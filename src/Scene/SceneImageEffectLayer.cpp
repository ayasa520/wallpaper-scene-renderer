#include "SceneImageEffectLayer.h"
#include "SceneNode.h"

#include "SpecTexs.hpp"
#include "Core/StringHelper.hpp"
#include "Utils/Logging.h"

using namespace wallpaper;

// The width and height parameters remain in the public constructor to preserve the existing parser
// call contract; effect geometry is copied from the resolved source/final meshes during
// ResolveEffect(), so the constructor only records the world node and ping-pong target names.
SceneImageEffectLayer::SceneImageEffectLayer(SceneNode* node, float /*w*/, float /*h*/,
                                             std::string_view pingpong_a,
                                             std::string_view pingpong_b)
    : m_worldNode(node),
      m_pingpong_a(pingpong_a),
      m_pingpong_b(pingpong_b),
      m_final_mesh(std::make_unique<SceneMesh>()),
      m_final_node(std::make_unique<SceneNode>()) {};

void SceneImageEffect::SetIdentity(int32_t owner_layer_id, int32_t effect_id,
                                   uint32_t effect_index, std::string effect_name) {
    m_owner_layer_id = owner_layer_id;
    m_effect_id      = effect_id;
    m_effect_index   = effect_index;
    m_effect_name    = std::move(effect_name);
}

void SceneImageEffect::SetLocalVisible(bool visible) {
    m_local_visible = visible;
    for (auto& node : nodes) {
        if (node.sceneNode != nullptr) {
            // Effect-local visibility is intentionally separate from layer visibility. The layer
            // tree still owns parent/child propagation through SceneNode::SetLayerVisible(), while
            // this flag lets scripts and animations disable only this effect without rebuilding
            // the render graph or hiding the owner layer.
            node.sceneNode->SetLocalVisible(visible);
        }
    }
}

void SceneImageEffect::SetBypassTargets(std::string src, std::string dst) {
    m_bypass_src = std::move(src);
    m_bypass_dst = std::move(dst);
}

bool SceneImageEffectLayer::HasFinalComposite() const {
    return m_final_node != nullptr && m_final_node->HasMaterial();
}

bool SceneImageEffectLayer::HasRuntimeVisibilityContract() const {
    for (const auto& effect : m_effects) {
        if (effect != nullptr && effect->HasRuntimeVisibilityContract()) return true;
    }
    return false;
}

bool SceneImageEffectLayer::HasVisibleRuntimeVisibilityContribution() const {
    for (const auto& effect : m_effects) {
        if (effect != nullptr && effect->HasRuntimeVisibilityContract() &&
            effect->LocalVisible()) {
            return true;
        }
    }
    return false;
}

bool SceneImageEffectLayer::HasVisibleSourceLessContribution() const {
    if (m_final_output_effect == nullptr) return false;

    if (HasRuntimeVisibilityContract()) {
        // Source-less compose helpers often model a user-selected generator effect followed by
        // always-visible filters such as fisheye or scroll. The filters are not valid sources on
        // their own; when every runtime-selected source effect is hidden, publishing the final
        // filter output would expose an empty or stale helper target as a rectangular quad.
        return HasVisibleRuntimeVisibilityContribution();
    }

    // Static source-less helpers without runtime visibility still need to draw their authored
    // chain normally. In that simpler contract the final effect's own visibility remains the best
    // indication that the chain currently contributes visible pixels.
    return m_final_output_effect->LocalVisible();
}

bool SceneImageEffectLayer::ShouldRunFinalComposite() const {
    // The neutral final composite has two modes. Ordinary image/text layers keep their historical
    // direct final writer while visible and use this pass only as a preserve-source hidden fallback.
    // Source-less passthrough helpers have no meaningful base image, so this pass becomes the only
    // screen publisher and draws only while the helper chain has a visible source contribution.
    if (!HasFinalComposite() || m_final_output_effect == nullptr) return false;
    if (m_publish_final_composite) return HasVisibleSourceLessContribution();
    if (m_final_output_effect->LocalVisible()) return false;
    return m_hidden_final_composite_policy == HiddenFinalCompositePolicy::PreserveSource;
}

void SceneImageEffectLayer::SetFinalCompositeSource(std::string source) {
    if (!HasFinalComposite()) return;

    auto* material = m_final_node->Mesh()->Material();
    if (material == nullptr) return;

    // The final composite is deliberately separate from every authored effect shader. Ordinary
    // layers use it to draw the bypassed ping-pong texture when the final authored effect is hidden.
    // Source-less passthrough helpers use the same neutral shader as their stable screen publisher,
    // with the authored final effect kept private so visibility changes cannot leave stale direct
    // framebuffer writes behind.
    if (material->textures.empty()) material->textures.resize(1);
    material->textures[0] = std::move(source);
}

void SceneImageEffectLayer::SyncResolvedOutputMesh() {
    if (m_resolved_output_node == nullptr || m_resolved_output_node->Mesh() == nullptr) return;
    if (!m_resolved_output_follows_world) {
        // Private dependency outputs deliberately use the effect camera's unit fullscreen mesh.
        // Runtime resize/transform refreshes should not copy the layer's world-space final mesh into
        // that node, otherwise offscreen dependencies render as if they were visible scene quads.
        return;
    }

    // Resource-only render-graph refreshes keep the already-resolved effect nodes alive and only
    // recreate their GPU resources. Runtime text updates therefore cannot rely on ResolveEffect()
    // running again to copy `m_final_mesh` into the currently active output node. Synchronizing
    // the resolved node mesh here keeps effect-backed text quads visually in lockstep with the
    // latest runtime map-rate/size changes even when the pass topology is intentionally reused.
    m_resolved_output_node->Mesh()->ChangeMeshDataFrom(*m_final_mesh);
    // `ChangeMeshDataFrom()` shares the CPU-side mesh payload but does not flip the render-pass
    // dirty flag. Resource-only refreshes look at the live pass mesh, not at `m_final_mesh`, so
    // the resolved output node must be marked dirty explicitly or Vulkan keeps drawing the stale
    // vertex buffer even though the debug logs already show the updated quad geometry.
    m_resolved_output_node->Mesh()->SetDirty();
    if (HasFinalComposite() && m_final_node->Mesh() != nullptr) {
        // The neutral final composite has its own node whether it is acting as a hidden fallback or
        // as the source-less helper publisher. Keep that mesh synchronized as well; otherwise a
        // runtime text/image resize could fix the authored path while leaving this publisher with
        // stale geometry.
        m_final_node->Mesh()->ChangeMeshDataFrom(*m_final_mesh);
        m_final_node->Mesh()->SetDirty();
    }
}

void SceneImageEffectLayer::SyncResolvedNodeToWorld() {
    if (m_worldNode == nullptr) return;

    m_worldNode->UpdateTrans();
    // Final effect nodes are emitted as render-graph-only nodes, not as real children of the
    // authored scene node. Copying only the local TRS loses virtual parent transforms used by
    // render-order proxy groups such as Wallpaper Engine compose layers. Resolve the full world
    // matrix here so the final screen writer lands in the same place as the authored layer.
    Eigen::Affine3f world_affine;
    world_affine.matrix() = m_worldNode->ModelTrans().cast<float>();
    m_final_node->SetLocalAffine(world_affine);
    m_final_node->UpdateTrans();

    if (m_resolved_output_node != nullptr && m_resolved_output_follows_world) {
        m_resolved_output_node->CopyTrans(*m_final_node);
        m_resolved_output_node->UpdateTrans();
    }
}

void SceneImageEffectLayer::SyncResolvedNodeToMatrix(const Eigen::Affine3f& world_affine) {
    m_final_node->SetLocalAffine(world_affine);
    m_final_node->UpdateTrans();

    if (m_resolved_output_node != nullptr && m_resolved_output_follows_world) {
        m_resolved_output_node->CopyTrans(*m_final_node);
        m_resolved_output_node->UpdateTrans();
    }
}

void SceneImageEffectLayer::ResolveEffect(const SceneMesh& default_mesh,
                                          std::string_view effect_cam,
                                          bool keep_final_output_private,
                                          const Eigen::Affine3f* resolved_world_affine) {
    std::string_view ppong_a = m_pingpong_a, ppong_b = m_pingpong_b;
    auto             swap_pp = [&ppong_a, &ppong_b]() {
        std::swap(ppong_a, ppong_b);
    };
    auto default_node = SceneNode();
    auto sync_resolved_world = [&]() {
        if (resolved_world_affine != nullptr) {
            // Render-order proxy routes keep some authored children root-owned in the physical
            // SceneNode tree while still drawing them under a virtual parent. When the render graph
            // already resolved that routed world matrix, trust it here instead of asking the node's
            // physical parent chain, which would drop the virtual parent transform.
            SyncResolvedNodeToMatrix(*resolved_world_affine);
            return;
        }
        SyncResolvedNodeToWorld();
    };

    m_resolved_output_node = nullptr;
    m_final_output_effect  = nullptr;
    m_resolved_output_follows_world = true;
    m_publish_final_composite = false;
    sync_resolved_world();

    SceneImageEffectNode* fallback_last_output { nullptr };
    for (auto& eff : m_effects) {
        // Each effect consumes the current input ping-pong target and normally writes the next
        // output ping-pong target. Capturing that pair after alias resolution gives the renderer a
        // topology-stable hidden path: when this effect is locally hidden, a conditional copy moves
        // input to output so later effects observe the correct current frame instead of the last
        // frame produced while the effect was visible.
        eff->SetBypassTargets(std::string(ppong_a), std::string(ppong_b));
        for (auto& cmd : eff->commands) {
            if (sstart_with(cmd.src, WE_EFFECT_PPONG_PREFIX_A)) cmd.src = ppong_a;

            if (sstart_with(cmd.dst, WE_EFFECT_PPONG_PREFIX_A)) cmd.dst = ppong_a;
        }
        for (auto it = eff->nodes.begin(); it != eff->nodes.end(); it++) {
            if (sstart_with(it->output, WE_EFFECT_PPONG_PREFIX_B) ||
                it->output == SpecTex_Default) {
                it->output           = ppong_b;
                fallback_last_output = &(*it);
                m_final_output_effect = eff.get();
            }

            assert(it->sceneNode->HasMaterial());

            auto& material = *(it->sceneNode->Mesh()->Material());
            {
                material.blenmode = BlendMode::Normal;
                it->sceneNode->SetCamera(effect_cam.data());
                it->sceneNode->CopyTrans(default_node);
                it->sceneNode->Mesh()->ChangeMeshDataFrom(default_mesh);
            }

            auto& texs = material.textures;
            std::replace_if(
                texs.begin(),
                texs.end(),
                [](auto& t) {
                    return sstart_with(t, WE_EFFECT_PPONG_PREFIX_A);
                },
                ppong_a);
        }
        swap_pp();
    }
    if (HasFinalComposite()) {
        // Prepare a neutral fallback output but do not make it the normal resolved writer. Visible
        // final effects keep Wallpaper Engine's authored behavior by drawing directly to the
        // inherited output target. When that final effect is hidden, the renderer gates this node on
        // and samples the bypassed ping-pong target so the layer still contributes its unmodified
        // input instead of going blank.
        SetFinalCompositeSource(std::string(ppong_a));
        auto& mesh          = *m_final_node->Mesh();
        auto& material      = *mesh.Material();
        if (m_fullscreen) {
            // The synthetic fallback uses a generic image shader with an MVP uniform. Fullscreen
            // postprocess layers are 2x2 clip-space quads, so the fallback must stay on the effect
            // camera/default mesh path; routing it through the active scene camera would shrink the
            // hidden-effect passthrough into world units instead of covering the framebuffer.
            material.blenmode = BlendMode::Normal;
            m_final_node->SetCamera(effect_cam.data());
            m_final_node->CopyTrans(default_node);
            mesh.ChangeMeshDataFrom(default_mesh);
            LOG_INFO("SceneEffectFinalCompositeResolve: layer=%d name='%s' fullscreen=true "
                     "camera='%.*s' output='%s' source='%s' blend=%d",
                     m_worldNode != nullptr ? m_worldNode->ID() : -1,
                     m_worldNode != nullptr ? m_worldNode->Name().c_str() : "",
                     static_cast<int>(effect_cam.size()),
                     effect_cam.data(),
                     SpecTex_Default.data(),
                     ppong_a.data(),
                     static_cast<int>(material.blenmode));
        } else {
            sync_resolved_world();
            material.blenmode = m_final_blend;
            m_final_node->SetCamera(std::string());
            mesh.ChangeMeshDataFrom(*m_final_mesh);
        }
    }

    const bool source_less_final_output =
        m_hidden_final_composite_policy == HiddenFinalCompositePolicy::SuppressOutput &&
        m_final_output_effect != nullptr;
    const bool keep_authored_final_private =
        keep_final_output_private || source_less_final_output;
    m_publish_final_composite = source_less_final_output && !keep_final_output_private;

    if (fallback_last_output != nullptr && !keep_authored_final_private) {
        // Keep the historical visible path: the final authored shader writes the screen/default
        // output directly, preserving shaders whose visual result depends on being the final
        // compositor. The synthetic final composite remains dormant unless this effect is hidden.
        m_resolved_output_node = fallback_last_output->sceneNode.get();
        fallback_last_output->output = SpecTex_Default;
        auto& mesh                   = *(fallback_last_output->sceneNode->Mesh());
        auto& material               = *mesh.Material();
        if (m_fullscreen) {
            // Fullscreen postprocess final passes, including dino_run's godrays_combine shader,
            // already draw the unit effect mesh and may still multiply by
            // g_ModelViewProjectionMatrix. Keep that final pass on the effect camera so the 2x2
            // utility quad remains a full-frame clip-space composite instead of becoming a tiny
            // active-camera world quad that leaves the rays effectively invisible.
            m_resolved_output_follows_world = false;
            material.blenmode = BlendMode::Normal;
            fallback_last_output->sceneNode->SetCamera(effect_cam.data());
            fallback_last_output->sceneNode->CopyTrans(default_node);
            mesh.ChangeMeshDataFrom(default_mesh);
            LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=true "
                     "camera='%.*s' output='%s' material='%s' blend=%d",
                     m_worldNode != nullptr ? m_worldNode->ID() : -1,
                     m_worldNode != nullptr ? m_worldNode->Name().c_str() : "",
                     static_cast<int>(effect_cam.size()),
                     effect_cam.data(),
                     SpecTex_Default.data(),
                     material.name.c_str(),
                     static_cast<int>(material.blenmode));
        } else {
            sync_resolved_world();
            material.blenmode = m_final_blend;
            fallback_last_output->sceneNode->SetCamera(std::string());
            fallback_last_output->sceneNode->CopyTrans(*m_final_node);
            mesh.ChangeMeshDataFrom(*m_final_mesh);
        }
    } else if (fallback_last_output != nullptr) {
        // Some final authored effects must remain private even when their owner is a visible screen
        // layer. Hidden dependency sources need this because consumers sample the resolved private
        // texture. Source-less passthrough helpers need the same topology because their base image
        // is intentionally empty; a stable neutral composite can then publish only real visible
        // source contributions and suppress the chain when user-selected generator effects are off.
        m_resolved_output_node = fallback_last_output->sceneNode.get();
        m_resolved_output_follows_world = false;
        sync_resolved_world();
        auto& mesh                   = *(fallback_last_output->sceneNode->Mesh());
        auto& material               = *mesh.Material();
        {
            material.blenmode = BlendMode::Normal;
            fallback_last_output->sceneNode->SetCamera(effect_cam.data());
            fallback_last_output->sceneNode->CopyTrans(default_node);
            mesh.ChangeMeshDataFrom(default_mesh);
        }
    }
}

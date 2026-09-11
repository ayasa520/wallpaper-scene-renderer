#include "SceneToRenderGraph.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "RenderGraph/RenderGraph.hpp"
#include "SpecTexs.hpp"
#include "Core/MapSet.hpp"
#include "Utils/Logging.h"

#include "VulkanRender/AllPasses.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace wallpaper;
namespace wallpaper::rg
{

void doCopy(RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc, TexNode* in, TexNode* out) {
    builder.read(in);
    builder.write(out);

    desc.src = in->key();
    desc.dst = out->key();
}
void addCopyPass(RenderGraph& rgraph, TexNode* in, TexNode* out,
                 std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [in, out, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc) {
            doCopy(builder, desc, in, out);
            desc.should_execute = should_execute;
        });
}

void addCopyPass(RenderGraph& rgraph, const TexNode::Desc& in, const TexNode::Desc& out,
                 std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [in, out, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc) {
            auto* in_node  = builder.createTexNode(in);
            auto* out_node = builder.createTexNode(out, true);
            doCopy(builder, desc, in_node, out_node);
            desc.should_execute = should_execute;
        });
}

TexNode* addCopyPass(RenderGraph& rgraph, TexNode* in, TexNode::Desc* out_desc = nullptr,
                     std::function<bool()> should_execute = {}) {
    TexNode* copy { nullptr };
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [&copy, in, out_desc, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& pdesc) {
            auto desc = out_desc == nullptr ? in->genDesc() : *out_desc;
            if (out_desc == nullptr) {
                desc.key += "_" + std::to_string(in->version()) + "_copy";
                desc.name += "_" + std::to_string(in->version()) + "_copy";
            }
            copy = builder.createTexNode(desc, true);
            doCopy(builder, pdesc, in, copy);
            pdesc.should_execute = should_execute;
        });
    return copy;
}

TexNode* addDefaultComposeSnapshot(RenderGraph& rgraph, TexNode* in,
                                   std::function<bool()> should_execute = {}) {
    // `_rt_default` is both the compose write target and the FullFrameBuffer sampler. A unique
    // `_rt_default_<version>_copy` per self-write keeps a screen-sized image alive for every
    // overlapping letter. Reuse one screen-sized partner for that snapshot instead.
    //
    // The transfer itself stays. Shader color-blend and refraction must sample the current
    // compose, including earlier letters, while Preserve/LOAD keeps uncovered pixels. Writing
    // an unseeded partner and publishing a mesh from it drops or smears those pixels.
    TexNode::Desc snapshot = in->genDesc();
    snapshot.key           = std::string(SpecTex_DefaultPingPong);
    snapshot.name          = snapshot.key;
    return addCopyPass(rgraph, in, &snapshot, std::move(should_execute));
}

void addClearPass(RenderGraph& rgraph, const TexNode::Desc& target,
                  std::array<float, 4> color = { 0.0f, 0.0f, 0.0f, 0.0f },
                  std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::ClearPass>(
        "clear",
        PassNode::Type::Clear,
        [target, color, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::ClearPass::Desc& desc) {
            auto* target_node = builder.createTexNode(target, true);
            builder.write(target_node);
            desc.target         = target_node->key();
            desc.clear_value    = VkClearValue { .color = { color[0], color[1], color[2], color[3] } };
            desc.should_execute = should_execute;
        });
}

static bool IsRuntimeRenderTarget(const Scene* scene, const std::string& path) {
    // Authored effect FBO names are not guaranteed to carry Wallpaper Engine's `_rt_` prefix after
    // the parser uniquifies them with the effect-layer address. The render-target table is the
    // authoritative runtime contract, so graph construction must consult it before classifying a
    // texture edge as an imported asset.
    return IsSpecTex(path) || (scene != nullptr && scene->renderTargets.count(path) != 0);
}

static TexNode::Desc createTexDesc(std::string path, const Scene* scene = nullptr) {
    return TexNode::Desc { .name = path,
                           .key  = path,
                           .type = IsRuntimeRenderTarget(scene, path) ? TexNode::TexType::Temp
                                                                      : TexNode::TexType::Imported };
}

void addShadowAtlasPass(RenderGraph& rgraph, Scene* scene) {
    rgraph.addPass<vulkan::ShadowAtlasPass>(
        "shadow_atlas",
        PassNode::Type::CustomShader,
        [scene](RenderGraphBuilder& builder, vulkan::ShadowAtlasPass::Desc& desc) {
            auto* dst =
                builder.createTexNode(createTexDesc(std::string(SpecTex_ShadowAtlas), scene), true);
            builder.write(dst);
            desc.target = dst->key();
            desc.scene  = scene;
        });
}

void addVolumetricsSingleFillPass(RenderGraph& rgraph, const Scene* scene) {
    // Scene-depth blit into `_rt_volumetricsSingle`.
    // Read `_rt_default` so this runs after model chunks have written shared scene depth.
    rgraph.addPass<vulkan::VolumetricsSingleFillPass>(
        "volumetrics_single_fill",
        PassNode::Type::CustomShader,
        [scene](RenderGraphBuilder& builder, vulkan::VolumetricsSingleFillPass::Desc& desc) {
            auto* src = builder.createTexNode(createTexDesc(std::string(SpecTex_Default), scene));
            auto* dst =
                builder.createTexNode(createTexDesc(std::string(SpecTex_VolumetricsSingle), scene),
                                      true);
            builder.read(src);
            builder.write(dst);
            desc.dst          = dst->key();
            desc.scene_output = src->key();
        });
}
} // namespace wallpaper::rg

static void CheckAndSetSprite(Scene& scene, vulkan::ShaderDrawRequest& desc,
                              const SceneMaterial& material) {
    for (usize i = 0; i < material.textures.size(); i++) {
        const auto& tex = material.Texture(i);
        if (! tex.empty() && ! IsSpecTex(tex) && scene.textures.count(tex) != 0) {
            const auto& stex = scene.textures.at(tex);
            if (stex.isSprite) {
                desc.sprites_map[i] = stex.spriteAnim;
            }
        }
    }
}

static int32_t NodeLayerId(const Scene& scene, SceneNode* node) {
    return scene.LayerIdForNode(node);
}

static bool ShouldExecuteHiddenDependency(Scene& scene, SceneDraw draw, std::string_view output) {
    const int32_t layer_id = draw.LayerId(scene);
    if (layer_id == 0) return false;
    if (! scene.IsLayerOffscreenDependencySource(layer_id)) return false;

    // Hidden dependencies may maintain their private sources for other readers. Main and
    // reflection targets are both enclosing destinations, so neither admits hidden publication.
    return output != SpecTex_Default && output != SpecTex_Reflection;
}

static bool ShouldExecuteLayerDestination(Scene& scene, SceneDraw draw,
                                          std::string_view output) {
    if (!draw.Valid() || draw.Visible(scene)) return true;
    return ShouldExecuteHiddenDependency(scene, draw, output);
}

struct ExtraInfo {
    rg::RenderGraph*           rgraph { nullptr };
    Scene*                     scene { nullptr };
    std::unordered_map<int32_t, size_t> layer_order_index {};
    // Model depth is shared per output target. Tracking the first model pass here lets the graph
    // clear depth once for each target, then load it for later chunks without touching 2D passes.
    std::unordered_set<std::string> model_depth_outputs_seen {};
    SceneImageEffect::FrameFboBindings frame_fbo_bindings;
    bool                       use_mipmap_framebuffer { false };
};

static bool IsOffscreenDependencyLayer(const ExtraInfo& extra, i32 imgId) {
    return extra.scene != nullptr && imgId != 0 &&
        extra.scene->IsLayerOffscreenDependencySource(imgId);
}

struct OrderedRenderGraphChild {
    SceneNode* node { nullptr };
    bool       authored { false };
    size_t     sequence { 0 };
};

struct DrawPassOptions {
    bool        resolved_scene_color { false };
    AlphaWritePolicy alpha_write_policy { AlphaWritePolicy::Preserve };
    std::optional<BlendMode> blend_override {};
    std::optional<bool> depth_test_override {};
    std::optional<bool> depth_write_override {};
    bool        destination_alpha_override { false };
    bool        clear_before_draw { false };
    std::string camera_override;
    bool        use_active_camera_for_parallax { false };
    bool        premultiplied_source_blend { false };
    bool        use_active_camera_for_uniforms { false };
    ShaderModelSpace model_space { ShaderModelSpace::Object };
    bool        reflection_pass { false };
    bool        reflection_raster { false };
    bool        reflection_snapshot { false };
    std::string effect_snapshot_camera;
};

struct TraversalRoute {
    // Traversal determines order and destination state. Object transforms are resolved from
    // SceneObject at draw time; membership in a composition is independent of node parenting.
    bool                           compose_source { false };
    std::string                    compose_source_camera;
    AlphaWritePolicy               compose_source_alpha_write_policy {
        AlphaWritePolicy::Preserve
    };
    bool                           premultiplied_source_blend { false };
    bool                           reflection_pass { false };
    // Private source and centered composition pushes restore ordinary raster matrices while
    // retaining the reflected frame's eye/basis. Publication restores this incoming raster state.
    bool                           reflection_raster { false };
};

static bool HasRenderableMeshMaterial(SceneNode* node) {
    return node != nullptr && node->Mesh() != nullptr && node->Mesh()->Material() != nullptr;
}

static size_t NodeLayerOrderIndex(SceneNode* node, const ExtraInfo& extra) {
    if (extra.scene == nullptr || node == nullptr) return std::numeric_limits<size_t>::max();
    const int32_t layer_id = NodeLayerId(*extra.scene, node);
    if (auto it = extra.layer_order_index.find(layer_id); it != extra.layer_order_index.end()) {
        return it->second;
    }
    return std::numeric_limits<size_t>::max();
}

static bool IsEffectLocalDependency(SceneNode* node, const ExtraInfo& extra) {
    if (extra.scene == nullptr || node == nullptr) return false;
    const int32_t layer_id = NodeLayerId(*extra.scene, node);
    return layer_id != 0 && extra.scene->IsLayerOffscreenDependencySource(layer_id);
}

static bool EffectSourceUsesProxyChildren(SceneImageEffectLayer* imgeff) {
    if (imgeff == nullptr) return false;
    const auto policy = imgeff->SourceContributionPolicy();
    return policy == SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren ||
        policy == SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly;
}

static bool EffectSourceUsesOwnerNode(SceneImageEffectLayer* imgeff) {
    if (imgeff == nullptr) return true;
    const auto policy = imgeff->SourceContributionPolicy();
    return policy == SceneImageEffectLayer::SourcePolicy::OwnerNode ||
        policy == SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren;
}

static bool DrawMaterialSamplesFramebuffer(SceneDraw draw) {
    if (draw.Mesh() == nullptr || draw.Mesh()->Material() == nullptr) return false;
    return draw.Mesh()->Material()->SamplesTexture(SpecTex_Default);
}

static bool DrawUsesPerspectiveCamera(const Scene& scene, SceneDraw draw) {
    const auto is_perspective = [&scene](const std::string& name) {
        const auto camera_it = scene.cameras.find(name);
        return camera_it != scene.cameras.end() && camera_it->second != nullptr &&
            camera_it->second->IsPerspective();
    };
    // Publication selects its camera explicitly. Source nodes still inherit projection from
    // their physical parent; keep that walk so perspective particles retain their depth scaling.
    if (draw.Phase() != nullptr) return is_perspective(draw.Camera());
    for (auto* current = draw.Node(); current != nullptr; current = current->Parent()) {
        if (!current->Camera().empty() && is_perspective(current->Camera())) return true;
    }
    return false;
}

struct EffectSourceRoutingDecision {
    bool        owner_node_samples_framebuffer { false };
    bool        owner_node_uses_perspective_camera { false };
    bool        owner_node_contributes_to_effect_source { false };
    bool        proxy_children_contribute_to_effect_source { false };
    std::string active_compose_source_camera;
    std::string child_compose_source_camera;
    bool        use_compose_camera_override { false };
};

static EffectSourceRoutingDecision ResolveEffectSourceRouting(SceneNode* node,
                                                              SceneImageEffectLayer* imgeff,
                                                              const ExtraInfo& extra,
                                                              const TraversalRoute& route) {
    EffectSourceRoutingDecision decision;

    // copybackground alone selects the owner-card contribution of a non-fullscreen composition.
    // Empty children and private dependency ownership do not authorize an extra framebuffer draw
    // after its transparent clear.
    decision.owner_node_samples_framebuffer = DrawMaterialSamplesFramebuffer(node);
    decision.owner_node_uses_perspective_camera =
        extra.scene != nullptr && DrawUsesPerspectiveCamera(*extra.scene, node);
    decision.owner_node_contributes_to_effect_source =
        EffectSourceUsesOwnerNode(imgeff);
    decision.proxy_children_contribute_to_effect_source = EffectSourceUsesProxyChildren(imgeff);
    decision.active_compose_source_camera =
        route.compose_source ? route.compose_source_camera : std::string();
    // A nested non-fullscreen composition establishes its own child projection/I state, then
    // restores the incoming state for publication. Fullscreen leaves the incoming child state
    // intact. Source and publication must not share one camera choice merely because both happen
    // inside the same recursive visit.
    decision.child_compose_source_camera =
        decision.proxy_children_contribute_to_effect_source && imgeff != nullptr &&
            !imgeff->IsFullscreen()
        ? imgeff->BridgeCameraName() : decision.active_compose_source_camera;
    decision.use_compose_camera_override =
        route.compose_source && imgeff == nullptr && !decision.active_compose_source_camera.empty();
    return decision;
}

static DrawPassOptions BuildOwnerSourcePassOptions(
    SceneImageEffectLayer* imgeff,
    std::string_view output,
    std::string_view inherited_output,
    const TraversalRoute& route,
    const EffectSourceRoutingDecision& source_route) {
    // Every private effect-source seed must start from a cleared target, not only composition
    // sources. Render-graph memory aliasing can hand this pass a target whose bytes still hold
    // another (now hidden) layer's last output, and a translucent owner draw with zero alpha
    // writes nothing at all - compositing would then resurrect the stale content full-screen.
    const bool clear_private_effect_source = imgeff != nullptr && output != inherited_output;
    const bool evaluate_framebuffer_source_with_active_camera =
        source_route.owner_node_samples_framebuffer &&
        !source_route.owner_node_uses_perspective_camera;

    // Owner-source emission is the one place where source routing affects actual pass state. Keep
    // these side effects grouped so future route types can extend the pass contract without adding
    // another cluster of loosely related booleans inside ToGraphPass().
    //
    // - Composition source routes write child layers into a parent-local source target. Their alpha
    //   policy comes from the parent composition layer's copybackground contract.
    // - An owner material that samples the live framebuffer normally evaluates world geometry
    //   against the active scene camera, including while writing a private source target. The
    //   composelayer vertex shader separately turns authored texture coordinates into the fullscreen
    //   output quad, so using the private effect camera for both roles creates the cursor-following
    //   rectangular region seen in incorrect implementations. An explicitly perspective node is
    //   the exception: its authored camera remains the draw camera even when its material samples
    //   the framebuffer. Forcing refractive perspective particles through the active orthographic
    //   camera removes all authored depth scaling and makes foreground rain appear uniformly small.
    // - Private effect source targets are cleared only at the seed step; clearing later authored
    //   effect passes would erase intermediate waterwaves/foliagesway/opacity results.
    return DrawPassOptions {
        .alpha_write_policy = route.compose_source
            ? route.compose_source_alpha_write_policy
            : AlphaWritePolicy::Preserve,
        .clear_before_draw = clear_private_effect_source,
        .camera_override = imgeff != nullptr
            ? imgeff->BridgeCameraName()
            : (source_route.use_compose_camera_override
                   ? source_route.active_compose_source_camera
                   : std::string()),
        .use_active_camera_for_parallax = source_route.use_compose_camera_override,
        .premultiplied_source_blend = route.premultiplied_source_blend,
        .use_active_camera_for_uniforms =
            evaluate_framebuffer_source_with_active_camera,
        // Ordinary source materials rasterize their geometry with identity I. Passthrough retains
        // its owner matrix. Neither choice mutates the authored transform or gives a source
        // camera to the owner's descendants.
        .model_space = imgeff != nullptr && !imgeff->Owner().Passthrough()
            ? ShaderModelSpace::Geometry : ShaderModelSpace::Object,
        .reflection_pass = route.reflection_pass,
        .reflection_raster = route.reflection_raster &&
            (imgeff == nullptr || imgeff->IsFullscreen() ||
             evaluate_framebuffer_source_with_active_camera),
        .reflection_snapshot = route.reflection_raster,
        .effect_snapshot_camera = source_route.active_compose_source_camera,
    };
}

static std::string_view EffectSourcePolicyName(SceneImageEffectLayer::SourcePolicy policy) {
    switch (policy) {
    case SceneImageEffectLayer::SourcePolicy::None: return "none";
    case SceneImageEffectLayer::SourcePolicy::OwnerNode: return "owner-node";
    case SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren:
        return "owner-node-and-proxy-children";
    case SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly: return "proxy-children-only";
    }
    return "unknown";
}

static bool HasCompositionAncestor(const Scene& scene, const SceneObject& object) {
    // Exclude a descendant from the scene list only when a passthrough ancestor owns its draw.
    // Ordinary parent/bone relations affect placement and visibility, but do not move the object
    // beside its parent in painter order.
    auto* parent = scene.FindSceneObject(object.ParentId());
    while (parent != nullptr) {
        if (parent->Passthrough()) return true;
        parent = scene.FindSceneObject(parent->ParentId());
    }
    return false;
}

static std::vector<OrderedRenderGraphChild> OrderedRenderGraphChildren(
    SceneNode* node, ExtraInfo& extra) {
    std::vector<OrderedRenderGraphChild> children;
    if (node == nullptr || extra.scene == nullptr) return children;

    auto& scene = *extra.scene;
    const bool scene_list = node == scene.sceneGraph.get();
    const auto* owner = scene.FindSceneObject(NodeLayerId(scene, node));
    const bool owner_visit = owner != nullptr && owner->LayerNode() == node;
    size_t sequence = 0;
    for (auto& child : node->GetChildren()) {
        if (!child) continue;
        const auto* child_owner = scene.FindSceneObject(NodeLayerId(scene, child.get()));
        // The physical node tree still retains resources, such as model chunks and emitters.
        // Authored objects are enumerated below from their own table; admitting their physical
        // edges here would duplicate a bone child or move a normal child out of scene order.
        if (child_owner != nullptr && child_owner->LayerNode() == child.get()) continue;
        children.push_back(OrderedRenderGraphChild {
            .node = child.get(),
            .sequence = sequence++,
        });
    }

    // A scene visit uses the authored list. Within a passthrough source, visit its child list and
    // then ordinary descendants; nested passthrough objects establish their own source phase.
    // Bone attachment does not change membership. Detached source/material resources never
    // enumerate their owner's children.
    if (scene_list || (owner_visit &&
        (owner->Passthrough() || HasCompositionAncestor(scene, *owner)))) {
        for (const auto layer_id : scene.layerOrder) {
            const auto* child_owner = scene.FindSceneObject(layer_id);
            if (child_owner == nullptr || child_owner->LayerNode() == nullptr) continue;
            if (scene_list ? HasCompositionAncestor(scene, *child_owner)
                           : child_owner->ParentId() != owner->Id()) continue;
            children.push_back(OrderedRenderGraphChild {
                .node = child_owner->LayerNode(),
                .authored = true,
                .sequence = sequence++,
            });
        }
    }

    // Resource-only children retain their insertion order relative to the same owner. Authored
    // objects retain scene-list order at the scene boundary and sibling order inside composition.
    std::stable_sort(children.begin(), children.end(), [&extra](const auto& lhs, const auto& rhs) {
        const auto lhs_index = NodeLayerOrderIndex(lhs.node, extra);
        const auto rhs_index = NodeLayerOrderIndex(rhs.node, extra);
        if (lhs_index != rhs_index) return lhs_index < rhs_index;
        return lhs.sequence < rhs.sequence;
    });
    return children;
}

template <typename PassT>
static void AddDrawPassImpl(SceneDraw draw, std::string_view output, i32 imgId, ExtraInfo& extra,
                            std::function<bool()> should_execute, DrawPassOptions options) {
    auto& rgraph = *extra.rgraph;
    auto& scene  = *extra.scene;

    if (draw.Mesh() == nullptr) {
        return;
    }
    auto* mesh = draw.Mesh();
    if (mesh->Material() == nullptr) {
        return;
    }
    auto* material = mesh->Material();
    const std::string output_key(output);
    const bool is_model_pass = material->modelRenderState.has_value();
    const bool clear_model_depth = is_model_pass && output_key != SpecTex_Default &&
        output_key != SpecTex_Reflection &&
        extra.model_depth_outputs_seen.insert(output_key).second;

    // The entire reflected draw is quality-gated, including preparatory texture copies. Receivers
    // retain their ordinary material and target while the checkbox is off; the independent stage
    // clear still supplies the empty scene color.
    if (options.reflection_pass) {
        should_execute = [inner = std::move(should_execute), &scene]() {
            if (!scene.reflectionsEnabled) return false;
            return !inner || inner();
        };
    }

    // A destination draw can contain preparatory copies as well as the shader submission itself.
    // Hidden layers skip the complete destination-draw operation, so all graph commands owned by
    // this pass share one runtime gate. This is especially important for framebuffer self-reads:
    // copying `_rt_default` for an invisible layer would still consume bandwidth and mutate its
    // snapshot target even though the subsequent shader is correctly skipped.
    auto pass_execution_gate =
        [draw, &scene, output_key, inner = std::move(should_execute)]() {
            if (inner && !inner()) return false;
            return ShouldExecuteLayerDestination(scene, draw, output_key);
        };

    std::string passName = material->name;
    rgraph.addPass<PassT>(
        passName,
        rg::PassNode::Type::CustomShader,
        [material, draw, output_key, imgId, &rgraph, &scene, &extra,
         clear_model_depth, options = std::move(options),
         pass_execution_gate = std::move(pass_execution_gate)](
            rg::RenderGraphBuilder& builder, typename PassT::Desc& pdesc) {
            const auto& pass = builder.workPassNode();
            // Passing the live scene into the prepared pass lets resource refreshes resolve current
            // render-target dependencies directly, which is what keeps first-class text bridges and
            // ordinary effect passes on the same stable render-graph contract.
            pdesc.scene      = &scene;
            pdesc.draw       = draw;
            pdesc.layer_id   = imgId;
            pdesc.execute_when_hidden = ShouldExecuteHiddenDependency(scene, draw, output_key);
            pdesc.should_execute      = pass_execution_gate;
            pdesc.output     = output_key;
            pdesc.resolved_scene_color = options.resolved_scene_color;
            pdesc.blend_override = options.blend_override;
            pdesc.depth_test_override = options.depth_test_override;
            pdesc.depth_write_override = options.depth_write_override;
            pdesc.destination_alpha_override = options.destination_alpha_override;
            pdesc.alpha_write_policy = output_key != SpecTex_Default
                ? options.alpha_write_policy
                : AlphaWritePolicy::Preserve;
            pdesc.premultiplied_source_blend = options.premultiplied_source_blend;
            pdesc.clear_before_draw = output_key != SpecTex_Default && options.clear_before_draw;
            pdesc.camera_override = options.camera_override;
            pdesc.use_active_camera_for_uniforms = options.use_active_camera_for_uniforms;
            pdesc.model_space = options.model_space;
            pdesc.reflection_pass = options.reflection_pass;
            pdesc.reflection_raster = options.reflection_raster;
            pdesc.reflection_snapshot = options.reflection_snapshot;
            pdesc.effect_snapshot_camera = options.effect_snapshot_camera;
            pdesc.use_active_camera_for_parallax =
                !pdesc.camera_override.empty() && options.use_active_camera_for_parallax;
            if (!pdesc.camera_override.empty()) {
                LOG_INFO("SceneRenderGraphComposeCameraOverride: layer=%d draw='%s' "
                         "output='%s' camera='%s' active-parallax=%s",
                         imgId,
                         draw.Valid() ? draw.Name().c_str() : "",
                         output_key.c_str(),
                         pdesc.camera_override.c_str(),
                         pdesc.use_active_camera_for_parallax ? "true" : "false");
            }
            if (pdesc.use_active_camera_for_uniforms) {
                LOG_INFO("SceneRenderGraphActiveCameraUniformOverride: layer=%d draw='%s' "
                         "output='%s'",
                         imgId,
                         draw.Valid() ? draw.Name().c_str() : "",
                         output_key.c_str());
            }
            // A material cannot create a depth attachment for a color-only effect target. Scene
            // destinations share the same attachment as models, while masked meshes retain their
            // independent stencil attachment. Capture effective booleans now: comparing live
            // material pointers during residency would observe the new value in both generations.
            // A resolved-color post-process can use the same logical output name as the scene,
            // but not the same attachments. It remains single-sample and color-only, leaving the
            // traversal's shared depth intact for all model/image/effect draws in that stage.
            pdesc.shared_depth = !options.resolved_scene_color &&
                (material->modelRenderState.has_value() ||
                 (draw.Mesh()->MaskedDraw().empty() &&
                  (output_key == SpecTex_Default || output_key == SpecTex_Reflection)));
            const auto blend = options.blend_override.value_or(material->blenmode);
            pdesc.depth_test = pdesc.shared_depth &&
                options.depth_test_override.value_or(material->depthTest);
            pdesc.depth_write = pdesc.depth_test &&
                options.depth_write_override.value_or(material->depthWrite) &&
                blend != BlendMode::Translucent && blend != BlendMode::Additive;
            pdesc.clear_depth = clear_model_depth || output_key == SpecTex_VolumetricsBack;
            if (const auto& model_state = material->modelRenderState; model_state.has_value()) {
                pdesc.model_pass = true;
                pdesc.depth_clear = model_state->depthClear;
            }
            CheckAndSetSprite(scene, pdesc, *material);
            for (usize i = 0; i < material->textures.size(); i++) {
                const auto&  url = material->Texture(i);
                if (std::getenv("WESCENE_TRACE_MEDIA_STATE") != nullptr &&
                    material->systemTextureBindings.contains(i)) {
                    LOG_INFO("SceneMaterialSystemTextureBind: layer=%d node='%s' "
                             "shader='%s' slot=%zu authored='%s' selected='%s'",
                             draw.LayerId(scene), draw.Name().c_str(), material->name.c_str(),
                             i, material->textures[i].c_str(), url.c_str());
                }
                rg::TexNode* input { nullptr };
                if (url.empty()) {
                    pdesc.textures.emplace_back("");
                    continue;
                } else {
                    rg::TexNode::Desc desc;
                    desc.key  = url;
                    desc.name = url;
                    // Private layer destinations participate in ordinary name/version
                    // dependencies here. Their readers must sample slot zero directly, without a
                    // layer-wide last-writer substitution or an additional GPU copy. Some
                    // effect-local FBOs use plain names such as `blur_start_2_<addr>`. Treat any
                    // key already registered in Scene::renderTargets as temporary graph storage
                    // so those edges order like internal render targets, not external material
                    // textures.
                    desc.type = ! rg::IsRuntimeRenderTarget(&scene, url)
                        ? rg::TexNode::TexType::Imported
                        : rg::TexNode::TexType::Temp;
                    input     = builder.createTexNode(desc);
                    if (rg::IsRuntimeRenderTarget(&scene, url)) builder.markVirtualWrite(input);
                    if (sstart_with(url, WE_MIP_MAPPED_FRAME_BUFFER))
                        extra.use_mipmap_framebuffer = true;
                }

                if (url == output_key) {
                    builder.markSelfWrite(input);
                    // Compose self-writes reuse one screen-sized snapshot. Other self-writes
                    // still get a versioned copy name because their destinations are not a
                    // shared compose buffer.
                    input = url == SpecTex_Default
                        ? rg::addDefaultComposeSnapshot(rgraph, input, pass_execution_gate)
                        : rg::addCopyPass(rgraph,
                                          input,
                                          static_cast<rg::TexNode::Desc*>(nullptr),
                                          pass_execution_gate);
                }
                builder.read(input);
                pdesc.textures.emplace_back(input->key());
            }

            // Mask textures belong to the mesh draw plan rather than to the visible material. The
            // selected masked pass declares them as ordinary imported reads, while an empty plan
            // leaves the regular CustomShaderPass with no mask-related graph resources at all.
            for (const auto& group : draw.Mesh()->MaskedDraw().groups) {
                rg::TexNode::Desc desc {
                    .name = group.maskTexture,
                    .key  = group.maskTexture,
                    .type = rg::TexNode::TexType::Imported,
                };
                auto* input = builder.createTexNode(desc);
                builder.read(input);
            }

            rg::TexNode* output_node { nullptr };
            output_node =
                builder.createTexNode(rg::TexNode::Desc { .name = output_key,
                                                          .key  = output_key,
                                                          .type = rg::TexNode::TexType::Temp },
                                      true);
            builder.write(output_node);
        });
}

static void AddDrawPass(SceneDraw draw, std::string_view output, i32 imgId, ExtraInfo& extra,
                        std::function<bool()> should_execute = {},
                        DrawPassOptions options = {}) {
    if (draw.Valid() && draw.Mesh() != nullptr && ! draw.Mesh()->MaskedDraw().empty()) {
        AddDrawPassImpl<vulkan::MaskedMeshPass>(draw,
                                                output,
                                                imgId,
                                                extra,
                                                std::move(should_execute),
                                                std::move(options));
        return;
    }
    AddDrawPassImpl<vulkan::CustomShaderPass>(draw,
                                              output,
                                              imgId,
                                              extra,
                                              std::move(should_execute),
                                              std::move(options));
}

static void AddTextNodePass(SceneNode* node, std::string_view output, i32 imgId, ExtraInfo& extra,
                            std::function<bool()> should_execute, DrawPassOptions options) {
    auto& rgraph = *extra.rgraph;
    auto& scene  = *extra.scene;

    if (node == nullptr || node->Text() == nullptr) {
        return;
    }

    const std::string output_key(output);
    std::string pass_name = node->Name().empty() ? std::string("text") : node->Name();
    rgraph.addPass<vulkan::TextPass>(
        pass_name,
        rg::PassNode::Type::Text,
        [node, output_key, imgId, should_execute = std::move(should_execute),
         options = std::move(options), &scene](
            rg::RenderGraphBuilder& builder, vulkan::TextPass::Desc& pdesc) {
            const auto& pass = builder.workPassNode();
            // Text is now emitted as its own render-graph pass. It shares the same constrained
            // hidden-dependency rule as mesh passes: invisible helper layers may render private
            // offscreen sources, but they must not composite text directly into `_rt_default`.
            pdesc.scene = &scene;
            pdesc.node = node;
            // Keep the authored layer id on the prepared pass so runtime text rerasters can
            // refresh the exact Clock/TextPass resources without broadening the dirty target set.
            pdesc.layer_id = imgId;
            pdesc.execute_when_hidden = ShouldExecuteHiddenDependency(scene, node, output_key);
            pdesc.should_execute = should_execute;
            // Owner-source routing grants clear ownership only to a private seed. Carry that
            // role explicitly into text material/color selection rather than classifying every
            // non-main destination (including reflection and composition) as a private source.
            pdesc.private_source = options.clear_before_draw;
            pdesc.clear_before_draw = pdesc.private_source && node->Text()->object.opaquebackground;
            pdesc.output = output_key;
            pdesc.camera_override = options.camera_override;
            pdesc.use_active_camera_for_parallax = options.use_active_camera_for_parallax;
            pdesc.model_space = options.model_space;
            pdesc.reflection_pass = options.reflection_pass;
            pdesc.reflection_raster = options.reflection_raster;
            pdesc.reflection_snapshot = options.reflection_snapshot;
            pdesc.effect_snapshot_camera = options.effect_snapshot_camera;
            pdesc.alpha_write_policy = output_key != SpecTex_Default
                ? options.alpha_write_policy
                : AlphaWritePolicy::Preserve;

            // Direct glyphs test the depth already owned by the main/reflection destination.
            // Private source/composition images stay color-only even when the text owner enables
            // testing. Capture the live glyph selection here; the background's retained selection
            // is resolved from its scene-owned material state only at an actual direct draw.
            pdesc.shared_depth = output_key == SpecTex_Default || output_key == SpecTex_Reflection;
            pdesc.glyph_depth_test = pdesc.shared_depth &&
                node->Text()->object.depthtest == "enabled";

            if (pdesc.private_source && !pdesc.clear_before_draw) {
                // Glyph blending loads the initializer's exact target version. This is an
                // attachment dependency, not a texture self-sample or an extra framebuffer copy.
                builder.read(builder.createTexNode(rg::createTexDesc(output_key, &scene)));
            }
            auto* output_node =
                builder.createTexNode(rg::TexNode::Desc { .name = output_key,
                                                          .key = output_key,
                                                          .type = rg::TexNode::TexType::Temp },
                                      true);
            builder.write(output_node);
            (void)pass;
        });
}

static void ToGraphPass(SceneNode* node, std::string_view inherited_output, i32 imgId,
                        ExtraInfo& extra, std::function<bool()> node_execute_gate = {},
                        TraversalRoute route = {}) {
    auto& scene = *extra.scene;

    // Gate the complete owner invocation, including text, private clears, composition children
    // and commands. A shader-only gate would still mutate feedback while reflections are off.
    if (route.reflection_pass) {
        node_execute_gate = [inner = std::move(node_execute_gate), &scene]() {
            return scene.reflectionsEnabled && (!inner || inner());
        };
    }

    if (const auto* object = scene.FindSceneObject(imgId);
        object != nullptr && object->LayerNode() == node &&
        std::getenv("WESCENE_TRACE_OBJECT_DRAW") != nullptr) {
        LOG_INFO("SceneObjectDrawVisit: layer=%d parent=%d passthrough=%s "
                 "composition-member=%s compose-source=%s reflection=%s output='%.*s'",
                 object->Id(), object->ParentId(), object->Passthrough() ? "true" : "false",
                 HasCompositionAncestor(scene, *object) ? "true" : "false",
                 route.compose_source ? "true" : "false",
                 route.reflection_pass ? "true" : "false",
                 static_cast<int>(inherited_output.size()), inherited_output.data());
    }

    std::string_view         output = inherited_output;
    SceneImageEffectLayer*   imgeff { nullptr };
    if (node != nullptr) {
        // Source geometry is owned by the authored object itself. Its visit emits the source
        // and effect sequence directly; auxiliary resource nodes cannot select this bridge.
        auto* candidate = scene.FindImageEffectLayer(NodeLayerId(scene, node));
        if (candidate != nullptr && candidate->Owner().LayerNode() == node) {
            // The zero-step, non-private passthrough arm returns before touching its source
            // target when there are no authored children. A nonempty composition must still
            // execute its child/source/publication phases. Effect visibility and authored
            // hierarchy changes rebuild this graph, so the predicate follows the current sequence
            // rather than the last authored effect.
            if (candidate->Owner().Passthrough() &&
                candidate->VisibleCompositionStepCount() == 0 &&
                candidate->ResolveFinalOutputCapability() ==
                    FinalOutputCapability::SceneAuthoredWriter &&
                scene.GetLayerChildren(candidate->Owner().Id()).empty()) {
                LOG_INFO("SceneRenderGraphEmptyCompositionSkip: layer=%d name='%s' "
                         "visible-steps=0 private=false authored-children=0",
                         candidate->Owner().Id(), candidate->Owner().Name().c_str());
                return;
            }
            // Resolve the owner's program, geometry and blend even when this build draws directly
            // into the destination; a previous visible chain may have left the prelighting source
            // program on the live material.
            candidate->ResolveOwnerDraw(scene);
            if (candidate->UsesDirectDraw()) {
                LOG_INFO("SceneRenderGraphLayerDirectDraw: layer=%d name='%s' output='%.*s' "
                         "visible-steps=0 private=false compose-source=%s primitive=%s "
                         "visible=%s local-visible=%s",
                         candidate->Owner().Id(), candidate->Owner().Name().c_str(),
                         static_cast<int>(inherited_output.size()), inherited_output.data(),
                         route.compose_source ? "true" : "false",
                         node->Text() != nullptr ? "text" : "image",
                         node->Visible() ? "true" : "false",
                         node->LocalVisible() ? "true" : "false");
            } else {
                imgeff = candidate;
                // A shape dispatches its effect directly in the incoming destination. Its
                // retained half-canvas resources do not imply an image source phase.
                output = imgeff->UsesShapeDraw() ? inherited_output : imgeff->SourceTarget();
            }
        }
    }

    const auto source_route =
        ResolveEffectSourceRouting(node, imgeff, extra, route);
    if (imgeff != nullptr &&
        imgeff->SourceContributionPolicy() == SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly) {
        LOG_INFO("SceneRenderGraphComposeSourceClear: layer=%d name='%s' output='%.*s' "
                 "camera='%s'",
                 NodeLayerId(scene, node),
                 node != nullptr ? node->Name().c_str() : "",
                 static_cast<int>(output.size()),
                 output.data(),
                 source_route.child_compose_source_camera.c_str());
        const std::string clear_output(output);
        rg::addClearPass(
            *extra.rgraph,
            rg::createTexDesc(clear_output, extra.scene),
            { 0.0f, 0.0f, 0.0f, 0.0f },
            [node, &scene, clear_output, outer_gate = node_execute_gate]() {
                if (outer_gate && !outer_gate()) return false;
                return ShouldExecuteLayerDestination(scene, node, clear_output);
            });
    }

    if (HasRenderableMeshMaterial(node) &&
        source_route.owner_node_contributes_to_effect_source) {
        AddDrawPass(node,
                    output,
                    imgId,
                    extra,
                    node_execute_gate,
                    BuildOwnerSourcePassOptions(
                        imgeff, output, inherited_output, route, source_route));
    } else if (HasRenderableMeshMaterial(node) && imgeff != nullptr) {
        LOG_INFO("SceneRenderGraphComposeOwnerSourceSkip: layer=%d name='%s' output='%.*s' "
                 "policy=%.*s",
                 NodeLayerId(scene, node),
                 node->Name().c_str(),
                 static_cast<int>(output.size()),
                 output.data(),
                 static_cast<int>(EffectSourcePolicyName(imgeff->SourceContributionPolicy()).size()),
                 EffectSourcePolicyName(imgeff->SourceContributionPolicy()).data());
    }
    // Text is now a first-class scene primitive. Whenever a node owns text we emit the dedicated
    // text pass directly from that primitive, keeping the render graph aligned with the same
    // authoritative text object that parser and runtime updates mutate.
    if (node != nullptr && node->Text() != nullptr) {
        auto source_options = BuildOwnerSourcePassOptions(
            imgeff, output, inherited_output, route, source_route);
        if (source_options.clear_before_draw && !node->Text()->object.opaquebackground) {
            // Non-opaque sources replace the complete private target with sampled framebuffer
            // RGB and zero alpha before glyphs accumulate coverage. The source card's UVs
            // control raster coverage; its sample coordinates use the incoming aligned owner
            // snapshot, not the private glyph camera or later composition-child raster I.
            // This is an ordinary material read of the named full framebuffer, including its
            // normal resolve/order rules. Do not retarget it to reflection or parent storage.
            AddDrawPass(*node->Text()->bridge.framebuffer_source, output, imgId, extra,
                        node_execute_gate, DrawPassOptions {
                            .camera_override = imgeff->BridgeCameraName(),
                            .use_active_camera_for_uniforms = true,
                            .model_space = ShaderModelSpace::LayerSnapshot,
                            .reflection_pass = route.reflection_pass,
                            .reflection_raster = route.reflection_raster,
                            .reflection_snapshot = route.reflection_raster,
                            .effect_snapshot_camera = source_route.active_compose_source_camera });
        }
        AddTextNodePass(node, output, imgId, extra, node_execute_gate, std::move(source_options));
    }

    const auto children = OrderedRenderGraphChildren(node, extra);
    const auto* owner = scene.FindSceneObject(imgId);
    const bool defer_authored_children = route.compose_source && owner != nullptr &&
        owner->LayerNode() == node && !owner->Passthrough();
    if (node != nullptr) {
        for (const auto& child : children) {
            if (child.node == nullptr) continue;
            // An ordinary child completes its own source/effect/publication sequence before its
            // children are visited. Those descendants still draw into the enclosing composition,
            // not into this object's private effect texture. Only passthrough owns a child source
            // phase; physical mesh/emitter resources remain part of their current owner draw.
            if (defer_authored_children && child.authored) continue;
            const bool child_compose_source_route =
                route.compose_source ||
                (child.authored && imgeff != nullptr &&
                 source_route.proxy_children_contribute_to_effect_source &&
                 !IsEffectLocalDependency(child.node, extra));
            const std::string child_compose_source_camera =
                child_compose_source_route ? source_route.child_compose_source_camera
                                           : std::string();
            const AlphaWritePolicy child_compose_source_alpha_write_policy =
                child_compose_source_route && imgeff != nullptr &&
                    source_route.proxy_children_contribute_to_effect_source
                    ? imgeff->CompositionChildAlphaWritePolicy()
                    : route.compose_source_alpha_write_policy;
            ToGraphPass(child.node,
                        output,
                        NodeLayerId(scene, child.node),
                        extra,
                        node_execute_gate,
                        TraversalRoute {
                            .compose_source = child_compose_source_route,
                            .compose_source_camera = child_compose_source_camera,
                            .compose_source_alpha_write_policy =
                                child_compose_source_alpha_write_policy,
                            .reflection_pass = route.reflection_pass,
                            .reflection_raster = route.reflection_raster &&
                                (imgeff == nullptr || imgeff->IsFullscreen()) });
        }
    }

    if (imgeff != nullptr) {
        // Finish this owner's source before its effect dispatcher, using the same owner-derived
        // publication contract in both. Ordinary children draw their authored final segment into
        // the enclosing composition; only the child's own dependency/puppet/blend contract
        // requires a private publisher.
        const auto final_output_capability = imgeff->ResolveFinalOutputCapability();
        const bool keep_final_output_private =
            final_output_capability != FinalOutputCapability::SceneAuthoredWriter;
        const auto private_target_it = scene.renderTargets.find(imgeff->SourceTarget());
        const auto final_target_it = scene.renderTargets.find(std::string(inherited_output));
        const SceneRenderTarget* private_target =
            private_target_it != scene.renderTargets.end() ? &private_target_it->second : nullptr;
        const SceneRenderTarget* final_target =
            final_target_it != scene.renderTargets.end() ? &final_target_it->second : nullptr;
        const TextureSample private_sample =
            private_target != nullptr ? private_target->sample : TextureSample {};
        const TextureSample final_sample =
            final_target != nullptr ? final_target->sample : TextureSample {};
        LOG_INFO("SceneRenderGraphEffectResolve: layer=%d name='%s' inherited-output='%.*s' "
                 "effect-output='%.*s' offscreen-dependency=%s visible=%s local-visible=%s "
                 "keep-final-private=%s compose-source-route=%s compose-camera='%s' "
                 "capability=%.*s private-target='%s' "
                 "private-target-size=[%d %d] "
                 "private-sampler=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "final-target='%.*s' final-target-size=[%d %d] "
                 "final-sampler=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s]",
                 NodeLayerId(scene, node),
                 node != nullptr ? node->Name().c_str() : "",
                 static_cast<int>(inherited_output.size()),
                 inherited_output.data(),
                 static_cast<int>(output.size()),
                 output.data(),
                 IsOffscreenDependencyLayer(extra, imgId) ? "true" : "false",
                 node != nullptr && node->Visible() ? "true" : "false",
                 node != nullptr && node->LocalVisible() ? "true" : "false",
                 keep_final_output_private ? "true" : "false",
                 route.compose_source ? "true" : "false",
                 source_route.active_compose_source_camera.c_str(),
                 static_cast<int>(FinalOutputCapabilityName(final_output_capability).size()),
                 FinalOutputCapabilityName(final_output_capability).data(),
                 imgeff->SourceTarget().c_str(),
                 private_target != nullptr ? private_target->width : 0,
                 private_target != nullptr ? private_target->height : 0,
                 static_cast<int>(TextureWrapName(private_sample.wrapS).size()),
                 TextureWrapName(private_sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(private_sample.wrapT).size()),
                 TextureWrapName(private_sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(private_sample.magFilter).size()),
                 TextureFilterName(private_sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(private_sample.minFilter).size()),
                 TextureFilterName(private_sample.minFilter).data(),
                 static_cast<int>(inherited_output.size()),
                 inherited_output.data(),
                 final_target != nullptr ? final_target->width : 0,
                 final_target != nullptr ? final_target->height : 0,
                 static_cast<int>(TextureWrapName(final_sample.wrapS).size()),
                 TextureWrapName(final_sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(final_sample.wrapT).size()),
                 TextureWrapName(final_sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(final_sample.magFilter).size()),
                 TextureFilterName(final_sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(final_sample.minFilter).size()),
                 TextureFilterName(final_sample.minFilter).data());
        const std::string layer_private_output(output);
        auto layer_destination_gate =
            [node, &scene, layer_private_output, outer_gate = node_execute_gate]() {
            if (outer_gate && !outer_gate()) return false;
            return ShouldExecuteLayerDestination(scene, node, layer_private_output);
        };
        imgeff->ResolveEffect(scene.default_effect_mesh, "effect", inherited_output,
                              extra.frame_fbo_bindings, layer_destination_gate());

        for (usize i = 0; i < imgeff->EffectCount(); i++) {
            auto& eff     = imgeff->GetEffect(i);
            // Resource retention does not select execution. Images dispatch the visible sequence;
            // shapes dispatch only their physical last effect, even with that effect hidden.
            if (!imgeff->ShouldExecuteEffect(*eff)) continue;
            auto  cmdItor = eff->commands.begin();
            auto  cmdEnd  = eff->commands.end();
            int   nodePos = 0;
            auto  effect_visible_gate = [eff, imgeff, layer_destination_gate]() {
                return layer_destination_gate() &&
                    imgeff->ShouldExecuteEffect(*eff);
            };
            const auto emit_commands = [&]() {
                // Commands share the authored step order with materials. Several copies can
                // occupy one boundary, including the boundary after the last material; each uses
                // the pair that ResolveEffect selected at that exact composition step.
                while (cmdItor != cmdEnd && nodePos == cmdItor->afterpos) {
                    // A target index of -1 does not submit a copy. Its compose marker still
                    // participated in ResolveEffect, so retain the entry and advance normally.
                    if (cmdItor->cmd == SceneImageEffect::CmdType::Copy && cmdItor->authored_dst) {
                        rg::addCopyPass(*extra.rgraph,
                                        rg::createTexDesc(cmdItor->src, extra.scene),
                                        rg::createTexDesc(cmdItor->dst, extra.scene),
                                        effect_visible_gate);
                    }
                    cmdItor++;
                }
            };
            for (auto& effect_node : eff->nodes) {
                emit_commands();
                // An authored final writer restores the enclosing composition's camera and alpha
                // state, just like the owner's neutral publisher. Private intermediate and puppet
                // surface draws retain their own camera. Otherwise removing the extra publication
                // pass would send this ordinary child through the scene camera while writing the
                // parent's texture.
                const bool uses_composition_camera = effect_node.uses_owner_transform &&
                    route.compose_source;
                // Explicit shape FBO draws retain the incoming camera/owner I just like other
                // shape records, but they do not inherit the enclosing composition's alpha
                // accumulation rule. Target/viewport selection is distinct from raster matrices.
                const bool composition_writer = uses_composition_camera &&
                    !effect_node.output_is_fbo;
                const bool composition_camera = uses_composition_camera &&
                    !source_route.active_compose_source_camera.empty();
                // Effect material nodes are private render-graph passes. They can carry a camera
                // override for layer-surface publication, but they must not be traversed through
                // ToGraphPass(): an override camera may itself own the same SceneImageEffectLayer,
                // and treating this internal pass as another image-effect owner recursively
                // rebuilds the chain while it is being emitted, overwriting the already-resolved
                // puppet and layer-surface publication route.
                AddDrawPass(effect_node.sceneNode.get(),
                            effect_node.output,
                            imgId,
                            extra,
                            effect_visible_gate,
                            DrawPassOptions {
                                .alpha_write_policy = composition_writer
                                    ? route.compose_source_alpha_write_policy
                                    : effect_node.alpha_write_policy,
                                .blend_override = effect_node.blend_override,
                                .depth_test_override = effect_node.depth_test_override,
                                .depth_write_override = effect_node.depth_write_override,
                                .destination_alpha_override = effect_node.destination_alpha_override,
                                .clear_before_draw = effect_node.clear_before_draw,
                                .camera_override = composition_camera
                                    ? source_route.active_compose_source_camera
                                    : effect_node.camera_override,
                                .use_active_camera_for_parallax =
                                    composition_camera ||
                                    effect_node.use_active_camera_for_parallax,
                                .reflection_pass = route.reflection_pass,
                                .reflection_raster = route.reflection_raster &&
                                    effect_node.uses_owner_transform,
                                .reflection_snapshot = route.reflection_raster,
                                .effect_snapshot_camera =
                                    source_route.active_compose_source_camera });
                nodePos++;
            }
            emit_commands();
        }

        if (imgeff->HasFinalComposite()) {
            // Emit publication directly from the object's raster phase. It has no children,
            // source routing or transform node to traverse. The layer still selects whether this
            // phase runs, and parent composition supplies the same camera/alpha contract as an
            // authored final material drawing into that destination. The dispatcher finishes
            // hidden dependency work before its visible-publication arm. A parent composition
            // target is still an enclosing destination, so the generic offscreen-source exemption
            // must not admit this draw merely because its output is not the main framebuffer.
            // Query the owner each frame so script visibility changes preserve the same rule.
            auto final_composite_gate = [imgeff, &scene, outer_gate = node_execute_gate]() {
                if (outer_gate && !outer_gate()) return false;
                return scene.IsLayerVisible(imgeff->Owner().Id()) &&
                    imgeff->ShouldRunFinalComposite();
            };
            const SceneDraw publication(imgeff->FinalCompositeDraw());
            const bool compose_camera = route.compose_source &&
                !source_route.active_compose_source_camera.empty();
            LOG_INFO("SceneLayerPublication: layer=%d name='%s' output='%.*s' "
                     "visible=%s dependency=%s composition-steps=%zu",
                     imgeff->Owner().Id(), imgeff->Owner().Name().c_str(),
                     static_cast<int>(inherited_output.size()), inherited_output.data(),
                     scene.IsLayerVisible(imgeff->Owner().Id()) ? "true" : "false",
                     scene.IsLayerOffscreenDependencySource(imgeff->Owner().Id())
                         ? "true" : "false",
                     imgeff->VisibleCompositionStepCount());
            // Text utility publication enables owner-selected depth testing from scene version
            // 2 onward, independently of the utility shader family's version-3 transition.
            // Select this per invocation rather than changing the shared material: live depth
            // and layout updates rebuild these descriptions, unlike the direct background's
            // separately retained first-draw selection. Disabled owners and earlier versions
            // keep the utility material's own state. Writing and attachment ownership remain
            // independent, so private/composition color targets do not acquire scene depth.
            std::optional<bool> publication_depth_test {};
            if (const auto* text = imgeff->Owner().LayerNode()->Text();
                text != nullptr && scene.authoredVersion >= 2 &&
                text->object.depthtest == "enabled") {
                publication_depth_test = true;
            }
            AddDrawPass(publication, inherited_output, imgId, extra, final_composite_gate,
                        DrawPassOptions {
                            .alpha_write_policy = route.compose_source
                                ? route.compose_source_alpha_write_policy
                                : AlphaWritePolicy::Preserve,
                            .depth_test_override = publication_depth_test,
                            .camera_override = compose_camera
                                ? source_route.active_compose_source_camera : std::string(),
                            .use_active_camera_for_parallax = compose_camera,
                            .use_active_camera_for_uniforms =
                                DrawMaterialSamplesFramebuffer(publication) &&
                                !DrawUsesPerspectiveCamera(scene, publication),
                            .reflection_pass = route.reflection_pass,
                            .reflection_raster = route.reflection_raster,
                            .reflection_snapshot = route.reflection_raster,
                            .effect_snapshot_camera =
                                source_route.active_compose_source_camera });
        }
    }

    if (defer_authored_children) {
        for (const auto& child : children) {
            if (!child.authored || child.node == nullptr) continue;
            ToGraphPass(child.node, inherited_output, NodeLayerId(scene, child.node),
                        extra, node_execute_gate, route);
        }
    }
}

static bool AddReflectionStage(Scene& scene, ExtraInfo& extra) {
    // Receiver presence enables the scene phase; reflected=true only selects producers inside
    // that phase. Inspect complete owners before emitting anything so a receiver appearing
    // earlier in painter order samples all producers, including owners appearing after it in
    // objects[].
    const bool has_receiver = std::any_of(scene.layerOrder.begin(), scene.layerOrder.end(),
        [&scene](int32_t id) {
            const auto* owner = scene.FindSceneObject(id);
            return owner != nullptr && owner->ReceivesReflection();
        });
    if (!has_receiver) return false;

    extra.rgraph->addPass<vulkan::ClearPass>(
        "reflection_clear", rg::PassNode::Type::Clear,
        [&scene](rg::RenderGraphBuilder& builder, vulkan::ClearPass::Desc& desc) {
            if (scene.shadows.quality != 0) {
                // This read orders the reflected phase after shadow rendering even when no
                // reflected material samples the atlas; it does not add a texture to those
                // materials.
                builder.read(builder.createTexNode(
                    rg::createTexDesc(std::string(SpecTex_ShadowAtlas), &scene)));
            }
            auto* target = builder.createTexNode(
                rg::createTexDesc(std::string(SpecTex_Reflection), &scene), true);
            builder.write(target);
            desc.target = target->key();
            desc.use_scene_clear_color = true;
            // Quality off has its own unconditional color-only clear; quality on uses the scene
            // clear setting for both attachments, while keeping the producer traversal
            // independent.
            desc.should_clear_color = [&scene]() {
                return !scene.reflectionsEnabled || scene.clearEnabled;
            };
            desc.should_clear_model_depth = [&scene]() {
                return scene.reflectionsEnabled && scene.clearEnabled;
            };
        });

    size_t producers = 0;
    for (const auto layer_id : scene.layerOrder) {
        if (!scene.reflectionsEnabled) break;
        const auto* owner = scene.FindSceneObject(layer_id);
        if (owner == nullptr ||
            owner->LayerNode() == nullptr || !owner->Reflected() || owner->ReceivesReflection() ||
            HasCompositionAncestor(scene, *owner)) continue;
        ++producers;
        ToGraphPass(owner->LayerNode(), SpecTex_Reflection, layer_id, extra, {},
                    TraversalRoute { .reflection_pass = true, .reflection_raster = true });
    }
    LOG_INFO("SceneReflectionStage: producers=%zu quality-enabled=%s "
             "shared-owner-resources=true before-main-scene=true",
             producers, scene.reflectionsEnabled ? "true" : "false");
    return true;
}

static std::unique_ptr<rg::RenderGraph> SceneToRenderGraphImpl(Scene& scene) {
    std::unique_ptr<rg::RenderGraph> rgraph = std::make_unique<rg::RenderGraph>();
    ExtraInfo                        extra { .rgraph = rgraph.get(), .scene = &scene };
    for (size_t index = 0; index < scene.layerOrder.size(); index++) {
        extra.layer_order_index[scene.layerOrder[index]] = index;
    }
    LOG_INFO("SceneRenderGraphOrderInit: layer-count=%zu", scene.layerOrder.size());
    if (scene.shadows.quality != 0) {
        // Publish the real atlas writer before building reflected/main consumers. A writer
        // appended after those consumers creates a later texture version and leaves them reading
        // an unwritten first-frame image (and previous-frame shadows thereafter). Keep ordinary
        // owner order unchanged; this is the independent light/shadow phase, not a dependency
        // sort.
        rg::addShadowAtlasPass(*rgraph, &scene);
    }
    const bool has_reflected_stage = AddReflectionStage(scene, extra);
    extra.model_depth_outputs_seen.clear();
    // The main scene stage clears after the reflected walk, immediately before the ordinary scene
    // walk. Keeping this boundary in the graph also lets reflection read the previous main
    // target; a renderer-wide prepass would discard that target too early.
    rgraph->addPass<vulkan::PrePass>(
        "scene_clear", rg::PassNode::Type::Clear,
        [&scene, has_reflected_stage](rg::RenderGraphBuilder& builder, vulkan::PrePass::Desc&) {
            if (has_reflected_stage) {
                // This is a phase-order edge, not a sampler binding. Graph creation
                // order alone does not constrain an independent main-color clear:
                // it must wait for the final reflected writer before discarding main.
                builder.read(builder.createTexNode(
                    rg::createTexDesc(std::string(SpecTex_Reflection), &scene)));
            } else if (scene.shadows.quality != 0) {
                // With no reflected phase, main follows the shadow phase directly.
                builder.read(builder.createTexNode(
                    rg::createTexDesc(std::string(SpecTex_ShadowAtlas), &scene)));
            }
            auto* target = builder.createTexNode(
                rg::createTexDesc(std::string(SpecTex_Default), &scene), true);
            builder.write(target);
        });
    ToGraphPass(scene.sceneGraph.get(), SpecTex_Default, scene.sceneGraph->ID(), extra);

    if (extra.use_mipmap_framebuffer) {
        rg::addCopyPass(*rgraph,
                        rg::TexNode::Desc { .name = SpecTex_Default.data(),
                                            .key  = SpecTex_Default.data(),
                                            .type = rg::TexNode::TexType::Temp },
                        rg::TexNode::Desc { .name = WE_MIP_MAPPED_FRAME_BUFFER.data(),
                                            .key  = WE_MIP_MAPPED_FRAME_BUFFER.data(),
                                            .type = rg::TexNode::TexType::Temp });
    }

    if (scene.volumetrics.active && !scene.volumetrics.lights.empty()) {
        // Volumetrics then bloom/HDR. Each light writes LightBuffer additively; blur/combine
        // run after every light has finished.
        rg::addClearPass(*rgraph,
                         rg::createTexDesc(std::string(SpecTex_VolumetricsLightBuffer), &scene),
                         { 0.0f, 0.0f, 0.0f, 0.0f });
        rg::addVolumetricsSingleFillPass(*rgraph, &scene);
        for (const auto& pass : scene.volumetrics.lights) {
            if (pass.light == nullptr) continue;
            SceneLight* light = pass.light;
            auto camera_inside = [&scene, light]() {
                if (scene.activeCamera == nullptr || light == nullptr) return false;
                // Use the same frame eye as the lighting uniforms; the orthographic projection
                // node can lie inside a light even when the actual frame eye is outside its hull.
                return light->CameraInsideVolume(scene.FrameEyePosition(),
                                                 scene.activeCamera->GetDirection().cast<float>());
            };
            if (pass.back) {
                AddDrawPass(pass.back.get(), SpecTex_VolumetricsBack, 0, extra);
            }
            if (pass.front) {
                AddDrawPass(pass.front.get(),
                            SpecTex_VolumetricsLightBuffer,
                            0,
                            extra,
                            [camera_inside]() { return ! camera_inside(); });
            }
            if (pass.fullscreen) {
                AddDrawPass(pass.fullscreen.get(),
                            SpecTex_VolumetricsLightBuffer,
                            0,
                            extra,
                            camera_inside);
            }
        }
        if (scene.volumetrics.quality < 3 && scene.volumetrics.blur_h &&
            scene.volumetrics.blur_v) {
            AddDrawPass(scene.volumetrics.blur_h.get(), SpecTex_VolumetricsLightBufferB, 0, extra);
            AddDrawPass(scene.volumetrics.blur_v.get(), SpecTex_VolumetricsLightBuffer, 0, extra);
        }
        if (scene.volumetrics.combine) {
            AddDrawPass(scene.volumetrics.combine.get(), SpecTex_Default, 0, extra);
        }
    }

    if (scene.bloom.quality > 0 && scene.bloom.enabled && !scene.bloom.nodes.empty()) {
        // Wallpaper Engine treats `general.bloom` as the execution switch for the complete Bloom
        // chain. Authored strength and threshold values remain stored while the switch is off, but
        // they do not make the post-process execute. Keeping disabled Bloom out of the graph avoids
        // three private blur targets, one full-frame feedback copy, and four GPU passes while still
        // retaining the parsed nodes for a later topology rebuild when a runtime binding enables it.
        // Scene Bloom is authored in `general`, so an enabled chain belongs after the complete layer
        // traversal, with each synthetic node bound to its explicit quarter/eighth output target.
        if (scene.bloom.nodes.size() != scene.bloom.outputs.size()) {
            LOG_ERROR("SceneBloomGraphBind: pass/output mismatch passes=%zu outputs=%zu",
                      scene.bloom.nodes.size(),
                      scene.bloom.outputs.size());
        } else {
            LOG_INFO("SceneBloomGraphBind: passes=%zu enabled=%s strength=%.3f threshold=%.3f",
                     scene.bloom.nodes.size(),
                     scene.bloom.enabled ? "true" : "false",
                     scene.bloom.strength,
                     scene.bloom.threshold);
            for (usize i = 0; i < scene.bloom.nodes.size(); ++i) {
                if (scene.bloom.nodes[i] == nullptr) {
                    LOG_ERROR("SceneBloomGraphBind: missing pass index=%zu output='%s'",
                              i,
                              scene.bloom.outputs[i].c_str());
                    continue;
                }
                LOG_INFO("SceneBloomGraphBind: pass=%zu node='%s' output='%s'",
                         i,
                         scene.bloom.nodes[i]->Name().c_str(),
                         scene.bloom.outputs[i].c_str());
                AddDrawPass(scene.bloom.nodes[i].get(), scene.bloom.outputs[i], 0, extra, {},
                            DrawPassOptions { .resolved_scene_color = true });
            }
        }
    } else if (scene.bloom.quality > 0 && scene.bloom.enabled && scene.bloom.node != nullptr) {
        // This fallback is intentionally retained for older parsed scene objects that may still
        // populate only the legacy single-node field before a full reparse has occurred.
        LOG_INFO("SceneBloomGraphBind: legacy-output='%s' enabled=%s strength=%.3f threshold=%.3f",
                 SpecTex_Default.data(),
                 scene.bloom.enabled ? "true" : "false",
                 scene.bloom.strength,
                 scene.bloom.threshold);
        AddDrawPass(scene.bloom.node.get(), SpecTex_Default, 0, extra, {},
                    DrawPassOptions { .resolved_scene_color = true });
    }

    scene.effectCommandPlanUsesVisibility = std::any_of(
        extra.frame_fbo_bindings.begin(), extra.frame_fbo_bindings.end(),
        [](const auto& entry) {
            const auto& commands = entry.first->commands;
            return std::any_of(commands.begin(), commands.end(), [](const auto& command) {
                return command.cmd == SceneImageEffect::CmdType::Swap &&
                    command.authored_src && command.authored_dst;
            });
        });
    // Reflection and main invocations have already advanced this frame-local table in order.
    // Commit only their final result, once a frame is submitted; graph warm-up never advances
    // history. An even number of swaps can leave the table unchanged and reuse the same graph.
    if (scene.effectCommandPlanUsesVisibility) {
        rgraph->onFrameSubmitted(
            [bindings = std::move(extra.frame_fbo_bindings), &scene]() {
                bool changed = false;
                for (const auto& [effect, final_bindings] : bindings) {
                    changed |= effect->CommitFboBindings(final_bindings);
                }
                if (changed) scene.MarkRenderGraphTopologyDirty();
            });
    }

    return rgraph;
}

std::unique_ptr<rg::RenderGraph> wallpaper::sceneToRenderGraph(Scene& scene) {
    return SceneToRenderGraphImpl(scene);
}

std::unique_ptr<rg::RenderGraph> wallpaper::sceneToPipelineWarmupRenderGraph(Scene& scene) {
    return SceneToRenderGraphImpl(scene);
}

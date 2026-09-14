#include "SceneImageEffectLayer.h"
#include "SceneNode.h"
#include "SceneObject.h"
#include "Scene.h"
#include "SceneMesh.h"
#include "SceneTextPrimitive.h"
#include "SceneDestinationTarget.h"

#include "SpecTexs.hpp"
#include "Core/StringHelper.hpp"
#include "Utils/Logging.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

using namespace wallpaper;

namespace
{
std::string ResolvePingPongInputAlias(std::string_view value,
                                      std::string_view initial_ppong_a,
                                      std::string_view initial_ppong_b,
                                      std::string_view ppong_a,
                                      std::string_view ppong_b) {
    // The old private render-target names also doubled as the symbolic A/B aliases because they
    // started with `_rt_effect_pingpong_a|b_`. Destination names (`sc.W.H.*`) contain no layer
    // identity and therefore do not carry that accidental type information. Match the bridge's
    // initial named targets explicitly so every authored `previous` input continues to follow the
    // current side after each effect swaps the pair. Without this distinction, effect two reads
    // and writes the same named target and the render graph must allocate a full-size snapshot
    // for every repeated language branch.
    if (sstart_with(value, WE_EFFECT_PPONG_PREFIX_A) || value == initial_ppong_a)
        return std::string(ppong_a);
    if (sstart_with(value, WE_EFFECT_PPONG_PREFIX_B) || value == initial_ppong_b)
        return std::string(ppong_b);
    return std::string(value);
}

std::string ResolvePingPongOutputAlias(std::string_view value,
                                       std::string_view initial_ppong_a,
                                       std::string_view initial_ppong_b,
                                       std::string_view ppong_a,
                                       std::string_view ppong_b) {
    if (value == SpecTex_Default) return std::string(ppong_b);
    return ResolvePingPongInputAlias(
        value, initial_ppong_a, initial_ppong_b, ppong_a, ppong_b);
}

bool IsCurrentEffectOutput(std::string_view authored_output) {
    return authored_output == SpecTex_Default ||
        sstart_with(authored_output, WE_EFFECT_PPONG_PREFIX_B);
}

std::string_view ResolveTemplateOrCurrent(const std::string& authored_value,
                                          const std::string& current_value) {
    return authored_value.empty() ? std::string_view(current_value)
                                  : std::string_view(authored_value);
}
} // namespace

std::string_view wallpaper::FinalOutputCapabilityName(FinalOutputCapability capability) {
    switch (capability) {
    case FinalOutputCapability::PrivateDependency: return "private-dependency";
    case FinalOutputCapability::PrivatePuppetPublication: return "private-puppet-publication";
    case FinalOutputCapability::SceneAuthoredWriter: return "scene-authored-writer";
    case FinalOutputCapability::PrivateThenPublish: return "private-then-publish";
    }
    return "unknown";
}

namespace
{
struct EffectOutputDiagnosticRoles {
    std::string_view authored_writer;
    std::string_view publication_writer;
    std::string_view private_parallax_owner;
    std::string_view publication_parallax_owner;
    uint32_t         parallax_application_count { 1 };
};

EffectOutputDiagnosticRoles ResolveEffectOutputDiagnosticRoles(
    FinalOutputCapability capability,
    bool keep_authored_final_private,
    bool has_authored_final_output) {
    if (!has_authored_final_output) {
        return { "none", "neutral-composite", "none", "neutral-composite", 1 };
    }
    if (!keep_authored_final_private) {
        return { "scene-authored-final", "authored-final", "none", "authored-final", 1 };
    }
    if (capability == FinalOutputCapability::PrivatePuppetPublication) {
        return { "private-authored-final", "skinned-publication", "none", "skinned-publication", 1 };
    }
    return { "private-authored-final", "neutral-composite", "none", "neutral-composite", 1 };
}

} // namespace

SceneImageEffectLayer::SceneImageEffectLayer(SceneObject& owner, float w, float h,
                                             std::string_view pingpong_a,
                                             std::string_view pingpong_b)
    : m_owner(owner),
      m_pingpong_a(pingpong_a),
      m_pingpong_b(pingpong_b),
      m_card_size { w, h },
      m_effect_matrix_size { w, h },
      m_source_mesh(std::make_unique<SceneMesh>()),
      m_final_mesh(std::make_unique<SceneMesh>()),
      m_direct_draw_mesh(m_source_mesh.get()),
      m_final_composite(owner) {};

void SceneImageEffect::SwapFboBindings(FboBindings& bindings, const std::string& source,
                                      const std::string& target) {
    // A swap exchanges references to two fixed FBO indices in every non-swap record. Composing
    // that permutation on the table values affects earlier records next frame and later records
    // in this frame.
    for (auto& [_, binding] : bindings) {
        if (binding == source) binding = target;
        else if (binding == target) binding = source;
    }
}

void SceneImageEffect::ResolveCommand(Command& command, FboBindings& bindings,
                                       std::string_view current_destination) {
    // The caller selects the active destination for this record; command operands themselves only
    // name declared FBOs. Keeping the dispatcher independent of image ping-pong advancement also
    // serves shape's source-slot -1 sequence.
    if (command.cmd == CmdType::Swap) {
        command.src = command.authored_src.value_or(std::string());
        command.dst = command.authored_dst.value_or(std::string());
        if (command.authored_src && command.authored_dst) {
            SwapFboBindings(bindings, command.src, command.dst);
        }
    } else {
        command.src = command.authored_src ? bindings.at(*command.authored_src)
                                           : std::string(current_destination);
        command.dst = command.authored_dst ? bindings.at(*command.authored_dst) : std::string();
    }
    if (std::getenv("WESCENE_TRACE_EFFECT_PHASES") != nullptr) {
        LOG_INFO("SceneEffectCommandResolve: layer=%d effect=%d position=%d "
                 "command=%s compose=%s final-destination=%s source='%s' target='%s' copy=%s",
                 OwnerLayerId(), EffectId(), command.afterpos,
                 command.cmd == CmdType::Swap ? "swap" : "copy",
                 command.advances_composition ? "true" : "false",
                 command.uses_final_destination ? "true" : "false",
                 command.src.c_str(), command.dst.c_str(),
                 command.cmd == CmdType::Copy && command.authored_dst ? "true" : "false");
    }
}

bool SceneImageEffect::CommitFboBindings(const FboBindings& bindings) {
    if (m_fbo_bindings == bindings) return false;
    m_fbo_bindings = bindings;
    if (std::getenv("WESCENE_TRACE_EFFECT_PHASES") != nullptr) {
        LOG_INFO("SceneEffectSwapCommit: layer=%d effect=%d bindings=%zu",
                 OwnerLayerId(), EffectId(), m_fbo_bindings.size());
    }
    return true;
}

void SceneImageEffectLayer::SetDestinationTargets(std::string first_target,
                                                  std::string second_target) {
    // Re-layout selects shared images by extent and recreates a private slot under its owner
    // name, while the authored effect chain retains its input/output roles. Update both the
    // parsed templates and the resolved bindings: graph reconstruction starts from the former,
    // whereas an already materialized final publisher still holds the latter.
    const auto rebind = [&](std::string& name) {
        if (name == m_pingpong_a) name = first_target;
        else if (name == m_pingpong_b) name = second_target;
    };
    for (auto& effect : m_effects) {
        for (auto& command : effect->commands) {
            if (command.authored_src) rebind(*command.authored_src);
            if (command.authored_dst) rebind(*command.authored_dst);
            rebind(command.src);
            rebind(command.dst);
        }
        for (auto& node : effect->nodes) {
            rebind(node.authored_output);
            rebind(node.output);
            for (auto& texture : node.authored_textures) rebind(texture);
            for (auto& texture : node.sceneNode->Mesh()->Material()->textures) rebind(texture);
        }
    }
    if (HasFinalComposite()) {
        for (auto& texture : m_final_composite.draw.Mesh()->Material()->textures) rebind(texture);
    }
    for (auto& name : m_runtime_render_target_names) rebind(name);
    m_pingpong_a = std::move(first_target);
    m_pingpong_b = std::move(second_target);
}

void SceneImageEffectLayer::RefreshDestinationTargets(
    Scene& scene, std::optional<std::array<int32_t, 2>> destination_extent,
    std::optional<TextureSample> destination_sampler) {
    // First/last-child callbacks rerun the parent's destination setup against its current
    // ancestry. The parsed source policy distinguishes current card dimensions from loaded
    // texture pixels. A concrete layer with its own destination sizing policy supplies that
    // extent; a primary-texture refresh also supplies its current sampler. A size property write
    // alone still changes geometry without rerunning resource setup. Copy the descriptor before
    // interning because insertion may rehash the scene target table.
    auto target = scene.renderTargets.at(m_pingpong_a);
    const std::array<int32_t, 2> previous_extent { target.width, target.height };
    if (destination_sampler) target.sample = *destination_sampler;
    if (destination_extent.has_value()) {
        target.width = target.mapWidth = (*destination_extent)[0];
        target.height = target.mapHeight = (*destination_extent)[1];
    } else if (m_destination_uses_card_size) {
        const auto extent = ResolveCardDestinationExtent(m_card_size);
        target.width = target.mapWidth = extent[0];
        target.height = target.mapHeight = extent[1];
    }
    const bool private_output =
        DeclaredFinalOutputCapability() != FinalOutputCapability::SceneAuthoredWriter;
    const auto names = ResolveSceneDestinationRenderTargets(
        scene, m_owner.Id(), m_owner.ParentId(), private_output, target);
    SetDestinationTargets(names[0], names[1]);
    ResizeEffectRenderTargets(scene, { static_cast<float>(target.width),
                                       static_cast<float>(target.height) });
    scene.MarkRenderGraphTopologyDirty();
    LOG_INFO("SceneCompositionChildBoundary: layer=%d parent=%d children=%zu "
             "target-a='%s' target-b='%s' previous=%dx%d extent=%dx%d card-derived=%s",
             m_owner.Id(), m_owner.ParentId(), scene.GetLayerChildren(m_owner.Id()).size(),
             names[0].c_str(), names[1].c_str(), previous_extent[0], previous_extent[1],
             target.width, target.height, m_destination_uses_card_size ? "true" : "false");
}

void SceneImageEffectLayer::AddEffectRenderTarget(std::string name, uint32_t scale, uint32_t fit) {
    AddRuntimeRenderTargetName(name);
    m_effect_render_targets.push_back({ std::move(name), scale, fit });
}

bool SceneImageEffectLayer::ResizeEffectRenderTargets(
    Scene& scene, std::array<float, 2> source_extent) {
    // Unlike destination slots, an existing FBO keeps its authored name and is resized in place.
    // Run every retained record at this owner's setup boundary, including when its own
    // destination extent did not change: another owner may have resized a shared FBO since this
    // one last ran. Per-frame transform updates do not enter this operation. Only changed
    // descriptors invalidate their GPU backing/history.
    bool changed = false;
    for (const auto& fbo : m_effect_render_targets) {
        auto& target = scene.renderTargets.at(fbo.name);
        const auto extent = ResolveEffectRenderTargetExtent(source_extent, fbo.scale, fbo.fit);
        if (target.width == extent[0] && target.height == extent[1] &&
            target.mapWidth == extent[0] && target.mapHeight == extent[1]) continue;
        LOG_INFO("SceneEffectFboResize: layer=%d target='%s' previous=%dx%d extent=%dx%d "
                 "source=[%.3f %.3f] scale=%u fit=%u",
                 m_owner.Id(), fbo.name.c_str(), target.width, target.height,
                 extent[0], extent[1], source_extent[0], source_extent[1], fbo.scale, fbo.fit);
        target.width = target.mapWidth = extent[0];
        target.height = target.mapHeight = extent[1];
        scene.MarkRenderTargetResourcesDirty(fbo.name);
        changed = true;
    }
    return changed;
}

bool SceneImageEffectLayer::CopyBackground() const {
    const auto* state = m_owner.ImageRuntimeState();
    return state != nullptr && state->copy_background;
}

SceneImageEffectLayer::SourcePolicy SceneImageEffectLayer::SourceContributionPolicy() const {
    if (UsesShapeDraw()) return SourcePolicy::None;
    if (!m_owner.Passthrough()) return SourcePolicy::OwnerNode;
    // Passthrough source admission follows copybackground independently of raster size. A
    // fullscreen composition preserves incoming child matrices, but still starts from transparent
    // when background copying is disabled. Sampling the owner's framebuffer in that state would
    // publish unrelated scene content again. Child count and private publication do not change
    // which source contributions belong to this composition.
    if (CopyBackground()) return SourcePolicy::OwnerNodeAndProxyChildren;
    return SourcePolicy::ProxyChildrenOnly;
}

BlendMode SceneImageEffectLayer::FinalBlend() const {
    // The parser resolves colorBlendMode precedence for both alternatives once; live
    // copybackground writes select between them during graph rebuild.
    return m_owner.Passthrough() && !CopyBackground()
        ? m_transparent_composition_blend : m_final_blend;
}

FinalOutputCapability SceneImageEffectLayer::ResolveFinalOutputCapability() const {
    // Private output follows the owner's publication flags. Neither composition ancestry nor an
    // effect visibility binding adds another private stage to an ordinary authored destination
    // segment.
    if (m_final_output_capability == FinalOutputCapability::PrivatePuppetPublication &&
        m_direct_puppet_source && !m_direct_puppet_source->private_publication &&
        VisibleCompositionStepCount() == 0) {
        return FinalOutputCapability::SceneAuthoredWriter;
    }
    return m_final_output_capability;
}

void SceneImageEffectLayer::RefreshPuppetPublicationState() {
    if (!m_direct_puppet_source || m_direct_puppet_source->private_publication ||
        VisibleCompositionStepCount() == 0) return;
    // Visibility refresh can promote an owner to private publication, and that state persists
    // when its effects become hidden again. Apply the transition when visibility changes, not
    // only when a frame is rendered: a show/hide pair between frames must still retain the
    // promoted publication state.
    m_direct_puppet_source->private_publication = true;
    LOG_INFO("ScenePuppetPublicationPromote: layer=%d name='%s' visible-steps=%zu",
             m_owner.Id(), m_owner.Name().c_str(), VisibleCompositionStepCount());
}

std::size_t SceneImageEffect::CompositionStepCount() const {
    std::size_t steps = 1;
    for (const auto& node : nodes) steps += node.advances_composition;
    for (const auto& command : commands) steps += command.advances_composition;
    return steps;
}

std::size_t SceneImageEffectLayer::VisibleCompositionStepCount() const {
    std::size_t steps = 0;
    for (const auto& effect : m_effects) {
        if (effect->LocalVisible()) steps += effect->CompositionStepCount();
    }
    return steps;
}

std::size_t SceneImageEffectLayer::SourceSlot() const {
    // Start at private-step parity so authored private completions arrive at physical slot zero.
    // Ordinary draws defer their last completion to the enclosing destination, while private
    // publication executes every authored step before publishing. With no visible authored steps,
    // the source-only bridge uses slot zero. Surface draws are deliberately excluded: their GPU
    // work does not belong to the authored composition count.
    const auto steps = VisibleCompositionStepCount();
    if (steps == 0) return 0;
    const bool private_publication =
        ResolveFinalOutputCapability() != FinalOutputCapability::SceneAuthoredWriter;
    return (steps - (private_publication ? 0 : 1)) % 2;
}

const std::string& SceneImageEffectLayer::SourceTarget() const {
    return SourceSlot() == 0 ? m_pingpong_a : m_pingpong_b;
}

bool SceneImageEffectLayer::UsesDirectDraw() const {
    // A resident effect bridge does not imply a source pass: zero visible steps with private
    // publication disabled send an ordinary image or text directly to the incoming destination.
    // Text publishes its existing glyph layout in that state; it needs neither the source
    // projection nor a second card draw just because hidden effect resources still exist.
    // Passthrough owns a separate child-source phase even when its effects are hidden.
    return (m_owner.Kind() == SceneObjectKind::Image || m_owner.Kind() == SceneObjectKind::Text) &&
        !m_owner.Passthrough() &&
        VisibleCompositionStepCount() == 0 &&
        ResolveFinalOutputCapability() == FinalOutputCapability::SceneAuthoredWriter;
}

bool SceneImageEffectLayer::UsesPrelightingSource() const {
    // Only visible authored effects, an actual private publisher, or a sprite source activate
    // this variant. A hidden effect's resident program and the internal puppet surface draw do
    // not contribute to the visible-effect count.
    return m_prelighting_source && !UsesDirectDraw() &&
        (VisibleCompositionStepCount() != 0 || m_prelighting_source->sprite ||
         ResolveFinalOutputCapability() != FinalOutputCapability::SceneAuthoredWriter);
}

const std::array<float, 2>& SceneImageEffectLayer::SourceTextureContentSize() const {
    return m_owner.ImageSource()->TextureMetadata().content_size;
}

void SceneImageEffectLayer::RefreshSourceTexture(
    Scene& scene, const SceneImageSource::Metadata& previous,
    const SceneImageSource::Metadata& next) {
    const auto& policy = m_source_texture_policy;
    const auto& allocation = next.allocation_size;
    const auto& content = next.content_size;
    const bool card_sized = policy.card_sized_destination && !next.sprite;
    const auto extent = card_sized ? ResolveCardDestinationExtent(m_card_size)
        : std::array { ClampDestinationRenderTargetExtent(static_cast<int32_t>(std::lround(content[0]))),
                       ClampDestinationRenderTargetExtent(static_cast<int32_t>(std::lround(content[1]))) };
    const auto sampler = DestinationRenderTargetSampler(
        policy.force_point_sampling || next.sample.magFilter == TextureFilter::NEAREST,
        policy.clamp_uvs);
    const bool texture_card = m_prelighting_source && m_prelighting_source->texture_card;

    LOG_INFO("SceneImageSourceTextureRefresh: layer=%d previous='%s' current='%s' "
             "allocation=[%d %d]->[%d %d] content=[%.3f %.3f]->[%.3f %.3f] "
             "texture-card=%s destination=%dx%d card-derived=%s filter=%s wrap=%s",
             m_owner.Id(), previous.texture_key.c_str(), next.texture_key.c_str(),
             previous.allocation_size[0], previous.allocation_size[1], allocation[0], allocation[1],
             previous.content_size[0], previous.content_size[1], content[0], content[1],
             texture_card ? "true" : "false", extent[0], extent[1],
             card_sized ? "true" : "false", TextureFilterName(sampler.magFilter).data(),
             TextureWrapName(sampler.wrapS).data());
    if (texture_card && previous.allocation_size != allocation) {
        // Mutate the retained source payload, not the owner's currently selected mesh. Direct
        // draws keep their authored card/UVs, imported puppet sources keep their auxiliary
        // coordinates, and reflected/main source consumers share the new payload revision.
        const float half_width = static_cast<float>(allocation[0]) * 0.5f;
        const float half_height = static_cast<float>(allocation[1]) * 0.5f;
        m_prelighting_source->mesh->GetVertexArray(0).SetVertex(WE_IN_POSITION, std::array {
            -half_width, half_height, 0.0f, -half_width, -half_height, 0.0f,
            half_width, half_height, 0.0f, half_width, -half_height, 0.0f,
        });
        m_prelighting_source->mesh->SetDirty();
    }
    m_destination_uses_card_size = card_sized;
    // Keep texture geometry notification distinct from effect-driven program selection. Do not
    // replace authored material values, refresh script instances, or change the retained sprite
    // selector here. Resource setup re-interns destination slots and resizes this owner's FBOs;
    // graph resolution then installs the corresponding source projection before submission.
    RefreshDestinationTargets(scene, extent, sampler);
}

void SceneImageEffectLayer::ResolveOwnerDraw(Scene& scene) {
    auto* mesh_ptr = m_owner.LayerNode()->Mesh();
    if (mesh_ptr == nullptr) return; // Text owns a dedicated primitive, without an image mesh.
    auto& mesh = *mesh_ptr;
    const bool direct = UsesDirectDraw();
    const bool prelighting = UsesPrelightingSource();
    // A composition source always draws the owner's card. Only an ordinary image selects the
    // alternate texture-space card or auxiliary puppet stream. Drawing into the restored
    // destination uses the authored image geometry/material instead. Re-select both blend and
    // mesh on every graph build so hiding and showing effects cannot retain the private source's
    // opaque blend or texture-space geometry.
    mesh.ChangeMeshDataFrom(prelighting && !m_owner.Passthrough()
                               ? *m_prelighting_source->mesh
                               : (direct ? *m_direct_draw_mesh : *m_source_mesh));
    mesh.Material()->blenmode = direct ? FinalBlend() : BlendMode::Normal;
    if (m_direct_puppet_source) {
        mesh.Material()->customShader.shader = direct
            ? m_direct_puppet_source->skinned_shader : m_direct_puppet_source->ordinary_shader;
        LOG_INFO("ScenePuppetDrawSelector: layer=%d name='%s' direct=%s visible-steps=%zu "
                 "private-publication=%s bones=%u",
                 m_owner.Id(), m_owner.Name().c_str(), direct ? "true" : "false",
                 VisibleCompositionStepCount(),
                 m_direct_puppet_source->private_publication ? "true" : "false",
                 mesh.Skinning().boneCount);
    }
    if (!m_prelighting_source) return;
    const auto& source = *m_prelighting_source;
    if (prelighting) mesh.Material()->customShader.shader = source.prelighting_shader;
    else if (!m_direct_puppet_source) mesh.Material()->customShader.shader = source.ordinary_shader;

    // The ordinary private dispatcher projects 0..target extent; the source draw separately
    // appends half the logical texture extent to I. Display-card dimensions neither size this
    // projection nor move the source mesh. Passthrough owns a different incoming I/camera state,
    // so keep its child camera intact.
    if (!m_owner.Passthrough()) {
        auto& camera = *scene.cameras.at(m_bridge_camera_name);
        if (prelighting) {
            const auto& target = scene.renderTargets.at(SourceTarget());
            camera.SetOrthographicViewRect(0.0, target.width, 0.0, target.height);
        } else {
            camera.SetOrthographicViewRect(-m_card_size[0] * 0.5, m_card_size[0] * 0.5,
                                           -m_card_size[1] * 0.5, m_card_size[1] * 0.5);
        }
    }
    LOG_INFO("SceneImagePrelightingSource: layer=%d name='%s' enabled=%s visible-effects=%s "
             "private=%s sprite=%s instanced=%s skinned=%s content=[%.3f %.3f] "
             "target='%s' camera='%s'",
             m_owner.Id(), m_owner.Name().c_str(), prelighting ? "true" : "false",
             VisibleCompositionStepCount() != 0 ? "true" : "false",
             ResolveFinalOutputCapability() != FinalOutputCapability::SceneAuthoredWriter
                 ? "true" : "false",
             source.sprite ? "true" : "false", source.instanced ? "true" : "false",
             mesh.Skinning().boneCount != 0 ? "true" : "false",
             SourceTextureContentSize()[0], SourceTextureContentSize()[1], SourceTarget().c_str(),
             m_bridge_camera_name.c_str());
}

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
            // Shape dispatch selects the physical last retained effect even when the effect
            // itself is hidden. Do not hide its private material nodes as a side effect of a
            // script-visible property write. The graph still gates every selected record on the
            // owner's draw admission.
            node.sceneNode->SetLocalVisible(
                m_visibility_policy == VisibilityPolicy::OwnerOnly || visible);
        }
    }
}

bool SceneImageEffectLayer::HasVisibleEffects() const {
    return std::any_of(m_effects.begin(), m_effects.end(), [](const auto& effect) {
        return effect->LocalVisible();
    });
}

bool SceneImageEffectLayer::UsesShapeDraw() const {
    return m_owner.Kind() == SceneObjectKind::Shape;
}

bool SceneImageEffectLayer::ShouldExecuteEffect(const SceneImageEffect& effect) const {
    // Retention and execution are different contracts. All effects remain addressable by
    // scripts and the first one supplies shape geometry; only the last retained shape effect
    // contributes ordered draw/copy/swap records, regardless of its local visibility value.
    return UsesShapeDraw() ? !m_effects.empty() && m_effects.back().get() == &effect
                           : effect.LocalVisible();
}

bool SceneImageEffectLayer::HasFinalComposite() const {
    return m_final_composite.draw.Mesh() != nullptr &&
        m_final_composite.draw.Mesh()->Material() != nullptr;
}

bool SceneImageEffectLayer::ShouldRunFinalComposite() const {
    if (!HasFinalComposite() || UsesDirectDraw()) return false;
    if (m_final_composite.publishes_private_output) return true;

    // A visible chain can consume every entry before the deferred segment, for example when its
    // last entry has compose=true. That empty segment draws nothing; it must not acquire a
    // neutral publisher merely because no authored scene writer was selected. Ordinary image/text
    // owners with no visible effects draw their own primitive directly. The traversal separately
    // skips a zero-step passthrough owner without children before emitting any of its phases; a
    // nonempty composition retains its child source.
    return !HasVisibleEffects();
}

void SceneImageEffectLayer::SetFinalCompositeSource(std::string source) {
    if (!HasFinalComposite()) return;

    auto* material = m_final_composite.draw.Mesh()->Material();
    if (material == nullptr) return;

    // The final composite is separate from every authored effect shader. It publishes a resolved
    // private output or the child source of a zero-step composition. An ordinary all-hidden
    // image/text owner uses its direct primitive instead, even while this resource stays resident.
    if (material->textures.empty()) material->textures.resize(1);
    material->textures[0] = std::move(source);
}

void SceneImageEffectLayer::SyncResolvedOutputMesh() {
    // Source/direct image draws share the retained card or imported mesh data. A size update
    // writes that retained resource, so also invalidate the active owner's GPU mesh handle.
    // Cold dynamic creation completes forward-reference dimensions before publishing its handle;
    // the retained effect meshes are already available at that point.
    if (auto* owner_node = m_owner.LayerNode(); owner_node != nullptr) {
        if (auto* owner_mesh = owner_node->Mesh(); owner_mesh != nullptr) owner_mesh->SetDirty();
    }
    for (auto& effect : m_effects) {
        for (auto& node : effect->nodes) {
            if (!node.mesh_follows_final_mesh) continue;
            // Resource-only graph refreshes retain the resolved materials. Update every layer-card
            // draw in the deferred segment, while private unit quads retain their own geometry.
            // ChangeMeshDataFrom shares CPU payload without marking GPU vertex buffers dirty.
            node.sceneNode->Mesh()->ChangeMeshDataFrom(*m_final_mesh);
            node.sceneNode->Mesh()->SetDirty();
        }
    }
    if (HasFinalComposite() && m_final_composite.draw.Mesh() != nullptr) {
        // Publication owns raster resources on the layer, without a transform or scene node.
        // Refresh this live mesh as well as the selected authored material so text re-layout
        // updates whichever phase currently publishes the layer.
        m_final_composite.draw.Mesh()->ChangeMeshDataFrom(*m_final_mesh);
        m_final_composite.draw.Mesh()->SetDirty();
    }
}

bool SceneImageEffectLayer::UsesOwnerTransform(const SceneNode* draw_node) const {
    for (const auto& effect : m_effects) {
        for (const auto& node : effect->nodes) {
            if (node.sceneNode.get() == draw_node) return node.uses_owner_transform;
        }
    }
    return false;
}

SceneImageEffectNode* SceneImageEffectLayer::ResolveEffectPingPongChain(
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam,
    std::string_view final_output,
    std::string_view& ppong_a,
    std::string_view& ppong_b,
    SceneImageEffect::FrameFboBindings& frame_bindings,
    bool admitted) {
    SceneImageEffectNode* fallback_last_output { nullptr };

    for (auto& eff : m_effects) {
        for (auto& node : eff->nodes) {
            node.uses_owner_transform = false;
            node.mesh_follows_final_mesh = false;
        }
        // A hidden effect contributes neither a destination step nor a new input texture. Retain
        // its authored materials for future visibility changes, but resolve the live chain from
        // only the visible sequence so its final geometry and matrix segment remain consistent.
        if (!eff->LocalVisible()) continue;
        // An empty visible effect contributes to the initial source-slot parity, but it never
        // binds a destination or completes an ordered record. Keep the current input pair so
        // later effects and private publication consume the image that was actually written.
        if (eff->nodes.empty() && eff->commands.empty()) continue;
        // Each compose marker advances the pair within this effect. Resolve material inputs and
        // commands at their authored positions so `previous` follows the last completed step;
        // resolving every binding at effect entry would make later steps resample the old input.
        auto [frame_it, inserted] = frame_bindings.try_emplace(eff, eff->CurrentFboBindings());
        auto fbo_bindings = frame_it->second;
        const auto resolve_commands_at = [&](i32 position) {
            for (auto& cmd : eff->commands) {
                if (cmd.afterpos != position) continue;
                eff->ResolveCommand(cmd, fbo_bindings,
                                    cmd.uses_final_destination ? final_output : ppong_b);
                // The marker is processed after the command, before the next entry at this same
                // boundary.
                if (cmd.advances_composition) std::swap(ppong_a, ppong_b);
            }
        };
        i32 position = 0;
        resolve_commands_at(position);

        for (auto it = eff->nodes.begin(); it != eff->nodes.end(); it++) {
            const auto authored_output = ResolveTemplateOrCurrent(it->authored_output, it->output);
            it->output = it->output_is_fbo ? fbo_bindings.at(it->authored_output)
                : ResolvePingPongOutputAlias(
                    authored_output, m_pingpong_a, m_pingpong_b, ppong_a, ppong_b);
            if (!it->output_is_fbo && IsCurrentEffectOutput(authored_output)) {
                fallback_last_output = &(*it);
            }

            assert(it->sceneNode->HasMaterial());

            auto& material = *(it->sceneNode->Mesh()->Material());
            it->blend_override.reset();
            it->depth_test_override.reset();
            it->depth_write_override.reset();
            // Owner preparation affects only the selected final material in the destination
            // segment, including an explicit FBO record there. Images inherit both source
            // selections; text replaces only testing. Resolve these values per invocation so
            // reflection/main draws and later text re-layout never mutate the retained material.
            if (it->uses_layer_space_effect_matrices && it->is_final_material) {
                if (const auto* text = m_owner.LayerNode()->Text(); text != nullptr) {
                    it->depth_test_override = text->object.depthtest == "enabled";
                } else if (auto* source = m_owner.LayerNode()->Mesh(); source != nullptr) {
                    it->depth_test_override = source->Material()->depthTest;
                    it->depth_write_override = source->Material()->depthWrite;
                }
            }
            it->destination_alpha_override = false;
            it->sceneNode->SetCamera(effect_cam.data());
            it->camera_override.clear();
            it->use_active_camera_for_parallax = false;
            it->clear_before_draw              = false;
            it->alpha_write_policy             = AlphaWritePolicy::Preserve;
            it->sceneNode->CopyTrans(default_node);
            it->sceneNode->Mesh()->ChangeMeshDataFrom(default_mesh);

            auto& texs = material.textures;
            if (!it->authored_textures.empty()) {
                texs = it->authored_textures;
            }
            for (auto& texture : texs) {
                texture = ResolvePingPongInputAlias(
                    texture, m_pingpong_a, m_pingpong_b, ppong_a, ppong_b);
            }
            for (const auto slot : it->fbo_texture_slots) {
                texs.at(slot) = fbo_bindings.at(it->authored_textures.at(slot));
            }
            if (std::getenv("WESCENE_TRACE_EFFECT_PHASES") != nullptr) {
                LOG_INFO("SceneEffectMatrixPhase: layer=%d node='%s' compose=%s "
                         "layer-space=%s input='%s' output='%s'",
                         it->sceneNode->ID(), it->sceneNode->Name().c_str(),
                         it->advances_composition ? "true" : "false",
                         it->uses_layer_space_effect_matrices ? "true" : "false",
                         texs.empty() ? "" : texs[0].c_str(), it->output.c_str());
            }
            if (it->advances_composition) std::swap(ppong_a, ppong_b);
            resolve_commands_at(++position);
        }

        // Retained but skipped passes must not advance the next invocation's inputs. Visibility
        // changes that admit such a pass rebuild the frame plan before it can execute.
        if (admitted) frame_it->second = std::move(fbo_bindings);
        std::swap(ppong_a, ppong_b);
    }

    return fallback_last_output;
}

void SceneImageEffectLayer::ResolveEffectMatrixPhases(bool keep_final_private) {
    std::size_t remaining_steps = VisibleCompositionStepCount();
    for (const auto& effect : m_effects) {
        if (!effect->LocalVisible()) {
            for (auto& node : effect->nodes) node.uses_layer_space_effect_matrices = false;
            for (auto& command : effect->commands) command.uses_final_destination = false;
            continue;
        }
        // Phase selection counts executed completions. An empty visible entry vector leaves
        // its initial counted step outstanding; consuming it here would move a later material
        // into the enclosing destination before that material's private work is complete.
        if (effect->nodes.empty() && effect->commands.empty()) continue;
        i32 position = 0;
        const auto advance_commands_at = [&](i32 boundary) {
            for (auto& command : effect->commands) {
                if (command.afterpos != boundary) continue;
                command.uses_final_destination = !keep_final_private && remaining_steps == 1;
                if (command.advances_composition) --remaining_steps;
            }
        };
        advance_commands_at(position);
        for (auto& node : effect->nodes) {
            // The final destination segment starts when only the ordinary final completion
            // remains. It can begin inside an effect and includes all subsequent FBO materials,
            // rather than being a property of just the final output node. Private publication
            // retains the scaled snapshots for every authored step.
            node.uses_layer_space_effect_matrices = !keep_final_private && remaining_steps == 1;
            if (node.advances_composition) --remaining_steps;
            advance_commands_at(++position);
        }
        if (effect->CompositionStepCount() != 0) --remaining_steps;
    }
}

bool SceneImageEffectLayer::UsesLayerSpaceEffectMatrices(const SceneNode* draw_node) const {
    for (const auto& effect : m_effects) {
        for (const auto& node : effect->nodes) {
            if (node.sceneNode.get() == draw_node) return node.uses_layer_space_effect_matrices;
        }
    }
    return false;
}

SceneImageEffectLayer::FinalOutputResolveDecision
SceneImageEffectLayer::ResolveFinalOutputDecision(
    FinalOutputCapability output_capability) {
    FinalOutputResolveDecision decision;

    const bool publish_private_final_composite =
        output_capability != FinalOutputCapability::SceneAuthoredWriter;
    decision.keep_authored_final_private = publish_private_final_composite;
    m_final_composite.publishes_private_output = publish_private_final_composite;

    return decision;
}

void SceneImageEffectLayer::ResolveFinalComposite(
    const SceneMesh& default_mesh,
    std::string_view effect_cam,
    std::string_view final_output,
    std::string_view final_composite_source) {
    if (!HasFinalComposite()) return;

    // The independent material publishes private layer output or a source-only sequence. An
    // ordinary visible authored segment selects its own material and leaves this phase dormant.
    SetFinalCompositeSource(std::string(final_composite_source));
    auto& mesh     = *m_final_composite.draw.Mesh();
    auto& material = *mesh.Material();
    if (m_fullscreen) {
        // A fullscreen publication is a 2x2 card in the effect camera. Its phase selects camera
        // space explicitly; there is no local transform to initialize or synchronize. Projecting
        // through the active layer camera would turn this card into a tiny world-space quad.
        material.blenmode = BlendMode::Normal;
        m_final_composite.draw.SetProjection(SceneDrawPhase::Space::Camera,
                                              std::string(effect_cam));
        mesh.ChangeMeshDataFrom(default_mesh);
        LOG_INFO("SceneEffectFinalCompositeResolve: layer=%d name='%s' fullscreen=true "
                 "camera='%.*s' output='%s' source='%s' blend=%d",
                 m_owner.Id(),
                 m_owner.Name().c_str(),
                 static_cast<int>(effect_cam.size()),
                 effect_cam.data(),
                 std::string(final_output).c_str(),
                 std::string(final_composite_source).c_str(),
                 static_cast<int>(material.blenmode));
        return;
    }

    material.blenmode = FinalBlend();
    m_final_composite.draw.SetProjection(SceneDrawPhase::Space::Layer, {});
    // The final mesh retains imported geometry and skinning attributes; it draws directly into
    // the enclosing destination with the utility material. Ordinary images and text use their own
    // retained destination card here.
    mesh.ChangeMeshDataFrom(*m_final_mesh);
    LOG_INFO("SceneEffectFinalCompositeResolve: layer=%d name='%s' fullscreen=false "
             "camera='' output='%s' source='%s' blend=%d "
             "publish-private=%s skinning-bones=%u",
             m_owner.Id(),
             m_owner.Name().c_str(),
             std::string(final_output).c_str(),
             std::string(final_composite_source).c_str(),
             static_cast<int>(material.blenmode),
             m_final_composite.publishes_private_output ? "true" : "false",
             mesh.Skinning().boneCount);
}

void SceneImageEffectLayer::ResolveVisibleFinalOutput(
    SceneImageEffectNode& final_output_node,
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam,
    std::string_view final_output) {
    // Every material drawing the current destination in this segment receives layer placement;
    // auxiliary FBO materials retain their private raster.
    final_output_node.uses_owner_transform = !m_fullscreen;
    final_output_node.mesh_follows_final_mesh = !m_fullscreen;
    final_output_node.output = std::string(final_output);
    final_output_node.destination_alpha_override = final_output_node.is_final_material;
    auto& mesh               = *(final_output_node.sceneNode->Mesh());
    auto& material           = *mesh.Material();
    if (m_fullscreen) {
        // Fullscreen postprocess final passes already draw the unit effect mesh and may still
        // multiply by g_ModelViewProjectionMatrix. Keep that final pass on the effect camera so
        // the 2x2 utility quad remains a full-frame clip-space composite instead of becoming a
        // tiny active-camera world quad that leaves the effect effectively invisible.
        final_output_node.blend_override = final_output_node.is_final_material
            ? std::optional { BlendMode::Normal } : std::nullopt;
        final_output_node.sceneNode->SetCamera(effect_cam.data());
        final_output_node.sceneNode->CopyTrans(default_node);
        mesh.ChangeMeshDataFrom(default_mesh);
        LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=true "
                 "camera='%.*s' output='%s' material='%s' blend=%d",
                 m_owner.Id(),
                 m_owner.Name().c_str(),
                 static_cast<int>(effect_cam.size()),
                 effect_cam.data(),
                 std::string(final_output).c_str(),
                 material.name.c_str(),
                 static_cast<int>(final_output_node.blend_override.value_or(material.blenmode)));
        return;
    }

    // The owner's final-material preparation is scoped to the selected material. Other
    // destination records keep their own blend, and private writers never acquire this override.
    final_output_node.blend_override = final_output_node.is_final_material
        ? std::optional { FinalBlend() } : std::nullopt;
    final_output_node.sceneNode->SetCamera(std::string());
    mesh.ChangeMeshDataFrom(*m_final_mesh);
    LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=false "
             "camera='' output='%s' material='%s' blend=%d private=false",
             m_owner.Id(),
             m_owner.Name().c_str(),
             std::string(final_output).c_str(),
             material.name.c_str(),
             static_cast<int>(final_output_node.blend_override.value_or(material.blenmode)));
}

void SceneImageEffectLayer::ResolvePrivateFinalOutput(
    SceneImageEffectNode& final_output_node,
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam) {
    // The owner's private-output contract keeps its authored final segment in the private
    // destination. Publication then draws that result with the owner's placement. An empty source
    // or a visibility binding alone does not select this extra publication phase.
    final_output_node.uses_owner_transform = false;
    final_output_node.mesh_follows_final_mesh = false;
    auto& mesh     = *(final_output_node.sceneNode->Mesh());
    auto& material = *mesh.Material();
    final_output_node.blend_override.reset();
    final_output_node.destination_alpha_override = false;
    final_output_node.sceneNode->SetCamera(effect_cam.data());
    final_output_node.sceneNode->CopyTrans(default_node);
    mesh.ChangeMeshDataFrom(default_mesh);
    LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=false "
             "camera='%.*s' output='%s' material='%s' blend=%d private=true "
             "publish-private=%s",
             m_owner.Id(),
             m_owner.Name().c_str(),
             static_cast<int>(effect_cam.size()),
             effect_cam.data(),
             final_output_node.output.c_str(),
             material.name.c_str(),
             static_cast<int>(material.blenmode),
             m_final_composite.publishes_private_output ? "true" : "false");
}

void SceneImageEffectLayer::ResolveShapeEffect(const SceneMesh& default_mesh,
                                               std::string_view final_output,
                                               SceneImageEffect::FrameFboBindings& frame_bindings,
                                               bool admitted) {
    // Shape effect dispatch has no source draw or image composition loop. Its physical last
    // retained effect receives source slot -1 and final-draw=true for every ordered record,
    // including commands and explicit FBO draws. Earlier effects still own their materials,
    // script bindings and first-effect geometry.
    for (const auto& effect : m_effects) {
        for (auto& node : effect->nodes) {
            node.uses_owner_transform = false;
            node.mesh_follows_final_mesh = false;
            node.uses_layer_space_effect_matrices = false;
        }
    }
    if (m_effects.empty()) return;

    auto& effect = *m_effects.back();
    auto [frame_it, inserted] = frame_bindings.try_emplace(
        m_effects.back(), effect.CurrentFboBindings());
    auto bindings = frame_it->second;
    LOG_INFO("SceneShapeEffectResolve: layer=%d name='%s' retained-effects=%zu "
             "selected-effect=%d selected-index=%u instance-visible=%s materials=%zu "
             "commands=%zu source-slot=-1 final-draw=true output='%.*s' publication=false",
             m_owner.Id(), m_owner.Name().c_str(), m_effects.size(), effect.EffectId(),
             effect.EffectIndex(), effect.LocalVisible() ? "true" : "false",
             effect.nodes.size(), effect.commands.size(), static_cast<int>(final_output.size()),
             final_output.data());
    const auto resolve_commands_at = [&](i32 position) {
        for (auto& command : effect.commands) {
            if (command.afterpos != position) continue;
            command.uses_final_destination = true;
            effect.ResolveCommand(command, bindings, final_output);
            // The shape callback does not inspect compose markers or advance a destination
            // pair. Only explicit copy/swap records change the shared command/FBO state.
        }
    };
    i32 position = 0;
    resolve_commands_at(position);
    for (auto& node : effect.nodes) {
        node.output = node.output_is_fbo ? bindings.at(node.authored_output)
                                        : std::string(final_output);
        node.uses_owner_transform = true;
        node.mesh_follows_final_mesh = node.is_final_material;
        node.uses_layer_space_effect_matrices = true;
        node.sceneNode->SetCamera(std::string());
        node.camera_override.clear();
        node.use_active_camera_for_parallax = false;
        node.clear_before_draw = false;
        node.alpha_write_policy = AlphaWritePolicy::Preserve;
        // Shapes select only the final blend; neither depth property is replaced by the owner.
        node.depth_test_override.reset();
        node.depth_write_override.reset();

        // The shape callback establishes I once for the whole record loop. Binding an FBO
        // changes its target/viewport, not that owner placement or incoming camera. The common
        // dispatcher selects the retained card and additive state only at its final-material
        // index; other records keep the unit mesh and authored material blend.
        auto& mesh = *node.sceneNode->Mesh();
        auto& material = *mesh.Material();
        mesh.ChangeMeshDataFrom(node.is_final_material ? *m_final_mesh : default_mesh);
        node.blend_override = node.is_final_material
            ? std::optional { FinalBlend() } : std::nullopt;
        node.destination_alpha_override = node.is_final_material;
        material.textures = node.authored_textures;
        for (const auto slot : node.fbo_texture_slots) {
            material.textures.at(slot) = bindings.at(node.authored_textures.at(slot));
        }
        if (std::getenv("WESCENE_TRACE_EFFECT_PHASES") != nullptr) {
            LOG_INFO("SceneShapeMaterialResolve: layer=%d effect=%d material=%d "
                     "final-material=%s compose=%s explicit-fbo=%s input='%s' output='%s' "
                     "owner-transform=true mesh=%s blend=%d",
                     m_owner.Id(), effect.EffectId(), position,
                     node.is_final_material ? "true" : "false",
                     node.advances_composition ? "true" : "false",
                     node.output_is_fbo ? "true" : "false",
                     material.textures.empty() ? "" : material.textures[0].c_str(),
                     node.output.c_str(), node.is_final_material ? "shape-card" : "unit",
                     static_cast<int>(node.blend_override.value_or(material.blenmode)));
        }
        resolve_commands_at(++position);
    }
    if (admitted) frame_it->second = std::move(bindings);
}

void SceneImageEffectLayer::ResolveEffect(const SceneMesh& default_mesh,
                                          std::string_view effect_cam,
                                          std::string_view final_output,
                                          SceneImageEffect::FrameFboBindings& frame_bindings,
                                          bool admitted) {
    if (UsesShapeDraw()) {
        m_final_composite.ResetForResolve();
        ResolveShapeEffect(default_mesh, final_output, frame_bindings, admitted);
        return;
    }
    const auto output_capability = ResolveFinalOutputCapability();
    std::string_view ppong_a = m_pingpong_a, ppong_b = m_pingpong_b;
    if (SourceSlot() != 0) std::swap(ppong_a, ppong_b);
    auto             default_node = SceneNode();

    m_final_composite.ResetForResolve();

    // The source pass, command bindings and material uniforms share the owner's publication
    // contract. Resolve phases before bindings, using the same source slot as graph emission and
    // ancestor collision lookup; the enclosing output cannot change this object's step count.
    ResolveEffectMatrixPhases(output_capability != FinalOutputCapability::SceneAuthoredWriter);
    auto* fallback_last_output =
        ResolveEffectPingPongChain(default_mesh, default_node, effect_cam, final_output,
                                  ppong_a, ppong_b, frame_bindings, admitted);
    const auto final_decision = ResolveFinalOutputDecision(output_capability);

    std::string_view final_composite_source = ppong_a;
    const auto diagnostic_roles = ResolveEffectOutputDiagnosticRoles(
        output_capability,
        final_decision.keep_authored_final_private,
        fallback_last_output != nullptr);
    const std::string_view authored_output_target =
        fallback_last_output != nullptr ? std::string_view(fallback_last_output->output)
                                        : std::string_view {};
    LOG_INFO("SceneEffectOutputContract: layer=%d name='%s' declared-capability=%.*s "
             "resolved-capability=%.*s authored-writer-role='%.*s' "
             "publication-writer-role='%.*s' effect-camera='%.*s' "
             "authored-output-target='%.*s' "
             "publication-source='%.*s' final-output-target='%.*s' "
             "private-parallax-owner='%.*s' publication-parallax-owner='%.*s' "
             "parallax-application-count=%u keep-authored-final-private=%s "
             "copybackground=%s",
             m_owner.Id(),
             m_owner.Name().c_str(),
             static_cast<int>(FinalOutputCapabilityName(m_final_output_capability).size()),
             FinalOutputCapabilityName(m_final_output_capability).data(),
             static_cast<int>(FinalOutputCapabilityName(output_capability).size()),
             FinalOutputCapabilityName(output_capability).data(),
             static_cast<int>(diagnostic_roles.authored_writer.size()),
             diagnostic_roles.authored_writer.data(),
             static_cast<int>(diagnostic_roles.publication_writer.size()),
             diagnostic_roles.publication_writer.data(),
             static_cast<int>(effect_cam.size()),
             effect_cam.data(),
             static_cast<int>(authored_output_target.size()),
             authored_output_target.data(),
             static_cast<int>(final_composite_source.size()),
             final_composite_source.data(),
             static_cast<int>(final_output.size()),
             final_output.data(),
             static_cast<int>(diagnostic_roles.private_parallax_owner.size()),
             diagnostic_roles.private_parallax_owner.data(),
             static_cast<int>(diagnostic_roles.publication_parallax_owner.size()),
             diagnostic_roles.publication_parallax_owner.data(),
             diagnostic_roles.parallax_application_count,
             final_decision.keep_authored_final_private ? "true" : "false",
             CopyBackground() ? "true" : "false");

    ResolveFinalComposite(default_mesh,
                              effect_cam,
                              final_output,
                              final_composite_source);

    std::size_t visible_writer_count = 0;
    if (!final_decision.keep_authored_final_private) {
        for (auto& effect : m_effects) {
            if (!effect->LocalVisible()) continue;
            for (auto& node : effect->nodes) {
                const auto authored_output =
                    ResolveTemplateOrCurrent(node.authored_output, node.output);
                if (!node.uses_layer_space_effect_matrices || node.output_is_fbo ||
                    !IsCurrentEffectOutput(authored_output)) continue;
                ResolveVisibleFinalOutput(node, default_mesh, default_node,
                                          effect_cam, final_output);
                ++visible_writer_count;
            }
        }
    } else if (fallback_last_output != nullptr) {
        ResolvePrivateFinalOutput(*fallback_last_output,
                                  default_mesh,
                                  default_node,
                                  effect_cam);
    }
    if (std::getenv("WESCENE_TRACE_EFFECT_PHASES") != nullptr) {
        LOG_INFO("SceneEffectFinalSegment: layer=%d private=%s visible-writers=%zu "
                 "source-publication=%s composition-steps=%zu source-slot=%zu source='%s'",
                 m_owner.Id(), final_decision.keep_authored_final_private ? "true" : "false",
                 visible_writer_count, ShouldRunFinalComposite() ? "true" : "false",
                 VisibleCompositionStepCount(), SourceSlot(), SourceTarget().c_str());
    }
}

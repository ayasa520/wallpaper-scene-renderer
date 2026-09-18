#include "Scene/SceneDestinationTarget.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "Scene/SceneImageEffectLayer.h"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"

namespace wallpaper
{

int32_t ClampDestinationRenderTargetExtent(int32_t extent) {
    return std::max(kMinDestinationRenderTargetExtent, extent);
}

std::array<int32_t, 2> ResolveCardDestinationExtent(std::array<float, 2> card_size) {
    // Resource setup reads the current card dimensions, ceils each axis, then applies the
    // destination minimum. This calculation is only selected by the source policy established by
    // the parser; changing a displayed card must not change a source-texture or shape pixel
    // extent.
    return { ClampDestinationRenderTargetExtent(static_cast<int32_t>(std::ceil(card_size[0]))),
             ClampDestinationRenderTargetExtent(static_cast<int32_t>(std::ceil(card_size[1]))) };
}

std::array<int32_t, 2> ResolveTextDestinationExtent(std::array<float, 2> layout_size) {
    // Text destination sizing truncates the shaped float bounds (including conditional padding)
    // before the shared allocator applies minimum four. Image cards use ceil instead; glyph
    // raster density does not redefine this pixel contract.
    return { ClampDestinationRenderTargetExtent(static_cast<int32_t>(layout_size[0])),
             ClampDestinationRenderTargetExtent(static_cast<int32_t>(layout_size[1])) };
}

std::array<int32_t, 2> ResolveShapeDestinationExtent(std::array<int32_t, 2> canvas_size) {
    // Shape destination setup obtains integer canvas halves independently of its current card
    // geometry and material control-point positions.
    return { ClampDestinationRenderTargetExtent(canvas_size[0] / 2),
             ClampDestinationRenderTargetExtent(canvas_size[1] / 2) };
}

std::array<int32_t, 2> ResolveEffectRenderTargetReferenceExtent(
    std::array<float, 2> source_extent, std::array<uint16_t, 2> authored_extent, uint16_t fit) {
    constexpr uint16_t max_authored_extent = 4096;
    int32_t width = authored_extent[0] <= max_authored_extent
        ? authored_extent[0] : static_cast<int32_t>(source_extent[0]);
    int32_t height = authored_extent[1] <= max_authored_extent
        ? authored_extent[1] : static_cast<int32_t>(source_extent[1]);
    if (fit <= max_authored_extent) {
        // Fit limits the longer selected axis; it never enlarges a smaller source. Preserve the
        // float ratio followed by multiplication and truncation, before the target's divisor is
        // applied. Keeping reference selection separate also lets a shared target retain its
        // first creator's divisor when a different owner reruns setup.
        if (width >= height) {
            const auto longer = std::min<int32_t>(width, fit);
            const float ratio = static_cast<float>(height) / static_cast<float>(width);
            height = static_cast<int32_t>(ratio * static_cast<float>(longer));
            width = longer;
        } else {
            const auto longer = std::min<int32_t>(height, fit);
            const float ratio = static_cast<float>(width) / static_cast<float>(height);
            width = static_cast<int32_t>(ratio * static_cast<float>(longer));
            height = longer;
        }
    }
    return { width, height };
}

static std::array<int32_t, 2> ResolveScaledRenderTargetExtent(
    std::array<int32_t, 2> reference_extent, uint32_t divisor) {
    return { std::max(2, reference_extent[0] / static_cast<int32_t>(divisor)),
             std::max(2, reference_extent[1] / static_cast<int32_t>(divisor)) };
}

SceneRenderTarget SceneRenderTarget::FromReferenceExtent(
    std::array<i32, 2> extent, uint32_t divisor) {
    const auto physical = ResolveScaledRenderTargetExtent(extent, divisor);
    return SceneRenderTarget {
        .width = physical[0],
        .height = physical[1],
        .mapWidth = physical[0],
        .mapHeight = physical[1],
        .reference_extent = extent,
        .resolution_divisor = divisor,
    };
}

bool SceneRenderTarget::ResizeReferenceExtent(std::array<i32, 2> extent) {
    if (reference_extent == extent) return false;
    const auto physical = ResolveScaledRenderTargetExtent(extent, resolution_divisor);
    reference_extent = extent;
    width = mapWidth = physical[0];
    height = mapHeight = physical[1];
    ++allocation_revision;
    return true;
}

TextureSample DestinationRenderTargetSampler(bool point_sampled, bool clamp_uvs) {
    const auto wrap = clamp_uvs ? TextureWrap::CLAMP_TO_EDGE : TextureWrap::REPEAT;
    const auto filter = point_sampled ? TextureFilter::NEAREST : TextureFilter::LINEAR;
    return TextureSample { wrap, wrap, filter, filter };
}

std::string SceneDestinationRenderTargetBaseName(int32_t width, int32_t height,
                                                 const TextureSample& sampler, size_t slot) {
    // Shared destination targets are fixed-size images: a layer whose source is the output
    // framebuffer takes that framebuffer's pixel size at load, so two layers that resolve to the
    // same pixel size legitimately share one backing image. There is no separate screen-following
    // identity.
    const char filter = sampler.magFilter == TextureFilter::NEAREST ? 'n' : 'b';
    const char wrap = sampler.wrapS == TextureWrap::CLAMP_TO_EDGE ? 'c' : 'r';
    return "sc." + std::to_string(ClampDestinationRenderTargetExtent(width)) + "." +
           std::to_string(ClampDestinationRenderTargetExtent(height)) + "." + filter + wrap +
           std::to_string(slot);
}

static std::array<std::string, 2> SceneDestinationRenderTargetNames(
    const Scene& scene, int32_t layer_id, int32_t parent_id, bool private_output,
    int32_t width, int32_t height, const TextureSample& sampler) {
    std::array<std::string, 2> names {
        private_output ? GenImageLayerCompositeTex(layer_id)
                       : SceneDestinationRenderTargetBaseName(width, height, sampler, 0),
        SceneDestinationRenderTargetBaseName(width, height, sampler, 1),
    };
    std::array<uint32_t, 2> ancestor_collisions {};
    std::unordered_set<int32_t> visited;

    // A passthrough parent's source target remains live while its children draw. Compare the
    // complete candidate prefix, including sampler and slot, against that active destination.
    // An ancestor may itself already have a collision suffix: exact equality would miss it and
    // make the third nesting level overwrite its parent. Count every matching ancestor before
    // appending the depth, independently for each slot. Siblings can still share scratch images.
    while (parent_id != 0 && visited.insert(parent_id).second) {
        const auto* parent = scene.FindSceneObject(parent_id);
        if (parent != nullptr && parent->Passthrough()) {
            if (const auto* effect_layer = scene.FindImageEffectLayer(parent_id)) {
                for (size_t slot = private_output ? 1 : 0; slot < names.size(); ++slot) {
                    if (effect_layer->SourceTarget().starts_with(names[slot])) {
                        ancestor_collisions[slot]++;
                    }
                }
            }
        }

        parent_id = parent != nullptr ? parent->ParentId() : 0;
    }

    for (size_t index = 0; index < names.size(); index++) {
        if (ancestor_collisions[index] != 0) {
            names[index] += std::to_string(ancestor_collisions[index]);
        }
    }
    return names;
}

std::array<std::string, 2> ResolveSceneDestinationRenderTargets(
    Scene& scene, int32_t layer_id, int32_t parent_id, bool private_output,
    const SceneRenderTarget& target) {
    const auto names = SceneDestinationRenderTargetNames(
        scene, layer_id, parent_id, private_output, target.width, target.height, target.sample);
    for (size_t slot = 0; slot < names.size(); ++slot) {
        if (private_output && slot == 0) {
            // Private output allocates an independent render target in owner slot zero. Its
            // identity must survive unrelated equal-sized layers and graph lifetime reuse. A
            // re-layout creates a replacement at the new extent under the same owner name;
            // invalidate that one GPU resource instead of treating the name as an intern hit.
            auto private_target = target;
            private_target.allowReuse = false;
            const auto [_, inserted] = scene.renderTargets.insert_or_assign(
                names[slot], private_target);
            if (!inserted) scene.MarkRenderTargetResourcesDirty(names[slot]);
            LOG_INFO("ScenePrivateDestinationTarget: layer=%d target='%s' extent=%dx%d "
                     "recreated=%s",
                     layer_id, names[slot].c_str(), target.width, target.height,
                     inserted ? "false" : "true");
        } else {
            InternNamedRenderTarget(scene, names[slot], target);
        }
    }
    return names;
}

const SceneRenderTarget& InternNamedRenderTarget(Scene& scene, const std::string& name,
                                                 SceneRenderTarget target) {
    // `try_emplace` makes the first-registration rule explicit and prevents a later hidden
    // language branch from silently replacing the descriptor shared by an earlier branch.
    const auto [it, inserted] = scene.renderTargets.try_emplace(name, target);
    // Storage retention is a requirement of every user of the shared name, independent
    // of which declaration supplied its dimensions. A seed-only effect can register a
    // transient target before a feedback effect requests the same image. Once any caller
    // needs retained contents, neither final-reader release nor a later transient
    // declaration may return that image to the reusable pool between submissions.
    it->second.allowReuse = it->second.allowReuse && target.allowReuse;
    if (! inserted &&
        (it->second.width != target.width || it->second.height != target.height ||
         it->second.ContentWidth() != target.ContentWidth() ||
         it->second.ContentHeight() != target.ContentHeight())) {
        LOG_INFO("SceneNamedRenderTargetIntern: name='%s' first-size=%dx%d first-map=%dx%d "
                 "ignored-size=%dx%d ignored-map=%dx%d",
                 name.c_str(),
                 it->second.width,
                 it->second.height,
                 it->second.ContentWidth(),
                 it->second.ContentHeight(),
                 target.width,
                 target.height,
                 target.ContentWidth(),
                 target.ContentHeight());
    }
    return it->second;
}

} // namespace wallpaper

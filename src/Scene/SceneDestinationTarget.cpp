#include "Scene/SceneDestinationTarget.h"

#include <algorithm>
#include <unordered_set>

#include "Scene/SceneImageEffectLayer.h"
#include "Utils/Logging.h"

namespace wallpaper
{

int32_t ClampDestinationRenderTargetExtent(int32_t extent) {
    return std::max(kMinDestinationRenderTargetExtent, extent);
}

std::string SceneDestinationRenderTargetBaseName(int32_t width, int32_t height, char suffix) {
    // Every destination target is a fixed-size image: a layer whose source is the output
    // framebuffer takes that framebuffer's pixel size at load, so two layers that resolve to the
    // same pixel size legitimately share one backing image. There is no separate screen-following
    // identity.
    return "sc." + std::to_string(ClampDestinationRenderTargetExtent(width)) + "." +
           std::to_string(ClampDestinationRenderTargetExtent(height)) + "." + suffix;
}

std::array<std::string, 2> SceneDestinationRenderTargetNames(
    const Scene& scene, int32_t parent_id, int32_t width, int32_t height, char suffix) {
    const char partner_suffix = suffix == 'n' ? 'b' : 'n';
    std::array<std::string, 2> names {
        SceneDestinationRenderTargetBaseName(width, height, suffix),
        SceneDestinationRenderTargetBaseName(width, height, partner_suffix),
    };
    std::array<uint32_t, 2> ancestor_collisions {};
    std::unordered_set<int32_t> visited;

    // Destination names start as `sc.W.H.<suffix>`. Walk only the current object's parent chain
    // and append the number of ancestors whose destination-slot name exactly equals that base.
    // Only ancestors flagged passthrough take part in the comparison; every other ancestor is
    // walked through without contributing. Siblings therefore intern the same named RT (the
    // multilingual case), while a nested same-sized effect under a passthrough parent does not
    // alias the target that parent still uses. Compare exact names rather than inventing a layer
    // id or a global occurrence counter.
    while (parent_id != 0 && visited.insert(parent_id).second) {
        const auto* parent = scene.FindSceneObject(parent_id);
        if (parent != nullptr && parent->Passthrough()) {
            if (const auto* effect_layer = scene.FindImageEffectLayer(parent_id)) {
                if (effect_layer->FirstTarget() == names[0]) {
                    ancestor_collisions[0]++;
                    ancestor_collisions[1]++;
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

const SceneRenderTarget& InternNamedRenderTarget(Scene& scene, const std::string& name,
                                                 SceneRenderTarget target) {
    // `try_emplace` makes the first-registration rule explicit and prevents a later hidden
    // language branch from silently replacing the descriptor shared by an earlier branch.
    const auto [it, inserted] = scene.renderTargets.try_emplace(name, target);
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

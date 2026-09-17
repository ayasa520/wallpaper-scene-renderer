#include "WPSceneScriptHostShared.hpp"

// Layer destruction nominates resources after the whole batch's surviving owners are known.
// Imported candidates receive a final reference check at the renderer's release boundary.

#include "Scene/Scene.h"
#include "Utils/Logging.h"

namespace wallpaper
{

void QueueLayerResourceRelease(Scene& scene, int32_t layer_id,
                               const LayerResidencyResources& resources,
                               const LayerResidencyResources& retained_resources,
                               const char* reason) {
    // The removed owner's descriptors were captured while it was alive. Retention is decided
    // only after the whole deletion batch, including surviving parents' destination reselection;
    // consulting the scene again for this owner would now find no identity or draw resources.
    std::size_t queued_static = 0;
    std::size_t queued_video = 0;
    std::size_t queued_render_targets = 0;

    for (const auto& key : resources.static_textures) {
        if (retained_resources.static_textures.count(key) != 0) continue;
        queued_static += scene.pendingStaticTextureReleaseKeys.insert(key).second ? 1 : 0;
    }
    for (const auto& key : resources.video_textures) {
        if (retained_resources.video_textures.count(key) != 0) continue;
        queued_video += scene.pendingVideoTextureReleaseKeys.insert(key).second ? 1 : 0;
    }
    for (const auto& key : resources.render_targets) {
        if (retained_resources.render_targets.count(key) != 0) continue;
        queued_render_targets += scene.pendingRenderTargetReleaseKeys.insert(key).second ? 1 : 0;
    }

    if (queued_static != 0 || queued_video != 0 || queued_render_targets != 0) {
        LOG_INFO("SceneResidencyQueueRelease: reason=%s layer=%d static=%zu video=%zu "
                 "render-target=%zu",
                 reason != nullptr ? reason : "unknown",
                 layer_id,
                 queued_static,
                 queued_video,
                 queued_render_targets);
    }
}

} // namespace wallpaper

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>

namespace wallpaper
{
class Scene;

// Retained scene references are independent of the currently admitted render commands. Hidden
// owners and inactive source variants still own their material inputs; GPU retirement must use
// this ownership view after all scene mutations at the resource boundary have completed.
struct LayerResidencyResources {
    std::unordered_set<std::string> static_textures;
    std::unordered_set<std::string> video_textures;
    std::unordered_set<std::string> render_targets;
};

LayerResidencyResources CollectLayerResidencyResources(const Scene& scene, int32_t layer_id);
LayerResidencyResources CollectRetainedResidencyResources(
    const Scene& scene, const std::unordered_set<int32_t>& excluded_layers = {});
void QueueReplacedImportedTextures(Scene& scene, const LayerResidencyResources& previous);
} // namespace wallpaper

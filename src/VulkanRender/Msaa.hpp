#pragma once

#include "Scene/Scene.h"
#include "SpecTexs.hpp"
#include "Vulkan/Device.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace wallpaper
{

inline void ConfigureSceneMsaa(Scene& scene) {
    const int         samples = scene.msaa.SampleCount();
    const std::string ms_name(SpecTex_DefaultMS);
    scene.msaa.built_quality = scene.msaa.quality;
    if (samples <= 1) {
        if (scene.renderTargets.erase(ms_name) > 0) {
            scene.pendingRenderTargetReleaseKeys.insert(ms_name);
        }
        return;
    }

    const auto def_it = scene.renderTargets.find(std::string(SpecTex_Default));
    if (def_it == scene.renderTargets.end()) return;
    const auto& def = def_it->second;
    scene.renderTargets[ms_name] = SceneRenderTarget {
        .width        = def.width,
        .height       = def.height,
        .mapWidth     = def.mapWidth,
        .mapHeight    = def.mapHeight,
        .sample_count = samples,
        .bind         = { .enable = true, .screen = true },
    };
}

namespace vulkan
{

inline VkSampleCountFlagBits MsaaSupportedSampleCount(const Device& device, int requested) {
    if (requested <= 1) return VK_SAMPLE_COUNT_1_BIT;
    const VkSampleCountFlags bits = device.limits().framebufferColorSampleCounts &
                                    device.limits().framebufferDepthSampleCounts;
    for (int n = requested; n > 1; n >>= 1) {
        const auto flag = static_cast<VkSampleCountFlagBits>(n);
        if ((bits & static_cast<VkSampleCountFlags>(flag)) ==
            static_cast<VkSampleCountFlags>(flag)) {
            return flag;
        }
    }
    return VK_SAMPLE_COUNT_1_BIT;
}

inline bool ComposeOutputUsesMsaa(const Scene& scene, std::string_view output) {
    return scene.msaa.SampleCount() > 1 && output == SpecTex_Default;
}

// Bloom/HDR combine sample `_rt_default` and replace it. Those posts stay 1x on
// the resolved color. Layer draws that sample the framebuffer still write the
// MS compose target; they read the resolved image, which is a different RT.
inline bool ShaderDrawSamplesResolvedDefault(const std::vector<std::string>& textures) {
    for (const auto& texture : textures) {
        if (texture == SpecTex_Default) return true;
    }
    return false;
}

inline bool ShaderDrawWritesResolvedDefault(const Scene& scene, std::string_view output,
                                            const SceneNode* node) {
    (void)scene;
    if (output != SpecTex_Default || node == nullptr) return false;
    const auto& name = node->Name();
    if (name.rfind("__hanabi_scene_bloom", 0) == 0) return true;
    if (name.rfind("__hanabi_scene_hdr_", 0) == 0) return true;
    return false;
}

inline bool ShaderDrawCanUseMsaa(const Scene& scene, std::string_view output,
                                 const SceneNode* node) {
    return ComposeOutputUsesMsaa(scene, output) &&
           ! ShaderDrawWritesResolvedDefault(scene, output, node);
}

struct RenderingResources;

void NoteComposeMsaaDraw(RenderingResources& rr, VkSampleCountFlagBits samples);
bool ResolveComposeMsaaIfNeeded(Scene& scene, const Device& device, RenderingResources& rr);

inline void SyncSceneMsaa(Scene& scene, const Device& device) {
    const int requested = Scene::MsaaSettings::RequestedSampleCount(scene.msaa.quality);
    scene.msaa.device_samples = static_cast<int>(MsaaSupportedSampleCount(device, requested));
    ConfigureSceneMsaa(scene);
}

} // namespace vulkan
} // namespace wallpaper

#pragma once

#include "Vulkan/TextureCache.hpp"

#include <string_view>

namespace wallpaper::vulkan
{

struct RenderingResources;

void ClearRenderTargetColor(vvk::CommandBuffer&, const ImageParameters&,
                            const VkClearColorValue&, VkImageLayout final_layout);

enum class SceneDepthAction { Unused, Preserved, LayoutInitialized, Cleared };

SceneDepthAction PrepareSceneModelDepth(RenderingResources&, std::string_view target, bool clear);
std::string_view SceneDepthActionName(SceneDepthAction);

} // namespace wallpaper::vulkan

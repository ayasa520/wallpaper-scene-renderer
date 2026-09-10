#pragma once

#include "Vulkan/TextureCache.hpp"

#include <string_view>

namespace wallpaper::vulkan
{

struct RenderingResources;
class Device;

void ClearRenderTargetColor(vvk::CommandBuffer&, const ImageParameters&,
                            const VkClearColorValue&, VkImageLayout final_layout);

enum class SceneDepthAction { Unused, Preserved, LayoutInitialized, Cleared };

// The destination owns depth independently of its draw kind. Callers supply the actual bound
// color extent and sample count, so model, material and glyph passes acquire the same image.
VmaImageParameters* AcquireSceneDepthImage(const Device&, RenderingResources&,
                                          std::string_view target, VkExtent3D,
                                          VkSampleCountFlagBits);
SceneDepthAction PrepareSceneModelDepth(RenderingResources&, std::string_view target, bool clear);
std::string_view SceneDepthActionName(SceneDepthAction);

} // namespace wallpaper::vulkan

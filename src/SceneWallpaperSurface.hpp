#pragma once
#include "SceneWallpaper.hpp"

#include <functional>
#include <memory>
#include <string_view>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <span>
#include <vector>

namespace wallpaper
{
using ReDrawCB = std::function<void()>;

namespace vulkan
{
class Device;
class VulkanExSwapchain;
}

enum class VulkanDevicePreference {
    Default,
    PreferIntegrated,
    PreferDiscrete,
};

enum class GpuPipelinePreference {
    Nvidia,
    NvidiaStateless,
    Va,
};

struct VulkanSurfaceInfo {
    std::function<VkResult(VkInstance, VkSurfaceKHR*)> createSurfaceOp;
    std::vector<std::string>                           instanceExts;
};

struct RenderInitInfo {
    bool enable_valid_layer { false };
    bool offscreen { false };
    ExternalFrameExportMode export_mode { ExternalFrameExportMode::OPAQUE_FD };

    std::span<const std::uint8_t> uuid;
    VulkanDevicePreference        device_preference { VulkanDevicePreference::Default };
    GpuPipelinePreference         gpu_pipeline_preference { GpuPipelinePreference::Nvidia };
    TexTiling                     offscreen_tiling { TexTiling::OPTIMAL };
    uint32_t                      export_drm_fourcc { 0 };
    std::vector<uint64_t>         export_drm_modifiers;
    VulkanSurfaceInfo             surface_info;

    uint16_t width { 1920 };
    uint16_t height { 1080 };
    double   render_scale { 1.0 };
    ReDrawCB redraw_callback;

    /*
     * Offscreen producers can inject the final export route after the renderer
     * has picked its physical device and created VkDevice. This mirrors
     * waywallen's ex_swapchain_factory: scene rendering and video decoding are
     * peers that publish into a route-owned swapchain, instead of either
     * backend being hard-coded as the owner of the whole producer output path.
     */
    struct ExSwapchainHandles {
        VkInstance       instance { VK_NULL_HANDLE };
        VkPhysicalDevice physical_device { VK_NULL_HANDLE };
        VkDevice         device { VK_NULL_HANDLE };
        VkQueue          graphics_queue { VK_NULL_HANDLE };
        uint32_t         graphics_queue_family { 0 };
        const vulkan::Device* renderer_device { nullptr };
    };
    std::function<std::unique_ptr<vulkan::VulkanExSwapchain>(const ExSwapchainHandles&)>
        ex_swapchain_factory;
};

} // namespace wallpaper

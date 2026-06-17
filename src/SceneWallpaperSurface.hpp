#pragma once
#include "SceneWallpaper.hpp"

#include <functional>
#include <memory>
#include <string>
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

enum class VideoTextureDecoderRoute {
    Nvidia,
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
    VulkanDevicePreference   device_preference { VulkanDevicePreference::Default };
    VideoTextureDecoderRoute video_texture_decoder_route { VideoTextureDecoderRoute::Nvidia };
    /*
     * DRM render node of the device selected via `uuid` (e.g.
     * "/dev/dri/renderD129"). Scene video textures use this as a runtime proof:
     * a VA decoder/postproc factory is accepted only when its readable
     * "device-path" property exactly matches this node. Empty means the caller
     * did not prove a renderer node, so the GPU-only video texture path must
     * fail instead of guessing a default GPU.
     */
    std::string                   render_node;
    TexTiling                     offscreen_tiling { TexTiling::OPTIMAL };
    uint32_t                      export_drm_fourcc { 0 };
    std::vector<uint64_t>         export_drm_modifiers;
    ExternalFrameMemoryPreference export_memory_preference {
        ExternalFrameMemoryPreference::Default
    };
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

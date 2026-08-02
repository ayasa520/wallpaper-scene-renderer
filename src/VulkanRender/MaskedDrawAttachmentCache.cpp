#include "MaskedDrawAttachmentCache.hpp"

#include "Utils/Logging.h"
#include "vvk/vma_wrapper.hpp"

#include <optional>
#include <utility>

namespace wallpaper::vulkan
{
namespace
{

std::optional<VmaImageParameters> CreateMaskedDrawAttachment(const Device& device,
                                                             VkExtent3D extent,
                                                             VkFormat format) {
    VmaImageParameters image;
    VkImageCreateInfo info {
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = nullptr,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = format,
        .extent                = extent,
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        .usage                 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    image.extent       = extent;
    image.mipmap_level = 1;

    VmaAllocationCreateInfo allocation_info {};
    allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VVK_CHECK_ACT(return std::nullopt,
                  vvk::CreateImage(device.vma_allocator(), info, allocation_info, image.handle));

    VkImageViewCreateInfo view_info {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .image    = *image.handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = format,
        .subresourceRange =
            VkImageSubresourceRange {
                .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
    };
    VVK_CHECK_ACT(return std::nullopt, device.handle().CreateImageView(view_info, image.view));
    return image;
}

} // namespace

VmaImageParameters* MaskedDrawAttachmentCache::acquire(const Device& device,
                                                       std::string_view output,
                                                       VkExtent3D extent,
                                                       VkFormat format) {
    auto& entry = m_entries[std::string(output)];
    const bool missing = ! entry.image.view || ! entry.image.handle;
    const bool wrong_size = entry.image.extent.width != extent.width ||
                            entry.image.extent.height != extent.height ||
                            entry.image.extent.depth != extent.depth;
    if (missing || wrong_size || entry.format != format) {
        auto replacement = CreateMaskedDrawAttachment(device, extent, format);
        if (! replacement.has_value()) return nullptr;

        // Log only allocation boundaries, never cache hits. This keeps normal frame preparation
        // quiet while making scene switches and output resizes auditable from run.log: every
        // replacement must be followed by exactly one cache clear or device-fault abandon.
        LOG_INFO("MaskedDrawAttachmentCache: %s output='%.*s' extent=[%u,%u,%u] format=%d",
                 missing ? "create" : "replace",
                 static_cast<int>(output.size()),
                 output.data(),
                 extent.width,
                 extent.height,
                 extent.depth,
                 static_cast<int>(format));
        entry.format = format;
        entry.image  = std::move(*replacement);
    }
    return &entry.image;
}

void MaskedDrawAttachmentCache::clear() {
    if (! m_entries.empty()) {
        LOG_INFO("MaskedDrawAttachmentCache: clear entries=%zu", m_entries.size());
    }
    m_entries.clear();
}

void MaskedDrawAttachmentCache::abandon() {
    if (! m_entries.empty()) {
        LOG_INFO("MaskedDrawAttachmentCache: abandon entries=%zu", m_entries.size());
    }
    for (auto& [_, entry] : m_entries) {
        entry.image.sampler.abandon();
        entry.image.view.abandon();
        entry.image.handle.abandon();
    }
    m_entries.clear();
}

} // namespace wallpaper::vulkan

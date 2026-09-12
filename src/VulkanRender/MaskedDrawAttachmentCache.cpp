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
                                                             VkExtent3D extent) {
    VmaImageParameters image;
    VkImageCreateInfo info {
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = nullptr,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = VK_FORMAT_R8_UNORM,
        .extent                = extent,
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        .usage                 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    image.extent       = extent;
    image.mipmap_level = 1;
    image.samples      = 1;

    VmaAllocationCreateInfo allocation_info {};
    allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VVK_CHECK_ACT(return std::nullopt,
                  vvk::CreateImage(device.vma_allocator(), info, allocation_info, image.handle));

    VkImageViewCreateInfo view_info {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .image    = *image.handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = VK_FORMAT_R8_UNORM,
        .subresourceRange =
            VkImageSubresourceRange {
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
    };
    VVK_CHECK_ACT(return std::nullopt, device.handle().CreateImageView(view_info, image.view));
    VkSamplerCreateInfo sampler_info {
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod       = 0.0f,
    };
    VVK_CHECK_ACT(return std::nullopt, device.handle().CreateSampler(sampler_info, image.sampler));
    return image;
}

} // namespace

std::shared_ptr<VmaImageParameters> MaskedDrawAttachmentCache::acquire(
    const Device& device, std::string_view output, VkExtent3D extent, Role role) {
    auto& images = m_entries[std::string(output)];
    auto& entry = role == Role::Accumulated ? images.accumulated : images.intermediate;
    const bool missing = entry == nullptr;
    const bool wrong_size = !missing && (entry->extent.width != extent.width ||
                            entry->extent.height != extent.height ||
                            entry->extent.depth != extent.depth);
    if (missing || wrong_size) {
        auto replacement = CreateMaskedDrawAttachment(device, extent);
        if (! replacement.has_value()) return nullptr;

        // Groups reuse accumulated coverage and allocate the intermediate only when a
        // parent chain needs it. Each intermediate is consumed before the next writer;
        // neither chain depth nor destination MSAA requires an image per puppet part.
        LOG_INFO("MaskedDrawAttachmentCache: %s output='%.*s' extent=[%u,%u,%u] format=R8 role=%s",
                 missing ? "create" : "replace",
                 static_cast<int>(output.size()),
                 output.data(),
                 extent.width,
                 extent.height,
                 extent.depth,
                 role == Role::Accumulated ? "accumulated" : "intermediate");
        entry = std::make_shared<VmaImageParameters>(std::move(*replacement));
    }
    return entry;
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
    for (auto& [_, images] : m_entries) {
        for (auto* entry : {images.accumulated.get(), images.intermediate.get()}) {
            if (entry == nullptr) continue;
            entry->sampler.abandon();
            entry->view.abandon();
            entry->handle.abandon();
        }
    }
    m_entries.clear();
}

} // namespace wallpaper::vulkan

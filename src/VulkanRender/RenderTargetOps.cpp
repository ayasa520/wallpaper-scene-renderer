#include "RenderTargetOps.hpp"

#include "Msaa.hpp"
#include "Resource.hpp"
#include "Utils/Logging.h"
#include "Vulkan/Device.hpp"

#include <cstdlib>
#include <optional>

namespace wallpaper::vulkan
{
namespace
{
std::optional<VmaImageParameters> CreateSceneDepthImage(const Device& device, VkExtent3D extent,
                                                       VkSampleCountFlagBits samples) {
    VmaImageParameters image;
    VkImageCreateInfo info {
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = ModelDepthAttachment::format,
        .extent                = extent,
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = samples,
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        // Scene depth is an attachment and a sampled input. Transfer destination access is
        // needed for the stage's far-depth clear; multisample resolves use attachments.
        .usage                 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    image.extent       = extent;
    image.mipmap_level = 1;
    image.samples      = static_cast<uint>(samples);

    VmaAllocationCreateInfo allocation {};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VVK_CHECK_ACT(return std::nullopt,
                  vvk::CreateImage(device.vma_allocator(), info, allocation, image.handle));
    VkImageViewCreateInfo view_info {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = *image.handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = ModelDepthAttachment::format,
        .subresourceRange = VkImageSubresourceRange {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    VVK_CHECK_ACT(return std::nullopt, device.handle().CreateImageView(view_info, image.view));
    return image;
}
} // namespace

VmaImageParameters* AcquireSceneDepthImage(const Device& device, RenderingResources& rr,
                                          std::string_view target, VkExtent3D extent,
                                          VkSampleCountFlagBits samples) {
    const std::string key(target);
    auto& attachment = rr.model_depth_images[key];
    auto& depth = attachment.image;
    const bool missing = !depth.view || !depth.handle;
    const bool wrong_size = depth.extent.width != extent.width ||
                            depth.extent.height != extent.height ||
                            depth.extent.depth != extent.depth;
    const bool wrong_samples = depth.samples != static_cast<uint>(samples);
    if (missing || wrong_size || wrong_samples) {
        // Allocation is shared across draw kinds, not copied into each pass. Resource refresh
        // drops destination framebuffers before replacing their views; the new allocation starts
        // uninitialized so the stage clear/layout operation runs on this generation as well.
        auto replacement = CreateSceneDepthImage(device, extent, samples);
        if (!replacement.has_value()) {
            LOG_ERROR("SceneDepthAttachment: cannot create output='%s' extent=[%u,%u] samples=%u",
                      key.c_str(), extent.width, extent.height, static_cast<unsigned>(samples));
            return nullptr;
        }
        if (wallpaper::diagnostics::Options().trace_depth_attachments) {
            LOG_INFO("SceneDepthAttachmentAllocate: output='%s' "
                     "old-image=%p old-view=%p old-extent=%ux%ux%u old-samples=%u "
                     "new-image=%p new-view=%p new-extent=%ux%ux%u new-samples=%u",
                     key.c_str(),
                     depth.handle ? reinterpret_cast<void*>(*depth.handle) : nullptr,
                     depth.view ? reinterpret_cast<void*>(*depth.view) : nullptr,
                     depth.extent.width, depth.extent.height, depth.extent.depth, depth.samples,
                     reinterpret_cast<void*>(*replacement->handle),
                     reinterpret_cast<void*>(*replacement->view),
                     replacement->extent.width, replacement->extent.height,
                     replacement->extent.depth, replacement->samples);
        }
        // Its framebuffer holds the old source view as well as the resolved view.
        // Destroy that dependency before replacing either image allocation.
        attachment.resolve.reset();
        depth = std::move(replacement.value());
        attachment.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.resolve_dirty = false;
    }
    if (samples > VK_SAMPLE_COUNT_1_BIT && !attachment.resolve.has_value()) {
        auto replacement = CreateSceneDepthImage(device, extent, VK_SAMPLE_COUNT_1_BIT);
        if (!replacement.has_value()) {
            LOG_ERROR("SceneDepthResolve: cannot create output='%s' extent=[%u,%u]",
                      key.c_str(), extent.width, extent.height);
            return nullptr;
        }
        ModelDepthResolve resolved;
        resolved.image = std::move(replacement.value());
        if (!CreateModelDepthResolve(device, depth, resolved)) return nullptr;
        attachment.resolve.emplace(std::move(resolved));
    }
    return &depth;
}

void ClearRenderTargetColor(vvk::CommandBuffer& cmd, const ImageParameters& image,
                            const VkClearColorValue& color, VkImageLayout final_layout) {
    const VkImageSubresourceRange range {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
    // A clear discards content, not prior users of the allocation. In particular, the
    // main stage follows reflection, which can have sampled the previous main color.
    // Order those reads/writes before the transfer even though oldLayout is UNDEFINED.
    const VkImageMemoryBarrier to_transfer {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.handle,
        .subresourceRange = range,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, to_transfer);
    cmd.ClearColorImage(image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, range);

    const VkImageMemoryBarrier to_final {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = final_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.handle,
        .subresourceRange = range,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, to_final);
}

SceneDepthAction PrepareSceneModelDepth(RenderingResources& rr, std::string_view target,
                                        bool clear) {
    const auto it = rr.model_depth_images.find(std::string(target));
    if (it == rr.model_depth_images.end() || !it->second.image.handle)
        return SceneDepthAction::Unused;
    auto& attachment = it->second;
    auto& image = attachment.image;
    auto& cmd = rr.command;
    if (!clear && attachment.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        return SceneDepthAction::Preserved;

    const VkImageSubresourceRange range {
        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const VkAccessFlags depth_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    const VkPipelineStageFlags depth_stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    const bool first_use = attachment.layout == VK_IMAGE_LAYOUT_UNDEFINED;
    const VkImageMemoryBarrier before {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = first_use ? 0u : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = clear ? VkAccessFlags { VK_ACCESS_TRANSFER_WRITE_BIT } : depth_access,
        .oldLayout = attachment.layout,
        .newLayout = clear ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                           : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image.handle,
        .subresourceRange = range,
    };
    cmd.PipelineBarrier(first_use ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                  : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        clear ? VkPipelineStageFlags { VK_PIPELINE_STAGE_TRANSFER_BIT }
                              : depth_stages,
                        0, before);

    if (clear) {
        // The stage, not its first visible model, owns the far-depth clear. A disabled clear only
        // establishes the new image's attachment layout; it must preserve any existing history.
        const VkClearDepthStencilValue value { 0.0f, 0 };
        cmd.ClearDepthStencilImage(*image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   &value, range);
        const VkImageMemoryBarrier after {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = depth_access,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image.handle,
            .subresourceRange = range,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, depth_stages, 0, after);
        // A stage clear writes depth even if every model is hidden. Its sampling
        // consumer must not reuse a resolved image from a preceding visible frame.
        if (image.samples > 1) attachment.resolve_dirty = true;
    }
    attachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    return clear ? SceneDepthAction::Cleared : SceneDepthAction::LayoutInitialized;
}

std::string_view SceneDepthActionName(SceneDepthAction action) {
    switch (action) {
    case SceneDepthAction::Unused: return "unused";
    case SceneDepthAction::Preserved: return "preserved";
    case SceneDepthAction::LayoutInitialized: return "layout-initialized";
    case SceneDepthAction::Cleared: return "cleared";
    }
    return {};
}

} // namespace wallpaper::vulkan

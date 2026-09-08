#include "RenderTargetOps.hpp"

#include "Resource.hpp"

namespace wallpaper::vulkan
{

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
        if (image.samples > 1) rr.model_depth_dirty.insert(it->first);
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

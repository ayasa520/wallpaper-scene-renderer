#include "Msaa.hpp"

#include "PassCommon.hpp"
#include "Resource.hpp"
#include "RenderCommandTrace.hpp"
#include "Utils/Logging.h"

#include <array>

namespace wallpaper::vulkan
{

void NoteComposeMsaaDraw(RenderingResources& rr, VkSampleCountFlagBits samples) {
    if (samples <= VK_SAMPLE_COUNT_1_BIT) return;
    rr.msaa_compose_dirty = true;
}

bool ResolveComposeMsaaIfNeeded(Scene& scene, const Device& device, RenderingResources& rr) {
    if (! rr.msaa_compose_dirty) return false;
    if (scene.MsaaSampleCount() <= 1) {
        rr.msaa_compose_dirty = false;
        return false;
    }

    const auto ms_name  = std::string(SpecTex_DefaultMS);
    const auto def_name = std::string(SpecTex_Default);
    const auto ms_it    = scene.renderTargets.find(ms_name);
    const auto def_it   = scene.renderTargets.find(def_name);
    if (ms_it == scene.renderTargets.end() || def_it == scene.renderTargets.end()) {
        rr.msaa_compose_dirty = false;
        return false;
    }

    auto ms = device.tex_cache().Query(ms_name, ToTexKey(ms_it->second), ! ms_it->second.allowReuse);
    auto resolved =
        device.tex_cache().Query(def_name, ToTexKey(def_it->second), ! def_it->second.allowReuse);
    if (! ms.has_value() || ! resolved.has_value() || ! ms->handle || ! resolved->handle) {
        LOG_ERROR("SceneAAResolve: missing compose images ms=%s resolved=%s",
                  ms.has_value() && ms->handle ? "ok" : "missing",
                  resolved.has_value() && resolved->handle ? "ok" : "missing");
        return false;
    }

    auto&                   cmd = rr.command;
    VkImageSubresourceRange range {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    std::array<VkImageMemoryBarrier, 2> to_transfer {
        VkImageMemoryBarrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image            = ms->handle,
            .subresourceRange = range,
        },
        VkImageMemoryBarrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask    = 0,
            .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
            // Resolve replaces the 1x image; discard prior destination contents.
            .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image            = resolved->handle,
            .subresourceRange = range,
        },
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        {},
                        {},
                        to_transfer);

    std::array<VkImageResolve, 1> regions {
        VkImageResolve {
            .srcSubresource =
                VkImageSubresourceLayers {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            .dstSubresource =
                VkImageSubresourceLayers {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            .extent = { ms->extent.width, ms->extent.height, 1 },
        },
    };
    cmd.ResolveImage(ms->handle,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     resolved->handle,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     regions);

    std::array<VkImageMemoryBarrier, 2> after_resolve {
        VkImageMemoryBarrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image            = ms->handle,
            .subresourceRange = range,
        },
        VkImageMemoryBarrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = resolved->handle,
            .subresourceRange = range,
        },
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        {},
                        {},
                        after_resolve);

    const auto command = TraceRenderCommand(rr, "compose-resolve", "recorded", def_name, *resolved);
    TraceRenderCommandInput(rr, command, "resolve-source", ms_name, *ms, true);
    rr.msaa_compose_dirty = false;
    return true;
}

bool CreateModelDepthResolve(const Device& device, const VmaImageParameters& source,
                             ModelDepthResolve& resolve) {
    // Resolve a completed scene depth attachment without redrawing the scene. LOAD/STORE
    // preserve its samples for later depth users; the single-sample destination is fully
    // replaced and ends ready for sampling. SAMPLE_ZERO is required by the depth-resolve
    // extension and retains an actual depth value instead of averaging unrelated surfaces.
    const std::array attachments {
        VkAttachmentDescription2 {
            .sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            .format         = ModelDepthAttachment::format,
            .samples        = static_cast<VkSampleCountFlagBits>(source.samples),
            .loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
        VkAttachmentDescription2 {
            .sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            .format         = ModelDepthAttachment::format,
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
    };
    const VkAttachmentReference2 source_ref {
        .sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
    };
    const VkAttachmentReference2 resolve_ref {
        .sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
        .attachment = 1,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
    };
    const VkSubpassDescriptionDepthStencilResolve depth_resolve {
        .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
        .depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
        .stencilResolveMode = VK_RESOLVE_MODE_NONE,
        .pDepthStencilResolveAttachment = &resolve_ref,
    };
    const VkSubpassDescription2 subpass {
        .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
        .pNext = &depth_resolve,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .pDepthStencilAttachment = &source_ref,
    };

    // Fixed-function depth resolves execute at COLOR_ATTACHMENT_OUTPUT and use color
    // attachment access masks, while depth LOAD/STORE use fragment-test stages. Include
    // both scopes. Discarding the destination's contents does not discard earlier shader
    // reads: those must finish before the resolve overwrites it. This dependency is global,
    // since the depth consumer samples a full-size image into a smaller lighting target.
    const VkPipelineStageFlags depth_stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    const VkPipelineStageFlags resolve_stages = depth_stages |
                                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkAccessFlags depth_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    const VkAccessFlags resolve_access = depth_access | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    const std::array dependencies {
        VkSubpassDependency2 {
            .sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = resolve_stages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
            .dstStageMask = resolve_stages,
            .srcAccessMask = depth_access | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = resolve_access,
        },
        VkSubpassDependency2 {
            .sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = resolve_stages,
            .dstStageMask = depth_stages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask = resolve_access,
            .dstAccessMask = depth_access | VK_ACCESS_SHADER_READ_BIT,
        },
    };
    const VkRenderPassCreateInfo2 pass_info {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies = dependencies.data(),
    };
    VVK_CHECK_BOOL_RE(device.handle().CreateRenderPass2KHR(pass_info, resolve.pass));

    const std::array views { *source.view, *resolve.image.view };
    const VkFramebufferCreateInfo framebuffer_info {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = *resolve.pass,
        .attachmentCount = static_cast<uint32_t>(views.size()),
        .pAttachments = views.data(),
        .width = source.extent.width,
        .height = source.extent.height,
        .layers = 1,
    };
    VVK_CHECK_BOOL_RE(device.handle().CreateFramebuffer(framebuffer_info, resolve.framebuffer));
    return true;
}

bool ResolveModelDepthIfNeeded(RenderingResources& rr, std::string_view output) {
    const auto it = rr.model_depth_images.find(std::string(output));
    if (it == rr.model_depth_images.end()) return false;
    auto& attachment = it->second;
    if (!attachment.resolve_dirty || !attachment.resolve.has_value()) return false;

    const auto& resolve = *attachment.resolve;
    const VkRenderPassBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = *resolve.pass,
        .framebuffer = *resolve.framebuffer,
        .renderArea = { { 0, 0 }, { attachment.image.extent.width, attachment.image.extent.height } },
    };
    rr.command.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
    rr.command.EndRenderPass();
    const auto command = TraceRenderCommand(rr, "depth-resolve", "recorded", output, resolve.image);
    TraceRenderCommandInput(rr, command, "resolve-source", output, attachment.image, true);
    attachment.resolve_dirty = false;
    return true;
}

} // namespace wallpaper::vulkan

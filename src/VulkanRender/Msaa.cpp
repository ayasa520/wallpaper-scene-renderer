#include "Msaa.hpp"

#include "PassCommon.hpp"
#include "Resource.hpp"
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
            // Official resolve replaces the 1x image; discard prior dest contents.
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

    rr.msaa_compose_dirty = false;
    return true;
}

} // namespace wallpaper::vulkan

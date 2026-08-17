#include "PrePass.hpp"
#include "PassCommon.hpp"
#include "Scene/Scene.h"
#include "Resource.hpp"

using namespace wallpaper::vulkan;

namespace
{

void BindPrePassTarget(wallpaper::Scene& scene, const Device& device, std::string_view name,
                       ImageParameters& image, bool& ok) {
    ok = false;
    const auto tex_name = std::string(name);
    if (scene.renderTargets.count(tex_name) == 0) return;
    auto& rt = scene.renderTargets.at(tex_name);
    if (auto opt = device.tex_cache().Query(tex_name, ToTexKey(rt), ! rt.allowReuse);
        opt.has_value()) {
        image = opt.value();
        ok    = static_cast<bool>(image.handle);
    }
}

void ClearPrePassTarget(vvk::CommandBuffer& cmd, const ImageParameters& image,
                        const VkClearColorValue& color, VkImageLayout final_layout,
                        VkAccessFlags dst_access, VkPipelineStageFlags dst_stage) {
    if (! image.handle) return;
    VkImageSubresourceRange range {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = VK_REMAINING_ARRAY_LAYERS,
        .baseArrayLayer = 0,
        .layerCount     = VK_REMAINING_MIP_LEVELS,
    };
    VkImageMemoryBarrier to_dst {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext            = nullptr,
        .srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT,
        .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image            = image.handle,
        .subresourceRange = range,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        to_dst);
    cmd.ClearColorImage(image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, range);
    VkImageMemoryBarrier to_final {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext            = nullptr,
        .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask    = dst_access,
        .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout        = final_layout,
        .image            = image.handle,
        .subresourceRange = range,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        dst_stage,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        to_final);
}

} // namespace

PrePass::PrePass(const Desc& desc): m_desc(desc) {}
PrePass::~PrePass() {}

bool PrePass::referencesRenderTarget(std::string_view render_target) const {
    // The pre-pass only clears the graph result target. It should stay out of text bridge refreshes
    // unless the default framebuffer-sized target itself was the resource that changed.
    return m_desc.result == render_target || render_target == SpecTex_DefaultMS;
}

void PrePass::prepare(Scene& scene, const Device& device, RenderingResources&) {
    bool have_result = false;
    BindPrePassTarget(scene, device, m_desc.result, m_desc.vk_result, have_result);
    if (! have_result) return;
    BindPrePassTarget(scene, device, SpecTex_DefaultMS, m_desc.vk_msaa, m_desc.has_msaa);
    {
        auto& sc           = scene.clearColor;
        m_desc.clear_value = VkClearValue { sc[0], sc[1], sc[2], 1.0f };
    }
    setPrepared();
}

void PrePass::refreshResources(Scene& scene, const Device& device, RenderingResources&) {
    // Resource-only rebuilds clear the texture cache without discarding the pre-pass object. The
    // clear pass therefore has to re-query the recreated render-target image here; otherwise the
    // next frame would keep clearing a stale Vulkan image handle and the renderer can fall into a
    // black frame or crash once text-driven effect resources are rebuilt in place.
    bool have_result = false;
    BindPrePassTarget(scene, device, m_desc.result, m_desc.vk_result, have_result);
    if (! have_result) {
        setPrepared(false);
        return;
    }
    BindPrePassTarget(scene, device, SpecTex_DefaultMS, m_desc.vk_msaa, m_desc.has_msaa);

    auto& sc = scene.clearColor;
    m_desc.clear_value = VkClearValue { sc[0], sc[1], sc[2], 1.0f };
}

void PrePass::execute(const Device&, RenderingResources& rr) {
    auto& cmd = rr.command;
    ClearPrePassTarget(cmd,
                       m_desc.vk_result,
                       m_desc.clear_value.color,
                       m_desc.layout,
                       VK_ACCESS_MEMORY_READ_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    if (m_desc.has_msaa) {
        // Compose draws write this target and do not write alpha. TextureCache
        // initializes new RTs to transparent black, so resolve would copy A=0
        // over the opaque `_rt_default` clear and the compositor shows black.
        ClearPrePassTarget(cmd,
                           m_desc.vk_msaa,
                           m_desc.clear_value.color,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
}
void PrePass::destory(const Device&, RenderingResources&) {
    setPrepared(false);
    clearReleaseTexs();
}

#include "FinPass.hpp"
#include "Vulkan/Shader.hpp"
#include "Resource.hpp"
#include "PassCommon.hpp"

#include <array>

using namespace wallpaper::vulkan;

constexpr std::string_view vert_code = R"(
struct VSInput {
    [[vk::location(0)]] float3 Position : POSITION0;
    [[vk::location(1)]] float2 Texcoord : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

VSOutput main_vs(VSInput input) {
    VSOutput output;
    output.texcoord = input.Texcoord;
    output.position = float4(input.Position, 1.0);
    return output;
}
)";

constexpr std::string_view frag_code = R"(
struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float4> u_Texture;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState u_Texture_ww_sampler;

float4 main_ps(PSInput input) : SV_Target0 {
    return u_Texture.Sample(u_Texture_ww_sampler, input.texcoord);
}
)";

struct VertexInput {
    std::array<float, 3> pos;
    std::array<float, 2> color;
};

constexpr std::array vertex_input = {
    VertexInput { { -1.0f, -1.0f, 0.0f }, { 0.0f, 1.0f } },
    VertexInput { { -1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
    VertexInput { { 1.0f, -1.0f, 0.0f }, { 1.0f, 1.0f } },
    VertexInput { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } },
};

FinPass::FinPass(const Desc&) {}
FinPass::~FinPass() {}

bool FinPass::referencesRenderTarget(std::string_view render_target) const {
    // The final present pass samples only the render-graph result target. Text bridge target
    // resizes never need to rebind this pass unless the default result image itself was recreated.
    return m_desc.result == render_target;
}

namespace
{
std::optional<vvk::RenderPass> CreateRenderPass(const vvk::Device& device, VkFormat format,
                                                VkImageLayout finalLayout) {
    VkAttachmentDescription attachment {
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = finalLayout,
    };
    VkAttachmentReference attachment_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass {
        .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &attachment_ref,
    };

    VkRenderPassCreateInfo creatinfo {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
    };
    vvk::RenderPass pass;
    if (auto res = device.CreateRenderPass(creatinfo, pass); res == VK_SUCCESS) {
        return pass;
    } else {
        VVK_CHECK(res);
        return std::nullopt;
    }
}
} // namespace

void FinPass::setPresent(ImageParameters img) { m_desc.vk_present = img; }
void FinPass::setPresentLayout(VkImageLayout layout) { m_desc.present_layout = layout; }
void FinPass::setPresentFormat(VkFormat format) { m_desc.present_format = format; }
void FinPass::setPresentQueueIndex(uint32_t i) { m_desc.present_queue_index = i; }

void FinPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    {
        auto tex_name = std::string(m_desc.result);
        if (scene.renderTargets.count(tex_name) == 0) return;
        auto& rt = scene.renderTargets.at(tex_name);
        if (auto opt = device.tex_cache().Query(tex_name, ToTexKey(rt), ! rt.allowReuse);
            opt.has_value()) {
            m_desc.vk_result = opt.value();
        }
    }
    std::vector<Uni_ShaderSpv> spvs;
    {
        ShaderCompOpt opt;
        opt.target_env = ShaderTargetEnv::VULKAN_1_1;

        std::array<ShaderCompUnit, 2> units;
        units[0] = ShaderCompUnit {
            .stage = wallpaper::ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "FinPass.vert",
            .entry_point = "main_vs",
            .src = std::string(vert_code),
        };
        units[1] = ShaderCompUnit {
            .stage = wallpaper::ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "FinPass.frag",
            .entry_point = "main_ps",
            .src = std::string(frag_code),
        };
        CompileAndLinkShaderUnits(units, opt, spvs);
    }

    VkVertexInputBindingDescription                bind_description;
    std::vector<VkVertexInputAttributeDescription> attr_descriptions;
    {
        bind_description.stride    = (sizeof(VertexInput));
        bind_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bind_description.binding   = (0);
        VkVertexInputAttributeDescription attr_pos, attr_color;
        attr_pos.binding    = (0);
        attr_pos.location   = (0);
        attr_pos.format     = VK_FORMAT_R32G32B32_SFLOAT;
        attr_pos.offset     = offsetof(VertexInput, pos);
        attr_color.binding  = (0);
        attr_color.location = (1);
        attr_color.format   = VK_FORMAT_R32G32_SFLOAT;
        attr_color.offset   = (offsetof(VertexInput, color));

        attr_descriptions.push_back(attr_pos);
        attr_descriptions.push_back(attr_color);

        {
            auto& buf = m_desc.vertex_buf;
            rr.vertex_buf->allocateSubRef(sizeof(decltype(vertex_input)), buf);
            rr.vertex_buf->writeToBuf(buf, { (uint8_t*)vertex_input.data(), buf.size });
        }
    }
    DescriptorSetInfo descriptor_info;
    {
        descriptor_info.push_descriptor = true;
        descriptor_info.bindings.resize(1);
        auto& binding           = descriptor_info.bindings.back();
        binding.binding         = (1);
        binding.descriptorCount = (1);
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    {
        auto opt = CreateRenderPass(device.handle(), m_desc.present_format, m_desc.present_layout);
        if (! opt.has_value()) return;
        auto pass = std::move(opt.value());

        descriptor_info.push_descriptor = true;
        GraphicsPipeline pipeline;
        pipeline.toDefault();
        m_desc.pipeline.debug_name = "FinPass";
        pipeline.addDescriptorSetInfo(spanone { descriptor_info })
            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
            .addInputBindingDescription(spanone { bind_description })
            .addInputAttributeDescription(attr_descriptions);
        for (auto& spv : spvs) pipeline.addStage(std::move(spv));

        if (! pipeline.create(device, pass, m_desc.pipeline)) return;
    }
    /*
    if(m_desc.present_layout == vk::ImageLayout::ePresentSrcKHR || m_desc.present_layout ==
    vk::ImageLayout::eSharedPresentKHR) m_desc.render_layout = m_desc.present_layout; else
    m_desc.render_layout = vk::ImageLayout::eColorAttachmentOptimal;
    */

    m_desc.render_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    {
        auto& sc           = scene.clearColor;
        m_desc.clear_value = VkClearValue { { sc[0], sc[1], sc[2], 1.0f } };
    }
    setPrepared();
}

void FinPass::refreshResources(Scene& scene, const Device& device, RenderingResources&) {
    // The final composite pass keeps its fullscreen mesh and pipeline across resource-only
    // refreshes. The only scene-owned object it samples is the render-graph result texture, so we
    // can re-query that cache entry and keep the existing pipeline hot instead of recompiling it.
    auto tex_name = std::string(m_desc.result);
    if (scene.renderTargets.count(tex_name) == 0) {
        setPrepared(false);
        return;
    }
    auto& rt = scene.renderTargets.at(tex_name);
    if (auto opt = device.tex_cache().Query(tex_name, ToTexKey(rt), !rt.allowReuse);
        opt.has_value()) {
        m_desc.vk_result = opt.value();
    } else {
        setPrepared(false);
    }
}

void FinPass::execute(const Device& device, RenderingResources& rr) {
    auto& cmd = rr.command;
    auto& src = m_desc.vk_result;
    auto& dst = m_desc.vk_present;
    if (!(src.handle && dst.handle) || src.extent.width == 0 || src.extent.height == 0 ||
        dst.extent.width == 0 || dst.extent.height == 0) {
        return;
    }

    // Present used to be a fullscreen sample+draw that also cleared the DMA-BUF
    // image. At 3200x2000 that pass alone was about half of gpu_draw. The
    // shader is an identity copy (negative viewport cancels the mesh UVs), so
    // a transfer matches the pixels and skips the fragment work plus the
    // per-frame framebuffer allocate.
    const uint32_t gfx_family = device.graphics_queue().family_index;
    const bool     qf_release = m_desc.present_queue_index != gfx_family;
    const uint32_t src_w      = src.extent.width;
    const uint32_t src_h      = src.extent.height;
    const uint32_t dst_w      = dst.extent.width;
    const uint32_t dst_h      = dst.extent.height;

    VkImageSubresourceRange srang {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    VkImageMemoryBarrier in_bar {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext               = nullptr,
        .srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT,
        .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = src.handle,
        .subresourceRange    = srang,
    };
    VkImageMemoryBarrier out_bar {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext               = nullptr,
        .srcAccessMask       = 0,
        .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = qf_release ? m_desc.present_queue_index : VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = qf_release ? gfx_family : VK_QUEUE_FAMILY_IGNORED,
        .image               = dst.handle,
        .subresourceRange    = srang,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        {},
                        {},
                        std::array { in_bar, out_bar });

    if (src_w == dst_w && src_h == dst_h) {
        VkImageCopy copy {
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
            .extent = { src_w, src_h, 1 },
        };
        cmd.CopyImage(src.handle,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      dst.handle,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      copy);
    } else {
        VkImageBlit blit {
            .srcSubresource =
                VkImageSubresourceLayers {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            .srcOffsets     = { VkOffset3D { 0, 0, 0 },
                                VkOffset3D { static_cast<int32_t>(src_w),
                                             static_cast<int32_t>(src_h),
                                             1 } },
            .dstSubresource =
                VkImageSubresourceLayers {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            .dstOffsets     = { VkOffset3D { 0, 0, 0 },
                                VkOffset3D { static_cast<int32_t>(dst_w),
                                             static_cast<int32_t>(dst_h),
                                             1 } },
        };
        cmd.BlitImage(src.handle,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      dst.handle,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      blit,
                      VK_FILTER_LINEAR);
    }

    in_bar.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    in_bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    in_bar.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    in_bar.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    out_bar.srcAccessMask      = VK_ACCESS_TRANSFER_WRITE_BIT;
    out_bar.dstAccessMask      = VK_ACCESS_MEMORY_READ_BIT;
    out_bar.oldLayout          = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    out_bar.newLayout          = m_desc.present_layout;
    out_bar.srcQueueFamilyIndex = qf_release ? gfx_family : VK_QUEUE_FAMILY_IGNORED;
    out_bar.dstQueueFamilyIndex = qf_release ? m_desc.present_queue_index : VK_QUEUE_FAMILY_IGNORED;
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        {},
                        {},
                        std::array { in_bar, out_bar });
}
void FinPass::destory(const Device&, RenderingResources& rr) {
    setPrepared(false);
    clearReleaseTexs();
    rr.vertex_buf->unallocateSubRef(m_desc.vertex_buf);
}

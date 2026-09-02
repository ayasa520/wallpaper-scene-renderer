#include "VolumetricsSingleFillPass.hpp"

#include "Core/ArrayHelper.hpp"
#include "Msaa.hpp"
#include "PassCommon.hpp"
#include "Resource.hpp"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include "Vulkan/Shader.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

using namespace wallpaper::vulkan;

namespace
{

constexpr std::string_view kVert = R"(
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
    output.position = float4(input.Position, 1.0);
    output.texcoord = input.Texcoord;
    return output;
}
)";

constexpr std::string_view kFrag = R"(
struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float> u_SceneDepth;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState u_SceneDepth_ww_sampler;

float4 main_ps(PSInput input) : SV_Target0 {
    // Depth framebuffer blit. Depth stretch is NEAREST.
    // Dest UV 0-1 maps onto the full scene depth, matching StretchRect/glBlitFramebuffer.
    const float depth = u_SceneDepth.SampleLevel(u_SceneDepth_ww_sampler, input.texcoord, 0);
    return float4(depth, depth, depth, 1.0);
}
)";

struct VertexInput {
    std::array<float, 3> pos;
    std::array<float, 2> uv;
};

// Vulkan-native viewport (y=0, +height): NDC y=-1 is the top of the framebuffer,
// matching depth-image texel (0,0).
constexpr std::array kQuad = {
    VertexInput { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
    VertexInput { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    VertexInput { { 1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } },
    VertexInput { { 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f } },
};

std::optional<vvk::RenderPass> CreateFillRenderPass(const vvk::Device& device) {
    VkAttachmentDescription attachment {
        .format         = VK_FORMAT_R8G8B8A8_UNORM,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkAttachmentReference color_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription subpass {
        .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &color_ref,
    };
    VkRenderPassCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
    };
    vvk::RenderPass pass;
    if (device.CreateRenderPass(info, pass) != VK_SUCCESS) return std::nullopt;
    return pass;
}

} // namespace

VolumetricsSingleFillPass::VolumetricsSingleFillPass(const Desc& desc): m_desc(desc) {}

VolumetricsSingleFillPass::~VolumetricsSingleFillPass() = default;

std::string VolumetricsSingleFillPass::residencyKey() const {
    return "VolumetricsSingleFill|dst=" + m_desc.dst + "|src=" + m_desc.scene_output;
}

bool VolumetricsSingleFillPass::referencesRenderTarget(std::string_view render_target) const {
    return m_desc.dst == render_target || m_desc.scene_output == render_target;
}

bool VolumetricsSingleFillPass::ensurePipeline(const Device& device, RenderingResources& rr) {
    if (m_pipeline.handle) return true;

    ShaderCompOpt opt;
    opt.target_env = ShaderTargetEnv::VULKAN_1_1;
    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage            = wallpaper::ShaderType::VERTEX,
            .source_language  = ShaderSourceLanguage::HLSL,
            .debug_name       = "VolumetricsSingleFill.vert",
            .entry_point      = "main_vs",
            .src              = std::string(kVert),
        },
        ShaderCompUnit {
            .stage            = wallpaper::ShaderType::FRAGMENT,
            .source_language  = ShaderSourceLanguage::HLSL,
            .debug_name       = "VolumetricsSingleFill.frag",
            .entry_point      = "main_ps",
            .src              = std::string(kFrag),
        },
    };
    std::vector<Uni_ShaderSpv> spvs;
    if (! CompileAndLinkShaderUnits(units, opt, spvs)) {
        LOG_ERROR("VolumetricsSingleFill: shader compile failed");
        return false;
    }

    auto pass_opt = CreateFillRenderPass(device.handle());
    if (! pass_opt.has_value()) return false;
    auto pass = std::move(pass_opt.value());

    VkVertexInputBindingDescription bind {
        .binding   = 0,
        .stride    = sizeof(VertexInput),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    std::array attr {
        VkVertexInputAttributeDescription {
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = offsetof(VertexInput, pos),
        },
        VkVertexInputAttributeDescription {
            .location = 1,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32_SFLOAT,
            .offset   = offsetof(VertexInput, uv),
        },
    };

    DescriptorSetInfo descriptor_info;
    descriptor_info.push_descriptor = true;
    descriptor_info.bindings        = {
        VkDescriptorSetLayoutBinding {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    GraphicsPipeline pipeline;
    pipeline.toDefault();
    m_pipeline.debug_name = "VolumetricsSingleFill";
    m_pipeline.cache_key  = "VolumetricsSingleFill|rgba8|nearest-depth-blit";
    pipeline.addDescriptorSetInfo(spanone { descriptor_info })
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        .addInputBindingDescription(spanone { bind })
        .addInputAttributeDescription(attr);
    for (auto& spv : spvs) pipeline.addStage(std::move(spv));
    if (! pipeline.create(device, pass, m_pipeline, rr.pipeline_cache.get())) {
        LOG_ERROR("VolumetricsSingleFill: pipeline create failed");
        return false;
    }

    if (! m_vertex_buf) {
        if (! rr.vertex_buf->allocateSubRef(sizeof(kQuad), m_vertex_buf)) return false;
        rr.vertex_buf->writeToBuf(m_vertex_buf, { (uint8_t*)kQuad.data(), m_vertex_buf.size });
    }

    if (! m_point_sampler) {
        VkSamplerCreateInfo sampler_info {
            .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter               = VK_FILTER_NEAREST,
            .minFilter               = VK_FILTER_NEAREST,
            .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod                  = 0.0f,
            .unnormalizedCoordinates = VK_FALSE,
        };
        if (device.handle().CreateSampler(sampler_info, m_point_sampler) != VK_SUCCESS) {
            LOG_ERROR("VolumetricsSingleFill: point sampler failed");
            return false;
        }
    }
    return true;
}

bool VolumetricsSingleFillPass::ensureFramebuffer(const Device& device) {
    const VkExtent2D extent { m_desc.vk_dst.extent.width, m_desc.vk_dst.extent.height };
    if (m_fb && m_fb_extent.width == extent.width && m_fb_extent.height == extent.height) {
        return true;
    }
    m_fb.reset();
    if (! m_pipeline.pass || m_desc.vk_dst.view == VK_NULL_HANDLE) return false;
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass      = *m_pipeline.pass,
        .attachmentCount = 1,
        .pAttachments    = &m_desc.vk_dst.view,
        .width           = extent.width,
        .height          = extent.height,
        .layers          = 1,
    };
    if (device.handle().CreateFramebuffer(info, m_fb) != VK_SUCCESS) return false;
    m_fb_extent = extent;
    return true;
}

void VolumetricsSingleFillPass::prepare(Scene& scene, const Device& device,
                                        RenderingResources& rr) {
    if (scene.renderTargets.count(m_desc.dst) == 0) {
        LOG_ERROR("VolumetricsSingleFill: missing dest '%s'", m_desc.dst.c_str());
        return;
    }
    const auto& rt = scene.renderTargets.at(m_desc.dst);
    auto        opt = device.tex_cache().Query(m_desc.dst, ToTexKey(rt), ! rt.allowReuse);
    if (! opt.has_value()) {
        LOG_ERROR("VolumetricsSingleFill: query dest failed");
        return;
    }
    m_desc.vk_dst = opt.value();
    if (! ensurePipeline(device, rr)) return;
    if (! ensureFramebuffer(device)) return;
    setPrepared();
}

void VolumetricsSingleFillPass::refreshResources(Scene& scene, const Device& device,
                                                 RenderingResources&) {
    if (scene.renderTargets.count(m_desc.dst) == 0) {
        setPrepared(false);
        return;
    }
    const auto& rt = scene.renderTargets.at(m_desc.dst);
    auto        opt = device.tex_cache().Query(m_desc.dst, ToTexKey(rt), ! rt.allowReuse);
    if (! opt.has_value()) {
        setPrepared(false);
        return;
    }
    m_desc.vk_dst = opt.value();
    m_fb.reset();
    m_fb_extent = {};
}

void VolumetricsSingleFillPass::clearFar(RenderingResources& rr) const {
    auto& img = m_desc.vk_dst;
    if (! img.handle) return;
    VkImageSubresourceRange range {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    VkImageMemoryBarrier to_transfer {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext            = nullptr,
        .srcAccessMask    = 0,
        .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image            = img.handle,
        .subresourceRange = range,
    };
    rr.command.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_DEPENDENCY_BY_REGION_BIT,
                               to_transfer);
    // Reversed scene depth: the far plane is 0.
    const VkClearColorValue far_depth { 0.0f, 0.0f, 0.0f, 1.0f };
    rr.command.ClearColorImage(img.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &far_depth, range);
    VkImageMemoryBarrier to_shader {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext            = nullptr,
        .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image            = img.handle,
        .subresourceRange = range,
    };
    rr.command.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                               VK_DEPENDENCY_BY_REGION_BIT,
                               to_shader);
}

void VolumetricsSingleFillPass::execute(const Device& device, RenderingResources& rr) {
    if (! m_desc.vk_dst.handle) return;

    // Model chunk draws only mark the shared multisampled depth dirty; materialize the
    // single-sample copy sampled below.
    ResolveModelDepthIfNeeded(rr, m_desc.scene_output);

    VmaImageParameters* depth_image = nullptr;
    if (auto resolved = rr.model_depth_resolved.find(m_desc.scene_output);
        resolved != rr.model_depth_resolved.end() && resolved->second.handle &&
        resolved->second.view) {
        depth_image = &resolved->second;
    } else if (auto raw = rr.model_depth_images.find(m_desc.scene_output);
               raw != rr.model_depth_images.end() && raw->second.handle && raw->second.view &&
               raw->second.samples <= 1) {
        depth_image = &raw->second;
    }
    if (depth_image == nullptr) {
        // No model wrote scene depth this frame: expose an all-far (0) limit depth.
        clearFar(rr);
        return;
    }
    if (! ensurePipeline(device, rr) || ! ensureFramebuffer(device) || ! m_fb) {
        clearFar(rr);
        return;
    }

    auto& depth = *depth_image;
    bool  from_resolved = false;
    if (auto it = rr.model_depth_resolved.find(m_desc.scene_output);
        it != rr.model_depth_resolved.end() && &it->second == depth_image) {
        from_resolved = true;
    }
    VkImageSubresourceRange depth_range {
        .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    VkImageMemoryBarrier depth_to_sample {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext            = nullptr,
        .srcAccessMask    = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image            = *depth.handle,
        .subresourceRange = depth_range,
    };
    if (! from_resolved) {
        rr.command.PipelineBarrier(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   VK_DEPENDENCY_BY_REGION_BIT,
                                   depth_to_sample);
    }

    const VkExtent2D extent { m_desc.vk_dst.extent.width, m_desc.vk_dst.extent.height };
    VkClearValue     unused {};
    VkRenderPassBeginInfo begin {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass      = *m_pipeline.pass,
        .framebuffer     = *m_fb,
        .renderArea      = VkRect2D { { 0, 0 }, extent },
        .clearValueCount = 1,
        .pClearValues    = &unused,
    };
    rr.command.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
    rr.command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_pipeline.handle);

    VkViewport viewport {
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(extent.width),
        .height   = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D scissor { { 0, 0 }, extent };
    rr.command.SetViewport(0, viewport);
    rr.command.SetScissor(0, scissor);

    VkDescriptorImageInfo image_info {
        *m_point_sampler,
        *depth.view,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkWriteDescriptorSet write {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding      = 0,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &image_info,
    };
    rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_pipeline.layout, 0, write);

    VkBuffer     gpu = rr.vertex_buf->gpuBuf();
    VkDeviceSize off = m_vertex_buf.offset;
    rr.command.BindVertexBuffers(0, 1, &gpu, &off);
    rr.command.Draw(4, 1, 0, 0);
    rr.command.EndRenderPass();

    VkImageMemoryBarrier depth_restore {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext            = nullptr,
        .srcAccessMask    = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask    = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .image            = *depth.handle,
        .subresourceRange = depth_range,
    };
    if (! from_resolved) {
        rr.command.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                   VK_DEPENDENCY_BY_REGION_BIT,
                                   depth_restore);
    }
}

void VolumetricsSingleFillPass::destory(const Device&, RenderingResources& rr) {
    setPrepared(false);
    clearReleaseTexs();
    m_fb.reset();
    m_fb_extent = {};
    if (m_vertex_buf) rr.vertex_buf->unallocateSubRef(m_vertex_buf);
}

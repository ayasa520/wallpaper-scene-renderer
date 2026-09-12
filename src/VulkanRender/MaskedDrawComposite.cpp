#include "MaskedDrawComposite.hpp"

#include "Core/ArrayHelper.hpp"
#include "ShaderDrawCore.hpp"
#include "Vulkan/Shader.hpp"

#include <array>

namespace wallpaper::vulkan
{
namespace
{

// A vertex-free full-target triangle leaves the puppet's bound vertex/index streams
// untouched. Both coverage attachments use the same extent, so fragment coordinates
// address the exact corresponding texel without introducing another filtering step.
constexpr std::string_view vertex_source = R"(
float4 main_vs(uint vertex : SV_VertexID) : SV_Position {
    float2 corner = float2((vertex << 1) & 2, vertex & 2);
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}
)";

constexpr std::string_view fragment_source = R"(
[[vk::binding(0, 0)]] Texture2D<float4> u_Coverage;

float4 main_ps(float4 position : SV_Position) : SV_Target0 {
    return u_Coverage.Load(int3(int2(position.xy), 0));
}
)";

} // namespace

bool MaskedDrawComposite::prepare(const Device& device, RenderingResources& resources) {
    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage = ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "MaskedDrawComposite.vert",
            .entry_point = "main_vs",
            .src = std::string(vertex_source),
        },
        ShaderCompUnit {
            .stage = ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "MaskedDrawComposite.frag",
            .entry_point = "main_ps",
            .src = std::string(fragment_source),
        },
    };
    ShaderCompOpt options;
    options.target_env = ShaderTargetEnv::VULKAN_1_1;
    std::vector<Uni_ShaderSpv> stages;
    if (!CompileAndLinkShaderUnits(units, options, stages)) return false;

    DescriptorSetInfo descriptors;
    descriptors.push_descriptor = true;
    descriptors.bindings.push_back({
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    });

    // Each source has already undergone its own alpha blending into an R8 image.
    // Multiplication must read that completed coverage and the retained destination
    // red, then store/quantize once. Reusing the source writer's SRC_ALPHA state here
    // would apply alpha a second time and change nested translucent masks.
    VkPipelineColorBlendAttachmentState blend {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT,
    };
    auto pass = CreateShaderDrawRenderPass(device.handle(), VK_FORMAT_R8_UNORM,
        VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!pass) return false;

    GraphicsPipeline pipeline;
    pipeline.toDefault();
    pipeline.addDescriptorSetInfo(spanone {descriptors})
        .setColorBlendStates(spanone {blend})
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    for (auto& stage : stages) pipeline.addStage(std::move(stage));
    m_pipeline.debug_name = "MaskedDrawComposite";
    m_pipeline.cache_key = "MaskedDrawComposite|format=r8|load=load|samples=1";
    return pipeline.create(device, *pass, m_pipeline, resources.pipeline_cache.get());
}

void MaskedDrawComposite::record(RenderingResources& resources, VkFramebuffer framebuffer,
                                 VkExtent3D extent, const ImageParameters& source) const {
    auto& command = resources.command;
    VkRenderPassBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = *m_pipeline.pass,
        .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {extent.width, extent.height}},
    };
    command.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
    command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_pipeline.handle);
    VkDescriptorImageInfo image {
        .imageView = source.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkWriteDescriptorSet write {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = &image,
    };
    command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_pipeline.layout, 0, write);
    command.Draw(3, 1, 0, 0);
    command.EndRenderPass();
}

void MaskedDrawComposite::destroy() { m_pipeline.resetCachedState(); }

} // namespace wallpaper::vulkan

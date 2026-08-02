#include "MaskedDrawShaderContract.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace wallpaper::vulkan
{

std::optional<std::vector<ShaderCode>> CompileMaskedDrawMaskShaderCodes(uint32_t bone_count) {
    static std::mutex mutex;
    static std::unordered_map<uint32_t, std::vector<ShaderCode>> cache;

    std::lock_guard lock(mutex);
    if (const auto it = cache.find(bone_count); it != cache.end()) return it->second;

    // This shader contract is isolated from pass orchestration because its bindings, explicit input
    // locations, and stencil coverage threshold must evolve together. The visible material and mask
    // writer share only transform and skinning uniforms; material tint, lighting, and blend state do
    // not participate in coverage generation.
    const std::string vertex_source = R"(
[[vk::binding(0, 0)]] cbuffer MaskedDrawUniformBlock {
    column_major float4x4 g_ModelViewProjectionMatrix;
    column_major float3x4 g_Bones[)" + std::to_string(bone_count) + R"(];
};

struct VSInput {
    [[vk::location(0)]] float3 a_Position : A_POSITION;
    [[vk::location(1)]] uint4 a_BlendIndices : A_BLENDINDICES;
    [[vk::location(2)]] float4 a_BlendWeights : A_BLENDWEIGHTS;
    [[vk::location(3)]] float2 a_TexCoord : A_TEXCOORD;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

VSOutput main_vs(VSInput input) {
    const float4 position = float4(input.a_Position, 1.0);
    const float3 skinned =
        mul(g_Bones[input.a_BlendIndices.x], position) * input.a_BlendWeights.x +
        mul(g_Bones[input.a_BlendIndices.y], position) * input.a_BlendWeights.y +
        mul(g_Bones[input.a_BlendIndices.z], position) * input.a_BlendWeights.z +
        mul(g_Bones[input.a_BlendIndices.w], position) * input.a_BlendWeights.w;

    VSOutput output;
    output.position = mul(g_ModelViewProjectionMatrix, float4(skinned, 1.0));
    output.texcoord = input.a_TexCoord;
    return output;
}
)";

    static const std::string fragment_source = R"(
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float4> g_Texture0;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState g_Texture0_ww_sampler;

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

float4 main_ps(PSInput input) : SV_Target0 {
    const float coverage = g_Texture0.Sample(g_Texture0_ww_sampler, input.texcoord).r;
    clip(coverage - (0.5 / 255.0));
    return 0.0;
}
)";

    ShaderCompOpt options {};
    options.target_env         = ShaderTargetEnv::VULKAN_1_0;
    options.auto_map_locations = false;
    options.auto_map_bindings  = false;

    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage           = ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name      = "MaskedDrawMask.vert",
            .entry_point     = "main_vs",
            .src             = vertex_source,
        },
        ShaderCompUnit {
            .stage           = ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name      = "MaskedDrawMask.frag",
            .entry_point     = "main_ps",
            .src             = fragment_source,
        },
    };

    std::vector<Uni_ShaderSpv> compiled;
    if (! CompileAndLinkShaderUnits(units, options, compiled)) return std::nullopt;

    std::vector<ShaderCode> codes;
    codes.reserve(compiled.size());
    for (auto& stage : compiled) codes.push_back(std::move(stage->spirv));
    cache.emplace(bone_count, codes);
    return codes;
}

} // namespace wallpaper::vulkan

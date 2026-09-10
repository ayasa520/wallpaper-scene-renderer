#include "TextPass.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneTextPrimitive.h"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include "Vulkan/ShaderComp.hpp"
#include "PassCommon.hpp"
#include "Msaa.hpp"
#include "Resource.hpp"
#include "RenderTargetOps.hpp"
#include "ShaderDrawCore.hpp"
#include "WPSceneScriptMedia.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

using namespace wallpaper::vulkan;

namespace
{
constexpr std::string_view kTextBackgroundTextureKey { "__text_layer_background_white" };

std::string TextPipelineCompatibilityKey(bool clear_before_draw,
                                         wallpaper::BlendMode blend_mode,
                                         wallpaper::AlphaWritePolicy alpha_write_policy,
                                         VkSampleCountFlagBits sample_count,
                                         bool resolve_msaa,
                                         bool shared_depth) {
    // Text PSOs are shared by render-pass compatibility plus the full GraphicsPipeline descriptor,
    // not by the layer that first requested them. This keeps visibility toggles on the same model
    // as engine-level PSO caches while still letting hidden text release atlas/framebuffer memory.
    return "TextPass|format=rgba8|final=shader-read|load=" +
           std::to_string(static_cast<int>(clear_before_draw ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                            : VK_ATTACHMENT_LOAD_OP_LOAD)) +
           "|blend=" + std::to_string(static_cast<int>(blend_mode)) +
           "|alpha-policy=" + std::to_string(static_cast<int>(alpha_write_policy)) +
           "|samples=" + std::to_string(static_cast<int>(sample_count)) +
           "|resolve=" + (resolve_msaa ? std::string("1") : std::string("0")) +
           "|depth-format=" + std::to_string(static_cast<int>(
               shared_depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED));
}

int IntendedTextSampleCount(const wallpaper::Scene* scene, std::string_view output) {
    // Compose text writes `_rt_FullFrameBufferMultiSampled`. Effect ping-pong /
    // text-bridge targets stay 1x: effect shaders sample them as Texture2D, and
    // there is no multisampled ping-pong render-target name.
    if (scene != nullptr && wallpaper::vulkan::ComposeOutputUsesMsaa(*scene, output)) {
        return std::max(1, scene->MsaaSampleCount());
    }
    return 1;
}

struct TextPassUniforms {
    float model_view_projection[16] {};
    float color[4] {};
};

struct PreparedTextShaders {
    std::vector<Uni_ShaderSpv> stages;
};

struct TextVertexInputLayout {
    VkVertexInputBindingDescription binding {};
    std::array<VkVertexInputAttributeDescription, 2> attributes {};
};

std::optional<TextVertexInputLayout> ResolveTextVertexInputLayout(
    const wallpaper::SceneTextPrimitive& primitive) {
    const wallpaper::SceneMesh* source_mesh { nullptr };
    for (const auto& page : primitive.glyph_pages) {
        if (page.mesh != nullptr && page.mesh->VertexCount() > 0) {
            source_mesh = page.mesh.get();
            break;
        }
    }
    if (source_mesh == nullptr && primitive.background_mesh != nullptr &&
        primitive.background_mesh->VertexCount() > 0) {
        source_mesh = primitive.background_mesh.get();
    }
    if (source_mesh == nullptr || source_mesh->VertexCount() == 0) return std::nullopt;

    const auto& vertex = source_mesh->GetVertexArray(0);
    const auto  attrs = vertex.GetAttrOffsetMap();
    const auto  position_it = attrs.find(std::string(wallpaper::WE_IN_POSITION));
    const auto  texcoord_it = attrs.find(std::string(wallpaper::WE_IN_TEXCOORD));
    if (position_it == attrs.end() || texcoord_it == attrs.end()) {
        LOG_ERROR("TextPass: generated text mesh is missing required position/texcoord attributes");
        return std::nullopt;
    }

    TextVertexInputLayout layout;
    // SceneVertexArray pads each attribute to Wallpaper Engine's vec4-style storage contract.
    // TextPass used to hardcode FLOAT3+FLOAT2 as a tightly packed 5-float vertex, but the actual
    // generated buffer is 8 floats per vertex. Reading the live SceneVertexArray stride/offsets
    // here keeps the dedicated text primitive on the same canonical mesh layout as generic image
    // passes and prevents every vertex after the first one from being fetched at the wrong byte.
    layout.binding = VkVertexInputBindingDescription {
        .binding = 0,
        .stride = static_cast<uint32_t>(vertex.OneSizeOf()),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    layout.attributes = {
        VkVertexInputAttributeDescription {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = static_cast<uint32_t>(position_it->second.offset),
        },
        VkVertexInputAttributeDescription {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = static_cast<uint32_t>(texcoord_it->second.offset),
        },
    };
    return layout;
}

std::optional<PreparedTextShaders> CompileTextShaders() {
    static const char* kVertexSource = R"(
[[vk::binding(0, 0)]] cbuffer TextUniformBlock {
    column_major float4x4 g_ModelViewProjectionMatrix;
    float4 g_Color4;
};

struct VSInput {
    [[vk::location(0)]] float3 a_Position : A_POSITION;
    [[vk::location(1)]] float2 a_TexCoord : A_TEXCOORD;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 v_TexCoord : TEXCOORD0;
    [[vk::location(1)]] float4 v_Color : COLOR0;
};

VSOutput main_vs(VSInput input) {
    VSOutput output;
    output.position = mul(g_ModelViewProjectionMatrix, float4(input.a_Position, 1.0));
    output.v_TexCoord = input.a_TexCoord;
    output.v_Color = g_Color4;
    return output;
}
)";

    static const char* kFragmentSource = R"(
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float4> g_Texture0;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState g_Texture0_ww_sampler;

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 v_TexCoord : TEXCOORD0;
    [[vk::location(1)]] float4 v_Color : COLOR0;
};

float4 main_ps(PSInput input) : SV_Target0 {
    // Glyph atlas pages use R8 coverage. The shared background texture is white in every channel,
    // so reading red keeps glyph and background draws on the same dedicated text shader.
    const float coverage = g_Texture0.Sample(g_Texture0_ww_sampler, input.v_TexCoord).r;
    return float4(input.v_Color.rgb, input.v_Color.a * coverage);
}
)";

    ShaderCompOpt options {};
    options.target_env = ShaderTargetEnv::VULKAN_1_0;
    options.auto_map_locations = false;
    options.auto_map_bindings = false;

    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage = wallpaper::ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "TextPass.vert",
            .entry_point = "main_vs",
            .src = kVertexSource,
        },
        ShaderCompUnit {
            .stage = wallpaper::ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "TextPass.frag",
            .entry_point = "main_ps",
            .src = kFragmentSource,
        },
    };

    PreparedTextShaders prepared;
    if (!CompileAndLinkShaderUnits(units, options, prepared.stages)) {
        LOG_ERROR("TextPass: failed to compile dedicated text shaders");
        return std::nullopt;
    }
    return prepared;
}

bool BindTextPassOutput(wallpaper::Scene& scene, const Device& device, TextPass::Desc& desc) {
    desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
    desc.resolve_msaa = false;
    desc.vk_resolve   = {};

    auto output_it = scene.renderTargets.find(desc.output);
    if (output_it == scene.renderTargets.end()) return false;
    auto& rt = output_it->second;

    // Direct clock/text that composites to `_rt_default` writes the same MS color
    // target as CustomShaderPass. A 1x write into the resolved image is overwritten
    // by the later compose resolve of `_rt_FullFrameBufferMultiSampled`.
    if (ComposeOutputUsesMsaa(scene, desc.output)) {
        const auto ms_name = std::string(wallpaper::SpecTex_DefaultMS);
        const auto ms_it   = scene.renderTargets.find(ms_name);
        if (ms_it != scene.renderTargets.end()) {
            if (auto ms_opt = device.tex_cache().Query(
                    ms_name, ToTexKey(ms_it->second), ! ms_it->second.allowReuse);
                ms_opt.has_value()) {
                desc.vk_output    = ms_opt.value();
                desc.sample_count = static_cast<VkSampleCountFlagBits>(
                    std::max(1u, ms_it->second.sample_count > 0
                                     ? static_cast<uint>(ms_it->second.sample_count)
                                     : 1u));
                desc.resolve_msaa = false;
                return true;
            }
        }
        LOG_ERROR("TextPass: MSAA compose target missing node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  desc.output.c_str());
    }

    if (auto opt = device.tex_cache().Query(desc.output, ToTexKey(rt), ! rt.allowReuse);
        opt.has_value()) {
        desc.vk_output = opt.value();
        return true;
    }
    return false;
}

void WriteMatrixToUniform(TextPassUniforms& uniforms, const Eigen::Matrix4f& matrix) {
    for (int column = 0; column < 4; column++) {
        for (int row = 0; row < 4; row++) {
            uniforms.model_view_projection[column * 4 + row] = matrix(row, column);
        }
    }
}

std::shared_ptr<wallpaper::Image> ResolveTextBackgroundImage() {
    // The direct text pipeline only needs one non-glyph texture: a 1x1 white coverage image for
    // the optional opaque background quad. Materializing it here keeps the text pass self-owned
    // and avoids routing primitive text rendering through unrelated image-parser infrastructure.
    static const std::shared_ptr<wallpaper::Image> image =
        wallpaper::CreateSceneScriptSolidImage(kTextBackgroundTextureKey, { 255, 255, 255, 255 });
    return image;
}

bool LoadTextPassTexture(const Device&                        device,
                         const std::shared_ptr<wallpaper::Image>& image,
                         ImageSlotsRef*                       out_slots) {
    if (out_slots == nullptr) return false;
    if (image == nullptr) {
        *out_slots = {};
        return true;
    }

    *out_slots = device.tex_cache().CreateTex(*image);
    return !out_slots->slots.empty();
}

bool CreateTextPipelinesForPrimitive(const Device&                         device,
                                    RenderingResources&                   rr,
                                    const wallpaper::SceneTextPrimitive&  primitive,
                                    bool                                  offscreen_output,
                                    bool                                  clear_before_draw,
                                    wallpaper::AlphaWritePolicy           alpha_write_policy,
                                    VkSampleCountFlagBits                 sample_count,
                                    bool                                  resolve_msaa,
                                    bool                                  shared_depth,
                                    bool                                  glyph_depth_test,
                                    std::string                           debug_name,
                                    PipelineParameters&                   glyph_pipeline,
                                    PipelineParameters&                   background_pipeline) {
    const auto compiled_shaders = CompileTextShaders();
    if (!compiled_shaders.has_value()) return false;

    DescriptorSetInfo descriptor_info;
    descriptor_info.push_descriptor = true;
    descriptor_info.bindings = {
        VkDescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        VkDescriptorSetLayoutBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const auto vertex_layout = ResolveTextVertexInputLayout(primitive);
    if (!vertex_layout.has_value()) return false;

    // colorBlendMode 31 is Wallpaper Engine's fixed-function additive case. Shader blend modes
    // 1..30 are handled later by the independent final passthrough, so their text source remains an
    // ordinary translucent offscreen raster.
    const auto blend_mode =
        !offscreen_output && primitive.object.colorBlendMode == 31
            ? wallpaper::BlendMode::Additive
            : wallpaper::BlendMode::Translucent;
    VkPipelineColorBlendAttachmentState blend_state {};
    blend_state.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    if (offscreen_output) blend_state.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
    SetBlend(blend_mode, blend_state);
    if (offscreen_output && alpha_write_policy != wallpaper::AlphaWritePolicy::Preserve) {
        // Composition attachments keep their authored RGB blend but use an explicit coverage
        // equation. copybackground=false selects Alpha-MAX so later transparent draws cannot erase
        // coverage accumulated by earlier routed children.
        blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        if (alpha_write_policy == wallpaper::AlphaWritePolicy::Max) {
            blend_state.alphaBlendOp = VK_BLEND_OP_MAX;
            blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        } else {
            blend_state.alphaBlendOp = VK_BLEND_OP_ADD;
            blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        }
    }

    GraphicsPipeline pipeline;
    pipeline.toDefault();
    pipeline.multisample.rasterizationSamples =
        sample_count > VK_SAMPLE_COUNT_1_BIT ? sample_count : VK_SAMPLE_COUNT_1_BIT;
    // Both ordinary translucent and additive glyph coverage are read-only depth consumers.
    // Enabling owner testing must not turn transparent atlas texels into depth writers.
    pipeline.depth.depthWriteEnable = false;
    pipeline.depth.depthCompareOp = VK_COMPARE_OP_GREATER;
    pipeline.addDescriptorSetInfo(std::span<const DescriptorSetInfo>(&descriptor_info, 1))
        .setColorBlendStates(std::span<const VkPipelineColorBlendAttachmentState>(&blend_state, 1))
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addInputBindingDescription(
            std::span<const VkVertexInputBindingDescription>(&vertex_layout->binding, 1))
        .addInputAttributeDescription(vertex_layout->attributes);
    for (const auto& stage : compiled_shaders->stages) {
        if (!stage) continue;
        pipeline.addStage(Uni_ShaderSpv(new ShaderSpv(*stage)));
    }

    const auto attachment = shared_depth ? SceneDepthAttachmentDescription(false)
                                         : ShaderDrawAttachmentDescription {};
    auto create_pipeline = [&](bool depth_test, const std::string& name,
                               PipelineParameters& parameters) {
        auto render_pass = CreateShaderDrawRenderPass(
            device.handle(), VK_FORMAT_R8G8B8A8_UNORM,
            clear_before_draw ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, attachment, sample_count, resolve_msaa);
        if (!render_pass.has_value()) return false;
        pipeline.depth.depthTestEnable = depth_test;
        parameters.debug_name = name;
        parameters.cache_key = TextPipelineCompatibilityKey(
            clear_before_draw, blend_mode, alpha_write_policy, sample_count, resolve_msaa,
            shared_depth);
        return pipeline.create(device, *render_pass, parameters, rr.pipeline_cache.get());
    };
    if (!create_pipeline(glyph_depth_test, debug_name, glyph_pipeline)) return false;
    // The opaque background has its own raster-state lifetime. Keep its existing selection
    // independent from live glyph testing, while using a render-pass-compatible pipeline so
    // it can draw into the same framebuffer without clearing or replacing shared scene depth.
    return !glyph_depth_test || create_pipeline(false, debug_name + " background", background_pipeline);
}

std::array<float, 4> ResolveTextColor(const wallpaper::SceneTextPrimitive& primitive,
                                      bool                                 background) {
    if (background) {
        return {
            primitive.object.backgroundcolor[0] * primitive.object.backgroundbrightness,
            primitive.object.backgroundcolor[1] * primitive.object.backgroundbrightness,
            primitive.object.backgroundcolor[2] * primitive.object.backgroundbrightness,
            primitive.object.alpha,
        };
    }
    return primitive.ForegroundColor();
}
} // namespace

TextPass::TextPass(const Desc& desc)
    : m_node_identity(desc.node != nullptr ? desc.node->RenderIdentity() : 0) {
    // The pass description intentionally stores only the authored/runtime identity fields here.
    // Vulkan handles such as framebuffers and pipeline objects are non-copyable and must always be
    // created during `prepare()` against the live device, so the constructor avoids copying any of
    // the prepared-state members from the temporary render-graph description.
    m_desc.scene               = desc.scene;
    m_desc.node                = desc.node;
    m_desc.layer_id            = desc.layer_id;
    m_desc.execute_when_hidden = desc.execute_when_hidden;
    m_desc.should_execute      = desc.should_execute;
    m_desc.clear_before_draw   = desc.clear_before_draw;
    m_desc.output              = desc.output;
    m_desc.alpha_write_policy  = desc.alpha_write_policy;
    m_desc.shared_depth        = desc.shared_depth;
    m_desc.glyph_depth_test    = desc.glyph_depth_test;
    m_desc.camera_override     = desc.camera_override;
    m_desc.use_active_camera_for_parallax = desc.use_active_camera_for_parallax;
    m_desc.model_space         = desc.model_space;
    m_desc.reflection_pass     = desc.reflection_pass;
    m_desc.reflection_raster   = desc.reflection_raster;
    m_desc.reflection_snapshot = desc.reflection_snapshot;
    m_desc.effect_snapshot_camera = desc.effect_snapshot_camera;
}
TextPass::~TextPass() = default;

std::string TextPass::residencyKey() const {
    return "TextPass|node=" + std::to_string(m_node_identity) +
           "|layer=" + std::to_string(m_desc.layer_id) + "|output=" + m_desc.output +
           "|reflection=" + (m_desc.reflection_pass ? "1" : "0");
}

bool TextPass::canReuseForResidency(const VulkanPass& next_pass) const {
    const auto* next = dynamic_cast<const TextPass*>(&next_pass);
    if (next == nullptr) return false;
    // The text pipeline depends on the text primitive's vertex layout and output target, both
    // represented by the stable node/layer/output residency key. Visibility gates and the live
    // scene pointer are safe to absorb without recreating shader modules or descriptor layouts.
    const int this_samples = static_cast<int>(m_desc.sample_count);
    const int next_samples = IntendedTextSampleCount(next->m_desc.scene, next->m_desc.output);
    return residencyKey() == next->residencyKey() &&
           m_desc.execute_when_hidden == next->m_desc.execute_when_hidden &&
           m_desc.clear_before_draw == next->m_desc.clear_before_draw &&
           m_desc.alpha_write_policy == next->m_desc.alpha_write_policy &&
           m_desc.shared_depth == next->m_desc.shared_depth &&
           m_desc.glyph_depth_test == next->m_desc.glyph_depth_test &&
           m_desc.camera_override == next->m_desc.camera_override &&
           m_desc.use_active_camera_for_parallax == next->m_desc.use_active_camera_for_parallax &&
           m_desc.model_space == next->m_desc.model_space &&
           m_desc.reflection_raster == next->m_desc.reflection_raster &&
           m_desc.reflection_snapshot == next->m_desc.reflection_snapshot &&
           m_desc.effect_snapshot_camera == next->m_desc.effect_snapshot_camera &&
           this_samples == next_samples &&
           ! m_desc.resolve_msaa;
}

void TextPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto* next = dynamic_cast<const TextPass*>(&next_pass);
    if (next == nullptr) return;
    m_desc.scene               = next->m_desc.scene;
    m_desc.node                = next->m_desc.node;
    m_node_identity            = next->m_node_identity;
    m_desc.layer_id            = next->m_desc.layer_id;
    m_desc.execute_when_hidden = next->m_desc.execute_when_hidden;
    m_desc.should_execute      = next->m_desc.should_execute;
    m_desc.clear_before_draw   = next->m_desc.clear_before_draw;
    m_desc.output              = next->m_desc.output;
    m_desc.alpha_write_policy  = next->m_desc.alpha_write_policy;
    m_desc.shared_depth        = next->m_desc.shared_depth;
    m_desc.glyph_depth_test    = next->m_desc.glyph_depth_test;
    m_desc.camera_override     = next->m_desc.camera_override;
    m_desc.use_active_camera_for_parallax = next->m_desc.use_active_camera_for_parallax;
    m_desc.model_space         = next->m_desc.model_space;
    m_desc.reflection_pass     = next->m_desc.reflection_pass;
    m_desc.reflection_raster   = next->m_desc.reflection_raster;
    m_desc.reflection_snapshot = next->m_desc.reflection_snapshot;
    m_desc.effect_snapshot_camera = next->m_desc.effect_snapshot_camera;
}

bool TextPass::referencesRenderTarget(std::string_view render_target) const {
    // A text pass only owns its compose/bridge output. Glyph atlas pages are imported texture-cache
    // entries, not render-graph targets. Compose text also writes `_rt_FullFrameBufferMultiSampled`
    // when MSAA is on, so that RT must refresh this pass.
    return m_desc.output == render_target ||
           (m_desc.output == wallpaper::SpecTex_Default &&
            render_target == wallpaper::SpecTex_DefaultMS);
}

bool TextPass::referencesTextLayer(int32_t layer_id) const {
    // Runtime text rerasters are scoped by authored layer id. Matching that id here lets a direct
    // Clock-style text pass refresh its atlas and mesh before command recording without touching
    // unrelated text layers that happen to draw to the same final render target.
    return layer_id != 0 && m_desc.layer_id == layer_id;
}

bool TextPass::refreshTextures(const Device& device) {
    const auto* primitive =
        m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) return false;

    if (!LoadTextPassTexture(device, ResolveTextBackgroundImage(), &m_desc.background_texture)) {
        return false;
    }

    m_desc.page_textures.resize(primitive->glyph_pages.size());
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        if (!LoadTextPassTexture(device,
                                 primitive->layout.glyph_pages[page_index].image,
                                 &m_desc.page_textures[page_index])) {
            return false;
        }
    }
    m_loaded_atlas_version = primitive->atlas_version;
    return true;
}

bool TextPass::recreateFramebuffer(const Device& device, RenderingResources& rr) {
    m_desc.framebuffer.reset();
    if (!m_desc.pipeline.pass || m_desc.vk_output.view == VK_NULL_HANDLE ||
        m_desc.vk_output.extent.width == 0 || m_desc.vk_output.extent.height == 0) {
        LOG_ERROR("TextPassRefresh: cannot recreate framebuffer node='%s' output='%s' "
                  "hasRenderPass=%s hasView=%s extent=[%u,%u]",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str(),
                  m_desc.pipeline.pass ? "true" : "false",
                  m_desc.vk_output.view != VK_NULL_HANDLE ? "true" : "false",
                  m_desc.vk_output.extent.width,
                  m_desc.vk_output.extent.height);
        return false;
    }
    if (m_desc.resolve_msaa && m_desc.vk_resolve.view == VK_NULL_HANDLE) {
        LOG_ERROR("TextPassRefresh: missing MSAA resolve view node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        return false;
    }
    m_desc.depth_image = m_desc.shared_depth
        ? AcquireSceneDepthImage(device, rr, m_desc.output, m_desc.vk_output.extent, m_desc.sample_count)
        : nullptr;
    if (m_desc.shared_depth && m_desc.depth_image == nullptr) return false;
    std::array<VkImageView, 3> attachments { m_desc.vk_output.view, VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t attachment_count = 1;
    if (m_desc.shared_depth) attachments[attachment_count++] = *m_desc.depth_image->view;
    if (m_desc.resolve_msaa) attachments[attachment_count++] = m_desc.vk_resolve.view;
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass      = *m_desc.pipeline.pass,
        .attachmentCount = attachment_count,
        .pAttachments    = attachments.data(),
        .width           = m_desc.vk_output.extent.width,
        .height          = m_desc.vk_output.extent.height,
        .layers          = 1,
    };
    const bool created = device.handle().CreateFramebuffer(info, m_desc.framebuffer) == VK_SUCCESS;
    if (created && std::getenv("WESCENE_TRACE_DEPTH_ATTACHMENTS") != nullptr) {
        LOG_INFO("SceneTextDepthFramebuffer: layer=%d output='%s' framebuffer=%p "
                 "color-view=%p depth-view=%p shared-depth=%s glyph-test=%s extent=%ux%u samples=%u",
                 m_desc.layer_id, m_desc.output.c_str(), reinterpret_cast<void*>(*m_desc.framebuffer),
                 reinterpret_cast<void*>(attachments[0]),
                 m_desc.shared_depth ? reinterpret_cast<void*>(attachments[1]) : nullptr,
                 m_desc.shared_depth ? "true" : "false", m_desc.glyph_depth_test ? "true" : "false",
                 info.width, info.height, static_cast<unsigned>(m_desc.sample_count));
    }
    return created;
}

bool TextPass::ensureMeshBuffers(SceneMesh& mesh, MeshBuffers& buffers, RenderingResources& rr) {
    auto* dyn_buf = rr.dyn_buf;
    if (dyn_buf == nullptr) return false;

    while (buffers.vertex_bufs.size() > mesh.VertexCount()) {
        dyn_buf->unallocateSubRef(buffers.vertex_bufs.back());
        buffers.vertex_bufs.pop_back();
    }
    buffers.vertex_bufs.resize(mesh.VertexCount());

    for (usize array_index = 0; array_index < mesh.VertexCount(); array_index++) {
        const auto& vertex = mesh.GetVertexArray(array_index);
        auto&       subref = buffers.vertex_bufs[array_index];
        const auto  required_size =
            static_cast<VkDeviceSize>(std::max<usize>(vertex.CapacitySizeOf(), vertex.OneSizeOf()));
        if (!subref || subref.size < required_size) {
            if (subref) dyn_buf->unallocateSubRef(subref);
            if (!dyn_buf->allocateSubRef(required_size, subref)) return false;
            buffers.force_upload = true;
        }
    }

    if (mesh.IndexCount() > 0) {
        const auto& index = mesh.GetIndexArray(0);
        const auto  required_size =
            static_cast<VkDeviceSize>(std::max<usize>(index.CapacitySizeof(), sizeof(uint16_t) * 6));
        if (!buffers.index_buf || buffers.index_buf.size < required_size) {
            if (buffers.index_buf) dyn_buf->unallocateSubRef(buffers.index_buf);
            if (!dyn_buf->allocateSubRef(required_size, buffers.index_buf)) return false;
            buffers.force_upload = true;
        }
    } else if (buffers.index_buf) {
        dyn_buf->unallocateSubRef(buffers.index_buf);
        buffers.index_buf = {};
    }

    const auto revision = mesh.DataRevision();
    const bool needs_upload = revision != buffers.uploaded_revision || buffers.force_upload;
    if (!needs_upload && mesh.Dirty().load() &&
        std::getenv("WESCENE_TRACE_MESH_UPLOADS") != nullptr) {
        // A rebuilt glyph page can reuse this allocation without increasing its byte capacity.
        // Record rejected CPU updates at that boundary so revision identity errors can be
        // distinguished from missing atlas data or a failed GPU upload.
        LOG_INFO("SceneTextMeshUploadSkipped: layer=%d output='%s' reflection=%s "
                 "revision=%llu cpu-dirty=true cpu-indices=%u uploaded-draw-count=%u",
                 m_desc.layer_id, m_desc.output.c_str(),
                 m_desc.reflection_pass ? "true" : "false",
                 static_cast<unsigned long long>(revision), mesh.LogicalIndexCount(),
                 buffers.draw_count);
    }
    if (!needs_upload) return true;

    for (usize array_index = 0; array_index < mesh.VertexCount(); array_index++) {
        const auto& vertex = mesh.GetVertexArray(array_index);
        auto&       subref = buffers.vertex_bufs[array_index];
        if (!dyn_buf->writeToBuf(
                subref,
                { reinterpret_cast<uint8_t*>(const_cast<float*>(vertex.Data())), vertex.DataSizeOf() })) {
            return false;
        }
    }

    if (mesh.IndexCount() > 0) {
        const auto& index = mesh.GetIndexArray(0);
        buffers.draw_count = static_cast<uint32_t>((index.RenderDataCount() * 2) / 3) * 3;
        if (!dyn_buf->writeToBuf(buffers.index_buf,
                                 { reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(index.Data())),
                                   index.DataSizeOf() })) {
            return false;
        }
    } else {
        buffers.draw_count = mesh.VertexCount() > 0
            ? static_cast<uint32_t>(mesh.GetVertexArray(0).VertexCount())
            : 0;
    }

    mesh.Dirty().store(false);
    buffers.uploaded_revision = revision;
    buffers.force_upload = false;
    if (std::getenv("WESCENE_TRACE_MESH_UPLOADS") != nullptr) {
        LOG_INFO("SceneTextMeshUpload: layer=%d output='%s' reflection=%s revision=%llu",
                 m_desc.layer_id, m_desc.output.c_str(),
                 m_desc.reflection_pass ? "true" : "false",
                 static_cast<unsigned long long>(revision));
    }
    return true;
}

void TextPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    const auto* primitive =
        m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) return;

    if (!refreshTextures(device)) return;

    // Text bridge render targets can resize while the TextPass object is intentionally kept alive.
    // The existing framebuffer references the old TextureCache image view, so it must be released
    // before `Query()` is allowed to replace the backing image for this output.
    m_desc.framebuffer.reset();
    if (!BindTextPassOutput(scene, device, m_desc)) return;

    const bool offscreen_output = m_desc.output != wallpaper::SpecTex_Default;
    // Direct text publishes glyphs into the currently inherited destination, including a parent's
    // composition target. Its offscreen name does not make that shared image a text source to
    // clear. Only BuildOwnerSourcePassOptions' private-source seed owns the clear; source
    // initialization runs before glyph rasterization in that branch. Keep load/clear independent
    // from the alpha-write policy, which still follows whether the destination is a composition
    // attachment. Otherwise even an empty late text layout clears every image child already drawn
    // into the parent's source.
    const auto debug_name =
        "TextPass[node=" + (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
        ",output=" + m_desc.output + "]";
    if (!CreateTextPipelinesForPrimitive(
            device,
            rr,
            *primitive,
            offscreen_output,
            m_desc.clear_before_draw,
            m_desc.alpha_write_policy,
            m_desc.sample_count,
            m_desc.resolve_msaa,
            m_desc.shared_depth,
            m_desc.glyph_depth_test,
            debug_name,
            m_desc.pipeline,
            m_desc.background_pipeline)) {
        return;
    }
    if (!recreateFramebuffer(device, rr)) return;

    rr.dyn_buf->allocateSubRef(sizeof(TextPassUniforms),
                               m_desc.ubo_buf,
                               device.limits().minUniformBufferOffsetAlignment);

    if (primitive->background_mesh != nullptr) {
        m_background_buffers.force_upload = true;
        if (!ensureMeshBuffers(*primitive->background_mesh, m_background_buffers, rr)) return;
    }
    m_page_buffers.resize(primitive->glyph_pages.size());
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        m_page_buffers[page_index].force_upload = true;
        if (!ensureMeshBuffers(*primitive->glyph_pages[page_index].mesh,
                               m_page_buffers[page_index],
                               rr)) {
            return;
        }
    }

    // The dedicated text pass only requests the shared transform uniform contract. All visual text
    // state such as glyph color and background color comes directly from the text primitive, so no
    // generic image-material bootstrap is involved anymore.
    if (scene.shaderValueUpdater != nullptr && m_desc.node != nullptr) {
        scene.shaderValueUpdater->InitUniforms(
            m_desc.node,
            [](std::string_view uniform_name) {
                return uniform_name == wallpaper::G_MVP;
            });
    }

    m_desc.clear_value = VkClearValue {
        .color = {
            offscreen_output ? 0.0f : scene.clearColor[0],
            offscreen_output ? 0.0f : scene.clearColor[1],
            offscreen_output ? 0.0f : scene.clearColor[2],
            offscreen_output ? 0.0f : 1.0f,
        },
    };
    setPrepared();
}

bool TextPass::warmupPipeline(Scene& scene, const Device& device, RenderingResources& rr) {
    const auto* primitive =
        m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) return false;

    const bool offscreen_output = m_desc.output != wallpaper::SpecTex_Default;
    const int intended_samples = IntendedTextSampleCount(&scene, m_desc.output);
    const auto sample_count = static_cast<VkSampleCountFlagBits>(intended_samples);
    const bool resolve_msaa = false;
    const auto debug_name =
        "TextPassWarmup[node=" +
        (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
        ",output=" + m_desc.output + "]";
    return CreateTextPipelinesForPrimitive(device,
                                          rr,
                                          *primitive,
                                          offscreen_output,
                                          m_desc.clear_before_draw,
                                          m_desc.alpha_write_policy,
                                          sample_count,
                                          resolve_msaa,
                                          m_desc.shared_depth,
                                          m_desc.glyph_depth_test,
                                          debug_name,
                                          m_desc.pipeline,
                                          m_desc.background_pipeline);
}

void TextPass::dropOutputFramebuffers() { m_desc.framebuffer.reset(); }

void TextPass::refreshResources(Scene& scene, const Device& device, RenderingResources& rr) {
    const int intended_samples = IntendedTextSampleCount(&scene, m_desc.output);
    if (static_cast<int>(m_desc.sample_count) != intended_samples || m_desc.resolve_msaa) {
        destory(device, rr);
        return;
    }
    if (!refreshTextures(device)) {
        LOG_ERROR("TextPassRefresh: texture refresh failed node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }
    auto* primitive = m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) {
        LOG_ERROR("TextPassRefresh: missing primitive node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }

    // Resource refresh happens before the next draw command records its dynamic-buffer upload.
    // Rebuilding and writing text meshes here keeps resized bridge text from binding freshly
    // allocated subranges that have not been copied to the GPU yet, which was the reason
    // effect-backed Date/Clock/Day could disappear immediately after a layout update.
    if (primitive->background_mesh != nullptr &&
        !ensureMeshBuffers(*primitive->background_mesh, m_background_buffers, rr)) {
        LOG_ERROR("TextPassRefresh: background mesh upload failed node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }
    if (m_page_buffers.size() != primitive->glyph_pages.size()) {
        m_page_buffers.resize(primitive->glyph_pages.size());
        for (auto& buffers : m_page_buffers) buffers.force_upload = true;
    }
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        if (!ensureMeshBuffers(*primitive->glyph_pages[page_index].mesh,
                               m_page_buffers[page_index],
                               rr)) {
            LOG_ERROR("TextPassRefresh: glyph mesh upload failed node='%s' output='%s' page=%zu",
                      m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                      m_desc.output.c_str(),
                      page_index);
            setPrepared(false);
            return;
        }
    }
    const auto previous_output_view = m_desc.vk_output.view;
    const auto previous_output_extent = m_desc.vk_output.extent;
    const auto bind_name = ComposeOutputUsesMsaa(scene, m_desc.output)
                               ? std::string(wallpaper::SpecTex_DefaultMS)
                               : m_desc.output;
    if (const auto bind_it = scene.renderTargets.find(bind_name);
        bind_it != scene.renderTargets.end() &&
        (previous_output_extent.width != static_cast<uint32_t>(bind_it->second.width) ||
         previous_output_extent.height != static_cast<uint32_t>(bind_it->second.height))) {
        m_desc.framebuffer.reset();
    }
    if (!BindTextPassOutput(scene, device, m_desc)) {
        setPrepared(false);
        return;
    }
    const bool output_extent_changed =
        previous_output_extent.width != m_desc.vk_output.extent.width ||
        previous_output_extent.height != m_desc.vk_output.extent.height;
    const bool output_view_changed = previous_output_view != m_desc.vk_output.view;
    if (output_extent_changed || output_view_changed || !m_desc.framebuffer) {
        if (!recreateFramebuffer(device, rr)) {
            setPrepared(false);
            return;
        }
    }
}

void TextPass::execute(const Device& device, RenderingResources& rr) {
    if (m_desc.should_execute && !m_desc.should_execute()) return;
    auto* node = m_desc.node;
    auto* primitive = node != nullptr ? node->Text() : nullptr;
    if (primitive == nullptr) return;
    if (!m_desc.pipeline.handle || !m_desc.framebuffer) return;
    if (node != nullptr && !node->Visible() && !m_desc.execute_when_hidden) return;

    // Log the first actual draw of each layout revision when investigating disappearing text.
    // Preparation alone cannot establish that an atlas, target and transform reached a draw.
    static const bool trace_destination = std::getenv("WESCENE_TRACE_TEXT_DESTINATION") != nullptr;
    const bool trace_revision = trace_destination &&
        m_traced_atlas_version != primitive->atlas_version;

    if (primitive->atlas_version != m_loaded_atlas_version ||
        m_desc.page_textures.size() != primitive->glyph_pages.size()) {
        // Text atlas content is owned by the scene primitive, not by render-graph pass creation.
        // Runtime text updates can therefore swap atlas pages or change page counts without a
        // graph rebuild. Refreshing the bound atlas images lazily here keeps the dedicated text
        // pass on the new scene-owned source of truth instead of depending on parser-time texture
        // registration.
        if (!refreshTextures(device)) return;
    }

    if (primitive->background_mesh != nullptr &&
        !ensureMeshBuffers(*primitive->background_mesh, m_background_buffers, rr)) {
        return;
    }
    if (m_page_buffers.size() != primitive->glyph_pages.size()) {
        m_page_buffers.resize(primitive->glyph_pages.size());
        for (auto& buffers : m_page_buffers) buffers.force_upload = true;
    }
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        if (!ensureMeshBuffers(*primitive->glyph_pages[page_index].mesh, m_page_buffers[page_index], rr)) {
            return;
        }
    }

    auto write_uniforms = [&](const std::array<float, 4>& color) {
        TextPassUniforms uniforms {};
        bool transform_written = false;
        if (m_desc.scene != nullptr && m_desc.scene->shaderValueUpdater != nullptr && node != nullptr) {
            sprite_map_t sprites;
            // The glyph source and its final effect draw share one authored text object.
            // Local glyph rasterization selects its bridge projection and identity I through
            // pass state, leaving live alignment, attachment and script transforms untouched.
            const ShaderUniformOverrides overrides {
                .camera_name = m_desc.camera_override,
                .use_camera_override = !m_desc.camera_override.empty(),
                .use_active_camera_for_parallax = m_desc.use_active_camera_for_parallax,
                .model_space = m_desc.model_space,
                .reflection_pass = m_desc.reflection_pass,
                .reflection_raster = m_desc.reflection_raster,
                .reflection_snapshot = m_desc.reflection_snapshot,
                .effect_snapshot_camera = m_desc.effect_snapshot_camera,
            };
            m_desc.scene->shaderValueUpdater->UpdateUniforms(
                node,
                sprites,
                [&uniforms, &transform_written](std::string_view name, wallpaper::ShaderValue value) {
                    if (name != wallpaper::G_MVP || value.size() < 16) return;
                    transform_written = true;
                    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
                    for (int column = 0; column < 4; column++) {
                        for (int row = 0; row < 4; row++) {
                            // ShaderValue uses a size_t index while the matrix loops are small
                            // signed integers; materializing the index keeps warning-clean builds
                            // without changing the column-major uniform contract.
                            const auto uniform_index = static_cast<size_t>(column * 4 + row);
                            matrix(row, column) = value[uniform_index];
                        }
                    }
                    WriteMatrixToUniform(uniforms, matrix);
                }, &overrides);
        }
        std::copy(color.begin(), color.end(), uniforms.color);
        if (trace_revision) {
            const auto* matrix = uniforms.model_view_projection;
            LOG_INFO("TextDestinationDraw: layer=%d name='%s' output='%s' extent=%ux%u "
                     "atlas=%u pages=%zu transform=%s diagonal=[%.6f %.6f %.6f %.6f] "
                     "translation=[%.6f %.6f %.6f] color=[%.3f %.3f %.3f %.3f]",
                     m_desc.layer_id, node->Name().c_str(), m_desc.output.c_str(),
                     m_desc.vk_output.extent.width, m_desc.vk_output.extent.height,
                     primitive->atlas_version, primitive->glyph_pages.size(),
                     transform_written ? "written" : "missing",
                     matrix[0], matrix[5], matrix[10], matrix[15],
                     matrix[12], matrix[13], matrix[14], color[0], color[1], color[2], color[3]);
        }
        rr.dyn_buf->writeToBuf(m_desc.ubo_buf,
                               { reinterpret_cast<uint8_t*>(const_cast<TextPassUniforms*>(&uniforms)),
                                 sizeof(uniforms) });
    };

    auto bind_uniforms = [&](const PipelineParameters& pipeline) {
        VkDescriptorBufferInfo buffer_info {
            rr.dyn_buf->gpuBuf(),
            m_desc.ubo_buf.offset,
            m_desc.ubo_buf.size,
        };
        VkWriteDescriptorSet write {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_info,
        };
        rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline.layout, 0, write);
    };

    auto bind_texture = [&](const ImageSlotsRef& slots, const PipelineParameters& pipeline) {
        if (slots.slots.empty()) return;
        const auto& image = slots.getActive();
        VkDescriptorImageInfo image_info {
            image.sampler,
            image.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet write {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
        };
        rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline.layout, 0, write);
    };

    const VkExtent2D output_extent {
        .width = m_desc.vk_output.extent.width,
        .height = m_desc.vk_output.extent.height,
    };
    std::array<VkClearValue, 3> clear_values { m_desc.clear_value, {}, {} };
    VkRenderPassBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = *m_desc.pipeline.pass,
        .framebuffer = *m_desc.framebuffer,
        .renderArea = VkRect2D { .offset = { 0, 0 }, .extent = output_extent },
        .clearValueCount = 1u + (m_desc.shared_depth ? 1u : 0u) + (m_desc.resolve_msaa ? 1u : 0u),
        .pClearValues = clear_values.data(),
    };
    if (trace_revision) {
        // Empty layouts still begin a render pass. Include the attachment operation in the
        // existing opt-in trace so clearing a shared composition is observable even when there
        // are no glyph draws and therefore no TextDestinationDraw uniform log for this layer.
        LOG_INFO("TextDestinationBegin: layer=%d name='%s' output='%s' image=%p "
                 "extent=%ux%u clear=%s atlas=%u pages=%zu opaque-background=%s",
                 m_desc.layer_id, node->Name().c_str(), m_desc.output.c_str(),
                 reinterpret_cast<void*>(m_desc.vk_output.handle), output_extent.width,
                 output_extent.height, m_desc.clear_before_draw ? "true" : "false",
                 primitive->atlas_version, primitive->glyph_pages.size(),
                 primitive->object.opaquebackground ? "true" : "false");
    }
    rr.command.BeginRenderPass(begin_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport {
        .x = 0.0f,
        .y = static_cast<float>(m_desc.vk_output.extent.height),
        .width = static_cast<float>(m_desc.vk_output.extent.width),
        .height = -static_cast<float>(m_desc.vk_output.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D scissor { { 0, 0 }, output_extent };
    rr.command.SetViewport(0, viewport);
    rr.command.SetScissor(0, scissor);

    auto draw_mesh = [&](MeshBuffers& buffers, const ImageSlotsRef& texture,
                         const std::array<float, 4>& color, const PipelineParameters& pipeline,
                         bool background) {
        if (buffers.draw_count == 0) return;
        rr.command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline.handle);
        write_uniforms(color);
        bind_uniforms(pipeline);
        bind_texture(texture, pipeline);
        auto gpu_buf = rr.dyn_buf->gpuBuf();
        for (usize binding_index = 0; binding_index < buffers.vertex_bufs.size(); binding_index++) {
            auto& subref = buffers.vertex_bufs[binding_index];
            rr.command.BindVertexBuffers(static_cast<uint32_t>(binding_index), 1, &gpu_buf, &subref.offset);
        }
        // Glyph page meshes are indexed, while the optional opaque background is a plain strip.
        // Supporting both draw modes keeps the direct text primitive self-contained instead of
        // depending on the old generic image pass behavior for one half of the text renderable.
        if (buffers.index_buf) {
            rr.command.BindIndexBuffer(gpu_buf, buffers.index_buf.offset, VK_INDEX_TYPE_UINT16);
            rr.command.DrawIndexed(buffers.draw_count, 1, 0, 0, 0);
        } else {
            rr.command.Draw(buffers.draw_count, 1, 0, 0);
        }
        if (std::getenv("WESCENE_TRACE_TEXT_DEPTH") != nullptr) {
            // Record the submitted consumer, not just the owner property's readback. Including
            // attachment and pipeline identity distinguishes live state replacement from atlas
            // refresh and proves that main/reflection draws retain their destination's storage.
            LOG_INFO("SceneTextDepthDraw: layer=%d output='%s' reflection=%s background=%s "
                     "shared-depth=%s test=%s write=false depth-view=%p pipeline=%p samples=%u count=%u",
                     m_desc.layer_id, m_desc.output.c_str(), m_desc.reflection_pass ? "true" : "false",
                     background ? "true" : "false", m_desc.shared_depth ? "true" : "false",
                     !background && m_desc.glyph_depth_test ? "true" : "false",
                     m_desc.depth_image != nullptr ? reinterpret_cast<void*>(*m_desc.depth_image->view)
                                                   : nullptr,
                     reinterpret_cast<void*>(*pipeline.handle),
                     static_cast<unsigned>(m_desc.sample_count), buffers.draw_count);
        }
    };

    if (primitive->object.opaquebackground && primitive->background_mesh != nullptr) {
        draw_mesh(m_background_buffers,
                  m_desc.background_texture,
                  ResolveTextColor(*primitive, true),
                  m_desc.glyph_depth_test ? m_desc.background_pipeline : m_desc.pipeline, true);
    }

    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        if (page_index >= m_desc.page_textures.size()) break;
        draw_mesh(m_page_buffers[page_index],
                  m_desc.page_textures[page_index],
                  ResolveTextColor(*primitive, false), m_desc.pipeline, false);
    }

    rr.command.EndRenderPass();
    if (m_desc.shared_depth) {
        rr.model_depth_images.at(m_desc.output).layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    if (trace_revision) m_traced_atlas_version = primitive->atlas_version;

    if (m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        m_desc.output == wallpaper::SpecTex_Default) {
        NoteComposeMsaaDraw(rr, m_desc.sample_count);
    }
}

void TextPass::destory(const Device&, RenderingResources& rr) {
    // Keep the cached text PSO alive through PipelineStateCache, but release every residency-bound
    // object that points at hidden-layer textures, render targets, or dynamic-buffer suballocations.
    m_desc.framebuffer.reset();
    m_desc.vk_output = {};
    m_desc.vk_resolve = {};
    m_desc.depth_image = nullptr;
    m_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
    m_desc.resolve_msaa = false;
    m_desc.background_texture = {};
    m_desc.page_textures.clear();
    for (auto& subref : m_background_buffers.vertex_bufs) {
        rr.dyn_buf->unallocateSubRef(subref);
    }
    if (m_background_buffers.index_buf) rr.dyn_buf->unallocateSubRef(m_background_buffers.index_buf);
    m_background_buffers = {};
    for (auto& page_buffers : m_page_buffers) {
        for (auto& subref : page_buffers.vertex_bufs) {
            rr.dyn_buf->unallocateSubRef(subref);
        }
        if (page_buffers.index_buf) rr.dyn_buf->unallocateSubRef(page_buffers.index_buf);
    }
    m_page_buffers.clear();
    rr.dyn_buf->unallocateSubRef(m_desc.ubo_buf);
    m_desc.ubo_buf = {};
    setPrepared(false);
}

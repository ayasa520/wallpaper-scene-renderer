#include "CustomShaderPass.hpp"
#include "Scene/Scene.h"
#include "Scene/SceneShader.h"
#include "Scene/SceneTextPrimitive.h"

#include "SpecTexs.hpp"
#include "Vulkan/Shader.hpp"
#include "Vulkan/VideoTextureCache.hpp"
#include "Utils/Logging.h"
#include "Utils/AutoDeletor.hpp"
#include "Resource.hpp"
#include "PassCommon.hpp"
#include "Interface/IImageParser.h"

#include "Core/ArrayHelper.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <sstream>

using namespace wallpaper::vulkan;

namespace
{
constexpr std::array<std::string_view, 6> kAudioSpectrumUniformNames {
    "g_AudioSpectrum16Left",
    "g_AudioSpectrum16Right",
    "g_AudioSpectrum32Left",
    "g_AudioSpectrum32Right",
    "g_AudioSpectrum64Left",
    "g_AudioSpectrum64Right",
};

bool HasAudioSpectrumUniforms(const ShaderReflected::Block& block) {
    return std::any_of(kAudioSpectrumUniformNames.begin(),
                       kAudioSpectrumUniformNames.end(),
                       [&block](std::string_view name) {
                           return wallpaper::exists(block.member_map, name);
                       });
}

const wallpaper::ShaderValue* FindMaterialUniformValue(const wallpaper::SceneMaterial& material,
                                                       std::string_view                name) {
    if (const auto it = material.customShader.constValues.find(name);
        it != material.customShader.constValues.end()) {
        return &it->second;
    }
    if (material.customShader.shader != nullptr) {
        if (const auto it = material.customShader.shader->default_uniforms.find(name);
            it != material.customShader.shader->default_uniforms.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

std::string DescribeMaterialUniformValue(const wallpaper::SceneMaterial& material,
                                         std::string_view                name) {
    const auto* value = FindMaterialUniformValue(material, name);
    if (value == nullptr) return "<missing>";

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < value->size(); i++) {
        if (i != 0) oss << ", ";
        oss << (*value)[i];
        if (i >= 3 && value->size() > 4) {
            oss << ", ...";
            break;
        }
    }
    oss << "]";
    return oss.str();
}

template <typename Map>
std::string DescribeUniformMap(const Map& values) {
    std::ostringstream oss;
    oss << "{";
    size_t count = 0;
    for (const auto& [name, value] : values) {
        if (count != 0) oss << ", ";
        oss << name << "=";
        oss << "[";
        for (size_t i = 0; i < value.size(); i++) {
            if (i != 0) oss << ", ";
            oss << value[i];
            if (i >= 3 && value.size() > 4) {
                oss << ", ...";
                break;
            }
        }
        oss << "]";
        count++;
        if (count >= 8 && values.size() > count) {
            oss << ", ...";
            break;
        }
    }
    oss << "}";
    return oss.str();
}

std::string DescribeTextureList(const wallpaper::SceneMaterial& material) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < material.textures.size(); i++) {
        if (i != 0) oss << ", ";
        oss << i << "=";
        oss << "'" << material.textures[i] << "'";
    }
    oss << "]";
    return oss.str();
}

std::string DescribeDescTextureList(const std::vector<std::string>& textures) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < textures.size(); i++) {
        if (i != 0) oss << ", ";
        oss << i << "=";
        oss << "'" << textures[i] << "'";
    }
    oss << "]";
    return oss.str();
}

std::string DescribeBlockMemberList(const ShaderReflected::Block& block) {
    std::ostringstream oss;
    oss << "{";
    size_t count = 0;
    for (const auto& [name, uniform] : block.member_map) {
        if (count != 0) oss << ", ";
        oss << name << "=(offset=" << uniform.offset << ", num=" << uniform.num << ")";
        count++;
    }
    oss << "}";
    return oss.str();
}

void LogAudioUniformBlockLayout(const CustomShaderPass::Desc& desc,
                                const wallpaper::SceneMaterial& material,
                                const ShaderReflected::Block& block) {
    if (!HasAudioSpectrumUniforms(block)) return;

    const auto* node = desc.node;
    const auto* shader = material.customShader.shader.get();
    LOG_INFO("SceneAudioUniformBlock: layer=%d name='%s' shader='%s' members=%s",
             node != nullptr ? const_cast<wallpaper::SceneNode*>(node)->ID() : -1,
             node != nullptr ? node->Name().c_str() : "<null>",
             shader != nullptr ? shader->name.c_str() : "<null>",
             DescribeBlockMemberList(block).c_str());
}

void LogAudioReactivePassConfig(const CustomShaderPass::Desc& desc,
                                const wallpaper::SceneMaterial& material,
                                const ShaderReflected::Block& block) {
    if (!HasAudioSpectrumUniforms(block)) return;

    const auto* node = desc.node;
    const auto* shader = material.customShader.shader.get();
    LOG_INFO(
        "SceneAudioPass: layer=%d name='%s' shader='%s' output='%s' visible=%s textures=%zu barCount=%s volumeFactor=%s barBounds=%s barOpacity=%s barSpacing=%s barColor=%s hideTs=%s dynamicHide=%s radius=%s segmentSpacing=%s segmentCount=%s segmentThreshold=%s textureList=%s",
        node != nullptr ? const_cast<wallpaper::SceneNode*>(node)->ID() : -1,
        node != nullptr ? node->Name().c_str() : "<null>",
        shader != nullptr ? shader->name.c_str() : "<null>",
        desc.output.c_str(),
        node != nullptr && node->Visible() ? "true" : "false",
        desc.textures.size(),
        DescribeMaterialUniformValue(material, "u_BarCount").c_str(),
        DescribeMaterialUniformValue(material, "u_VolumeFactor").c_str(),
        DescribeMaterialUniformValue(material, "u_BarBounds").c_str(),
        DescribeMaterialUniformValue(material, "u_BarOpacity").c_str(),
        DescribeMaterialUniformValue(material, "u_BarSpacing").c_str(),
        DescribeMaterialUniformValue(material, "u_BarColor").c_str(),
        DescribeMaterialUniformValue(material, "u_TsOfHiding").c_str(),
        DescribeMaterialUniformValue(material, "u_DynamicHiding").c_str(),
        DescribeMaterialUniformValue(material, "u_Radius").c_str(),
        DescribeMaterialUniformValue(material, "u_SegmentSpacing").c_str(),
        DescribeMaterialUniformValue(material, "u_SegmentCount").c_str(),
        DescribeMaterialUniformValue(material, "u_SegmentThreshold").c_str(),
        DescribeTextureList(material).c_str());
}

void LogAudioCompositePassConfig(const CustomShaderPass::Desc&  desc,
                                 const wallpaper::SceneMaterial& material) {
    const auto* node = desc.node;
    const auto* shader = material.customShader.shader.get();
    if (shader == nullptr || shader->name != "genericimage3") return;

    LOG_INFO(
        "SceneAudioCompositeConfig: layer=%d name='%s' ptr=%p shader='%s' output='%s' visible=%s textures=%zu texture0='%s' blendMode=%d color4=%s alpha=%s userAlpha=%s brightness=%s textureList=%s",
        node != nullptr ? const_cast<wallpaper::SceneNode*>(node)->ID() : -1,
        node != nullptr ? node->Name().c_str() : "<null>",
        static_cast<const void*>(node),
        shader->name.c_str(),
        desc.output.c_str(),
        node != nullptr && node->Visible() ? "true" : "false",
        material.textures.size(),
        !material.textures.empty() ? material.textures[0].c_str() : "<none>",
        static_cast<int>(material.blenmode),
        DescribeMaterialUniformValue(material, "g_Color4").c_str(),
        DescribeMaterialUniformValue(material, "g_Alpha").c_str(),
        DescribeMaterialUniformValue(material, "g_UserAlpha").c_str(),
        DescribeMaterialUniformValue(material, "g_Brightness").c_str(),
        DescribeTextureList(material).c_str());
    LOG_INFO("SceneAudioCompositeConfig: constValues=%s",
             DescribeUniformMap(material.customShader.constValues).c_str());
    if (shader != nullptr) {
        LOG_INFO("SceneAudioCompositeConfig: defaultUniforms=%s",
                 DescribeUniformMap(shader->default_uniforms).c_str());
    }
}

void LogAudioReactivePassExecute(const CustomShaderPass::Desc& desc) {
    const auto* node = desc.node;
    auto*       mutable_node = const_cast<wallpaper::SceneNode*>(node);
    if (mutable_node == nullptr || mutable_node->Mesh() == nullptr ||
        mutable_node->Mesh()->Material() == nullptr) {
        return;
    }

    const auto& material = *mutable_node->Mesh()->Material();
    const auto* shader = material.customShader.shader.get();
    if (shader == nullptr) return;
    const bool is_audio_bar_shader =
        shader->name.find("Simple_Audio_Bars") != std::string::npos;
    if (!is_audio_bar_shader) return;

    static std::unordered_map<int32_t, size_t> s_log_counts;
    const auto layer_id = mutable_node->ID();
    const auto count = ++s_log_counts[layer_id];
    if (count != 1 && (count % 300) != 0) return;

    const auto& t = node->Translate();
    const auto& s = node->Scale();
    const auto& r = node->Rotation();
    LOG_INFO(
        "SceneAudioPassExecute: layer=%d name='%s' output='%s' visible=%s drawCount=%u indexed=%s rt=%ux%u translate=(%.3f,%.3f,%.3f) scale=(%.3f,%.3f,%.3f) rotation=(%.3f,%.3f,%.3f)",
        layer_id,
        node->Name().c_str(),
        desc.output.c_str(),
        node->Visible() ? "true" : "false",
        desc.draw_count,
        desc.index_buf ? "true" : "false",
        desc.vk_output.extent.width,
        desc.vk_output.extent.height,
        t.x(),
        t.y(),
        t.z(),
        s.x(),
        s.y(),
        s.z(),
        r.x(),
        r.y(),
        r.z());
    LOG_INFO("SceneAudioPassExecute: textures=%s",
             DescribeTextureList(material).c_str());
    LOG_INFO("SceneAudioPassExecute: desc-textures=%s",
             DescribeDescTextureList(desc.textures).c_str());
}

void LogAudioCompositePassExecute(const CustomShaderPass::Desc& desc) {
    const auto* node = desc.node;
    auto*       mutable_node = const_cast<wallpaper::SceneNode*>(node);
    if (mutable_node == nullptr || mutable_node->Mesh() == nullptr ||
        mutable_node->Mesh()->Material() == nullptr) {
        return;
    }

    const auto* shader = mutable_node->Mesh()->Material()->customShader.shader.get();
    const auto* material = mutable_node->Mesh()->Material();
    if (shader == nullptr || material == nullptr) return;
    if (shader->name != "genericimage3") {
        return;
    }

    static std::unordered_map<int32_t, size_t> s_log_counts;
    const auto layer_id = mutable_node->ID();
    const auto count = ++s_log_counts[layer_id];
    if (count != 1 && (count % 300) != 0) return;

    const auto& t = node->Translate();
    const auto& s = node->Scale();
    const auto& r = node->Rotation();
    LOG_INFO(
        "SceneAudioCompositePassExecute: layer=%d name='%s' ptr=%p output='%s' visible=%s drawCount=%u indexed=%s rt=%ux%u translate=(%.3f,%.3f,%.3f) scale=(%.3f,%.3f,%.3f) rotation=(%.3f,%.3f,%.3f)",
        layer_id,
        node->Name().c_str(),
        static_cast<const void*>(node),
        desc.output.c_str(),
        node->Visible() ? "true" : "false",
        desc.draw_count,
        desc.index_buf ? "true" : "false",
        desc.vk_output.extent.width,
        desc.vk_output.extent.height,
        t.x(),
        t.y(),
        t.z(),
        s.x(),
        s.y(),
        s.z(),
        r.x(),
        r.y(),
        r.z());
    LOG_INFO("SceneAudioCompositePassExecute: texture0='%s'",
             !material->textures.empty() ? material->textures[0].c_str() : "<none>");
    LOG_INFO("SceneAudioCompositePassExecute: textures=%s",
             DescribeTextureList(*material).c_str());
    LOG_INFO("SceneAudioCompositePassExecute: desc-textures=%s",
             DescribeDescTextureList(desc.textures).c_str());
}

}

CustomShaderPass::CustomShaderPass(const Desc& desc) {
    // The render graph builder already classifies hidden offscreen dependencies and gives passes
    // a live scene pointer for diagnostics. Preserve that prepared intent here; dropping these
    // fields forced text/effect passes back through generic visibility and null-scene behavior.
    m_desc.scene               = desc.scene;
    m_desc.node                = desc.node;
    m_desc.execute_when_hidden = desc.execute_when_hidden;
    m_desc.should_execute      = desc.should_execute;
    m_desc.textures            = desc.textures;
    m_desc.output              = desc.output;
    m_desc.sprites_map         = desc.sprites_map;
};
CustomShaderPass::~CustomShaderPass() {}

bool CustomShaderPass::referencesRenderTarget(std::string_view render_target) const {
    // Custom shader passes are affected when either their output framebuffer is the dirty target or
    // one of their descriptor inputs samples it. This lets a resized text bridge update the exact
    // effect chain that consumes it instead of refreshing every other shader in the wallpaper.
    if (m_desc.output == render_target) return true;
    for (const auto& texture : m_desc.textures) {
        if (texture == render_target) return true;
    }
    return false;
}

std::optional<vvk::RenderPass> CreateRenderPass(const vvk::Device& device, VkFormat format,
                                                VkAttachmentLoadOp loadOp,
                                                VkImageLayout      finalLayout) {
    VkAttachmentDescription attachment {
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = loadOp, // VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = finalLayout, // ShaderReadOnlyOptimal
    };

    if (loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) {
        attachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkAttachmentReference attachment_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass {
        .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &attachment_ref,
    };

    VkSubpassDependency dependency {
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = {},
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPassCreateInfo creatinfo {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &dependency,
    };
    vvk::RenderPass pass;
    if (auto res = device.CreateRenderPass(creatinfo, pass); res == VK_SUCCESS) {
        return pass;
    } else {
        VVK_CHECK(res);
        return std::nullopt;
    }
}

static void UpdateUniform(StagingBuffer* buf, const StagingBufferRef& bufref,
                          const ShaderReflected::Block& block, std::string_view name,
                          const wallpaper::ShaderValue& value) {
    using namespace wallpaper;
    std::span<uint8_t> value_u8 { (uint8_t*)value.data(),
                                  value.size() * sizeof(ShaderValue::value_type) };
    auto               uni = block.member_map.find(name);
    if (uni == block.member_map.end()) {
        // log
        return;
    }

    const size_t offset        = uni->second.offset;
    const size_t reflectedSize = uni->second.size;
    const size_t packedSize    = value_u8.size();

    if (reflectedSize == packedSize || value.size() <= 1) {
        buf->writeToBuf(bufref, value_u8, offset);
        return;
    }

    // SPIR-V reflection reports std140 array sizes for uniforms such as
    // `float g_AudioSpectrum32Left[32]`, which occupy 16 bytes per element.
    // Our runtime values are stored densely as `float[N]`, so copy them using
    // the reflected stride instead of writing the packed blob directly.
    if (reflectedSize > packedSize && reflectedSize % value.size() == 0) {
        const size_t stride = reflectedSize / value.size();
        if (stride >= sizeof(ShaderValue::value_type)) {
            for (size_t i = 0; i < value.size(); i++) {
                std::span<uint8_t> elem {
                    reinterpret_cast<uint8_t*>(const_cast<ShaderValue::value_type*>(&value[i])),
                    sizeof(ShaderValue::value_type)
                };
                buf->writeToBuf(bufref, elem, offset + i * stride);
            }
            return;
        }
    }

    buf->writeToBuf(bufref, value_u8, offset);
}

static void WriteMaterialUniforms(StagingBuffer*                    buf,
                                  const StagingBufferRef&           bufref,
                                  const ShaderReflected::Block&     block,
                                  const wallpaper::SceneMaterial&   material) {
    auto write_values = [&](const auto& values) {
        for (const auto& [name, value] : values) {
            if (!wallpaper::exists(block.member_map, name)) continue;
            UpdateUniform(buf, bufref, block, name, value);
        }
    };

    if (material.customShader.shader != nullptr) {
        write_values(material.customShader.shader->default_uniforms);
    }
    write_values(material.customShader.constValues);
}

void LogStagingAllocRequest(std::string_view kind, const CustomShaderPass::Desc& desc,
                            VkDeviceSize size) {
    LOG_INFO("StagingAllocRequest: kind=%.*s node='%s' camera='%s' shader='%s' output='%s' size=%zu dyn=%s",
             static_cast<int>(kind.size()),
             kind.data(),
             desc.node ? desc.node->Name().c_str() : "<null>",
             desc.node ? desc.node->Camera().c_str() : "<null>",
             desc.node && desc.node->Mesh() && desc.node->Mesh()->Material() &&
                     desc.node->Mesh()->Material()->customShader.shader
                 ? desc.node->Mesh()->Material()->customShader.shader->name.c_str()
                 : "<null>",
             desc.output.c_str(),
             static_cast<size_t>(size),
             desc.dyn_vertex ? "true" : "false");
}

constexpr VkDeviceSize kInitialDynamicSuballocationSize = 64 * 1024;

VkDeviceSize InitialDynamicSuballocationSize(VkDeviceSize capacity,
                                             VkDeviceSize live_size,
                                             VkDeviceSize element_size) {
    if (capacity == 0) return 0;

    const VkDeviceSize non_empty_element = std::max<VkDeviceSize>(element_size, 1);
    const VkDeviceSize required_live = std::max<VkDeviceSize>(live_size, non_empty_element);
    const VkDeviceSize bootstrap =
        std::min<VkDeviceSize>(capacity,
                               std::max<VkDeviceSize>(kInitialDynamicSuballocationSize,
                                                      non_empty_element));

    // Dynamic particle meshes often advertise a very large theoretical capacity while starting
    // with zero live vertices. Reserve only a small bootstrap range up front, but never choose a
    // range smaller than the data that is already live and must be uploaded immediately.
    return std::min<VkDeviceSize>(capacity, std::max(required_live, bootstrap));
}

VkDeviceSize DynamicVertexUploadSize(const wallpaper::SceneVertexArray& vertex) {
    // Vertex arrays expose both live bytes and authored capacity. Use the live byte count for the
    // first upload so character-rain style particle systems do not reserve their entire theoretical
    // maximum before any spawned particles exist.
    return InitialDynamicSuballocationSize(
        static_cast<VkDeviceSize>(vertex.CapacitySizeOf()),
        static_cast<VkDeviceSize>(vertex.DataSizeOf()),
        static_cast<VkDeviceSize>(vertex.OneSizeOf()));
}

VkDeviceSize DynamicIndexUploadSize(const wallpaper::SceneIndexArray& indice) {
    // Index buffers follow the same bootstrap rule as vertices. A single quad's index footprint is
    // used as the non-empty floor because empty dynamic meshes still need a sane first allocation
    // when the scene begins spawning particles later.
    return InitialDynamicSuballocationSize(
        static_cast<VkDeviceSize>(indice.CapacitySizeof()),
        static_cast<VkDeviceSize>(indice.DataSizeOf()),
        static_cast<VkDeviceSize>(sizeof(uint32_t) * 6));
}

VkDeviceSize GrowDynamicSuballocationSize(VkDeviceSize current_size,
                                          VkDeviceSize required_live_size,
                                          VkDeviceSize capacity,
                                          VkDeviceSize element_size) {
    if (capacity == 0) return 0;

    // Growth is geometric but clamped to authored capacity. That keeps normal particle expansion
    // amortized while still refusing to cross the renderer-side maximum promised by the scene data.
    VkDeviceSize next_size = current_size == 0
        ? InitialDynamicSuballocationSize(capacity, required_live_size, element_size)
        : current_size;
    const VkDeviceSize required =
        std::min<VkDeviceSize>(capacity,
                               std::max<VkDeviceSize>(required_live_size,
                                                      std::max<VkDeviceSize>(element_size, 1)));

    while (next_size < required && next_size < capacity) {
        const VkDeviceSize doubled = next_size > capacity / 2 ? capacity : next_size * 2;
        next_size = std::max<VkDeviceSize>(required, doubled);
        next_size = std::min<VkDeviceSize>(next_size, capacity);
    }
    return next_size;
}

bool RefreshCustomShaderPassTextures(wallpaper::Scene& scene,
                                     const Device&    device,
                                     CustomShaderPass::Desc& desc) {
    desc.vk_textures.resize(desc.textures.size());
    for (wallpaper::usize i = 0; i < desc.textures.size(); i++) {
        auto& tex_name = desc.textures[i];
        if (tex_name.empty()) {
            desc.vk_textures[i] = {};
            continue;
        }

        ImageSlotsRef img_slots;
        if (wallpaper::IsSpecTex(tex_name)) {
            const auto render_target_it = scene.renderTargets.find(tex_name);
            if (render_target_it == scene.renderTargets.end()) {
                LOG_ERROR("CustomShaderPassRefresh: missing input render target node='%s' "
                          "output='%s' slot=%zu texture='%s'",
                          desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                          desc.output.c_str(),
                          static_cast<size_t>(i),
                          tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
            auto& rt  = render_target_it->second;
            auto  opt = device.tex_cache().Query(
                tex_name, wallpaper::vulkan::ToTexKey(rt), !rt.allowReuse);
            if (!opt.has_value()) {
                LOG_ERROR("CustomShaderPassRefresh: query input failed node='%s' output='%s' "
                          "slot=%zu texture='%s'",
                          desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                          desc.output.c_str(),
                          static_cast<size_t>(i),
                          tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
            img_slots.slots.clear();
            img_slots.slots.push_back(opt.value());
        } else {
            auto image = scene.imageParser->Parse(tex_name);
            if (image) {
                if (scene.textures.count(tex_name) != 0 && scene.textures.at(tex_name).isVideo) {
                    const auto paused_it = scene.videoTexturePaused.find(tex_name);
                    // Hidden video passes are kept prepared so visibility flips are cheap, but the
                    // backing decoder should still start paused unless a scene script explicitly
                    // requested playback for this texture.
                    const bool initially_paused =
                        paused_it != scene.videoTexturePaused.end()
                            ? paused_it->second
                            : (desc.node != nullptr && !desc.node->Visible());
                    img_slots = device.video_tex_cache().Acquire(
                        tex_name, scene.textures.at(tex_name), *image, initially_paused);
                } else {
                    img_slots = device.tex_cache().CreateTex(*image);
                }
            } else {
                LOG_ERROR("parse tex \"%s\" failed", tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
        }
        desc.vk_textures[i] = img_slots;
    }

    auto& tex_name = desc.output;
    if (!wallpaper::IsSpecTex(tex_name)) {
        LOG_ERROR("CustomShaderPassRefresh: non-render-target output node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  tex_name.c_str());
        return false;
    }
    const auto output_it = scene.renderTargets.find(tex_name);
    if (output_it == scene.renderTargets.end()) {
        LOG_ERROR("CustomShaderPassRefresh: missing output render target node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  tex_name.c_str());
        return false;
    }
    auto& rt = output_it->second;
    if (auto opt = device.tex_cache().Query(
            tex_name, wallpaper::vulkan::ToTexKey(rt), !rt.allowReuse);
        opt.has_value()) {
        desc.vk_output = opt.value();
        return true;
    }
    LOG_ERROR("CustomShaderPassRefresh: query output failed node='%s' output='%s'",
              desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
              tex_name.c_str());
    return false;
}

bool RecreateCustomShaderPassFramebuffer(const Device& device, CustomShaderPass::Desc& desc) {
    desc.fb.reset();
    if (!desc.pipeline.pass || desc.vk_output.view == VK_NULL_HANDLE ||
        desc.vk_output.extent.width == 0 || desc.vk_output.extent.height == 0) {
        LOG_ERROR("CustomShaderPassRefresh: cannot recreate framebuffer node='%s' output='%s' "
                  "hasRenderPass=%s hasView=%s extent=[%u,%u]",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  desc.output.c_str(),
                  desc.pipeline.pass ? "true" : "false",
                  desc.vk_output.view != VK_NULL_HANDLE ? "true" : "false",
                  desc.vk_output.extent.width,
                  desc.vk_output.extent.height);
        return false;
    }
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .renderPass      = *desc.pipeline.pass,
        .attachmentCount = 1,
        .pAttachments    = &desc.vk_output.view,
        .width           = desc.vk_output.extent.width,
        .height          = desc.vk_output.extent.height,
        .layers          = 1,
    };
    return device.handle().CreateFramebuffer(info, desc.fb) == VK_SUCCESS;
}

void CustomShaderPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    // Prepared passes can survive resource-only refreshes, so keep the live scene pointer current
    // before binding render targets and before text/effect diagnostics read bridge metadata.
    m_desc.scene = &scene;
    // A retrying prepared pass may still own a framebuffer whose attachment points at the previous
    // TextureCache image view. Drop it before `Query()` can resize and destroy that output image,
    // otherwise Vulkan sees a framebuffer referencing a dead attachment during minute-rollover
    // text bridge updates.
    m_desc.fb.reset();
    m_desc.vk_tex_binding.clear();
    if (!RefreshCustomShaderPassTextures(scene, device, m_desc)) return;
    SceneMesh& mesh = *(m_desc.node->Mesh());

    std::vector<Uni_ShaderSpv> spvs;
    DescriptorSetInfo          descriptor_info;
    ShaderReflected            ref;
    {
        SceneShader& shader = *(mesh.Material()->customShader.shader);

        if (! GenReflect(shader.codes, spvs, ref)) {
            LOG_ERROR("gen spv reflect failed, %s", shader.name.c_str());
            return;
        }

        auto& bindings = descriptor_info.bindings;
        bindings.resize(ref.binding_map.size());

        /*
        LOG_INFO("----shader------");
        LOG_INFO("%s", shader.name.c_str());
        LOG_INFO("--inputs:");
        for (auto& i : ref.input_location_map) {
            LOG_INFO("%d %s", i.second, i.first.c_str());
        }
        LOG_INFO("--bindings:");
        */

        std::transform(
            ref.binding_map.begin(), ref.binding_map.end(), bindings.begin(), [](auto& item) {
                // LOG_INFO("%d %s", item.second.binding, item.first.c_str());
                return item.second;
            });

        for (usize i = 0; i < m_desc.vk_textures.size(); i++) {
            i32 binding { -1 };
            if (exists(ref.binding_map, WE_GLTEX_NAMES[i]))
                binding = (i32)ref.binding_map.at(WE_GLTEX_NAMES[i]).binding;
            m_desc.vk_tex_binding.push_back(binding);
        }
    }

    if (!ref.blocks.empty() && mesh.Material() != nullptr) {
        LogAudioReactivePassConfig(m_desc, *mesh.Material(), ref.blocks.front());
        LogAudioCompositePassConfig(m_desc, *mesh.Material());
    }

    m_desc.draw_count = 0;
    std::vector<VkVertexInputBindingDescription>   bind_descriptions;
    std::vector<VkVertexInputAttributeDescription> attr_descriptions;
    {
        m_desc.dyn_vertex = mesh.Dynamic();
        // Dynamic meshes allocate fresh staging/GPU subranges every time the render graph is
        // recompiled. A static-looking text layer can therefore become "clean" long before a new
        // pass instance is created, which leaves the newly allocated buffer ranges uninitialized if
        // we only upload on `mesh.Dirty()`. Marking the pass for one mandatory upload keeps
        // long-lived text quads valid across unrelated render-graph rebuilds triggered by other
        // animated layers such as effect-backed clocks and dates.
        m_desc.force_dyn_upload = m_desc.dyn_vertex;
        m_desc.vertex_bufs.resize(mesh.VertexCount());

        for (uint i = 0; i < mesh.VertexCount(); i++) {
            const auto& vertex    = mesh.GetVertexArray(i);
            auto        attrs_map = vertex.GetAttrOffsetMap();

            VkVertexInputBindingDescription bind_desc {
                .binding   = i,
                .stride    = (uint32_t)vertex.OneSizeOf(),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };
            bind_descriptions.push_back(bind_desc);

            for (auto& item : ref.input_location_map) {
                auto& name   = item.first;
                auto& input  = item.second;
                usize offset = exists(attrs_map, name) ? attrs_map[name].offset : 0;

                VkVertexInputAttributeDescription attr_desc {
                    .location = input.location,
                    .binding  = i,
                    .format   = input.format,
                    .offset   = (u32)offset,
                };
                attr_descriptions.push_back(attr_desc);
            }
            {
                auto& buf = m_desc.vertex_bufs[i];
                if (! m_desc.dyn_vertex) {
                    if (vertex.CapacitySizeOf() >= 1024 * 1024) {
                        LogStagingAllocRequest("vertex-static", m_desc, vertex.CapacitySizeOf());
                    }
                    if (! rr.vertex_buf->allocateSubRef(vertex.CapacitySizeOf(), buf)) return;
                    if (! rr.vertex_buf->writeToBuf(buf, { (uint8_t*)vertex.Data(), buf.size }))
                        return;
                } else {
                    if (vertex.CapacitySizeOf() >= 1024 * 1024) {
                        LogStagingAllocRequest("vertex-dynamic", m_desc, vertex.CapacitySizeOf());
                    }
                    const auto initial_size = DynamicVertexUploadSize(vertex);
                    if (! rr.dyn_buf->allocateSubRef(initial_size, buf)) return;
                }
            }
            m_desc.draw_count += (u32)(vertex.DataSize() / vertex.OneSize());
        }

        if (mesh.IndexCount() > 0) {
            auto&  indice     = mesh.GetIndexArray(0);
            size_t count      = (indice.DataCount() * 2) / 3;
            m_desc.draw_count = (u32)count * 3;
            auto& buf         = m_desc.index_buf;
            if (! m_desc.dyn_vertex) {
                if (indice.CapacitySizeof() >= 1024 * 1024) {
                    LogStagingAllocRequest("index-static", m_desc, indice.CapacitySizeof());
                }
                if (! rr.vertex_buf->allocateSubRef(indice.CapacitySizeof(), buf)) return;
                if (! rr.vertex_buf->writeToBuf(buf, { (uint8_t*)indice.Data(), buf.size })) return;
            } else {
                if (indice.CapacitySizeof() >= 1024 * 1024) {
                    LogStagingAllocRequest("index-dynamic", m_desc, indice.CapacitySizeof());
                }
                const auto initial_size = DynamicIndexUploadSize(indice);
                if (! rr.dyn_buf->allocateSubRef(initial_size, buf)) return;
            }
        }
    }
    {
        VkPipelineColorBlendAttachmentState color_blend;
        VkAttachmentLoadOp                  loadOp { VK_ATTACHMENT_LOAD_OP_DONT_CARE };
        {
            VkColorComponentFlags colorMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
            const auto camera_name = m_desc.node != nullptr
                ? std::string_view(m_desc.node->Camera())
                : std::string_view {};
            // CustomShaderPass now handles only authored image/effect nodes. First-class text
            // bridge sources are rendered by TextPass, so alpha-write policy should come from the
            // pass node itself instead of walking synthetic text helper parents.
            bool alpha = ! (camera_name.empty() || sstart_with(camera_name, "global"));

            if (alpha) colorMask |= VK_COLOR_COMPONENT_A_BIT;
            color_blend.colorWriteMask = colorMask;

            auto blendmode = mesh.Material()->blenmode;
            SetBlend(blendmode, color_blend);
            m_desc.blending = color_blend.blendEnable;

            SetAttachmentLoadOp(blendmode, loadOp);
        }
        auto opt = CreateRenderPass(device.handle(),
                                    VK_FORMAT_R8G8B8A8_UNORM,
                                    loadOp,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (! opt.has_value()) return;
        auto& pass = opt.value();

        descriptor_info.push_descriptor = true;
        GraphicsPipeline pipeline;
        pipeline.toDefault();
        m_desc.pipeline.debug_name =
            "CustomShaderPass[node=" + (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
            ",output=" + m_desc.output + "]";
        pipeline.addDescriptorSetInfo(spanone { descriptor_info })
            .setColorBlendStates(spanone { color_blend })
            .setTopology(m_desc.index_buf ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                          : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
            .addInputBindingDescription(bind_descriptions)
            .addInputAttributeDescription(attr_descriptions);
        for (auto& spv : spvs) pipeline.addStage(std::move(spv));

        if (! pipeline.create(device, pass, m_desc.pipeline)) return;
    }
    {
        // The helper above already converts framebuffer creation into a plain success/failure
        // contract so that both the initial prepare path and the lightweight resource-refresh path
        // can share the same code. Keeping it as an explicit boolean check avoids routing a
        // non-VkResult helper through the `VVK_CHECK_*` macros, which only understand raw Vulkan
        // return codes.
        if (!RecreateCustomShaderPassFramebuffer(device, m_desc)) return;
    }

    if (! ref.blocks.empty()) {
        auto& block = ref.blocks.front();
        if (block.size >= 1024 * 1024) {
            LogStagingAllocRequest("ubo", m_desc, block.size);
        }
        rr.dyn_buf->allocateSubRef(
            block.size, m_desc.ubo_buf, device.limits().minUniformBufferOffsetAlignment);
    }

    if (! ref.blocks.empty()) {
        std::function<void()> update_dyn_buf_op;
        if (m_desc.dyn_vertex) {
            auto& mesh             = *m_desc.node->Mesh();
            auto* dyn_buf          = rr.dyn_buf;
            auto& vertex_bufs      = m_desc.vertex_bufs;
            auto& draw_count       = m_desc.draw_count;
            auto& index_buf        = m_desc.index_buf;
            auto& force_dyn_upload = m_desc.force_dyn_upload;
            update_dyn_buf_op =
                [&mesh, &vertex_bufs, &draw_count, &index_buf, dyn_buf, &force_dyn_upload]() {
                const bool dirty = mesh.Dirty().load();
                const bool needs_upload = dirty || force_dyn_upload;
                if (needs_upload) {
                    auto ensure_vertex_subref = [&](usize array_index,
                                                    const wallpaper::SceneVertexArray& vertex) {
                        if (vertex_bufs.size() <= array_index) {
                            vertex_bufs.resize(array_index + 1);
                        }

                        auto& buf = vertex_bufs[array_index];
                        const auto required_live_size =
                            static_cast<VkDeviceSize>(std::max<usize>(vertex.DataSizeOf(),
                                                                      vertex.OneSizeOf()));
                        if (buf && buf.size >= required_live_size) return true;

                        // Dynamic custom-shader meshes may grow after a pass was prepared. Keep
                        // this as a renderer-level buffer refresh mechanism for authored dynamic
                        // meshes, while first-class text is handled by TextPass and never enters
                        // CustomShaderPass as glyph helper nodes.
                        const auto required_size = GrowDynamicSuballocationSize(
                            buf ? buf.size : 0,
                            required_live_size,
                            static_cast<VkDeviceSize>(vertex.CapacitySizeOf()),
                            static_cast<VkDeviceSize>(vertex.OneSizeOf()));
                        if (required_size < required_live_size) {
                            LOG_ERROR("DynamicVertexUpload: live data exceeds capacity node='%s' "
                                      "live=%zu capacity=%zu",
                                      mesh.Material() != nullptr ? mesh.Material()->name.c_str()
                                                                 : "<unknown>",
                                      static_cast<size_t>(required_live_size),
                                      static_cast<size_t>(vertex.CapacitySizeOf()));
                            return false;
                        }
                        if (buf) {
                            dyn_buf->unallocateSubRef(buf);
                            buf = {};
                        }
                        if (!dyn_buf->allocateSubRef(required_size, buf)) {
                            return false;
                        }
                        force_dyn_upload = true;
                        return true;
                    };

                    auto release_unused_vertex_subrefs = [&]() {
                        while (vertex_bufs.size() > mesh.VertexCount()) {
                            auto& stale_buf = vertex_bufs.back();
                            if (stale_buf) dyn_buf->unallocateSubRef(stale_buf);
                            vertex_bufs.pop_back();
                        }
                    };

                    auto ensure_index_subref = [&](const wallpaper::SceneIndexArray& indice) {
                        const auto required_live_size =
                            static_cast<VkDeviceSize>(std::max<usize>(indice.DataSizeOf(),
                                                                      sizeof(uint32_t) * 6));
                        if (index_buf && index_buf.size >= required_live_size) return true;

                        const auto required_size = GrowDynamicSuballocationSize(
                            index_buf ? index_buf.size : 0,
                            required_live_size,
                            static_cast<VkDeviceSize>(indice.CapacitySizeof()),
                            static_cast<VkDeviceSize>(sizeof(uint32_t) * 6));
                        if (required_size < required_live_size) {
                            LOG_ERROR("DynamicIndexUpload: live data exceeds capacity live=%zu "
                                      "capacity=%zu",
                                      static_cast<size_t>(required_live_size),
                                      static_cast<size_t>(indice.CapacitySizeof()));
                            return false;
                        }
                        if (index_buf) {
                            dyn_buf->unallocateSubRef(index_buf);
                            index_buf = {};
                        }
                        if (!dyn_buf->allocateSubRef(required_size, index_buf)) {
                            return false;
                        }
                        force_dyn_upload = true;
                        return true;
                    };

                    release_unused_vertex_subrefs();
                    for (usize i = 0; i < mesh.VertexCount(); i++) {
                        const auto& vertex = mesh.GetVertexArray(i);
                        if (!ensure_vertex_subref(i, vertex)) {
                            mesh.SetDirty();
                            return;
                        }
                        auto& buf = vertex_bufs[i];
                        if (! dyn_buf->writeToBuf(buf,
                                                  { (uint8_t*)vertex.Data(), vertex.DataSizeOf() }))
                        {
                            mesh.SetDirty();
                            return;
                        }
                    }
                    if (mesh.IndexCount() > 0) {
                        auto& indice = mesh.GetIndexArray(0);
                        if (!ensure_index_subref(indice)) {
                            mesh.SetDirty();
                            return;
                        }
                        u32   count  = (u32)((indice.RenderDataCount() * 2) / 3);
                        draw_count   = count * 3;
                        auto& buf    = index_buf;
                        if (! dyn_buf->writeToBuf(buf,
                                                  { (uint8_t*)indice.Data(), indice.DataSizeOf() }))
                        {
                            mesh.SetDirty();
                            return;
                        }
                    } else {
                        // Dynamic non-indexed meshes are still drawable. Text effect outputs use a
                        // four-vertex triangle-strip card that is resized in place when Date/Day/
                        // Clock content changes; clearing draw_count here made the bridge and
                        // effect passes execute successfully while submitting no final composite
                        // geometry at all. The first vertex binding defines the vertex count for
                        // non-indexed draws, matching the static prepare path's draw contract.
                        draw_count = mesh.VertexCount() > 0
                            ? static_cast<u32>(mesh.GetVertexArray(0).DataSize() /
                                                mesh.GetVertexArray(0).OneSize())
                            : 0;
                        if (index_buf) {
                            dyn_buf->unallocateSubRef(index_buf);
                            index_buf = {};
                        }
                    }
                    // Clearing the pass-local bootstrap flag only after all writes succeed keeps a
                    // freshly compiled dynamic pass from getting stuck with empty GPU buffers if an
                    // earlier upload attempt bails out partway through due to an allocation/write
                    // failure. Subsequent frames will keep retrying until the first complete upload
                    // lands in the new subranges.
                    mesh.Dirty().store(false);
                    force_dyn_upload = false;
                }
            };
        }

        auto  block  = ref.blocks.front();
        auto* buf    = rr.dyn_buf;
        auto* bufref = &m_desc.ubo_buf;

        auto* node           = m_desc.node;
        auto* shader_updater = scene.shaderValueUpdater.get();
        auto& sprites        = m_desc.sprites_map;
        auto& vk_textures    = m_desc.vk_textures;

        auto* material       = mesh.Material();
        if (material != nullptr) {
            LogAudioUniformBlockLayout(m_desc, *material, block);
        }
        // Keep Star-River-style dynamic mesh uploads separate from general pass updates. Only the
        // vertex/index bytes need to move before m_dyn_buf->recordUpload(); uniform and sprite
        // updates stay in execute() so Date/Clock effect composites keep their original layout
        // timing and do not shift when the upload fix is active.
        m_desc.update_dynamic_mesh_op = update_dyn_buf_op;
        m_desc.update_op = [shader_updater,
                            block,
                            buf,
                            bufref,
                            node,
                            material,
                            &sprites,
                            &vk_textures]() {
            auto update_unf_op = [&block, buf, bufref](std::string_view       name,
                                                       wallpaper::ShaderValue value) {
                UpdateUniform(buf, *bufref, block, name, value);
            };
            if (material != nullptr) {
                WriteMaterialUniforms(buf, *bufref, block, *material);
            }
            shader_updater->UpdateUniforms(node, sprites, update_unf_op);
            // update image slot for sprites
            {
                for (auto& [i, sp] : sprites) {
                    if (i >= vk_textures.size()) continue;
                    vk_textures.at(i).active = sp.GetCurFrame().imageId;
                }
            }
        };

        auto exists_unf_op = [&block](std::string_view name) {
            return exists(block.member_map, name);
        };
        shader_updater->InitUniforms(node, exists_unf_op);

        // memset uniform buf
        buf->fillBuf(*bufref, 0, bufref->size, 0);
        WriteMaterialUniforms(buf, *bufref, block, *mesh.Material());
        m_desc.update_op();
        if (m_desc.update_dynamic_mesh_op) m_desc.update_dynamic_mesh_op();
    }

    {
        auto& sc           = scene.clearColor;
        // TextPass owns transparent clears for exact-size bridge sources. Generic custom shader
        // passes should keep the scene clear contract so image/effect behavior is no longer
        // coupled to text-specific source-pass rules.
        m_desc.clear_value = VkClearValue {
            .color = { sc[0], sc[1], sc[2], 1.0f },
        };
    }
    for (auto& tex : releaseTexs()) {
        device.tex_cache().MarkShareReady(tex);
    }
    setPrepared();
}

void CustomShaderPass::refreshResources(Scene& scene, const Device& device,
                                        RenderingResources& rr) {
    // Resource refreshes reuse the pass object. Refresh the scene pointer first so dependency
    // checks observe the current render-target table and text bridge state without logging.
    m_desc.scene = &scene;
    // Resource-only refreshes intentionally keep the compiled shader pipeline, reflected bindings,
    // and uploaded mesh/UBO allocations intact. The expensive part that changes for effect-backed
    // minute updates is the texture-cache-backed image handle set and the framebuffer that wraps
    // the resized render target. Rebinding only those pieces avoids recompiling every shader pass
    // in the scene when the clock/date text changes shape.
    if (m_desc.node != nullptr && m_desc.node->Mesh() != nullptr) {
        auto& mesh = *m_desc.node->Mesh();
        if (!mesh.Dynamic() && mesh.Dirty().load()) {
            // Resource-only refreshes were originally written for effects whose geometry never
            // changes after graph build. Refactored text effects break that assumption: runtime
            // updates now mutate the static blur/compose quads of already-compiled effect passes.
            // If we only recreate textures/framebuffers here, the pass keeps drawing the old GPU
            // vertex buffer even though the SceneMesh carries the new map-rate-adjusted quad. By
            // dropping back to the normal prepare path for dirty static meshes we force the pass
            // to re-upload its mesh data and make runtime text-effect geometry changes actually
            // visible on screen.
            destory(device, rr);
            return;
        }
    }

    const auto output_target_it = scene.renderTargets.find(m_desc.output);
    if (output_target_it == scene.renderTargets.end()) {
        LOG_ERROR("CustomShaderPassRefresh: output target not found before refresh node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }
    const auto previous_output_view = m_desc.vk_output.view;
    const auto previous_output_extent = m_desc.vk_output.extent;
    const auto desired_output_key = wallpaper::vulkan::ToTexKey(output_target_it->second);
    const bool output_extent_changed =
        previous_output_extent.width != static_cast<uint32_t>(desired_output_key.width) ||
        previous_output_extent.height != static_cast<uint32_t>(desired_output_key.height);
    if (output_extent_changed) {
        // Only an output size change requires releasing the framebuffer before TextureCache::Query.
        // Recreating every framebuffer on every resource refresh was both wasteful and unsafe for
        // unchanged `_rt_default` passes: those passes can keep their attachment while only their
        // sampled input texture changes, exactly like a particle pass consuming a live source.
        m_desc.fb.reset();
    }
    if (!RefreshCustomShaderPassTextures(scene, device, m_desc)) {
        LOG_ERROR("CustomShaderPassRefresh: texture refresh failed node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }
    const bool output_view_changed = previous_output_view != m_desc.vk_output.view;
    const bool framebuffer_missing = !m_desc.fb;
    if (framebuffer_missing || output_extent_changed || output_view_changed) {
        if (!RecreateCustomShaderPassFramebuffer(device, m_desc)) {
            setPrepared(false);
            return;
        }
    }
    if (m_desc.dyn_vertex && m_desc.update_dynamic_mesh_op != nullptr && m_desc.node != nullptr &&
        m_desc.node->Mesh() != nullptr &&
        (m_desc.force_dyn_upload || m_desc.node->Mesh()->Dirty().load())) {
        // Text-backed effect passes keep their render-graph topology stable while the final
        // source quad changes size. Uploading the dirty dynamic mesh during the resource-refresh
        // phase lets the compile-time dynamic-buffer copy include the new quad before the first
        // post-refresh draw, instead of binding a fresh suballocation that still contains old data.
        m_desc.update_dynamic_mesh_op();
    }
    for (auto& tex : releaseTexs()) {
        device.tex_cache().MarkShareReady(tex);
    }
}

void CustomShaderPass::updateBeforeUpload() {
    if (!m_desc.update_dynamic_mesh_op) return;

    if (m_desc.should_execute && !m_desc.should_execute()) {
        return;
    }

    if (m_desc.node != nullptr && !m_desc.node->LocalVisible()) {
        return;
    }

    const bool node_visible = m_desc.node == nullptr ? true : m_desc.node->Visible();
    if (m_desc.node != nullptr && ! node_visible && !m_desc.execute_when_hidden) {
        return;
    }

    // VulkanRender records the shared dynamic-buffer upload before pass execution. Updating only
    // dynamic vertex/index bytes here preserves the original execute-time uniform ordering for
    // text/effect composites while still preventing reused-source particle fields, such as
    // 3308867900's Star River layer, from drawing from stale or never-uploaded subranges.
    m_desc.update_dynamic_mesh_op();
}

void CustomShaderPass::execute(const Device&, RenderingResources& rr) {
    if (m_desc.should_execute && !m_desc.should_execute()) {
        // Runtime-gated helper passes stay in the render graph so visibility flips do not rebuild
        // framebuffer topology. Returning before uniform updates and draw submission makes the pass a
        // true no-op on frames where its fallback branch is not active.
        return;
    }

    if (m_desc.node != nullptr && !m_desc.node->LocalVisible()) {
        // execute_when_hidden is only for layer-level invisibility, such as offscreen dependency
        // sources that must keep rendering while their authored layer is hidden in the main scene.
        // Effect-local visibility is a stricter contract: a hidden effect must not run its shader
        // pass, otherwise the hidden branch would still overwrite the ping-pong output that the
        // bypass copy is responsible for preserving.
        return;
    }

    const bool node_visible = m_desc.node == nullptr ? true : m_desc.node->Visible();
    if (m_desc.node != nullptr && ! node_visible && !m_desc.execute_when_hidden) {
        return;
    }

    if (m_desc.update_op) m_desc.update_op();
    LogAudioReactivePassExecute(m_desc);
    LogAudioCompositePassExecute(m_desc);

    auto&                   cmd    = rr.command;
    auto&                   outext = m_desc.vk_output.extent;
    VkImageSubresourceRange base_srang {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = VK_REMAINING_ARRAY_LAYERS,
        .baseArrayLayer = 0,
        .layerCount     = VK_REMAINING_MIP_LEVELS,
    };
    for (usize i = 0; i < m_desc.vk_textures.size(); i++) {
        auto& slot    = m_desc.vk_textures[i];
        int   binding = m_desc.vk_tex_binding[i];
        if (binding < 0) continue;
        if (slot.slots.empty()) continue;
        auto&                 img = slot.getActive();
        VkDescriptorImageInfo desc_img { img.sampler,
                                         img.view,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet  wset {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext           = nullptr,
            .dstSet          = {},
            .dstBinding      = (uint32_t)binding,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &desc_img,
        };
        cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.layout, 0, wset);

        VkImageMemoryBarrier imb {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            // Render-target inputs are often produced by the immediately previous pass in the
            // same command buffer. That producer writes through the color-attachment path, not the
            // fragment-shader read path, so the consumer must wait on color attachment writes
            // before sampling. Without this, first-class text bridges can be drawn correctly by
            // TextPass and still appear missing when their effect/composite pass samples the
            // freshly written offscreen image.
            .srcAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = img.handle,
            .subresourceRange = base_srang,
        };

        cmd.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            imb);
    }

    if (m_desc.ubo_buf) {
        VkDescriptorBufferInfo desc_buf {
            rr.dyn_buf->gpuBuf(),
            m_desc.ubo_buf.offset,
            m_desc.ubo_buf.size,
        };
        VkWriteDescriptorSet wset {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext           = nullptr,
            .dstSet          = {},
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &desc_buf,
        };
        cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.layout, 0, wset);
    }

    VkRenderPassBeginInfo pass_begin_info {
        .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext       = nullptr,
        .renderPass  = *m_desc.pipeline.pass,
        .framebuffer = *m_desc.fb,
        .renderArea =
            VkRect2D {
                .offset = { 0, 0 },
                .extent = { outext.width, outext.height },
            },
        .clearValueCount = 1,
        .pClearValues    = &m_desc.clear_value,
    };
    cmd.BeginRenderPass(pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.handle);
    VkViewport viewport {
        .x        = 0,
        .y        = (float)outext.height,
        .width    = (float)outext.width,
        .height   = -(float)outext.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D scissor { { 0, 0 }, { outext.width, outext.height } };

    cmd.SetViewport(0, viewport);
    cmd.SetScissor(0, scissor);

    auto gpu_buf = m_desc.dyn_vertex ? rr.dyn_buf->gpuBuf() : rr.vertex_buf->gpuBuf();

    for (usize i = 0; i < m_desc.vertex_bufs.size(); i++) {
        auto& buf = m_desc.vertex_bufs[i];
        cmd.BindVertexBuffers((u32)i, 1, &gpu_buf, &buf.offset);
    }
    if (m_desc.index_buf) {
        cmd.BindIndexBuffer(gpu_buf, m_desc.index_buf.offset, VK_INDEX_TYPE_UINT16);
        cmd.DrawIndexed(m_desc.draw_count, 1, 0, 0, 0);
    } else {
        cmd.Draw(m_desc.draw_count, 1, 0, 0);
    }

    cmd.EndRenderPass();
}

void CustomShaderPass::destory(const Device&, RenderingResources& rr) {
    m_desc.update_op = {};
    m_desc.update_dynamic_mesh_op = {};
    {
        auto& buf = m_desc.dyn_vertex ? rr.dyn_buf : rr.vertex_buf;
        for (auto& bufref : m_desc.vertex_bufs) {
            buf->unallocateSubRef(bufref);
        }
    }
    rr.dyn_buf->unallocateSubRef(m_desc.ubo_buf);
    // Resource-only render-graph refreshes keep the pass objects alive inside the graph, so the
    // only signal that forces a new Vulkan preparation round is the prepared flag. Leaving it set
    // after we release buffer suballocations lets the next frame reuse stale descriptor/image
    // bindings that still point at destroyed cache entries, which is exactly the kind of
    // use-after-destroy that can crash inside vkCmdPushDescriptorSetKHR on the render thread.
    setPrepared(false);
}

void CustomShaderPass::setDescTex(u32 index, std::string_view tex_key) {
    assert(index < m_desc.textures.size());
    if (index >= m_desc.textures.size()) return;
    m_desc.textures[index] = tex_key;
}

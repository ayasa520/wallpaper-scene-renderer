#include "ShaderDrawCore.hpp"
#include "Scene/Scene.h"
#include "Scene/SceneShader.h"
#include "Scene/SceneTextPrimitive.h"

#include "SpecTexs.hpp"
#include "Vulkan/Shader.hpp"
#include "Vulkan/VideoTextureCache.hpp"
#include "vvk/vma_wrapper.hpp"
#include "Utils/Logging.h"
#include "Utils/AutoDeletor.hpp"
#include "Resource.hpp"
#include "RenderCommandTrace.hpp"
#include "RenderTargetOps.hpp"
#include "PassCommon.hpp"
#include "Msaa.hpp"
#include "Interface/IImageParser.h"

#include "Core/ArrayHelper.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace wallpaper::vulkan;

std::string wallpaper::vulkan::ShaderDrawPipelineCompatibilityKey(
    VkAttachmentLoadOp load_op, bool model_pass, VkAttachmentLoadOp depth_load_op,
    const ShaderDrawAttachmentDescription& attachment, VkSampleCountFlagBits samples,
    bool resolve_msaa) {
    // Keep this key limited to Vulkan render-pass compatibility. GraphicsPipeline adds the shader,
    // descriptor, vertex-input, blend, depth, and topology state to the final cache key, matching
    // the descriptor-driven PSO caches used by larger renderers instead of tying immutable PSOs to
    // a transient layer/pass identity.
    return "ShaderDraw|format=rgba8|final=shader-read|load=" +
           std::to_string(static_cast<int>(load_op)) +
           "|model=" + (model_pass ? std::string("1") : std::string("0")) +
           "|depth-format=d32|depth-load=" + std::to_string(static_cast<int>(depth_load_op)) +
           "|extra-tag=" + std::string(attachment.cache_tag) +
           "|extra-format=" + std::to_string(static_cast<int>(attachment.format)) +
           "|extra-depth-load=" +
           std::to_string(static_cast<int>(attachment.depth_load_op)) +
           "|extra-stencil-load=" +
           std::to_string(static_cast<int>(attachment.stencil_load_op)) +
           "|samples=" + std::to_string(static_cast<int>(samples)) +
           "|resolve=" + (resolve_msaa ? std::string("1") : std::string("0"));
}

namespace
{

void PopulateTextureBindingsFromReflection(wallpaper::vulkan::ShaderDrawData& desc,
                                           const wallpaper::vulkan::ShaderReflected& ref,
                                           size_t texture_count) {
    desc.vk_tex_binding.clear();
    desc.vk_tex_binding.reserve(texture_count);
    for (size_t i = 0; i < texture_count; i++) {
        wallpaper::i32 binding { -1 };
        if (i < wallpaper::WE_GLTEX_NAMES.size() &&
            wallpaper::exists(ref.binding_map, wallpaper::WE_GLTEX_NAMES[i])) {
            binding = static_cast<wallpaper::i32>(
                ref.binding_map.at(wallpaper::WE_GLTEX_NAMES[i]).binding);
        }
        desc.vk_tex_binding.push_back(binding);
    }
}

// Mesh primitive and index presence are the complete input-assembly contract. Warmup and real
// preparation must derive the same value so their pipeline keys match the pipeline actually used.
VkPrimitiveTopology ToTopology(const wallpaper::SceneMesh& mesh) {
    switch (mesh.Primitive()) {
    case wallpaper::MeshPrimitive::POINT: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case wallpaper::MeshPrimitive::TRIANGLE:
        return mesh.IndexCount() > 0 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                     : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    }
    assert(false);
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

} // namespace

ShaderDrawCore::ShaderDrawCore(const ShaderDrawRequest& desc)
    : m_draw_identity(desc.draw.Valid() ? desc.draw.RenderIdentity() : 0) {
    // The render graph builder already classifies hidden offscreen dependencies and gives passes
    // a live scene pointer for diagnostics. Preserve that prepared intent here; dropping these
    // fields forced text/effect passes back through generic visibility and null-scene behavior.
    m_desc.scene               = desc.scene;
    m_desc.draw                = desc.draw;
    m_desc.layer_id            = desc.layer_id;
    m_desc.execute_when_hidden = desc.execute_when_hidden;
    m_desc.should_execute      = desc.should_execute;
    m_desc.textures            = desc.textures;
    m_desc.output              = desc.output;
    m_desc.resolved_scene_color = desc.resolved_scene_color;
    m_desc.alpha_write_policy  = desc.alpha_write_policy;
    m_desc.blend_override      = desc.blend_override;
    m_desc.depth_test_override = desc.depth_test_override;
    m_desc.depth_write_override = desc.depth_write_override;
    m_desc.destination_alpha_override = desc.destination_alpha_override;
    m_desc.premultiplied_source_blend = desc.premultiplied_source_blend;
    m_desc.clear_before_draw   = desc.clear_before_draw;
    m_desc.camera_override     = desc.camera_override;
    m_desc.use_active_camera_for_uniforms = desc.use_active_camera_for_uniforms;
    m_desc.use_active_camera_for_parallax = desc.use_active_camera_for_parallax;
    m_desc.suppress_destination_parallax = desc.suppress_destination_parallax;
    m_desc.model_space = desc.model_space;
    m_desc.reflection_pass = desc.reflection_pass;
    m_desc.reflection_raster = desc.reflection_raster;
    m_desc.reflection_snapshot = desc.reflection_snapshot;
    m_desc.effect_snapshot_camera = desc.effect_snapshot_camera;
    m_desc.sprites_map         = desc.sprites_map;
    m_desc.model_pass          = desc.model_pass;
    m_desc.shared_depth        = desc.shared_depth;
    m_desc.depth_test          = desc.depth_test;
    m_desc.depth_write         = desc.depth_write;
    m_desc.clear_depth         = desc.clear_depth;
    m_desc.depth_clear         = desc.depth_clear;
    if (desc.draw.Valid() && desc.draw.Mesh() != nullptr &&
        desc.draw.Mesh()->Material() != nullptr) {
        m_material_blend = desc.draw.Mesh()->Material()->blenmode;
        m_material_cull = desc.draw.Mesh()->Material()->cullMode;
        m_material_alpha_writing = desc.draw.Mesh()->Material()->alphaWriting;
    }
};

std::string ShaderDrawCore::residencyKey(std::string_view pass_kind) const {
    return std::string(pass_kind) + "|layer=" + std::to_string(m_desc.layer_id) +
           "|node=" + std::to_string(m_draw_identity) +
           "|output=" + m_desc.output +
           "|reflection=" + (m_desc.reflection_pass ? "1" : "0");
}

std::string ShaderDrawCore::profileName(std::string_view pass_kind) const {
    // Frame-profiling identity: the residency key plus the authored node name, so aggregated GPU
    // timings can be attributed to scene content without cross-referencing node pointers.
    std::string name = residencyKey(pass_kind);
    if (m_desc.draw.Valid() && ! m_desc.draw.Name().empty()) {
        name += "|name=" + m_desc.draw.Name();
    }
    return name;
}

static int IntendedShaderDrawSampleCount(const ShaderDrawData& desc) {
    if (desc.scene != nullptr &&
        ShaderDrawCanUseMsaa(*desc.scene, desc.output, desc.resolved_scene_color)) {
        return std::max(1, desc.scene->MsaaSampleCount());
    }
    return 1;
}

bool ShaderDrawCore::canReuseForResidency(const ShaderDrawCore& next) const {
    // A prepared pass may be reused only when its immutable GPU contract is the same. Runtime
    // visibility gates and descriptor texture keys can be refreshed in place, but changing model
    // depth state, material blending/culling or the owning SceneNode requires a new pipeline.
    const int this_samples = static_cast<int>(m_desc.sample_count);
    const int next_samples = IntendedShaderDrawSampleCount(next.m_desc);
    return m_desc.layer_id == next.m_desc.layer_id &&
           m_draw_identity == next.m_draw_identity &&
           m_desc.output == next.m_desc.output &&
           m_desc.resolved_scene_color == next.m_desc.resolved_scene_color &&
           m_desc.execute_when_hidden == next.m_desc.execute_when_hidden &&
           m_desc.model_pass == next.m_desc.model_pass &&
           m_desc.shared_depth == next.m_desc.shared_depth &&
           m_desc.depth_test == next.m_desc.depth_test &&
           m_desc.depth_write == next.m_desc.depth_write &&
           m_desc.clear_depth == next.m_desc.clear_depth &&
           m_desc.depth_clear == next.m_desc.depth_clear &&
           // Alpha policy changes the prepared pipeline's color write mask, blend operation, and
           // factors. Reusing a pass across that boundary would keep stale composition coverage.
           m_desc.alpha_write_policy == next.m_desc.alpha_write_policy &&
           m_desc.destination_alpha_override == next.m_desc.destination_alpha_override &&
           // Final destination state and MAX coverage do not consume the material enum. A live
           // edit still refreshes graph descriptions, but these invocations keep their pipeline.
           (m_desc.destination_alpha_override || m_desc.alpha_write_policy == AlphaWritePolicy::Max ||
            m_material_alpha_writing == next.m_material_alpha_writing) &&
           m_desc.blend_override == next.m_desc.blend_override &&
           // Compare the captured effective value. A changed raw material blend requires a
           // new pipeline unless both invocations still select the same owner-final override.
           m_desc.blend_override.value_or(m_material_blend) ==
               next.m_desc.blend_override.value_or(next.m_material_blend) &&
           m_material_cull == next.m_material_cull &&
           m_desc.premultiplied_source_blend ==
               next.m_desc.premultiplied_source_blend &&
           m_desc.clear_before_draw == next.m_desc.clear_before_draw &&
           // The uniform update lambda captures the pass camera route. Treat it as part of the
           // prepared uniform contract so a source-space composition route cannot reuse a screen-space
           // publisher after a topology rebuild.
           m_desc.camera_override == next.m_desc.camera_override &&
           m_desc.use_active_camera_for_uniforms ==
               next.m_desc.use_active_camera_for_uniforms &&
           m_desc.use_active_camera_for_parallax ==
               next.m_desc.use_active_camera_for_parallax &&
           m_desc.suppress_destination_parallax ==
               next.m_desc.suppress_destination_parallax &&
           m_desc.model_space == next.m_desc.model_space &&
           m_desc.reflection_pass == next.m_desc.reflection_pass &&
           m_desc.reflection_raster == next.m_desc.reflection_raster &&
           m_desc.reflection_snapshot == next.m_desc.reflection_snapshot &&
           m_desc.effect_snapshot_camera == next.m_desc.effect_snapshot_camera &&
           this_samples == next_samples &&
           ! m_desc.resolve_msaa &&
           m_desc.textures.size() == next.m_desc.textures.size();
}

void ShaderDrawCore::absorbResidencyGraphState(const ShaderDrawCore& next) {
    // Render-graph diffing keeps this pass's expensive Vulkan objects alive while replacing only
    // the declarative state that can change as layers move between hidden and visible residency.
    // Texture handles are rebound by refreshResources()/prepare(), and the runtime gate must follow
    // the newly built graph so effect bypass/final-composite branches stay correct.
    m_desc.scene          = next.m_desc.scene;
    m_desc.draw           = next.m_desc.draw;
    m_draw_identity       = next.m_draw_identity;
    m_material_blend      = next.m_material_blend;
    m_material_cull       = next.m_material_cull;
    m_material_alpha_writing = next.m_material_alpha_writing;
    m_desc.layer_id       = next.m_desc.layer_id;
    m_desc.should_execute = next.m_desc.should_execute;
    m_desc.textures       = next.m_desc.textures;
    m_desc.output         = next.m_desc.output;
    m_desc.alpha_write_policy = next.m_desc.alpha_write_policy;
    m_desc.destination_alpha_override = next.m_desc.destination_alpha_override;
    m_desc.blend_override = next.m_desc.blend_override;
    m_desc.depth_test_override = next.m_desc.depth_test_override;
    m_desc.depth_write_override = next.m_desc.depth_write_override;
    m_desc.premultiplied_source_blend = next.m_desc.premultiplied_source_blend;
    m_desc.clear_before_draw = next.m_desc.clear_before_draw;
    m_desc.camera_override = next.m_desc.camera_override;
    m_desc.use_active_camera_for_uniforms = next.m_desc.use_active_camera_for_uniforms;
    m_desc.use_active_camera_for_parallax = next.m_desc.use_active_camera_for_parallax;
    m_desc.suppress_destination_parallax = next.m_desc.suppress_destination_parallax;
    m_desc.model_space = next.m_desc.model_space;
    m_desc.reflection_pass = next.m_desc.reflection_pass;
    m_desc.reflection_raster = next.m_desc.reflection_raster;
    m_desc.reflection_snapshot = next.m_desc.reflection_snapshot;
    m_desc.effect_snapshot_camera = next.m_desc.effect_snapshot_camera;
    m_desc.sprites_map    = next.m_desc.sprites_map;
}

bool ShaderDrawCore::referencesRenderTarget(std::string_view render_target) const {
    // Custom shader passes are affected when either their output framebuffer is the dirty target or
    // one of their descriptor inputs samples it. This lets a resized text bridge update the exact
    // effect chain that consumes it instead of refreshing every other shader in the wallpaper.
    if (m_desc.output == render_target) return true;
    return referencesImportedTexture(render_target);
}

bool ShaderDrawCore::referencesImportedTexture(std::string_view texture_key) const {
    for (const auto& texture : m_desc.textures) {
        if (texture == texture_key) return true;
    }
    if (m_extension != nullptr && m_desc.draw.Valid() && m_desc.draw.Mesh() != nullptr) {
        for (const auto texture : m_extension->resourceTextures(*m_desc.draw.Mesh())) {
            if (texture == texture_key) return true;
        }
    }
    return false;
}

std::optional<vvk::RenderPass> wallpaper::vulkan::CreateShaderDrawRenderPass(
    const vvk::Device& device, VkFormat format, VkAttachmentLoadOp loadOp,
    VkImageLayout finalLayout, const ShaderDrawAttachmentDescription& extra_attachment,
    VkSampleCountFlagBits samples, bool resolve_msaa) {
    const bool store_ms = samples > VK_SAMPLE_COUNT_1_BIT;
    const bool msaa     = store_ms && resolve_msaa;
    const VkSampleCountFlagBits color_samples =
        store_ms ? samples : VK_SAMPLE_COUNT_1_BIT;

    VkAttachmentDescription attachment {
        .format         = format,
        .samples        = color_samples,
        .loadOp         = loadOp,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = store_ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : finalLayout,
    };

    if (loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) {
        attachment.initialLayout = store_ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkAttachmentReference attachment_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentDescription depth_attachment {
        .format         = extra_attachment.format,
        .samples        = color_samples,
        .loadOp         = extra_attachment.depth_load_op,
        .storeOp        = extra_attachment.depth_store_op,
        .stencilLoadOp  = extra_attachment.stencil_load_op,
        .stencilStoreOp = extra_attachment.stencil_store_op,
        .initialLayout  = extra_attachment.initial_layout,
        .finalLayout    = extra_attachment.final_layout,
    };
    VkAttachmentReference depth_attachment_ref {
        .attachment = 1,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentDescription resolve_attachment {
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = finalLayout,
    };
    const uint32_t resolve_index = extra_attachment.enabled() ? 2u : 1u;
    VkAttachmentReference resolve_ref {
        .attachment = resolve_index,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    std::array<VkAttachmentDescription, 3> attachments {
        attachment,
        extra_attachment.enabled() ? depth_attachment : resolve_attachment,
        resolve_attachment,
    };

    VkSubpassDescription subpass {
        .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount    = 1,
        .pColorAttachments       = &attachment_ref,
        .pResolveAttachments     = msaa ? &resolve_ref : nullptr,
        .pDepthStencilAttachment = extra_attachment.enabled() ? &depth_attachment_ref : nullptr,
    };

    // Rebound attachments can still contain writes from a preceding pass, even
    // when this pass discards their contents with UNDEFINED/CLEAR. In particular,
    // the per-light Back depth is stored in late fragment tests. Make those writes
    // available before the next automatic layout transition, not just before reads.
    VkSubpassDependency dependency {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    uint32_t attachment_count = 1;
    if (extra_attachment.enabled()) attachment_count++;
    if (msaa) attachment_count++;

    VkRenderPassCreateInfo creatinfo {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachment_count,
        .pAttachments    = attachments.data(),
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

void wallpaper::vulkan::UpdateShaderDrawUniform(StagingBuffer* buf,
                                                const StagingBufferRef& bufref,
                                                const ShaderReflected::Block& block,
                                                std::string_view name,
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

    if (reflectedSize < packedSize) {
        // Some first-party model shaders declare compact matrix uniforms such as `mat3
        // g_ModelMatrix`, while the runtime updater naturally owns a full 4x4 scene transform.
        // Writing the whole packed matrix would overflow the reflected member slot and corrupt the
        // following uniforms. Clamp only at the upload boundary so 2D shader generation and the
        // shared ShaderValue representation do not need a model-specific branch.
        buf->writeToBuf(bufref, value_u8.subspan(0, reflectedSize), offset);
        return;
    }

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
                std::span<uint8_t> elem { reinterpret_cast<uint8_t*>(
                                              const_cast<ShaderValue::value_type*>(&value[i])),
                                          sizeof(ShaderValue::value_type) };
                buf->writeToBuf(bufref, elem, offset + i * stride);
            }
            return;
        }
    }

    buf->writeToBuf(bufref, value_u8, offset);
}

static void WriteMaterialUniforms(StagingBuffer* buf, const StagingBufferRef& bufref,
                                  const ShaderReflected::Block&   block,
                                  const wallpaper::SceneMaterial& material) {
    auto write_values = [&](const auto& values) {
        for (const auto& [name, value] : values) {
            if (! wallpaper::exists(block.member_map, name)) continue;
            UpdateShaderDrawUniform(buf, bufref, block, name, value);
        }
    };

    if (material.customShader.shader != nullptr) {
        write_values(material.customShader.shader->default_uniforms);
    }
    write_values(material.customShader.constValues);
}

constexpr VkDeviceSize kInitialDynamicSuballocationSize = 64 * 1024;
constexpr VkDeviceSize kDynamicIndexQuadFloorSize       = sizeof(uint16_t) * 6;

VkDeviceSize InitialDynamicSuballocationSize(VkDeviceSize capacity, VkDeviceSize live_size,
                                             VkDeviceSize element_size) {
    if (capacity == 0) return 0;

    const VkDeviceSize non_empty_element = std::max<VkDeviceSize>(element_size, 1);
    const VkDeviceSize required_live     = std::max<VkDeviceSize>(live_size, non_empty_element);
    const VkDeviceSize bootstrap         = std::min<VkDeviceSize>(
        capacity, std::max<VkDeviceSize>(kInitialDynamicSuballocationSize, non_empty_element));

    // Dynamic particle meshes often advertise a very large theoretical capacity while starting
    // with zero live vertices. Reserve only a small bootstrap range up front, but never choose a
    // range smaller than the data that is already live and must be uploaded immediately.
    return std::min<VkDeviceSize>(capacity, std::max(required_live, bootstrap));
}

VkDeviceSize DynamicVertexUploadSize(const wallpaper::SceneVertexArray& vertex) {
    // Vertex arrays expose both live bytes and authored capacity. Use the live byte count for the
    // first upload so character-rain style particle systems do not reserve their entire theoretical
    // maximum before any spawned particles exist.
    return InitialDynamicSuballocationSize(static_cast<VkDeviceSize>(vertex.CapacitySizeOf()),
                                           static_cast<VkDeviceSize>(vertex.DataSizeOf()),
                                           static_cast<VkDeviceSize>(vertex.OneSizeOf()));
}

VkDeviceSize DynamicIndexUploadSize(const wallpaper::SceneIndexArray& indice) {
    // Index buffers follow the same bootstrap rule as vertices, but CustomShaderPass binds them as
    // VK_INDEX_TYPE_UINT16 at draw time. SceneIndexArray stores both 32-bit model indices and
    // packed 16-bit particle indices behind the same byte-count API, so the non-empty dynamic floor
    // must match the GPU binding size. The effect-dependency route added for private image
    // composites can expose one-quad particle helpers with only 12 bytes of authored capacity;
    // using a 24-byte uint32_t floor makes those valid helpers fail before their first dynamic
    // upload.
    return InitialDynamicSuballocationSize(static_cast<VkDeviceSize>(indice.CapacitySizeof()),
                                           static_cast<VkDeviceSize>(indice.DataSizeOf()),
                                           kDynamicIndexQuadFloorSize);
}

VkCullModeFlags ToVkCullMode(wallpaper::SceneCullMode mode) {
    switch (mode) {
    case wallpaper::SceneCullMode::None: return VK_CULL_MODE_NONE;
    case wallpaper::SceneCullMode::Back: return VK_CULL_MODE_BACK_BIT;
    case wallpaper::SceneCullMode::Front: return VK_CULL_MODE_FRONT_BIT;
    }
    return VK_CULL_MODE_NONE;
}

bool ShouldWriteCustomShaderAlpha(const wallpaper::SceneMaterial& material,
                                  std::string_view                camera_name,
                                  wallpaper::AlphaWritePolicy      alpha_write_policy,
                                  std::string_view                output,
                                  bool destination_alpha_override) {
    // Accumulated composition coverage is independent of ordinary material alpha writes. A
    // disabled material must not suppress MAX coverage. The selected final material restores
    // destination alpha state after material selection: without that coverage scope it preserves
    // destination alpha, even when an offscreen destination supplies a private camera.
    if (alpha_write_policy == wallpaper::AlphaWritePolicy::Max) return true;
    if (destination_alpha_override) return false;
    switch (material.alphaWriting) {
    case wallpaper::SceneAlphaWriting::Enabled: return true;
    case wallpaper::SceneAlphaWriting::Disabled: return false;
    case wallpaper::SceneAlphaWriting::Default: break;
    }

    const bool is_model_pass = material.modelRenderState.has_value();
    // Model shaders may output non-opaque alpha for their own material math even when the authored
    // object is visually opaque. Allowing that alpha into `_rt_default` makes FinPass present a
    // translucent frame and visually crushes the lighting. Keep the RGB blend factors intact for
    // translucent model materials, but preserve the target alpha just like global 2D passes.
    // Offscreen targets such as `_rt_volumetricsLightBuffer` must keep the shader alpha:
    // volumetricsfront writes a=1 and the additive combine (SRC_ALPHA, ONE) multiplies by the
    // sampled LightBuffer alpha. Suppressing A left the buffer at the clear value 0 and the
    // passthrough combine added nothing.
    if (is_model_pass && output == wallpaper::SpecTex_Default) return false;
    // Volumetric nodes have an empty camera name, so the 2D compositor gate below would still
    // drop A. Offscreen model targets must store the shader alpha: the additive combine
    // (SRC_ALPHA, ONE) samples LightBuffer.a, and volumetricsfront writes a=1.
    if (is_model_pass && output != wallpaper::SpecTex_Default) return true;

    // Explicit compositor policies opt in to camera-less alpha writes. Authored effect shaders such
    // as auto_sway otherwise retain the historical camera-derived mask because their helper regions
    // can legally output alpha=1 without representing final layer coverage.
    if (alpha_write_policy != wallpaper::AlphaWritePolicy::Preserve) return true;

    return ! (camera_name.empty() || wallpaper::sstart_with(camera_name, "global"));
}

std::string_view EffectiveCustomShaderCamera(
    const wallpaper::vulkan::ShaderDrawData& desc) {
    if (! desc.camera_override.empty()) return desc.camera_override;
    return desc.draw.Valid() ? std::string_view(desc.draw.Camera()) : std::string_view {};
}

void ApplyAlphaWritePolicy(wallpaper::AlphaWritePolicy                policy,
                           bool                                       writes_alpha,
                           VkPipelineColorBlendAttachmentState&       blend_state) {
    if (!writes_alpha || policy == wallpaper::AlphaWritePolicy::Preserve) return;

    // Alpha coverage is independent from the authored RGB equation. SourceOver publishes a
    // resolved private silhouette, while Max matches Wallpaper Engine's copybackground=false
    // attachment state: later transparent fragments may expand coverage but never reduce coverage
    // already written by an earlier attachment.
    if (!blend_state.blendEnable) {
        blend_state.blendEnable = true;
        blend_state.colorBlendOp = VK_BLEND_OP_ADD;
        blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    }
    blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    if (policy == wallpaper::AlphaWritePolicy::Max) {
        blend_state.alphaBlendOp = VK_BLEND_OP_MAX;
        blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        return;
    }
    blend_state.alphaBlendOp = VK_BLEND_OP_ADD;
    blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
}

void ApplyPremultipliedSourceBlend(
    const wallpaper::vulkan::ShaderDrawData& desc,
    bool                                             writes_alpha,
    VkPipelineColorBlendAttachmentState&             blend_state) {
    if (! desc.premultiplied_source_blend || ! blend_state.blendEnable) return;

    // The private layer-surface puppet pass has already composited straight-alpha texture samples
    // into a transparent render target. Its sampled RGB is premultiplied, while alpha remains the
    // correct source-over coverage. Keep the authored destination factor from the material blend
    // mode, but consume RGB as premultiplied color so the final publisher does not fade animated
    // eye/eyelid pixels before the puppet mesh actually covers them.
    blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    if (writes_alpha) {
        blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    }
}

void ApplyModelPassDesc(const wallpaper::SceneMaterial&            material,
                        wallpaper::vulkan::ShaderDrawData& desc,
                        VkAttachmentLoadOp&                        load_op) {
    const auto& model_state = material.modelRenderState;
    if (! model_state.has_value()) return;

    desc.model_pass    = true;
    desc.depth_clear   = model_state->depthClear;
    // Model passes carry an explicit attachment-load policy. The parser chooses it per output
    // target, so offscreen buffers are cleared once per frame before later chunks load and
    // composite into the same image.
    switch (model_state->colorLoadMode) {
    case wallpaper::SceneModelColorLoadMode::DontCare: break;
    case wallpaper::SceneModelColorLoadMode::Load: load_op = VK_ATTACHMENT_LOAD_OP_LOAD; break;
    case wallpaper::SceneModelColorLoadMode::Clear: load_op = VK_ATTACHMENT_LOAD_OP_CLEAR; break;
    }
    // Stage initialization is independent of the first visible chunk. Every reflected draw loads
    // the color/depth already cleared by that stage, even when this shared material is the first
    // model in the ordinary scene walk.
    if (desc.output == wallpaper::SpecTex_Reflection) load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
}

std::string_view ModelColorLoadModeName(wallpaper::SceneModelColorLoadMode mode) {
    switch (mode) {
    case wallpaper::SceneModelColorLoadMode::DontCare: return "dont-care";
    case wallpaper::SceneModelColorLoadMode::Load: return "load";
    case wallpaper::SceneModelColorLoadMode::Clear: return "clear";
    }
    return "unknown";
}

VkClearValue BuildCustomShaderClearValue(const wallpaper::Scene&         scene,
                                         const wallpaper::SceneMaterial& material,
                                         bool                            transparent_clear) {
    if (transparent_clear) {
        return VkClearValue {
            .color = { 0.0f, 0.0f, 0.0f, 0.0f },
        };
    }

    if (material.modelRenderState.has_value() &&
        material.modelRenderState->colorLoadMode == wallpaper::SceneModelColorLoadMode::Clear) {
        // Model-only offscreen targets are sampled as textures by later passes. Transparent black
        // is the neutral clear value for those buffers: uncovered pixels contribute no stale color,
        // no alpha, and no previous-frame reflection when the current model geometry shrinks.
        return VkClearValue {
            .color = { 0.0f, 0.0f, 0.0f, 0.0f },
        };
    }

    auto& sc = scene.clearColor;
    // Non-model and main-target custom shader passes retain the existing scene clear color
    // contract. Keeping this branch shared avoids changing ordinary image/effect behavior while
    // still letting model state opt into transparent offscreen clears explicitly.
    return VkClearValue {
        .color = { sc[0], sc[1], sc[2], 1.0f },
    };
}

void ApplyExplicitClearPolicy(const wallpaper::vulkan::ShaderDrawData& desc,
                              const wallpaper::SceneMaterial&                  material,
                              VkAttachmentLoadOp&                              load_op) {
    if (!desc.clear_before_draw) return;

    load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    LOG_INFO("CustomShaderExplicitClearPolicy: layer=%d node='%s' material='%s' "
             "output='%s'",
             desc.layer_id,
             desc.draw.Valid() ? desc.draw.Name().c_str() : "",
             material.name.c_str(),
             desc.output.c_str());
}

ShaderDrawRenderState wallpaper::vulkan::BuildShaderDrawRenderState(
    const wallpaper::SceneMaterial& material, wallpaper::vulkan::ShaderDrawData& desc) {
    ShaderDrawRenderState state;
    VkColorComponentFlags   color_mask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    const auto camera_name = EffectiveCustomShaderCamera(desc);
    const bool writes_alpha =
        ShouldWriteCustomShaderAlpha(material, camera_name, desc.alpha_write_policy, desc.output,
                                     desc.destination_alpha_override);

    if (writes_alpha) color_mask |= VK_COLOR_COMPONENT_A_BIT;
    state.color_blend.colorWriteMask = color_mask;

    const auto blend_mode = desc.blend_override.value_or(material.blenmode);
    SetBlend(blend_mode, state.color_blend);
    ApplyPremultipliedSourceBlend(desc, writes_alpha, state.color_blend);
    ApplyAlphaWritePolicy(desc.alpha_write_policy, writes_alpha, state.color_blend);
    desc.blending = state.color_blend.blendEnable;

    if (wallpaper::diagnostics::Options().trace_material_state) {
        // Capture both persistent and effective values at pipeline construction. Uniform readback
        // alone cannot show whether a scoped owner override reached the actual raster pipeline.
        LOG_INFO("SceneMaterialRasterState: layer=%d node='%s' material='%s' output='%s' "
                 "stored-blend=%d override-blend=%d effective-blend=%d samples=%u "
                 "coverage=%s blend=%u rgb=[%u %u %u] alpha=[%u %u %u] mask=%u",
                 desc.layer_id, desc.draw.Valid() ? desc.draw.Name().c_str() : "",
                 material.name.c_str(), desc.output.c_str(), static_cast<int>(material.blenmode),
                 desc.blend_override ? static_cast<int>(*desc.blend_override) : -1,
                 static_cast<int>(blend_mode), static_cast<unsigned>(desc.sample_count),
                 desc.alpha_to_coverage && desc.sample_count > VK_SAMPLE_COUNT_1_BIT
                     ? "true" : "false",
                 state.color_blend.blendEnable, state.color_blend.srcColorBlendFactor,
                 state.color_blend.dstColorBlendFactor, state.color_blend.colorBlendOp,
                 state.color_blend.srcAlphaBlendFactor, state.color_blend.dstAlphaBlendFactor,
                 state.color_blend.alphaBlendOp, state.color_blend.colorWriteMask);
    }

    SetAttachmentLoadOp(blend_mode, state.color_load_op);
    ApplyModelPassDesc(material, desc, state.color_load_op);
    ApplyExplicitClearPolicy(desc, material, state.color_load_op);
    if ((!writes_alpha || desc.sample_count > VK_SAMPLE_COUNT_1_BIT ||
         desc.output == wallpaper::SpecTex_Default || desc.output == wallpaper::SpecTex_Reflection) &&
        state.color_load_op == VK_ATTACHMENT_LOAD_OP_DONT_CARE) {
        // Blending disabled does not mean a draw covers every target pixel. Scene destinations
        // must LOAD the stage clear or retained history, including alpha and uncovered MSAA
        // samples; only the stage may clear. An RGB-only draw also needs LOAD on a private
        // target: DONT_CARE would discard the previous alpha before the write mask preserves it.
        state.color_load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    if (wallpaper::diagnostics::Options().trace_material_state) {
        LOG_INFO("SceneMaterialAlphaState: layer=%d node='%s' material='%s' output='%s' "
                 "stored-alpha=%d destination-override=%s coverage-policy=%d writes-alpha=%s "
                 "mask=%u color-load=%d",
                 desc.layer_id, desc.draw.Valid() ? desc.draw.Name().c_str() : "",
                 material.name.c_str(), desc.output.c_str(), static_cast<int>(material.alphaWriting),
                 desc.destination_alpha_override ? "true" : "false",
                 static_cast<int>(desc.alpha_write_policy), writes_alpha ? "true" : "false",
                 state.color_blend.colorWriteMask, static_cast<int>(state.color_load_op));
    }
    return state;
}

void wallpaper::vulkan::ApplyShaderDrawMaterialPipelineState(const wallpaper::SceneMaterial& material,
                                const wallpaper::vulkan::ShaderDrawData& desc,
                                GraphicsPipeline& pipeline) {
    // Source, intermediate and final effect draws all consume the material's own cull state.
    // An owner-final blend override does not replace it. Ordinary effect materials need no depth
    // attachment to select one-sided drawing.
    // The reflected destination and projection-Y inversion cancel under the top-down viewport;
    // neither reflection nor an authored owner scale rewrites the material's front-face rule.
    pipeline.raster.cullMode = ToVkCullMode(material.cullMode);
    if (wallpaper::diagnostics::Options().trace_material_state) {
        LOG_INFO("SceneMaterialCullState: layer=%d node='%s' material='%s' output='%s' "
                 "stored-cull=%d cull=%u front-face=%u model=%s reflection-raster=%s",
                 desc.layer_id, desc.draw.Valid() ? desc.draw.Name().c_str() : "",
                 material.name.c_str(), desc.output.c_str(), static_cast<int>(material.cullMode),
                 static_cast<unsigned>(pipeline.raster.cullMode),
                 static_cast<unsigned>(pipeline.raster.frontFace),
                 material.modelRenderState.has_value() ? "true" : "false",
                 desc.reflection_raster ? "true" : "false");
    }
    // Effective depth is captured by the graph after target and owner-state selection. The same
    // booleans drive ordinary and model pipelines; raw readback and color-only targets remain
    // independent. Test-disabled and translucent/additive draws never write scene depth.
    pipeline.depth.depthTestEnable       = desc.depth_test;
    pipeline.depth.depthWriteEnable      = desc.depth_write;
    // Reversed depth: the only compare mode is GREATER against a 0-cleared buffer.
    pipeline.depth.depthCompareOp        = VK_COMPARE_OP_GREATER;
    pipeline.depth.depthBoundsTestEnable = false;
    pipeline.depth.stencilTestEnable     = false;
    if (wallpaper::diagnostics::Options().trace_material_state) {
        LOG_INFO("SceneMaterialDepthState: layer=%d node='%s' material='%s' output='%s' "
                 "stored-test=%s stored-write=%s override-test=%d override-write=%d "
                 "resolved-scene-color=%s shared-depth=%s effective-test=%s effective-write=%s clear=%s",
                 desc.layer_id, desc.draw.Valid() ? desc.draw.Name().c_str() : "",
                 material.name.c_str(), desc.output.c_str(),
                 material.depthTest ? "true" : "false", material.depthWrite ? "true" : "false",
                 desc.depth_test_override ? static_cast<int>(*desc.depth_test_override) : -1,
                 desc.depth_write_override ? static_cast<int>(*desc.depth_write_override) : -1,
                 desc.resolved_scene_color ? "true" : "false",
                 desc.shared_depth ? "true" : "false", desc.depth_test ? "true" : "false",
                 desc.depth_write ? "true" : "false", desc.clear_depth ? "true" : "false");
    }
    const auto& model_state = material.modelRenderState;
    if (! model_state.has_value()) return;
    LOG_INFO("ModelRenderStateBind: node='%s' shader='%s' output='%s' color-load=%s "
             "reflection-pass=%s depth-test=%s depth-write=%s depth-clear=%s "
             "depth-compare=%s depth-clear-z=%.3f cull=%u",
             desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
             material.customShader.shader != nullptr ? material.customShader.shader->name.c_str()
                                                     : "<null>",
             desc.output.c_str(),
             ModelColorLoadModeName(model_state->colorLoadMode).data(),
             desc.reflection_pass ? "true" : "false",
             desc.depth_test ? "true" : "false",
             desc.depth_write ? "true" : "false",
             desc.clear_depth ? "true" : "false",
             "greater",
             desc.depth_clear,
             static_cast<unsigned>(pipeline.raster.cullMode));
}

VkDeviceSize GrowDynamicSuballocationSize(VkDeviceSize current_size,
                                          VkDeviceSize required_live_size, VkDeviceSize capacity,
                                          VkDeviceSize element_size) {
    if (capacity == 0) return 0;

    // Growth is geometric but clamped to authored capacity. That keeps normal particle expansion
    // amortized while still refusing to cross the renderer-side maximum promised by the scene data.
    VkDeviceSize next_size =
        current_size == 0
            ? InitialDynamicSuballocationSize(capacity, required_live_size, element_size)
            : current_size;
    const VkDeviceSize required = std::min<VkDeviceSize>(
        capacity,
        std::max<VkDeviceSize>(required_live_size, std::max<VkDeviceSize>(element_size, 1)));

    while (next_size < required && next_size < capacity) {
        const VkDeviceSize doubled = next_size > capacity / 2 ? capacity : next_size * 2;
        next_size                  = std::max<VkDeviceSize>(required, doubled);
        next_size                  = std::min<VkDeviceSize>(next_size, capacity);
    }
    return next_size;
}

bool RefreshCustomShaderPassTextures(wallpaper::Scene& scene, const Device& device,
                                     ShaderDrawData& desc) {
    desc.vk_textures.resize(desc.textures.size());
    for (wallpaper::usize i = 0; i < desc.textures.size(); i++) {
        auto& tex_name = desc.textures[i];
        if (tex_name.empty()) {
            desc.vk_textures[i] = {};
            continue;
        }

        ImageSlotsRef img_slots;
        const auto    render_target_it = scene.renderTargets.find(tex_name);
        if (render_target_it != scene.renderTargets.end()) {
            // The scene render-target table is the authoritative source for internal effect FBOs.
            // Some authored blur chains use plain names like `blur_start_2_<addr>`, so relying only
            // on the `_rt_` prefix would send valid runtime targets through the material-file
            // parser.
            auto& rt  = render_target_it->second;
            auto  opt = device.tex_cache().Query(
                tex_name, wallpaper::vulkan::ToTexKey(rt), ! rt.allowReuse);
            if (! opt.has_value()) {
                LOG_ERROR("CustomShaderPassRefresh: query input failed node='%s' output='%s' "
                          "slot=%zu texture='%s'",
                          desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
                          desc.output.c_str(),
                          static_cast<size_t>(i),
                          tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
            img_slots.slots.clear();
            img_slots.slots.push_back(opt.value());
        } else if (wallpaper::IsSpecTex(tex_name)) {
            LOG_ERROR("CustomShaderPassRefresh: missing input render target node='%s' "
                      "output='%s' slot=%zu texture='%s'",
                      desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
                      desc.output.c_str(),
                      static_cast<size_t>(i),
                      tex_name.c_str());
            desc.vk_textures[i] = {};
            continue;
        } else {
            if (scene.dirtyImportedTextureKeys.count(tex_name) == 0) {
                if (auto cached_slots = device.tex_cache().FindTex(tex_name);
                    cached_slots.has_value()) {
                    desc.vk_textures[i] = *cached_slots;
                    continue;
                }
            } else {
                scene.DropParsedImageCache(tex_name);
            }

            const auto texture_it = scene.textures.find(tex_name);
            const bool static_scene_texture =
                texture_it != scene.textures.end() && ! texture_it->second.isVideo;
            auto image = static_scene_texture ? scene.GetParsedImageIfReady(tex_name) : nullptr;
            if (image == nullptr) {
                image = static_scene_texture
                            ? scene.ParseImageBlockingCached(tex_name)
                            : (scene.imageParser != nullptr ? scene.imageParser->Parse(tex_name)
                                                            : nullptr);
            }
            if (image) {
                if (scene.textures.count(tex_name) != 0 && scene.textures.at(tex_name).isVideo) {
                    const auto paused_it = scene.videoTexturePaused.find(tex_name);
                    const bool stopped   = scene.videoTextureStopped.count(tex_name) != 0;
                    // Hidden video passes are kept prepared so visibility flips are cheap, but the
                    // backing decoder should still start paused unless a scene script explicitly
                    // requested playback for this texture.
                    const bool initially_paused =
                        paused_it != scene.videoTexturePaused.end()
                            ? paused_it->second
                            : (desc.draw.Valid() && ! desc.draw.Visible(scene));
                    const auto initial_state =
                        stopped
                            ? wallpaper::VideoTexturePlaybackState::Stopped
                            : (initially_paused ? wallpaper::VideoTexturePlaybackState::Paused
                                                : wallpaper::VideoTexturePlaybackState::Playing);
                    img_slots = device.video_tex_cache().Acquire(
                        tex_name, scene.textures.at(tex_name), *image, initial_state);
                } else {
                    img_slots = device.tex_cache().CreateTex(*image);
                    if (static_scene_texture) {
                        scene.DropParsedImageCache(tex_name);
                    }
                }
            } else {
                LOG_ERROR("parse tex \"%s\" failed", tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
        }
        desc.vk_textures[i] = img_slots;
    }

    auto&      tex_name  = desc.output;
    const auto output_it = scene.renderTargets.find(tex_name);
    if (output_it == scene.renderTargets.end()) {
        // Outputs must be registered render targets, but they do not have to be `_rt_`-prefixed:
        // effect-local FBOs are uniquified from their authored names and are still valid Vulkan
        // framebuffer destinations once WPSceneParser has inserted them into scene.renderTargets.
        LOG_ERROR("CustomShaderPassRefresh: missing output render target node='%s' output='%s'",
                  desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
                  tex_name.c_str());
        return false;
    }
    auto& rt = output_it->second;
    desc.sample_count      = VK_SAMPLE_COUNT_1_BIT;
    desc.resolve_msaa      = false;
    desc.alpha_to_coverage = false;
    desc.vk_resolve        = {};
    if (ShaderDrawCanUseMsaa(scene, tex_name, desc.resolved_scene_color)) {
        const auto ms_name = std::string(wallpaper::SpecTex_DefaultMS);
        const auto ms_it   = scene.renderTargets.find(ms_name);
        if (ms_it != scene.renderTargets.end()) {
            if (auto ms_opt = device.tex_cache().Query(
                    ms_name, wallpaper::vulkan::ToTexKey(ms_it->second), ! ms_it->second.allowReuse);
                ms_opt.has_value()) {
                desc.vk_output    = ms_opt.value();
                desc.sample_count = static_cast<VkSampleCountFlagBits>(
                    std::max(1u, ms_it->second.sample_count > 0
                                     ? static_cast<uint>(ms_it->second.sample_count)
                                     : 1u));
                if (desc.draw.Valid() && desc.draw.Mesh() != nullptr &&
                    desc.draw.Mesh()->Material() != nullptr) {
                    desc.alpha_to_coverage = desc.blend_override.value_or(
                        desc.draw.Mesh()->Material()->blenmode) ==
                        wallpaper::BlendMode::AlphaToCoverage;
                }
                return true;
            }
        }
        LOG_ERROR("CustomShaderPassRefresh: MSAA compose target missing node='%s' output='%s'",
                  desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
                  tex_name.c_str());
    }
    if (auto opt =
            device.tex_cache().Query(tex_name, wallpaper::vulkan::ToTexKey(rt), ! rt.allowReuse);
        opt.has_value()) {
        desc.vk_output = opt.value();
        return true;
    }
    LOG_ERROR("CustomShaderPassRefresh: query output failed node='%s' output='%s'",
              desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
              tex_name.c_str());
    return false;
}

ShaderDrawAttachmentDescription wallpaper::vulkan::SceneDepthAttachmentDescription(bool clear_depth) {
    const auto load_op = clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    return ShaderDrawAttachmentDescription {
        .format           = VK_FORMAT_D32_SFLOAT,
        .depth_load_op    = load_op,
        .depth_store_op   = VK_ATTACHMENT_STORE_OP_STORE,
        .stencil_load_op  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initial_layout   = clear_depth ? VK_IMAGE_LAYOUT_UNDEFINED
                                        : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .final_layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .cache_tag        = "model-depth",
    };
}

ShaderDrawAttachmentDescription ResolveShaderDrawAttachment(
    const ShaderDrawData& desc) {
    if (desc.shared_depth) {
        return SceneDepthAttachmentDescription(desc.clear_depth);
    }
    return {};
}

bool RecreateCustomShaderPassFramebuffer(const Device& device, RenderingResources& rr,
                                         ShaderDrawData& desc,
                                         ShaderDrawExtension* extension) {
    desc.fb.reset();
    if (! desc.pipeline.pass || desc.vk_output.view == VK_NULL_HANDLE ||
        desc.vk_output.extent.width == 0 || desc.vk_output.extent.height == 0) {
        LOG_ERROR("CustomShaderPassRefresh: cannot recreate framebuffer node='%s' output='%s' "
                  "hasRenderPass=%s hasView=%s extent=[%u,%u]",
                  desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
                  desc.output.c_str(),
                  desc.pipeline.pass ? "true" : "false",
                  desc.vk_output.view != VK_NULL_HANDLE ? "true" : "false",
                  desc.vk_output.extent.width,
                  desc.vk_output.extent.height);
        return false;
    }
    const auto attachment = ResolveShaderDrawAttachment(desc);
    if (desc.shared_depth) {
        desc.depth_stencil_image_ref = AcquireSceneDepthImage(
            device, rr, desc.output, desc.vk_output.extent, desc.sample_count);
    } else {
        desc.depth_stencil_image_ref = nullptr;
    }
    if (attachment.enabled() && desc.depth_stencil_image_ref == nullptr) return false;

    std::array<VkImageView, 3> attachments {
        desc.vk_output.view,
        desc.depth_stencil_image_ref != nullptr && desc.depth_stencil_image_ref->view
            ? *desc.depth_stencil_image_ref->view
            : VK_NULL_HANDLE,
        desc.vk_resolve.view,
    };
    if (desc.resolve_msaa && desc.vk_resolve.view == VK_NULL_HANDLE) {
        LOG_ERROR("CustomShaderPassRefresh: missing MSAA resolve view node='%s' output='%s'",
                  desc.draw.Valid() ? desc.draw.Name().c_str() : "<null>",
                  desc.output.c_str());
        return false;
    }
    uint32_t attachment_count = 1;
    if (attachment.enabled()) attachment_count++;
    if (desc.resolve_msaa) {
        if (! attachment.enabled()) {
            attachments[1] = desc.vk_resolve.view;
        }
        attachment_count++;
    }
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .renderPass      = *desc.pipeline.pass,
        .attachmentCount = attachment_count,
        .pAttachments    = attachments.data(),
        .width           = desc.vk_output.extent.width,
        .height          = desc.vk_output.extent.height,
        .layers          = 1,
    };
    const bool created = device.handle().CreateFramebuffer(info, desc.fb) == VK_SUCCESS;
    if (created && wallpaper::diagnostics::Options().trace_depth_attachments) {
        LOG_INFO("SceneDepthFramebuffer: layer=%d node='%s' output='%s' framebuffer=%p "
                 "color-view=%p depth-view=%p resolved-scene-color=%s shared-depth=%s "
                 "extent=%ux%u samples=%u",
                 desc.layer_id, desc.draw.Valid() ? desc.draw.Name().c_str() : "",
                 desc.output.c_str(), reinterpret_cast<void*>(*desc.fb),
                 reinterpret_cast<void*>(attachments[0]),
                 attachment.enabled() ? reinterpret_cast<void*>(attachments[1]) : nullptr,
                 desc.resolved_scene_color ? "true" : "false",
                 desc.shared_depth ? "true" : "false", info.width, info.height,
                 static_cast<unsigned>(desc.sample_count));
    }
    return created && (extension == nullptr || extension->refreshFramebuffers(device, rr, desc));
}

bool ShaderDrawCore::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    // Prepared passes can survive resource-only refreshes, so keep the live scene pointer current
    // before binding render targets and before text/effect diagnostics read bridge metadata.
    m_desc.scene = &scene;
    // A retrying prepared pass may still own a framebuffer whose attachment points at the previous
    // TextureCache image view. Drop it before `Query()` can resize and destroy that output image,
    // otherwise Vulkan sees a framebuffer referencing a dead attachment during minute-rollover
    // text bridge updates.
    dropOutputFramebuffers();
    m_desc.vk_tex_binding.clear();
    if (!m_desc.draw.Valid() || m_desc.draw.Mesh() == nullptr ||
        m_desc.draw.Mesh()->Material() == nullptr ||
        m_desc.draw.Mesh()->Material()->customShader.shader == nullptr) {
        LOG_ERROR("ShaderDrawPrepare: incomplete scene contract node='%s' output='%s'",
                  m_desc.draw.Valid() ? m_desc.draw.Name().c_str() : "<null>",
                  m_desc.output.c_str());
        return false;
    }
    SceneMesh& mesh = *(m_desc.draw.Mesh());
    if (m_extension != nullptr && ! m_extension->configure(device, m_desc, mesh)) return false;
    if (! RefreshCustomShaderPassTextures(scene, device, m_desc)) return false;
    if (m_extension != nullptr && ! m_extension->refreshTextures(scene, device, m_desc)) {
        return false;
    }

    std::vector<Uni_ShaderSpv> spvs;
    DescriptorSetInfo          descriptor_info;
    ShaderReflected            ref;
    {
        SceneShader& shader = *(mesh.Material()->customShader.shader);

        if (! GenReflect(shader.codes, spvs, ref)) {
            LOG_ERROR("gen spv reflect failed, %s", shader.name.c_str());
            return false;
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

        PopulateTextureBindingsFromReflection(m_desc, ref, m_desc.vk_textures.size());
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
                auto&      name     = item.first;
                auto&      input    = item.second;
                const bool has_attr = exists(attrs_map, name);
                usize      offset   = has_attr ? attrs_map[name].offset : 0;

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
                if (m_desc.dyn_vertex) {
                    const auto initial_size = DynamicVertexUploadSize(vertex);
                    if (! rr.dyn_buf->allocateSubRef(initial_size, buf)) return false;
                } else if (! mesh.FileImmutable()) {
                    if (! rr.vertex_buf->allocateSubRef(vertex.CapacitySizeOf(), buf)) return false;
                    if (! rr.vertex_buf->writeToBuf(buf, { (uint8_t*)vertex.Data(), buf.size }))
                        return false;
                }
            }
            m_desc.draw_count += (u32)(vertex.DataSize() / vertex.OneSize());
        }

        if (! m_desc.dyn_vertex && mesh.FileImmutable()) {
            m_desc.immutable_mesh = rr.immutable_meshes.getOrCreate(device, mesh);
            if (! m_desc.immutable_mesh) return false;
        }

        if (mesh.IndexCount() > 0) {
            m_desc.index_element_bytes = mesh.IndexElementBytes();
            m_desc.draw_count          = mesh.LogicalIndexCount();
            auto& buf                  = m_desc.index_buf;
            if (m_desc.dyn_vertex) {
                auto& indice = mesh.GetIndexArray(0);
                const auto initial_size = DynamicIndexUploadSize(indice);
                if (! rr.dyn_buf->allocateSubRef(initial_size, buf)) return false;
            } else if (! mesh.FileImmutable()) {
                auto& indice = mesh.GetIndexArray(0);
                if (! rr.vertex_buf->allocateSubRef(indice.CapacitySizeof(), buf)) return false;
                if (! rr.vertex_buf->writeToBuf(buf, { (uint8_t*)indice.Data(), buf.size })) {
                    return false;
                }
            }
        }
    }
    const auto render_state = BuildShaderDrawRenderState(*mesh.Material(), m_desc);
    if (const char* trace_layer = wallpaper::diagnostics::Options().transform_layer;
        trace_layer != nullptr && std::to_string(m_desc.layer_id) == trace_layer) {
        const auto& blend = render_state.color_blend;
        LOG_INFO("SceneShaderDrawPrepare: layer=%d node='%s' output='%s' count=%u "
                 "load=%u blend=%u rgb=[%u %u %u] alpha=[%u %u %u] mask=%u",
                 m_desc.layer_id, m_desc.draw.Name().c_str(), m_desc.output.c_str(),
                 m_desc.draw_count, render_state.color_load_op, blend.blendEnable,
                 blend.srcColorBlendFactor, blend.dstColorBlendFactor, blend.colorBlendOp,
                 blend.srcAlphaBlendFactor, blend.dstAlphaBlendFactor, blend.alphaBlendOp,
                 blend.colorWriteMask);
        // Record small generated cards before upload so their actual UV/position bytes can be
        // compared with the submitted transforms. Imported immutable payloads may already be
        // released; diagnostics must not recreate or retain them just to print this information.
        for (size_t binding = 0; binding < mesh.VertexCount(); ++binding) {
            const auto& vertices = mesh.GetVertexArray(binding);
            if (vertices.Data() == nullptr || vertices.VertexCount() > 4) continue;
            for (size_t vertex = 0; vertex < vertices.VertexCount(); ++vertex) {
                std::string values;
                for (size_t component = 0; component < vertices.OneSize(); ++component) {
                    if (component != 0) values += ' ';
                    values += std::to_string(
                        vertices.Data()[vertex * vertices.OneSize() + component]);
                }
                LOG_INFO("SceneShaderDrawVertex: layer=%d node='%s' binding=%zu "
                         "vertex=%zu values=[%s]",
                         m_desc.layer_id, m_desc.draw.Name().c_str(), binding, vertex,
                         values.c_str());
            }
        }
    }
    {
        const auto attachment = ResolveShaderDrawAttachment(m_desc);
        auto opt = CreateShaderDrawRenderPass(device.handle(),
                                              VK_FORMAT_R8G8B8A8_UNORM,
                                              render_state.color_load_op,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                              attachment,
                                              m_desc.sample_count,
                                              m_desc.resolve_msaa);
        if (! opt.has_value()) return false;
        auto& pass = opt.value();

        descriptor_info.push_descriptor = true;
        GraphicsPipeline pipeline;
        pipeline.toDefault();
        pipeline.multisample.rasterizationSamples = m_desc.sample_count;
        pipeline.multisample.alphaToCoverageEnable =
            m_desc.alpha_to_coverage && m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT;
        ApplyShaderDrawMaterialPipelineState(*mesh.Material(), m_desc, pipeline);
        m_desc.pipeline.debug_name =
            "CustomShaderPass[node=" +
            (m_desc.draw.Valid() ? m_desc.draw.Name() : std::string("(null)")) +
            ",output=" + m_desc.output + "]";
        pipeline.addDescriptorSetInfo(spanone { descriptor_info })
            .setColorBlendStates(spanone { render_state.color_blend })
            .setTopology(ToTopology(mesh))
            .addInputBindingDescription(bind_descriptions)
            .addInputAttributeDescription(attr_descriptions);
        for (auto& spv : spvs) pipeline.addStage(std::move(spv));

        m_desc.pipeline.cache_key = ShaderDrawPipelineCompatibilityKey(
            render_state.color_load_op,
            m_desc.model_pass,
            m_desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            attachment,
            m_desc.sample_count,
            m_desc.resolve_msaa);
        if (! pipeline.create(device, pass, m_desc.pipeline, rr.pipeline_cache.get())) return false;

        if (m_extension != nullptr &&
            ! m_extension->preparePipelines(
                device,
                rr,
                ShaderDrawPipelineContext {
                    .data                   = m_desc,
                    .mesh                   = mesh,
                    .material               = *mesh.Material(),
                    .descriptor_info        = descriptor_info,
                    .binding_descriptions   = bind_descriptions,
                    .attribute_descriptions = attr_descriptions,
                    .render_state           = render_state,
                })) {
            return false;
        }
    }
    {
        // The helper above already converts framebuffer creation into a plain success/failure
        // contract so that both the initial prepare path and the lightweight resource-refresh path
        // can share the same code. Keeping it as an explicit boolean check avoids routing a
        // non-VkResult helper through the `VVK_CHECK_*` macros, which only understand raw Vulkan
        // return codes.
        if (! RecreateCustomShaderPassFramebuffer(device, rr, m_desc, m_extension)) return false;
    }

    // Geometry residency belongs to the mesh, not to the shader's uniform layout. A dynamic
    // effect card still needs its vertex/index bytes when the shader uses only constants or
    // samplers. Install its upload operation independently so initial preparation, resource
    // refresh and per-frame updates all populate the same pass-owned GPU subranges.
    {
        std::function<void()> update_dyn_buf_op;
        if (m_desc.dyn_vertex) {
            auto&       mesh             = *m_desc.draw.Mesh();
            auto*       dyn_buf          = rr.dyn_buf;
            auto&       vertex_bufs      = m_desc.vertex_bufs;
            auto&       draw_count       = m_desc.draw_count;
            auto&       index_buf        = m_desc.index_buf;
            auto&       force_dyn_upload = m_desc.force_dyn_upload;
            auto&       uploaded_revision = m_desc.uploaded_mesh_revision;
            // The callback already refers to pass-owned storage. Read diagnostic names from
            // that same descriptor when logging instead of copying strings into the closure;
            // compiled-out INFO calls must not leave string allocations during preparation.
            update_dyn_buf_op                = [&mesh,
                                                &vertex_bufs,
                                                &draw_count,
                                                &index_buf,
                                                dyn_buf,
                                                &force_dyn_upload,
                                                &uploaded_revision,
                                                &desc = m_desc]() {
                const auto revision = mesh.DataRevision();
                const bool bootstrap_upload = force_dyn_upload;
                const bool needs_upload = revision != uploaded_revision || bootstrap_upload;
                if (needs_upload) {
                    auto ensure_vertex_subref = [&](usize                              array_index,
                                                    const wallpaper::SceneVertexArray& vertex) {
                        if (vertex_bufs.size() <= array_index) {
                            vertex_bufs.resize(array_index + 1);
                        }

                        auto&      buf                = vertex_bufs[array_index];
                        const auto required_live_size = static_cast<VkDeviceSize>(
                            std::max<usize>(vertex.DataSizeOf(), vertex.OneSizeOf()));
                        if (buf && buf.size >= required_live_size) return true;

                        // Dynamic custom-shader meshes may grow after a pass was prepared. Keep
                        // this as a renderer-level buffer refresh mechanism for authored
                        // dynamic meshes, while first-class text is handled by TextPass and
                        // never enters CustomShaderPass as glyph helper nodes.
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
                        if (! dyn_buf->allocateSubRef(required_size, buf)) {
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
                        const auto required_live_size = static_cast<VkDeviceSize>(
                            std::max<usize>(indice.DataSizeOf(), kDynamicIndexQuadFloorSize));
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
                        if (! dyn_buf->allocateSubRef(required_size, index_buf)) {
                            return false;
                        }
                        force_dyn_upload = true;
                        return true;
                    };

                    release_unused_vertex_subrefs();
                    for (usize i = 0; i < mesh.VertexCount(); i++) {
                        const auto& vertex = mesh.GetVertexArray(i);
                        if (! ensure_vertex_subref(i, vertex)) {
                            mesh.SetDirty();
                            return;
                        }
                        auto& buf = vertex_bufs[i];
                        if (! dyn_buf->writeToBuf(
                                buf, { (uint8_t*)vertex.Data(), vertex.DataSizeOf() })) {
                            mesh.SetDirty();
                            return;
                        }
                    }
                    if (mesh.IndexCount() > 0) {
                        auto& indice = mesh.GetIndexArray(0);
                        if (! ensure_index_subref(indice)) {
                            mesh.SetDirty();
                            return;
                        }
                        draw_count = mesh.LogicalIndexCount();
                        if (mesh.IndexElementBytes() == 4) {
                            draw_count = static_cast<u32>(indice.RenderDataCount());
                        } else {
                            const u32 count = (u32)((indice.RenderDataCount() * 2) / 3);
                            draw_count      = count * 3;
                        }
                        auto& buf  = index_buf;
                        if (! dyn_buf->writeToBuf(
                                buf, { (uint8_t*)indice.Data(), indice.DataSizeOf() })) {
                            mesh.SetDirty();
                            return;
                        }
                    } else {
                        // Dynamic non-indexed meshes are still drawable. Text effect outputs
                        // use a four-vertex triangle-strip card that is resized in place when
                        // Date/Day/ Clock content changes; clearing draw_count here made the
                        // bridge and effect passes execute successfully while submitting no
                        // final composite geometry at all. The first vertex binding defines the
                        // vertex count for non-indexed draws, matching the static prepare
                        // path's draw contract.
                        draw_count = mesh.VertexCount() > 0
                                         ? static_cast<u32>(mesh.GetVertexArray(0).DataSize() /
                                                            mesh.GetVertexArray(0).OneSize())
                                         : 0;
                        if (index_buf) {
                            dyn_buf->unallocateSubRef(index_buf);
                            index_buf = {};
                        }
                    }
                    // Clearing the pass-local bootstrap flag only after all writes succeed
                    // keeps a freshly compiled dynamic pass from getting stuck with empty GPU
                    // buffers if an earlier upload attempt bails out partway through due to an
                    // allocation/write failure. Subsequent frames will keep retrying until the
                    // first complete upload lands in the new subranges.
                    mesh.Dirty().store(false);
                    uploaded_revision = revision;
                    force_dyn_upload = false;
                    if (wallpaper::diagnostics::Options().trace_mesh_uploads) {
                        LOG_INFO("SceneMeshUpload: layer=%d output='%s' reflection=%s "
                                 "revision=%llu draw-count=%u node='%s' bootstrap=%s",
                                 desc.layer_id, desc.output.c_str(), desc.reflection_pass ? "true" : "false",
                                 static_cast<unsigned long long>(revision), draw_count,
                                 desc.draw.Name().c_str(), bootstrap_upload ? "true" : "false");
                    }
                }
            };
        }
        m_desc.update_dynamic_mesh_op = std::move(update_dyn_buf_op);
    }

    if (!ref.blocks.empty() || m_extension != nullptr) {
        std::optional<ShaderReflected::Block> block;
        if (!ref.blocks.empty()) {
            block = ref.blocks.front();
            if (!rr.dyn_buf->allocateSubRef(
                    block->size, m_desc.ubo_buf, device.limits().minUniformBufferOffsetAlignment)) {
                return false;
            }
        }
        auto* buf    = rr.dyn_buf;
        auto* bufref = &m_desc.ubo_buf;
        const auto draw      = m_desc.draw;
        auto* shader_updater = scene.shaderValueUpdater.get();
        auto* extension      = m_extension;
        auto& sprites        = m_desc.sprites_map;
        auto& vk_textures    = m_desc.vk_textures;
        // Keep the material dependency explicit in the capture list because the updater writes the
        // authored uniforms directly and should not rediscover the material through the scene node.
        auto* material = mesh.Material();

        m_desc.update_op =
            [shader_updater, block, buf, bufref, extension,
             draw, material, &sprites, &vk_textures,
             camera_override = m_desc.camera_override,
             use_active_camera_for_uniforms = m_desc.use_active_camera_for_uniforms,
             use_active_camera_for_parallax = m_desc.use_active_camera_for_parallax,
             suppress_destination_parallax = m_desc.suppress_destination_parallax,
             model_space = m_desc.model_space,
             reflection_pass = m_desc.reflection_pass,
             reflection_raster = m_desc.reflection_raster,
             reflection_snapshot = m_desc.reflection_snapshot,
             effect_snapshot_camera = m_desc.effect_snapshot_camera,
             uniform_writes = &m_trace_uniform_writes,
             textures = &m_desc.textures]() {
                const auto trace_uniforms = uniform_writes->beginUpdate();
                auto update_unf_op = [block, buf, bufref, extension, trace_uniforms, uniform_writes](
                                         std::string_view name, wallpaper::ShaderValue value) {
                    if (block) UpdateShaderDrawUniform(buf, *bufref, *block, name, value);
                    if (trace_uniforms && block) {
                        const auto member = block->member_map.find(name);
                        if (member != block->member_map.end()) {
                            // Record only writes admitted by the visible executable's layout.
                            // Extension-only uniforms use separate buffers and cannot establish
                            // what this draw consumed. Keep reflected and packed sizes distinct
                            // so compact matrices and strided arrays retain their upload meaning.
                            uniform_writes->record(name, value,
                                member->second.offset, member->second.size);
                        }
                    }
                    if (extension != nullptr) extension->updateUniform(buf, name, value);
                };
                if (material != nullptr) {
                    if (block) WriteMaterialUniforms(buf, *bufref, *block, *material);
                    if (extension != nullptr) extension->updateMaterialUniforms(buf, *material);
                }
                const ShaderUniformOverrides overrides {
                    .camera_name = camera_override,
                    .use_camera_override = !camera_override.empty(),
                    .use_active_camera_for_uniforms = use_active_camera_for_uniforms,
                    .use_active_camera_for_parallax = use_active_camera_for_parallax,
                    .suppress_destination_parallax = suppress_destination_parallax,
                    .model_space = model_space,
                    .reflection_pass = reflection_pass,
                    .reflection_raster = reflection_raster,
                    .reflection_snapshot = reflection_snapshot,
                    .effect_snapshot_camera = effect_snapshot_camera,
                    .textures = std::span<const std::string>(*textures),
                };
                shader_updater->UpdateUniforms(
                    draw,
                    sprites,
                    update_unf_op,
                    &overrides);
                // update image slot for sprites
                {
                    for (auto& [i, sp] : sprites) {
                        if (i >= vk_textures.size()) continue;
                        vk_textures.at(i).active = sp.GetCurFrame().imageId;
                    }
                }
            };

        // An extension can consume matrices or pose uniforms that the visible executable
        // omits. Initialize the updater from the union, then fan out one current-frame value
        // into each independently reflected layout before the shared staging upload.
        auto exists_unf_op = [&block, extension](std::string_view name) {
            return (block && exists(block->member_map, name)) ||
                (extension != nullptr && extension->hasUniform(name));
        };
        shader_updater->InitUniforms(draw, exists_unf_op);

        // memset uniform buf
        if (block) buf->fillBuf(*bufref, 0, bufref->size, 0);
        if (m_extension != nullptr) m_extension->initializeUniforms(buf);
        m_desc.update_op();
    }
    // Bootstrap geometry even when no uniform updater exists. Preserve the uniform-then-mesh
    // ordering used by updateBeforeUpload(): all CPU writes for this draw must precede the
    // shared staging-buffer copy recorded by VulkanRender.
    if (m_desc.update_dynamic_mesh_op) m_desc.update_dynamic_mesh_op();

    {
        m_desc.clear_value =
            BuildCustomShaderClearValue(scene, *mesh.Material(), m_desc.clear_before_draw);
    }
    return true;
}

bool ShaderDrawCore::warmupPipeline(Scene& scene, const Device& device, RenderingResources& rr) {
    m_desc.scene = &scene;
    if (!m_desc.draw.Valid() || m_desc.draw.Mesh() == nullptr ||
        m_desc.draw.Mesh()->Material() == nullptr) {
        return false;
    }

    SceneMesh& mesh = *(m_desc.draw.Mesh());
    if (m_extension != nullptr && ! m_extension->configure(device, m_desc, mesh)) return false;

    std::vector<Uni_ShaderSpv> spvs;
    DescriptorSetInfo          descriptor_info;
    ShaderReflected            ref;
    {
        SceneShader& shader = *(mesh.Material()->customShader.shader);
        if (! GenReflect(shader.codes, spvs, ref)) {
            LOG_ERROR("pipeline warmup reflect failed, %s", shader.name.c_str());
            return false;
        }

        auto& bindings = descriptor_info.bindings;
        bindings.resize(ref.binding_map.size());
        std::transform(
            ref.binding_map.begin(), ref.binding_map.end(), bindings.begin(), [](auto& item) {
                return item.second;
            });
        PopulateTextureBindingsFromReflection(m_desc, ref, mesh.Material()->textures.size());
    }

    std::vector<VkVertexInputBindingDescription>   bind_descriptions;
    std::vector<VkVertexInputAttributeDescription> attr_descriptions;
    for (uint i = 0; i < mesh.VertexCount(); i++) {
        const auto& vertex    = mesh.GetVertexArray(i);
        auto        attrs_map = vertex.GetAttrOffsetMap();

        bind_descriptions.push_back(VkVertexInputBindingDescription {
            .binding   = i,
            .stride    = static_cast<uint32_t>(vertex.OneSizeOf()),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        });

        for (auto& item : ref.input_location_map) {
            const auto& name     = item.first;
            const auto& input    = item.second;
            const bool  has_attr = exists(attrs_map, name);
            const auto  offset   = has_attr ? attrs_map[name].offset : 0;
            attr_descriptions.push_back(VkVertexInputAttributeDescription {
                .location = input.location,
                .binding  = i,
                .format   = input.format,
                .offset   = static_cast<uint32_t>(offset),
            });
        }
    }
    if (ShaderDrawCanUseMsaa(scene, m_desc.output, m_desc.resolved_scene_color)) {
        m_desc.sample_count = static_cast<VkSampleCountFlagBits>(
            std::max(1, scene.MsaaSampleCount()));
        m_desc.resolve_msaa = false;
        if (mesh.Material() != nullptr) {
            m_desc.alpha_to_coverage = m_desc.blend_override.value_or(
                mesh.Material()->blenmode) == BlendMode::AlphaToCoverage;
        }
    }
    auto render_state = BuildShaderDrawRenderState(*mesh.Material(), m_desc);
    const auto attachment = ResolveShaderDrawAttachment(m_desc);
    auto opt = CreateShaderDrawRenderPass(device.handle(),
                                          VK_FORMAT_R8G8B8A8_UNORM,
                                          render_state.color_load_op,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                          attachment,
                                          m_desc.sample_count,
                                          m_desc.resolve_msaa);
    if (! opt.has_value()) return false;
    auto& pass = opt.value();

    descriptor_info.push_descriptor = true;
    GraphicsPipeline pipeline;
    pipeline.toDefault();
    pipeline.multisample.rasterizationSamples = m_desc.sample_count;
    pipeline.multisample.alphaToCoverageEnable =
        m_desc.alpha_to_coverage && m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT;
    ApplyShaderDrawMaterialPipelineState(*mesh.Material(), m_desc, pipeline);
    m_desc.pipeline.debug_name =
        "CustomShaderPassWarmup[node=" +
        (m_desc.draw.Valid() ? m_desc.draw.Name() : std::string("(null)")) +
        ",output=" + m_desc.output + "]";
    m_desc.pipeline.cache_key = ShaderDrawPipelineCompatibilityKey(
        render_state.color_load_op,
        m_desc.model_pass,
        m_desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        attachment,
        m_desc.sample_count,
        m_desc.resolve_msaa);
    pipeline.addDescriptorSetInfo(spanone { descriptor_info })
        .setColorBlendStates(spanone { render_state.color_blend })
        .setTopology(ToTopology(mesh))
        .addInputBindingDescription(bind_descriptions)
        .addInputAttributeDescription(attr_descriptions);
    for (auto& spv : spvs) pipeline.addStage(std::move(spv));

    return pipeline.create(device, pass, m_desc.pipeline, rr.pipeline_cache.get());
}

bool ShaderDrawCore::refreshResources(Scene& scene, const Device& device,
                                      RenderingResources& rr) {
    // Resource refreshes reuse the pass object. Refresh the scene pointer first so dependency
    // checks observe the current render-target table and text bridge state without logging.
    m_desc.scene = &scene;
    // Resource-only refreshes intentionally keep the compiled shader pipeline, reflected bindings,
    // and uploaded mesh/UBO allocations intact. The expensive part that changes for effect-backed
    // minute updates is the texture-cache-backed image handle set and the framebuffer that wraps
    // the resized render target. Rebinding only those pieces avoids recompiling every shader pass
    // in the scene when the clock/date text changes shape.
    if (m_desc.draw.Valid() && m_desc.draw.Mesh() != nullptr) {
        auto& mesh = *m_desc.draw.Mesh();
        if (! mesh.Dynamic() && mesh.Dirty().load()) {
            // Resource-only refreshes were originally written for effects whose geometry never
            // changes after graph build. Refactored text effects break that assumption: runtime
            // updates now mutate the static blur/compose quads of already-compiled effect passes.
            // If we only recreate textures/framebuffers here, the pass keeps drawing the old GPU
            // vertex buffer even though the SceneMesh carries the new map-rate-adjusted quad. By
            // dropping back to the normal prepare path for dirty static meshes we force the pass
            // to re-upload its mesh data and make runtime text-effect geometry changes actually
            // visible on screen.
            destroy(rr);
            return false;
        }
    }

    const auto output_target_it = scene.renderTargets.find(m_desc.output);
    if (output_target_it == scene.renderTargets.end()) {
        LOG_ERROR(
            "CustomShaderPassRefresh: output target not found before refresh node='%s' output='%s'",
            m_desc.draw.Valid() ? m_desc.draw.Name().c_str() : "<null>",
            m_desc.output.c_str());
        return false;
    }
    const auto previous_output_view   = m_desc.vk_output.view;
    const auto previous_output_extent = m_desc.vk_output.extent;
    const auto previous_samples       = m_desc.sample_count;
    const auto previous_resolve       = m_desc.resolve_msaa;
    const auto desired_output_key     = wallpaper::vulkan::ToTexKey(output_target_it->second);
    const bool output_extent_changed =
        previous_output_extent.width != static_cast<uint32_t>(desired_output_key.width) ||
        previous_output_extent.height != static_cast<uint32_t>(desired_output_key.height);
    const int intended_samples = IntendedShaderDrawSampleCount(m_desc);
    if (output_extent_changed || intended_samples != static_cast<int>(previous_samples)) {
        // Drop the framebuffer before TextureCache replaces `_rt_FullFrameBufferMultiSampled`.
        // Keeping a live framebuffer across that resize leaves a destroyed MSAA image attached
        // and the next submit waits on the frame fence forever.
        dropOutputFramebuffers();
    }
    if (! RefreshCustomShaderPassTextures(scene, device, m_desc)) {
        LOG_ERROR("CustomShaderPassRefresh: texture refresh failed node='%s' output='%s'",
                  m_desc.draw.Valid() ? m_desc.draw.Name().c_str() : "<null>",
                  m_desc.output.c_str());
        return false;
    }
    if (m_desc.sample_count != previous_samples || m_desc.resolve_msaa != previous_resolve) {
        destroy(rr);
        return false;
    }
    if (m_extension != nullptr && ! m_extension->refreshTextures(scene, device, m_desc)) {
        return false;
    }
    const bool output_view_changed = previous_output_view != m_desc.vk_output.view;
    const bool framebuffer_missing = ! m_desc.fb;
    if (framebuffer_missing || output_extent_changed || output_view_changed) {
        if (! RecreateCustomShaderPassFramebuffer(device, rr, m_desc, m_extension)) {
            return false;
        }
    }
    if (m_desc.dyn_vertex && m_desc.update_dynamic_mesh_op != nullptr && m_desc.draw.Valid() &&
        m_desc.draw.Mesh() != nullptr &&
        (m_desc.force_dyn_upload ||
         m_desc.uploaded_mesh_revision != m_desc.draw.Mesh()->DataRevision())) {
        // Text-backed effect passes keep their render-graph topology stable while the final
        // source quad changes size. Uploading the dirty dynamic mesh during the resource-refresh
        // phase lets the compile-time dynamic-buffer copy include the new quad before the first
        // post-refresh draw, instead of binding a fresh suballocation that still contains old data.
        m_desc.update_dynamic_mesh_op();
    }
    return true;
}

bool ShaderDrawCore::refreshImportedTextureBindings(Scene& scene, const Device& device) {
    m_desc.scene = &scene;

    for (usize texture_index = 0; texture_index < m_desc.textures.size(); ++texture_index) {
        const auto& texture_key = m_desc.textures[texture_index];
        if (scene.dirtyImportedTextureResourceKeys.count(texture_key) == 0) continue;

        const auto cached_slots = device.tex_cache().FindTex(texture_key);
        if (!cached_slots.has_value() || cached_slots->slots.empty()) {
            LOG_ERROR("ImportedTexturePassRebind: cached texture missing layer=%d node='%s' "
                      "output='%s' slot=%zu key='%s'",
                      m_desc.layer_id,
                      m_desc.draw.Valid() ? m_desc.draw.Name().c_str() : "<null>",
                      m_desc.output.c_str(),
                      static_cast<size_t>(texture_index),
                      texture_key.c_str());
            return false;
        }

        if (m_desc.vk_textures.size() < m_desc.textures.size()) {
            m_desc.vk_textures.resize(m_desc.textures.size());
        }
        m_desc.vk_textures[texture_index] = *cached_slots;
    }

    bool extension_affected = false;
    if (m_extension != nullptr && m_desc.draw.Valid() && m_desc.draw.Mesh() != nullptr) {
        for (const auto texture_key : m_extension->resourceTextures(*m_desc.draw.Mesh())) {
            if (scene.dirtyImportedTextureResourceKeys.count(std::string(texture_key)) == 0) {
                continue;
            }
            extension_affected = true;
            break;
        }
        if (extension_affected && !m_extension->refreshTextures(scene, device, m_desc)) {
            return false;
        }
    }
    return true;
}

void ShaderDrawCore::dropOutputFramebuffers() {
    m_desc.fb.reset();
    if (m_extension != nullptr) m_extension->dropFramebuffers();
}

void ShaderDrawCore::updateBeforeUpload() {
    if (m_desc.should_execute && ! m_desc.should_execute()) {
        return;
    }

    if (m_desc.draw.Valid() && ! m_desc.draw.LocalVisible()) {
        return;
    }

    const bool node_visible = !m_desc.draw.Valid() ? true : m_desc.draw.Visible(*m_desc.scene);
    if (m_desc.draw.Valid() && ! node_visible && ! m_desc.execute_when_hidden) {
        return;
    }

    // recordUpload() flushes the mapped staging allocation, records the staging-to-GPU copies, and
    // clears its dirty ranges before any pass executes. Writing uniforms from execute() therefore
    // made every animated material value arrive one submitted frame late: on a media switch the new
    // current texture was visible while the blend pass still read the previous animation endpoint.
    // Update both the pass UBO and dynamic geometry here, in render-graph order, so the buffer copy
    // and the draw recorded for this submit describe one coherent frame.
    if (m_desc.update_op) m_desc.update_op();
    if (m_desc.update_dynamic_mesh_op) m_desc.update_dynamic_mesh_op();
}

void ShaderDrawCore::execute(const Device& device, RenderingResources& rr) {
    const char* trace_layer = wallpaper::diagnostics::Options().transform_layer;
    const char* trace_target = wallpaper::diagnostics::Options().render_target;
    // A named target can connect otherwise unrelated owners. Select both its producer and
    // consumers in the same process so the trace can compare actual GPU image identities,
    // execution gates and ordering without changing the render graph or reading pixels.
    const bool trace_draw = RenderCommandTraceActive(rr) || (trace_layer != nullptr &&
        std::to_string(m_desc.layer_id) == trace_layer) ||
        (trace_target != nullptr && (m_desc.output == trace_target ||
            std::find(m_desc.textures.begin(), m_desc.textures.end(), trace_target) !=
                m_desc.textures.end()));
    if (trace_draw) ++m_trace_draw_sequence;
    const auto trace_result = [&](const char* result) {
        if constexpr (!wallpaper::diagnostics::Enabled) return;
        if (RenderCommandTraceActive(rr)) {
            const bool recorded_draw = std::string_view(result) == "draw" && m_desc.draw_count > 0;
            const auto command = TraceRenderCommand(rr, "shader", result, m_desc.output,
                m_desc.vk_output, m_desc.layer_id, m_desc.reflection_pass, m_desc.draw_count);
            for (size_t index = 0; index < m_desc.vk_textures.size(); ++index) {
                const auto& slots = m_desc.vk_textures[index];
                if (slots.slots.empty()) continue;
                const int binding = m_desc.vk_tex_binding[index];
                TraceRenderCommandInput(rr, command, "sampler", m_desc.textures[index],
                    slots.getActive(), recorded_draw && binding >= 0, binding);
            }
            if (recorded_draw) {
                for (const auto& write : m_trace_uniform_writes) {
                    rr.frame_trace_dump.addUniform(command, write.name, write.offset,
                        write.reflected_size, write.value.data(), write.value.size());
                    if (!rr.trace_render_commands) continue;
                    std::string values;
                    for (size_t index = 0; index < write.value.size(); ++index) {
                        char component[32];
                        // Nine significant decimal digits round-trip every uploaded float.
                        // Preserve the packed order, including matrix padding, rather than
                        // transposing values into a presentation-specific matrix convention.
                        std::snprintf(component, sizeof(component), "%.9g", write.value[index]);
                        if (!values.empty()) values += ' ';
                        values += component;
                    }
                    LOG_INFO("SceneRenderCommandUniform: frame=%llu command=%llu name='%s' "
                             "offset=%zu reflected-bytes=%zu packed-bytes=%zu values=[%s]",
                             static_cast<unsigned long long>(rr.trace_render_frame),
                             static_cast<unsigned long long>(command), write.name.c_str(),
                             write.offset, write.reflected_size,
                             write.value.size() * sizeof(ShaderValue::value_type), values.c_str());
                }
            }
        }
        // Pass-begin profiling also includes skipped draws. Log the execution decision and exact
        // bound images at sparse checkpoints so an absent element can be traced without guessing
        // from visibility flags or flooding every frame with descriptor and geometry metadata.
        // A one-frame media transition needs consecutive decisions rather than sparse
        // checkpoints. This opt-in keeps the existing layer/target filter and adds the
        // scene clock so resource generations can be matched to SceneMediaDispatch.
        const bool trace_every_frame = rr.trace_render_commands ||
            wallpaper::diagnostics::Options().trace_draw_every_frame;
        if (!trace_draw || (!trace_every_frame && m_trace_draw_sequence != 1 &&
                            m_trace_draw_sequence != 121 && m_trace_draw_sequence != 601)) return;
        LOG_INFO("SceneShaderDrawExecute: sequence=%llu time=%.6f layer=%d node='%s' result=%s "
                 "count=%u output='%s' image=%p extent=%ux%u",
                 static_cast<unsigned long long>(m_trace_draw_sequence),
                 m_desc.scene != nullptr ? m_desc.scene->elapsingTime : 0.0, m_desc.layer_id,
                 m_desc.draw.Name().c_str(), result, m_desc.draw_count, m_desc.output.c_str(),
                 reinterpret_cast<void*>(m_desc.vk_output.handle),
                 m_desc.vk_output.extent.width, m_desc.vk_output.extent.height);
        for (size_t index = 0; index < m_desc.vk_textures.size(); ++index) {
            const auto& slots = m_desc.vk_textures[index];
            if (slots.slots.empty()) continue;
            const auto& input = slots.getActive();
            LOG_INFO("SceneShaderDrawInput: sequence=%llu layer=%d node='%s' index=%zu "
                     "key='%s' binding=%d slot=%u image=%p extent=%ux%u",
                     static_cast<unsigned long long>(m_trace_draw_sequence), m_desc.layer_id,
                     m_desc.draw.Name().c_str(), index, m_desc.textures[index].c_str(),
                     m_desc.vk_tex_binding[index], static_cast<unsigned>(slots.active),
                     reinterpret_cast<void*>(input.handle), input.extent.width, input.extent.height);
        }
        if (const char* trace_uniform = wallpaper::diagnostics::Options().material_uniform;
            trace_uniform != nullptr && m_desc.draw.Mesh() != nullptr) {
            const auto& material = *m_desc.draw.Mesh()->Material();
            const auto log_value = [&](const auto& values, const char* source) {
                const auto it = values.find(trace_uniform);
                if (it == values.end()) return;
                std::string value;
                for (size_t component = 0; component < it->second.size(); ++component) {
                    if (!value.empty()) value += ' ';
                    value += std::to_string(it->second[component]);
                }
                LOG_INFO("SceneShaderDrawMaterialValue: sequence=%llu layer=%d node='%s' "
                         "uniform='%s' source=%s value=[%s]",
                         static_cast<unsigned long long>(m_trace_draw_sequence), m_desc.layer_id,
                         m_desc.draw.Name().c_str(), trace_uniform, source, value.c_str());
            };
            // Preserve upload precedence in the report: authored/live values override the
            // executable's defaults in WriteMaterialUniforms before this submit is recorded.
            log_value(material.customShader.shader->default_uniforms, "program-default");
            log_value(material.customShader.constValues, "material");
        }
    };
    if (m_desc.should_execute && ! m_desc.should_execute()) {
        trace_result("execution-gate");
        // Runtime-gated helper passes stay in the render graph so visibility flips do not rebuild
        // framebuffer topology. Returning before uniform updates and draw submission makes the pass
        // a true no-op on frames where its fallback branch is not active.
        return;
    }

    if (m_desc.draw.Valid() && ! m_desc.draw.LocalVisible()) {
        trace_result("local-hidden");
        // execute_when_hidden is only for layer-level invisibility, such as offscreen dependency
        // sources that must keep rendering while their authored layer is hidden in the main scene.
        // Effect-local visibility is a stricter contract: a hidden effect must not run its shader
        // pass, otherwise the hidden branch would still overwrite the ping-pong output that the
        // bypass copy is responsible for preserving.
        return;
    }

    const bool node_visible = !m_desc.draw.Valid() ? true : m_desc.draw.Visible(*m_desc.scene);
    if (m_desc.draw.Valid() && ! node_visible && ! m_desc.execute_when_hidden) {
        trace_result("owner-hidden");
        // The render graph has still reached this pass's ordering point even when authored
        // visibility turns the shader into a no-op for the frame. Releasing final-read keys here
        // prevents temporary render targets from staying pinned only because no draw was recorded.
        return;
    }

    if (auto* scene = m_desc.scene != nullptr ? m_desc.scene : rr.scene;
        scene != nullptr && ShaderDrawSamplesResolvedDefault(m_desc.textures)) {
        ResolveComposeMsaaIfNeeded(*scene, device, rr);
    }

    auto&                   cmd    = rr.command;
    auto&                   outext = m_desc.vk_output.extent;
    const auto is_comparison_depth = [&](usize i) {
        if (i >= m_desc.textures.size()) return false;
        const auto& name = m_desc.textures[i];
        if (name == SpecTex_ShadowAtlas) return true;
        if (m_desc.scene == nullptr) return false;
        const auto it = m_desc.scene->renderTargets.find(name);
        return it != m_desc.scene->renderTargets.end() && it->second.comparisonDepth;
    };
    const auto push_visible_descriptors = [&](VkPipelineLayout layout) {
        for (usize i = 0; i < m_desc.vk_textures.size(); i++) {
            const int binding = m_desc.vk_tex_binding[i];
            if (binding < 0 || m_desc.vk_textures[i].slots.empty()) continue;
            const auto& image = m_desc.vk_textures[i].getActive();
            VkDescriptorImageInfo image_info {
                image.sampler,
                image.view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            VkWriteDescriptorSet write {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext           = nullptr,
                .dstSet          = {},
                .dstBinding      = static_cast<uint32_t>(binding),
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &image_info,
            };
            cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, write);
        }
        if (m_desc.ubo_buf) {
            VkDescriptorBufferInfo buffer_info {
                rr.dyn_buf->gpuBuf(),
                m_desc.ubo_buf.offset,
                m_desc.ubo_buf.size,
            };
            VkWriteDescriptorSet write {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext           = nullptr,
                .dstSet          = {},
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &buffer_info,
            };
            cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, write);
        }
    };
    push_visible_descriptors(*m_desc.pipeline.layout);

    for (usize i = 0; i < m_desc.vk_textures.size(); i++) {
        auto& slot = m_desc.vk_textures[i];
        if (slot.slots.empty()) continue;
        auto& img = slot.getActive();
        const bool comparison_depth = is_comparison_depth(i);
        VkImageSubresourceRange srang {
            .aspectMask     = comparison_depth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                               : VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = VK_REMAINING_ARRAY_LAYERS,
            .baseArrayLayer = 0,
            .layerCount     = VK_REMAINING_MIP_LEVELS,
        };

        VkImageMemoryBarrier imb {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            // Color inputs wait on the previous color-attachment write. The shadow atlas is a
            // depth comparison target written by late fragment tests, then sampled with
            // texSample2DCompare.
            .srcAccessMask    = comparison_depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                                 : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = img.handle,
            .subresourceRange = srang,
        };

        cmd.PipelineBarrier(comparison_depth ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                                             : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            imb);
    }

    m_desc.depth_clear_value.depthStencil = { m_desc.depth_clear, 0 };
    std::array<VkClearValue, 3> clear_values { m_desc.clear_value, m_desc.depth_clear_value, {} };
    uint32_t clear_count = 1;
    if (ResolveShaderDrawAttachment(m_desc).enabled()) clear_count++;
    if (m_desc.resolve_msaa) clear_count++;
    VkRenderPassBeginInfo       pass_begin_info {
        .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext       = nullptr,
        .renderPass  = *m_desc.pipeline.pass,
        .framebuffer = *m_desc.fb,
        .renderArea =
            VkRect2D {
                .offset = { 0, 0 },
                .extent = { outext.width, outext.height },
            },
        .clearValueCount = clear_count,
        .pClearValues    = clear_values.data(),
    };
    cmd.BeginRenderPass(pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
    if (m_desc.clear_before_draw) {
        TraceRenderCommand(rr, "shader-clear", "recorded", m_desc.output, m_desc.vk_output,
                           m_desc.layer_id, m_desc.reflection_pass);
    }

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

    if (m_desc.immutable_mesh) {
        for (usize i = 0; i < m_desc.immutable_mesh->vertices.size(); i++) {
            VkBuffer           gpu = m_desc.immutable_mesh->vertices[i].handle();
            const VkDeviceSize off = 0;
            if (gpu == VK_NULL_HANDLE) continue;
            cmd.BindVertexBuffers((u32)i, 1, &gpu, &off);
        }
        if (m_desc.immutable_mesh->has_index) {
            const VkIndexType index_type = m_desc.immutable_mesh->index_element_bytes == 4
                                               ? VK_INDEX_TYPE_UINT32
                                               : VK_INDEX_TYPE_UINT16;
            cmd.BindIndexBuffer(m_desc.immutable_mesh->index.handle(), 0, index_type);
            if (m_extension == nullptr) {
                cmd.DrawIndexed(m_desc.draw_count, 1, 0, 0, 0);
            } else {
                m_extension->recordIndexed(ShaderDrawRecordContext {
                    .data                     = m_desc,
                    .device                   = device,
                    .resources                = rr,
                    .push_visible_descriptors = push_visible_descriptors,
                });
            }
        } else {
            cmd.Draw(m_desc.draw_count, 1, 0, 0);
        }
    } else {
        auto gpu_buf = m_desc.dyn_vertex ? rr.dyn_buf->gpuBuf() : rr.vertex_buf->gpuBuf();

        for (usize i = 0; i < m_desc.vertex_bufs.size(); i++) {
            auto& buf = m_desc.vertex_bufs[i];
            cmd.BindVertexBuffers((u32)i, 1, &gpu_buf, &buf.offset);
        }
        if (m_desc.index_buf) {
            const VkIndexType index_type = m_desc.index_element_bytes == 4
                                               ? VK_INDEX_TYPE_UINT32
                                               : VK_INDEX_TYPE_UINT16;
            cmd.BindIndexBuffer(gpu_buf, m_desc.index_buf.offset, index_type);
            if (m_extension == nullptr) {
                cmd.DrawIndexed(m_desc.draw_count, 1, 0, 0, 0);
            } else {
                m_extension->recordIndexed(ShaderDrawRecordContext {
                    .data                     = m_desc,
                    .device                   = device,
                    .resources                = rr,
                    .push_visible_descriptors = push_visible_descriptors,
                });
            }
        } else {
            cmd.Draw(m_desc.draw_count, 1, 0, 0);
        }
    }

    cmd.EndRenderPass();

    // Successful draw records follow any on-demand input resolve and attachment clear.
    // The shared frame trace therefore reflects recording order across different pass
    // types, including the resolve attachment written when this render pass ends.
    trace_result("draw");
    if (m_desc.resolve_msaa) {
        const auto command = TraceRenderCommand(rr, "shader-resolve", "recorded", m_desc.output,
            m_desc.vk_resolve, m_desc.layer_id, m_desc.reflection_pass);
        TraceRenderCommandInput(rr, command, "resolve-source", m_desc.output, m_desc.vk_output, true);
    }

    if (m_desc.shared_depth) {
        rr.model_depth_images.at(m_desc.output).layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    if (m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        m_desc.output == wallpaper::SpecTex_Default) {
        NoteComposeMsaaDraw(rr, m_desc.sample_count);
    }

    if (m_desc.shared_depth && m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        m_desc.depth_stencil_image_ref != nullptr && m_desc.depth_stencil_image_ref->handle) {
        // Only mark the shared multisampled depth dirty. The depth-sampling consumer materializes
        // the single-sample copy once on demand (ResolveModelDepthIfNeeded); resolving here would
        // pay one full-extent depth resolve per model chunk pass, which dominates the frame on
        // model-heavy scenes.
        rr.model_depth_images.at(m_desc.output).resolve_dirty = true;
    }
    // Temporary render targets may only be returned to TextureCache after the pass has actually
    // consumed them in the recorded frame. Releasing during prepare/resource-refresh is unsafe:
    // all passes are prepared before any pass executes, so a later pass can accidentally bind a
    // same-sized but unrelated physical image for a still-live logical key.
}

void ShaderDrawCore::destroy(RenderingResources& rr) {
    m_desc.update_op              = {};
    m_desc.update_dynamic_mesh_op = {};
    if (m_extension != nullptr) m_extension->destroy(rr);
    // Retiring a hidden layer must drop framebuffer/image/buffer residency while leaving cached
    // PSO ownership to GraphicsPipelineStateCache. This mirrors game-engine visibility handling:
    // textures and render targets can be evicted, but the immutable shader pipeline remains warm
    // for the next show transition instead of recompiling on the visible frame.
    m_desc.fb.reset();
    m_desc.vk_textures.clear();
    m_desc.vk_tex_binding.clear();
    m_desc.vk_output              = {};
    m_desc.vk_resolve             = {};
    m_desc.depth_stencil_image_ref = nullptr;
    m_desc.immutable_mesh.reset();
    auto* mesh_buf = m_desc.dyn_vertex ? rr.dyn_buf : rr.vertex_buf;
    for (auto& bufref : m_desc.vertex_bufs) {
        if (mesh_buf) mesh_buf->unallocateSubRef(bufref);
    }
    m_desc.vertex_bufs.clear();
    if (m_desc.index_buf && mesh_buf) {
        mesh_buf->unallocateSubRef(m_desc.index_buf);
        m_desc.index_buf = {};
    }
    rr.dyn_buf->unallocateSubRef(m_desc.ubo_buf);
    m_desc.ubo_buf = {};
}

void ShaderDrawCore::setTexture(u32 index, std::string_view tex_key) {
    assert(index < m_desc.textures.size());
    if (index >= m_desc.textures.size()) return;
    m_desc.textures[index] = tex_key;
}

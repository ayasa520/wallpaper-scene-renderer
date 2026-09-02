#include "ShadowAtlasPass.hpp"

#include "Core/ArrayHelper.hpp"
#include "Core/Literals.hpp"
#include "Interface/IShaderValueUpdater.h"
#include "PassCommon.hpp"
#include "Resource.hpp"
#include "Scene/ShadowAtlas.hpp"
#include "Scene/SceneMaterial.h"
#include "Scene/SceneMesh.h"
#include "SkinningShaderContract.hpp"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include "Vulkan/Shader.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

using namespace wallpaper::vulkan;

namespace
{

constexpr VkDeviceSize kShadowMvpBytes = sizeof(float) * 16;

std::string ShadowVertexSource(uint32_t bone_count) {
    if (bone_count == 0) {
        return R"(
struct VSInput {
    [[vk::location(0)]] float3 Position : POSITION0;
};

struct VSOutput {
    float4 position : SV_Position;
};

[[vk::binding(0, 0)]] cbuffer ShadowCaster {
    column_major float4x4 u_MVP;
};

VSOutput main_vs(VSInput input) {
    VSOutput output;
    output.position = mul(u_MVP, float4(input.Position, 1.0));
    return output;
}
)";
    }

    return R"(
struct VSInput {
    [[vk::location(0)]] float3 Position : POSITION0;
    [[vk::location(1)]] uint4 BlendIndices : BLENDINDICES0;
    [[vk::location(2)]] float4 BlendWeights : BLENDWEIGHT0;
};

struct VSOutput {
    float4 position : SV_Position;
};

[[vk::binding(0, 0)]] cbuffer ShadowCaster {
    column_major float4x4 u_MVP;
    column_major float3x4 g_Bones[)" + std::to_string(bone_count) + R"(];
};

VSOutput main_vs(VSInput input) {
    const float4 position = float4(input.Position, 1.0);
    const float3 skinned =
        mul(g_Bones[input.BlendIndices.x], position) * input.BlendWeights.x +
        mul(g_Bones[input.BlendIndices.y], position) * input.BlendWeights.y +
        mul(g_Bones[input.BlendIndices.z], position) * input.BlendWeights.z +
        mul(g_Bones[input.BlendIndices.w], position) * input.BlendWeights.w;

    VSOutput output;
    output.position = mul(u_MVP, float4(skinned, 1.0));
    return output;
}
)";
}

constexpr std::string_view kFrag = R"(
void main_ps() {}
)";

std::optional<vvk::RenderPass> CreateShadowRenderPass(const vvk::Device& device) {
    VkAttachmentDescription attachment {
        .format         = VK_FORMAT_D32_SFLOAT,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkAttachmentReference depth_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription subpass {
        .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount    = 0,
        .pDepthStencilAttachment = &depth_ref,
    };
    std::array deps {
        VkSubpassDependency {
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = 0,
            .srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        },
        VkSubpassDependency {
            .srcSubpass    = 0,
            .dstSubpass    = VK_SUBPASS_EXTERNAL,
            .srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        },
    };
    VkRenderPassCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = static_cast<uint32_t>(deps.size()),
        .pDependencies   = deps.data(),
    };
    vvk::RenderPass pass;
    if (device.CreateRenderPass(info, pass) != VK_SUCCESS) return std::nullopt;
    return pass;
}

void WalkCasters(wallpaper::SceneNode* node,
                 const std::function<void(wallpaper::SceneNode&, wallpaper::SceneMesh&)>& fn) {
    if (node == nullptr) return;
    const auto& name = node->Name();
    if (name.rfind("volumetrics_", 0) == 0) return;
    if (auto* mesh = node->Mesh(); mesh != nullptr) {
        auto* material = mesh->Material();
        if (material != nullptr && material->modelRenderState.has_value() &&
            node->CastsShadows() && mesh->VertexCount() > 0) {
            fn(*node, *mesh);
        }
    }
    for (auto& child : node->GetChildren()) {
        WalkCasters(child.get(), fn);
    }
}

std::string ShadowVertexLayoutKey(uint32_t stride, uint32_t position_offset, uint32_t bone_count,
                                  uint32_t blend_indices_offset,
                                  uint32_t blend_weights_offset) {
    std::string key = "stride=" + std::to_string(stride) + "|pos=" +
                      std::to_string(position_offset) + "|bones=" +
                      std::to_string(bone_count);
    if (bone_count > 0) {
        key += "|indices=" + std::to_string(blend_indices_offset) + "|weights=" +
               std::to_string(blend_weights_offset);
    }
    return key;
}

VkDeviceSize ShadowUniformSize(uint32_t bone_count) {
    return kShadowMvpBytes +
           static_cast<VkDeviceSize>(bone_count) *
               static_cast<VkDeviceSize>(wallpaper::kDxcSkinningMatrixFloatCount * sizeof(float));
}

VkDeviceSize AlignUniformSize(VkDeviceSize size, VkDeviceSize alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace

ShadowAtlasPass::ShadowAtlasPass(const Desc& desc): m_desc(desc) {}

ShadowAtlasPass::~ShadowAtlasPass() = default;

std::string ShadowAtlasPass::residencyKey() const {
    return "ShadowAtlas|target=" + m_desc.target;
}

bool ShadowAtlasPass::referencesRenderTarget(std::string_view render_target) const {
    return m_desc.target == render_target;
}

bool ShadowAtlasPass::ensureClearPass(const Device& device) {
    if (m_clear_pass) return true;
    auto pass_opt = CreateShadowRenderPass(device.handle());
    if (! pass_opt.has_value()) return false;
    m_clear_pass = std::move(pass_opt.value());
    return true;
}

bool ShadowAtlasPass::ensureShadowShaders(uint32_t bone_count) {
    if (const auto it = m_shader_spvs.find(bone_count);
        it != m_shader_spvs.end() && !it->second.empty()) {
        return true;
    }

    ShaderCompOpt opt;
    opt.target_env = ShaderTargetEnv::VULKAN_1_1;
    std::vector<Uni_ShaderSpv> compiled;
    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage           = wallpaper::ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name      = "ShadowAtlas.vert[bones=" + std::to_string(bone_count) + "]",
            .entry_point     = "main_vs",
            .src             = ShadowVertexSource(bone_count),
        },
        ShaderCompUnit {
            .stage           = wallpaper::ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name      = "ShadowAtlas.frag",
            .entry_point     = "main_ps",
            .src             = std::string(kFrag),
        },
    };
    if (! CompileAndLinkShaderUnits(units, opt, compiled)) {
        LOG_ERROR("ShadowAtlas: shader compile failed bones=%u", bone_count);
        return false;
    }
    m_shader_spvs.emplace(bone_count, std::move(compiled));
    return true;
}

bool ShadowAtlasPass::ensurePipeline(const Device& device, RenderingResources& rr,
                                    const CasterMesh& caster) {
    if (caster.stride == 0) return false;
    const std::string key = ShadowVertexLayoutKey(caster.stride,
                                                  caster.position_offset,
                                                  caster.bone_count,
                                                  caster.blend_indices_offset,
                                                  caster.blend_weights_offset);
    if (auto it = m_pipelines.find(key); it != m_pipelines.end() && it->second.handle) return true;
    if (! ensureShadowShaders(caster.bone_count)) return false;
    const auto shader_it = m_shader_spvs.find(caster.bone_count);
    if (shader_it == m_shader_spvs.end()) return false;

    auto pass_opt = CreateShadowRenderPass(device.handle());
    if (! pass_opt.has_value()) return false;
    auto pass = std::move(pass_opt.value());

    VkVertexInputBindingDescription bind {
        .binding   = 0,
        .stride    = caster.stride,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    std::vector<VkVertexInputAttributeDescription> attrs {
        VkVertexInputAttributeDescription {
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = caster.position_offset,
        },
    };
    if (caster.bone_count > 0) {
        attrs.push_back(VkVertexInputAttributeDescription {
            .location = 1,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32A32_UINT,
            .offset   = caster.blend_indices_offset,
        });
        attrs.push_back(VkVertexInputAttributeDescription {
            .location = 2,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset   = caster.blend_weights_offset,
        });
    }

    DescriptorSetInfo descriptor_info;
    descriptor_info.push_descriptor = true;
    descriptor_info.bindings        = {
        VkDescriptorSetLayoutBinding {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT,
        },
    };

    PipelineParameters params;
    GraphicsPipeline   pipeline;
    pipeline.toDefault();
    // The atlas stores reversed depth (near = 1, far = 0) like the scene, so the nearest caster
    // wins with GREATER against a 0-cleared tile.
    pipeline.depth.depthTestEnable  = VK_TRUE;
    pipeline.depth.depthWriteEnable = VK_TRUE;
    pipeline.depth.depthCompareOp   = VK_COMPARE_OP_GREATER;
    // A dedicated slope-scaled rasterizer bias pushes casters away from the light while the atlas
    // is drawn; with reversed depth "away" is the negative direction. A constant-only clip-space
    // offset cannot follow the receiver slope and produces regular self-shadow bands on large
    // oblique surfaces such as roads.
    pipeline.raster.depthBiasEnable      = VK_TRUE;
    pipeline.raster.depthBiasSlopeFactor = -4.0f;
    params.debug_name = "ShadowAtlas";
    params.cache_key  = "ShadowAtlas|d32|greater|slope-bias|no-color|" + key;
    pipeline.addDescriptorSetInfo(spanone { descriptor_info })
        .setColorBlendStates(std::span<const VkPipelineColorBlendAttachmentState>())
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addInputBindingDescription(spanone { bind })
        .addInputAttributeDescription(attrs);
    for (const auto& spv : shader_it->second) {
        if (! spv) return false;
        pipeline.addStage(std::make_unique<ShaderSpv>(*spv));
    }
    if (! pipeline.create(device, pass, params, rr.pipeline_cache.get())) {
        LOG_ERROR("ShadowAtlas: pipeline create failed %s", key.c_str());
        return false;
    }
    m_pipelines[key] = std::move(params);
    return true;
}

bool ShadowAtlasPass::ensureFramebuffer(const Device& device) {
    const VkExtent2D extent { m_desc.vk_target.extent.width, m_desc.vk_target.extent.height };
    if (m_fb && m_fb_extent.width == extent.width && m_fb_extent.height == extent.height) {
        return true;
    }
    m_fb.reset();
    if (! m_clear_pass || m_desc.vk_target.view == VK_NULL_HANDLE) return false;
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass      = *m_clear_pass,
        .attachmentCount = 1,
        .pAttachments    = &m_desc.vk_target.view,
        .width           = extent.width,
        .height          = extent.height,
        .layers          = 1,
    };
    if (device.handle().CreateFramebuffer(info, m_fb) != VK_SUCCESS) return false;
    m_fb_extent = extent;
    return true;
}

void ShadowAtlasPass::releaseCasters() { m_casters.clear(); }

void ShadowAtlasPass::collectCasters(Scene& scene, const Device& device, RenderingResources& rr) {
    releaseCasters();
    if (scene.sceneGraph == nullptr) return;

    WalkCasters(scene.sceneGraph.get(), [&](SceneNode& node, SceneMesh& mesh) {
        if (mesh.VertexCount() == 0) return;
        const auto& vertex = mesh.GetVertexArray(0);
        const auto  attrs  = vertex.GetAttrOffsetMap();
        const auto  position_it = attrs.find(std::string(wallpaper::WE_IN_POSITION));
        if (position_it == attrs.end() || vertex.VertexCount() == 0) return;

        const uint32_t bone_count = mesh.Skinning().boneCount;
        auto blend_indices_it = attrs.end();
        auto blend_weights_it = attrs.end();
        if (bone_count > 0) {
            blend_indices_it = attrs.find(std::string(wallpaper::WE_IN_BLENDINDICES));
            blend_weights_it = attrs.find(std::string(wallpaper::WE_IN_BLENDWEIGHTS));
            if (blend_indices_it == attrs.end() || blend_weights_it == attrs.end() ||
                blend_indices_it->second.attr.type != wallpaper::VertexType::UINT4 ||
                blend_weights_it->second.attr.type != wallpaper::VertexType::FLOAT4) {
                LOG_ERROR("ShadowAtlas: invalid skinning vertex layout node='%s' bones=%u",
                          node.Name().c_str(),
                          bone_count);
                return;
            }
        }

        auto gpu = rr.immutable_meshes.getOrCreate(device, mesh);
        if (! gpu || gpu->vertices.empty() || ! gpu->vertices.front()) return;

        CasterMesh caster;
        caster.node             = &node;
        caster.mesh             = std::move(gpu);
        caster.stride           = static_cast<uint32_t>(vertex.OneSizeOf());
        caster.position_offset  = static_cast<uint32_t>(position_it->second.offset);
        caster.bone_count       = bone_count;
        if (bone_count > 0) {
            caster.blend_indices_offset =
                static_cast<uint32_t>(blend_indices_it->second.offset);
            caster.blend_weights_offset =
                static_cast<uint32_t>(blend_weights_it->second.offset);
        }
        caster.vertex_count     = static_cast<uint32_t>(vertex.VertexCount());
        caster.index_element_bytes = mesh.IndexElementBytes();
        caster.index_count      = mesh.LogicalIndexCount();
        caster.indexed          = caster.mesh->has_index && caster.index_count > 0;
        m_casters.push_back(std::move(caster));
    });
}

void ShadowAtlasPass::rebuildDrawList() {
    m_draws.clear();
    auto* scene = m_desc.scene;
    if (scene == nullptr) return;

    // Resolve one immutable palette per caster before expanding it across lights and cascades.
    // Every generated draw then reads the same pose that visible materials received after
    // PrepareFrame(), while shared puppet runtimes remain owned exclusively by the scene updater.
    for (auto& caster : m_casters) {
        caster.skinning_pose     = {};
        caster.pose_revision     = 0;
        caster.pose_frame_serial = 0;
        if (caster.bone_count == 0) continue;

        const auto pose = scene->shaderValueUpdater != nullptr
                              ? scene->shaderValueUpdater->SkinningPose(caster.node)
                              : std::nullopt;
        if (!pose.has_value() || pose->matrices.size() != caster.bone_count) {
            if (!caster.pose_error_reported) {
                LOG_ERROR("ShadowAtlas: skinning pose mismatch node='%s' expected=%u actual=%zu",
                          caster.node != nullptr ? caster.node->Name().c_str() : "<null>",
                          caster.bone_count,
                          pose.has_value() ? pose->matrices.size() : 0);
                caster.pose_error_reported = true;
            }
            continue;
        }

        caster.pose_error_reported = false;
        caster.skinning_pose       = pose->matrices;
        caster.pose_revision       = pose->revision;
        caster.pose_frame_serial   = pose->frame_serial;
    }

    const DirectionalShadowView cascade_view = scene->ShadowCascadeView();

    for (auto& light_ptr : scene->lights) {
        if (! light_ptr) continue;
        SceneLight& light = *light_ptr;
        const Eigen::Vector3f origin = light.WorldOrigin();

        auto emit_tile = [&](const SceneLight::ShadowAtlasSlot& atlas, const Eigen::Matrix4f& mvp_light,
                             bool point_faces) {
            if (! atlas.packed) return;
            for (uint32_t ci = 0; ci < m_casters.size(); ++ci) {
                const auto& caster = m_casters[ci];
                auto*       node   = caster.node;
                if (node == nullptr || ! node->Visible()) continue;
                if (caster.bone_count > 0 && caster.skinning_pose.empty()) continue;
                node->UpdateTrans();
                Eigen::Matrix4f model = node->ModelTrans().cast<float>();
                if (auto* mesh = node->Mesh(); mesh != nullptr) {
                    model = model * mesh->GeometryTransform().matrix();
                }
                if (point_faces) {
                    const Eigen::Matrix4f proj = ShadowPointProjection(
                        light.ShadowNearPlane(), light.ShadowFarPlane(),
                        ShadowPointFovDegrees(atlas.quality), true);
                    for (int face = 0; face < 6; ++face) {
                        const auto vp = ShadowPointFaceViewport(atlas.x, atlas.y, atlas.size, face);
                        DrawItem item;
                        item.caster_index = ci;
                        item.vp_x         = vp.x;
                        item.vp_y         = vp.y;
                        item.vp_w         = vp.width;
                        item.vp_h         = vp.height;
                        item.scissor_x    = vp.scissor_x;
                        item.scissor_y    = vp.scissor_y;
                        item.scissor_w    = vp.scissor_w;
                        item.scissor_h    = vp.scissor_h;
                        item.mvp          = proj * ShadowPointView(face, origin) * model;
                        m_draws.push_back(item);
                    }
                } else {
                    const auto vp = ShadowSpotViewport(atlas.x, atlas.y, atlas.size);
                    DrawItem item;
                    item.caster_index = ci;
                    item.vp_x         = vp.x;
                    item.vp_y         = vp.y;
                    item.vp_w         = vp.width;
                    item.vp_h         = vp.height;
                    item.scissor_x    = vp.scissor_x;
                    item.scissor_y    = vp.scissor_y;
                    item.scissor_w    = vp.scissor_w;
                    item.scissor_h    = vp.scissor_h;
                    item.mvp          = mvp_light * model;
                    m_draws.push_back(item);
                }
            }
        };

        if (light.type() == SceneLightType::Directional) {
            for (int cascade = 0; cascade < 3; ++cascade) {
                const auto& atlas = light.cascadeAtlasSlot(cascade);
                emit_tile(atlas,
                          light.ShadowCascadeWorldToLightClip(
                              cascade, cascade_view, atlas.size, true),
                          false);
            }
            continue;
        }

        const auto& atlas = light.shadowAtlasSlot();
        if (! atlas.packed) continue;
        if (atlas.point) {
            emit_tile(atlas, Eigen::Matrix4f::Identity(), true);
        } else {
            emit_tile(atlas, light.ShadowWorldToLightClip(), false);
        }
    }
}

void ShadowAtlasPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    m_desc.scene = &scene;
    m_dyn_buf    = rr.dyn_buf;
    m_ubo_align  = std::max(device.limits().minUniformBufferOffsetAlignment, VkDeviceSize { 64 });

    if (scene.renderTargets.count(m_desc.target) == 0) {
        LOG_ERROR("ShadowAtlas: missing target '%s'", m_desc.target.c_str());
        return;
    }
    const auto& rt = scene.renderTargets.at(m_desc.target);
    auto        opt = device.tex_cache().Query(m_desc.target, ToTexKey(rt), ! rt.allowReuse);
    if (! opt.has_value()) {
        LOG_ERROR("ShadowAtlas: query target failed");
        return;
    }
    m_desc.vk_target = opt.value();
    if (! ensureClearPass(device)) return;
    collectCasters(scene, device, rr);
    for (const auto& caster : m_casters) {
        if (! ensurePipeline(device, rr, caster)) return;
    }
    if (! ensureFramebuffer(device)) return;
    setPrepared();
}

void ShadowAtlasPass::refreshResources(Scene& scene, const Device& device, RenderingResources&) {
    m_desc.scene = &scene;
    if (scene.renderTargets.count(m_desc.target) == 0) {
        setPrepared(false);
        return;
    }
    const auto& rt = scene.renderTargets.at(m_desc.target);
    auto        opt = device.tex_cache().Query(m_desc.target, ToTexKey(rt), ! rt.allowReuse);
    if (! opt.has_value()) {
        setPrepared(false);
        return;
    }
    m_desc.vk_target = opt.value();
    m_fb.reset();
    m_fb_extent = {};
}

void ShadowAtlasPass::updateBeforeUpload() {
    rebuildDrawList();
    if (m_dyn_buf == nullptr) return;

    VkDeviceSize bytes = 0;
    for (auto& draw : m_draws) {
        if (draw.caster_index >= m_casters.size()) continue;
        draw.ubo_offset = bytes;
        draw.ubo_size   = ShadowUniformSize(m_casters[draw.caster_index].bone_count);
        bytes += AlignUniformSize(draw.ubo_size, m_ubo_align);
    }
    if (bytes == 0) bytes = m_ubo_align;

    if (m_ubo_buf && m_ubo_buf.size < bytes) {
        m_dyn_buf->unallocateSubRef(m_ubo_buf);
    }
    if (! m_ubo_buf) {
        if (! m_dyn_buf->allocateSubRef(bytes, m_ubo_buf, m_ubo_align)) {
            LOG_ERROR("ShadowAtlas: ubo allocate failed draws=%zu", m_draws.size());
            m_draws.clear();
            return;
        }
    }
    for (const auto& draw : m_draws) {
        if (draw.caster_index >= m_casters.size()) continue;
        const auto& caster = m_casters[draw.caster_index];
        const auto  offset = static_cast<size_t>(draw.ubo_offset);
        Eigen::Matrix4f mvp = draw.mvp;
        m_dyn_buf->writeToBuf(m_ubo_buf,
                              { reinterpret_cast<uint8_t*>(mvp.data()),
                                static_cast<size_t>(kShadowMvpBytes) },
                              offset);
        if (caster.bone_count == 0) continue;

        auto packed_pose = wallpaper::PackDxcRowVectorSkinningUniform(caster.skinning_pose);
        m_dyn_buf->writeToBuf(
            m_ubo_buf,
            { reinterpret_cast<uint8_t*>(packed_pose.data()),
              packed_pose.size() * sizeof(float) },
            offset + static_cast<size_t>(kShadowMvpBytes));
    }
}

void ShadowAtlasPass::execute(const Device& device, RenderingResources& rr) {
    if (m_desc.vk_target.handle == VK_NULL_HANDLE) return;
    // First bind always far-clears the atlas (reversed depth 0.0), including zero casters.
    // Empty tiles stay at far so GREATER comparison sampling keeps the unoccluded path.
    if (! ensureClearPass(device) || ! ensureFramebuffer(device) || ! m_fb) return;

    const VkExtent2D extent { m_desc.vk_target.extent.width, m_desc.vk_target.extent.height };
    VkClearValue     clear {};
    clear.depthStencil = { 0.0f, 0 };
    VkRenderPassBeginInfo begin {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass      = *m_clear_pass,
        .framebuffer     = *m_fb,
        .renderArea      = VkRect2D { { 0, 0 }, extent },
        .clearValueCount = 1,
        .pClearValues    = &clear,
    };
    rr.command.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);

    PipelineParameters* bound = nullptr;
    for (const auto& draw : m_draws) {
        if (draw.caster_index >= m_casters.size()) continue;
        const auto& caster = m_casters[draw.caster_index];
        if (! caster.mesh || caster.mesh->vertices.empty()) continue;
        VkBuffer gpu = caster.mesh->vertices.front().handle();
        if (gpu == VK_NULL_HANDLE) continue;
        if (! ensurePipeline(device, rr, caster)) continue;
        const auto pipeline_key = ShadowVertexLayoutKey(caster.stride,
                                                        caster.position_offset,
                                                        caster.bone_count,
                                                        caster.blend_indices_offset,
                                                        caster.blend_weights_offset);
        auto pipe_it = m_pipelines.find(pipeline_key);
        if (pipe_it == m_pipelines.end() || ! pipe_it->second.handle) continue;
        auto& pipeline = pipe_it->second;
        if (bound != &pipeline) {
            rr.command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline.handle);
            bound = &pipeline;
        }

        const int32_t max_w = static_cast<int32_t>(extent.width);
        const int32_t max_h = static_cast<int32_t>(extent.height);
        const int32_t sx    = std::clamp(draw.scissor_x, 0, std::max(max_w - 1, 0));
        const int32_t sy    = std::clamp(draw.scissor_y, 0, std::max(max_h - 1, 0));
        const int32_t sw    = std::clamp(draw.scissor_w, 1, std::max(max_w - sx, 1));
        const int32_t sh    = std::clamp(draw.scissor_h, 1, std::max(max_h - sy, 1));

        VkViewport viewport {
            .x        = draw.vp_x,
            .y        = draw.vp_y,
            .width    = draw.vp_w,
            .height   = draw.vp_h,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        VkRect2D scissor {
            .offset = { sx, sy },
            .extent = { static_cast<uint32_t>(sw), static_cast<uint32_t>(sh) },
        };
        rr.command.SetViewport(0, viewport);
        rr.command.SetScissor(0, scissor);

        if (m_dyn_buf != nullptr && m_ubo_buf) {
            VkDescriptorBufferInfo buffer_info {
                .buffer = m_dyn_buf->gpuBuf(),
                .offset = m_ubo_buf.offset + draw.ubo_offset,
                .range  = draw.ubo_size,
            };
            VkWriteDescriptorSet write {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &buffer_info,
            };
            rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline.layout, 0,
                                            write);
        }

        const VkDeviceSize off = 0;
        rr.command.BindVertexBuffers(0, 1, &gpu, &off);
        if (caster.indexed && caster.mesh->has_index) {
            const VkIndexType index_type = caster.index_element_bytes == 4
                                               ? VK_INDEX_TYPE_UINT32
                                               : VK_INDEX_TYPE_UINT16;
            rr.command.BindIndexBuffer(caster.mesh->index.handle(), 0, index_type);
            rr.command.DrawIndexed(caster.index_count, 1, 0, 0, 0);
        } else {
            rr.command.Draw(caster.vertex_count, 1, 0, 0);
        }
    }
    rr.command.EndRenderPass();
}

void ShadowAtlasPass::destory(const Device&, RenderingResources&) {
    setPrepared(false);
    clearReleaseTexs();
    m_fb.reset();
    m_fb_extent = {};
    m_draws.clear();
    if (m_dyn_buf != nullptr && m_ubo_buf) m_dyn_buf->unallocateSubRef(m_ubo_buf);
    releaseCasters();
    m_pipelines.clear();
    m_shader_spvs.clear();
    m_clear_pass.reset();
    m_dyn_buf = nullptr;
}

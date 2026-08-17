#include "ShadowAtlasPass.hpp"

#include "Core/ArrayHelper.hpp"
#include "Core/Literals.hpp"
#include "PassCommon.hpp"
#include "Resource.hpp"
#include "Scene/ShadowAtlas.hpp"
#include "Scene/SceneMaterial.h"
#include "Scene/SceneMesh.h"
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

constexpr std::string_view kVert = R"(
struct VSInput {
    [[vk::location(0)]] float3 Position : POSITION0;
};

struct VSOutput {
    float4 position : SV_Position;
};

struct ShadowCaster {
    float4x4 u_MVP;
};

[[vk::binding(0, 0)]] ConstantBuffer<ShadowCaster> g_cb;

VSOutput main_vs(VSInput input) {
    VSOutput output;
    output.position = mul(g_cb.u_MVP, float4(input.Position, 1.0));
    return output;
}
)";

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

uint64_t ShadowVertexLayoutKey(uint32_t stride, uint32_t position_offset) {
    return (static_cast<uint64_t>(stride) << 32) | position_offset;
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

bool ShadowAtlasPass::ensureShadowShaders() {
    if (! m_shader_spvs.empty()) return true;

    ShaderCompOpt opt;
    opt.target_env = ShaderTargetEnv::VULKAN_1_1;
    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage           = wallpaper::ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name      = "ShadowAtlas.vert",
            .entry_point     = "main_vs",
            .src             = std::string(kVert),
        },
        ShaderCompUnit {
            .stage           = wallpaper::ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name      = "ShadowAtlas.frag",
            .entry_point     = "main_ps",
            .src             = std::string(kFrag),
        },
    };
    if (! CompileAndLinkShaderUnits(units, opt, m_shader_spvs)) {
        LOG_ERROR("ShadowAtlas: shader compile failed");
        m_shader_spvs.clear();
        return false;
    }
    return true;
}

bool ShadowAtlasPass::ensurePipeline(const Device& device, RenderingResources& rr, uint32_t stride,
                                    uint32_t position_offset) {
    if (stride == 0) return false;
    const uint64_t key = ShadowVertexLayoutKey(stride, position_offset);
    if (auto it = m_pipelines.find(key); it != m_pipelines.end() && it->second.handle) return true;
    if (! ensureShadowShaders()) return false;

    auto pass_opt = CreateShadowRenderPass(device.handle());
    if (! pass_opt.has_value()) return false;
    auto pass = std::move(pass_opt.value());

    VkVertexInputBindingDescription bind {
        .binding   = 0,
        .stride    = stride,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    VkVertexInputAttributeDescription attr {
        .location = 0,
        .binding  = 0,
        .format   = VK_FORMAT_R32G32B32_SFLOAT,
        .offset   = position_offset,
    };

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
    pipeline.depth.depthTestEnable  = VK_TRUE;
    pipeline.depth.depthWriteEnable = VK_TRUE;
    pipeline.depth.depthCompareOp   = VK_COMPARE_OP_LESS;
    params.debug_name               = "ShadowAtlas";
    params.cache_key                = "ShadowAtlas|d32|less|no-color|stride=" +
                       std::to_string(stride) + "|pos=" + std::to_string(position_offset);
    pipeline.addDescriptorSetInfo(spanone { descriptor_info })
        .setColorBlendStates(std::span<const VkPipelineColorBlendAttachmentState>())
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addInputBindingDescription(spanone { bind })
        .addInputAttributeDescription(spanone { attr });
    for (const auto& spv : m_shader_spvs) {
        if (! spv) return false;
        pipeline.addStage(std::make_unique<ShaderSpv>(*spv));
    }
    if (! pipeline.create(device, pass, params, rr.pipeline_cache.get())) {
        LOG_ERROR("ShadowAtlas: pipeline create failed stride=%u pos=%u", stride, position_offset);
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
        const auto  it     = attrs.find(std::string(wallpaper::WE_IN_POSITION));
        if (it == attrs.end() || vertex.VertexCount() == 0) return;

        auto gpu = rr.immutable_meshes.getOrCreate(device, mesh);
        if (! gpu || gpu->vertices.empty() || ! gpu->vertices.front()) return;

        CasterMesh caster;
        caster.node             = &node;
        caster.mesh             = std::move(gpu);
        caster.stride           = static_cast<uint32_t>(vertex.OneSizeOf());
        caster.position_offset  = static_cast<uint32_t>(it->second.offset);
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

    uint32_t slot = 0;
    const Eigen::Vector3f cascade_center = scene->ShadowCascadeCenter();

    for (auto& light_ptr : scene->lights) {
        if (! light_ptr) continue;
        SceneLight& light = *light_ptr;
        const Eigen::Vector3f origin = light.WorldOrigin();

        auto emit_tile = [&](const SceneLight::ShadowAtlasSlot& atlas, const Eigen::Matrix4f& mvp_light,
                             bool point_faces) {
            if (! atlas.packed) return;
            for (uint32_t ci = 0; ci < m_casters.size(); ++ci) {
                auto* node = m_casters[ci].node;
                if (node == nullptr || ! node->Visible()) continue;
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
                        item.ubo_slot     = slot++;
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
                    item.ubo_slot     = slot++;
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
                          light.ShadowCascadeWorldToLightClip(cascade, cascade_center, true),
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
        if (! ensurePipeline(device, rr, caster.stride, caster.position_offset)) return;
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
    const VkDeviceSize bytes =
        m_draws.empty() ? m_ubo_align : static_cast<VkDeviceSize>(m_draws.size()) * m_ubo_align;
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
        const auto offset = static_cast<size_t>(draw.ubo_slot) * static_cast<size_t>(m_ubo_align);
        Eigen::Matrix4f mvp = draw.mvp;
        m_dyn_buf->writeToBuf(m_ubo_buf,
                              { reinterpret_cast<uint8_t*>(mvp.data()), sizeof(float) * 16 },
                              offset);
    }
}

void ShadowAtlasPass::execute(const Device& device, RenderingResources& rr) {
    if (m_desc.vk_target.handle == VK_NULL_HANDLE) return;
    // First bind always far-clears the atlas (depth 1.0), including zero casters.
    // Empty tiles stay at far so comparison sampling keeps the unoccluded path.
    if (! ensureClearPass(device) || ! ensureFramebuffer(device) || ! m_fb) return;

    const VkExtent2D extent { m_desc.vk_target.extent.width, m_desc.vk_target.extent.height };
    VkClearValue     clear {};
    clear.depthStencil = { 1.0f, 0 };
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
        if (! ensurePipeline(device, rr, caster.stride, caster.position_offset)) continue;
        auto pipe_it = m_pipelines.find(ShadowVertexLayoutKey(caster.stride, caster.position_offset));
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
                .offset = m_ubo_buf.offset + static_cast<VkDeviceSize>(draw.ubo_slot) * m_ubo_align,
                .range  = sizeof(float) * 16,
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
    m_clear_pass.reset();
    m_dyn_buf = nullptr;
}

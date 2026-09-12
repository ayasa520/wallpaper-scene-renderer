#include "MaskedDrawRenderer.hpp"

#include "Core/ArrayHelper.hpp"
#include "Interface/IImageParser.h"
#include "PassCommon.hpp"
#include "Utils/Logging.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

namespace wallpaper::vulkan
{
namespace
{

using MaskMaterials = SceneMesh::MaskedDrawMaterials;

bool SameRange(const SceneMesh::DrawRange& lhs, const SceneMesh::DrawRange& rhs) {
    return lhs.firstIndex == rhs.firstIndex && lhs.indexCount == rhs.indexCount;
}

bool SameRanges(const std::vector<SceneMesh::DrawRange>& lhs,
                const std::vector<SceneMesh::DrawRange>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
        if (!SameRange(lhs[i], rhs[i])) return false;
    }
    return true;
}

void PublishCoverage(RenderingResources& resources, const VmaImageParameters& image) {
    // Mask and composite render passes finish in shader-read layout. Publish their color
    // writes before a later fragment program samples the image. Incoming render-pass
    // dependencies separately order subsequent attachment loads/clears after those reads.
    VkImageMemoryBarrier barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image.handle,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    resources.command.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, barrier);
}

} // namespace

bool MaskedDrawRenderer::configure(const Device& device, const ShaderDrawData& data,
                                   const SceneMesh& mesh) {
    const auto& plan = mesh.MaskedDraw();
    if (plan.empty() || plan.materials == nullptr || data.model_pass ||
        mesh.IndexCount() == 0 || plan.orderedRanges.empty()) {
        LOG_ERROR("MaskedDrawPrepare: invalid mesh/program contract node='%s' groups=%zu",
                  data.draw.Name().c_str(), plan.groups.size());
        return false;
    }
    const auto features = device.gpu().GetFormatProperties(VK_FORMAT_R8_UNORM).optimalTilingFeatures;
    constexpr auto required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if ((features & required) != required) {
        LOG_ERROR("MaskedDrawPrepare: R8 color coverage unsupported node='%s' features=%u",
                  data.draw.Name().c_str(), features);
        return false;
    }
    m_materials = plan.materials;
    m_has_parents = std::any_of(plan.groups.begin(), plan.groups.end(), [](const auto& group) {
        return group.coverageGroups.size() > 1;
    });
    m_inverted.clear();
    for (size_t i = 0; i < plan.groups.size(); ++i) {
        const auto& group = plan.groups[i];
        if (group.maskTexture.empty() || group.maskRanges.empty() || group.contentRanges.empty()) {
            LOG_ERROR("MaskedDrawPrepare: invalid group node='%s' group=%zu texture='%s'",
                      data.draw.Name().c_str(), i, group.maskTexture.c_str());
            return false;
        }
        m_inverted.push_back(group.inverted);
    }
    for (const auto& ordered : plan.orderedRanges) {
        if (ordered.groupIndex < -1 || (ordered.groupIndex >= 0 &&
            static_cast<size_t>(ordered.groupIndex) >= plan.groups.size())) {
            LOG_ERROR("MaskedDrawPrepare: invalid range group node='%s' group=%d groups=%zu",
                      data.draw.Name().c_str(), ordered.groupIndex, plan.groups.size());
            return false;
        }
    }
    return true;
}

std::vector<std::string_view> MaskedDrawRenderer::resourceTextures(const SceneMesh& mesh) const {
    std::vector<std::string_view> textures;
    for (const auto& group : mesh.MaskedDraw().groups) textures.push_back(group.maskTexture);
    return textures;
}

bool MaskedDrawRenderer::refreshTextures(Scene& scene, const Device& device,
                                         const ShaderDrawData& data) {
    const auto& groups = data.draw.Mesh()->MaskedDraw().groups;
    m_textures.resize(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& name = groups[i].maskTexture;
        if (scene.dirtyImportedTextureKeys.count(name) == 0) {
            if (auto cached = device.tex_cache().FindTex(name); cached.has_value()) {
                m_textures[i] = *cached;
                continue;
            }
        } else {
            scene.DropParsedImageCache(name);
        }
        const auto texture = scene.textures.find(name);
        if (texture == scene.textures.end() || texture->second.isVideo) {
            LOG_ERROR("MaskedDrawTexture: invalid imported mask node='%s' group=%zu texture='%s'",
                      data.draw.Name().c_str(), i, name.c_str());
            return false;
        }
        auto image = scene.GetParsedImageIfReady(name);
        if (image == nullptr) image = scene.ParseImageBlockingCached(name);
        if (image == nullptr) {
            LOG_ERROR("MaskedDrawTexture: parse failed node='%s' group=%zu texture='%s'",
                      data.draw.Name().c_str(), i, name.c_str());
            return false;
        }
        auto slots = device.tex_cache().CreateTex(*image);
        scene.DropParsedImageCache(name);
        if (slots.slots.empty()) {
            LOG_ERROR("MaskedDrawTexture: upload failed node='%s' group=%zu texture='%s'",
                      data.draw.Name().c_str(), i, name.c_str());
            return false;
        }
        m_textures[i] = std::move(slots);
    }
    return true;
}

bool MaskedDrawRenderer::prepareProgram(const Device& device, RenderingResources& resources,
                                         const ShaderDrawPipelineContext& context,
                                         const SceneMaterial& material, bool mask,
                                         Program& program) {
    const std::vector<BlendMode> blends = mask ? std::vector {material.blenmode}
        : std::vector {BlendMode::Translucent, BlendMode::Additive};
    program.pipelines.resize(blends.size());
    for (size_t index = 0; index < blends.size(); ++index) {
        std::vector<Uni_ShaderSpv> stages;
        if (!ReflectMaskedDrawShaderContract(context.mesh, material, stages, program.contract)) {
            return false;
        }
        const auto& contract = program.contract;
        if ((mask && (contract.texture_bindings[0] < 0 ||
                      contract.texture_bindings[MaskMaterials::MaskTextureSlot] < 0)) ||
            (!mask && contract.texture_bindings[MaskMaterials::CoverageTextureSlot] < 0)) {
            LOG_ERROR("MaskedDrawPrepare: missing coverage sampler node='%s' shader='%s' mask=%s",
                      context.data.draw.Name().c_str(), material.name.c_str(), mask ? "true" : "false");
            return false;
        }
        for (size_t slot = 0; slot < contract.texture_bindings.size(); ++slot) {
            if (contract.texture_bindings[slot] < 0 ||
                (mask && slot == MaskMaterials::MaskTextureSlot) ||
                (!mask && slot == MaskMaterials::CoverageTextureSlot)) continue;
            if (slot >= context.data.vk_textures.size() ||
                context.data.vk_textures[slot].slots.empty()) {
                LOG_ERROR("MaskedDrawPrepare: missing input node='%s' shader='%s' slot=%zu",
                          context.data.draw.Name().c_str(), material.name.c_str(), slot);
                return false;
            }
        }

        // The clipped material retains the invocation's destination alpha and raster route,
        // but the group owns its translucent/additive blend selection. The mask writer has
        // a separate R8 target and material; it never inherits scene depth or clears color
        // belonging to an earlier visible range.
        ShaderDrawData invocation;
        static_cast<ShaderDrawRequest&>(invocation) = context.data;
        invocation.sample_count = mask ? VK_SAMPLE_COUNT_1_BIT : context.data.sample_count;
        invocation.resolve_msaa = !mask && context.data.resolve_msaa;
        invocation.alpha_to_coverage = !mask && material.alpha_to_coverage;
        invocation.blend_override = blends[index];
        invocation.clear_before_draw = false;
        if (mask) {
            invocation.depth_test = false;
            invocation.depth_write = false;
        }
        VkPipelineColorBlendAttachmentState blend {};
        if (mask) {
            SetBlend(material.blenmode, blend);
            blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        } else {
            blend = BuildShaderDrawRenderState(material, invocation).color_blend;
        }
        const auto load = mask ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        auto pass = CreateShaderDrawRenderPass(device.handle(),
            mask ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8B8A8_UNORM, load,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {},
            invocation.sample_count, invocation.resolve_msaa);
        if (!pass) return false;

        GraphicsPipeline pipeline;
        pipeline.toDefault();
        pipeline.multisample.rasterizationSamples = invocation.sample_count;
        pipeline.multisample.alphaToCoverageEnable =
            invocation.alpha_to_coverage && invocation.sample_count > VK_SAMPLE_COUNT_1_BIT;
        ApplyShaderDrawMaterialPipelineState(material, invocation, pipeline);
        pipeline.addDescriptorSetInfo(spanone {contract.descriptors})
            .setColorBlendStates(spanone {blend})
            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .addInputBindingDescription(contract.bindings)
            .addInputAttributeDescription(contract.attributes);
        for (auto& stage : stages) pipeline.addStage(std::move(stage));
        auto& prepared = program.pipelines[index];
        prepared.debug_name = std::string(mask ? "MaskedDrawCoverage" : "MaskedDrawClipped") +
            "[node=" + context.data.draw.Name() + ",output=" + context.data.output + "]";
        prepared.cache_key = mask ? "MaskedDrawCoverage|format=r8|load=clear|samples=1"
            : "MaskedDrawClipped|" + ShaderDrawPipelineCompatibilityKey(load, false,
                VK_ATTACHMENT_LOAD_OP_DONT_CARE, {}, invocation.sample_count,
                invocation.resolve_msaa);
        if (!pipeline.create(device, *pass, prepared, resources.pipeline_cache.get())) return false;
    }

    if (!program.contract.reflection.blocks.empty()) {
        // All groups are recorded before the shared staging upload is consumed by the GPU.
        // Inversion therefore needs one stable uniform range per group, not repeated CPU
        // writes into an address already referenced by an earlier mask command.
        program.uniforms.resize(mask ? m_inverted.size() : 1);
        for (auto& uniform : program.uniforms) {
            if (!resources.dyn_buf->allocateSubRef(program.contract.reflection.blocks.front().size,
                    uniform, device.limits().minUniformBufferOffsetAlignment)) return false;
        }
    }
    return true;
}

bool MaskedDrawRenderer::preparePipelines(const Device& device, RenderingResources& resources,
                                           const ShaderDrawPipelineContext& context) {
    return prepareProgram(device, resources, context, m_materials->mask, true, m_mask) &&
        prepareProgram(device, resources, context, m_materials->clipped, false, m_clipped) &&
        (!m_has_parents || m_composite.prepare(device, resources));
}

bool MaskedDrawRenderer::refreshFramebuffers(const Device& device, RenderingResources& resources,
                                              const ShaderDrawData& data) {
    dropFramebuffers();
    using Role = MaskedDrawAttachmentCache::Role;
    m_coverage = resources.masked_draw_attachments.acquire(device, data.output,
        data.vk_output.extent, Role::Accumulated);
    if (m_coverage == nullptr) return false;
    VkImageView view = *m_coverage->view;
    VkFramebufferCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = *m_mask.pipelines.front().pass,
        .attachmentCount = 1,
        .pAttachments = &view,
        .width = data.vk_output.extent.width,
        .height = data.vk_output.extent.height,
        .layers = 1,
    };
    VVK_CHECK_ACT(return false, device.handle().CreateFramebuffer(info, m_mask_framebuffer));
    if (m_has_parents) {
        m_intermediate = resources.masked_draw_attachments.acquire(device, data.output,
            data.vk_output.extent, Role::Intermediate);
        if (m_intermediate == nullptr) return false;
        view = *m_intermediate->view;
        VVK_CHECK_ACT(return false,
                      device.handle().CreateFramebuffer(info, m_intermediate_framebuffer));
    }
    return true;
}

void MaskedDrawRenderer::dropFramebuffers() {
    m_mask_framebuffer.reset();
    m_intermediate_framebuffer.reset();
    m_coverage.reset();
    m_intermediate.reset();
}

bool MaskedDrawRenderer::hasUniform(std::string_view name) const {
    return m_mask.contract.hasUniform(name) || m_clipped.contract.hasUniform(name);
}

void MaskedDrawRenderer::initializeUniforms(StagingBuffer* buffer) {
    for (auto* program : {&m_mask, &m_clipped}) {
        for (const auto& uniform : program->uniforms) buffer->fillBuf(uniform, 0, uniform.size, 0);
    }
}

void MaskedDrawRenderer::updateMaterialUniforms(StagingBuffer* buffer,
                                                const SceneMaterial& invocation) {
    const auto write = [&](const Program& program, const auto& values) {
        if (program.uniforms.empty()) return;
        const auto& block = program.contract.reflection.blocks.front();
        for (const auto& [name, value] : values) {
            for (const auto& uniform : program.uniforms) {
                UpdateShaderDrawUniform(buffer, uniform, block, name, value);
            }
        }
    };
    write(m_mask, m_materials->mask.customShader.shader->default_uniforms);
    write(m_mask, m_materials->mask.customShader.constValues);
    write(m_clipped, m_materials->clipped.customShader.shader->default_uniforms);
    write(m_clipped, m_materials->clipped.customShader.constValues);
    // Per-invocation values include the resolved publication modulation or the current direct
    // material controls. They override the clipped executable's initial defaults without
    // mutating a program shared by another draw or advancing its canonical owner pose.
    write(m_clipped, invocation.customShader.constValues);
    for (size_t group = 0; group < m_inverted.size(); ++group) {
        UpdateShaderDrawUniform(buffer, m_mask.uniforms[group],
            m_mask.contract.reflection.blocks.front(), G_RV0,
            std::array<float, 4> {m_inverted[group] ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f});
    }
    ++m_uniform_epoch;
}

void MaskedDrawRenderer::updateUniform(StagingBuffer* buffer, std::string_view name,
                                        const ShaderValue& value) {
    for (auto* program : {&m_mask, &m_clipped}) {
        if (!program->contract.hasUniform(name) || (program == &m_mask && name == G_RV0)) continue;
        for (const auto& uniform : program->uniforms) {
            UpdateShaderDrawUniform(buffer, uniform, program->contract.reflection.blocks.front(),
                                    name, value);
        }
    }
    if (name == G_BONES && std::getenv("WESCENE_TRACE_MASKED_DRAW") != nullptr) {
        // This diagnostic fingerprints the exact shared pose bytes sent to both programs.
        // It is not another pose cache and performs no work outside the opt-in trace.
        m_pose_hash = 14695981039346656037ull;
        const auto* bytes = reinterpret_cast<const uint8_t*>(value.data());
        for (size_t i = 0; i < value.size() * sizeof(ShaderValue::value_type); ++i) {
            m_pose_hash = (m_pose_hash ^ bytes[i]) * 1099511628211ull;
        }
    }
}

void MaskedDrawRenderer::recordIndexed(const ShaderDrawRecordContext& context) {
    const auto& data = context.data;
    auto& command = context.resources.command;
    const auto& plan = data.draw.Mesh()->MaskedDraw();
    const auto& extent = data.vk_output.extent;
    ++m_draw_sequence;
    const bool trace = std::getenv("WESCENE_TRACE_MASKED_DRAW") != nullptr &&
        (m_draw_sequence % 120 == 1 || std::getenv("WESCENE_TRACE_DRAW_EVERY_FRAME") != nullptr);
    const auto trace_range = [&](const char* role, const SceneMesh::DrawRange& range,
                                  int32_t group, const char* shader,
                                  const VmaImageParameters* target = nullptr) {
        if (!trace) return;
        LOG_INFO("MaskedDrawCommand: sequence=%llu time=%.6f uniform-epoch=%llu pose=%016llx "
                 "layer=%d node='%s' role=%s shader='%s' group=%d first=%u count=%u "
                 "destination=%p coverage=%p extent=%ux%u samples=%u attachment=%p",
                 static_cast<unsigned long long>(m_draw_sequence),
                 data.scene->elapsingTime, static_cast<unsigned long long>(m_uniform_epoch),
                 static_cast<unsigned long long>(m_pose_hash), data.layer_id,
                 data.draw.Name().c_str(), role, shader, group, range.firstIndex, range.indexCount,
                 reinterpret_cast<void*>(data.vk_output.handle),
                 reinterpret_cast<void*>(*m_coverage->handle), extent.width, extent.height,
                 static_cast<unsigned>(data.sample_count),
                 reinterpret_cast<void*>(target != nullptr ? *target->handle : data.vk_output.handle));
    };
    const auto push_descriptors = [&](const Program& program, size_t pipeline_index,
                                       size_t group_index, bool mask) {
        const auto layout = *program.pipelines[pipeline_index].layout;
        for (size_t slot = 0; slot < program.contract.texture_bindings.size(); ++slot) {
            const auto binding = program.contract.texture_bindings[slot];
            if (binding < 0) continue;
            const ImageParameters image =
                mask && slot == MaskMaterials::MaskTextureSlot ? m_textures[group_index].getActive()
                : !mask && slot == MaskMaterials::CoverageTextureSlot ? ImageParameters(*m_coverage)
                : data.vk_textures[slot].getActive();
            VkDescriptorImageInfo image_info {
                image.sampler, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            VkWriteDescriptorSet write {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = static_cast<uint32_t>(binding),
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_info,
            };
            command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, write);
        }
        if (!program.uniforms.empty()) {
            const auto& uniform = program.uniforms[mask ? group_index : 0];
            VkDescriptorBufferInfo buffer_info {
                context.resources.dyn_buf->gpuBuf(), uniform.offset, uniform.size,
            };
            VkWriteDescriptorSet write {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = program.contract.uniform_binding,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_info,
            };
            command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, write);
        }
    };

    enum class BoundPipeline { Visible, Mask, Clipped };
    auto bound = BoundPipeline::Visible;
    size_t bound_clipped = 0;
    int32_t active_group = -1;
    // Preserve the imported part schedule, including ordinary ranges between clipped ranges.
    // The scratch mask is consumed immediately. Pre-rendering every group into this shared
    // image would make earlier content sample the final group's coverage instead of its own.
    for (const auto& ordered : plan.orderedRanges) {
        const auto& range = ordered.range;
        if (range.indexCount == 0) continue;
        if (ordered.groupIndex < 0) {
            if (bound != BoundPipeline::Visible) {
                command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *data.pipeline.handle);
                context.push_visible_descriptors(*data.pipeline.layout);
                bound = BoundPipeline::Visible;
            }
            trace_range("visible", range, -1, data.draw.Mesh()->Material()->name.c_str());
            command.DrawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
            continue;
        }
        const auto group_index = static_cast<size_t>(ordered.groupIndex);
        const auto& group = plan.groups[group_index];
        const size_t clipped_index = group.blend == BlendMode::Additive ? 1 : 0;
        if (active_group != ordered.groupIndex) {
            command.EndRenderPass();
            // The parser supplies immediate-parent-to-outer order, with the current mask
            // last. Every writer needs its own clear and source-alpha blend before its
            // result participates in the product. Combining raw writer fragments directly
            // in accumulated coverage would change translucent and inverted masks.
            for (size_t stage = 0; stage < group.coverageGroups.size(); ++stage) {
                const auto source_index = group.coverageGroups[stage];
                const auto& source_group = plan.groups[source_index];
                auto& target = stage == 0 ? m_coverage : m_intermediate;
                VkClearValue clear {
                    .color = {{source_group.inverted ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f}},
                };
                VkRenderPassBeginInfo mask_begin {
                    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    .renderPass = *m_mask.pipelines.front().pass,
                    .framebuffer = stage == 0 ? *m_mask_framebuffer : *m_intermediate_framebuffer,
                    .renderArea = {{0, 0}, {extent.width, extent.height}},
                    .clearValueCount = 1,
                    .pClearValues = &clear,
                };
                command.BeginRenderPass(mask_begin, VK_SUBPASS_CONTENTS_INLINE);
                command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      *m_mask.pipelines.front().handle);
                push_descriptors(m_mask, 0, source_index, true);
                if (trace) {
                    LOG_INFO("MaskedDrawInputs: sequence=%llu layer=%d node='%s' group=%u "
                             "identity=%llu inverted=%s blend=%d albedo=%p mask=%p "
                             "mask-uniform-offset=%llu clipped-uniform-offset=%llu "
                             "consumer-group=%zu stage=%zu attachment=%p",
                             static_cast<unsigned long long>(m_draw_sequence), data.layer_id,
                             data.draw.Name().c_str(), source_index,
                             static_cast<unsigned long long>(source_group.identity),
                             source_group.inverted ? "true" : "false",
                             static_cast<int>(source_group.blend),
                             reinterpret_cast<void*>(data.vk_textures[0].getActive().handle),
                             reinterpret_cast<void*>(m_textures[source_index].getActive().handle),
                             static_cast<unsigned long long>(m_mask.uniforms[source_index].offset),
                             static_cast<unsigned long long>(m_clipped.uniforms.empty()
                                 ? 0 : m_clipped.uniforms.front().offset),
                             group_index, stage, reinterpret_cast<void*>(*target->handle));
                }
                for (const auto& mask_range : source_group.maskRanges) {
                    if (mask_range.indexCount == 0) continue;
                    trace_range("mask", mask_range, static_cast<int32_t>(source_index),
                                m_materials->mask.name.c_str(), target.get());
                    command.DrawIndexed(mask_range.indexCount, 1, mask_range.firstIndex, 0, 0);
                }
                command.EndRenderPass();
                PublishCoverage(context.resources, *target);

                if (stage != 0) {
                    m_composite.record(context.resources, *m_mask_framebuffer, extent,
                                       ImageParameters(*m_intermediate));
                    if (trace) {
                        LOG_INFO("MaskedDrawComposite: sequence=%llu layer=%d node='%s' group=%zu "
                                 "source-group=%u stage=%zu intermediate=%p coverage=%p "
                                 "extent=%ux%u",
                                 static_cast<unsigned long long>(m_draw_sequence), data.layer_id,
                                 data.draw.Name().c_str(), group_index, source_index, stage,
                                 reinterpret_cast<void*>(*m_intermediate->handle),
                                 reinterpret_cast<void*>(*m_coverage->handle),
                                 extent.width, extent.height);
                    }
                }
            }
            if (group.coverageGroups.size() > 1) PublishCoverage(context.resources, *m_coverage);

            // Resume the same destination with LOAD. Its framebuffer, sample/resolve state
            // and bound puppet geometry survive the full-target composite, preserving earlier
            // visible ranges and uncovered multisample color through every mask interruption.
            VkRenderPassBeginInfo begin {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = *m_clipped.pipelines[clipped_index].pass,
                .framebuffer = *data.fb,
                .renderArea = {{0, 0}, {extent.width, extent.height}},
            };
            command.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
            bound = BoundPipeline::Mask;
            active_group = ordered.groupIndex;
        }
        if (bound != BoundPipeline::Clipped || bound_clipped != clipped_index) {
            command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  *m_clipped.pipelines[clipped_index].handle);
            push_descriptors(m_clipped, clipped_index, group_index, false);
            bound = BoundPipeline::Clipped;
            bound_clipped = clipped_index;
        }
        trace_range("clipped", range, ordered.groupIndex, m_materials->clipped.name.c_str());
        command.DrawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
    }
}

void MaskedDrawRenderer::destroy(RenderingResources& resources) {
    dropFramebuffers();
    m_composite.destroy();
    m_has_parents = false;
    m_textures.clear();
    m_inverted.clear();
    m_materials.reset();
    for (auto* program : {&m_mask, &m_clipped}) {
        for (auto& uniform : program->uniforms) resources.dyn_buf->unallocateSubRef(uniform);
        program->uniforms.clear();
        program->contract = {};
    }
}

bool MaskedDrawRenderer::SamePlan(const SceneMesh::MaskedDrawPlan& lhs,
                                  const SceneMesh::MaskedDrawPlan& rhs) {
    if (lhs.materials != rhs.materials || !SameRanges(lhs.unmaskedRanges, rhs.unmaskedRanges) ||
        lhs.groups.size() != rhs.groups.size() || lhs.orderedRanges.size() != rhs.orderedRanges.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.groups.size(); ++i) {
        const auto& a = lhs.groups[i];
        const auto& b = rhs.groups[i];
        if (a.identity != b.identity || a.maskTexture != b.maskTexture ||
            a.blend != b.blend || a.inverted != b.inverted ||
            a.coverageGroups != b.coverageGroups ||
            !SameRanges(a.maskRanges, b.maskRanges) || !SameRanges(a.contentRanges, b.contentRanges)) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs.orderedRanges.size(); ++i) {
        if (lhs.orderedRanges[i].groupIndex != rhs.orderedRanges[i].groupIndex ||
            !SameRange(lhs.orderedRanges[i].range, rhs.orderedRanges[i].range)) return false;
    }
    return true;
}

} // namespace wallpaper::vulkan

#include "MaskedDrawRenderer.hpp"

#include "Core/ArrayHelper.hpp"
#include "Interface/IImageParser.h"
#include "Utils/Logging.h"

#include <algorithm>
#include <array>

namespace wallpaper::vulkan
{
namespace
{

std::optional<VkFormat> ResolveMaskedDrawStencilFormat(const Device& device) {
    constexpr std::array formats {
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
    };
    for (const auto format : formats) {
        const auto properties = device.gpu().GetFormatProperties(format);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) !=
            0) {
            return format;
        }
    }
    return std::nullopt;
}

bool SameRange(const SceneMesh::DrawRange& lhs, const SceneMesh::DrawRange& rhs) {
    return lhs.firstIndex == rhs.firstIndex && lhs.indexCount == rhs.indexCount;
}

bool SameRanges(const std::vector<SceneMesh::DrawRange>& lhs,
                const std::vector<SceneMesh::DrawRange>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
        if (! SameRange(lhs[i], rhs[i])) return false;
    }
    return true;
}

} // namespace

bool MaskedDrawRenderer::configure(const Device& device, const ShaderDrawData& data,
                                   const SceneMesh& mesh) {
    m_stencil_format = VK_FORMAT_UNDEFINED;
    m_uniform_block.reset();

    const auto& plan = mesh.MaskedDraw();
    if (plan.empty()) {
        LOG_ERROR("MaskedDrawPrepare: empty plan node='%s'",
                  data.node != nullptr ? data.node->Name().c_str() : "<null>");
        return false;
    }
    if (data.model_pass || mesh.IndexCount() == 0 || plan.orderedRanges.empty()) {
        LOG_ERROR("MaskedDrawPrepare: invalid mesh contract node='%s' model=%s indices=%zu "
                  "ordered-ranges=%zu groups=%zu",
                  data.node != nullptr ? data.node->Name().c_str() : "<null>",
                  data.model_pass ? "true" : "false",
                  mesh.IndexCount(),
                  plan.orderedRanges.size(),
                  plan.groups.size());
        return false;
    }

    for (size_t group_index = 0; group_index < plan.groups.size(); group_index++) {
        const auto& group = plan.groups[group_index];
        if (group.maskTexture.empty() || group.maskRanges.empty() ||
            group.contentRanges.empty()) {
            LOG_ERROR("MaskedDrawPrepare: invalid group node='%s' group=%zu texture='%s' "
                      "mask-ranges=%zu content-ranges=%zu",
                      data.node != nullptr ? data.node->Name().c_str() : "<null>",
                      group_index,
                      group.maskTexture.c_str(),
                      group.maskRanges.size(),
                      group.contentRanges.size());
            return false;
        }
    }
    for (const auto& ordered : plan.orderedRanges) {
        if (ordered.groupIndex < -1 ||
            (ordered.groupIndex >= 0 &&
             static_cast<size_t>(ordered.groupIndex) >= plan.groups.size())) {
            LOG_ERROR("MaskedDrawPrepare: ordered range references invalid group node='%s' "
                      "group=%d groups=%zu",
                      data.node != nullptr ? data.node->Name().c_str() : "<null>",
                      ordered.groupIndex,
                      plan.groups.size());
            return false;
        }
    }

    const auto stencil_format = ResolveMaskedDrawStencilFormat(device);
    if (! stencil_format.has_value()) {
        LOG_ERROR("MaskedDrawPrepare: no depth/stencil attachment format node='%s'",
                  data.node != nullptr ? data.node->Name().c_str() : "<null>");
        return false;
    }
    m_stencil_format = *stencil_format;
    return true;
}

std::vector<std::string_view>
MaskedDrawRenderer::resourceTextures(const SceneMesh& mesh) const {
    std::vector<std::string_view> textures;
    textures.reserve(mesh.MaskedDraw().groups.size());
    for (const auto& group : mesh.MaskedDraw().groups) textures.push_back(group.maskTexture);
    return textures;
}

bool MaskedDrawRenderer::refreshTextures(Scene& scene, const Device& device,
                                         const ShaderDrawData& data) {
    if (data.node == nullptr || data.node->Mesh() == nullptr) {
        LOG_ERROR("MaskedDrawTexture: missing mesh node='%s'",
                  data.node != nullptr ? data.node->Name().c_str() : "<null>");
        return false;
    }

    const auto& groups = data.node->Mesh()->MaskedDraw().groups;
    m_textures.resize(groups.size());
    for (size_t i = 0; i < groups.size(); i++) {
        const auto& texture_name = groups[i].maskTexture;
        if (scene.dirtyImportedTextureKeys.count(texture_name) == 0) {
            if (auto cached = device.tex_cache().FindTex(texture_name); cached.has_value()) {
                m_textures[i] = *cached;
                continue;
            }
        } else {
            scene.DropParsedImageCache(texture_name);
        }

        const auto texture_it = scene.textures.find(texture_name);
        if (texture_it == scene.textures.end() || texture_it->second.isVideo) {
            LOG_ERROR("MaskedDrawTexture: invalid imported mask node='%s' group=%zu texture='%s'",
                      data.node != nullptr ? data.node->Name().c_str() : "<null>",
                      i,
                      texture_name.c_str());
            return false;
        }

        auto image = scene.GetParsedImageIfReady(texture_name);
        if (image == nullptr) {
            image = scene.ParseImageBlockingCached(texture_name);
        }
        if (image == nullptr) {
            LOG_ERROR("MaskedDrawTexture: parse failed node='%s' group=%zu texture='%s'",
                      data.node != nullptr ? data.node->Name().c_str() : "<null>",
                      i,
                      texture_name.c_str());
            return false;
        }

        auto slots = device.tex_cache().CreateTex(*image);
        scene.DropParsedImageCache(texture_name);
        if (slots.slots.empty()) {
            LOG_ERROR("MaskedDrawTexture: upload failed node='%s' group=%zu texture='%s'",
                      data.node != nullptr ? data.node->Name().c_str() : "<null>",
                      i,
                      texture_name.c_str());
            return false;
        }
        m_textures[i] = std::move(slots);
    }
    return true;
}

ShaderDrawAttachmentDescription MaskedDrawRenderer::attachmentDescription() const {
    if (! enabled()) return {};
    return ShaderDrawAttachmentDescription {
        .format           = m_stencil_format,
        .depth_load_op    = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .depth_store_op   = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencil_load_op  = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initial_layout   = VK_IMAGE_LAYOUT_UNDEFINED,
        .final_layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .cache_tag        = "masked",
    };
}

VmaImageParameters* MaskedDrawRenderer::acquireAttachment(const Device& device,
                                                          RenderingResources& resources,
                                                          const ShaderDrawData& data) {
    auto* stencil_image = resources.masked_draw_attachments.acquire(
        device, data.output, data.vk_output.extent, m_stencil_format, data.sample_count);
    if (stencil_image == nullptr) {
        LOG_ERROR("MaskedDrawAttachment: allocation failed node='%s' output='%s' "
                  "extent=[%u,%u] format=%d",
                  data.node != nullptr ? data.node->Name().c_str() : "<null>",
                  data.output.c_str(),
                  data.vk_output.extent.width,
                  data.vk_output.extent.height,
                  static_cast<int>(m_stencil_format));
    }
    return stencil_image;
}

bool MaskedDrawRenderer::preparePipelines(const Device& device, RenderingResources& resources,
                                          const ShaderDrawPipelineContext& context) {
    std::vector<Uni_ShaderSpv> test_spvs;
    ShaderReflected            test_ref;
    if (! GenReflect(context.material.customShader.shader->codes, test_spvs, test_ref)) {
        LOG_ERROR("MaskedDrawPrepare: visible shader reflection failed node='%s'",
                  context.data.node != nullptr ? context.data.node->Name().c_str() : "<null>");
        return false;
    }

    auto test_pass = CreateShaderDrawRenderPass(device.handle(),
                                                VK_FORMAT_R8G8B8A8_UNORM,
                                                context.render_state.color_load_op,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                attachmentDescription(),
                                                context.data.sample_count,
                                                context.data.resolve_msaa);
    if (! test_pass.has_value()) return false;

    GraphicsPipeline test_pipeline;
    test_pipeline.toDefault();
    test_pipeline.multisample.rasterizationSamples = context.data.sample_count;
    test_pipeline.depth.stencilTestEnable = true;
    test_pipeline.depth.front = VkStencilOpState {
        .failOp      = VK_STENCIL_OP_KEEP,
        .passOp      = VK_STENCIL_OP_KEEP,
        .depthFailOp = VK_STENCIL_OP_KEEP,
        .compareOp   = VK_COMPARE_OP_EQUAL,
        .compareMask = 0xff,
        .writeMask   = 0x00,
        .reference   = kMaskedDrawStencilReference,
    };
    test_pipeline.depth.back = test_pipeline.depth.front;
    m_test_pipeline.debug_name =
        "MaskedDrawTest[node=" +
        (context.data.node != nullptr ? context.data.node->Name() : std::string("(null)")) +
        ",output=" + context.data.output + "]";
    m_test_pipeline.cache_key = ShaderDrawPipelineCompatibilityKey(
        context.render_state.color_load_op,
        false,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        attachmentDescription(),
        context.data.sample_count,
        context.data.resolve_msaa);
    test_pipeline.addDescriptorSetInfo(spanone { context.descriptor_info })
        .setColorBlendStates(spanone { context.render_state.color_blend })
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addInputBindingDescription(context.binding_descriptions)
        .addInputAttributeDescription(context.attribute_descriptions);
    for (auto& spv : test_spvs) test_pipeline.addStage(std::move(spv));
    if (! test_pipeline.create(device,
                               *test_pass,
                               m_test_pipeline,
                               resources.pipeline_cache.get())) {
        return false;
    }

    const auto bone_count = context.mesh.Skinning().boneCount;
    if (bone_count == 0) {
        LOG_ERROR("MaskedDrawPrepare: mesh has no bone count node='%s'",
                  context.data.node != nullptr ? context.data.node->Name().c_str() : "<null>");
        return false;
    }
    auto mask_shader_codes = CompileMaskedDrawMaskShaderCodes(bone_count);
    if (! mask_shader_codes.has_value()) {
        LOG_ERROR("MaskedDrawPrepare: mask shader compilation failed node='%s' bones=%u",
                  context.data.node != nullptr ? context.data.node->Name().c_str() : "<null>",
                  bone_count);
        return false;
    }

    std::vector<Uni_ShaderSpv> mask_spvs;
    ShaderReflected            mask_ref;
    if (! GenReflect(*mask_shader_codes, mask_spvs, mask_ref)) {
        LOG_ERROR("MaskedDrawPrepare: mask shader reflection failed node='%s'",
                  context.data.node != nullptr ? context.data.node->Name().c_str() : "<null>");
        return false;
    }
    if (mask_ref.blocks.size() != 1) {
        LOG_ERROR("MaskedDrawPrepare: expected one mask uniform block node='%s' blocks=%zu",
                  context.data.node != nullptr ? context.data.node->Name().c_str() : "<null>",
                  mask_ref.blocks.size());
        return false;
    }

    DescriptorSetInfo mask_descriptor_info;
    mask_descriptor_info.push_descriptor = true;
    mask_descriptor_info.bindings.resize(mask_ref.binding_map.size());
    std::transform(mask_ref.binding_map.begin(),
                   mask_ref.binding_map.end(),
                   mask_descriptor_info.bindings.begin(),
                   [](const auto& item) { return item.second; });

    std::vector<VkVertexInputBindingDescription> mask_bindings;
    std::vector<VkVertexInputAttributeDescription> mask_attributes;
    for (uint32_t stream_index = 0; stream_index < context.mesh.VertexCount(); stream_index++) {
        const auto& vertex    = context.mesh.GetVertexArray(stream_index);
        const auto  attrs_map = vertex.GetAttrOffsetMap();
        mask_bindings.push_back(VkVertexInputBindingDescription {
            .binding   = stream_index,
            .stride    = static_cast<uint32_t>(vertex.OneSizeOf()),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        });
        for (const auto& [name, input] : mask_ref.input_location_map) {
            if (input.location >= kMaskedDrawVertexAttributes.size()) {
                LOG_ERROR("MaskedDrawPrepare: unexpected mask shader input location=%u "
                          "name='%s' node='%s'",
                          input.location,
                          name.c_str(),
                          context.data.node != nullptr ? context.data.node->Name().c_str()
                                                       : "<null>");
                return false;
            }
            const auto expected_name = kMaskedDrawVertexAttributes[input.location];
            const auto attr_it       = attrs_map.find(std::string(expected_name));
            if (attr_it == attrs_map.end()) {
                LOG_ERROR("MaskedDrawPrepare: mask shader input '%s' expects attribute '%.*s' "
                          "node='%s'",
                          name.c_str(),
                          static_cast<int>(expected_name.size()),
                          expected_name.data(),
                          context.data.node != nullptr ? context.data.node->Name().c_str()
                                                       : "<null>");
                return false;
            }
            mask_attributes.push_back(VkVertexInputAttributeDescription {
                .location = input.location,
                .binding  = stream_index,
                .format   = input.format,
                .offset   = static_cast<uint32_t>(attr_it->second.offset),
            });
        }
    }

    auto mask_pass = CreateShaderDrawRenderPass(device.handle(),
                                                VK_FORMAT_R8G8B8A8_UNORM,
                                                context.render_state.color_load_op,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                attachmentDescription(),
                                                context.data.sample_count,
                                                context.data.resolve_msaa);
    if (! mask_pass.has_value()) return false;

    VkPipelineColorBlendAttachmentState mask_color {};
    mask_color.colorWriteMask = 0;
    GraphicsPipeline mask_pipeline;
    mask_pipeline.toDefault();
    mask_pipeline.multisample.rasterizationSamples = context.data.sample_count;
    mask_pipeline.depth.stencilTestEnable = true;
    mask_pipeline.depth.front = VkStencilOpState {
        .failOp      = VK_STENCIL_OP_KEEP,
        .passOp      = VK_STENCIL_OP_REPLACE,
        .depthFailOp = VK_STENCIL_OP_KEEP,
        .compareOp   = VK_COMPARE_OP_ALWAYS,
        .compareMask = 0xff,
        .writeMask   = 0xff,
        .reference   = kMaskedDrawStencilReference,
    };
    mask_pipeline.depth.back = mask_pipeline.depth.front;
    m_mask_pipeline.debug_name =
        "MaskedDrawMask[node=" +
        (context.data.node != nullptr ? context.data.node->Name() : std::string("(null)")) +
        ",output=" + context.data.output + "]";
    m_mask_pipeline.cache_key = ShaderDrawPipelineCompatibilityKey(
        context.render_state.color_load_op,
        false,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        attachmentDescription(),
        context.data.sample_count,
        context.data.resolve_msaa);
    mask_pipeline.addDescriptorSetInfo(spanone { mask_descriptor_info })
        .setColorBlendStates(spanone { mask_color })
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addInputBindingDescription(mask_bindings)
        .addInputAttributeDescription(mask_attributes);
    for (auto& spv : mask_spvs) mask_pipeline.addStage(std::move(spv));
    if (! mask_pipeline.create(device,
                               *mask_pass,
                               m_mask_pipeline,
                               resources.pipeline_cache.get())) {
        return false;
    }

    m_uniform_block = mask_ref.blocks.front();
    if (! resources.dyn_buf->allocateSubRef(
            m_uniform_block->size,
            m_ubo_buf,
            device.limits().minUniformBufferOffsetAlignment)) {
        return false;
    }
    return true;
}

void MaskedDrawRenderer::updateUniform(StagingBuffer* buffer, std::string_view name,
                                       const ShaderValue& value) {
    if (! m_uniform_block.has_value() || ! m_ubo_buf) return;
    UpdateShaderDrawUniform(buffer, m_ubo_buf, *m_uniform_block, name, value);
}

void MaskedDrawRenderer::initializeUniforms(StagingBuffer* buffer) {
    if (! m_ubo_buf) return;
    buffer->fillBuf(m_ubo_buf, 0, m_ubo_buf.size, 0);
}

void MaskedDrawRenderer::recordIndexed(const ShaderDrawRecordContext& context) {
    auto&       data      = context.data;
    auto&       command   = context.resources.command;
    const auto& plan      = data.node->Mesh()->MaskedDraw();
    const auto& out_extent = data.vk_output.extent;

    const auto push_mask_descriptors = [&](size_t group_index) {
        const auto& image = m_textures[group_index].getActive();
        VkDescriptorImageInfo image_info {
            image.sampler,
            image.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet image_write {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext           = nullptr,
            .dstSet          = {},
            .dstBinding      = kMaskedDrawTextureBinding,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &image_info,
        };
        command.PushDescriptorSetKHR(
            VK_PIPELINE_BIND_POINT_GRAPHICS, *m_mask_pipeline.layout, 0, image_write);

        VkDescriptorBufferInfo buffer_info {
            context.resources.dyn_buf->gpuBuf(),
            m_ubo_buf.offset,
            m_ubo_buf.size,
        };
        VkWriteDescriptorSet buffer_write {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext           = nullptr,
            .dstSet          = {},
            .dstBinding      = kMaskedDrawUniformBinding,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &buffer_info,
        };
        command.PushDescriptorSetKHR(
            VK_PIPELINE_BIND_POINT_GRAPHICS, *m_mask_pipeline.layout, 0, buffer_write);
    };

    enum class BoundPipeline
    {
        Visible,
        Mask,
        Clipped,
    };
    BoundPipeline bound_pipeline  = BoundPipeline::Visible;
    int32_t       active_group    = -1;

    // The ordered range schedule preserves the MDLV part order exactly. Drawing all unmasked parts
    // first would move eyelids, pupils, and other layered parts across one another even though the
    // same ranges and stencil masks were eventually submitted.
    for (const auto& ordered : plan.orderedRanges) {
        const auto& range = ordered.range;
        if (range.indexCount == 0) continue;

        if (ordered.groupIndex < 0) {
            if (bound_pipeline != BoundPipeline::Visible) {
                command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *data.pipeline.handle);
                context.push_visible_descriptors(*data.pipeline.layout);
                bound_pipeline = BoundPipeline::Visible;
            }
            command.DrawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
            continue;
        }

        const auto group_index = static_cast<size_t>(ordered.groupIndex);
        if (active_group != ordered.groupIndex) {
            VkClearAttachment clear_attachment {
                .aspectMask      = VK_IMAGE_ASPECT_STENCIL_BIT,
                .colorAttachment = 0,
                .clearValue      = VkClearValue { .depthStencil = { 1.0f, 0 } },
            };
            VkClearRect clear_rect {
                .rect = VkRect2D {
                    .offset = { 0, 0 },
                    .extent = { out_extent.width, out_extent.height },
                },
                .baseArrayLayer = 0,
                .layerCount     = 1,
            };
            command.ClearAttachments(spanone { clear_attachment }, spanone { clear_rect });

            command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_mask_pipeline.handle);
            push_mask_descriptors(group_index);
            bound_pipeline = BoundPipeline::Mask;
            for (const auto& mask_range : plan.groups[group_index].maskRanges) {
                if (mask_range.indexCount == 0) continue;
                command.DrawIndexed(
                    mask_range.indexCount, 1, mask_range.firstIndex, 0, 0);
            }
            active_group = ordered.groupIndex;
        }

        if (bound_pipeline != BoundPipeline::Clipped) {
            command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_test_pipeline.handle);
            context.push_visible_descriptors(*m_test_pipeline.layout);
            bound_pipeline = BoundPipeline::Clipped;
        }
        command.DrawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
    }
}

void MaskedDrawRenderer::destroy(RenderingResources& resources) {
    m_textures.clear();
    m_uniform_block.reset();
    if (m_ubo_buf) resources.dyn_buf->unallocateSubRef(m_ubo_buf);
    m_ubo_buf = {};
}

bool MaskedDrawRenderer::SamePlan(const SceneMesh::MaskedDrawPlan& lhs,
                                  const SceneMesh::MaskedDrawPlan& rhs) {
    if (! SameRanges(lhs.unmaskedRanges, rhs.unmaskedRanges) ||
        lhs.groups.size() != rhs.groups.size() ||
        lhs.orderedRanges.size() != rhs.orderedRanges.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.groups.size(); i++) {
        const auto& lhs_group = lhs.groups[i];
        const auto& rhs_group = rhs.groups[i];
        if (lhs_group.maskTexture != rhs_group.maskTexture ||
            ! SameRanges(lhs_group.maskRanges, rhs_group.maskRanges) ||
            ! SameRanges(lhs_group.contentRanges, rhs_group.contentRanges)) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs.orderedRanges.size(); i++) {
        if (lhs.orderedRanges[i].groupIndex != rhs.orderedRanges[i].groupIndex ||
            ! SameRange(lhs.orderedRanges[i].range, rhs.orderedRanges[i].range)) {
            return false;
        }
    }
    return true;
}

} // namespace wallpaper::vulkan

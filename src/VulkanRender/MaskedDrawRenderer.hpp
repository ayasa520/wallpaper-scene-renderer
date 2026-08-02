#pragma once

#include "MaskedDrawShaderContract.hpp"
#include "ShaderDrawCore.hpp"

#include <optional>

namespace wallpaper::vulkan
{

class MaskedDrawRenderer final : public ShaderDrawExtension {
public:
    bool configure(const Device&, const ShaderDrawData&, const SceneMesh&) override;
    std::vector<std::string_view> resourceTextures(const SceneMesh&) const override;
    bool refreshTextures(Scene&, const Device&, const ShaderDrawData&) override;
    ShaderDrawAttachmentDescription attachmentDescription() const override;
    VmaImageParameters* acquireAttachment(const Device&, RenderingResources&,
                                          const ShaderDrawData&) override;
    bool preparePipelines(const Device&, RenderingResources&,
                          const ShaderDrawPipelineContext&) override;
    void updateUniform(StagingBuffer*, std::string_view, const ShaderValue&) override;
    void initializeUniforms(StagingBuffer*) override;
    void recordIndexed(const ShaderDrawRecordContext&) override;
    void destroy(RenderingResources&) override;

    static bool SamePlan(const SceneMesh::MaskedDrawPlan&, const SceneMesh::MaskedDrawPlan&);

private:
    bool enabled() const { return m_stencil_format != VK_FORMAT_UNDEFINED; }

    std::vector<ImageSlotsRef>       m_textures;
    VkFormat                         m_stencil_format { VK_FORMAT_UNDEFINED };
    PipelineParameters               m_mask_pipeline;
    PipelineParameters               m_test_pipeline;
    StagingBufferRef                 m_ubo_buf;
    std::optional<ShaderReflected::Block> m_uniform_block;
};

} // namespace wallpaper::vulkan

#pragma once

#include "MaskedDrawShaderContract.hpp"
#include "MaskedDrawComposite.hpp"
#include "ShaderDrawCore.hpp"

#include <optional>

namespace wallpaper::vulkan
{

class MaskedDrawRenderer final : public ShaderDrawExtension {
public:
    bool configure(const Device&, const ShaderDrawData&, const SceneMesh&) override;
    std::vector<std::string_view> resourceTextures(const SceneMesh&) const override;
    bool refreshTextures(Scene&, const Device&, const ShaderDrawData&) override;
    bool refreshFramebuffers(const Device&, RenderingResources&, const ShaderDrawData&) override;
    void dropFramebuffers() override;
    bool preparePipelines(const Device&, RenderingResources&,
                          const ShaderDrawPipelineContext&) override;
    void updateUniform(StagingBuffer*, std::string_view, const ShaderValue&) override;
    bool hasUniform(std::string_view) const override;
    void updateMaterialUniforms(StagingBuffer*, const SceneMaterial&) override;
    void initializeUniforms(StagingBuffer*) override;
    void recordIndexed(const ShaderDrawRecordContext&) override;
    void destroy(RenderingResources&) override;

    static bool SamePlan(const SceneMesh::MaskedDrawPlan&, const SceneMesh::MaskedDrawPlan&);

private:
    struct Program {
        MaskedDrawShaderContract contract;
        std::vector<PipelineParameters> pipelines;
        std::vector<StagingBufferRef> uniforms;
    };

    bool prepareProgram(const Device&, RenderingResources&, const ShaderDrawPipelineContext&,
                         const SceneMaterial&, bool mask, Program&);

    std::shared_ptr<const SceneMesh::MaskedDrawMaterials> m_materials;
    std::vector<ImageSlotsRef> m_textures;
    std::vector<bool> m_inverted;
    Program m_mask;
    Program m_clipped;
    MaskedDrawComposite m_composite;
    bool m_has_parents { false };
    std::shared_ptr<VmaImageParameters> m_coverage;
    std::shared_ptr<VmaImageParameters> m_intermediate;
    vvk::Framebuffer m_mask_framebuffer;
    vvk::Framebuffer m_intermediate_framebuffer;
    uint64_t m_uniform_epoch { 0 };
    uint64_t m_draw_sequence { 0 };
    uint64_t m_pose_hash { 0 };
};

} // namespace wallpaper::vulkan

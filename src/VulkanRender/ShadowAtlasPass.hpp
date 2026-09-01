#pragma once

#include "VulkanPass.hpp"

#include "Scene/Scene.h"
#include "Vulkan/Device.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/ImmutableMeshStore.hpp"
#include "Vulkan/Parameters.hpp"
#include "Vulkan/Spv.hpp"
#include "Vulkan/StagingBuffer.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace wallpaper
{
namespace vulkan
{

// Fills `_rt_shadowAtlas` with per-light square depth slots. Point lights use
// six 2×3 viewports; spots use one. Casters are 3D model meshes. First bind
// far-clears depth to 1.0 even when the draw list is empty.
class ShadowAtlasPass : public VulkanPass {
public:
    struct Desc {
        std::string target;
        Scene*      scene { nullptr };
        ImageParameters vk_target;
    };

    explicit ShadowAtlasPass(const Desc&);
    ~ShadowAtlasPass() override;

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void updateBeforeUpload() override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    struct CasterMesh {
        SceneNode*                        node { nullptr };
        std::shared_ptr<ImmutableMeshGpu> mesh;
        uint32_t                          stride { 0 };
        uint32_t                          position_offset { 0 };
        uint32_t                          blend_indices_offset { 0 };
        uint32_t                          blend_weights_offset { 0 };
        uint32_t                          bone_count { 0 };
        uint32_t                          index_count { 0 };
        uint32_t                          vertex_count { 0 };
        uint32_t                          index_element_bytes { 2 };
        bool                              indexed { false };
        bool                              pose_error_reported { false };
        std::span<const Eigen::Affine3f>  skinning_pose;
        uint64_t                          pose_revision { 0 };
        uint64_t                          pose_frame_serial { 0 };
    };

    struct DrawItem {
        uint32_t caster_index { 0 };
        VkDeviceSize ubo_offset { 0 };
        VkDeviceSize ubo_size { 0 };
        float    vp_x { 0.0f };
        float    vp_y { 0.0f };
        float    vp_w { 0.0f };
        float    vp_h { 0.0f };
        int32_t  scissor_x { 0 };
        int32_t  scissor_y { 0 };
        int32_t  scissor_w { 1 };
        int32_t  scissor_h { 1 };
        Eigen::Matrix4f mvp { Eigen::Matrix4f::Identity() };
    };

    bool ensureClearPass(const Device&);
    bool ensureShadowShaders(uint32_t bone_count);
    bool ensurePipeline(const Device&, RenderingResources&, const CasterMesh&);
    bool ensureFramebuffer(const Device&);
    void collectCasters(Scene&, const Device&, RenderingResources&);
    void releaseCasters();
    void rebuildDrawList();

    Desc               m_desc;
    StagingBuffer*     m_dyn_buf { nullptr };
    StagingBufferRef   m_ubo_buf;
    VkDeviceSize       m_ubo_align { 256 };
    vvk::RenderPass    m_clear_pass;
    vvk::Framebuffer   m_fb;
    std::unordered_map<uint32_t, std::vector<Uni_ShaderSpv>> m_shader_spvs;
    std::unordered_map<std::string, PipelineParameters> m_pipelines;
    VkExtent2D         m_fb_extent {};
    std::vector<CasterMesh> m_casters;
    std::vector<DrawItem>   m_draws;
};

} // namespace vulkan
} // namespace wallpaper

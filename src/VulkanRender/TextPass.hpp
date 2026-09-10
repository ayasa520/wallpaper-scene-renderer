#pragma once

#include "VulkanPass.hpp"
#include "Interface/IShaderValueUpdater.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "Vulkan/Device.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/StagingBuffer.hpp"

namespace wallpaper
{
class SceneTextPrimitive;
class SceneNode;
class SceneMesh;

namespace vulkan
{

class TextPass : public VulkanPass {
public:
    struct Desc {
        Scene*      scene { nullptr };
        SceneNode*  node { nullptr };
        // Authored layer ownership is separate from the SceneNode pointer because runtime text
        // rebuilds swap primitives under stable nodes; the layer id remains the durable refresh key.
        int32_t     layer_id { 0 };
        bool        execute_when_hidden { false };
        std::function<bool()> should_execute;
        // A private seed owns source initialization; a direct draw loads the inherited target.
        // Reflection and composition targets do not imply a private text source by themselves.
        bool        private_source { false };
        std::string output;
        std::string camera_override;
        bool use_active_camera_for_parallax { false };
        ShaderModelSpace model_space { ShaderModelSpace::Object };
        bool reflection_pass { false };
        bool reflection_raster { false };
        bool reflection_snapshot { false };
        std::string effect_snapshot_camera;
        AlphaWritePolicy alpha_write_policy { AlphaWritePolicy::Preserve };
        // Capture graph-selected raster state by value. Live text properties mutate the same
        // primitive in both graph generations, so comparing primitive pointers loses the change.
        bool shared_depth { false };
        bool glyph_depth_test { false };

        ImageParameters          vk_output;
        ImageParameters          vk_resolve;
        VmaImageParameters*       depth_image { nullptr };
        VkSampleCountFlagBits    sample_count { VK_SAMPLE_COUNT_1_BIT };
        bool                     resolve_msaa { false };
        vvk::Framebuffer         framebuffer;
        PipelineParameters       pipeline;
        // Opposite read-only depth variant for a direct background whose retained material
        // selection differs from the current glyph selection. Both are prepared before drawing.
        PipelineParameters       background_pipeline;
        StagingBufferRef         ubo_buf;
        StagingBufferRef         background_ubo_buf;
        ImageSlotsRef            background_texture;
        std::vector<ImageSlotsRef> page_textures;
    };

    TextPass(const Desc&);
    ~TextPass() override;

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void dropOutputFramebuffers() override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool warmupPipeline(Scene&, const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool canReuseForResidency(const VulkanPass& next_pass) const override;
    void absorbResidencyGraphState(const VulkanPass&) override;
    bool referencesRenderTarget(std::string_view) const override;
    bool referencesTextLayer(int32_t) const override;

private:
    struct MeshBuffers {
        std::vector<StagingBufferRef> vertex_bufs;
        StagingBufferRef              index_buf;
        uint32_t                      draw_count { 0 };
        bool                          force_upload { true };
        uint64_t                      uploaded_revision { 0 };
    };

    // The direct text pass owns its own dynamic mesh uploads because text geometry can change
    // without changing render-graph topology. Keeping the buffers inside the pass lets a single
    // pass instance absorb atlas page count and quad changes in place.
    bool ensureMeshBuffers(SceneMesh&, MeshBuffers&, RenderingResources&);
    bool refreshTextures(const Device&);
    bool recreateFramebuffer(const Device&, RenderingResources&);

    Desc m_desc;
    // Store the node identity independently of the raw node pointer because an old text pass can be
    // queried for residency after a topology rebuild has released its SceneNode owner.
    uint64_t m_node_identity { 0 };
    MeshBuffers m_background_buffers;
    std::vector<MeshBuffers> m_page_buffers;
    uint32_t m_loaded_atlas_version { std::numeric_limits<uint32_t>::max() };
    uint32_t m_traced_atlas_version { std::numeric_limits<uint32_t>::max() };
};

} // namespace vulkan
} // namespace wallpaper

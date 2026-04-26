#pragma once
#include "VulkanPass.hpp"
#include <functional>
#include <string>
#include <vector>

#include "Vulkan/Device.hpp"
#include "Scene/Scene.h"
#include "Vulkan/StagingBuffer.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "SpriteAnimation.hpp"
#include "Interface/IShaderValueUpdater.h"

namespace wallpaper
{

namespace vulkan
{

class CustomShaderPass : public VulkanPass {
public:
    struct Desc {
        // in
        // Keeping a scene pointer inside the prepared pass description lets execution-time
        // diagnostics inspect the live render-target metadata that the pass is sampling from and
        // writing to. That is essential for tracking text/effect bugs where physical Vulkan image
        // sizes diverge from the authored logical content rectangle.
        Scene*                   scene { nullptr };
        SceneNode*               node { nullptr };
        bool                     execute_when_hidden { false };
        // Optional runtime gate for topology-stable helper passes. This is used for the synthetic
        // hidden-final-effect composite: the pass exists in the graph, but it must only draw on
        // frames where the final authored effect is locally hidden.
        std::function<bool()>    should_execute;
        std::vector<std::string> textures;
        std::string              output;
        sprite_map_t             sprites_map;

        // -----prepared
        // vulkan texs
        std::vector<ImageSlotsRef> vk_textures;
        std::vector<i32>           vk_tex_binding;
        ImageParameters            vk_output;

        // bufs
        bool                          dyn_vertex { false };
        bool                          force_dyn_upload { false };
        std::vector<StagingBufferRef> vertex_bufs;
        StagingBufferRef              index_buf;
        StagingBufferRef              ubo_buf;

        // pipeline
        VkClearValue       clear_value;
        VkClearValue       depth_clear_value;
        bool               blending { false };
        bool               model_pass { false };
        bool               depth_test { false };
        bool               depth_write { false };
        bool               clear_depth { true };
        vvk::Framebuffer   fb;
        VmaImageParameters* depth_image_ref { nullptr };
        PipelineParameters pipeline;
        u32                draw_count { 0 };

        // Dynamic mesh bytes are uploaded before command recording, while uniforms and sprite
        // frame selection remain on the historical execute-time path for text/effect stability.
        std::function<void()> update_dynamic_mesh_op;
        // uniforms
        std::function<void()> update_op;
    };

    CustomShaderPass(const Desc&);
    virtual ~CustomShaderPass();

    void setDescTex(u32 index, std::string_view tex_key);

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void updateBeforeUpload() override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    Desc m_desc;
};

} // namespace vulkan
} // namespace wallpaper

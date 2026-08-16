#pragma once

#include "VulkanPass.hpp"

#include "Scene/Scene.h"
#include "Vulkan/Device.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/StagingBuffer.hpp"

#include <string>
#include <string_view>

namespace wallpaper
{
namespace vulkan
{

// Fills `_rt_volumetricsSingle` with a depth blit from the scene framebuffer.
// Single is created at the same 1/8 or 1/4 scale as LightBuffer. Depth stretch
// uses NEAREST (D3D/GL reject LINEAR for depth).
class VolumetricsSingleFillPass : public VulkanPass {
public:
    struct Desc {
        std::string     dst;
        std::string     scene_output;
        ImageParameters vk_dst;
    };

    explicit VolumetricsSingleFillPass(const Desc&);
    ~VolumetricsSingleFillPass() override;

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    bool ensurePipeline(const Device&, RenderingResources&);
    bool ensureFramebuffer(const Device&);
    void clearFar(RenderingResources&) const;

    Desc               m_desc;
    StagingBufferRef   m_vertex_buf;
    vvk::Sampler       m_point_sampler;
    vvk::Framebuffer   m_fb;
    PipelineParameters m_pipeline;
    VkExtent2D         m_fb_extent {};
};

} // namespace vulkan
} // namespace wallpaper

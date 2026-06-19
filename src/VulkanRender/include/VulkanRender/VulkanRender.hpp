#pragma once

#include "RenderGraph/RenderGraph.hpp"
#include "SceneWallpaperSurface.hpp"
#include "Swapchain/ExSwapchain.hpp"
#include "Type.hpp"
#include "VulkanRender/OffscreenFrameReleaseCallback.hpp"

#include <cstdio>
#include <memory>

namespace wallpaper
{
class Scene;

namespace vulkan
{
class FinPass;

class VulkanRender {
public:
    VulkanRender();
    ~VulkanRender();

    bool init(RenderInitInfo);

    void destroy();

    void drawFrame(Scene&);
    void setPaused(bool paused);
    void setOffscreenFrameReleaseCallback(OffscreenFrameReleaseCallback callback);
    bool reconfigureOffscreenExport(uint32_t width,
                                    uint32_t height,
                                    TexTiling tiling,
                                    ExternalFrameExportMode export_mode,
                                    uint32_t export_drm_fourcc,
                                    const std::vector<uint64_t>& export_drm_modifiers,
                                    ExternalFrameMemoryPreference memory_preference);

    void clearLastRenderGraph(bool clear_scene_caches = false);
    void clearRenderGraphResources();
    void compileRenderGraph(Scene&, rg::RenderGraph&, bool refresh_resources_only = false);
    void warmupRenderGraphPipelines(Scene&, rg::RenderGraph&);
    void refreshImportedTextures(Scene&);
    void UpdateCameraFillMode(Scene&, wallpaper::FillMode);

    ExSwapchain* exSwapchain() const;
    bool inited() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
} // namespace vulkan
} // namespace wallpaper

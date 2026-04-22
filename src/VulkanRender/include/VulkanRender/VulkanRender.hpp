#pragma once

#include "RenderGraph/RenderGraph.hpp"
#include "SceneWallpaperSurface.hpp"
#include "Swapchain/ExSwapchain.hpp"
#include "Type.hpp"

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

    void clearLastRenderGraph();
    void clearRenderGraphResources();
    void compileRenderGraph(Scene&, rg::RenderGraph&, bool refresh_resources_only = false);
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

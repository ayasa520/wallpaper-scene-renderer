#pragma once
#include "VulkanPass.hpp"
#include <functional>
#include <string>

#include "Vulkan/Device.hpp"
#include "Scene/Scene.h"

namespace wallpaper
{
namespace vulkan
{

class CopyPass : public VulkanPass {
public:
    struct Desc {
        std::string src;
        std::string dst;

        ImageParameters vk_src;
        ImageParameters vk_dst;

        // Runtime visibility gates let the graph keep a stable topology while a pass becomes a
        // cheap no-op for the current frame. Hidden image effects use this for both sides of their
        // contract: their authored copy commands only run while the effect is visible, and their
        // bypass copy only runs while the effect is hidden.
        std::function<bool()> should_execute;
    };

    CopyPass(const Desc&);
    virtual ~CopyPass();

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    Desc m_desc;
};

}
}

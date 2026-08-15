#pragma once

#include "VulkanPass.hpp"

#include <string>
#include <string_view>

#include "Scene/Scene.h"
#include "Vulkan/Device.hpp"

namespace wallpaper
{
namespace vulkan
{

class ClearPass : public VulkanPass {
public:
    struct Desc {
        std::string     target;
        ImageParameters vk_target;
        VkClearValue    clear_value {};
    };

    explicit ClearPass(const Desc&);
    ~ClearPass() override;

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    Desc m_desc;
};

} // namespace vulkan
} // namespace wallpaper

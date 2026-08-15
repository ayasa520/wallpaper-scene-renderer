#pragma once
#include "VulkanPass.hpp"
#include <string>

#include "Vulkan/Device.hpp"

#include "SpecTexs.hpp"

namespace wallpaper
{
namespace vulkan
{

class PrePass : public VulkanPass {
public:
    struct Desc {
        // in
        const std::string_view result { SpecTex_Default };
        const VkImageLayout    layout { VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        // prepared
        ImageParameters vk_result;
        VkClearValue    clear_value;
    };

    PrePass(const Desc&);
    virtual ~PrePass();

    // void setClearValue(vk::ClearValue);

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    Desc m_desc;
};

} // namespace vulkan
} // namespace wallpaper

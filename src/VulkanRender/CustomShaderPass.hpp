#pragma once
#include "ShaderDrawCore.hpp"

namespace wallpaper
{

namespace vulkan
{

class CustomShaderPass : public VulkanPass {
public:
    using Desc = ShaderDrawRequest;

    CustomShaderPass(const Desc&);
    virtual ~CustomShaderPass();

    void setDescTex(u32 index, std::string_view tex_key);

    // Diagnostic presentation borrows the framebuffer's exact prepared image at this draw's
    // ordering point. Looking it up again by logical name could select a different pooled image.
    const ImageParameters& outputImage() const { return m_core.data().vk_output; }

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void refreshImportedTextureBindings(Scene&, const Device&) override;
    void dropOutputFramebuffers() override;
    void updateBeforeUpload() override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool warmupPipeline(Scene&, const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    std::string profileName() const override;
    bool canReuseForResidency(const VulkanPass& next_pass) const override;
    void absorbResidencyGraphState(const VulkanPass&) override;
    bool referencesRenderTarget(std::string_view) const override;
    bool referencesImportedTexture(std::string_view) const override;

private:
    ShaderDrawCore m_core;
};

} // namespace vulkan
} // namespace wallpaper

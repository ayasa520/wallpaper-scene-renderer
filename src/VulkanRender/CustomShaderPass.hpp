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

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void prepareDeferred(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void dropOutputFramebuffers() override;
    void updateBeforeUpload() override;
    DeferredPrepareResourcesState requestDeferredPrepareResources(Scene&, const Device&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool warmupPipeline(Scene&, const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool canReuseForResidency(const VulkanPass& next_pass) const override;
    void absorbResidencyGraphState(const VulkanPass&) override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    ShaderDrawCore m_core;
};

} // namespace vulkan
} // namespace wallpaper

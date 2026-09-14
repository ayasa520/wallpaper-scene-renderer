#pragma once

#include "VulkanPass.hpp"
#include "Vulkan/Parameters.hpp"

#include <memory>
#include <string>

namespace wallpaper::vulkan
{

struct MipHistoryState;

class MipHistoryPass : public VulkanPass {
public:
    struct Submission {
        void Commit();

        MipHistoryState* state { nullptr };
        uint64_t disable_revision { 0 };
        uint64_t frame { 0 };
        VkImage image { VK_NULL_HANDLE };
        uint32_t mip_levels { 0 };
        bool trace { false };
        std::string target;
    };

    struct Desc {
        std::string target;
        std::shared_ptr<Submission> submission;
    };

    explicit MipHistoryPass(const Desc& desc): m_desc(desc) {}

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    void absorbResidencyGraphState(const VulkanPass&) override;
    bool referencesRenderTarget(std::string_view) const override;

private:
    void Bind(Scene&, const Device&, RenderingResources&);

    Desc m_desc;
    ImageParameters m_target {};
};

} // namespace wallpaper::vulkan

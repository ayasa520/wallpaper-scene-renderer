#include "PrePass.hpp"
#include "PassCommon.hpp"
#include "Scene/Scene.h"
#include "Resource.hpp"
#include "RenderTargetOps.hpp"
#include "RenderCommandTrace.hpp"
#include "Utils/Logging.h"

#include <cstdlib>

using namespace wallpaper::vulkan;

namespace
{

void BindPrePassTarget(wallpaper::Scene& scene, const Device& device, std::string_view name,
                       ImageParameters& image, bool& ok) {
    ok = false;
    const auto tex_name = std::string(name);
    if (scene.renderTargets.count(tex_name) == 0) return;
    auto& rt = scene.renderTargets.at(tex_name);
    if (auto opt = device.tex_cache().Query(tex_name, ToTexKey(rt), ! rt.allowReuse);
        opt.has_value()) {
        image = opt.value();
        ok    = static_cast<bool>(image.handle);
    }
}

} // namespace

PrePass::PrePass(const Desc& desc): m_desc(desc) {}
PrePass::~PrePass() {}

std::string PrePass::residencyKey() const {
    return "PrePass|target=" + std::string(m_desc.result);
}

bool PrePass::referencesRenderTarget(std::string_view render_target) const {
    // The pre-pass only clears the graph result target. It should stay out of text bridge refreshes
    // unless the default framebuffer-sized target itself was the resource that changed.
    return m_desc.result == render_target || render_target == SpecTex_DefaultMS;
}

void PrePass::prepare(Scene& scene, const Device& device, RenderingResources&) {
    bool have_result = false;
    BindPrePassTarget(scene, device, m_desc.result, m_desc.vk_result, have_result);
    if (! have_result) return;
    BindPrePassTarget(scene, device, SpecTex_DefaultMS, m_desc.vk_msaa, m_desc.has_msaa);
    setPrepared();
}

void PrePass::refreshResources(Scene& scene, const Device& device, RenderingResources&) {
    // Resource-only rebuilds clear the texture cache without discarding the pre-pass object. The
    // clear pass therefore has to re-query the recreated render-target image here; otherwise the
    // next frame would keep clearing a stale Vulkan image handle and the renderer can fall into a
    // black frame or crash once text-driven effect resources are rebuilt in place.
    bool have_result = false;
    BindPrePassTarget(scene, device, m_desc.result, m_desc.vk_result, have_result);
    if (! have_result) {
        setPrepared(false);
        return;
    }
    BindPrePassTarget(scene, device, SpecTex_DefaultMS, m_desc.vk_msaa, m_desc.has_msaa);
}

void PrePass::execute(const Device&, RenderingResources& rr) {
    // This pass is the main scene boundary after reflection, not a device-wide frame
    // initialization. Read live scene values here so scripts/user bindings need no topology
    // rebuild, and let every model LOAD depth regardless of which model is visible first.
    const auto& scene = *rr.scene;
    const auto& color = scene.clearColor;
    const VkClearColorValue clear_color { color[0], color[1], color[2], 1.0f };
    if (scene.clearEnabled) {
        ClearRenderTargetColor(rr.command, m_desc.vk_result, clear_color, m_desc.layout);
        if (m_desc.has_msaa) {
            // Both representations must start with the same opaque scene clear. The
            // resolve copies alpha as well, while most compose draws only write RGB.
            ClearRenderTargetColor(rr.command, m_desc.vk_msaa, clear_color,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
    }
    const auto depth_action = PrepareSceneModelDepth(rr, m_desc.result, scene.clearEnabled);
    TraceRenderCommand(rr, "clear", scene.clearEnabled ? "recorded" : "preserved",
                       m_desc.result, m_desc.vk_result);
    if (m_desc.has_msaa) {
        TraceRenderCommand(rr, "clear", scene.clearEnabled ? "recorded" : "preserved",
                           SpecTex_DefaultMS, m_desc.vk_msaa);
    }
    if (wallpaper::diagnostics::Options().trace_scene_clear) {
        LOG_INFO("SceneStageClear: target='%.*s' clear-enabled=%s color-action=%s "
                 "color=[%.6f %.6f %.6f %.6f] model-depth=%s msaa=%s",
                 static_cast<int>(m_desc.result.size()), m_desc.result.data(),
                 scene.clearEnabled ? "true" : "false",
                 scene.clearEnabled ? "cleared" : "preserved",
                 color[0], color[1], color[2], 1.0f, SceneDepthActionName(depth_action).data(),
                 m_desc.has_msaa ? "true" : "false");
    }
}
void PrePass::destory(const Device&, RenderingResources&) {
    setPrepared(false);
    clearReleaseTexs();
    m_desc.vk_result = {};
    m_desc.vk_msaa = {};
}

#include "ClearPass.hpp"

#include "PassCommon.hpp"
#include "Resource.hpp"
#include "RenderTargetOps.hpp"
#include "RenderCommandTrace.hpp"
#include "Utils/Logging.h"

#include <cstdlib>

using namespace wallpaper::vulkan;

ClearPass::ClearPass(const Desc& desc): m_desc(desc) {}

ClearPass::~ClearPass() = default;

std::string ClearPass::residencyKey() const {
    return "ClearPass|target=" + m_desc.target;
}

void ClearPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto& next = static_cast<const ClearPass&>(next_pass);
    // Residency matching preserves the target identity, but a rebuilt graph can assign
    // this clear to a different traversal. For example, removing the reflected walk can
    // hand its resident clear to the main walk, whose execution must remain independent
    // of reflection enablement. Replace every graph-owned clear decision while retaining
    // the prepared image; the normal resource refresh binds this graph's target image.
    m_desc.clear_value              = next.m_desc.clear_value;
    m_desc.should_execute           = next.m_desc.should_execute;
    m_desc.use_scene_clear_color     = next.m_desc.use_scene_clear_color;
    m_desc.should_clear_color        = next.m_desc.should_clear_color;
    m_desc.should_clear_model_depth  = next.m_desc.should_clear_model_depth;
}

bool ClearPass::referencesRenderTarget(std::string_view render_target) const {
    return m_desc.target == render_target;
}

void ClearPass::prepare(Scene& scene, const Device& device, RenderingResources&) {
    if (scene.renderTargets.count(m_desc.target) == 0) {
        LOG_ERROR("ClearPass: target render target not found: %s", m_desc.target.c_str());
        return;
    }

    const auto& rt = scene.renderTargets.at(m_desc.target);
    if (auto opt = device.tex_cache().Query(m_desc.target, ToTexKey(rt), !rt.allowReuse);
        opt.has_value()) {
        m_desc.vk_target = opt.value();
    } else {
        LOG_ERROR("ClearPass: query image from cache failed: %s", m_desc.target.c_str());
        return;
    }

    setPrepared();
}

void ClearPass::refreshResources(Scene& scene, const Device& device, RenderingResources&) {
    if (scene.renderTargets.count(m_desc.target) == 0) {
        setPrepared(false);
        return;
    }

    const auto& rt = scene.renderTargets.at(m_desc.target);
    if (auto opt = device.tex_cache().Query(m_desc.target, ToTexKey(rt), !rt.allowReuse);
        opt.has_value()) {
        m_desc.vk_target = opt.value();
    } else {
        setPrepared(false);
    }
}

void ClearPass::execute(const Device&, RenderingResources& rr) {
    if (m_desc.should_execute && !m_desc.should_execute()) {
        TraceRenderCommand(rr, "clear", "execution-gate", m_desc.target, m_desc.vk_target);
        return;
    }

    auto& img = m_desc.vk_target;
    if (!img.handle) {
        TraceRenderCommand(rr, "clear", "missing-image", m_desc.target, img);
        setPrepared(false);
        return;
    }

    VkClearValue clear_value = m_desc.clear_value;
    if (m_desc.use_scene_clear_color) {
        const auto& color = rr.scene->clearColor;
        clear_value.color = { color[0], color[1], color[2], 1.0f };
    }
    const bool clear_color = !m_desc.should_clear_color || m_desc.should_clear_color();
    if (clear_color) {
        ClearRenderTargetColor(rr.command, img, clear_value.color,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    // Trace the already evaluated decision; visibility and clear callbacks can observe
    // live scene state and must never be invoked a second time just for diagnostics.
    TraceRenderCommand(rr, "clear", clear_color ? "recorded" : "preserved", m_desc.target, img);

    // A live disabled clear preserves both attachments. A freshly allocated model depth
    // still needs an attachment layout before LOAD, even when no content clear is requested.
    const auto depth_action = m_desc.should_clear_model_depth
        ? PrepareSceneModelDepth(rr, m_desc.target, m_desc.should_clear_model_depth())
        : SceneDepthAction::Unused;
    if (m_desc.use_scene_clear_color &&
        (std::getenv("WESCENE_TRACE_SCENE_CLEAR") != nullptr ||
         std::getenv("WESCENE_TRACE_REFLECTION") != nullptr)) {
        LOG_INFO("SceneStageClear: target='%s' clear-enabled=%s color-action=%s "
                 "color=[%.6f %.6f %.6f %.6f] model-depth=%s",
                 m_desc.target.c_str(), rr.scene->clearEnabled ? "true" : "false",
                 clear_color ? "cleared" : "preserved",
                 clear_value.color.float32[0], clear_value.color.float32[1],
                 clear_value.color.float32[2], clear_value.color.float32[3],
                 SceneDepthActionName(depth_action).data());
    }
}

void ClearPass::destory(const Device&, RenderingResources&) {
    setPrepared(false);
    clearReleaseTexs();
    m_desc.vk_target = {};
}

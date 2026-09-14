#include "MipHistoryPass.hpp"

#include "PassCommon.hpp"
#include "RenderCommandTrace.hpp"
#include "RenderTargetOps.hpp"
#include "Resource.hpp"
#include "Scene/Scene.h"
#include "Utils/Logging.h"
#include "Vulkan/Device.hpp"

#include <cstdlib>

using namespace wallpaper::vulkan;

void MipHistoryPass::Submission::Commit() {
    if (state == nullptr) return;
    // The graph's callback runs only after successful queue submission. Failed or
    // abandoned recording therefore leaves both allocation initialization and live
    // disable requests pending, even if another graph is prepared in the meantime.
    state->submitted_disable_revision = disable_revision;
    state->creation_clear_pending = false;
    if (trace) {
        LOG_INFO("SceneMipHistorySubmit: frame=%llu target='%s' image=%p generation=%llu "
                 "disable-revision=%llu mip-levels=%u",
                 static_cast<unsigned long long>(frame), target.c_str(),
                 reinterpret_cast<void*>(image),
                 static_cast<unsigned long long>(state->generation),
                 static_cast<unsigned long long>(disable_revision), mip_levels);
    }
    state = nullptr;
}

std::string MipHistoryPass::residencyKey() const {
    return "MipHistoryPass|target=" + m_desc.target;
}

void MipHistoryPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    // Retain the prepared binding while reconnecting recording to the new graph's
    // submission callback. The allocation-owned history remains in RenderingResources.
    m_desc.submission = static_cast<const MipHistoryPass&>(next_pass).m_desc.submission;
}

bool MipHistoryPass::referencesRenderTarget(std::string_view target) const {
    return m_desc.target == target;
}

void MipHistoryPass::Bind(Scene& scene, const Device& device, RenderingResources& rr) {
    const auto& target = scene.renderTargets.at(m_desc.target);
    const auto image = device.tex_cache().Query(m_desc.target, ToTexKey(target), !target.allowReuse);
    if (!image) {
        LOG_ERROR("MipHistoryPass: cannot bind target '%s'", m_desc.target.c_str());
        setPrepared(false);
        return;
    }
    m_target = *image;
    const uint64_t generation = device.tex_cache().RenderTargetGeneration(m_desc.target);
    auto& history = rr.mip_history;
    const bool replaced = history.generation != generation;
    if (replaced) {
        // A new enabled allocation has no disabled-content requirement. Disables that
        // preceded its existence do not reset it. A disabled allocation records its
        // creation request now and retains it through all preparation-only visits.
        history = { generation, scene.ReflectionDisableRevision(), !scene.reflectionsEnabled };
    }
    if (rr.trace_render_commands || std::getenv("WESCENE_TRACE_SCENE_CLEAR") != nullptr) {
        LOG_INFO("SceneMipHistoryBind: target='%s' image=%p generation=%llu replaced=%s "
                 "creation-pending=%s disable-revision=%llu applied-revision=%llu mip-levels=%u",
                 m_desc.target.c_str(), reinterpret_cast<void*>(m_target.handle),
                 static_cast<unsigned long long>(generation), replaced ? "true" : "false",
                 history.creation_clear_pending ? "true" : "false",
                 static_cast<unsigned long long>(scene.ReflectionDisableRevision()),
                 static_cast<unsigned long long>(history.submitted_disable_revision),
                 m_target.mipmap_level);
    }
    setPrepared();
}

void MipHistoryPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    Bind(scene, device, rr);
}

void MipHistoryPass::refreshResources(Scene& scene, const Device& device, RenderingResources& rr) {
    Bind(scene, device, rr);
}

void MipHistoryPass::execute(const Device&, RenderingResources& rr) {
    auto& history = rr.mip_history;
    auto& submission = *m_desc.submission;
    submission.state = nullptr;
    const uint64_t revision = rr.scene->ReflectionDisableRevision();
    const bool clear = history.creation_clear_pending ||
                       history.submitted_disable_revision != revision;
    if (clear) {
        // Clear every mip level before either traversal's first sampler. A uniform
        // opaque-black chain needs no downsampling, and later disabled frames retain
        // it without repeating this work. The post-main copy owns enabled updates.
        const VkClearColorValue opaque_black { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
        ClearRenderTargetColor(rr.command, m_target, opaque_black,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        submission.state = &history;
        submission.disable_revision = revision;
        submission.frame = rr.trace_render_frame;
        submission.image = m_target.handle;
        submission.mip_levels = m_target.mipmap_level;
        submission.trace = rr.trace_render_commands ||
                           std::getenv("WESCENE_TRACE_SCENE_CLEAR") != nullptr;
        submission.target = m_desc.target;
    }
    TraceRenderCommand(rr, "clear", clear ? "recorded" : "preserved", m_desc.target, m_target);
    if (rr.trace_render_commands || std::getenv("WESCENE_TRACE_SCENE_CLEAR") != nullptr) {
        LOG_INFO("SceneMipHistory: frame=%llu target='%s' image=%p generation=%llu "
                 "quality-enabled=%s disable-revision=%llu applied-revision=%llu "
                 "creation-pending=%s action=%s color=[0 0 0 1] mip-levels=%u",
                 static_cast<unsigned long long>(rr.trace_render_frame), m_desc.target.c_str(),
                 reinterpret_cast<void*>(m_target.handle),
                 static_cast<unsigned long long>(history.generation),
                 rr.scene->reflectionsEnabled ? "true" : "false",
                 static_cast<unsigned long long>(revision),
                 static_cast<unsigned long long>(history.submitted_disable_revision),
                 history.creation_clear_pending ? "true" : "false",
                 clear ? "cleared" : "preserved", m_target.mipmap_level);
    }
}

void MipHistoryPass::destory(const Device&, RenderingResources&) {
    setPrepared(false);
    clearReleaseTexs();
}

#pragma once

#include "Resource.hpp"
#include "Scene/Scene.h"
#include "Utils/Logging.h"
#include "Vulkan/Parameters.hpp"

#include <string_view>

namespace wallpaper::vulkan
{

// Command diagnostics have two sinks that share one command numbering: the human-readable log
// and the structural per-draw JSON dump (FrameTraceDump). Either
// one being active is enough to number commands and collect their inputs and uniforms.
inline bool RenderCommandLogActive(const RenderingResources& rr) {
    return wallpaper::diagnostics::Enabled && rr.trace_render_commands;
}

inline bool RenderCommandTraceActive(const RenderingResources& rr) {
    return wallpaper::diagnostics::Enabled &&
        (rr.trace_render_commands || rr.frame_trace_dump.active());
}

inline void TraceRenderCommandFrame(const RenderingResources& rr, const char* phase) {
    if constexpr (!wallpaper::diagnostics::Enabled) return;
    if (!rr.trace_render_commands) return;
    LOG_INFO("SceneRenderCommandFrame: frame=%llu time=%.6f phase=%s commands=%llu "
             "renderer=%p scene=%p",
             static_cast<unsigned long long>(rr.trace_render_frame), rr.scene->elapsingTime,
             phase, static_cast<unsigned long long>(rr.trace_render_command),
             static_cast<const void*>(&rr), static_cast<const void*>(rr.scene));
}

inline void BeginRenderCommandTrace(RenderingResources& rr) {
    if constexpr (!wallpaper::diagnostics::Enabled) return;
    ++rr.draw_index;
    rr.frame_trace_dump.beginFrame(rr.draw_index, rr.scene ? rr.scene->elapsingTime : 0.0);
    rr.trace_render_commands = wallpaper::diagnostics::Options().trace_render_commands;
    if (!RenderCommandTraceActive(rr)) return;
    ++rr.trace_render_frame;
    rr.trace_render_command = 0;
    // A scene clock can repeat and graph preparation can run without submission. Keep
    // recording attempts distinct for the renderer lifetime, and pair their commands
    // with an explicit successful queue-submit record before treating them as GPU work.
    TraceRenderCommandFrame(rr, "recording");
}

// Closes the structural dump for a frame whose GPU work was queued. The log sink has no
// per-frame footer beyond the "submitted" phase record.
inline void EndRenderCommandTrace(RenderingResources& rr, uint32_t output_width,
                                  uint32_t output_height, size_t tracked_images) {
    if constexpr (!wallpaper::diagnostics::Enabled) return;
    TraceRenderCommandFrame(rr, "submitted");
    rr.frame_trace_dump.finishFrame("submitted", output_width, output_height, tracked_images);
}

inline uint64_t TraceRenderCommand(RenderingResources& rr, const char* kind, const char* result,
                                    std::string_view output, const ImageParameters& image,
                                    int32_t layer = 0, bool reflection = false, uint32_t count = 0) {
    if (!RenderCommandTraceActive(rr)) return 0;
    const auto command = ++rr.trace_render_command;
    if (rr.trace_render_commands) {
        LOG_INFO("SceneRenderCommand: frame=%llu command=%llu time=%.6f kind=%s result=%s "
                 "layer=%d reflection=%s count=%u output='%.*s' image=%p extent=%ux%u samples=%u",
                 static_cast<unsigned long long>(rr.trace_render_frame),
                 static_cast<unsigned long long>(command), rr.scene->elapsingTime, kind, result,
                 layer, reflection ? "true" : "false", count,
                 static_cast<int>(output.size()), output.data(), reinterpret_cast<void*>(image.handle),
                 image.extent.width, image.extent.height, image.samples);
    }
    rr.frame_trace_dump.addCommand(command, kind, result, output, image, layer, reflection, count);
    return command;
}

inline void TraceRenderCommandInput(RenderingResources& rr, uint64_t command,
                                    const char* role, std::string_view key,
                                    const ImageParameters& image, bool read, int32_t binding = -1) {
    if (!RenderCommandTraceActive(rr)) return;
    // Skipped passes and optimized-out samplers retain their binding metadata. Mark
    // those explicitly so an ordering audit cannot mistake a descriptor for a read.
    if (rr.trace_render_commands) {
        LOG_INFO("SceneRenderCommandInput: frame=%llu command=%llu role=%s access=%s "
                 "binding=%d key='%.*s' image=%p extent=%ux%u samples=%u",
                 static_cast<unsigned long long>(rr.trace_render_frame),
                 static_cast<unsigned long long>(command), role, read ? "read" : "metadata", binding,
                 static_cast<int>(key.size()), key.data(), reinterpret_cast<void*>(image.handle),
                 image.extent.width, image.extent.height, image.samples);
    }
    rr.frame_trace_dump.addInput(command, role, key, image, read, binding);
}

} // namespace wallpaper::vulkan

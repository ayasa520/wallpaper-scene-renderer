#pragma once

#include "Resource.hpp"
#include "Scene/Scene.h"
#include "Utils/Logging.h"
#include "Vulkan/Parameters.hpp"

#include <cstdlib>
#include <string_view>

namespace wallpaper::vulkan
{

inline void TraceRenderCommandFrame(const RenderingResources& rr, const char* phase) {
    if (!rr.trace_render_commands) return;
    LOG_INFO("SceneRenderCommandFrame: frame=%llu time=%.6f phase=%s commands=%llu "
             "renderer=%p scene=%p",
             static_cast<unsigned long long>(rr.trace_render_frame), rr.scene->elapsingTime,
             phase, static_cast<unsigned long long>(rr.trace_render_command),
             static_cast<const void*>(&rr), static_cast<const void*>(rr.scene));
}

inline void BeginRenderCommandTrace(RenderingResources& rr) {
    rr.trace_render_commands = std::getenv("WESCENE_TRACE_RENDER_COMMANDS") != nullptr;
    if (!rr.trace_render_commands) return;
    ++rr.trace_render_frame;
    rr.trace_render_command = 0;
    // A scene clock can repeat and graph preparation can run without submission. Keep
    // recording attempts distinct for the renderer lifetime, and pair their commands
    // with an explicit successful queue-submit record before treating them as GPU work.
    TraceRenderCommandFrame(rr, "recording");
}

inline uint64_t TraceRenderCommand(RenderingResources& rr, const char* kind, const char* result,
                                    std::string_view output, const ImageParameters& image,
                                    int32_t layer = 0, bool reflection = false, uint32_t count = 0) {
    if (!rr.trace_render_commands) return 0;
    const auto command = ++rr.trace_render_command;
    LOG_INFO("SceneRenderCommand: frame=%llu command=%llu time=%.6f kind=%s result=%s "
             "layer=%d reflection=%s count=%u output='%.*s' image=%p extent=%ux%u samples=%u",
             static_cast<unsigned long long>(rr.trace_render_frame),
             static_cast<unsigned long long>(command), rr.scene->elapsingTime, kind, result,
             layer, reflection ? "true" : "false", count,
             static_cast<int>(output.size()), output.data(), reinterpret_cast<void*>(image.handle),
             image.extent.width, image.extent.height, image.samples);
    return command;
}

inline void TraceRenderCommandInput(const RenderingResources& rr, uint64_t command,
                                     const char* role, std::string_view key,
                                     const ImageParameters& image, bool read, int32_t binding = -1) {
    if (!rr.trace_render_commands) return;
    // Skipped passes and optimized-out samplers retain their binding metadata. Mark
    // those explicitly so an ordering audit cannot mistake a descriptor for a read.
    LOG_INFO("SceneRenderCommandInput: frame=%llu command=%llu role=%s access=%s "
             "binding=%d key='%.*s' image=%p extent=%ux%u samples=%u",
             static_cast<unsigned long long>(rr.trace_render_frame),
             static_cast<unsigned long long>(command), role, read ? "read" : "metadata", binding,
             static_cast<int>(key.size()), key.data(), reinterpret_cast<void*>(image.handle),
             image.extent.width, image.extent.height, image.samples);
}

} // namespace wallpaper::vulkan

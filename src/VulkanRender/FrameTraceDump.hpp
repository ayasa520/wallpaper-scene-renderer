#pragma once

/*
 * Structural frame trace for golden-frame regression tests.
 *
 * A pixel golden only says that something changed; this dump says what. For selected draws it
 * records every render command in execution order (kind, result, destination target, layer,
 * reflection flag, draw count, target extent and sample count), each command's texture inputs
 * (role, texture key, binding, whether the recorded GPU work reads it) and the uniform values
 * uploaded for shader draws, then writes one JSON document per draw. Everything that varies
 * between processes for the same behavior (Vulkan handles, host pointers, timings) is left out,
 * so two runs of a deterministic capture produce byte-identical files and any difference names
 * the command, texture or uniform that changed.
 *
 *   WESCENE_DUMP_FRAME_TRACE=<dir>          enables the dump and selects the output directory
 *   WESCENE_DUMP_FRAME_TRACE_DRAWS=1,60,120 1-based draw numbers to dump, or "all"
 *
 * Files are named draw-NNNNNN.json.
 */

#include "Vulkan/Parameters.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace wallpaper::vulkan
{

class FrameTraceDump {
public:
    struct Input {
        std::string role;
        std::string key;
        bool        read { false };
        int32_t     binding { -1 };
        uint32_t    width { 0 };
        uint32_t    height { 0 };
        uint32_t    samples { 0 };
        uint64_t    allocation_revision { 0 };
        uint64_t    allocation_generation { 0 };
    };
    struct Uniform {
        std::string        name;
        size_t             offset { 0 };
        size_t             reflected_bytes { 0 };
        std::vector<float> values;
    };
    struct Command {
        uint64_t             index { 0 };
        std::string          kind;
        std::string          result;
        std::string          output;
        int32_t              layer { 0 };
        bool                 reflection { false };
        uint32_t             count { 0 };
        uint32_t             width { 0 };
        uint32_t             height { 0 };
        uint32_t             samples { 0 };
        uint64_t             allocation_revision { 0 };
        uint64_t             allocation_generation { 0 };
        std::vector<Input>   inputs;
        std::vector<Uniform> uniforms;
    };

    // True when the environment selects a dump directory. Cheap enough to call per frame.
    static bool Configured();

    // Called once per draw before any command is recorded. Decides whether this draw is dumped.
    void beginFrame(uint64_t draw_index, double scene_time);
    bool active() const { return m_active; }

    uint64_t addCommand(uint64_t index, const char* kind, const char* result,
                        std::string_view output, const ImageParameters& image, int32_t layer,
                        bool reflection, uint32_t count);
    void     addInput(uint64_t command, const char* role, std::string_view key,
                      const ImageParameters& image, bool read, int32_t binding);
    void     addUniform(uint64_t command, std::string_view name, size_t offset,
                        size_t reflected_bytes, const float* values, size_t count);

    // Writes the document for the current draw. `phase` names how the frame ended
    // ("submitted" for a frame whose GPU work was queued).
    void finishFrame(const char* phase, uint32_t output_width, uint32_t output_height,
                     size_t tracked_images);

private:
    Command* find(uint64_t command);

    bool                 m_active { false };
    uint64_t             m_draw_index { 0 };
    double               m_scene_time { 0.0 };
    std::vector<Command> m_commands;
};

} // namespace wallpaper::vulkan

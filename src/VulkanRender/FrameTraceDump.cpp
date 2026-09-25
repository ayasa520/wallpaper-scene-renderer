#include "FrameTraceDump.hpp"

#include "Utils/Logging.h"

#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>

using namespace wallpaper::vulkan;

namespace
{

struct DumpSettings {
    std::string        directory;
    bool               all_draws { false };
    std::set<uint64_t> draws;
};

const DumpSettings& Settings() {
    static const DumpSettings settings = [] {
        DumpSettings result;
        const char* directory = wallpaper::diagnostics::Options().frame_trace_directory;
        if (directory == nullptr || *directory == '\0') return result;
        result.directory = directory;
        const char* draws = wallpaper::diagnostics::Options().frame_trace_draws;
        if (draws == nullptr || *draws == '\0' || std::string_view(draws) == "all") {
            result.all_draws = true;
            return result;
        }
        std::string_view list(draws);
        while (! list.empty()) {
            const auto comma = list.find(',');
            const auto item  = list.substr(0, comma);
            uint64_t   value = 0;
            if (std::from_chars(item.data(), item.data() + item.size(), value).ec == std::errc {}) {
                result.draws.insert(value);
            }
            if (comma == std::string_view::npos) break;
            list.remove_prefix(comma + 1);
        }
        return result;
    }();
    return settings;
}

void WriteString(std::ostream& out, std::string_view text) {
    out << '"';
    for (const unsigned char c : text) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out << buffer;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
}

void WriteFloat(std::ostream& out, float value) {
    // Nine significant digits round-trip every float, so identical uploads produce identical
    // text and any difference is a real value change.
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(value));
    std::string_view text(buffer);
    if (text == "nan" || text == "-nan" || text == "inf" || text == "-inf") {
        WriteString(out, text);
    } else {
        out << text;
    }
}

} // namespace

bool FrameTraceDump::Configured() { return ! Settings().directory.empty(); }

void FrameTraceDump::beginFrame(uint64_t draw_index, double scene_time) {
    m_commands.clear();
    m_draw_index = draw_index;
    m_scene_time = scene_time;
    const auto& settings = Settings();
    m_active = ! settings.directory.empty() &&
               (settings.all_draws || settings.draws.count(draw_index) != 0);
}

FrameTraceDump::Command* FrameTraceDump::find(uint64_t command) {
    // Commands are appended in order and inputs/uniforms arrive for the most recent one.
    for (auto it = m_commands.rbegin(); it != m_commands.rend(); ++it) {
        if (it->index == command) return &*it;
    }
    return nullptr;
}

uint64_t FrameTraceDump::addCommand(uint64_t index, const char* kind, const char* result,
                                    std::string_view output, const ImageParameters& image,
                                    int32_t layer, bool reflection, uint32_t count) {
    if (! m_active) return index;
    Command command;
    command.index      = index;
    command.kind       = kind;
    command.result     = result;
    command.output     = std::string(output);
    command.layer      = layer;
    command.reflection = reflection;
    command.count      = count;
    command.width      = image.extent.width;
    command.height     = image.extent.height;
    command.samples    = image.samples;
    command.allocation_revision = image.allocation_revision;
    command.allocation_generation = image.allocation_generation;
    m_commands.push_back(std::move(command));
    return index;
}

void FrameTraceDump::addInput(uint64_t command, const char* role, std::string_view key,
                              const ImageParameters& image, bool read, int32_t binding) {
    if (! m_active) return;
    Command* target = find(command);
    if (target == nullptr) return;
    target->inputs.push_back(Input {
        .role    = role,
        .key     = std::string(key),
        .read    = read,
        .binding = binding,
        .width   = image.extent.width,
        .height  = image.extent.height,
        .samples = image.samples,
        .allocation_revision = image.allocation_revision,
        .allocation_generation = image.allocation_generation,
    });
}

void FrameTraceDump::addUniform(uint64_t command, std::string_view name, size_t offset,
                                size_t reflected_bytes, const float* values, size_t count) {
    if (! m_active) return;
    Command* target = find(command);
    if (target == nullptr) return;
    target->uniforms.push_back(Uniform {
        .name            = std::string(name),
        .offset          = offset,
        .reflected_bytes = reflected_bytes,
        .values          = std::vector<float>(values, values + count),
    });
}

void FrameTraceDump::finishFrame(const char* phase, uint32_t output_width,
                                 uint32_t output_height, size_t tracked_images) {
    if (! m_active) return;
    m_active = false;

    std::ostringstream out;
    out << "{\n";
    out << "  \"draw\": " << m_draw_index << ",\n";
    out << "  \"scene_time\": ";
    WriteFloat(out, static_cast<float>(m_scene_time));
    out << ",\n";
    out << "  \"phase\": ";
    WriteString(out, phase);
    out << ",\n";
    out << "  \"output\": [" << output_width << ", " << output_height << "],\n";
    out << "  \"resources\": {\"tracked_images\": " << tracked_images << "},\n";
    out << "  \"commands\": [\n";
    for (size_t i = 0; i < m_commands.size(); ++i) {
        const auto& command = m_commands[i];
        out << "    {\"index\": " << command.index << ", \"kind\": ";
        WriteString(out, command.kind);
        out << ", \"result\": ";
        WriteString(out, command.result);
        out << ", \"output\": ";
        WriteString(out, command.output);
        out << ", \"layer\": " << command.layer
            << ", \"reflection\": " << (command.reflection ? "true" : "false")
            << ", \"count\": " << command.count << ", \"extent\": [" << command.width << ", "
            << command.height << "], \"samples\": " << command.samples;
        // Only explicitly reconstructed targets need this extra identity. Read it from each
        // command's retained image binding, not from the scene's latest requested descriptor:
        // stale bindings and missing same-size replacements must remain observable to the gate.
        if (command.allocation_revision != 0) {
            out << ", \"allocation\": {\"revision\": " << command.allocation_revision
                << ", \"generation\": " << command.allocation_generation << "}";
        }
        if (! command.inputs.empty()) {
            out << ",\n     \"inputs\": [";
            for (size_t j = 0; j < command.inputs.size(); ++j) {
                const auto& input = command.inputs[j];
                out << (j ? ",\n       " : "\n       ") << "{\"role\": ";
                WriteString(out, input.role);
                out << ", \"key\": ";
                WriteString(out, input.key);
                out << ", \"access\": " << (input.read ? "\"read\"" : "\"metadata\"")
                    << ", \"binding\": " << input.binding << ", \"extent\": [" << input.width
                    << ", " << input.height << "], \"samples\": " << input.samples;
                if (input.allocation_revision != 0) {
                    out << ", \"allocation\": {\"revision\": " << input.allocation_revision
                        << ", \"generation\": " << input.allocation_generation << "}";
                }
                out << "}";
            }
            out << "]";
        }
        if (! command.uniforms.empty()) {
            out << ",\n     \"uniforms\": [";
            for (size_t j = 0; j < command.uniforms.size(); ++j) {
                const auto& uniform = command.uniforms[j];
                out << (j ? ",\n       " : "\n       ") << "{\"name\": ";
                WriteString(out, uniform.name);
                out << ", \"offset\": " << uniform.offset << ", \"bytes\": "
                    << uniform.reflected_bytes << ", \"values\": [";
                for (size_t k = 0; k < uniform.values.size(); ++k) {
                    if (k) out << ", ";
                    WriteFloat(out, uniform.values[k]);
                }
                out << "]}";
            }
            out << "]";
        }
        out << "}" << (i + 1 < m_commands.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";

    const auto& settings = Settings();
    std::error_code ec;
    std::filesystem::create_directories(settings.directory, ec);
    char name[64];
    std::snprintf(name, sizeof(name), "draw-%06llu.json",
                  static_cast<unsigned long long>(m_draw_index));
    const std::filesystem::path path = std::filesystem::path(settings.directory) / name;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (! file) {
        LOG_ERROR("FrameTraceDump: cannot write %s", path.c_str());
        return;
    }
    file << out.str();
    LOG_INFO("FrameTraceDump: draw=%llu commands=%zu path=%s",
             static_cast<unsigned long long>(m_draw_index), m_commands.size(), path.c_str());
}

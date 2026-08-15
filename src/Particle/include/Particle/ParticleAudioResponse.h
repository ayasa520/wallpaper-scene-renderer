#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>

namespace wallpaper
{

// Particle audio consumers share sixteen bands and the same authored parameter block.
inline constexpr std::size_t kParticleAudioBandCount = 16;

struct ParticleAudioResponseParams {
    uint32_t             mode { 0 }; // 0 = bypass, 1 = left, 2 = right, 3 = stereo per-band mean.
    float                exponent { 2.0f };
    std::array<float, 2> bounds { 0.8f, 1.0f };
    uint32_t             frequency_start { 0 };
    uint32_t             frequency_end { 1 };
};

// audioprocessingbounds is a space-separated string. A missing key keeps the default [0.8, 1.0].
// An empty string writes [0, 0]; a single token leaves the second endpoint at 0. Do not reuse the
// generic JSON array helper for the string form — a size mismatch would preserve the default pair.
inline void AssignAudioBoundsFromAuthoredString(std::string_view text,
                                                std::array<float, 2>& bounds) {
    if (text.empty()) {
        bounds = { 0.0f, 0.0f };
        return;
    }
    const std::string owned { text };
    bounds[0] = std::strtof(owned.c_str(), nullptr);
    const char* cursor = owned.c_str();
    while (*cursor != '\0' && *cursor != ' ') ++cursor;
    while (*cursor == ' ') ++cursor;
    bounds[1] = std::strtof(cursor, nullptr);
}

// An empty callback is the authored mode-zero bypass. Nonzero modes bind one live response
// evaluation so every consumer applies the same curve to the same frame's spectrum snapshot.
using ParticleAudioResponseFactor = std::function<double()>;

} // namespace wallpaper

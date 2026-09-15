#pragma once

/*
 * Lockstep test mode.
 *
 * Golden-frame regression tests need two runs of the same wallpaper to produce identical
 * frames. In production the renderer owns its own clock (a periodic timer posts draws, the
 * scene advances by the measured frame duration, particles seed their generators from the
 * hardware entropy source and scripts read the calendar clock). All of those are inputs that
 * only the process environment controls, so the harness pins them through environment
 * variables read once at first use:
 *
 *   WESCENE_LOCKSTEP=1        The frame timer never runs on its own and never re-requests a
 *                             draw after a slow frame; every draw is one explicit request from
 *                             the embedding producer, and the scene clock advances by the fixed
 *                             step below instead of the measured duration.
 *   WESCENE_FIXED_DT=<sec>    Scene time advanced per draw in lockstep mode. Defaults to the
 *                             configured 1/fps when absent.
 *   WESCENE_RANDOM_SEED=<n>   Seed applied to the thread-local random engine of every thread
 *                             that emits particles or parses random ranges, once per scene load.
 *   WESCENE_FIXED_EPOCH=<sec> Unix time used as the calendar clock at scene time zero. Scripts
 *                             and time-of-day bindings see this epoch plus the scene clock, so
 *                             clock-driven layers still animate, identically on every run.
 *
 * Nothing here changes behavior unless the variables are set; the production path stays on the
 * measured timer clock and hardware-seeded generators.
 */

#include "Core/Random.hpp"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <system_error>

namespace wallpaper::determinism
{

namespace detail
{

inline std::optional<std::string_view> Env(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return std::nullopt;
    return std::string_view(value);
}

inline std::optional<double> EnvDouble(const char* name) {
    const auto text = Env(name);
    if (! text) return std::nullopt;
    char*        end    = nullptr;
    const double parsed = std::strtod(text->data(), &end);
    if (end == text->data()) return std::nullopt;
    return parsed;
}

inline std::optional<uint64_t> EnvU64(const char* name) {
    const auto text = Env(name);
    if (! text) return std::nullopt;
    uint64_t   parsed = 0;
    const auto result = std::from_chars(text->data(), text->data() + text->size(), parsed);
    if (result.ec != std::errc {}) return std::nullopt;
    return parsed;
}

} // namespace detail

inline bool LockstepEnabled() {
    static const bool enabled = detail::Env("WESCENE_LOCKSTEP") == std::string_view("1");
    return enabled;
}

// Scene seconds advanced by each lockstep draw. `fallback` is the embedding's 1/fps.
inline double FixedFrameTime(double fallback) {
    static const std::optional<double> configured = detail::EnvDouble("WESCENE_FIXED_DT");
    if (configured && *configured > 0.0) return *configured;
    return fallback;
}

inline std::optional<uint64_t> RandomSeed() {
    static const std::optional<uint64_t> seed = detail::EnvU64("WESCENE_RANDOM_SEED");
    return seed;
}

// Reseeds the calling thread's random engine. The engine is thread-local, so this has to run
// on every thread whose random draws influence the frame: the loading thread (particle range
// parsing and prewarm) and the render thread (per-frame emission).
inline void SeedThreadRandom() {
    const auto seed = RandomSeed();
    if (! seed) return;
    Random::seed(static_cast<Random::engine_type::result_type>(*seed));
}

inline std::optional<double> FixedEpoch() {
    static const std::optional<double> epoch = detail::EnvDouble("WESCENE_FIXED_EPOCH");
    return epoch;
}

} // namespace wallpaper::determinism

#pragma once

/*
 * Lockstep test mode.
 *
 * Golden-frame regression tests need two runs of the same wallpaper to produce identical
 * frames. In production the renderer owns its own clock (a periodic timer posts draws, the
 * scene advances by the measured frame duration, particles seed their generators from the
 * hardware entropy source and scripts read the calendar clock). All of those are inputs that
 * the harness pins through the development diagnostic configuration. Lockstep disables the
 * periodic timer and advances scene time by one fixed step per explicit draw request. A seed
 * replays random draws, and a fixed epoch anchors calendar time to the advancing scene clock.
 *
 * These inputs exist only in builds with WESCENE_ENABLE_DIAGNOSTICS. Production uses the
 * measured timer clock and hardware-seeded generators, with no environment reads or static
 * initialization guards at the call sites in its frame and script paths.
 */

#include "Core/Random.hpp"
#include "Utils/Diagnostics.h"

#include <cstdint>
#include <optional>

namespace wallpaper::determinism
{

inline bool LockstepEnabled() {
    return diagnostics::Options().lockstep;
}

// Scene seconds advanced by each lockstep draw. The embedding supplies its configured 1/fps.
inline double FixedFrameTime(double frame_time) {
    const auto configured = diagnostics::Options().fixed_dt;
    if (configured && *configured > 0.0) return *configured;
    return frame_time;
}

inline std::optional<uint64_t> RandomSeed() {
    return diagnostics::Options().random_seed;
}

// Reseeds the calling thread's random engine. The engine is thread-local, so this has to run
// on every thread whose random draws influence the frame: the loading thread (particle range
// parsing and prewarm) and the render thread (per-frame emission).
inline void SeedThreadRandom() {
    if constexpr (diagnostics::Enabled) {
        if (const auto seed = RandomSeed())
            Random::seed(static_cast<Random::engine_type::result_type>(*seed));
    }
}

inline std::optional<double> FixedEpoch() {
    return diagnostics::Options().fixed_epoch;
}

} // namespace wallpaper::determinism

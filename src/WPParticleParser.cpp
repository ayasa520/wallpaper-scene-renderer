#include "WPParticleParser.hpp"
#include "Particle/ParticleEmitter.h"
#include "Particle/ParticleModify.h"
#include "Particle/ParticleParallelExecutor.h"
#include "Particle/ParticleSystem.h"
#include <random>
#include <memory>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Utils/Logging.h"
#include "Utils/Algorism.h"
#include "Core/Random.hpp"

using namespace wallpaper;
using namespace Eigen;
namespace PM = ParticleModify;

namespace
{

// Keep a tiny dead-zone around the force center so particles exactly under the
// mouse/control point do not generate an undefined normalized direction.
constexpr double kControlPointForceMinDistance = 0.0001;
constexpr double kParticleDirectionMinVectorLengthSquared = 1e-12;

inline bool IsFiniteNonZeroVector(const Vector3d& value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z()) &&
           value.squaredNorm() > kParticleDirectionMinVectorLengthSquared;
}

inline Vector3d NormalizeOr(const Vector3d& value, const Vector3d& fallback) {
    if (IsFiniteNonZeroVector(value)) return value.normalized();
    return fallback;
}

inline i32 NormalizeControlPointIndex(i32 index) {
    if (index < 0) {
        LOG_ERROR("wrong controlpoint index %d", index);
        return 0;
    }
    if (index >= static_cast<i32>(wpscene::kParticleControlpointSlotCount)) {
        LOG_ERROR("wrong controlpoint index %d", index);
        return index % static_cast<i32>(wpscene::kParticleControlpointSlotCount);
    }
    return index;
}

inline float StableParticleRandom01(uint64_t sequence) {
    uint64_t value = sequence + 0x9e3779b97f4a7c15ULL;
    value          = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value          = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<float>(value >> 40) * (1.0f / 16777216.0f);
}

inline float MapSequenceCount(float authored_count) {
    if (! std::isfinite(authored_count) || authored_count <= 1.0f) return 1.0f;
    return authored_count;
}

inline double MapSequencePhase(uint64_t sequence, float count) {
    const double step  = 1.0 / static_cast<double>(count);
    double       phase = static_cast<double>(sequence) * step;
    if (phase > 1.0) phase = std::fmod(phase, 1.0);
    if (phase < 0.0) phase += 1.0;
    return phase;
}

inline ParticleEmitterTiming MakeEmitterTiming(const wpscene::Emitter& wpe,
                                               ParticleAudioResponseFactor audio_rate_factor) {
    ParticleEmitterTiming timing;
    timing.emit_speed              = wpe.rate;
    timing.one_per_frame           = wpe.flags[wpscene::Emitter::FlagEnum::one_per_frame];
    timing.periodic                = wpe.flags[wpscene::Emitter::FlagEnum::periodic];
    timing.instantaneous           = wpe.instantaneous;
    timing.max_to_emit_per_period  = wpe.maxtoemitperperiod;
    timing.min_periodic_duration   = wpe.minperiodicduration;
    timing.max_periodic_duration   = wpe.maxperiodicduration;
    timing.min_periodic_delay      = wpe.minperiodicdelay;
    timing.max_periodic_delay      = wpe.maxperiodicdelay;
    timing.duration                = wpe.duration;
    timing.delay                   = wpe.delay;
    timing.audio_rate_factor       = std::move(audio_rate_factor);
    return timing;
}

inline Eigen::Vector3f DecodeParticleColorEndpoint(
    const std::array<float, 3>& encoded_color) {
    // Wallpaper Engine serializes colorrandom endpoints as RGB byte triplets while the particle
    // simulation and shaders consume normalized RGB values.
    constexpr float kByteToNormalized = 1.0f / 255.0f;
    return { encoded_color[0] * kByteToNormalized,
             encoded_color[1] * kByteToNormalized,
             encoded_color[2] * kByteToNormalized };
}

inline Vector3d GenRandomVec3(const std::array<float, 3>& min, const std::array<float, 3>& max) {
    Vector3d result(3);
    for (int32_t i = 0; i < 3; i++) {
        result[i] = Random::get(min[i], max[i]);
    }
    return result;
}

} // namespace

struct SingleRandom {
    float       min { 0.0f };
    float       max { 0.0f };
    float       exponent { 1.0f };
    static void ReadFromJson(const nlohmann::json& j, SingleRandom& r) {
        GET_JSON_NAME_VALUE_NOWARN(j, "min", r.min);
        GET_JSON_NAME_VALUE_NOWARN(j, "max", r.max);
    };
};
struct VecRandom {
    std::array<float, 3> min { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> max { 0.0f, 0.0f, 0.0f };
    float                exponent { 1.0f };

    static void ReadFromJson(const nlohmann::json& j, VecRandom& r) {
        GET_JSON_NAME_VALUE_NOWARN(j, "min", r.min);
        GET_JSON_NAME_VALUE_NOWARN(j, "max", r.max);
    };
};
struct TurbulentRandom {
    float  scale { 1.0f };
    double timescale { 1.0f };
    float  offset { 0.0f };
    float  speedmin { 100.0f };
    float  speedmax { 250.0f };
    float  phasemin { 0.0f };
    float  phasemax { 0.1f };

    std::array<float, 3> forward { 0.0f, 1.0f, 0.0f }; // x y z
    std::array<float, 3> right { 0.0f, 0.0f, 1.0f };
    std::array<float, 3> up { 1.0f, 0.0f, 0.0f };

    static void ReadFromJson(const nlohmann::json& j, TurbulentRandom& r) {
        GET_JSON_NAME_VALUE_NOWARN(j, "scale", r.scale);
        GET_JSON_NAME_VALUE_NOWARN(j, "timescale", r.timescale);
        GET_JSON_NAME_VALUE_NOWARN(j, "offset", r.offset);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedmin", r.speedmin);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedmax", r.speedmax);
        GET_JSON_NAME_VALUE_NOWARN(j, "phasemin", r.phasemin);
        GET_JSON_NAME_VALUE_NOWARN(j, "phasemax", r.phasemax);
        GET_JSON_NAME_VALUE_NOWARN(j, "forward", r.forward);
        GET_JSON_NAME_VALUE_NOWARN(j, "right", r.right);
        GET_JSON_NAME_VALUE_NOWARN(j, "up", r.up);
    };
};

struct MapSequenceAroundControlPoint {
    i32                  controlpoint { 0 };
    float                count { 32.0f };
    std::array<float, 3> speedmin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> speedmax { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> axis { 0.0f, 0.0f, 1.0f };
    std::array<float, 2> bounds { 0.0f, 1.0f };
    bool                 mirror { false };

    static auto ReadFromJson(const nlohmann::json& j) {
        MapSequenceAroundControlPoint v;
        GET_JSON_NAME_VALUE_NOWARN(j, "controlpoint", v.controlpoint);
        if (v.controlpoint < 0) {
            LOG_ERROR("wrong contropoint index %d", v.controlpoint);
            v.controlpoint = 0;
        } else if (v.controlpoint >= 8) {
            LOG_ERROR("wrong contropoint index %d", v.controlpoint);
            v.controlpoint %= 8;
        }

        GET_JSON_NAME_VALUE_NOWARN(j, "count", v.count);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedmin", v.speedmin);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedmax", v.speedmax);
        GET_JSON_NAME_VALUE_NOWARN(j, "axis", v.axis);
        GET_JSON_NAME_VALUE_NOWARN(j, "bounds", v.bounds);
        GET_JSON_NAME_VALUE_NOWARN(j, "mirror", v.mirror);
        return v;
    }
};

ParticleInitOp WPParticleParser::genParticleInitOp(const nlohmann::json&       wpj,
                                                   ParticleAudioResponseFactor audio_factor) {
    using namespace std::placeholders;
    do {
        if (! wpj.contains("name")) break;
        std::string name;
        GET_JSON_NAME_VALUE(wpj, "name", name);

        if (name == "colorrandom") {
            VecRandom r;
            r.min = { 0.0f, 0.0f, 0.0f };
            r.max = { 255.0f, 255.0f, 255.0f };
            VecRandom::ReadFromJson(wpj, r);
            const auto min_color = DecodeParticleColorEndpoint(r.min);
            const auto max_color = DecodeParticleColorEndpoint(r.max);
            return [min_color, max_color](Particle& p, const ParticleInitInfo&) {
                // A single random value selects a point on the authored RGB gradient and is retained
                // so later instanceoverride color edits can transform the endpoints first.
                PM::InitColorRandom(
                    p, min_color, max_color, static_cast<float>(Random::get(0.0, 1.0)));
            };
        } else if (name == "lifetimerandom") {
            SingleRandom r = { 0.0f, 1.0f };
            SingleRandom::ReadFromJson(wpj, r);
            return [=](Particle& p, const ParticleInitInfo&) {
                PM::InitLifetime(p, Random::get(r.min, r.max));
            };
        } else if (name == "sizerandom") {
            SingleRandom r = { 0.0f, 20.0f };
            SingleRandom::ReadFromJson(wpj, r);
            return [=](Particle& p, const ParticleInitInfo&) {
                PM::InitSize(p, Random::get(r.min, r.max));
            };
        } else if (name == "alpharandom") {
            SingleRandom r = { 0.05f, 1.0f };
            SingleRandom::ReadFromJson(wpj, r);
            return [=](Particle& p, const ParticleInitInfo&) {
                PM::InitAlpha(p, Random::get(r.min, r.max));
            };
        } else if (name == "velocityrandom") {
            VecRandom r;
            r.min[0] = r.min[1] = -32.0f;
            r.max[0] = r.max[1] = 32.0f;
            VecRandom::ReadFromJson(wpj, r);
            return [=](Particle& p, const ParticleInitInfo&) {
                auto result = GenRandomVec3(r.min, r.max);
                // Simulation is in the particle node's local space; the node transform is applied
                // at render, so this velocity stays local.
                PM::ChangeVelocity(p, result[0], result[1], result[2]);
            };
        } else if (name == "rotationrandom") {
            VecRandom r;
            r.max[2] = 2 * M_PI;
            VecRandom::ReadFromJson(wpj, r);
            return [=](Particle& p, const ParticleInitInfo&) {
                auto result = GenRandomVec3(r.min, r.max);
                PM::ChangeRotation(p, result[0], result[1], result[2]);
            };
        } else if (name == "angularvelocityrandom") {
            VecRandom r;
            r.min[2] = -5.0f;
            r.max[2] = 5.0f;
            VecRandom::ReadFromJson(wpj, r);
            return [=](Particle& p, const ParticleInitInfo&) {
                auto result = GenRandomVec3(r.min, r.max);
                PM::ChangeAngularVelocity(p, result[0], result[1], result[2]);
            };
        } else if (name == "mapsequencearoundcontrolpoint") {
            MapSequenceAroundControlPoint m     = MapSequenceAroundControlPoint::ReadFromJson(wpj);
            const float                   count = MapSequenceCount(m.count);
            const Vector3d                axis =
                NormalizeOr((Vector3f { m.axis.data() }).cast<double>(), Vector3d::UnitZ());
            const double bounds_min  = static_cast<double>(m.bounds[0]);
            const double bounds_span = static_cast<double>(m.bounds[1] - m.bounds[0]);
            return [=](Particle& p, const ParticleInitInfo& info) {
                Vector3d center = Vector3d::Zero();
                if (m.controlpoint >= 0 &&
                    static_cast<usize>(m.controlpoint) < info.controlpoints.size()) {
                    center = info.controlpoints[static_cast<usize>(m.controlpoint)].offset;
                }

                /*
                 * Each spawn advances a shared 0..1 phase by 1/count. The phase is used before
                 * wrapping, so a value of exactly 1 stays 1 (a full turn, same pose as 0). Bounds
                 * scale that phase into an angle around the authored axis. Position is rotated
                 * around the control point; sequenced speed is added to the emitter velocity.
                 */
                double phase = MapSequencePhase(info.sequence, count);
                if (m.mirror) {
                    const double folded = std::fmod(phase * 2.0, 2.0);
                    phase               = folded <= 1.0 ? folded : 2.0 - folded;
                }
                const double           angle = (bounds_min + phase * bounds_span) * (2.0 * M_PI);
                const AngleAxisd       orbit(angle, axis);
                const Vector3d         relative = PM::GetPos(p).cast<double>() - center;
                PM::MoveTo(p, center + orbit * relative);
                PM::ChangeVelocity(p, orbit * GenRandomVec3(m.speedmin, m.speedmax));
            };
        } else if (name == "turbulentvelocityrandom") {
            TurbulentRandom r;
            TurbulentRandom::ReadFromJson(wpj, r);
            Vector3f forward(r.forward.data());
            Vector3f right(r.right.data());
            return [=](Particle& p, const ParticleInitInfo& info) {
                /*
                 * Audio scales the stored phasemax-phasemin span. The phase argument is
                 * (rng01 * span + phasemin + time) * timescale. HashNoise1D turns that scalar into
                 * a 1D hashed-gradient sample; the sample is then an angle noise*pi*scale + offset
                 * that rotates authored forward around authored right. Simulation is in the
                 * particle node's local space and the node transform is applied at render, so the
                 * direction stays local. Final speed is an independent rng in [speedmin, speedmax]
                 * and is not multiplied by the audio response.
                 */
                const float response = static_cast<float>(audio_factor ? audio_factor() : 1.0);
                const float span     = (r.phasemax - r.phasemin) * response;
                const float phase_arg =
                    (Random::get(0.0f, 1.0f) * span + r.phasemin +
                     static_cast<float>(info.time)) *
                    static_cast<float>(r.timescale);
                const float noise = algorism::HashNoise1D(phase_arg);
                const float angle = noise * 3.1415927410125732f * r.scale + r.offset;
                Vector3f direction = forward;
                if (IsFiniteNonZeroVector(right.cast<double>())) {
                    direction = AngleAxisf(angle, right.normalized()) * forward;
                }
                const float speed = Random::get(r.speedmin, r.speedmax);
                direction *= speed;
                PM::ChangeVelocity(p, direction[0], direction[1], direction[2]);
            };
        }
    } while (false);
    return [](Particle&, const ParticleInitInfo&) {
    };
}

ParticleInitOp WPParticleParser::genOverrideInitOp(const wpscene::ParticleInstanceoverride& over) {
    return [=](Particle& p, const ParticleInitInfo&) {
        PM::MutiplyInitLifeTime(p, over.lifetime);
        PM::MutiplyInitAlpha(p, over.alpha);
        PM::MutiplyInitSize(p, over.size);
        PM::MutiplyVelocity(p, over.speed);
        if (over.overColor) {
            PM::InitColorOverride(p, DecodeParticleColorEndpoint(over.color));
        } else if (over.overColorn) {
            PM::InitColorOverride(
                p, Eigen::Vector3f { over.colorn[0], over.colorn[1], over.colorn[2] });
        }
    };
}
float FadeValueChange(float life, float start, float end, float startValue,
                      float endValue) noexcept {
    if (life <= start)
        return startValue;
    else if (life > end)
        return endValue;
    const float pass = (life - start) / (end - start);
    return startValue + pass * (endValue - startValue);
}

struct ValueChange {
    float starttime { 0 };
    float endtime { 1.0f };
    float startvalue { 1.0f };
    float endvalue { 0.0f };

    static auto ReadFromJson(const nlohmann::json& j) {
        ValueChange v;
        GET_JSON_NAME_VALUE_NOWARN(j, "starttime", v.starttime);
        GET_JSON_NAME_VALUE_NOWARN(j, "endtime", v.endtime);
        GET_JSON_NAME_VALUE_NOWARN(j, "startvalue", v.startvalue);
        GET_JSON_NAME_VALUE_NOWARN(j, "endvalue", v.endvalue);
        return v;
    }
};
float FadeValueChange(float life, const ValueChange& v) noexcept {
    return FadeValueChange(life, v.starttime, v.endtime, v.startvalue, v.endvalue);
}

struct VecChange {
    float                starttime { 0 };
    float                endtime { 1.0f };
    std::array<float, 3> startvalue { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> endvalue { 0.0f, 0.0f, 0.0f };

    static auto ReadFromJson(const nlohmann::json& j) {
        VecChange v;
        GET_JSON_NAME_VALUE_NOWARN(j, "starttime", v.starttime);
        GET_JSON_NAME_VALUE_NOWARN(j, "endtime", v.endtime);
        GET_JSON_NAME_VALUE_NOWARN(j, "startvalue", v.startvalue);
        GET_JSON_NAME_VALUE_NOWARN(j, "endvalue", v.endvalue);
        return v;
    }
};

struct FrequencyValue {
    std::array<float, 3> mask { 1.0f, 1.0f, 0.0f };

    float frequencymin { 0.0f };
    float frequencymax { 10.0f };
    float scalemin { 0.0f };
    float scalemax { 1.0f };
    float phasemin { 0.0f };
    float phasemax { static_cast<float>(2 * M_PI) };

    struct StorageRandom {
        bool  reset { true };
        float frequency { 0.0f };
        float scale { 1.0f };
        float phase { 0.0f };
    };

    std::vector<StorageRandom> storage;

    static auto ReadFromJson(const nlohmann::json& j, std::string_view name) {
        FrequencyValue v;
        if (name == "oscillatesize") {
            v.scalemin = 0.8f;
            v.scalemax = 1.2f;
        } else if (name == "oscillateposition") {
            v.frequencymax = 5.0f;
        }
        GET_JSON_NAME_VALUE_NOWARN(j, "frequencymin", v.frequencymin);
        GET_JSON_NAME_VALUE_NOWARN(j, "frequencymax", v.frequencymax);
        if (v.frequencymax == 0.0f) v.frequencymax = v.frequencymin;
        GET_JSON_NAME_VALUE_NOWARN(j, "scalemin", v.scalemin);
        GET_JSON_NAME_VALUE_NOWARN(j, "scalemax", v.scalemax);
        GET_JSON_NAME_VALUE_NOWARN(j, "phasemin", v.phasemin);
        GET_JSON_NAME_VALUE_NOWARN(j, "phasemax", v.phasemax);
        GET_JSON_NAME_VALUE_NOWARN(j, "mask", v.mask);
        return v;
    };
    inline void CheckAndResize(size_t s) {
        if (storage.size() < s) storage.resize(2 * s, StorageRandom {});
    }
    inline void GenFrequency(Particle& p, uint32_t index) {
        auto& st = storage.at(index);
        if (! PM::LifetimeOk(p)) st.reset = true;
        if (st.reset) {
            st.frequency = Random::get(frequencymin, frequencymax);
            st.scale     = Random::get(scalemin, scalemax);
            st.phase     = (float)Random::get((double)phasemin, phasemax + 2.0 * M_PI);
            st.reset     = false;
        }
    }
    inline double GetScale(uint32_t index, double time) {
        const auto& st = storage.at(index);
        double      f  = st.frequency / (2.0f * M_PI);
        double      w  = 2.0f * M_PI * f;
        return algorism::lerp((std::cos(w * time + st.phase) + 1.0f) * 0.5f, scalemin, scalemax);
    }
    inline double GetMove(uint32_t index, double time, double timePass) {
        const auto& st = storage.at(index);
        double      f  = st.frequency / (2.0f * M_PI);
        double      w  = 2.0f * M_PI * f;
        return -1.0f * st.scale * w * std::sin(w * time + st.phase) * timePass;
    }
};

struct Turbulence {
    // the minimum time offset of the noise field for a particle.
    float phasemin { 0 };
    // the maximum time offset of the noise field for a particle.
    float phasemax { 0 };
    // the minimum velocity applied to particles.
    float speedmin { 500.0f };
    // the maximum velocity applied to particles.
    float speedmax { 1000.0f };
    // how fast the noise field changes shape.
    float timescale { 20.0f };

    float scale { 0.01f };

    std::array<int32_t, 3> mask { 1, 1, 0 };

    static auto ReadFromJson(const nlohmann::json& j) {
        Turbulence v;
        GET_JSON_NAME_VALUE_NOWARN(j, "phasemin", v.phasemin);
        GET_JSON_NAME_VALUE_NOWARN(j, "phasemax", v.phasemax);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedmin", v.speedmin);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedmax", v.speedmax);
        GET_JSON_NAME_VALUE_NOWARN(j, "timescale", v.timescale);
        GET_JSON_NAME_VALUE_NOWARN(j, "mask", v.mask);
        GET_JSON_NAME_VALUE_NOWARN(j, "scale", v.scale);
        return v;
    };
};

struct Vortex {
    enum class FlagEnum
    {
        infinit_axis = 0,
        maintain_distance = 1,
        ring_shape = 2,
    };
    using EFlags = BitFlags<FlagEnum>;

    i32 controlpoint { 0 };

    // anything below this distance receives force multiplied with speed inner.
    float distanceinner { 500.0f };
    // anything above this distance receives force multiplied with speed outer.
    float distanceouter { 650.0f };
    // amount of force applied to inner ring.
    float speedinner { 2500.0f };
    // amount of force applied to outer ring.
    float speedouter { 0 };
    float centerforce { 0.0f };
    float ringradius { 0.0f };
    float ringwidth { 0.0f };
    float ringpulldistance { 0.0f };
    float ringpullforce { 0.0f };

    EFlags flags { 0 };

    // positional offset from the center of the control point.
    std::array<float, 3> offset { 0.0f, 0.0f, 0.0f };

    // the axis to rotate around.
    std::array<float, 3> axis { 0.0f, 0.0f, 1.0f };

    static auto ReadFromJson(const nlohmann::json& j) {
        Vortex v;
        GET_JSON_NAME_VALUE_NOWARN(j, "controlpoint", v.controlpoint);
        v.controlpoint = NormalizeControlPointIndex(v.controlpoint);

        GET_JSON_NAME_VALUE_NOWARN(j, "distanceinner", v.distanceinner);
        GET_JSON_NAME_VALUE_NOWARN(j, "distanceouter", v.distanceouter);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedinner", v.speedinner);
        GET_JSON_NAME_VALUE_NOWARN(j, "speedouter", v.speedouter);
        GET_JSON_NAME_VALUE_NOWARN(j, "centerforce", v.centerforce);
        GET_JSON_NAME_VALUE_NOWARN(j, "ringradius", v.ringradius);
        GET_JSON_NAME_VALUE_NOWARN(j, "ringwidth", v.ringwidth);
        GET_JSON_NAME_VALUE_NOWARN(j, "ringpulldistance", v.ringpulldistance);
        GET_JSON_NAME_VALUE_NOWARN(j, "ringpullforce", v.ringpullforce);

        i32 _flags { 0 };
        GET_JSON_NAME_VALUE_NOWARN(j, "flags", _flags);
        v.flags = EFlags(_flags);

        GET_JSON_NAME_VALUE_NOWARN(j, "offset", v.offset);
        GET_JSON_NAME_VALUE_NOWARN(j, "axis", v.axis);

        return v;
    };
};

struct ControlPointForce {
    i32 controlpoint { 0 };
    u32 flags { 0 };

    // how strongly the control point attracts or repels.
    float scale { 512.0f };
    // the maximum distance between particle and control point where the force takes effect.
    float threshold { 512.0f };
    // Particle-path default when the key is omitted. The attract loop uses threshold, not this
    // field.
    float deletethreshold { 15.0f };

    // positional offset from the center of the control point.
    std::array<float, 3> origin { 0.0f, 0.0f, 0.0f };

    static auto ReadFromJson(const nlohmann::json& j) {
        ControlPointForce v;
        GET_JSON_NAME_VALUE_NOWARN(j, "controlpoint", v.controlpoint);
        v.controlpoint = NormalizeControlPointIndex(v.controlpoint);

        GET_JSON_NAME_VALUE_NOWARN(j, "scale", v.scale);
        // Wallpaper Engine assets commonly serialize this field with the historical
        // "threadhold" typo, while hand-authored tests and docs often use "threshold".
        // Reading both names keeps the operator compatible with either spelling.
        GET_JSON_NAME_VALUE_NOWARN(j, "threshold", v.threshold);
        GET_JSON_NAME_VALUE_NOWARN(j, "threadhold", v.threshold);
        GET_JSON_NAME_VALUE_NOWARN(j, "deletethreshold", v.deletethreshold);
        GET_JSON_NAME_VALUE_NOWARN(j, "flags", v.flags);

        GET_JSON_NAME_VALUE_NOWARN(j, "offset", v.origin);
        return v;
    };
};

ParticleOperatorOp
WPParticleParser::genParticleOperatorOp(const nlohmann::json&                    wpj,
                                        const wpscene::ParticleInstanceoverride& over,
                                        ParticleAudioResponseFactor              audio_factor) {
    do {
        if (! wpj.contains("name")) break;
        std::string name;
        GET_JSON_NAME_VALUE(wpj, "name", name);
        if (name == "movement") {
            float drag { 0.0f };
            auto  speed = over.speed;

            std::array<float, 3> gravity { 0, 0, 0 };
            GET_JSON_NAME_VALUE_NOWARN(wpj, "drag", drag);
            GET_JSON_NAME_VALUE_NOWARN(wpj, "gravity", gravity);
            Vector3d vecG = Vector3f(gravity.data()).cast<double>();
            return [=](const ParticleInfo& info) {
                for (auto& p : info.particles) {
                    // Keep the authored gravity vector untouched so every particle system follows
                    // the scene's normal coordinate convention. Flipping Y here made generic
                    // effects such as fireworks accelerate upward after they exploded.
                    //
                    // Movement is semi-implicit: v_mid = v + g*dt, x += v_mid*dt, then
                    // v = v_mid * (1 - min(drag*dt, almost 1)) so a huge dt cannot reverse the
                    // particle and never quite reaches a hard zero.
                    PM::Accelerate(p, speed * vecG, info.time_pass);
                    PM::MoveByTime(p, info.time_pass);

                    constexpr double kDragDtMax = 0.99999988079071045;
                    const double drag_factor =
                        1.0 - std::min(static_cast<double>(drag) * info.time_pass, kDragDtMax);
                    PM::MutiplyVelocity(p, drag_factor);
                }
            };
        } else if (name == "angularmovement") {
            float                drag { 0.0f };
            std::array<float, 3> force { 0, 0, 0 };
            GET_JSON_NAME_VALUE_NOWARN(wpj, "drag", drag);
            GET_JSON_NAME_VALUE_NOWARN(wpj, "force", force);
            Vector3d vecF = Vector3f(force.data()).cast<double>();
            return [=](const ParticleInfo& info) {
                for (auto& p : info.particles) {
                    Vector3d acc =
                        algorism::DragForce(PM::GetAngular(p).cast<double>(), drag) + vecF;
                    PM::AngularAccelerate(p, acc, info.time_pass);
                    PM::RotateByTime(p, info.time_pass);
                }
            };
        } else if (name == "sizechange") {
            auto vc        = ValueChange::ReadFromJson(wpj);
            auto size_over = over.size;
            return [vc, size_over](const ParticleInfo& info) {
                for (auto& p : info.particles) {
                    PM::MutiplySize(p, size_over * FadeValueChange(PM::LifetimePos(p), vc));
                }
            };

        } else if (name == "alphafade") {
            float fadeintime { 0.5f }, fadeouttime { 0.5f };
            GET_JSON_NAME_VALUE_NOWARN(wpj, "fadeintime", fadeintime);
            GET_JSON_NAME_VALUE_NOWARN(wpj, "fadeouttime", fadeouttime);
            return [fadeintime, fadeouttime](const ParticleInfo& info) {
                for (auto& p : info.particles) {
                    auto life = PM::LifetimePos(p);
                    if (life <= fadeintime)
                        PM::MutiplyAlpha(p, FadeValueChange(life, 0, fadeintime, 0, 1.0f));
                    else if (life > fadeouttime)
                        PM::MutiplyAlpha(p,
                                         1.0f - FadeValueChange(life, fadeouttime, 1.0f, 0, 1.0f));
                }
            };
        } else if (name == "alphachange") {
            auto vc = ValueChange::ReadFromJson(wpj);
            return [vc](const ParticleInfo& info) {
                for (auto& p : info.particles) {
                    PM::MutiplyAlpha(p, FadeValueChange(PM::LifetimePos(p), vc));
                }
            };
        } else if (name == "colorchange") {
            auto vc = VecChange::ReadFromJson(wpj);
            return [vc](const ParticleInfo& info) {
                for (auto& p : info.particles) {
                    auto     life = PM::LifetimePos(p);
                    Vector3f result;
                    for (uint i = 0; i < 3; i++)
                        result[i] = FadeValueChange(
                            life, vc.starttime, vc.endtime, vc.startvalue[i], vc.endvalue[i]);
                    PM::MutiplyColor(p, result);
                }
            };
        } else if (name == "oscillatealpha") {
            FrequencyValue fv = FrequencyValue::ReadFromJson(wpj, name);
            return [fv](const ParticleInfo& info) mutable {
                fv.CheckAndResize(info.particles.size());
                for (uint i = 0; i < info.particles.size(); i++) {
                    auto& p = info.particles[i];
                    fv.GenFrequency(p, i);
                    PM::MutiplyAlpha(p, fv.GetScale(i, PM::LifetimePassed(p)));
                }
            };
        } else if (name == "oscillatesize") {
            FrequencyValue fv = FrequencyValue::ReadFromJson(wpj, name);
            return [fv](const ParticleInfo& info) mutable {
                fv.CheckAndResize(info.particles.size());
                for (uint i = 0; i < info.particles.size(); i++) {
                    auto& p = info.particles[i];
                    fv.GenFrequency(p, i);
                    PM::MutiplySize(p, fv.GetScale(i, PM::LifetimePassed(p)));
                }
            };

        } else if (name == "oscillateposition") {
            std::vector<Vector3f>         lastMove;
            FrequencyValue                fvx = FrequencyValue::ReadFromJson(wpj, name);
            std::array<FrequencyValue, 3> fxp = { fvx, fvx, fvx };
            return [=](const ParticleInfo& info) mutable {
                for (auto& f : fxp) f.CheckAndResize(info.particles.size());
                for (uint i = 0; i < info.particles.size(); i++) {
                    auto&    p = info.particles[i];
                    Vector3d del { Vector3d::Zero() };
                    auto     time = PM::LifetimePassed(p);
                    for (uint d = 0; d < 3; d++) {
                        if (fxp[0].mask[d] < 0.01) continue;
                        fxp[d].GenFrequency(p, i);
                        del[d] = fxp[d].GetMove(i, time, info.time_pass);
                    }

                    PM::Move(p, del);
                }
            };
        } else if (name == "turbulence") {
            Turbulence tur = Turbulence::ReadFromJson(wpj);
            return [=](const ParticleInfo& info) {
                const double audio_scale = audio_factor ? audio_factor() : 1.0;
                if (! std::isfinite(audio_scale) || audio_scale == 0.0) return;

                const float  audio_scale_f = static_cast<float>(audio_scale);
                const float  time_phase    = static_cast<float>(tur.timescale * info.time);
                const float  scale_f       = tur.scale;
                const float  dt_f          = static_cast<float>(info.time_pass);

                ParticleParallelExecutor::Instance().ParallelFor(
                    info.particles.size(), [&](size_t begin, size_t end) {
                    const auto apply_curl = [&](Particle& p, Vector3f curl, float speed) {
                        const float curl_norm = curl.norm();
                        if (! std::isfinite(curl_norm) || curl_norm <= 0.0f) return;
                        Vector3f result = curl * (speed / curl_norm);
                        for (usize i = 0; i < 3; i++) {
                            if (tur.mask[i] == 0) result[i] = 0;
                        }
                        PM::Accelerate(p, result * audio_scale_f, dt_f);
                    };

                    const auto phase_and_speed = [&](const Particle& particle) {
                        const float random = StableParticleRandom01(particle.spawnSequence);
                        return std::array<float, 2> {
                            static_cast<float>(
                                algorism::lerp(random, tur.phasemin, tur.phasemax)) +
                                time_phase,
                            static_cast<float>(
                                algorism::lerp(random, tur.speedmin, tur.speedmax)),
                        };
                    };

                    const auto apply_batch = [&](const std::array<size_t, 8>& particle_indices) {
                        float px[8], py[8], pz[8], cx[8], cy[8], cz[8];
                        float speeds[8];
                        for (int lane = 0; lane < 8; lane++) {
                            const auto& particle = info.particles[particle_indices[lane]];
                            const auto [phase, speed] = phase_and_speed(particle);
                            speeds[lane] = speed;

                            Vector3f pos = PM::GetPos(particle);
                            pos.array() += phase;
                            pos *= scale_f;
                            px[lane] = pos.x();
                            py[lane] = pos.y();
                            pz[lane] = pos.z();
                        }
                        algorism::CurlNoise8(px, py, pz, cx, cy, cz);
                        for (int lane = 0; lane < 8; lane++) {
                            apply_curl(info.particles[particle_indices[lane]],
                                       Vector3f { cx[lane], cy[lane], cz[lane] },
                                       speeds[lane]);
                        }
                    };

                    // Particle storage keeps reusable dead slots. Compact only live indices into
                    // SIMD batches so sparse systems do not evaluate stale slots merely to discard
                    // their result after the noise calculation.
                    std::array<size_t, 8> live_indices;
                    size_t live_lane_count = 0;
                    for (size_t particle_index = begin; particle_index < end; particle_index++) {
                        if (! PM::LifetimeOk(info.particles[particle_index])) continue;
                        live_indices[live_lane_count++] = particle_index;
                        if (live_lane_count == live_indices.size()) {
                            apply_batch(live_indices);
                            live_lane_count = 0;
                        }
                    }

                    for (size_t lane = 0; lane < live_lane_count; lane++) {
                        auto& p = info.particles[live_indices[lane]];
                        const auto [phase, speed] = phase_and_speed(p);
                        Vector3f pos = PM::GetPos(p);
                        pos.array() += phase;
                        apply_curl(p, algorism::CurlNoise(pos * scale_f), speed);
                    }
                }, 1);
            };
        } else if (name == "vortex" || name == "vortex_v2") {
            Vortex v = Vortex::ReadFromJson(wpj);
            const bool extended = name == "vortex_v2";
            return [=](const ParticleInfo& info) {
                const double audio_scale = audio_factor ? audio_factor() : 1.0;
                if (! std::isfinite(audio_scale) ||
                    static_cast<usize>(v.controlpoint) >= info.controlpoints.size()) {
                    return;
                }

                Vector3d offset = info.controlpoints[v.controlpoint].offset +
                                  (Vector3f { v.offset.data() }).cast<double>();
                Vector3d axis = NormalizeOr((Vector3f { v.axis.data() }).cast<double>(),
                                            Vector3d::UnitZ());
                const bool infinite_axis = v.flags[Vortex::FlagEnum::infinit_axis];
                const bool maintain_distance =
                    extended && v.flags[Vortex::FlagEnum::maintain_distance];
                const bool ring_shape = extended && v.flags[Vortex::FlagEnum::ring_shape];
                const double distance_span = static_cast<double>(v.distanceouter) -
                                             static_cast<double>(v.distanceinner);
                const double min_distance_squared =
                    kControlPointForceMinDistance * kControlPointForceMinDistance;

                for (auto& p : info.particles) {
                    if (! PM::LifetimeOk(p)) continue;
                    const Vector3d relative = p.position.cast<double>() - offset;
                    const Vector3d radial   = relative - axis * relative.dot(axis);
                    const double radial_distance = radial.norm();
                    const double distance = infinite_axis ? radial_distance : relative.norm();
                    if (distance <= kControlPointForceMinDistance ||
                        radial_distance <= kControlPointForceMinDistance) {
                        continue;
                    }

                    const Vector3d tangent         = relative.cross(axis);
                    const double   tangent_squared = tangent.squaredNorm();
                    if (tangent_squared <= min_distance_squared) continue;
                    const Vector3d direct = tangent / std::sqrt(tangent_squared);

                    Vector3d acceleration = Vector3d::Zero();
                    if (ring_shape) {
                        const double half_width =
                            std::max(0.0, static_cast<double>(v.ringwidth) * 0.5);
                        const double ring_delta =
                            radial_distance - static_cast<double>(v.ringradius);
                        const double distance_to_ring = std::abs(ring_delta);

                        if (distance_to_ring <= half_width) {
                            double t = half_width > kControlPointForceMinDistance
                                ? (ring_delta + half_width) / (half_width * 2.0)
                                : 0.0;
                            t = std::clamp(t, 0.0, 1.0);
                            acceleration +=
                                direct * algorism::lerp(t, v.speedinner, v.speedouter);
                        } else if (v.ringpulldistance > 0.0f &&
                                   distance_to_ring <=
                                       half_width + static_cast<double>(v.ringpulldistance)) {
                            const double pull =
                                1.0 - (distance_to_ring - half_width) /
                                          static_cast<double>(v.ringpulldistance);
                            const Vector3d radial_direction = radial / radial_distance;
                            acceleration +=
                                (ring_delta > 0.0 ? -radial_direction : radial_direction) *
                                static_cast<double>(v.ringpullforce) * pull;
                            acceleration += direct * static_cast<double>(v.speedouter) * pull;
                        }
                    } else {
                        double t = 0.0;
                        if (std::abs(distance_span) > kControlPointForceMinDistance) {
                            t = (distance - static_cast<double>(v.distanceinner)) / distance_span;
                        }
                        t = std::clamp(t, 0.0, 1.0);
                        acceleration +=
                            direct * algorism::lerp(t, v.speedinner, v.speedouter);
                    }

                    if (maintain_distance && IsFiniteNonZeroVector(relative)) {
                        acceleration -=
                            relative.normalized() * static_cast<double>(v.centerforce);
                    }

                    PM::Accelerate(p, acceleration * audio_scale, info.time_pass);
                }
            };
        } else if (name == "controlpointattract") {
            ControlPointForce c = ControlPointForce::ReadFromJson(wpj);
            return [=](const ParticleInfo& info) {
                if (c.controlpoint < 0 || static_cast<usize>(c.controlpoint) >= info.controlpoints.size())
                    return;

                Vector3d offset = info.controlpoints[c.controlpoint].offset +
                                  Vector3f { c.origin.data() }.cast<double>();
                /*
                 * Attract within the authored threshold with linear falloff
                 * (1 - distance/threshold) * scale. Flags bit 1 replaces that impulse with the
                 * remaining distance when the step would overshoot.
                 */
                const double threshold = static_cast<double>(c.threshold);
                if (!std::isfinite(threshold) || threshold <= 0.0 || !std::isfinite(c.scale))
                    return;

                const bool delete_in_center = (c.flags & 0x1) != 0;
                const bool limit_overshoot = (c.flags & 0x2) != 0;
                for (auto& p : info.particles) {
                    if (! PM::LifetimeOk(p)) continue;
                    const Vector3d diff     = offset - PM::GetPos(p).cast<double>();
                    const double   distance = diff.norm();

                    if (delete_in_center && std::isfinite(c.deletethreshold) &&
                        c.deletethreshold >= 0.0f &&
                        distance < static_cast<double>(c.deletethreshold)) {
                        PM::Delete(p);
                        continue;
                    }
                    if (distance >= threshold || distance <= kControlPointForceMinDistance)
                        continue;

                    Vector3d direction = diff / distance;
                    double   accel = (1.0 - distance / threshold) * static_cast<double>(c.scale);
                    if (limit_overshoot && info.time_pass > 0.0) {
                        const double step = accel * info.time_pass;
                        if (distance < step)
                            accel = distance / info.time_pass;
                    }
                    PM::Accelerate(p, direction * accel, info.time_pass);
                }
            };
        }
    } while (false);
    LOG_ERROR("unsupported particle operator '%s'",
              wpj.value("name", std::string("(missing)")).c_str());
    return [](const ParticleInfo&) {
    };
}

ParticleEmittOp WPParticleParser::genParticleEmittOp(const wpscene::Emitter& wpe,
                                                     ParticleAudioResponseFactor  audio_rate_factor) {
    if (wpe.name == "boxrandom") {
        ParticleBoxEmitterArgs box;
        box.minDistance   = wpe.distancemin;
        box.maxDistance   = wpe.distancemax;
        box.directions    = wpe.directions;
        box.orgin         = wpe.origin;
        box.controlpoint  = wpe.controlpoint;
        box.minSpeed      = wpe.speedmin;
        box.maxSpeed      = wpe.speedmax;
        box.timing        = MakeEmitterTiming(wpe, std::move(audio_rate_factor));
        return ParticleBoxEmitterArgs::MakeEmittOp(box);
    } else if (wpe.name == "sphererandom") {
        ParticleSphereEmitterArgs sphere;
        sphere.minDistance   = wpe.distancemin[0];
        sphere.maxDistance   = wpe.distancemax[0];
        sphere.directions    = wpe.directions;
        sphere.orgin         = wpe.origin;
        sphere.controlpoint  = wpe.controlpoint;
        sphere.sign          = wpe.sign;
        sphere.minSpeed      = wpe.speedmin;
        sphere.maxSpeed      = wpe.speedmax;
        sphere.timing        = MakeEmitterTiming(wpe, std::move(audio_rate_factor));
        return ParticleSphereEmitterArgs::MakeEmittOp(sphere);
    } else
        return [](std::vector<Particle>&, std::vector<ParticleInitOp>&,
                  std::span<const ParticleControlpoint>, uint32_t, double, double, uint64_t&) {
        };
}

#include "ParticleEmitter.h"
#include "ParticleModify.h"
#include "Utils/Algorism.h"
#include "Core/Random.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <random>
#include <tuple>

using namespace wallpaper;

namespace
{

inline std::tuple<u32, bool> FindDeadParticle(std::span<const Particle> particles, u32 start) {
    for (u32 i = start; i < particles.size(); i++) {
        if (! ParticleModify::LifetimeOk(particles[i])) return { i, true };
    }
    return { 0, false };
}

inline u32 GetEmitNum(double& timer, float speed) {
    double emitDur = 1.0f / speed;
    if (emitDur > timer) return 0;
    u32 num = timer / emitDur;
    while (emitDur < timer) timer -= emitDur;
    if (timer < 0) timer = 0;
    return num;
}

template<typename SpawnOp>
void EmitParticles(std::vector<Particle>& particles, u32 num, u32 maxcount,
                   uint64_t& next_spawn_sequence, SpawnOp&& spawn_particle) {
    u32  next_search_index = 0;
    bool has_dead          = true;

    for (u32 i = 0; i < num; i++) {
        if (has_dead) {
            auto [dead_index, found] = FindDeadParticle(particles, next_search_index);
            next_search_index        = dead_index;
            has_dead                 = found;
        }
        if (! has_dead && maxcount == particles.size()) break;

        Particle spawned      = spawn_particle();
        spawned.spawnSequence = next_spawn_sequence++;
        if (has_dead) {
            particles[next_search_index] = std::move(spawned);
        } else {
            particles.push_back(std::move(spawned));
        }
    }
}

template<typename GenerateOp>
Particle SpawnParticle(GenerateOp&& generate, std::vector<ParticleInitOp>& initializers,
                       const ParticleInitInfo& info) {
    auto particle = generate();
    // Cherry_Blossoms_2.json relies on all initializers for one spawned particle seeing the same
    // control-point snapshot and sequence slot; otherwise the five cursor petals drift into noise.
    for (auto& initializer : initializers) initializer(particle, info);
    return particle;
}

inline void ApplySign(Eigen::Vector3d& p, int32_t x, int32_t y, int32_t z) noexcept {
    if (x != 0) {
        p.x() = std::abs(p.x()) * (float)x;
    }
    if (y != 0) {
        p.y() = std::abs(p.y()) * (float)y;
    }
    if (z != 0) {
        p.z() = std::abs(p.z()) * (float)z;
    }
}
} // namespace

ParticleEmittOp ParticleBoxEmitterArgs::MakeEmittOp(ParticleBoxEmitterArgs a) {
    double timer { 0.0f };
    // Keep a per-emitter sequence counter so mapsequencearoundcontrolpoint repeats the authored
    // five slots independently of reusable particle storage indices.
    uint64_t sequence { 0 };
    return [a, timer, sequence](std::vector<Particle>&       ps,
                                std::vector<ParticleInitOp>& inis,
                                std::span<const ParticleControlpoint> controlpoints,
                                u32                          maxcount,
                                double                       timepass,
                                uint64_t&                    next_spawn_sequence) mutable {
        timer += timepass;
        auto GenBox = [&]() {
            Eigen::Vector3d pos;
            for (int32_t i = 0; i < 3; i++)
                pos[i] = algorism::lerp(Random::get(-1.0, 1.0), a.minDistance[i], a.maxDistance[i]);
            auto p = Particle();
            pos    = pos.cwiseProduct(Eigen::Vector3f { a.directions.data() }.cast<double>());
            ParticleModify::MoveTo(p, pos);
            ParticleModify::ChangeVelocity(p,
                                           Random::get(a.minSpeed, a.maxSpeed) * pos.normalized());

            Eigen::Vector3d origin = Eigen::Vector3f { a.orgin.data() }.cast<double>();
            if (a.controlpoint >= 0 && (usize)a.controlpoint < controlpoints.size()) {
                origin += controlpoints[(usize)a.controlpoint].offset;
            }
            ParticleModify::Move(p, origin);
            return p;
        };
        u32 emit_num = GetEmitNum(timer, a.emitSpeed);
        emit_num     = a.one_per_frame ? 1 : emit_num;
        emit_num     = a.instantaneous > 0 && ps.empty() ? a.instantaneous : emit_num;
        EmitParticles(ps, emit_num, maxcount, next_spawn_sequence, [&]() {
            ParticleInitInfo init_info;
            init_info.duration      = 1.0f / a.emitSpeed;
            init_info.controlpoints = controlpoints;
            init_info.sequence      = sequence++;
            return SpawnParticle(GenBox, inis, init_info);
        });
    };
}

ParticleEmittOp ParticleSphereEmitterArgs::MakeEmittOp(ParticleSphereEmitterArgs a) {
    using namespace Eigen;
    double timer { 0.0f };
    // Cherry_Blossoms_2.json uses this sphere emitter. The sequence counter is intentionally tied to
    // the emitter instance, not to particle array indices, so each burst stays in the fixed 5-point
    // order expected by mapsequencearoundcontrolpoint.
    uint64_t sequence { 0 };
    return [a, timer, sequence](std::vector<Particle>&       ps,
                                std::vector<ParticleInitOp>& inis,
                                std::span<const ParticleControlpoint> controlpoints,
                                u32                          maxcount,
                                double                       timepass,
                                uint64_t&                    next_spawn_sequence) mutable {
        timer += timepass;
        auto GenSphere = [&]() {
            auto   p = Particle();
            double r = algorism::lerp(
                std::pow(Random::get(0.0, 1.0), 1.0 / 3.0), a.minDistance, a.maxDistance);
            Eigen::Vector3d sp = r * algorism::GenSphereSurfaceNormal(
                                         [](double u, double o) {
                                             return Random::get<std::normal_distribution<>>(u, o);
                                         },
                                         Eigen::Vector3f { a.directions.data() }.cast<double>());
            ApplySign(sp, a.sign[0], a.sign[1], a.sign[2]);

            ParticleModify::MoveTo(p, sp);
            ParticleModify::ChangeVelocity(p,
                                           Random::get(a.minSpeed, a.maxSpeed) * sp.normalized());

            Eigen::Vector3d origin = Eigen::Vector3f { a.orgin.data() }.cast<double>();
            if (a.controlpoint >= 0 && (usize)a.controlpoint < controlpoints.size()) {
                origin += controlpoints[(usize)a.controlpoint].offset;
            }
            ParticleModify::Move(p, origin);
            return p;
        };
        u32 emit_num = GetEmitNum(timer, a.emitSpeed);
        emit_num     = a.one_per_frame ? 1 : emit_num;
        emit_num     = a.instantaneous > 0 && ps.empty() ? a.instantaneous : emit_num;
        EmitParticles(ps, emit_num, maxcount, next_spawn_sequence, [&]() {
            ParticleInitInfo init_info;
            init_info.duration      = 1.0f / a.emitSpeed;
            init_info.controlpoints = controlpoints;
            init_info.sequence      = sequence++;
            return SpawnParticle(GenSphere, inis, init_info);
        });
    };
}

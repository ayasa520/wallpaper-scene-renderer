#pragma once

#include <Eigen/Core>
#include <cstdint>

namespace wallpaper
{

struct Particle {
    struct InitValue {
        Eigen::Vector3f color { 1.0f, 1.0f, 1.0f };
        float           alpha { 1.0f };
        float           size { 20 };
        float           lifetime { 1.0f };
    };
    struct AuthoredColorRandom {
        Eigen::Vector3f minimum { 1.0f, 1.0f, 1.0f };
        Eigen::Vector3f maximum { 1.0f, 1.0f, 1.0f };
        Eigen::Vector3f referenceHsv { 0.0f, 0.0f, 1.0f };
        float           interpolation { 0.0f };
        bool            active { false };
    };
    Eigen::Vector3f position { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f color { 1.0f, 1.0f, 1.0f };
    // Wallpaper Engine shifts both authored colorrandom endpoints in HSV space before performing
    // the particle's RGB interpolation. Preserve that source range, its reference HSV value, and
    // the per-spawn interpolation factor so initial parsing and runtime property edits are identical.
    AuthoredColorRandom authoredColorRandom {};
    float           alpha { 1.0f };
    float           size { 20 };
    float           lifetime { 1.0f };

    Eigen::Vector3f rotation { 0.0f, 0.0f, 0.0f }; // radian  z x y
    Eigen::Vector3f velocity { 0.0f, 0.0f, 0.0f };
    // Optional shader-facing trail axis. When unset, spritetrail uses the physics velocity.
    Eigen::Vector3f renderVelocity { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f acceleration { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f angularVelocity { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f angularAcceleration { 0.0f, 0.0f, 0.0f };

    bool      mark_new { true };
    bool      hasRenderVelocity { false };
    bool      operatorDeleted { false };
    // Slot indices are reusable and therefore cannot define rope connectivity. Every spawn gets a
    // subsystem-wide monotonic sequence so the current rope chain can be reconstructed without
    // reordering ParticleInstance::m_particles (which ropetrail history keeps parallel to slots).
    uint64_t  spawnSequence { 0 };
    // First spawn unit sample in [0, 1], stored as IEEE bits. Remapvalue simplex/fbm mixes
    // these bits into the lattice hash so each particle keeps a stable noise identity.
    uint32_t  remap_seed { 0 };
    InitValue init {};
};
} // namespace wallpaper

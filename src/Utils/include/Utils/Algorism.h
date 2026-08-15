#pragma once
#include <functional>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>

#include "Core/Literals.hpp"

namespace wallpaper
{
namespace algorism
{
double CalculatePersperctiveDistance(double fov, double height) noexcept;
double CalculatePersperctiveFov(double distence, double height) noexcept;

constexpr u32 PowOfTwo(u32 x) {
    u32 pow2 { 8 };
    while (pow2 < x) pow2 *= 2;
    return pow2;
}

constexpr bool IsPowOfTwo(u32 x) { return (x > 1) && ((x & (x - 1)) == 0); }

inline Eigen::Vector3d sph2cart(const Eigen::Vector3d& sph) noexcept {
    double azimuth   = sph.x();
    double elevation = sph.y();
    double radius    = sph.z();
    return radius * Eigen::Vector3d {
        std::cos(azimuth) * std::cos(elevation),
        std::sin(azimuth) * std::cos(elevation),
        std::sin(elevation),
    };
}

template<typename TFUNC>
Eigen::Vector3d GenSphereSurface(TFUNC&& random) noexcept {
    double azimuth = 2.0 * EIGEN_PI * random();
    // not uniform distribution
    double elevation = std::asin(2.0 * random() - 1.0);
    return sph2cart({
        azimuth,
        elevation,
        1.0,
    });
}

template<typename TFUNC>
Eigen::Vector3d GenSphereSurfaceNormal(TFUNC&&                normal_random,
                                       const Eigen::Vector3d& direct) noexcept {
    double u    = direct.x() > 0.0 ? normal_random(0.0, direct.x()) : 0.0;
    double v    = direct.y() > 0.0 ? normal_random(0.0, direct.y()) : 0.0;
    double w    = direct.z() > 0.0 ? normal_random(0.0, direct.z()) : 0.0;
    double norm = std::sqrt((u * u + v * v + w * w));
    return Eigen::Vector3d(u, v, w) / norm;
}

template<typename TFUNC>
Eigen::Vector3d GenSphereIn(TFUNC&& random) noexcept {
    // not uniform distribution
    return std::pow(random(), 1.0 / 3.0) * GenSphereSurface(random);
}

constexpr double DragForce(double speed, double strength, double density) {
    // return -0.5 * speed*speed * strength * density;
    return -2.0 * speed * strength * density;
}
inline Eigen::Vector3d DragForce(Eigen::Vector3d v, double strength,
                                 double density = 1.0) noexcept {
    return v.normalized() * DragForce(v.norm(), strength, density);
}

constexpr double lerp(double t, double a, double b) noexcept { return a + t * (b - a); }

constexpr double PerlinEase(double t) noexcept { return t * t * t * (t * (t * 6 - 15) + 10); };
double           PerlinNoise(double x, double y, double z) noexcept;
Eigen::Vector3d  PerlinNoiseGradient(double x, double y, double z) noexcept;
Eigen::Vector3f  PerlinNoiseGradient(float x, float y, float z) noexcept;
// Eight analytic gradients at once. Same lattice, fade, and 16-entry table as the scalar
// float overload. Supported x86 CPUs use AVX2/FMA after a runtime feature check; every other
// target uses the scalar implementation.
void             PerlinNoiseGradient8(const float x[8], const float y[8], const float z[8],
                                      float gx[8], float gy[8], float gz[8]) noexcept;
void             CurlNoise8(const float px[8], const float py[8], const float pz[8],
                            float cx[8], float cy[8], float cz[8]) noexcept;
// 1D hashed-gradient interpolant used by turbulentvelocityrandom, not a 3D curl sample.
float            HashNoise1D(float x) noexcept;

// curl need a vec
inline Eigen::Vector3d PerlinNoiseVec3(Eigen::Vector3d p) noexcept {
    return Eigen::Vector3d { PerlinNoise(p[0], p[1], p[2]),
                             PerlinNoise(p[0] + 89.2, p[1] + 33.1, p[2] + 57.3),
                             PerlinNoise(p[0] + 100.3, p[1] + 120.1, p[2] + 142.2) };
}
inline Eigen::Vector3f CurlNoise(Eigen::Vector3f p) noexcept {
    /*
     * The three vector-field components use the same fixed offsets as PerlinNoiseVec3. Analytic
     * gradients replace the previous six-point central difference (18 PerlinNoise evaluations);
     * the turbulence particle operator calls this once per live particle per frame, which made
     * PerlinNoise the single largest CPU consumer on emitter-heavy scenes. The hot path stays in
     * float because particle positions and velocities are already stored as float3.
     */
    const Eigen::Vector3f gx = PerlinNoiseGradient(p[0], p[1], p[2]);
    const Eigen::Vector3f gy = PerlinNoiseGradient(p[0] + 89.2f, p[1] + 33.1f, p[2] + 57.3f);
    const Eigen::Vector3f gz = PerlinNoiseGradient(p[0] + 100.3f, p[1] + 120.1f, p[2] + 142.2f);
    // curl = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
    return { gz.y() - gy.z(), gx.z() - gz.x(), gy.x() - gx.y() };
}

} // namespace algorism
} // namespace wallpaper

#pragma once

#include "Particle.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace wallpaper
{

namespace ParticleModify
{

inline void Move(Particle& p, const Eigen::Vector3d& acc) noexcept {
    p.position = (p.position.cast<double>() + acc).cast<float>();
}
inline void Move(Particle& p, double x, double y, double z) noexcept { Move(p, { x, y, z }); }

inline void MoveTo(Particle& p, const Eigen::Vector3d& pos) noexcept {
    p.position = pos.cast<float>();
}
inline void MoveTo(Particle& p, double x, double y, double z) noexcept { MoveTo(p, { x, y, z }); }

inline void MoveToNegZ(Particle& p) noexcept { p.position.z() = -std::abs(p.position.z()); }

inline void MoveByTime(Particle& p, double t) noexcept { Move(p, p.velocity.cast<double>() * t); }

inline void MoveMultiply(Particle& p, const Eigen::Vector3d& para) noexcept {
    p.position = para.cwiseProduct(p.position.cast<double>()).cast<float>();
}
inline void MoveMultiply(Particle& p, double x, double y, double z) noexcept {
    MoveMultiply(p, { x, y, z });
}

inline void MoveApplySign(Particle& p, int32_t x, int32_t y, int32_t z) noexcept {
    if (x != 0) {
        p.position[0] = std::abs(p.position[0]) * (float)x;
    }
    if (y != 0) {
        p.position[1] = std::abs(p.position[1]) * (float)y;
    }
    if (z != 0) {
        p.position[2] = std::abs(p.position[2]) * (float)z;
    }
}
inline void SphereDirectOffset(Particle& p, const Eigen::Vector3d& base, double direct) noexcept {
    using namespace Eigen;
    Vector3d axis  = base.cross(p.position.cast<double>()).normalized();
    Affine3d trans = Affine3d::Identity();
    trans.prerotate(AngleAxis<double>(direct, axis));
    p.position = (trans * p.position.cast<double>()).cast<float>();
}

inline void RotatePos(Particle& p, double x, double y, double z) noexcept {
    using namespace Eigen;
    Affine3d trans = Affine3d::Identity();

    trans.prerotate(AngleAxis<double>(y, Vector3d::UnitY()));  // y
    trans.prerotate(AngleAxis<double>(x, Vector3d::UnitX()));  // x
    trans.prerotate(AngleAxis<double>(-z, Vector3d::UnitZ())); // z
    p.position = (trans * p.position.cast<double>()).cast<float>();
}

inline void ChangeLifetime(Particle& p, double l) noexcept { p.lifetime += l; }
inline void Delete(Particle& p) noexcept {
    p.lifetime        = 0.0f;
    p.operatorDeleted = true;
}

inline float LifetimePos(const Particle& p) {
    if (p.lifetime < 0) return 1.0f;
    return 1.0f - (p.lifetime / p.init.lifetime);
}

inline double LifetimePassed(const Particle& p) noexcept { return p.init.lifetime - p.lifetime; }

inline bool LifetimeOk(const Particle& p) noexcept { return p.lifetime > 0.0f; }

void ChangeRotation(Particle&, float x, float y, float z);

inline void ChangeColor(Particle& p, const Eigen::Vector3d& c) noexcept {
    p.color = (p.color.cast<double>() + c).cast<float>();
}
inline void ChangeColor(Particle& p, double r, double g, double b) { ChangeColor(p, { r, g, b }); }

inline void ChangeRotation(Particle& p, const Eigen::Vector3d& r) noexcept {
    p.rotation = (p.rotation.cast<double>() + r).cast<float>();
}
inline void ChangeRotation(Particle& p, double x, double y, double z) {
    ChangeRotation(p, { x, y, z });
}

inline void ChangeVelocity(Particle& p, const Eigen::Vector3d& v) noexcept {
    p.velocity = (p.velocity.cast<double>() + v).cast<float>();
}
inline void ChangeVelocity(Particle& p, double x, double y, double z) noexcept {
    ChangeVelocity(p, { x, y, z });
}
inline void Accelerate(Particle& p, const Eigen::Vector3d& acc, double t) noexcept {
    ChangeVelocity(p, acc * t);
}
inline void Accelerate(Particle& p, const Eigen::Vector3f& acc, float t) noexcept {
    p.velocity += acc * t;
}

inline void ChangeAngularVelocity(Particle& p, const Eigen::Vector3d& v) noexcept {
    p.angularVelocity = (p.angularVelocity.cast<double>() + v).cast<float>();
}
inline void ChangeAngularVelocity(Particle& p, double x, double y, double z) noexcept {
    ChangeAngularVelocity(p, { x, y, z });
}
inline void AngularAccelerate(Particle& p, const Eigen::Vector3d& acc, double t) noexcept {
    ChangeAngularVelocity(p, acc * t);
}

inline void Rotate(Particle& p, const Eigen::Vector3d& r) noexcept {
    p.rotation = (p.rotation.cast<double>() + r).cast<float>();
}
inline void Rotate(Particle& p, double x, double y, double z) noexcept { Rotate(p, { x, y, z }); }

inline void RotateByTime(Particle& p, double t) noexcept {
    Rotate(p, p.angularVelocity.cast<double>() * t);
}

inline void MutiplyAlpha(Particle& p, double a) { p.alpha *= static_cast<float>(a); }
inline void MutiplyAlpha(Particle& p, float a) { p.alpha *= a; }
inline void MutiplySize(Particle& p, double s) { p.size *= static_cast<float>(s); }
inline void MutiplySize(Particle& p, float s) { p.size *= s; }

inline void MutiplyColor(Particle& p, const Eigen::Vector3d& c) {
    p.color = c.cwiseProduct(p.color.cast<double>()).cast<float>();
}
inline void MutiplyColor(Particle& p, double r, double g, double b) {
    MutiplyColor(p, { r, g, b });
}
inline void MutiplyColor(Particle& p, const Eigen::Vector3f& c) {
    p.color.x() *= c.x();
    p.color.y() *= c.y();
    p.color.z() *= c.z();
}
inline void MutiplyColor(Particle& p, float r, float g, float b) {
    p.color.x() *= r;
    p.color.y() *= g;
    p.color.z() *= b;
}
inline void MutiplyVelocity(Particle& p, double m) { p.velocity *= m; }

inline void ChangeSize(Particle& p, double s) { p.size += s; }
inline void ChangeAlpha(Particle& p, double a) { p.alpha += a; }

inline void InitLifetime(Particle& p, float l) noexcept {
    p.lifetime      = l;
    p.init.lifetime = l;
}
inline void InitSize(Particle& p, double s) {
    p.size      = s;
    p.init.size = s;
}
inline void InitAlpha(Particle& p, double a) {
    p.alpha      = a;
    p.init.alpha = a;
}
inline Eigen::Vector3f ParticleRgbToHsv(const Eigen::Vector3f& rgb) {
    constexpr float kColorDeltaEpsilon = 0.00001f;

    const float maximum = rgb.maxCoeff();
    const float minimum = rgb.minCoeff();
    const float delta   = maximum - minimum;

    Eigen::Vector3f hsv { 0.0f, 0.0f, maximum };
    if (delta < kColorDeltaEpsilon || maximum <= 0.0f) return hsv;

    hsv.y() = delta / maximum;
    if (rgb.x() >= maximum) {
        hsv.x() = (rgb.y() - rgb.z()) / delta;
    } else if (rgb.y() >= maximum) {
        hsv.x() = ((rgb.z() - rgb.x()) / delta) + 2.0f;
    } else {
        hsv.x() = ((rgb.x() - rgb.y()) / delta) + 4.0f;
    }
    hsv.x() /= 6.0f;
    if (hsv.x() < 0.0f) hsv.x() += 1.0f;
    return hsv;
}

inline Eigen::Vector3f ParticleHsvToRgb(const Eigen::Vector3f& hsv) {
    const float hue_sector = hsv.x() * 6.0f;
    const float chroma     = hsv.z() * hsv.y();
    const float secondary  = chroma * (1.0f - std::abs(std::fmod(hue_sector, 2.0f) - 1.0f));
    const float match      = hsv.z() - chroma;

    Eigen::Vector3f rgb;
    if (hue_sector < 1.0f) {
        rgb = { chroma, secondary, 0.0f };
    } else if (hue_sector < 2.0f) {
        rgb = { secondary, chroma, 0.0f };
    } else if (hue_sector < 3.0f) {
        rgb = { 0.0f, chroma, secondary };
    } else if (hue_sector < 4.0f) {
        rgb = { 0.0f, secondary, chroma };
    } else if (hue_sector < 5.0f) {
        rgb = { secondary, 0.0f, chroma };
    } else {
        rgb = { chroma, 0.0f, secondary };
    }
    return rgb.array() + match;
}

inline Eigen::Vector3f ShiftAuthoredParticleColor(const Eigen::Vector3f& authored,
                                                   const Eigen::Vector3f& hsv_delta) {
    Eigen::Vector3f hsv = ParticleRgbToHsv(authored);
    hsv.x() += hsv_delta.x();
    hsv.x() -= std::floor(hsv.x());
    hsv.y() = std::clamp(hsv.y() + hsv_delta.y(), 0.0f, 1.0f);
    hsv.z() = std::clamp(hsv.z() + hsv_delta.z(), 0.0f, 1.0f);
    return ParticleHsvToRgb(hsv);
}

inline Eigen::Vector3f ResolveParticleColorOverride(const Particle& p,
                                                    const Eigen::Vector3f& color) {
    if (!p.authoredColorRandom.active) return color;

    // The particle compiler derives its reference color from the RGB midpoint of the authored
    // colorrandom range. The current layer color and that reference are converted to HSV; their
    // delta is added to each endpoint independently, with hue wrapping and saturation/value
    // clamping. Wallpaper Engine then performs the particle's random interpolation in RGB space.
    const Eigen::Vector3f hsv_delta =
        ParticleRgbToHsv(color) - p.authoredColorRandom.referenceHsv;
    const Eigen::Vector3f minimum = ShiftAuthoredParticleColor(
        p.authoredColorRandom.minimum, hsv_delta);
    const Eigen::Vector3f maximum = ShiftAuthoredParticleColor(
        p.authoredColorRandom.maximum, hsv_delta);
    return minimum + (maximum - minimum) * p.authoredColorRandom.interpolation;
}

inline void InitColorOverride(Particle& p, const Eigen::Vector3f& color) {
    const Eigen::Vector3f resolved = ResolveParticleColorOverride(p, color);
    p.color                         = resolved;
    p.init.color                    = resolved;
}

inline void InitVelocity(Particle& p, const Eigen::Vector3d& v) { p.velocity = v.cast<float>(); }
inline void InitVelocity(Particle& p, double x, double y, double z) {
    InitVelocity(p, { x, y, z });
}

inline void InitRenderVelocity(Particle& p, const Eigen::Vector3d& v) {
    p.renderVelocity    = v.cast<float>();
    p.hasRenderVelocity = true;
}
inline void InitRenderVelocity(Particle& p, double x, double y, double z) {
    InitRenderVelocity(p, { x, y, z });
}

inline void MutiplyInitLifeTime(Particle& p, double m) {
    p.lifetime *= m;
    p.init.lifetime = p.lifetime;
}
inline void MutiplyInitAlpha(Particle& p, double m) {
    p.alpha *= m;
    p.init.alpha = p.alpha;
}
inline void MutiplyInitSize(Particle& p, double m) {
    p.size *= m;
    p.init.size = p.size;
}
inline void InitColorRandom(Particle& p, const Eigen::Vector3f& minimum,
                            const Eigen::Vector3f& maximum, float interpolation) {
    p.authoredColorRandom.minimum       = minimum;
    p.authoredColorRandom.maximum       = maximum;
    p.authoredColorRandom.referenceHsv  = ParticleRgbToHsv((minimum + maximum) * 0.5f);
    p.authoredColorRandom.interpolation = std::clamp(interpolation, 0.0f, 1.0f);
    p.authoredColorRandom.active        = true;
    p.color      = minimum + (maximum - minimum) * p.authoredColorRandom.interpolation;
    p.init.color = p.color;
}

inline void Reset(Particle& p) {
    p.alpha = p.init.alpha;
    p.size  = p.init.size;
    p.color = p.init.color;
}

inline void MarkOld(Particle& p) { p.mark_new = false; }
inline bool IsNew(const Particle& p) { return p.mark_new; }

inline const Eigen::Vector3f& GetPos(const Particle& p) { return p.position; }
inline const Eigen::Vector3f& GetVelocity(const Particle& p) { return p.velocity; }
inline const Eigen::Vector3f& GetRenderVelocity(const Particle& p) {
    return p.hasRenderVelocity ? p.renderVelocity : p.velocity;
}
inline const Eigen::Vector3f& GetAngular(const Particle& p) { return p.rotation; }

}; // namespace ParticleModify
} // namespace wallpaper

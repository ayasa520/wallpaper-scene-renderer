#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <Eigen/Dense>

namespace wallpaper
{

// Face size by shadows quality: low/medium=256, high=512, ultra=1024.
inline int ShadowFaceSize(int quality) {
    if (quality >= 4) return 1024;
    if (quality >= 3) return 512;
    return 256;
}

// Point-light shadow FOV by quality. Sampling scales view-space xy/z by
// 0.47/0.48/0.49, which is 0.5/tan(fov/2) for 94°/92°/91.2°.
inline float ShadowPointFovDegrees(int quality) {
    if (quality >= 4) return 91.2f;
    if (quality >= 3) return 92.0f;
    return 94.0f;
}

inline constexpr float kShadowNearDefault = 0.05f;

// Constant bias on the render projection's B term only (not sampling projectionInfo).
inline constexpr float kShadowDepthBiasB = 0.0005f;

// Six cube faces packed as a 2×3 grid inside each square slot.
inline constexpr float kShadowPointViewportScaleX = 0.5f;
inline constexpr float kShadowPointViewportScaleY = 0.3333f;

// Sampling's project matrix is identity-xy plus projectionInfo in z/w:
//   clip.z = info.x * view_z + info.y
//   clip.w = info.z * view_z + info.w
// The six face views in CalculateProjectedCoordsPoint put a point in front of
// the light at negative view-z (e.g. +X: view_z = light.x - world.x). The
// right-handed reversed pair maps distance d in [near, far] to ndc.z in
// [1, 0], the same convention as the scene camera: the near plane is 1 and an
// empty (far-cleared) atlas tile is 0, so SampleCmp GREATER lights every
// unoccluded ray.
inline Eigen::Vector4f ShadowProjectionInfo(float nearp, float farp) {
    const float n     = nearp;
    const float f     = std::max(farp, n + 0.01f);
    const float denom = std::max(f - n, 1.0e-6f);
    return Eigen::Vector4f(n / denom, n * f / denom, -1.0f, 0.0f);
}

// Same z/w as ShadowProjectionInfo. Rendering also bakes FOV into xy so NDC
// [-1, 1] matches the 0.47-compensated sample UVs.
inline Eigen::Matrix4f ShadowPointProjection(float nearp, float farp, float fov_degrees,
                                             bool render_bias) {
    const float n = nearp;
    const float f = std::max(farp, n + 0.01f);
    const float half =
        std::max(fov_degrees, 0.1f) * 0.5f * 0.01745329238474369f;
    const float fl    = 1.0f / std::tan(half);
    const float denom = std::max(f - n, 1.0e-6f);
    const float a     = n / denom;
    float       b     = n * f / denom;
    // The constant render bias moves stored casters slightly toward the light; with reversed
    // depth that is the positive direction.
    if (render_bias) b += kShadowDepthBiasB;
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    proj(0, 0)           = fl;
    proj(1, 1)           = fl;
    proj(2, 2)           = a;
    proj(2, 3)           = b;
    proj(3, 2)           = -1.0f;
    return proj;
}

// HLSL `mul(p, viewMatrix)` in common_pbr_2.h is row-vector. Eigen is column-vector,
// so these are the transposes of the six `mat4(...)` constructors (faces +X,-X,+Y,-Y,+Z,-Z).
inline Eigen::Matrix4f ShadowPointView(int face, const Eigen::Vector3f& origin) {
    const float ox = origin.x();
    const float oy = origin.y();
    const float oz = origin.z();
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    switch (face) {
    case 0: // +X
        view << 0, 0, 1, -oz, 0, 1, 0, -oy, -1, 0, 0, ox, 0, 0, 0, 1;
        break;
    case 1: // -X
        view << 0, 0, -1, oz, 0, 1, 0, -oy, 1, 0, 0, -ox, 0, 0, 0, 1;
        break;
    case 2: // +Y
        view << 1, 0, 0, -ox, 0, 0, 1, -oz, 0, -1, 0, oy, 0, 0, 0, 1;
        break;
    case 3: // -Y
        view << 1, 0, 0, -ox, 0, 0, -1, oz, 0, 1, 0, -oy, 0, 0, 0, 1;
        break;
    case 4: // +Z
        view << -1, 0, 0, ox, 0, 1, 0, -oy, 0, 0, -1, oz, 0, 0, 0, 1;
        break;
    default: // -Z
        view << 1, 0, 0, -ox, 0, 1, 0, -oy, 0, 0, 1, -oz, 0, 0, 0, 1;
        break;
    }
    return view;
}

struct ShadowFaceViewport {
    float x { 0.0f };
    float y { 0.0f };
    float width { 0.0f };
    float height { 0.0f };
    int   scissor_x { 0 };
    int   scissor_y { 0 };
    int   scissor_w { 1 };
    int   scissor_h { 1 };
};

// D3D NDC Y+ is the top of the viewport. Vulkan default NDC Y+ is the bottom.
// Negative height maps D3D Y+ onto the top of the atlas tile so sampling's
// `viewportPointCompensation.y = -0.47` matches the rasterized faces.
inline ShadowFaceViewport ShadowPointFaceViewport(int slot_x, int slot_y, int face_size,
                                                  int face_index) {
    const int   col = face_index & 1;
    const int   row = face_index / 2;
    const float vw  = static_cast<float>(face_size) * kShadowPointViewportScaleX;
    const float vh  = static_cast<float>(face_size) * kShadowPointViewportScaleY;
    const float x0  = static_cast<float>(slot_x) + static_cast<float>(col) * vw;
    const float y0  = static_cast<float>(slot_y) + static_cast<float>(row) * vh;
    ShadowFaceViewport out;
    out.x      = x0;
    out.y      = y0 + vh;
    out.width  = vw;
    out.height = -vh;
    const int x1 = static_cast<int>(std::floor(x0));
    const int y1 = static_cast<int>(std::floor(y0));
    const int x2 = static_cast<int>(std::ceil(x0 + vw));
    const int y2 = static_cast<int>(std::ceil(y0 + vh));
    out.scissor_x = std::max(x1, 0);
    out.scissor_y = std::max(y1, 0);
    out.scissor_w = std::max(x2 - out.scissor_x, 1);
    out.scissor_h = std::max(y2 - out.scissor_y, 1);
    return out;
}

inline ShadowFaceViewport ShadowSpotViewport(int slot_x, int slot_y, int face_size) {
    ShadowFaceViewport out;
    out.x         = static_cast<float>(slot_x);
    out.y         = static_cast<float>(slot_y + face_size);
    out.width     = static_cast<float>(face_size);
    out.height    = -static_cast<float>(face_size);
    out.scissor_x = std::max(slot_x, 0);
    out.scissor_y = std::max(slot_y, 0);
    out.scissor_w = std::max(face_size, 1);
    out.scissor_h = std::max(face_size, 1);
    return out;
}

} // namespace wallpaper

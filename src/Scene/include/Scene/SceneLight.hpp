#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include "SceneNode.h"

namespace wallpaper
{

enum class SceneLightType
{
    Point,
    Spot,
    Directional,
    Tube,
    Other,
};

class SceneLight {
public:
    SceneLight(Eigen::Vector3f color, float radius, float intensity)
        : m_color(color), m_radius(radius), m_intensity(intensity) {
        m_premultiplied_color = m_color * m_intensity * m_radius * m_radius;
    }
    ~SceneLight() = default;

    Eigen::Vector3f color() const { return m_color; }
    float           radius() const { return m_radius; }
    float           intensity() const { return m_intensity; }
    SceneNode*      node() const { return m_node.get(); }

    void setColor(Eigen::Vector3f color) {
        m_color               = color;
        m_premultiplied_color = m_color * m_intensity * m_radius * m_radius;
    }
    void setRadius(float radius) {
        m_radius              = radius;
        m_premultiplied_color = m_color * m_intensity * m_radius * m_radius;
    }
    void setIntensity(float intensity) {
        m_intensity           = intensity;
        m_premultiplied_color = m_color * m_intensity * m_radius * m_radius;
    }

    // g_LightsColorRadius stores the non-radius-premultiplied light color in rgb and the authored
    // radius in w. Derive that color from the constructor-time premultiplied payload so the model
    // and legacy 2D light uniforms stay numerically tied to the same parsed light energy.
    Eigen::Vector3f colorIntensity() const {
        const float radius_sq = m_radius * m_radius;
        if (radius_sq <= 1e-6f) return m_color * m_intensity;
        return m_premultiplied_color / radius_sq;
    }

    Eigen::Vector3f premultipliedColor() const { return m_premultiplied_color; }

    void setNode(std::shared_ptr<SceneNode> node) { m_node = node; }

    SceneLightType type() const { return m_type; }
    void           setType(SceneLightType type) { m_type = type; }

    bool castVolumetrics() const { return m_cast_volumetrics; }
    void setCastVolumetrics(bool enabled) { m_cast_volumetrics = enabled; }

    // Volumetric lighting is Point/Spot only. Directional and tube keep their
    // authored castvolumetrics flag for editor round-trips but never enter the
    // volumetric graph.
    bool typeSupportsVolumetrics() const {
        return m_type == SceneLightType::Point || m_type == SceneLightType::Spot;
    }

    float density() const { return m_density; }
    void  setDensity(float density) { m_density = density; }

    float volumetricsExponent() const { return m_volumetrics_exponent; }
    void  setVolumetricsExponent(float exponent) { m_volumetrics_exponent = exponent; }

    float innerCone() const { return m_inner_cone; }
    void  setInnerCone(float degrees) { m_inner_cone = degrees; }

    float outerCone() const { return m_outer_cone; }
    void  setOuterCone(float degrees) { m_outer_cone = degrees; }

    bool               hasCookie() const { return m_has_cookie; }
    const std::string& cookie() const { return m_cookie; }
    void               setCookie(std::string cookie) {
        m_cookie     = std::move(cookie);
        m_has_cookie = ! m_cookie.empty();
    }

    bool castsShadows() const { return m_casts_shadows; }
    void setCastsShadows(bool enabled) { m_casts_shadows = enabled; }

    float exponent() const { return m_exponent; }
    void  setExponent(float exponent) { m_exponent = exponent; }

    float cascadeDistance(int index) const {
        if (index <= 0) return m_cascade0;
        if (index == 1) return m_cascade1;
        return m_cascade2;
    }
    void setCascadeDistances(float d0, float d1, float d2) {
        m_cascade0 = d0;
        m_cascade1 = d1;
        m_cascade2 = d2;
    }

    // One square `_rt_shadowAtlas` slot. Point lights pack six faces in a 2×3
    // grid; spots occupy the full square. Directional lights use three cascade
    // slots instead of this single slot.
    struct ShadowAtlasSlot {
        bool packed { false };
        bool point { true };
        bool directional { false };
        int  cascade_index { 0 };
        int  quality { 2 };
        int  x { 0 };
        int  y { 0 };
        int  size { 0 };
        int  atlas_w { 2 };
        int  atlas_h { 2 };
    };

    void                   setShadowAtlasSlot(ShadowAtlasSlot slot) { m_shadow_slot = slot; }
    void                   clearShadowAtlasSlot() {
        m_shadow_slot = {};
        m_cascade_slots[0] = {};
        m_cascade_slots[1] = {};
        m_cascade_slots[2] = {};
    }
    const ShadowAtlasSlot& shadowAtlasSlot() const { return m_shadow_slot; }
    void setCascadeAtlasSlot(int index, ShadowAtlasSlot slot) {
        if (index < 0 || index > 2) return;
        m_cascade_slots[index] = slot;
    }
    const ShadowAtlasSlot& cascadeAtlasSlot(int index) const {
        if (index <= 0) return m_cascade_slots[0];
        if (index == 1) return m_cascade_slots[1];
        return m_cascade_slots[2];
    }

    Eigen::Vector4f ShadowAtlasUv() const {
        if (! m_shadow_slot.packed || m_shadow_slot.atlas_w <= 0 || m_shadow_slot.atlas_h <= 0) {
            return Eigen::Vector4f::Zero();
        }
        const float w = static_cast<float>(m_shadow_slot.atlas_w);
        const float h = static_cast<float>(m_shadow_slot.atlas_h);
        return Eigen::Vector4f(static_cast<float>(m_shadow_slot.x) / w,
                               static_cast<float>(m_shadow_slot.y) / h,
                               static_cast<float>(m_shadow_slot.size) / w,
                               static_cast<float>(m_shadow_slot.size) / h);
    }

    float ShadowNearPlane() const { return 0.05f; }

    float ShadowFarPlane() const { return std::max(m_radius, ShadowNearPlane() + 0.01f); }

    // Point lights with SHADOW write projectionInfo into g_RenderVar3.
    // Must match CalculateProjectedCoordsPoint: negative view-z in front of
    // each cube face, so clip.w = -view_z and ndc.z stays in [0, 1].
    Eigen::Vector4f ShadowProjectionInfo() const {
        const float n     = ShadowNearPlane();
        const float f     = ShadowFarPlane();
        const float denom = std::max(f - n, 1.0e-6f);
        return Eigen::Vector4f(-f / denom, -n * f / denom, -1.0f, 0.0f);
    }

    Eigen::Vector3f WorldOrigin() const {
        if (m_node == nullptr) return Eigen::Vector3f::Zero();
        m_node->UpdateTrans();
        return m_node->ModelTrans().block<3, 1>(0, 3).cast<float>();
    }

    Eigen::Vector3f WorldForward() const {
        if (m_node == nullptr) return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        m_node->UpdateTrans();
        Eigen::Vector3f forward = m_node->ModelTrans().block<3, 1>(0, 2).cast<float>();
        if (forward.squaredNorm() > 1e-12f) return forward.normalized();
        return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }

    // Degrees to radians for g_RenderVar1 inner/outer cone cosines.
    static constexpr float Deg2Rad() { return 0.01745329238474369f; }

    Eigen::Matrix4f LightAxes() const {
        const Eigen::Vector3f z = WorldForward();
        Eigen::Vector3f       up(0.0f, 1.0f, 0.0f);
        if (std::abs(z.dot(up)) > 0.99f) up = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f x = up.cross(z);
        if (x.squaredNorm() <= 1e-12f) x = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        x.normalize();
        const Eigen::Vector3f y = z.cross(x).normalized();
        Eigen::Matrix4f       axes = Eigen::Matrix4f::Identity();
        axes.block<3, 1>(0, 0)     = x;
        axes.block<3, 1>(0, 1)     = y;
        axes.block<3, 1>(0, 2)     = z;
        return axes;
    }

    // Point AltVP: scale by radius and translate to the light origin.
    Eigen::Matrix4f PointAltViewProjection() const {
        const Eigen::Vector3f origin = WorldOrigin();
        Eigen::Matrix4f       m      = Eigen::Matrix4f::Identity();
        m(0, 0)                      = m_radius;
        m(1, 1)                      = m_radius;
        m(2, 2)                      = m_radius;
        m(0, 3)                      = origin.x();
        m(1, 3)                      = origin.y();
        m(2, 3)                      = origin.z();
        return m;
    }

    Eigen::Matrix4f SpotLocalToWorld() const {
        const float rad     = std::max(m_outer_cone, 0.0f) * Deg2Rad();
        const float lateral = m_radius * std::tan(rad);
        Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
        scale(0, 0)           = lateral;
        scale(1, 1)           = lateral;
        scale(2, 2)           = m_radius;
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
        const Eigen::Vector3f origin = WorldOrigin();
        trans(0, 3)                  = origin.x();
        trans(1, 3)                  = origin.y();
        trans(2, 3)                  = origin.z();
        return trans * LightAxes() * scale;
    }

    Eigen::Matrix4f WorldToLightClip() const {
        const Eigen::Vector3f origin  = WorldOrigin();
        const Eigen::Matrix4f axes    = LightAxes();
        Eigen::Matrix4f       world_to_light = Eigen::Matrix4f::Identity();
        world_to_light.block<3, 3>(0, 0)     = axes.block<3, 3>(0, 0).transpose();
        const Eigen::Vector3f t = -world_to_light.block<3, 3>(0, 0) * origin;
        world_to_light(0, 3)    = t.x();
        world_to_light(1, 3)    = t.y();
        world_to_light(2, 3)    = t.z();

        const float fov   = std::max(m_outer_cone * 2.0f, 0.1f) * Deg2Rad();
        const float nearp = std::max(m_radius * 0.001f, 0.01f);
        const float farp  = std::max(m_radius, nearp + 0.01f);
        const float f     = 1.0f / std::tan(fov * 0.5f);
        Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
        proj(0, 0)           = f;
        proj(1, 1)           = f;
        proj(2, 2)           = farp / (farp - nearp);
        proj(2, 3)           = -nearp * farp / (farp - nearp);
        proj(3, 2)           = 1.0f;
        return proj * world_to_light;
    }

    // Same clip transform as WorldToLightClip, with a constant B bias for the
    // shadow rasterizer. Sampling still uses the unbiased matrix.
    Eigen::Matrix4f ShadowWorldToLightClip() const {
        Eigen::Matrix4f clip = WorldToLightClip();
        clip(2, 3) -= 0.0005f;
        return clip;
    }

    // Directional cascade VP. Authored cascadedistanceN is the ortho half-extent
    // of cascade N. The eye sits cascadeDistance along -forward from `center`
    // so the frustum covers that box. Sampling uses render_bias=false.
    Eigen::Matrix4f ShadowCascadeWorldToLightClip(int cascade, const Eigen::Vector3f& center,
                                                  bool render_bias) const {
        const float               half = std::max(cascadeDistance(cascade), 0.01f);
        const Eigen::Vector3f     fwd  = WorldForward();
        const Eigen::Vector3f     eye  = center - fwd * half;
        const Eigen::Matrix4f     axes = LightAxes();
        Eigen::Matrix4f           world_to_light = Eigen::Matrix4f::Identity();
        world_to_light.block<3, 3>(0, 0)         = axes.block<3, 3>(0, 0).transpose();
        const Eigen::Vector3f t = -world_to_light.block<3, 3>(0, 0) * eye;
        world_to_light(0, 3)    = t.x();
        world_to_light(1, 3)    = t.y();
        world_to_light(2, 3)    = t.z();

        const float n = ShadowNearPlane();
        const float f = std::max(half * 2.0f, n + 0.01f);
        Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
        proj(0, 0)           = 1.0f / half;
        proj(1, 1)           = 1.0f / half;
        // D3D / Vulkan clip z is [0, 1]. Official cascade sampling feeds this z
        // straight to texSample2DCompare and treats abs(ndc) > 0.99 as out of tile.
        proj(2, 2)           = 1.0f / (f - n);
        float b              = -n / (f - n);
        if (render_bias) b += 0.0005f;
        proj(2, 3) = b;
        proj(3, 3) = 1.0f;
        return proj * world_to_light;
    }

    Eigen::Matrix4f AltViewProjection() const {
        if (m_type == SceneLightType::Point) return PointAltViewProjection();
        if (m_has_cookie) {
            const Eigen::Matrix4f clip = WorldToLightClip();
            if (std::abs(clip.determinant()) > 1e-12f) return clip.inverse();
        }
        return SpotLocalToWorld();
    }

    // Camera-in-volume test. Sample point is eye + lookdir * 0.2 (point/spot)
    // or 0.1 (cookie 6-plane).
    bool CameraInsideVolume(const Eigen::Vector3f& eye, const Eigen::Vector3f& lookdir) const {
        Eigen::Vector3f dir = lookdir;
        if (dir.squaredNorm() > 1e-12f) dir.normalize();
        const float             offset     = m_has_cookie ? 0.1f : 0.2f;
        const Eigen::Vector3f   camera_pos = eye + dir * offset;
        const Eigen::Vector3f   origin     = WorldOrigin();
        const Eigen::Vector3f   delta      = camera_pos - origin;

        if (m_type != SceneLightType::Spot) {
            return delta.squaredNorm() < m_radius * m_radius;
        }

        if (m_has_cookie) {
            const Eigen::Vector4f clip =
                WorldToLightClip() * Eigen::Vector4f(camera_pos.x(), camera_pos.y(), camera_pos.z(), 1.0f);
            if (clip.w() <= 0.0f) return false;
            return std::abs(clip.x()) <= clip.w() && std::abs(clip.y()) <= clip.w() &&
                   std::abs(clip.z()) <= clip.w();
        }

        const Eigen::Vector3f forward = WorldForward();
        const float           along   = delta.dot(forward);
        if (along <= 0.0f || along > m_radius) return false;
        const Eigen::Vector3f perp        = delta - forward * along;
        const float           outer_rad   = m_outer_cone * Deg2Rad();
        const float           cone_radius = along * std::tan(outer_rad);
        return perp.norm() <= cone_radius;
    }

private:
    Eigen::Vector3f m_color { Eigen::Vector3f::Zero() };
    float           m_radius { 0.0f };
    float           m_intensity { 1.0f };

    Eigen::Vector3f            m_premultiplied_color { Eigen::Vector3f::Zero() };
    std::shared_ptr<SceneNode> m_node { nullptr };

    SceneLightType m_type { SceneLightType::Other };
    bool           m_cast_volumetrics { false };
    float          m_density { 1.0f };
    float          m_volumetrics_exponent { 1.0f };
    float          m_inner_cone { 0.0f };
    float          m_outer_cone { 45.0f };
    bool           m_has_cookie { false };
    std::string    m_cookie;
    bool           m_casts_shadows { false };
    float          m_exponent { 1.0f };
    float          m_cascade0 { 25.0f };
    float          m_cascade1 { 50.0f };
    float          m_cascade2 { 200.0f };
    ShadowAtlasSlot m_shadow_slot {};
    ShadowAtlasSlot m_cascade_slots[3] {};
};
} // namespace wallpaper

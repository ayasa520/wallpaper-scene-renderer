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

struct DirectionalShadowView {
    Eigen::Vector3f eye { Eigen::Vector3f::Zero() };
    Eigen::Vector3f forward { 0.0f, 0.0f, -1.0f };
    bool            orthographic { false };
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

    Eigen::Vector2f ShadowCascadeExtents(int index) const {
        const float xy_extent = cascadeDistance(index);
        const float shared_depth_extent = m_cascade1 * 4.0f;
        const float depth_extent = index < 2
                                       ? shared_depth_extent
                                       : std::max(m_cascade2 * 1.5f, shared_depth_extent);
        return { xy_extent, depth_extent };
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
    // each cube face, so clip.w = -view_z and ndc.z runs reversed from 1 at the
    // near plane to 0 at the far plane, like the atlas it is compared against.
    Eigen::Vector4f ShadowProjectionInfo() const {
        const float n     = ShadowNearPlane();
        const float f     = ShadowFarPlane();
        const float denom = std::max(f - n, 1.0e-6f);
        return Eigen::Vector4f(n / denom, n * f / denom, -1.0f, 0.0f);
    }

    // Parented lights compose the authored ancestor chain at draw time like every other routed
    // layer. The shader-value updater publishes that composed transform once per frame; without
    // it the physical scene-graph transform is the authoritative world placement.
    void SetResolvedWorldTransform(const Eigen::Matrix4d& transform) {
        m_resolvedWorld    = transform;
        m_hasResolvedWorld = true;
    }

    Eigen::Matrix4d WorldTransform() const {
        if (m_hasResolvedWorld) return m_resolvedWorld;
        if (m_node == nullptr) return Eigen::Matrix4d::Identity();
        m_node->UpdateTrans();
        return m_node->ModelTrans();
    }

    Eigen::Vector3f WorldOrigin() const {
        if (m_node == nullptr && ! m_hasResolvedWorld) return Eigen::Vector3f::Zero();
        return WorldTransform().block<3, 1>(0, 3).cast<float>();
    }

    Eigen::Vector3f WorldForward() const {
        // Wallpaper Engine lights travel along their authored local +X axis. Its light-array
        // builder reads transform column zero for spot direction and negates the same column for
        // directional LightingV1. Using the conventional +Z model-forward axis rotates both the
        // direct-light lobe and every shadow camera away from the authored sun direction.
        if (m_node == nullptr && ! m_hasResolvedWorld) return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f forward = WorldTransform().block<3, 1>(0, 0).cast<float>();
        if (forward.squaredNorm() > 1e-12f) return forward.normalized();
        return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
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

    Eigen::Matrix4f WorldToLight() const {
        const Eigen::Vector3f origin  = WorldOrigin();
        const Eigen::Matrix4f axes    = LightAxes();
        Eigen::Matrix4f       world_to_light = Eigen::Matrix4f::Identity();
        world_to_light.block<3, 3>(0, 0)     = axes.block<3, 3>(0, 0).transpose();
        const Eigen::Vector3f t = -world_to_light.block<3, 3>(0, 0) * origin;
        world_to_light(0, 3)    = t.x();
        world_to_light(1, 3)    = t.y();
        world_to_light(2, 3)    = t.z();
        return world_to_light;
    }

    // Spot cone clip used by the cookie projection and the volumetric hull: light space looks
    // down +z, near maps to clip depth 0 and far to 1.
    Eigen::Matrix4f WorldToLightClip() const {
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
        return proj * WorldToLight();
    }

    // Spot shadow clip: same cone, but depth runs reversed (near = 1, far = 0) like the shadow
    // atlas it is rasterized into and sampled against. The render variant adds a constant B
    // bias that moves stored casters slightly toward the light; sampling stays unbiased.
    Eigen::Matrix4f ShadowSpotWorldToLightClip(bool render_bias) const {
        const float fov   = std::max(m_outer_cone * 2.0f, 0.1f) * Deg2Rad();
        const float nearp = std::max(m_radius * 0.001f, 0.01f);
        const float farp  = std::max(m_radius, nearp + 0.01f);
        const float f     = 1.0f / std::tan(fov * 0.5f);
        Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
        proj(0, 0)           = f;
        proj(1, 1)           = f;
        proj(2, 2)           = -nearp / (farp - nearp);
        proj(2, 3)           = nearp * farp / (farp - nearp);
        if (render_bias) proj(2, 3) += 0.0005f;
        proj(3, 2)           = 1.0f;
        return proj * WorldToLight();
    }

    Eigen::Matrix4f ShadowWorldToLightClip() const { return ShadowSpotWorldToLightClip(true); }

    Eigen::Vector3f ShadowCascadeCenter(int cascade,
                                        const DirectionalShadowView& view) const {
        const float half_xy = ShadowCascadeExtents(cascade).x() * 0.5f;
        const Eigen::Vector3f light_forward = WorldForward();

        // Move each cascade away from the eye by half of its authored XY extent. Before applying
        // that distance, remove half of the camera-forward component parallel to the light travel
        // direction. An eye-centered square spends half of every authored cascade behind the
        // camera, causing distant receivers to leave the far cascade and nearer receivers to use
        // a coarser cascade than intended. The perspective path applies the offset in all axes;
        // the orthographic path pins the resulting world Z coordinate to zero.
        const Eigen::Vector3f offset_direction =
            view.forward - light_forward * (0.5f * view.forward.dot(light_forward));
        Eigen::Vector3f center = view.eye + offset_direction * half_xy;
        if (view.orthographic) center.z() = 0.0f;
        return center;
    }

    Eigen::Vector3f StabilizedShadowCascadeCenter(int cascade,
                                                  const DirectionalShadowView& view,
                                                  int shadow_map_size) const {
        Eigen::Vector3f center = ShadowCascadeCenter(cascade, view);
        const float world_texel = ShadowCascadeExtents(cascade).x() /
                                  static_cast<float>(shadow_map_size);
        const Eigen::Matrix4f axes = LightAxes();
        const Eigen::Vector3f light_x = axes.block<3, 1>(0, 0);
        const Eigen::Vector3f light_y = axes.block<3, 1>(0, 1);

        // Project the moving cascade center onto both axes of the shadow plane, remove each
        // coordinate's floating remainder by one world-space shadow texel, and only then build
        // the light view. Keeping render and sample matrices on this identical quantized center
        // prevents camera motion and camera shake from sliding stationary geometry through the
        // far-cascade texels.
        const float remainder_x = std::fmod(light_x.dot(center), world_texel);
        const float remainder_y = std::fmod(light_y.dot(center), world_texel);
        return center - light_x * remainder_x - light_y * remainder_y;
    }

    // Directional cascade VP. Authored cascadedistanceN is the full XY extent, while the
    // light-space depth is a separate full extent: 4*d1 for cascades 0/1 and max(1.5*d2, 4*d1)
    // for cascade 2. Keeping depth tied to each cascade's XY size clips distant casters as a
    // receiver enters a nearer cascade, which appears as a shadow being cut at a fixed
    // screen-space boundary. Center the orthographic Z interval on the cascade center so the
    // near/far planes stay symmetric. Sampling uses render_bias=false.
    Eigen::Matrix4f ShadowCascadeWorldToLightClip(int cascade,
                                                  const DirectionalShadowView& view,
                                                  int shadow_map_size, bool render_bias) const {
        const Eigen::Vector2f extents    = ShadowCascadeExtents(cascade);
        const float           half_xy    = extents.x() * 0.5f;
        const float           half_depth = extents.y() * 0.5f;
        const Eigen::Matrix4f axes       = LightAxes();
        const Eigen::Vector3f stable_center =
            StabilizedShadowCascadeCenter(cascade, view, shadow_map_size);
        Eigen::Matrix4f       world_to_light = Eigen::Matrix4f::Identity();
        world_to_light.block<3, 3>(0, 0)     = axes.block<3, 3>(0, 0).transpose();
        const Eigen::Vector3f t = -world_to_light.block<3, 3>(0, 0) * stable_center;
        world_to_light(0, 3)    = t.x();
        world_to_light(1, 3)    = t.y();
        world_to_light(2, 3)    = t.z();

        Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
        proj(0, 0)           = 1.0f / half_xy;
        proj(1, 1)           = 1.0f / half_xy;
        // Reversed depth over the symmetric [-halfDepth, +halfDepth] interval: light-space zero
        // maps to 0.5, the near limit to 1 and the far limit to 0. The render bias pushes stored
        // casters slightly away from the light, which is the negative direction here.
        proj(2, 2)           = -1.0f / (half_depth * 2.0f);
        float b              = 0.5f;
        if (render_bias) b -= 0.0005f;
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
    Eigen::Matrix4d            m_resolvedWorld { Eigen::Matrix4d::Identity() };
    bool                       m_hasResolvedWorld { false };

    SceneLightType m_type { SceneLightType::Other };
    bool           m_cast_volumetrics { false };
    float          m_density { 1.0f };
    float          m_volumetrics_exponent { 1.0f };
    float          m_inner_cone { 0.0f };
    float          m_outer_cone { 45.0f };
    bool           m_has_cookie { false };
    std::string    m_cookie;
    bool           m_casts_shadows { false };
    float          m_exponent { 2.0f };
    float          m_cascade0 { 25.0f };
    float          m_cascade1 { 50.0f };
    float          m_cascade2 { 200.0f };
    ShadowAtlasSlot m_shadow_slot {};
    ShadowAtlasSlot m_cascade_slots[3] {};
};
} // namespace wallpaper

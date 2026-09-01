#pragma once
#include "Core/Literals.hpp"
#include "Core/NoCopyMove.hpp"
#include "Core/MapSet.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace wallpaper
{
class SceneNode;
class SceneCamera;
class SceneShader;
class ShaderValue;
class SpriteAnimation;

using sprite_map_t    = Map<usize, SpriteAnimation>;
using UpdateUniformOp = std::function<void(std::string_view, ShaderValue)>;
using ExistsUniformOp = std::function<bool(std::string_view)>;

struct ShaderUniformOverrides {
    std::string_view camera_name;
    bool             use_camera_override { false };
    // Some source-seed passes still write into an offscreen effect target owned by the node's
    // camera, but their shader samples screen-space data. Those passes need uniforms evaluated
    // against the live active camera without permanently changing the node's authored camera.
    bool             use_active_camera_for_uniforms { false };
    bool             use_active_camera_for_parallax { false };
};

struct ShaderSkinningPose {
    // The matrices are immutable for the current prepared frame. Render passes must consume or copy
    // the span before the next PrepareFrame(), which is the only operation allowed to advance it.
    std::span<const Eigen::Affine3f> matrices;
    uint64_t                         revision { 0 };
    uint64_t                         frame_serial { 0 };
};

class IShaderValueUpdater : NoCopy, NoMove {
public:
    IShaderValueUpdater()          = default;
    virtual ~IShaderValueUpdater() = default;

    // Resource-affecting pose/surface preparation must finish before render-graph refresh. The
    // later FrameBegin hook remains the draw-phase boundary for uniform/cache consumers.
    virtual void PrepareFrame()                                                    = 0;
    virtual void FrameBegin()                                                      = 0;
    virtual void InitUniforms(SceneNode*, const ExistsUniformOp&)                  = 0;
    virtual void UpdateUniforms(SceneNode*, sprite_map_t&, const UpdateUniformOp&,
                                const ShaderUniformOverrides* overrides = nullptr) = 0;
    virtual void FrameEnd()                                                        = 0;

    // Projection-driven resources must use the same inherited-parent, attachment, and camera
    // parallax transform contract as shader uniforms. Exposing that matrix here keeps sizing code
    // independent from the concrete Wallpaper Engine updater implementation.
    virtual Eigen::Matrix4d ResolveModelTransformForProjection(
        SceneNode* node, const SceneCamera* camera, bool apply_parallax) = 0;

    // Depth-only material passes need the exact palette selected for the visible material in the
    // same frame. Keeping this query on the updater interface prevents render backends from owning
    // or advancing Wallpaper Engine animation state.
    virtual std::optional<ShaderSkinningPose> SkinningPose(SceneNode* node) const = 0;

    virtual void MouseInput(double x, double y) = 0;
    virtual void SetTexelSize(float x, float y) = 0;
    virtual void SetScreenSize(i32 w, i32 h)    = 0;
};
} // namespace wallpaper

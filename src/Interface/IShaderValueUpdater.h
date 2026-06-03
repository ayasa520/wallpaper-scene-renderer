#pragma once
#include "Core/Literals.hpp"
#include "Core/NoCopyMove.hpp"
#include "Core/MapSet.hpp"

#include <functional>
#include <string_view>

namespace wallpaper
{
class SceneNode;
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

class IShaderValueUpdater : NoCopy, NoMove {
public:
    IShaderValueUpdater()          = default;
    virtual ~IShaderValueUpdater() = default;

    virtual void FrameBegin()                                                      = 0;
    virtual void InitUniforms(SceneNode*, const ExistsUniformOp&)                  = 0;
    virtual void UpdateUniforms(SceneNode*, sprite_map_t&, const UpdateUniformOp&,
                                const ShaderUniformOverrides* overrides = nullptr) = 0;
    virtual void FrameEnd()                                                        = 0;

    virtual void MouseInput(double x, double y) = 0;
    virtual void SetTexelSize(float x, float y) = 0;
    virtual void SetScreenSize(i32 w, i32 h)    = 0;
};
} // namespace wallpaper

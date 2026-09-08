#pragma once
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <optional>

#include <Eigen/Dense>

#include "Core/Core.hpp"
#include "Interface/IShaderValueUpdater.h"
#include "Core/MapSet.hpp"
#include "SpriteAnimation.hpp"
#include "WPPuppet.hpp"
#include "Scene/SceneLight.hpp"

namespace wallpaper
{

class Scene;
class SceneNode;
class SceneMesh;
class SceneImageEffectLayer;
class WPNodeTransformResolver;

struct WPUniformInfo {
    bool has_MI { false };
    bool has_M { false };
    bool has_AM { false };
    bool has_ANM { false };
    bool has_AVP { false };
    bool has_EM { false };
    bool has_RV0 { false };
    bool has_RV1 { false };
    bool has_RV2 { false };
    bool has_RV3 { false };
    bool has_RV4 { false };
    bool has_MVP { false };
    // Wallpaper Engine effect passes can render through helper meshes while shader math still
    // depends on the authored layer transform. Track these matrix uniforms separately so effect
    // shaders get that contract without requiring a g_ModelViewProjectionMatrix declaration too.
    bool has_LMM { false };
    bool has_EMVP { false };
    bool has_EMVPI { false };
    bool has_MVPI { false };
    bool has_ETVP { false };
    bool has_ETVPI { false };
    bool has_VP { false };

    bool has_BONES { false };
    bool has_TIME { false };
    bool has_DAYTIME { false };
    // Cursor feedback shaders need these values as a coherent per-frame set. Tracking them beside
    // the older pointer position bit keeps UpdateUniforms data-driven: ordinary materials do not pay
    // for cursor state writes, while effects like cursorripple receive every uniform they declare.
    bool has_FRAMETIME { false };
    bool has_POINTERPOSITION { false };
    bool has_POINTERPOSITIONLAST { false };
    bool has_POINTERSTATE { false };
    bool has_PARALLAXPOSITION { false };
    bool has_TEXELSIZE { false };
    bool has_TEXELSIZEHALF { false };
    bool has_SCREEN { false };
    bool has_LP { false };
    // These extra light payloads and the camera basis uniforms are part of the 3D model shader
    // contract. InitUniforms enables them only on materialized model nodes so 2D shaders keep their
    // previous uniform surface even if they happen to declare similarly named values.
    bool has_model_LCP { false };
    bool has_LCR { false };
    bool has_LPOINT_ORIGIN { false };
    bool has_LPOINT_COLOR { false };
    bool has_LSPOT_ORIGIN { false };
    bool has_LSPOT_COLOR { false };
    bool has_LSPOT_DIRECTION { false };
    bool has_LSPOT_EXPONENT { false };
    bool has_LDIR_COLOR { false };
    bool has_LDIR_DIRECTION { false };
    bool has_LTUBE_ORIGINA { false };
    bool has_LTUBE_ORIGINB { false };
    bool has_LTUBE_COLOR { false };
    bool has_LFEAT_SHADOW_POINT_PROJ { false };
    bool has_LFEAT_SHADOW_POINT_XFORM { false };
    bool has_LFEAT_SHADOW_PROJ { false };
    bool has_LFEAT_SHADOW_PROJ_XFORM { false };
    bool has_EYE_POSITION { false };
    bool has_NORMAL_MODEL_MATRIX { false };
    bool has_VIEWUP { false };
    bool has_VIEWRIGHT { false };
    bool has_VIEWFORWARD { false };
    std::array<bool, 3> has_audio_spectrum_left { false, false, false };
    std::array<bool, 3> has_audio_spectrum_right { false, false, false };

    struct Tex {
        bool has_resolution { false };
        bool has_texel { false };
        bool has_mipmap { false };
    };
    std::array<Tex, 12> texs;
};

enum class WPNodeTransformBindingMode
{
    None,
    InheritParent,
    BoneAttachment,
};

struct WPNodeTransformBinding {
    WPNodeTransformBindingMode mode { WPNodeTransformBindingMode::None };
    SceneNode*                 parent { nullptr };
    uint32_t                   bone_index { 0xFFFFFFFFu };
    Eigen::Affine3f            bind_transform { Eigen::Affine3f::Identity() };

    bool InheritsParentTransform() const {
        return mode == WPNodeTransformBindingMode::InheritParent;
    }

    bool IsBoneAttachment() const {
        return mode == WPNodeTransformBindingMode::BoneAttachment;
    }
};

struct EffectLayerProjectionBinding {
    SceneImageEffectLayer* layer { nullptr };
};

struct WPShaderValueData {
    std::array<float, 2> parallaxDepth { 0.0f, 0.0f };
    // index + name

    WPPuppetLayer puppet_layer;
    WPNodeTransformBinding transform_binding {};
    EffectLayerProjectionBinding effect_layer_projection {};
    // Private source/surface phases rasterize in local coordinates. Their visible destination
    // applies the authored root's displacement once; raw object matrices never contain it.
    bool                  suppress_model_parallax { false };

    // Volumetric util materials pack g_RenderVar0–4 and the light-volume matrices per light.
    SceneLight*           volumetric_light { nullptr };
    bool                  volumetric_pass { false };

    void SetParallaxContract(const std::array<float, 2>& depth,
                             bool suppress_own_model_parallax = false) {
        parallaxDepth           = depth;
        suppress_model_parallax = suppress_own_model_parallax;
    }

    void SetEffectLayerProjection(SceneImageEffectLayer* layer) {
        effect_layer_projection.layer = layer;
    }

    void SuppressOwnModelParallax() { suppress_model_parallax = true; }

    void CopyParallaxContractFrom(const WPShaderValueData& source) {
        SetParallaxContract(source.parallaxDepth, source.suppress_model_parallax);
    }

    void InheritParentTransform(SceneNode* parent) {
        transform_binding.mode = WPNodeTransformBindingMode::InheritParent;
        transform_binding.parent = parent;
    }

    void AttachToBone(SceneNode* parent, uint32_t bone_index,
                      const Eigen::Affine3f& bind_transform) {
        transform_binding.mode             = WPNodeTransformBindingMode::BoneAttachment;
        transform_binding.parent           = parent;
        transform_binding.bone_index       = bone_index;
        transform_binding.bind_transform   = bind_transform;
    }

    bool InheritsSceneParentTransform() const {
        return transform_binding.InheritsParentTransform();
    }

    bool IsBoneAttached() const { return transform_binding.IsBoneAttachment(); }

    bool AppliesModelParallax() const {
        return ! suppress_model_parallax;
    }

    SceneNode* TransformParent() const { return transform_binding.parent; }
};

struct WPCameraParallax {
    bool  enable { false };
    float amount;
    float delay;
    float mouseinfluence;
};

class WPShaderValueUpdater : public IShaderValueUpdater {
public:
    WPShaderValueUpdater(Scene* scene): m_scene(scene) {}
    virtual ~WPShaderValueUpdater() {}

    void PrepareFrame() override;
    void FrameBegin() override;

    void InitUniforms(const SceneDraw&, const ExistsUniformOp&) override;
    void UpdateUniforms(const SceneDraw&, sprite_map_t&, const UpdateUniformOp&,
                        const ShaderUniformOverrides* overrides = nullptr) override;
    void FrameEnd() override;
    Eigen::Matrix4d ResolveModelTransformForProjection(
        const SceneDraw& draw, const SceneCamera* camera, bool apply_parallax) override;
    std::optional<ShaderSkinningPose> SkinningPose(SceneNode* node) const override;
    void MouseInput(double, double) override;
    void SetTexelSize(float x, float y) override;

    void SetNodeData(void*, const WPShaderValueData&);
    void RemoveDrawData(const SceneDraw& draw);
    const WPShaderValueData* GetNodeData(const void* node_addr) const;
    WPShaderValueData*       GetNodeData(const void* node_addr);
    void SetCameraParallax(const WPCameraParallax& value) {
        m_parallax = value;
        // Camera parallax changes alter the derived model transforms even when the authored layer
        // transform data is unchanged. Clear the per-frame caches immediately so a runtime toggle
        // cannot leave puppet/model layers using offsets computed with the previous global state.
        m_modelTransformCache.clear();
        m_parallaxOffsetCache.clear();
    }
    void     AdvanceAllPuppets();
    uint64_t NextPuppetFrameSerial() const noexcept { return m_puppet_frame_serial + 1; }

    void SetScreenSize(i32 w, i32 h) override { m_screen_size = { (float)w, (float)h }; }

private:
    struct EffectProjectionSnapshot {
        Eigen::Matrix4d layer_model;
        Eigen::Matrix4d placed_model;
        Eigen::Matrix4d view_projection;
        Eigen::Matrix4d incoming_view_projection;
    };
    EffectProjectionSnapshot ResolveEffectProjectionSnapshot(const SceneImageEffectLayer& layer);
    void UpdatePointerState();

    Scene*               m_scene;
    WPCameraParallax     m_parallax;
    double               m_dayTime { 0.0f };
    std::array<float, 2> m_texelSize { 1.0f / 1920.0f, 1.0f / 1080.0f };

    // Pointer uniforms expose the unsmoothed engine input. Camera-parallax delay owns a separate
    // sample because Wallpaper Engine applies that temporal filter only to its look-at target;
    // sharing one sample would delay cursor feedback shaders whenever a scene authored parallax
    // delay, including scenes where camera parallax itself is disabled.
    std::array<float, 2> m_pointerPos { 0.5f, 0.5f };
    std::array<float, 2> m_pointerPosLast { 0.5f, 0.5f };
    std::array<float, 2> m_pointerPosInput { 0.5f, 0.5f };
    std::array<float, 2> m_parallaxPointerPos { 0.5f, 0.5f };

    std::array<float, 2> m_screen_size { 1920, 1080 };

    uint64_t                     m_puppet_frame_serial { 0 };
    Map<void*, Eigen::Matrix4d>  m_modelTransformCache;
    Map<void*, Eigen::Vector3f>  m_parallaxOffsetCache;
    Map<void*, WPShaderValueData> m_nodeDataMap;
    Map<void*, WPUniformInfo>     m_nodeUniformInfoMap;
};
} // namespace wallpaper

#pragma once
// Internal surface shared between the WPSceneParser translation units (the core parser and
// WPSceneParserPostFx). This header is deliberately not part of the public include tree: it
// exists so scene.json parsing can be split into cohesive files without giving parser
// internals a public API. Everything here stays in the global namespace because the parser
// units compile with `using namespace wallpaper;` and historically defined these symbols at
// global scope; moving them into a named namespace would churn thousands of references for
// no behavioral gain.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>
#include <nlohmann/json_fwd.hpp>

#include "Core/Literals.hpp"
#include "Fs/VFS.h"
#include "WPDynamicValue.hpp"
#include "WPUserSetting.hpp"
#include "Scene/LightingV1.hpp"
#include "Scene/Scene.h"
#include "WPShaderParser.hpp"
#include "WPShaderValueUpdater.hpp"
#include "Particle/Particle.h"
#include "WPUserProperties.hpp"
#include "wpscene/WPImageObject.h"
#include "wpscene/WPParticleObject.h"

struct ParseContext {
    std::shared_ptr<wallpaper::Scene> scene;
    wallpaper::WPShaderValueUpdater*  shader_updater;
    wallpaper::i32                    ortho_w;
    wallpaper::i32                    ortho_h;
    wallpaper::fs::VFS*               vfs;
    const wallpaper::UserPropertyMap* user_properties;

    wallpaper::ShaderValueMap                                          global_base_uniforms;
    std::shared_ptr<wallpaper::SceneNode>                              effect_camera_node;
    std::shared_ptr<wallpaper::SceneNode>                              global_camera_node;
    std::shared_ptr<wallpaper::SceneNode>                              global_perspective_camera_node;
    std::unordered_set<int32_t>                                        dependent_parent_ids;
    std::unordered_map<int32_t, std::shared_ptr<wallpaper::SceneNode>> object_nodes;
    std::unordered_map<int32_t, const wallpaper::WPPuppet*>            object_puppets;
    // A forward source reference has no texture extent while its reader is materialized.
    // Retain the exact source geometry payload and material until all initial targets exist;
    // effect/direct draw wrappers may share that payload without sharing a mesh wrapper.
    struct PendingImageSourceMapping {
        std::shared_ptr<wallpaper::SceneMesh>     geometry;
        std::shared_ptr<wallpaper::SceneMaterial> material;
        int32_t                                 layer_id;
        bool                                    crop_card_uvs;
    };
    std::vector<PendingImageSourceMapping> pending_image_source_mappings;
    // Main-scene model chunks preserve the color produced by preceding chunks. Reflection
    // has its own graph-stage clear and uses these same materials with per-draw load state.
    wallpaper::usize model_pass_count { 0 };
};

enum class GeometryStagePolicy {
    Disabled,
    MatchMaterial,
    Required,
};

struct MaterialLoadResult {
    bool geometry_stage_loaded { false };
};

// Defined in WPSceneParser.cpp; also consumed by the post-process configuration unit.
std::optional<MaterialLoadResult>
LoadMaterial(wallpaper::fs::VFS& vfs, const wallpaper::wpscene::WPMaterial& wpmat,
             wallpaper::Scene* pScene,
             wallpaper::SceneMaterial* pMaterial, wallpaper::WPShaderValueData* pSvData,
             const wallpaper::UserPropertyMap* user_properties = nullptr,
             wallpaper::WPShaderInfo*          pWPShaderInfo   = nullptr,
             GeometryStagePolicy geometry_stage = GeometryStagePolicy::Disabled);

// Defined in WPSceneParserPostFx.cpp; consumed by the core parser (scene setup and the
// lighting uniform block LoadMaterial seeds).
bool                      ConfigureSceneBloomPass(ParseContext& context);
bool                      ConfigureSceneVolumetricsImpl(wallpaper::Scene& scene,
                                                        wallpaper::fs::VFS& vfs);
bool                      SceneHasShadowLights(const wallpaper::Scene& scene);
wallpaper::LightingV1Desc LightingDescFromScene(const wallpaper::Scene& scene);

// The authored 3D model object. Parsed by the core parser's object dispatch, materialized by
// WPSceneParserModel.cpp.
struct WPModelObject {
    int32_t              id { 0 };
    std::string          name;
    std::array<float, 3> origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> angles { 0.0f, 0.0f, 0.0f };
    bool                 visible { true };
    wallpaper::VisibleBinding visible_binding;
    int32_t              parent { 0 };
    std::string          attachment;
    std::string          model;
    uint32_t             model_token { 0 };
    bool                 perspective { false };
    int32_t              skin { 0 };
    bool                 reflected { true };
    std::vector<wallpaper::WPPuppetLayer::AnimationLayer> animation_layers;
    // 3D models omit this key when they should cast. Non-casters write false.
    bool                 castshadow { true };

    bool FromJson(const nlohmann::json& json, wallpaper::fs::VFS&);
};

// The isolated model camera keeps authored 3D view transforms away from `global_perspective`;
// the reflection target is registered only after model materialization identifies an active
// receiver sampler. Reflected-list membership alone does not allocate a scene target.
inline constexpr std::string_view kSceneModelPerspectiveCameraName { "__hanabi_model_perspective" };
inline constexpr std::string_view kModelReflectionTargetName { "_rt_Reflection" };

// Defined in WPSceneParser.cpp; shared with the model unit.
std::string DescribeIndexVec(const std::vector<wallpaper::usize>& values);
bool ConfigureBoneAttachment(ParseContext& context, int32_t parent_id, std::string_view attachment,
                             std::string_view object_kind,
                             std::string_view object_name, wallpaper::WPShaderValueData& node_data);
void AttachNodeToScene(ParseContext& context, const std::shared_ptr<wallpaper::SceneNode>& node,
                       int32_t parent_id, const std::string& object_name);
void RegisterLayerSceneState(ParseContext& context, int32_t layer_id, int32_t parent_id,
                             std::string_view attachment, bool visible);
void LoadConstvalue(wallpaper::SceneMaterial& material, const wallpaper::wpscene::WPMaterial& wpmat,
                    const wallpaper::WPShaderInfo& info);
void RegisterUserShaderValueBindings(ParseContext& context,
                                     const wallpaper::wpscene::WPMaterial& wpmat,
                                     const wallpaper::WPShaderInfo& info,
                                     wallpaper::SceneNode* node, int32_t object_id,
                                     std::string_view object_name);
void RegisterConstantShaderValueBindings(ParseContext& context,
                                         const wallpaper::wpscene::WPMaterial& wpmat,
                                         const wallpaper::WPShaderInfo& info,
                                         wallpaper::SceneNode* node, int32_t object_id,
                                         std::string_view object_name, int32_t effect_id,
                                         int32_t effect_index, wallpaper::usize material_index);
void LoadUserShaderValue(wallpaper::SceneMaterial& material,
                         const wallpaper::wpscene::WPMaterial& wpmat,
                         const wallpaper::WPShaderInfo& info,
                         const wallpaper::UserPropertyMap* user_properties);

// Defined in WPSceneParserModel.cpp; the core parser dispatches model objects into it.
void ParseModelObj(ParseContext& context, WPModelObject& model_obj);
void RefreshModelObj(ParseContext& context, wallpaper::SceneObject& owner);

// Defined in WPSceneParser.cpp; shared with the bindings unit.
bool IsTextLayerObjectJson(const nlohmann::json& object_json);
void LogTextLayerRegistration(const char* event_name, int32_t object_id,
                              const std::string& object_name, std::string_view property_name,
                              wallpaper::WPDynamicValue::Type hint,
                              const wallpaper::WPUserSetting& setting,
                              const std::optional<wallpaper::WPDynamicValue>& base_value);
bool IsCameraLayerObjectJson(const nlohmann::json& object_json);
bool IsCameraLayerRuntimeProperty(std::string_view property_name);

// Defined in WPSceneParserBindings.cpp; the core parser registers layer and effect property
// bindings after objects and their complete effect chains are materialized.
void RegisterSceneScripts(ParseContext& context, const nlohmann::json& json);
void RegisterSceneScriptsForObject(ParseContext& context, const nlohmann::json& object_json);

// Defined in WPSceneParser.cpp; shared with the particle unit.
bool  LayerUsesRoutedParent(int32_t parent_id, std::string_view attachment);
float RandomParticleFrameLifetime(const wallpaper::Particle& p, float sprite_frame_count_value);
void ConfigureInheritedParentBinding(ParseContext& context, int32_t parent_id,
                                     wallpaper::WPShaderValueData& node_data);
// Parent linkage for recursive particle child parsing. Members are the parsing-time capacity
// contract for child subsystems; see the field comments at the use sites in
// WPSceneParserParticle.cpp.
struct ParticleChildPtr {
    wallpaper::wpscene::ParticleChild* child { nullptr };
    wallpaper::SceneNode*              node_parent { nullptr };
    wallpaper::ParticleSubSystem*      particle_parent { nullptr };
    wallpaper::i32                     max_instancecount { 1 };
    wallpaper::u32                     parent_live_particle_slots { 1 };
};

// Defined in WPSceneParserParticle.cpp; the core parser dispatches particle objects into it,
// and LoadMaterial resolves authored blend strings through the same table particles use.
void ParseParticleObj(ParseContext& context, wallpaper::wpscene::WPParticleObject& wppartobj,
                      ParticleChildPtr child_ptr = {});
wallpaper::BlendMode ParseBlendMode(std::string_view str);

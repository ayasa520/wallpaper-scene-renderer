#pragma once
// Internal surface shared between the WPSceneParser translation units (the core parser and
// WPSceneParserPostFx). This header is deliberately not part of the public include tree: it
// exists so scene.json parsing can be split into cohesive files without giving parser
// internals a public API. Everything here stays in the global namespace because the parser
// units compile with `using namespace wallpaper;` and historically defined these symbols at
// global scope; moving them into a named namespace would churn thousands of references for
// no behavioral gain.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "Core/Literals.hpp"
#include "Fs/VFS.h"
#include "Scene/LightingV1.hpp"
#include "Scene/Scene.h"
#include "WPShaderParser.hpp"
#include "WPShaderValueUpdater.hpp"
#include "WPUserProperties.hpp"
#include "wpscene/WPImageObject.h"

// A hidden layer whose visibility can flip at runtime is registered as a lightweight logical
// placeholder first; the kind records which materializer owns it.
enum class LazyMaterializeKind
{
    None,
    Image,
    Particle,
    Text,
};

struct VisibilityContract {
    bool                authored_visible { true };
    bool                initial_visible { true };
    bool                has_user_binding { false };
    bool                has_script { false };
    bool                has_animation { false };
    bool                referenced_by_script { false };
    bool                dependency_source { false };
    bool                requires_runtime_contract { false };
    bool                can_prune_at_parse_time { false };
    LazyMaterializeKind lazy_materialize_kind { LazyMaterializeKind::None };
};

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
    std::unordered_map<int32_t, VisibilityContract>                    layer_visibility_contracts;
    std::unordered_map<int32_t, int32_t>                               initial_parent_by_layer_id;
    std::unordered_map<int32_t, std::shared_ptr<wallpaper::SceneNode>> object_nodes;
    std::unordered_map<int32_t, const wallpaper::WPPuppet*>            object_puppets;
    // Model chunk passes share the main scene target. Tracking their parse order here lets the
    // model-only material state preserve color after the first model pass without changing the
    // legacy load-op behavior of 2D image/effect passes.
    std::unordered_map<std::string, wallpaper::usize> model_pass_count_by_output;
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
             wallpaper::Scene* pScene, wallpaper::SceneNode* pNode,
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

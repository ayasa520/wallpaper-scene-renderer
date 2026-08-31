#include "WPSceneParser.hpp"
#include "WPSceneParserShared.hpp"

// 3D model (MDL) layer materialization: model material path resolution, the material loader
// and layer materializer, and ParseModelObj. Split from WPSceneParser.cpp as a cohesive unit;
// the parser internals it consumes (ParseContext, LoadMaterial, attachment/scene-state
// helpers) and the ParseModelObj entry point the core parser dispatches into are declared in
// the shared header.

#include "Utils/Logging.h"
#include "Utils/Algorism.h"
#include "Core/StringHelper.hpp"
#include "SpecTexs.hpp"
#include "Scene/ShadowAtlas.hpp"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneTexture.h"
#include "WPJson.hpp"
#include "WPMdlParser.hpp"
#include "WPTexImageParser.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <Eigen/Dense>

using namespace wallpaper;
using namespace Eigen;

namespace
{
std::string StripJsonExtension(std::string value) {
    constexpr std::string_view extension { ".json" };
    if (value.size() >= extension.size() &&
        value.compare(value.size() - extension.size(), extension.size(), extension) == 0) {
        value.resize(value.size() - extension.size());
    }
    return value;
}

std::string MaterialStem(std::string_view material_path) {
    const auto slash = material_path.find_last_of('/');
    const auto start = slash == std::string_view::npos ? 0 : slash + 1;
    return StripJsonExtension(std::string(material_path.substr(start)));
}

std::string MaterialDirectory(std::string_view material_path) {
    const auto slash = material_path.find_last_of('/');
    if (slash == std::string_view::npos) return {};
    return std::string(material_path.substr(0, slash + 1));
}

std::string ResolveModelMaterialPath(const std::string&    material_path,
                                     const nlohmann::json* sidecar_json, int32_t skin) {
    if (sidecar_json == nullptr || ! sidecar_json->is_object() ||
        ! sidecar_json->contains("skins") || ! sidecar_json->at("skins").is_array()) {
        return material_path;
    }

    const auto& skins = sidecar_json->at("skins");
    if (skin < 0 || static_cast<size_t>(skin) >= skins.size() || ! skins.at(skin).is_object()) {
        return material_path;
    }

    const auto  stem   = MaterialStem(material_path);
    const auto& remaps = skins.at(skin);
    if (! remaps.contains(stem) || ! remaps.at(stem).is_string()) return material_path;

    auto remapped = remaps.at(stem).get<std::string>();
    if (remapped.find('/') == std::string::npos) {
        remapped = MaterialDirectory(material_path) + StripJsonExtension(remapped) + ".json";
    } else if (remapped.size() < 5 || remapped.substr(remapped.size() - 5) != ".json") {
        remapped += ".json";
    }
    return remapped;
}

std::string ResolveStaticChunkMaterialPath(const WPMdl::StaticChunk& chunk, int32_t skin) {
    if (chunk.material_json_variants.empty()) return chunk.material_json_file;
    if (skin >= 0 && static_cast<size_t>(skin) < chunk.material_json_variants.size()) {
        // MDLV0004 static models can carry several material paths for one geometry payload. The
        // scene object owns the skin index, so material selection belongs here rather than in the
        // low-level binary parser that only knows the model file bytes.
        return chunk.material_json_variants[skin];
    }

    LOG_ERROR("ModelMaterialSkin: skin=%d out of range variants=%zu fallback='%s'",
              skin,
              chunk.material_json_variants.size(),
              chunk.material_json_file.c_str());
    return chunk.material_json_file;
}

std::optional<nlohmann::json> LoadModelSidecarJson(fs::VFS& vfs, std::string_view model_path) {
    auto path = std::string(model_path);
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".mdl") {
        path.resize(path.size() - 4);
        path += ".json";
    }

    const auto asset_path = "/assets/" + path;
    if (! vfs.Contains(asset_path)) return std::nullopt;

    const auto content = fs::GetFileContent(vfs, asset_path);
    if (content.empty()) return std::nullopt;

    nlohmann::json json;
    if (! PARSE_JSON(content, json)) {
        LOG_ERROR("ModelSidecar: parse failed path='%s'", path.c_str());
        return std::nullopt;
    }
    return json;
}

std::optional<SceneCullMode> ParseModelCullModeValue(std::string_view value) {
    static const std::unordered_map<std::string_view, SceneCullMode> modes {
        { "nocull", SceneCullMode::None }, { "none", SceneCullMode::None },
        { "normal", SceneCullMode::Back }, { "back", SceneCullMode::Back },
        { "front", SceneCullMode::Front },
    };
    if (auto it = modes.find(value); it != modes.end()) return it->second;
    return std::nullopt;
}

std::string_view ModelCullModeMaterialString(SceneCullMode mode) {
    switch (mode) {
    case SceneCullMode::None: return "nocull";
    case SceneCullMode::Back: return "back";
    case SceneCullMode::Front: return "front";
    }
    return "back";
}

std::optional<bool> ParseModelRenderStateSwitch(std::string_view value) {
    if (value == "enabled" || value == "enable" || value == "true" || value == "1") return true;
    if (value == "disabled" || value == "disable" || value == "false" || value == "0")
        return false;
    return std::nullopt;
}

bool ModelBlendUsesTransparency(std::string_view blending) {
    return blending == "translucent" || blending == "additive";
}

bool ModelMaterialSamplesReflection(const wpscene::WPMaterial& material) {
    return std::any_of(
        material.textures.begin(), material.textures.end(), [](const std::string& texture) {
            return texture == kModelReflectionTargetName;
        });
}

struct ModelMaterialRenderPolicy {
    std::string   blending;
    bool          transparent { false };
    bool          depthTest { true };
    bool          depthWrite { true };
    SceneCullMode cullMode { SceneCullMode::Back };
};

std::optional<ModelMaterialRenderPolicy>
BuildModelMaterialRenderPolicy(const wpscene::WPMaterial& material,
                               const std::string&         material_path) {
    ModelMaterialRenderPolicy policy {};
    policy.blending    = material.blendingAuthored ? material.blending : "normal";
    policy.transparent = ModelBlendUsesTransparency(policy.blending);

    if (material.depthtestAuthored) {
        const auto depth_test = ParseModelRenderStateSwitch(material.depthtest);
        if (! depth_test.has_value()) {
            LOG_ERROR("ModelMaterialState: invalid depthtest '%s' path='%s'",
                      material.depthtest.c_str(),
                      material_path.c_str());
            return std::nullopt;
        }
        policy.depthTest = *depth_test;
    }

    if (material.depthwriteAuthored) {
        const auto depth_write = ParseModelRenderStateSwitch(material.depthwrite);
        if (! depth_write.has_value()) {
            LOG_ERROR("ModelMaterialState: invalid depthwrite '%s' path='%s'",
                      material.depthwrite.c_str(),
                      material_path.c_str());
            return std::nullopt;
        }
        policy.depthWrite = *depth_write;
    }
    if (policy.transparent) {
        // Alpha-blended model chunks should test against opaque geometry but avoid writing depth;
        // otherwise transparent quads such as glass, shadow blobs, or reflection grids can occlude
        // later model chunks. This is a model-material rule, not a parser recovery path.
        policy.depthWrite = false;
    }

    if (material.cullmodeAuthored) {
        const auto cull_mode = ParseModelCullModeValue(material.cullmode);
        if (! cull_mode.has_value()) {
            LOG_ERROR("ModelMaterialState: invalid cullmode '%s' path='%s'",
                      material.cullmode.c_str(),
                      material_path.c_str());
            return std::nullopt;
        }
        policy.cullMode = *cull_mode;
    } else if (ModelMaterialSamplesReflection(material) || policy.transparent) {
        // Model reflection and alpha-blended shell surfaces are intentionally double-sided when the
        // material omits culling. Opaque model chunks keep the stricter back-face default, and the
        // policy is scoped here so the legacy 2D WPMaterial defaults remain unchanged.
        policy.cullMode = SceneCullMode::None;
    }

    return policy;
}

wpscene::WPMaterial BuildEffectiveModelMaterial(wpscene::WPMaterial         material,
                                                const ModelMaterialRenderPolicy& policy) {
    // LoadMaterial still consumes WPMaterial because shader, texture, combo, and binding parsing is
    // shared with 2D layers. This effective copy writes the already-validated model policy into the
    // string fields before that shared loader runs, so missing model fields are resolved once at the
    // model boundary instead of being patched later by renderer code.
    material.blending   = policy.blending;
    material.depthtest  = policy.depthTest ? "enabled" : "disabled";
    material.depthwrite = policy.depthWrite ? "enabled" : "disabled";
    material.cullmode   = std::string(ModelCullModeMaterialString(policy.cullMode));
    return material;
}

bool LoadModelMaterialJson(ParseContext& context, const std::string& material_path,
                           nlohmann::json& material_json) {
    const auto material_source = fs::GetFileContent(*context.vfs, "/assets/" + material_path);
    if (material_source.empty()) return false;

    // Wallpaper Engine model materials are JSON-with-comments in practice. Parse that dialect
    // directly with nlohmann's comment-aware mode so valid first-party materials are accepted on the
    // first pass and malformed assets still produce one clear diagnostic.
    material_json = nlohmann::json::parse(material_source, nullptr, false, true);
    if (! material_json.is_discarded()) return true;

    LOG_ERROR("ModelMaterialJson: parse failed path='%s'", material_path.c_str());
    return false;
}

void SeedModelCameraUniforms(ParseContext& context, WPShaderInfo& shader_info) {
    auto& scene     = *context.scene;
    auto  camera_it = scene.cameras.find(std::string(kSceneModelPerspectiveCameraName));
    if (camera_it == scene.cameras.end() || ! camera_it->second) return;

    const auto eye     = camera_it->second->GetPosition().cast<float>();
    Vector3f   forward = camera_it->second->GetDirection().cast<float>();
    if (forward.norm() > 1e-6f) forward.normalize();
    Vector3f up = camera_it->second->GetUp().cast<float>();
    if (up.norm() > 1e-6f) up.normalize();
    Vector3f right = forward.cross(up);
    if (right.norm() > 1e-6f) right.normalize();

    // These constants are seeded only for model materials. Runtime updates in WPShaderValueUpdater
    // keep them animated for camera paths, while 2D materials never receive this 3D camera
    // contract.
    shader_info.baseConstSvs[std::string(G_EYE_POSITION)] =
        std::array<float, 3> { eye.x(), eye.y(), eye.z() };
    shader_info.baseConstSvs[std::string(G_VIEWUP)] =
        std::array<float, 3> { up.x(), up.y(), up.z() };
    shader_info.baseConstSvs[std::string(G_VIEWRIGHT)] =
        std::array<float, 3> { right.x(), right.y(), right.z() };
    shader_info.baseConstSvs[std::string(G_VIEWFORWARD)] =
        std::array<float, 3> { forward.x(), forward.y(), forward.z() };
}

struct ModelMaterialSource {
    std::string               path;
    wpscene::WPMaterial       material;
    ModelMaterialRenderPolicy renderPolicy;
};

class ModelMaterialLoader {
public:
    ModelMaterialLoader(ParseContext& context, const WPModelObject& model_obj,
                        const nlohmann::json* sidecar_json)
        : context_(context), model_obj_(model_obj), sidecar_json_(sidecar_json) {}

    bool AnyChunkSamplesReflection(const WPMdl& mdl) const {
        return std::any_of(
            mdl.static_chunks.begin(), mdl.static_chunks.end(), [this](const auto& chunk) {
                return ChunkSamplesReflection(chunk);
            });
    }

    bool ChunkSamplesReflection(const WPMdl::StaticChunk& chunk) const {
        const auto source = LoadSource(chunk);
        if (!source.has_value()) {
            LOG_ERROR("ModelReflectionMaterial: failed to inspect layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        return ModelMaterialSamplesReflection(source->material);
    }

    bool UsesTransparentBlend(const WPMdl::StaticChunk& chunk) const {
        const auto source = LoadSource(chunk);
        if (! source.has_value()) {
            LOG_ERROR("ModelRenderOrder: failed to inspect layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        return source->renderPolicy.transparent;
    }

    bool LoadChunkMaterial(const WPMdl::StaticChunk& chunk, SceneNode* chunk_node,
                           SceneMaterial& material, WPShaderValueData& node_data,
                           wpscene::WPMaterial& resolved_wp_material,
                           WPShaderInfo& resolved_shader_info,
                           SceneModelColorLoadMode color_load_mode,
                           bool mirrored_handedness, std::string output_override) const {
        const auto source = LoadSource(chunk);
        if (! source.has_value()) {
            LOG_ERROR("ModelMaterialLoad: failed to parse layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        WPShaderInfo shader_info;
        shader_info.baseConstSvs = context_.global_base_uniforms;
        SeedModelCameraUniforms(context_, shader_info);
        if (! LoadMaterial(*context_.vfs,
                           source->material,
                           context_.scene.get(),
                           chunk_node,
                           &material,
                           &node_data,
                           context_.user_properties,
                           &shader_info)) {
            LOG_ERROR("ModelMaterialLoad: shader load failed layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      source->path.c_str());
            return false;
        }

        LoadConstvalue(material, source->material, shader_info);
        LoadUserShaderValue(material, source->material, shader_info, context_.user_properties);
        const auto render_state =
            BuildRenderState(color_load_mode,
                             mirrored_handedness,
                             std::move(output_override),
                             source->renderPolicy);
        material.modelRenderState = render_state;
        // Model material JSON and shader metadata are returned to the caller so binding
        // registration can happen after mesh->AddMaterial() and node->AddMesh(). That keeps 3D
        // model chunks on the same material-ready registration path as ordinary scene layers.
        resolved_wp_material = source->material;
        resolved_shader_info = shader_info;
        return true;
    }

private:
    SceneModelRenderState BuildRenderState(SceneModelColorLoadMode color_load_mode,
                                           bool                    mirrored_handedness,
                                           std::string             output_override,
                                           const ModelMaterialRenderPolicy& policy) const {
        // The renderer-facing model state is derived from the same validated policy used to build
        // the effective WPMaterial, keeping shader loading, depth rules, culling, and reflection
        // output routing on one explicit model-material contract.
        return SceneModelRenderState {
            .colorLoadMode      = color_load_mode,
            .depthTest          = policy.depthTest,
            .depthWrite         = policy.depthWrite,
            .cullMode           = policy.cullMode,
            .mirroredHandedness = mirrored_handedness,
            .outputOverride     = std::move(output_override),
        };
    }

    std::optional<ModelMaterialSource> LoadSource(const WPMdl::StaticChunk& chunk) const {
        const auto material_path = ResolvePath(chunk);
        if (const auto cached = source_cache_.find(material_path); cached != source_cache_.end()) {
            return cached->second;
        }
        if (failed_sources_.count(material_path) != 0) return std::nullopt;

        auto source = LoadSourceFromPath(material_path);
        if (! source.has_value()) {
            failed_sources_.insert(material_path);
            return std::nullopt;
        }

        const auto [cached, inserted] = source_cache_.emplace(material_path, std::move(*source));
        return cached->second;
    }

    std::optional<ModelMaterialSource> LoadSourceFromPath(const std::string& material_path) const {
        nlohmann::json material_json;
        if (! LoadModelMaterialJson(context_, material_path, material_json)) return std::nullopt;

        wpscene::WPMaterial wp_material;
        if (! wp_material.FromJson(material_json)) return std::nullopt;

        const auto render_policy = BuildModelMaterialRenderPolicy(wp_material, material_path);
        if (! render_policy.has_value()) return std::nullopt;

        return ModelMaterialSource {
            .path         = material_path,
            .material     = BuildEffectiveModelMaterial(std::move(wp_material), *render_policy),
            .renderPolicy = *render_policy,
        };
    }

    std::string ResolvePath(const WPMdl::StaticChunk& chunk) const {
        return ResolveModelMaterialPath(
            ResolveStaticChunkMaterialPath(chunk, model_obj_.skin), sidecar_json_, model_obj_.skin);
    }

    ParseContext&         context_;
    const WPModelObject&  model_obj_;
    const nlohmann::json* sidecar_json_ { nullptr };
    mutable std::unordered_map<std::string, ModelMaterialSource> source_cache_;
    mutable std::unordered_set<std::string>                      failed_sources_;
};

void EnsureModelReflectionTarget(ParseContext& context) {
    auto& scene = *context.scene;
    if (scene.renderTargets.count(std::string(kModelReflectionTargetName)) != 0) return;

    // Reflection is a model-only render target. It is registered lazily when a model layer requests
    // reflection so ordinary 2D scenes do not gain another render target or graph edge.
    scene.renderTargets[std::string(kModelReflectionTargetName)] = {
        .width                  = context.ortho_w,
        .height                 = context.ortho_h,
        .mapWidth               = context.ortho_w,
        .mapHeight              = context.ortho_h,
        .allowReuse             = true,
        .withDepth              = true,
        .screenSpaceSampleYFlip = true,
        .bind                   = { .enable = true, .screen = true },
    };
    LOG_INFO("ModelReflectionTarget: registered name='_rt_Reflection' size=%ux%u map-size=%ux%u "
             "with-depth=true screen-aligned=true screen-space-sample-y-flip=true",
             context.ortho_w,
             context.ortho_h,
             context.ortho_w,
             context.ortho_h);
}

struct ModelChunkOrder {
    std::vector<usize> opaque;
    std::vector<usize> transparent;
    std::vector<usize> ordered;

    static ModelChunkOrder Build(const WPMdl& mdl, const ModelMaterialLoader& material_loader,
                                 const WPModelObject& model_obj) {
        ModelChunkOrder order;
        order.opaque.reserve(mdl.static_chunks.size());
        order.transparent.reserve(mdl.static_chunks.size());

        for (usize chunk_index = 0; chunk_index < mdl.static_chunks.size(); chunk_index++) {
            const auto& chunk = mdl.static_chunks[chunk_index];
            if (material_loader.UsesTransparentBlend(chunk)) {
                order.transparent.push_back(chunk_index);
            } else {
                order.opaque.push_back(chunk_index);
            }
        }

        order.ordered.reserve(mdl.static_chunks.size());
        order.ordered.insert(order.ordered.end(), order.opaque.begin(), order.opaque.end());
        order.ordered.insert(
            order.ordered.end(), order.transparent.begin(), order.transparent.end());
        order.Log(model_obj);
        return order;
    }

    void Log(const WPModelObject& model_obj) const {
        if (transparent.empty()) return;

        // Transparent model chunks must be appended after opaque chunks from the same authored
        // model. They still depth-test against the opaque depth buffer, but drawing them last keeps
        // later opaque chunks from overwriting glass that does not write depth. The diagnostic
        // records the exact parser-side order used by run.log.
        LOG_INFO("ModelRenderOrder: layer=%d name='%s' opaque=%s transparent=%s final=%s",
                 model_obj.id,
                 model_obj.name.c_str(),
                 DescribeIndexVec(opaque).c_str(),
                 DescribeIndexVec(transparent).c_str(),
                 DescribeIndexVec(ordered).c_str());
    }
};

struct ModelChunkNodeRequest {
    std::string name;
    std::string output_override;
    SceneModelColorLoadMode color_load_mode { SceneModelColorLoadMode::DontCare };
    bool        mirrored_handedness { false };
};

class ModelLayerMaterializer {
public:
    ModelLayerMaterializer(ParseContext& context, const WPModelObject& model_obj)
        : context_(context),
          model_obj_(model_obj),
          sidecar_json_(LoadModelSidecarJson(*context.vfs, model_obj.model)),
          material_loader_(context_, model_obj_, SidecarJson()) {}

    void Materialize(const WPMdl& mdl) {
        root_ = CreateRootNode();
        RegisterRootNode();
        const bool material_samples_reflection = material_loader_.AnyChunkSamplesReflection(mdl);
        if (model_obj_.reflected || material_samples_reflection) {
            EnsureModelReflectionTarget(context_);
        }

        const auto order = ModelChunkOrder::Build(mdl, material_loader_, model_obj_);
        AppendChunks(mdl, order);
        ApplyCastsShadows(root_.get(), model_obj_.castshadow);

        context_.scene->ApplyLayerVisibility(model_obj_.id);
    }

private:
    const nlohmann::json* SidecarJson() const {
        return sidecar_json_.has_value() ? &*sidecar_json_ : nullptr;
    }

    std::shared_ptr<SceneNode> CreateRootNode() const {
        auto root  = std::make_shared<SceneNode>(Vector3f(model_obj_.origin.data()),
                                                 Vector3f(model_obj_.scale.data()),
                                                 Vector3f(model_obj_.angles.data()),
                                                 model_obj_.name);
        root->ID() = model_obj_.id;
        return root;
    }

    void RegisterRootNode() {
        WPShaderValueData root_data;
        ConfigureBoneAttachment(context_,
                                model_obj_.parent,
                                model_obj_.attachment,
                                Eigen::Affine3f(root_->GetLocalTrans().cast<float>()),
                                "model",
                                model_obj_.name,
                                root_data);
        if (LayerUsesRoutedParent(model_obj_.parent, model_obj_.attachment)) {
            // A parented model composes the full authored ancestor chain at draw time, exactly
            // like parented image/text/particle layers. Physically nesting the model root under
            // its immediate parent node would only apply that parent's local transform, because
            // group ancestors are themselves root-owned routed layers; scripted group scaling and
            // rotation above the model would silently drop out of the model's world transform.
            ConfigureInheritedParentBinding(context_, model_obj_.parent, root_data);
            context_.scene->sceneGraph->AppendChild(root_);
        } else {
            AttachNodeToScene(context_, root_, model_obj_.parent, model_obj_.name, &root_data);
        }

        context_.object_nodes[model_obj_.id] = root_;
        context_.scene->AddLayerRuntimeNode(model_obj_.id, root_.get());
        context_.shader_updater->SetNodeData(root_.get(), root_data);
        RegisterLayerSceneState(
            context_, model_obj_.id, model_obj_.parent, model_obj_.attachment, model_obj_.visible);
    }

    void AppendChunks(const WPMdl& mdl, const ModelChunkOrder& order) {
        // Receiver materials and producer passes are separate contracts. A material may reference
        // `_rt_Reflection` only so its shader can bind the runtime target, while mirrored producer
        // geometry is authored by the model object's `"reflected": true` flag. Keeping those paths
        // separate prevents receiver-only models from drawing phantom mirrored geometry, without
        // disabling reflected scenes that intentionally populate the target.
        if (model_obj_.reflected) {
            for (usize chunk_index : order.ordered) {
                const auto& chunk = mdl.static_chunks[chunk_index];
                AppendReflectionChunk(chunk, chunk_index);
            }
        }

        for (usize chunk_index : order.ordered) {
            const auto& chunk = mdl.static_chunks[chunk_index];
            AppendMainChunk(chunk, chunk_index);
        }
    }

    void AppendReflectionChunk(const WPMdl::StaticChunk& chunk, usize chunk_index) {
        if (material_loader_.ChunkSamplesReflection(chunk)) {
            // A material that samples `_rt_Reflection` is the receiver surface, not a producer for
            // that same target. Skipping it avoids feedback/self-copy edges while still allowing
            // authored reflected models to populate the target before the receiver draws.
            return;
        }

        auto reflection_node = MakeChunkNode(
            chunk,
            ModelChunkNodeRequest {
                .name                = model_obj_.name + "::__hanabi_model_reflection_chunk_" +
                                       std::to_string(chunk_index),
                .output_override     = std::string(kModelReflectionTargetName),
                .color_load_mode     = NextModelOutputColorLoadMode(kModelReflectionTargetName),
                .mirrored_handedness = true,
            });
        if (reflection_node == nullptr) return;

        // Some authored reflection receivers sample `_rt_Reflection` as a screen-space floor
        // mirror. Mirroring reflected chunks across the authored Y=0 floor plane gives the target
        // the expected geometry, while `mirroredHandedness` above lets the render pass fix
        // winding/culling without weakening cull behavior for normal 3D or any 2D scene.
        reflection_node->SetScale(Vector3f { 1.0f, -1.0f, 1.0f });
        root_->AppendChild(reflection_node);
    }

    void AppendMainChunk(const WPMdl::StaticChunk& chunk, usize chunk_index) {
        auto node = MakeChunkNode(
            chunk,
            ModelChunkNodeRequest {
                .name = model_obj_.name + "::__hanabi_model_chunk_" + std::to_string(chunk_index),
                .color_load_mode = NextModelOutputColorLoadMode(SpecTex_Default),
            });
        if (node != nullptr) root_->AppendChild(node);
    }

    std::shared_ptr<SceneNode> MakeChunkNode(const WPMdl::StaticChunk& chunk,
                                             ModelChunkNodeRequest     request) {
        auto node = std::make_shared<SceneNode>();
        node->SetName(std::move(request.name));
        node->ID() = model_obj_.id;
        // Model chunks use the isolated model camera so authored 3D view transforms cannot move
        // legacy 2D perspective particles that still render through `global_perspective`.
        node->SetCamera(std::string(kSceneModelPerspectiveCameraName));

        auto mesh = std::make_shared<SceneMesh>();
        WPMdlParser::GenStaticMesh(*mesh, chunk);

        SceneMaterial       material;
        WPShaderValueData   node_data;
        wpscene::WPMaterial wp_material;
        WPShaderInfo        shader_info;
        if (! material_loader_.LoadChunkMaterial(chunk,
                                                 node.get(),
                                                 material,
                                                 node_data,
                                                 wp_material,
                                                 shader_info,
                                                 request.color_load_mode,
                                                 request.mirrored_handedness,
                                                 std::move(request.output_override))) {
            return nullptr;
        }

        mesh->AddMaterial(std::move(material));
        node->AddMesh(mesh);
        RegisterUserShaderValueBindings(
            context_, wp_material, shader_info, node.get(), model_obj_.id, model_obj_.name);
        // Model materials can author script/user/animation-driven constants (for example an
        // alpha constant toggled from panel scripts through shared state). Register them so the
        // script host drives the chunk material uniform each frame instead of leaving the
        // parse-time fallback value on screen forever.
        RegisterConstantShaderValueBindings(context_,
                                            wp_material,
                                            shader_info,
                                            node.get(),
                                            model_obj_.id,
                                            model_obj_.name,
                                            0,
                                            0,
                                            0);
        // Chunk draws resolve their world transform through the model root's node data, so a
        // routed model root's inherited ancestor transform reaches every chunk. A physically
        // attached root resolves to its plain scene-graph transform through the same path, which
        // keeps unparented and bone-attached models unchanged.
        node_data.InheritParentTransform(root_.get(), false);
        context_.shader_updater->SetNodeData(node.get(), node_data);
        context_.scene->AddLayerRuntimeNode(model_obj_.id, node.get());
        return node;
    }

    void ApplyCastsShadows(SceneNode* node, bool value) {
        if (node == nullptr) return;
        if (node->Name().find("__hanabi_model_reflection") != std::string::npos) return;
        node->SetCastsShadows(value);
        for (auto& child : node->GetChildren()) {
            ApplyCastsShadows(child.get(), value);
        }
    }

    SceneModelColorLoadMode NextModelOutputColorLoadMode(std::string_view output) {
        auto  key   = output.empty() ? std::string(SpecTex_Default) : std::string(output);
        auto& count = context_.model_pass_count_by_output[key];
        if (count++ > 0) return SceneModelColorLoadMode::Load;

        // The default scene target is already owned by the renderer pre-pass, while private model
        // render targets have no standalone clear pass. Clearing the first model writer to an
        // offscreen target prevents transparent pixels from loading the previous frame, and later
        // writers still load so multi-chunk models compose into the same target.
        return key == SpecTex_Default ? SceneModelColorLoadMode::DontCare
                                      : SceneModelColorLoadMode::Clear;
    }

    ParseContext&                 context_;
    const WPModelObject&          model_obj_;
    std::optional<nlohmann::json> sidecar_json_;
    ModelMaterialLoader           material_loader_;
    std::shared_ptr<SceneNode>    root_;
};

} // namespace

void ParseModelObj(ParseContext& context, WPModelObject& model_obj) {
    if (! model_obj.visible) return;

    WPMdl mdl;
    if (! WPMdlParser::ParseStaticModel(model_obj.model, *context.vfs, mdl)) {
        LOG_ERROR("ModelObjectParse: static mdl parse failed layer=%d name='%s' model='%s'",
                  model_obj.id,
                  model_obj.name.c_str(),
                  model_obj.model.c_str());
        return;
    }

    ModelLayerMaterializer(context, model_obj).Materialize(mdl);
}


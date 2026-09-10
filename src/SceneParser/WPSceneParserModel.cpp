#include "WPSceneParser.hpp"
#include "WPSceneParserShared.hpp"

// Model layer materialization: geometry-source adaptation, material path resolution, the
// material loader and layer materializer. Split from WPSceneParser.cpp as a cohesive unit;
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
    const auto* camera = scene.activeCamera;
    if (camera == nullptr) return;

    const auto eye     = scene.FrameEyePosition();
    Vector3f   forward = camera->GetDirection().cast<float>();
    if (forward.norm() > 1e-6f) forward.normalize();
    Vector3f up = camera->GetUp().cast<float>();
    if (up.norm() > 1e-6f) up.normalize();
    Vector3f right = forward.cross(up);
    if (right.norm() > 1e-6f) right.normalize();

    // Model materials consume the selected scene frame's world-space eye and basis, even when
    // their raster projection is an auxiliary perspective view. Runtime updates keep these
    // constants synchronized with camera paths, layers and reflected frame state.
    shader_info.baseConstSvs[std::string(G_EYE_POSITION)] =
        std::array<float, 3> { eye.x(), eye.y(), eye.z() };
    shader_info.baseConstSvs[std::string(G_VIEWUP)] =
        std::array<float, 3> { up.x(), up.y(), up.z() };
    shader_info.baseConstSvs[std::string(G_VIEWRIGHT)] =
        std::array<float, 3> { right.x(), right.y(), right.z() };
    shader_info.baseConstSvs[std::string(G_VIEWFORWARD)] =
        std::array<float, 3> { forward.x(), forward.y(), forward.z() };
}

// Both source kinds feed the same layer/material lifecycle. File geometry is decoded once;
// generated geometry supplies a layer-local mesh wrapper over its shared CPU payload. Keeping
// this adapter independent from materials avoids duplicating layer ownership or manufacturing
// an MDL file for script-owned buffers.
struct ModelGeometryChunk {
    std::string material_path;
    std::shared_ptr<SceneMesh> geometry;
};

struct ModelMaterialSource {
    std::string               path;
    wpscene::WPMaterial       material;
};

class ModelMaterialLoader {
public:
    ModelMaterialLoader(ParseContext& context, const WPModelObject& model_obj,
                        const nlohmann::json* sidecar_json)
        : context_(context), model_obj_(model_obj), sidecar_json_(sidecar_json) {}

    bool UsesTransparentBlend(const ModelGeometryChunk& chunk) const {
        const auto source = LoadSource(chunk);
        if (! source.has_value()) {
            LOG_ERROR("ModelRenderOrder: failed to inspect layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        return source->material.blending == "translucent" || source->material.blending == "additive";
    }

    bool LoadChunkMaterial(const WPMdl* file_model, const ModelGeometryChunk& chunk,
                           SceneMaterial& material, WPShaderValueData& node_data,
                           wpscene::WPMaterial& resolved_wp_material,
                           WPShaderInfo& resolved_shader_info,
                           SceneModelColorLoadMode color_load_mode) const {
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
        auto effective_material = source->material;
        if (file_model != nullptr && file_model->puppet != nullptr) {
            // MDLV0023 model chunks retain their authored interleaved blend-index/weight
            // attributes. Enabling the stock shader combos here declares the g_Bones float4x3
            // array consumed by those attributes on every material chunk.
            WPMdlParser::AddPuppetMatInfo(effective_material, *file_model);
            WPMdlParser::AddPuppetShaderInfo(shader_info, *file_model);
        }
        if (! LoadMaterial(*context_.vfs,
                           effective_material,
                           context_.scene.get(),
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

        LoadConstvalue(material, effective_material, shader_info);
        LoadUserShaderValue(material, effective_material, shader_info, context_.user_properties);
        // Raw depth values come from LoadMaterial. Model-only state selects attachment lifetime
        // and color initialization; common draw policy applies blend-based write suppression.
        material.modelRenderState = SceneModelRenderState { .colorLoadMode = color_load_mode };
        // Model material JSON and shader metadata are returned to the caller so binding
        // registration can happen after mesh->AddMaterial() and node->AddMesh(). That keeps 3D
        // model chunks on the same material-ready registration path as ordinary scene layers.
        resolved_wp_material = std::move(effective_material);
        resolved_shader_info = shader_info;
        return true;
    }

private:
    std::optional<ModelMaterialSource> LoadSource(const ModelGeometryChunk& chunk) const {
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

        return ModelMaterialSource {
            .path         = material_path,
            .material     = std::move(wp_material),
        };
    }

    std::string ResolvePath(const ModelGeometryChunk& chunk) const {
        return ResolveModelMaterialPath(chunk.material_path, sidecar_json_, model_obj_.skin);
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

    // Only an active receiver material requires this screen-sized target. The reflected walk
    // changes destination state and projection before normal scene drawing; sampling uses the
    // authored shader coordinates, with no additional texture-side Y inversion.
    scene.renderTargets[std::string(kModelReflectionTargetName)] = {
        .width                  = context.ortho_w,
        .height                 = context.ortho_h,
        .mapWidth               = context.ortho_w,
        .mapHeight              = context.ortho_h,
        .allowReuse             = true,
        .withDepth              = true,
        .bind                   = { .enable = true, .screen = true },
    };
    LOG_INFO("ModelReflectionTarget: registered name='_rt_Reflection' size=%ux%u map-size=%ux%u "
             "with-depth=true screen-aligned=true screen-space-sample-y-flip=false",
             context.ortho_w,
             context.ortho_h,
             context.ortho_w,
             context.ortho_h);
}

struct ModelChunkOrder {
    std::vector<usize> opaque;
    std::vector<usize> transparent;
    std::vector<usize> ordered;

    static ModelChunkOrder Build(const std::vector<ModelGeometryChunk>& chunks,
                                 const ModelMaterialLoader& material_loader,
                                 const WPModelObject& model_obj) {
        ModelChunkOrder order;
        order.opaque.reserve(chunks.size());
        order.transparent.reserve(chunks.size());

        for (usize chunk_index = 0; chunk_index < chunks.size(); chunk_index++) {
            const auto& chunk = chunks[chunk_index];
            if (material_loader.UsesTransparentBlend(chunk)) {
                order.transparent.push_back(chunk_index);
            } else {
                order.opaque.push_back(chunk_index);
            }
        }

        order.ordered.reserve(chunks.size());
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

class ModelLayerMaterializer {
public:
    ModelLayerMaterializer(ParseContext& context, const WPModelObject& model_obj,
                           const std::vector<ModelGeometryChunk>& chunks,
                           const WPMdl* file_model,
                           std::shared_ptr<SceneModelData> model_data = {})
        : context_(context),
          model_obj_(model_obj),
          chunks_(chunks),
          file_model_(file_model),
          model_data_(std::move(model_data)),
          sidecar_json_(file_model != nullptr
                            ? LoadModelSidecarJson(*context.vfs, model_obj.model) : std::nullopt),
          material_loader_(context_, model_obj_, SidecarJson()) {}

    void Materialize() {
        root_ = CreateRootNode();
        if (file_model_ != nullptr && file_model_->puppet != nullptr) {
            // One model owns one animation-layer stack even when its geometry is split across
            // several material chunks. Copies of WPPuppetLayer share the same runtime state, so
            // scripts mutate the logical root once and every chunk uploads the identical pose
            // snapshot during the frame transaction.
            shared_puppet_pose_ = WPPuppetLayer(file_model_->puppet);
            shared_puppet_pose_.prepared(model_obj_.animation_layers);
        }
        RegisterRootNode();
        auto& owner = context_.scene->EnsureSceneObject(model_obj_.id);
        owner.SetModelData(model_data_);
        owner.SetModelPerspective(model_obj_.perspective);
        MaterializeChunks();
    }

    void Refresh(std::shared_ptr<SceneNode> root) {
        // Only chunk resources are being replaced. Re-registering the root would reset its
        // transform/parent binding and replace the identity addressed by live layer scripts.
        root_ = std::move(root);
        MaterializeChunks();
    }

private:
    void MaterializeChunks() {
        auto& owner = context_.scene->EnsureSceneObject(model_obj_.id);
        owner.SetReceivesReflection(false);
        const auto order = ModelChunkOrder::Build(chunks_, material_loader_, model_obj_);
        AppendChunks(order);
        if (owner.ReceivesReflection()) EnsureModelReflectionTarget(context_);
        ApplyCastsShadows(root_.get(), model_obj_.castshadow);

        context_.scene->ApplyLayerVisibility(model_obj_.id);
        if (model_data_) owner.SetModelDataRevision(model_data_->StructureRevision());
    }

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
        if (shared_puppet_pose_.hasPuppet()) root_data.puppet_layer = shared_puppet_pose_;
        ConfigureBoneAttachment(context_,
                                model_obj_.parent,
                                model_obj_.attachment,
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
            AttachNodeToScene(context_, root_, model_obj_.parent, model_obj_.name);
        }

        context_.object_nodes[model_obj_.id] = root_;
        context_.scene->AddLayerRuntimeNode(model_obj_.id, root_.get());
        context_.shader_updater->SetNodeData(root_.get(), root_data);
        RegisterLayerSceneState(
            context_, model_obj_.id, model_obj_.parent, model_obj_.attachment, model_obj_.visible);
    }

    void AppendChunks(const ModelChunkOrder& order) {
        // The reflected list retains these same owners/resources. Materialize each chunk once,
        // register its script bindings once, and let the render graph submit it in the
        // independent reflection phase when needed.
        for (usize chunk_index : order.ordered) {
            const auto& chunk = chunks_[chunk_index];
            auto node = MakeChunkNode(chunk, chunk_index);
            if (node != nullptr) root_->AppendChild(node);
        }
    }

    std::shared_ptr<SceneNode> MakeChunkNode(const ModelGeometryChunk& chunk,
                                             usize chunk_index) {
        auto node = std::make_shared<SceneNode>();
        node->SetName(model_obj_.name + "::__hanabi_model_chunk_" + std::to_string(chunk_index));
        node->ID() = model_obj_.id;
        // Model chunks inherit the current destination camera. The owner's perspective flag
        // changes only its submitted projection/destination matrices; a named particle camera
        // would discard the incoming view, framing and camera-layer displacement.

        auto mesh = chunk.geometry;
        if (shared_puppet_pose_.hasPuppet()) {
            mesh->SetSkinning(
                { .boneCount = static_cast<uint32_t>(shared_puppet_pose_.Puppet()->bones.size()) });
        }

        SceneMaterial       material;
        WPShaderValueData   node_data;
        wpscene::WPMaterial wp_material;
        WPShaderInfo        shader_info;
        if (! material_loader_.LoadChunkMaterial(file_model_,
                                                 chunk,
                                                 material,
                                                 node_data,
                                                 wp_material,
                                                 shader_info,
                                                 NextModelColorLoadMode())) {
            return nullptr;
        }

        if (material.SamplesTexture(kModelReflectionTargetName)) {
            Set<uint> active_slots;
            if (!WPShaderParser::ReflectTextureSlots(material.customShader.shader->codes,
                                                     active_slots)) {
                LOG_ERROR("ModelReflectionMaterial: descriptor reflection failed layer=%d "
                          "name='%s' chunk=%zu", model_obj_.id, model_obj_.name.c_str(), chunk_index);
                return nullptr;
            }
            for (usize slot = 0; slot < material.textures.size(); ++slot) {
                if (material.Texture(slot) != kModelReflectionTargetName) continue;
                if (active_slots.contains(static_cast<uint>(slot))) {
                    context_.scene->EnsureSceneObject(model_obj_.id).SetReceivesReflection(true);
                    LOG_INFO("ModelReflectionReceiver: layer=%d name='%s' chunk=%zu "
                             "shader='%s' active-slot=%zu whole-owner-excluded=true",
                             model_obj_.id, model_obj_.name.c_str(), chunk_index,
                             material.name.c_str(), slot);
                } else {
                    // An optimized-out sampler is neither a receiver nor a graph read.
                    // Retaining its name would still synthesize a self-copy when this model
                    // is drawn into reflection, despite the GPU never sampling that slot.
                    material.textures[slot].clear();
                    material.systemTextureBindings.erase(slot);
                    LOG_INFO("ModelReflectionSamplerUnused: layer=%d chunk=%zu slot=%zu",
                             model_obj_.id, chunk_index, slot);
                }
            }
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
        node_data.InheritParentTransform(root_.get());
        if (shared_puppet_pose_.hasPuppet()) node_data.puppet_layer = shared_puppet_pose_;
        context_.shader_updater->SetNodeData(node.get(), node_data);
        context_.scene->AddLayerRuntimeNode(model_obj_.id, node.get());
        return node;
    }

    void ApplyCastsShadows(SceneNode* node, bool value) {
        if (node == nullptr) return;
        node->SetCastsShadows(value);
        for (auto& child : node->GetChildren()) {
            ApplyCastsShadows(child.get(), value);
        }
    }

    SceneModelColorLoadMode NextModelColorLoadMode() {
        return context_.model_pass_count++ == 0 ? SceneModelColorLoadMode::DontCare
                                               : SceneModelColorLoadMode::Load;
    }

    ParseContext&                 context_;
    const WPModelObject&          model_obj_;
    const std::vector<ModelGeometryChunk>& chunks_;
    const WPMdl*                  file_model_;
    std::shared_ptr<SceneModelData> model_data_;
    std::optional<nlohmann::json> sidecar_json_;
    ModelMaterialLoader           material_loader_;
    std::shared_ptr<SceneNode>    root_;
    WPPuppetLayer                 shared_puppet_pose_;
};

std::vector<ModelGeometryChunk> GeneratedModelChunks(const SceneModelData& model_data) {
    std::vector<ModelGeometryChunk> chunks;
    chunks.reserve(model_data.Shapes().size());
    for (const auto& shape : model_data.Shapes()) {
        auto mesh = std::make_shared<SceneMesh>(shape.geometry->Dynamic());
        mesh->ChangeMeshDataFrom(*shape.geometry);
        chunks.push_back({ shape.material, std::move(mesh) });
    }
    return chunks;
}

} // namespace

void RefreshModelObj(ParseContext& context, SceneObject& owner) {
    const auto root = context.object_nodes.at(owner.Id());
    WPModelObject model_obj;
    model_obj.id = owner.Id();
    model_obj.name = owner.Name();
    model_obj.castshadow = root->CastsShadows();
    const auto chunks = GeneratedModelChunks(*owner.ModelData());
    ModelLayerMaterializer(context, model_obj, chunks, nullptr, owner.ModelData()).Refresh(root);
}

void ParseModelObj(ParseContext& context, WPModelObject& model_obj) {
    if (model_obj.model_token != 0) {
        auto model_data = context.scene->modelData.Find(model_obj.model_token);
        if (! model_data) {
            LOG_ERROR("ModelObjectParse: invalid generated model token=%u layer=%d",
                      model_obj.model_token, model_obj.id);
            return;
        }
        const auto chunks = GeneratedModelChunks(*model_data);
        ModelLayerMaterializer(context, model_obj, chunks, nullptr, model_data).Materialize();
        LOG_INFO("ModelDataLayer: token=%u layer=%d name='%s' shapes=%zu perspective=%s",
                 model_obj.model_token, model_obj.id, model_obj.name.c_str(), chunks.size(),
                 model_obj.perspective ? "true" : "false");
        return;
    }

    WPMdl mdl;
    if (! WPMdlParser::ParseStaticModel(model_obj.model,
                                        *context.vfs,
                                        mdl,
                                        ! model_obj.animation_layers.empty())) {
        LOG_ERROR("ModelObjectParse: static mdl parse failed layer=%d name='%s' model='%s'",
                  model_obj.id,
                  model_obj.name.c_str(),
                  model_obj.model.c_str());
        return;
    }

    std::vector<ModelGeometryChunk> chunks;
    chunks.reserve(mdl.static_chunks.size());
    for (const auto& chunk : mdl.static_chunks) {
        auto mesh = std::make_shared<SceneMesh>();
        WPMdlParser::GenStaticMesh(*mesh, chunk);
        chunks.push_back({ ResolveStaticChunkMaterialPath(chunk, model_obj.skin), std::move(mesh) });
    }
    ModelLayerMaterializer(context, model_obj, chunks, &mdl).Materialize();
}

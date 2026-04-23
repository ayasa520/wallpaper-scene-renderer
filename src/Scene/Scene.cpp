#include "Scene.h"

#include "SceneCamera.h"

#include "Fs/VFS.h"
#include "Interface/IImageParser.h"
#include "Interface/IShaderValueUpdater.h"
#include "Particle/ParticleSystem.h"
#include "Utils/Logging.h"

#include <cmath>
#include <unordered_set>

namespace wallpaper 
{

namespace
{
std::string FindDinoRunScoreLayerName(const Scene& scene, int32_t layer_id) {
    if (scene.scene_id.find("dino_run") == std::string::npos) return {};

    if (auto it = scene.textLayers.find(layer_id); it != scene.textLayers.end()) {
        if (it->second.object.name == "label_coins" || it->second.object.name == "label_top") {
            return it->second.object.name;
        }
    }

    if (auto it = scene.layerNodes.find(layer_id); it != scene.layerNodes.end() && it->second != nullptr) {
        const auto& name = it->second->Name();
        if (name == "label_coins" || name == "label_top") return name;
    }

    for (const auto& [name, id] : scene.layerNameToId) {
        if (id == layer_id && (name == "label_coins" || name == "label_top")) return name;
    }
    return {};
}

bool IsLayerVisibleImpl(const Scene& scene, int32_t layer_id, std::unordered_set<int32_t>& visiting) {
    if (layer_id == 0) return true;
    if (!visiting.insert(layer_id).second) return true;

    const auto visible_it = scene.layerLocalVisibility.find(layer_id);
    const bool local_visible =
        visible_it == scene.layerLocalVisibility.end() ? true : visible_it->second;
    if (!local_visible) return false;

    const auto binding_it = scene.layerParentBindings.find(layer_id);
    if (binding_it == scene.layerParentBindings.end() || binding_it->second.parent_id == 0) {
        return true;
    }

    return IsLayerVisibleImpl(scene, binding_it->second.parent_id, visiting);
}

void CollectLayerEffectNodes(const Scene& scene, int32_t layer_id, std::vector<SceneNode*>& nodes) {
    auto camera_names_it = scene.objectRuntimeCameraNames.find(layer_id);
    if (camera_names_it == scene.objectRuntimeCameraNames.end()) return;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = scene.cameras.find(camera_name);
        if (camera_it == scene.cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr) continue;

        if (effect_layer->HasFinalComposite()) {
            // Final composite nodes are owned by the image-effect bridge instead of the authored
            // scene tree, so they will not be reached by normal parent/child propagation. Treat
            // them as layer-owned runtime nodes here to keep layer visibility authoritative while
            // preserving effect-local visibility on the internal shader nodes.
            nodes.push_back(&effect_layer->FinalNode());
        }

        for (size_t effect_index = 0; effect_index < effect_layer->EffectCount(); effect_index++) {
            auto& effect = effect_layer->GetEffect(effect_index);
            for (auto& effect_node : effect->nodes) {
                if (effect_node.sceneNode) nodes.push_back(effect_node.sceneNode.get());
            }
        }
    }
}

void ApplyLayerVisibilityRecursive(Scene& scene, int32_t layer_id, std::unordered_set<int32_t>& visited) {
    if (layer_id == 0 || !visited.insert(layer_id).second) return;

    std::unordered_set<int32_t> visiting;
    const bool effective_visible = IsLayerVisibleImpl(scene, layer_id, visiting);
    const auto trace_name = FindDinoRunScoreLayerName(scene, layer_id);

    if (auto runtime_nodes_it = scene.objectRuntimeNodes.find(layer_id);
        runtime_nodes_it != scene.objectRuntimeNodes.end()) {
        for (auto* node : runtime_nodes_it->second) {
            if (node != nullptr) {
                // Layer visibility propagation must not overwrite a node's own local visibility
                // contract. Runtime-owned support nodes may intentionally stay hidden even while
                // their authored layer is visible, so the scene system only updates the
                // layer-level flag.
                node->SetLayerVisible(effective_visible);
            }
        }
    }

    std::vector<SceneNode*> effect_nodes;
    CollectLayerEffectNodes(scene, layer_id, effect_nodes);
    for (auto* node : effect_nodes) {
        if (node != nullptr) {
            // Effect nodes are also owned by the layer-visibility system, but they still need to
            // preserve any explicit local visibility decisions that the effect pipeline may make.
            node->SetLayerVisible(effective_visible);
        }
    }

    if (!trace_name.empty()) {
        const auto binding = scene.GetLayerParentBinding(layer_id);
        const auto runtime_nodes_it = scene.objectRuntimeNodes.find(layer_id);
        const size_t runtime_node_count =
            runtime_nodes_it == scene.objectRuntimeNodes.end() ? 0 : runtime_nodes_it->second.size();
    }

    for (const auto& [child_id, binding] : scene.layerParentBindings) {
        if (binding.parent_id == layer_id) {
            ApplyLayerVisibilityRecursive(scene, child_id, visited);
        }
    }
}

std::pair<int32_t, Scene::CameraLayerRuntimeState*> FindActiveCameraLayer(Scene& scene) {
    // Wallpaper Engine uses the bottom-most visible camera layer as the active view. Scene JSON is
    // parsed in layer order, so walking the recorded camera layer order backwards gives later
    // camera layers precedence while still letting user/script visibility changes disable them.
    for (auto it = scene.cameraLayerOrder.rbegin(); it != scene.cameraLayerOrder.rend(); ++it) {
        auto layer_it = scene.cameraLayers.find(*it);
        if (layer_it == scene.cameraLayers.end() || !layer_it->second.node) continue;
        if (!scene.IsLayerVisible(*it)) continue;
        return { *it, &layer_it->second };
    }
    return { 0, nullptr };
}

double SanitizeCameraZoom(double zoom, int32_t layer_id) {
    if (std::isfinite(zoom) && zoom > 0.0001) return zoom;

    // Invalid authored/user zoom values would collapse the orthographic projection to infinity.
    // Log the offending camera layer and keep a neutral zoom so the wallpaper remains visible.
    LOG_ERROR("SceneCameraLayer: invalid zoom %.6f on layer=%d, using 1.0", zoom, layer_id);
    return 1.0;
}

void ApplyCameraProjectionState(Scene& scene,
                                const std::string& camera_name,
                                SceneCamera& camera,
                                double zoom,
                                float fov,
                                int32_t layer_id) {
    if (camera.IsPerspective()) {
        camera.SetFov(fov);
    } else {
        const double safe_zoom = SanitizeCameraZoom(zoom, layer_id);
        camera.SetWidth(std::max(1.0, static_cast<double>(scene.ortho[0]) / safe_zoom));
        camera.SetHeight(std::max(1.0, static_cast<double>(scene.ortho[1]) / safe_zoom));
    }

    camera.Update();
    scene.UpdateLinkedCamera(camera_name);
}
} // namespace

Scene::Scene(): sceneGraph(std::make_shared<SceneNode>()) ,paritileSys(std::make_unique<ParticleSystem>(*this)) {}
Scene::~Scene() = default;

void Scene::SetLayerParentBinding(int32_t layer_id, int32_t parent_id, std::string attachment) {
    if (layer_id == 0) return;
    if (parent_id == 0 && attachment.empty()) {
        layerParentBindings.erase(layer_id);
        return;
    }
    layerParentBindings[layer_id] = LayerParentBinding {
        .parent_id = parent_id,
        .attachment = std::move(attachment),
    };
}

Scene::LayerParentBinding Scene::GetLayerParentBinding(int32_t layer_id) const {
    auto it = layerParentBindings.find(layer_id);
    return it == layerParentBindings.end() ? LayerParentBinding {} : it->second;
}

void Scene::ClearLayerParentBinding(int32_t layer_id) {
    layerParentBindings.erase(layer_id);
}

std::vector<int32_t> Scene::GetLayerChildren(int32_t layer_id) const {
    std::vector<int32_t> children;
    for (const auto& [child_id, binding] : layerParentBindings) {
        if (binding.parent_id == layer_id) children.push_back(child_id);
    }
    return children;
}

void Scene::SetLayerLocalVisibility(int32_t layer_id, bool visible) {
    if (layer_id == 0) return;
    layerLocalVisibility[layer_id] = visible;
}

bool Scene::GetLayerLocalVisibility(int32_t layer_id) const {
    auto it = layerLocalVisibility.find(layer_id);
    return it == layerLocalVisibility.end() ? true : it->second;
}

bool Scene::IsLayerVisible(int32_t layer_id) const {
    std::unordered_set<int32_t> visiting;
    return IsLayerVisibleImpl(*this, layer_id, visiting);
}

void Scene::ApplyLayerVisibility(int32_t layer_id) {
    std::unordered_set<int32_t> visited;
    ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    if (!cameraLayers.empty()) UpdateActiveCameraLayer();
}

void Scene::ApplyAllLayerVisibility() {
    std::unordered_set<int32_t> visited;
    for (const auto layer_id : layerOrder) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    for (const auto& [layer_id, _] : layerNodes) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    if (!cameraLayers.empty()) UpdateActiveCameraLayer();
}

Eigen::Vector3f Scene::ResolveCameraLayerNodeTranslation(
    const std::array<float, 3>& authored_origin) const {
    // WE 2D camera origins are authored around the static camera origin, where 0/0 means the
    // default centered wallpaper view. Hanabi's orthographic camera node is centered in render
    // coordinates, so add the canvas half-size before attaching the SceneCamera to this layer.
    return Eigen::Vector3f {
        static_cast<float>(ortho[0]) * 0.5f + authored_origin[0],
        static_cast<float>(ortho[1]) * 0.5f + authored_origin[1],
        authored_origin[2],
    };
}

void Scene::UpdateActiveCameraLayer() {
    auto [next_layer_id, camera_layer] = FindActiveCameraLayer(*this);

    std::string camera_name = "global";
    std::shared_ptr<SceneNode> camera_node = defaultGlobalCameraNode;
    double zoom = defaultGlobalCameraZoom;
    float fov = 50.0f;

    if (camera_layer != nullptr) {
        camera_name = camera_layer->camera_name.empty() ? "global" : camera_layer->camera_name;
        camera_node = camera_layer->node;
        zoom = camera_layer->zoom;
        fov = camera_layer->fov;
    }

    auto camera_it = cameras.find(camera_name);
    if (camera_it == cameras.end() || !camera_it->second) {
        LOG_ERROR("SceneCameraLayer: target camera '%s' for layer=%d is missing",
                  camera_name.c_str(),
                  next_layer_id);
        camera_it = cameras.find("global");
        camera_name = "global";
    }
    if (camera_it == cameras.end() || !camera_it->second || !camera_node) return;

    camera_it->second->AttatchNode(camera_node);
    ApplyCameraProjectionState(*this,
                               camera_name,
                               *camera_it->second,
                               zoom,
                               fov,
                               next_layer_id);
    activeCamera = camera_it->second.get();

    if (activeCameraLayerId != next_layer_id) {
        // This transition log is intentionally sparse: it proves which authored camera layer owns
        // the view without flooding frame logs while keyframed zoom/origin values animate.
        LOG_INFO("SceneCameraLayerActive: previous=%d active=%d camera='%s' zoom=%.3f origin=[%.3f, %.3f, %.3f]",
                 activeCameraLayerId,
                 next_layer_id,
                 camera_name.c_str(),
                 zoom,
                 camera_layer != nullptr ? camera_layer->origin[0] : 0.0f,
                 camera_layer != nullptr ? camera_layer->origin[1] : 0.0f,
                 camera_layer != nullptr ? camera_layer->origin[2] : 0.0f);
        activeCameraLayerId = next_layer_id;
    }
}

SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id, uint32_t effect_index) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr || effect_index >= effect_layer->EffectCount()) continue;
        return effect_layer->GetEffect(effect_index).get();
    }

    return nullptr;
}

const SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id,
                                               uint32_t effect_index) const {
    return const_cast<Scene*>(this)->FindImageEffect(owner_layer_id, effect_index);
}

SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id, int32_t effect_id) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr) continue;

        for (std::size_t effect_index = 0; effect_index < effect_layer->EffectCount();
             effect_index++) {
            auto& effect = effect_layer->GetEffect(effect_index);
            if (effect != nullptr && effect->EffectId() == effect_id) return effect.get();
        }
    }

    return nullptr;
}

const SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id,
                                                   int32_t effect_id) const {
    return const_cast<Scene*>(this)->FindImageEffectById(owner_layer_id, effect_id);
}

bool Scene::SetEffectLocalVisibility(int32_t owner_layer_id, uint32_t effect_index,
                                     bool visible) {
    auto* effect = FindImageEffect(owner_layer_id, effect_index);
    if (effect == nullptr) return false;

    // Only the effect-local bit changes here. The render graph topology remains valid because
    // hidden effects are handled by conditional execution and a bypass copy, while layer visibility
    // propagation still owns parent/child effective visibility.
    effect->SetLocalVisible(visible);
    ApplyLayerVisibility(owner_layer_id);
    return true;
}

bool Scene::SetEffectLocalVisibilityById(int32_t owner_layer_id, int32_t effect_id,
                                         bool visible) {
    auto* effect = FindImageEffectById(owner_layer_id, effect_id);
    if (effect == nullptr) return false;

    effect->SetLocalVisible(visible);
    ApplyLayerVisibility(owner_layer_id);
    return true;
}

}

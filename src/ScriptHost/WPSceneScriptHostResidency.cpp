#include "WPSceneScriptHostShared.hpp"

// Layer GPU-resource ownership bookkeeping for the scene script host. Dynamic deletion releases
// only resources no surviving layer still references.

#include "Scene/Scene.h"
#include "Scene/include/Scene/SceneImageEffectLayer.h"
#include "Scene/include/Scene/SceneMaterial.h"
#include "Scene/include/Scene/SceneMesh.h"
#include "Scene/include/Scene/SceneNode.h"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace wallpaper
{

void MergeResidencyResources(LayerResidencyResources& target,
                             const LayerResidencyResources& source) {
    target.static_textures.insert(source.static_textures.begin(), source.static_textures.end());
    target.video_textures.insert(source.video_textures.begin(), source.video_textures.end());
    target.render_targets.insert(source.render_targets.begin(), source.render_targets.end());
}

template<typename Callback>
void VisitLayerTree(const Scene& scene, int32_t root_layer_id, Callback&& callback) {
    std::vector<int32_t>        stack { root_layer_id };
    std::unordered_set<int32_t> visited;

    while (!stack.empty()) {
        const auto layer_id = stack.back();
        stack.pop_back();
        if (layer_id == 0 || !visited.insert(layer_id).second) continue;

        callback(layer_id);

        for (const auto child_id : scene.GetLayerChildren(layer_id)) {
            stack.push_back(child_id);
        }
    }
}

void PushUniqueResidencyNode(SceneNode* node, std::vector<SceneNode*>& nodes,
                             std::unordered_set<SceneNode*>& seen) {
    if (node == nullptr || !seen.insert(node).second) return;
    nodes.push_back(node);
}

void CollectLayerEffectResidencyNodes(const Scene& scene, int32_t layer_id,
                                      std::vector<SceneNode*>& nodes,
                                      std::unordered_set<SceneNode*>& seen) {
    auto* effect_layer = const_cast<Scene&>(scene).FindImageEffectLayer(layer_id);
    if (effect_layer == nullptr) return;

    if (effect_layer->HasFinalComposite()) {
        PushUniqueResidencyNode(&effect_layer->FinalNode(), nodes, seen);
    }

    for (size_t effect_index = 0; effect_index < effect_layer->EffectCount(); effect_index++) {
        auto& effect = effect_layer->GetEffect(effect_index);
        if (!effect) continue;
        for (auto& effect_node : effect->nodes) {
            PushUniqueResidencyNode(effect_node.sceneNode.get(), nodes, seen);
        }
    }
}

std::vector<SceneNode*> CollectLayerResidencyNodes(const Scene& scene, int32_t layer_id) {
    std::vector<SceneNode*> nodes;
    std::unordered_set<SceneNode*> seen;

    for (auto* node : scene.GetLayerRuntimeNodes(layer_id)) {
        PushUniqueResidencyNode(node, nodes, seen);
    }
    CollectLayerEffectResidencyNodes(scene, layer_id, nodes, seen);
    return nodes;
}

void CollectResidencyTextureKey(const Scene& scene, const std::string& key,
                                LayerResidencyResources& resources) {
    if (key.empty() || IsSpecLinkTex(key) || key == SpecTex_Default) return;

    if (scene.renderTargets.count(key) != 0 || IsSpecTex(key)) {
        resources.render_targets.insert(key);
        return;
    }

    if (auto texture_it = scene.textures.find(key);
        texture_it != scene.textures.end() && texture_it->second.isVideo) {
        resources.video_textures.insert(key);
        return;
    }

    resources.static_textures.insert(key);
}

void CollectResidencyMaterialResources(const Scene& scene, const SceneMaterial& material,
                                       LayerResidencyResources& resources) {
    for (const auto& key : material.textures) {
        CollectResidencyTextureKey(scene, key, resources);
    }
}

void CollectResidencyNodeResources(const Scene& scene, SceneNode* node,
                                   LayerResidencyResources& resources) {
    if (node == nullptr) return;

    if (auto* mesh = node->Mesh(); mesh != nullptr && mesh->Material() != nullptr) {
        CollectResidencyMaterialResources(scene, *mesh->Material(), resources);
    }

    if (auto* text = node->Text(); text != nullptr) {
        for (const auto& page : text->layout.glyph_pages) {
            if (!page.texture_key.empty()) resources.static_textures.insert(page.texture_key);
        }
    }
}

LayerResidencyResources CollectLayerResidencyResources(const Scene& scene, int32_t layer_id) {
    LayerResidencyResources resources;

    for (auto* node : CollectLayerResidencyNodes(scene, layer_id)) {
        CollectResidencyNodeResources(scene, node, resources);
    }

    if (const auto* effect_layer = scene.FindImageEffectLayer(layer_id)) {
        for (const auto& key : effect_layer->RuntimeRenderTargetNames()) {
            if (!key.empty() && key != SpecTex_Default) resources.render_targets.insert(key);
        }
    }

    return resources;
}

LayerResidencyResources CollectRetainedResidencyResources(
    const Scene& scene, const std::unordered_set<int32_t>& excluded_layers = {}) {
    LayerResidencyResources resources;
    std::unordered_set<int32_t> visited_layers;

    // Iterate the identity map and skip layers without live draw handles: the former
    // objectRuntimeNodes map only ever held non-empty entries, so this reproduces its key set.
    for (const auto& [layer_id, object] : scene.sceneObjects) {
        if (object == nullptr || object->RuntimeNodes().empty()) continue;
        if (!visited_layers.insert(layer_id).second) continue;
        if (excluded_layers.count(layer_id) != 0) continue;
        MergeResidencyResources(resources, CollectLayerResidencyResources(scene, layer_id));
    }

    for (const auto& node : scene.bloom.nodes) {
        if (node) CollectResidencyNodeResources(scene, node.get(), resources);
    }
    if (scene.bloom.node) CollectResidencyNodeResources(scene, scene.bloom.node.get(), resources);

    return resources;
}

void QueueLayerResourceRelease(Scene& scene, int32_t layer_id,
                               const LayerResidencyResources& retained_resources,
                               const char* reason) {
    const auto resources = CollectLayerResidencyResources(scene, layer_id);
    std::size_t queued_static = 0;
    std::size_t queued_video = 0;
    std::size_t queued_render_targets = 0;

    for (const auto& key : resources.static_textures) {
        if (retained_resources.static_textures.count(key) != 0) continue;
        queued_static += scene.pendingStaticTextureReleaseKeys.insert(key).second ? 1 : 0;
    }
    for (const auto& key : resources.video_textures) {
        if (retained_resources.video_textures.count(key) != 0) continue;
        queued_video += scene.pendingVideoTextureReleaseKeys.insert(key).second ? 1 : 0;
    }
    for (const auto& key : resources.render_targets) {
        if (retained_resources.render_targets.count(key) != 0) continue;
        queued_render_targets += scene.pendingRenderTargetReleaseKeys.insert(key).second ? 1 : 0;
    }

    if (queued_static != 0 || queued_video != 0 || queued_render_targets != 0) {
        LOG_INFO("SceneResidencyQueueRelease: reason=%s layer=%d static=%zu video=%zu "
                 "render-target=%zu",
                 reason != nullptr ? reason : "unknown",
                 layer_id,
                 queued_static,
                 queued_video,
                 queued_render_targets);
    }
}

} // namespace wallpaper

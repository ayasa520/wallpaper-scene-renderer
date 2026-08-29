#include "WPSceneScriptHostShared.hpp"

// Layer GPU-residency bookkeeping and deferred-layer materialization for the scene script
// host: collecting the textures/render targets a layer tree holds resident, queueing and
// cancelling hidden-layer resource release, and materializing deferred image/particle/text
// layers when scripts or visibility first need them.

#include "Scene/Scene.h"
#include "Scene/include/Scene/SceneImageEffectLayer.h"
#include "Scene/include/Scene/SceneMaterial.h"
#include "Scene/include/Scene/SceneMesh.h"
#include "Scene/include/Scene/SceneNode.h"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include "WPSceneParser.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace wallpaper
{

bool MaterializeDeferredParticleLayerIfNeeded(WPSceneScriptHost::Opaque* opaque, int32_t layer_id) {
    if (opaque == nullptr || opaque->scene == nullptr) return false;
    if (! opaque->scene->IsLayerDeferredRuntime(layer_id, SceneDeferredRuntimeKind::Particle)) {
        return true;
    }

    const auto registration_range = CaptureSceneRegistrationRange(opaque);

    if (! wallpaper::MaterializeDeferredParticleLayer(
            *opaque->scene, layer_id, &opaque->user_properties)) {
        LOG_ERROR("DeferredRuntimeParticleRealize: failed for layer=%d", layer_id);
        return false;
    }

    auto* node = FindNodeById(opaque, layer_id);
    if (node == nullptr) {
        LOG_ERROR("DeferredRuntimeParticleRealize: missing realized node for layer=%d", layer_id);
        return false;
    }

    RebindLayerRegistrations(opaque, layer_id, node);
    RegisterSceneRegistrationRange(opaque, registration_range);
    EnsureTextureAnimationStatesForNode(opaque, node);
    // Realizing a deferred particle layer inserts the actual runtime scene node and the particle
    // render work that did not exist when the graph was compiled with only the hidden placeholder.
    // A resource refresh cannot add that missing pass, so the first false->true visibility toggle
    // must force a topology rebuild before the layer can become visible on screen.
    opaque->scene->MarkRenderGraphTopologyDirty();
    return true;
}

bool MaterializeDeferredImageLayerIfNeeded(WPSceneScriptHost::Opaque* opaque, int32_t layer_id) {
    if (opaque == nullptr || opaque->scene == nullptr) return false;
    if (! opaque->scene->IsLayerDeferredRuntime(layer_id, SceneDeferredRuntimeKind::Image)) {
        return true;
    }

    const auto registration_range = CaptureSceneRegistrationRange(opaque);

    if (! wallpaper::MaterializeDeferredImageLayer(
            *opaque->scene, layer_id, &opaque->user_properties)) {
        LOG_ERROR("DeferredRuntimeImageRealize: failed for layer=%d", layer_id);
        return false;
    }

    auto* node = FindNodeById(opaque, layer_id);
    if (node == nullptr) {
        LOG_ERROR("DeferredRuntimeImageRealize: missing realized node for layer=%d", layer_id);
        return false;
    }

    RebindLayerRegistrations(opaque, layer_id, node);
    RegisterSceneRegistrationRange(opaque, registration_range);
    EnsureTextureAnimationStatesForNode(opaque, node);
    // Deferred image layers are the expensive case for multilingual scenes: the hidden placeholder
    // had no material/effect passes, so turning it visible changes graph topology and must rebuild
    // before any newly-created render targets or pipelines can contribute to the frame.
    opaque->scene->MarkRenderGraphTopologyDirty();
    LOG_INFO("DeferredRuntimeImageRealize: materialized layer=%d topology-dirty=true", layer_id);
    return true;
}

bool MaterializeDeferredTextLayerIfNeeded(WPSceneScriptHost::Opaque* opaque, int32_t layer_id) {
    if (opaque == nullptr || opaque->scene == nullptr) return false;
    if (! opaque->scene->IsLayerDeferredRuntime(layer_id, SceneDeferredRuntimeKind::Text)) {
        return true;
    }

    const auto registration_range = CaptureSceneRegistrationRange(opaque);

    if (! wallpaper::MaterializeDeferredTextLayer(
            *opaque->scene, layer_id, &opaque->user_properties)) {
        LOG_ERROR("DeferredRuntimeTextRealize: failed for layer=%d", layer_id);
        return false;
    }

    auto* node = FindNodeById(opaque, layer_id);
    if (node == nullptr) {
        LOG_ERROR("DeferredRuntimeTextRealize: missing realized node for layer=%d", layer_id);
        return false;
    }

    RebindLayerRegistrations(opaque, layer_id, node);
    RegisterSceneRegistrationRange(opaque, registration_range);
    EnsureTextureAnimationStatesForNode(opaque, node);
    // Deferred text materialization follows the same contract as particles: the render graph built
    // while a hidden placeholder was present cannot draw the newly inserted text node until its
    // topology is rebuilt, even if ordinary uniforms or textures would only need a resource
    // refresh.
    opaque->scene->MarkRenderGraphTopologyDirty();
    LOG_INFO("DeferredRuntimeTextRealize: materialized layer=%d topology-dirty=true", layer_id);
    return true;
}

bool MaterializeDeferredVisibleLayerTreeIfNeeded(WPSceneScriptHost::Opaque* opaque,
                                                 int32_t                    root_layer_id) {
    if (opaque == nullptr || opaque->scene == nullptr || root_layer_id == 0) return false;

    std::vector<int32_t>        stack { root_layer_id };
    std::unordered_set<int32_t> visited;
    while (! stack.empty()) {
        const int32_t layer_id = stack.back();
        stack.pop_back();
        if (layer_id == 0 || ! visited.insert(layer_id).second) continue;

        if (! opaque->scene->IsLayerVisible(layer_id)) {
            // Deferred descendants inherit effective visibility from their parent chain. If this
            // layer is still effectively hidden after the current local-visible write, none of its
            // children can be visible either, so keep that branch logical and cheap.
            continue;
        }

        if (! MaterializeDeferredImageLayerIfNeeded(opaque, layer_id)) return false;
        if (! MaterializeDeferredParticleLayerIfNeeded(opaque, layer_id)) return false;
        if (! MaterializeDeferredTextLayerIfNeeded(opaque, layer_id)) return false;

        for (const auto child_id : opaque->scene->GetLayerChildren(layer_id)) {
            stack.push_back(child_id);
        }
    }

    return true;
}

bool RetainsGpuResidencyWhileHidden(const Scene& scene, int32_t layer_id) {
    if (layer_id == 0) return true;
    if (scene.IsLayerVisible(layer_id)) return true;
    return scene.IsLayerOffscreenDependencySource(layer_id);
}

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
        if (!RetainsGpuResidencyWhileHidden(scene, layer_id)) continue;

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

void QueueHiddenLayerTreeResourceRelease(WPSceneScriptHost::Opaque* opaque,
                                         int32_t                    root_layer_id) {
    if (opaque == nullptr || opaque->scene == nullptr || root_layer_id == 0) return;

    auto& scene = *opaque->scene;
    const auto retained_resources = CollectRetainedResidencyResources(scene);

    std::size_t queued_static = 0;
    std::size_t queued_video = 0;
    std::size_t queued_render_targets = 0;

    VisitLayerTree(scene, root_layer_id, [&](int32_t layer_id) {
        if (!RetainsGpuResidencyWhileHidden(scene, layer_id)) {
            const auto static_before = scene.pendingStaticTextureReleaseKeys.size();
            const auto video_before = scene.pendingVideoTextureReleaseKeys.size();
            const auto render_target_before = scene.pendingRenderTargetReleaseKeys.size();
            QueueLayerResourceRelease(scene, layer_id, retained_resources, "hidden");
            queued_static += scene.pendingStaticTextureReleaseKeys.size() - static_before;
            queued_video += scene.pendingVideoTextureReleaseKeys.size() - video_before;
            queued_render_targets +=
                scene.pendingRenderTargetReleaseKeys.size() - render_target_before;
        }
    });

    if (queued_static != 0 || queued_video != 0 || queued_render_targets != 0) {
        LOG_INFO("SceneResidencyQueueRelease: root-layer=%d static=%zu video=%zu render-target=%zu",
                 root_layer_id,
                 queued_static,
                 queued_video,
                 queued_render_targets);
    }
}

void CancelLayerResourceRelease(Scene& scene, int32_t layer_id, const char* reason,
                                std::size_t& cancelled_static,
                                std::size_t& cancelled_video,
                                std::size_t& cancelled_render_targets) {
    const auto resources = CollectLayerResidencyResources(scene, layer_id);
    std::size_t layer_static = 0;
    std::size_t layer_video = 0;
    std::size_t layer_render_targets = 0;
    for (const auto& key : resources.static_textures) {
        layer_static += scene.pendingStaticTextureReleaseKeys.erase(key);
    }
    for (const auto& key : resources.video_textures) {
        layer_video += scene.pendingVideoTextureReleaseKeys.erase(key);
    }
    for (const auto& key : resources.render_targets) {
        layer_render_targets += scene.pendingRenderTargetReleaseKeys.erase(key);
    }

    cancelled_static += layer_static;
    cancelled_video += layer_video;
    cancelled_render_targets += layer_render_targets;

    if (layer_static != 0 || layer_video != 0 || layer_render_targets != 0) {
        LOG_INFO("SceneResidencyCancelRelease: reason=%s layer=%d static=%zu video=%zu "
                 "render-target=%zu",
                 reason != nullptr ? reason : "unknown",
                 layer_id,
                 layer_static,
                 layer_video,
                 layer_render_targets);
    }
}

void CancelLayerTreeResourceRelease(WPSceneScriptHost::Opaque* opaque,
                                    int32_t                    root_layer_id) {
    if (opaque == nullptr || opaque->scene == nullptr || root_layer_id == 0) return;

    auto& scene = *opaque->scene;
    std::size_t cancelled_static = 0;
    std::size_t cancelled_video = 0;
    std::size_t cancelled_render_targets = 0;

    VisitLayerTree(scene, root_layer_id, [&](int32_t layer_id) {
        // Visibility can flip more than once before the render thread drains pending releases. When
        // a hidden branch becomes visible again in the same frame, its resources are retained by the
        // new graph and must be removed from the eviction queue before Vulkan sees them. This is the
        // same generation-cancel idea mature engines use for streamed residency requests, just
        // scoped to the synchronous render-thread queue we already own.
        CancelLayerResourceRelease(scene,
                                   layer_id,
                                   "visible",
                                   cancelled_static,
                                   cancelled_video,
                                   cancelled_render_targets);
    });
}


} // namespace wallpaper

#include "WPSceneScriptHostShared.hpp"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include "Scene/Scene.h"
#include "Scene/SceneModelData.h"
#include "Scene/SceneNode.h"
#include "Utils/Logging.h"
#include "WPSceneParser.hpp"
#include "WPShaderValueUpdater.hpp"

namespace wallpaper
{
namespace
{
size_t RetireMaterialRegistrations(WPSceneScriptHost::Opaque& opaque,
                                    const std::unordered_set<SceneMaterial*>& materials) {
    const auto retired_registration = [&](const WPSceneScriptRegistration& registration) {
        return registration.target_kind == WPSceneScriptTargetKind::MaterialUniform &&
               materials.contains(registration.material);
    };
    // Snapshot instances outside dispatch. Destroy callbacks can append scripted layers, so do
    // not traverse a mutating registry while freeing them. Keep each old identity discoverable
    // until its destroy callback returns: thisObject queries still address that live material.
    // The old registrations and materials are released before any subsequent frame dispatch.
    std::vector<ScriptInstance*> retired_instances;
    std::unordered_set<uint32_t> retired_ids;
    for (const auto& instance : opaque.instances) {
        if (! retired_registration(instance->registration)) continue;
        retired_ids.insert(instance->instance_id);
        retired_instances.push_back(instance.get());
    }
    std::erase_if(opaque.property_bindings, retired_registration);
    std::erase_if(opaque.property_animations, [&](const PropertyAnimationInstance& animation) {
        return retired_registration(animation.registration);
    });
    std::erase_if(opaque.scene->bindingRegistrations, retired_registration);
    std::erase_if(opaque.scene->scriptRegistrations, retired_registration);
    std::erase_if(opaque.scene->propertyAnimationRegistrations, retired_registration);

    for (auto* instance : retired_instances) {
        FreeScriptInstance(opaque.runtime.context, *instance);
        opaque.hovered_instances.erase(instance->instance_id);
        opaque.pressed_instances.erase(instance->instance_id);
    }
    std::erase_if(opaque.instances, [&](const std::unique_ptr<ScriptInstance>& instance) {
        return retired_ids.contains(instance->instance_id);
    });
    // Include timers authored by a destroy callback itself. The instance has ended, so those
    // closures must not outlive the material while ordinary layer-owned timers keep running.
    std::erase_if(opaque.timers, [&](ScriptTimer& timer) {
        if (! retired_ids.contains(timer.owner_instance_id)) return false;
        JS_FreeValue(opaque.runtime.context, timer.callback);
        timer.callback = JS_UNDEFINED;
        return true;
    });
    return retired_ids.size();
}
} // namespace

void ProcessPendingSceneModelRefresh(WPSceneScriptHost::Opaque& opaque) {
    if (opaque.pending_model_refresh_tokens.empty()) return;
    const auto pending = std::exchange(opaque.pending_model_refresh_tokens, {});
    auto& scene = *opaque.scene;
    auto& updater = *static_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());

    // Snapshot identities, not map iterators: material destruction/init can create owners and
    // rehash the identity map. Newly created owners already materialize the current revision.
    // A callback can also request another replacement; it stays in the next bounded batch
    // rather than recursively destroying a newly executing script or looping until quiescent.
    const auto layer_ids = scene.layerOrder;
    std::vector<std::pair<int32_t, LayerResidencyResources>> retired_resources;
    for (const auto layer_id : layer_ids) {
        auto* owner = scene.FindSceneObject(layer_id);
        if (owner == nullptr || ! owner->ModelData()) continue;
        const auto model = owner->ModelData();
        if (! pending.contains(model->Token()) ||
            owner->ModelDataRevision() == model->StructureRevision()) continue;

        auto* root = owner->LayerNode();
        const auto previous_revision = owner->ModelDataRevision();
        retired_resources.emplace_back(layer_id, CollectLayerResidencyResources(scene, layer_id));
        std::unordered_set<SceneNode*> chunk_nodes;
        std::unordered_set<SceneMaterial*> materials;
        for (auto* node : owner->RuntimeNodes()) {
            if (node == root) continue;
            chunk_nodes.insert(node);
            if (node->Mesh() != nullptr) materials.insert(node->Mesh()->Material());
        }
        const auto retired_scripts = RetireMaterialRegistrations(opaque, materials);

        // Keep old chunks alive until all raw material/node registrations have been removed.
        // Authored child layers attached to the model root are not chunk resources and retain
        // their placement, scripts, visibility, ordering and input capture throughout refresh.
        std::vector<std::shared_ptr<SceneNode>> retired_chunks;
        for (const auto& child : root->GetChildren()) {
            if (chunk_nodes.contains(child.get())) retired_chunks.push_back(child);
        }
        for (const auto& chunk : retired_chunks) root->RemoveChild(chunk.get());
        for (auto* node : chunk_nodes) {
            updater.RemoveDrawData(node);
            opaque.texture_states.erase(node);
        }
        scene.ClearLayerRuntimeNodes(layer_id);
        scene.AddLayerRuntimeNode(layer_id, root);

        const auto range = CaptureSceneRegistrationRange(&opaque);
        if (! RefreshGeneratedSceneModelLayer(scene, layer_id, &opaque.user_properties)) {
            LOG_ERROR("SceneModelRefresh: layer=%d token=%u materialization failed",
                      layer_id, model->Token());
        }
        scene.MarkRenderGraphTopologyDirty();
        RegisterSceneRegistrationRange(&opaque, range);
        if (std::getenv("WESCENE_TRACE_MODEL_DATA") != nullptr) {
            LOG_INFO("SceneModelRefresh: layer=%d token=%u root=%p revision=%llu->%llu "
                     "retired-chunks=%zu chunks=%zu retired-scripts=%zu parent=%d",
                     layer_id, model->Token(), static_cast<void*>(root),
                     static_cast<unsigned long long>(previous_revision),
                     static_cast<unsigned long long>(owner->ModelDataRevision()),
                     retired_chunks.size(), owner->RuntimeNodes().size() - 1,
                     retired_scripts, owner->ParentId());
        }
    }

    // Descriptor retirement is decided after all referenced owners have refreshed, so a texture
    // still used by another owner or by a newly initialized material remains resident. The
    // existing topology rebuild owns GPU submission synchronization and allocation retirement.
    if (! retired_resources.empty()) {
        // New chunks were appended without moving authored children. Restore the canonical
        // layer order after the whole batch, preserving transparent-chunk order within an owner.
        ResortSceneLayerTree(opaque);
        const auto retained = CollectRetainedResidencyResources(scene, {});
        for (const auto& [layer_id, resources] : retired_resources) {
            QueueLayerResourceRelease(scene, layer_id, resources, retained, "model-replace");
        }
    }
}
} // namespace wallpaper

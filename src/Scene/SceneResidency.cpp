#include "SceneResidency.h"

#include "Scene.h"
#include "SceneImageEffectLayer.h"
#include "SceneImageSource.h"
#include "SceneMaterial.h"
#include "SceneMesh.h"
#include "SceneNode.h"
#include "SpecTexs.hpp"

namespace wallpaper
{
namespace
{
void MergeResources(LayerResidencyResources& target, const LayerResidencyResources& source) {
    target.static_textures.insert(source.static_textures.begin(), source.static_textures.end());
    target.video_textures.insert(source.video_textures.begin(), source.video_textures.end());
    target.render_targets.insert(source.render_targets.begin(), source.render_targets.end());
}

void CollectTexture(const Scene& scene, const std::string& key,
                    LayerResidencyResources& resources) {
    if (key.empty() || key == SpecTex_Default) return;
    if (scene.renderTargets.contains(key) || IsSpecTex(key)) {
        resources.render_targets.insert(key);
    } else if (const auto texture = scene.textures.find(key);
               texture != scene.textures.end() && texture->second.isVideo) {
        resources.video_textures.insert(key);
    } else {
        resources.static_textures.insert(key);
    }
}

void CollectMaterial(const Scene& scene, const SceneMaterial& material,
                     LayerResidencyResources& resources) {
    // Resolve every retained slot using the same precedence as the draw. An inactive authored
    // input is not an additional reference: selecting it again loads that image normally.
    for (usize slot = 0; slot < material.textures.size(); ++slot) {
        CollectTexture(scene, material.Texture(slot), resources);
    }
}

void CollectMesh(const Scene& scene, const SceneMesh* mesh, LayerResidencyResources& resources) {
    if (mesh == nullptr) return;
    if (const auto* material = mesh->Material()) CollectMaterial(scene, *material, resources);
    // A source wrapper can retain mask geometry while another wrapper is selected for drawing.
    // These per-group images are draw inputs even though they are outside the primary material's
    // texture array. Keep their lifetime coupled to the shared mesh payload that consumes them.
    const auto& masks = mesh->MaskedDraw();
    for (const auto& group : masks.groups) CollectTexture(scene, group.maskTexture, resources);
    if (masks.materials) {
        CollectMaterial(scene, masks.materials->mask, resources);
        CollectMaterial(scene, masks.materials->clipped, resources);
    }
}

void CollectNode(const Scene& scene, const SceneNode* node, LayerResidencyResources& resources) {
    if (node == nullptr) return;
    CollectMesh(scene, node->Mesh(), resources);
    if (const auto* text = node->Text()) {
        if (text->bridge.framebuffer_source) {
            CollectMesh(scene, text->bridge.framebuffer_source->Mesh(), resources);
        }
        for (const auto& page : text->layout.glyph_pages) {
            if (!page.texture_key.empty()) resources.static_textures.insert(page.texture_key);
        }
    }
}
} // namespace

LayerResidencyResources CollectLayerResidencyResources(const Scene& scene, int32_t layer_id) {
    LayerResidencyResources resources;
    const auto* owner = scene.FindSceneObject(layer_id);
    if (owner == nullptr) return resources;
    for (const auto* node : owner->RuntimeNodes()) CollectNode(scene, node, resources);

    // Image setup retains the authored material separately from the active draw mesh. Private
    // publication, prelighting and effect visibility can replace that mesh without ending the
    // source's lifetime. Enumerate all owned variants instead of inferring ownership from the
    // visible command list or requiring a live node on the owner.
    if (const auto& source = owner->ImageSource()) {
        CollectMaterial(scene, source->Material(), resources);
        CollectMesh(scene, &source->SourceMesh(), resources);
    }
    if (const auto& layer = owner->ImageEffectLayer()) {
        CollectMesh(scene, &layer->SourceMesh(), resources);
        CollectMesh(scene, &layer->FinalMesh(), resources);
        CollectMesh(scene, layer->FinalCompositeDraw().Mesh(), resources);
        if (const auto* prelighting = layer->GetPrelightingSource()) {
            CollectMesh(scene, prelighting->mesh.get(), resources);
        }
        for (size_t index = 0; index < layer->EffectCount(); ++index) {
            for (const auto& node : layer->GetEffect(index)->nodes) {
                CollectNode(scene, node.sceneNode.get(), resources);
            }
        }
        for (const auto& key : layer->RuntimeRenderTargetNames()) {
            if (!key.empty() && key != SpecTex_Default) resources.render_targets.insert(key);
        }
    }
    return resources;
}

LayerResidencyResources CollectRetainedResidencyResources(
    const Scene& scene, const std::unordered_set<int32_t>& excluded_layers) {
    LayerResidencyResources resources;
    for (const auto& [layer_id, owner] : scene.sceneObjects) {
        if (owner == nullptr || excluded_layers.contains(layer_id)) continue;
        MergeResources(resources, CollectLayerResidencyResources(scene, layer_id));
    }
    for (const auto& node : scene.bloom.nodes) CollectNode(scene, node.get(), resources);
    CollectNode(scene, scene.bloom.node.get(), resources);
    // Deletion callbacks can retire the originating owner before its requested clear is
    // submitted. The scene queue is an independent target owner until submission consumes it.
    for (const auto& request : scene.PendingRenderTargetClears()) {
        resources.render_targets.insert(request.target);
    }
    return resources;
}

void QueueReplacedImportedTextures(Scene& scene, const LayerResidencyResources& previous) {
    // A binding change nominates previously resolved imported inputs. Final retention is checked
    // after property/script callbacks and graph replacement, because another owner can acquire a
    // candidate during the same batch. Named targets retain their destination-owner lifetime;
    // changing a reader does not retire the target's writer or its history.
    scene.pendingStaticTextureReleaseKeys.insert(
        previous.static_textures.begin(), previous.static_textures.end());
    scene.pendingVideoTextureReleaseKeys.insert(
        previous.video_textures.begin(), previous.video_textures.end());
}
} // namespace wallpaper

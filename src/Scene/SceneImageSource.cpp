#include "SceneImageSource.h"

#include "Scene.h"
#include "SceneCamera.h"
#include "SceneImageEffectLayer.h"
#include "SceneMaterial.h"
#include "SceneMesh.h"
#include "SceneNode.h"
#include "SceneObject.h"
#include "SpecTexs.hpp"
#include "WPImageAlignment.hpp"
#include "Utils/Logging.h"

namespace wallpaper
{
SceneImageSource::SceneImageSource(SceneObject& owner, std::shared_ptr<SceneNode> node,
                                   const SceneMesh& source_mesh, Policy policy,
                                   std::optional<Metadata> metadata)
    : m_owner(owner), m_node(std::move(node)), m_material(source_mesh.SharedMaterial()),
      m_source_mesh(std::make_unique<SceneMesh>()), m_policy(policy),
      m_metadata(metadata ? std::move(*metadata) : Metadata {}) {
    m_source_mesh->ChangeMeshDataFrom(source_mesh);
}

SceneImageSource::~SceneImageSource() = default;

std::optional<SceneImageSource::Metadata> SceneImageSource::ResolveMetadata(
    const Scene& scene, std::string_view name) {
    Metadata result;
    result.texture_key = name;
    if (const auto it = scene.textures.find(result.texture_key); it != scene.textures.end()) {
        const auto& texture = it->second;
        result.allocation_size = { texture.width, texture.height };
        result.content_size = texture.isSprite
            ? std::array { texture.spriteAnim.Frames().front().width,
                           texture.spriteAnim.Frames().front().height }
            : std::array { static_cast<float>(texture.mapWidth),
                           static_cast<float>(texture.mapHeight) };
        result.sample = texture.sample;
        result.sprite = texture.isSprite;
    } else if (const auto it = scene.renderTargets.find(result.texture_key);
               it != scene.renderTargets.end()) {
        const auto& target = it->second;
        result.allocation_size = { target.width, target.height };
        result.content_size = { static_cast<float>(target.ContentWidth()),
                                static_cast<float>(target.ContentHeight()) };
        result.sample = target.sample;
    } else {
        return std::nullopt;
    }
    result.display_size = result.content_size;
    return result;
}

std::optional<SceneImageSource::Metadata> SceneImageSource::ResolveMetadata(
    const Scene& scene, const SceneMaterial& material) {
    if (material.textures.empty()) return std::nullopt;
    auto result = ResolveMetadata(scene, material.Texture(0));
    if (!result) return result;

    // System replacements carry a material reference size independently of the primary
    // texture's sampling metadata. Resolve active slots in authored order: the last successful
    // replacement supplies the material size. An absent system value has no size override, and
    // a keepAspect binding deliberately uses its selected image's dimensions. Retaining both
    // sizes in this snapshot lets autosize update display geometry without resizing the private
    // texture-space card, changing source UVs, or losing same-key resource refresh detection.
    for (usize slot = 0; slot < material.textures.size(); ++slot) {
        const auto reference = material.systemTextureReferences.find(slot);
        if (reference == material.systemTextureReferences.end()) continue;
        const auto binding = material.systemTextureBindings.find(slot);
        if (binding == material.systemTextureBindings.end() || binding->second->empty()) continue;
        const auto selected = ResolveMetadata(scene, material.Texture(slot));
        if (!selected) continue;
        result->display_size = reference->second.keepAspect
            ? std::array {
                static_cast<float>(static_cast<uint16_t>(static_cast<int32_t>(selected->content_size[0]))),
                static_cast<float>(static_cast<uint16_t>(static_cast<int32_t>(selected->content_size[1]))),
            }
            : reference->second.size;
    }
    return result;
}

std::string_view SceneImageSource::TextureName() const {
    return m_material->textures.empty() ? std::string_view {} : m_material->Texture(0);
}

void SceneImageSource::ApplyAutosize(Scene& scene, const Metadata& metadata) {
    if (!m_policy.autosize || metadata.sprite) return;
    auto& state = *m_owner.ImageRuntimeState();
    const auto& size = metadata.display_size;
    if (state.size == size) return;

    LOG_INFO("SceneImageAutosize: layer=%d texture='%s' previous=[%.3f %.3f] "
             "size=[%.3f %.3f] allocation=[%d %d] content=[%.3f %.3f]",
             m_owner.Id(), metadata.texture_key.c_str(), state.size[0], state.size[1],
             size[0], size[1], metadata.allocation_size[0], metadata.allocation_size[1],
             metadata.content_size[0], metadata.content_size[1]);
    // Generated display cards share these bytes with their direct/source consumers. Their
    // logical size does not resize the private texture-space card or an imported mesh, and the
    // authored origin remains the pivot while alignment follows the new display dimensions.
    const float half_width = size[0] * 0.5f;
    const float half_height = size[1] * 0.5f;
    const auto resize_card = [&](SceneMesh& mesh) {
        mesh.GetVertexArray(0).SetVertex(WE_IN_POSITION, std::array {
            -half_width, half_height, 0.0f, -half_width, -half_height, 0.0f,
            half_width, half_height, 0.0f, half_width, -half_height, 0.0f,
        });
        mesh.SetDirty();
    };
    resize_card(*m_source_mesh);
    // Static passes consult the drawable wrapper's dirty flag before uploading shared payloads.
    // Invalidating only the retained source wrapper updates CPU readback but leaves a direct
    // image drawing its old GPU vertices. Dynamic consumers still use the shared revision.
    m_node->Mesh()->SetDirty();
    state.size = size;
    m_node->SetAlignmentOffset(ResolveImageAlignmentOffset(state.alignment, size));
    if (const auto& layer = m_owner.ImageEffectLayer()) {
        layer->SetCardSize(size);
        resize_card(layer->FinalMesh());
        layer->SyncResolvedOutputMesh();
        // The private prelighting projection is refreshed from destination resources by the
        // effect bridge. An ordinary source projection instead spans the resized display card.
        if (!layer->UsesPrelightingSource()) {
            scene.cameras.at(layer->BridgeCameraName())->SetOrthographicViewRect(
                -half_width, half_width, -half_height, half_height);
        }
    }
    scene.MarkRenderGraphResourcesDirty();
}

void SceneImageSource::CompleteInitialTexture(Scene& scene) {
    if (auto metadata = ResolveMetadata(scene, *m_material)) {
        // Forward references complete after every owner has materialized. Apply their initial
        // logical size before destination setup remembers this descriptor as the applied state.
        ApplyAutosize(scene, *metadata);
        m_metadata = std::move(*metadata);
    }
}

void SceneImageSource::Refresh(Scene& scene) {
    if (!m_policy.observe_changes) return;
    auto next = ResolveMetadata(scene, *m_material);
    if (!next || m_metadata == *next) return;

    // A changed key or descriptor is a texture setup boundary. Copy it before destination
    // interning can rehash the scene tables, and leave unchanged metadata entirely inert so
    // script-written sizes and target history survive unrelated resource-only updates.
    ApplyAutosize(scene, *next);
    if (m_policy.crop_card_uvs && !next->sprite &&
        (m_metadata.allocation_size != next->allocation_size ||
         m_metadata.content_size != next->content_size)) {
        const float u = next->content_size[0] / static_cast<float>(next->allocation_size[0]);
        const float v = next->content_size[1] / static_cast<float>(next->allocation_size[1]);
        m_source_mesh->GetVertexArray(0).SetVertex(
            WE_IN_TEXCOORD, std::array { 0.0f, 0.0f, 0.0f, v, u, 0.0f, u, v });
        m_source_mesh->SetDirty();
        m_node->Mesh()->SetDirty();
    }
    if (const auto& layer = m_owner.ImageEffectLayer()) {
        layer->RefreshSourceTexture(scene, m_metadata, *next);
    }
    m_metadata = std::move(*next);
}
} // namespace wallpaper

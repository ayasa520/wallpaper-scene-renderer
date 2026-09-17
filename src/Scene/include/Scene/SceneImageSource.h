#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "Type.hpp"

namespace wallpaper
{
class Scene;
struct SceneMaterial;
class SceneMesh;
class SceneNode;
class SceneObject;

// Primary-texture observation belongs to the image owner, including images without effects.
// Keep the last applied metadata separate from live display size: scripts may resize a card
// without changing its texture, and an unrelated resource refresh must preserve that size.
class SceneImageSource {
public:
    struct Metadata {
        std::string texture_key;
        std::array<int32_t, 2> allocation_size {};
        std::array<float, 2> content_size {};
        std::array<float, 2> display_size {};
        TextureSample sample {};
        bool sprite { false };
        bool operator==(const Metadata&) const = default;
    };

    struct Policy {
        bool observe_changes;
        bool autosize;
        bool crop_card_uvs;
    };

    SceneImageSource(SceneObject& owner, std::shared_ptr<SceneNode> node,
                      const SceneMesh& source_mesh, Policy policy,
                      std::optional<Metadata> metadata);
    ~SceneImageSource();

    static std::optional<Metadata> ResolveMetadata(const Scene& scene, std::string_view name);
    static std::optional<Metadata> ResolveMetadata(const Scene& scene, const SceneMaterial& material);
    std::string_view TextureName() const;
    const SceneMaterial& Material() const { return *m_material; }
    const SceneMesh& SourceMesh() const { return *m_source_mesh; }
    const Metadata& TextureMetadata() const { return m_metadata; }
    void CompleteInitialTexture(Scene& scene);
    void Refresh(Scene& scene);

private:
    void ApplyAutosize(Scene& scene, const Metadata& metadata);

    SceneObject& m_owner;
    std::shared_ptr<SceneNode> m_node;
    std::shared_ptr<SceneMaterial> m_material;
    // This wrapper shares the authored source payload even when the draw node selects a private
    // allocation card. Only parser-identified generated cards enable size or UV mutation.
    std::unique_ptr<SceneMesh> m_source_mesh;
    Policy m_policy;
    Metadata m_metadata;
};
} // namespace wallpaper

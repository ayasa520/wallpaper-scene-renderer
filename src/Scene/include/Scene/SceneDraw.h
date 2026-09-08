#pragma once

#include "Core/NoCopyMove.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace wallpaper
{
class Scene;
class SceneNode;
class SceneObject;
class SceneMesh;

// Nodes and object-owned drawing phases share the same allocation sequence. A graph residency
// key must remain unique even when a destroyed node's address is reused by a publication phase.
inline uint64_t AllocateSceneDrawIdentity() {
    static std::atomic<uint64_t> next { 1 };
    return next.fetch_add(1, std::memory_order_relaxed);
}

// A drawing phase owns only raster resources and projection selection. It has no transform,
// parent, children, script identity, or visibility state. Layer-space placement and visibility
// are read from the authored owner; camera-space placement uses the projection's local origin.
class SceneDrawPhase : NoCopy, NoMove {
public:
    enum class Space { Layer, Camera };

    explicit SceneDrawPhase(SceneObject& owner) : m_owner(owner) {}
    SceneObject& Owner() const { return m_owner; }
    uint64_t RenderIdentity() const { return m_identity; }
    SceneMesh* Mesh() const { return m_mesh.get(); }
    void SetMesh(std::shared_ptr<SceneMesh> mesh) { m_mesh = std::move(mesh); }
    const std::string& Name() const { return m_name; }
    void SetName(std::string name) { m_name = std::move(name); }
    const std::string& Camera() const { return m_camera; }
    Space PlacementSpace() const { return m_space; }
    void SetProjection(Space space, std::string camera) {
        m_space = space;
        m_camera = std::move(camera);
    }

private:
    SceneObject& m_owner;
    const uint64_t m_identity { AllocateSceneDrawIdentity() };
    std::shared_ptr<SceneMesh> m_mesh;
    std::string m_name;
    std::string m_camera;
    Space m_space { Space::Layer };
};

// Non-owning input to the common draw/uniform pipeline. Scene traversal supplies a node, while
// an object's publication supplies a phase directly. This view carries no copied render state
// and cannot participate in the scene hierarchy. The graph copies RenderIdentity() before its
// owner can be retired, and keeps resource access live for text and puppet mesh revisions.
class SceneDraw {
public:
    SceneDraw() = default;
    SceneDraw(SceneNode* node) : m_node(node) {}
    SceneDraw(const SceneDrawPhase& phase) : m_phase(&phase) {}

    bool Valid() const { return m_node != nullptr || m_phase != nullptr; }
    SceneNode* Node() const { return m_node; }
    const SceneDrawPhase* Phase() const { return m_phase; }
    void* DataKey() const;
    uint64_t RenderIdentity() const;
    int32_t LayerId(const Scene&) const;
    SceneMesh* Mesh() const;
    const std::string& Name() const;
    const std::string& Camera() const;
    bool LocalVisible() const;
    bool Visible(const Scene&) const;

private:
    SceneNode* m_node { nullptr };
    const SceneDrawPhase* m_phase { nullptr };
};
} // namespace wallpaper

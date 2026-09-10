#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "Core/NoCopyMove.hpp"
#include "SceneMesh.h"

namespace wallpaper
{

struct SceneModelVertexAttribute {
    const char* constant;
    const char* name;
    std::string_view shader_name;
    uint32_t mask;
    VertexType type;
};

// Format entries describe a canonical packed layout, not a request to rearrange input floats.
std::span<const SceneModelVertexAttribute> ModelVertexAttributes();

struct SceneModelBounds {
    Eigen::Vector3f min { Eigen::Vector3f::Zero() };
    Eigen::Vector3f max { Eigen::Vector3f::Zero() };
};

using SceneModelIndices =
    std::variant<std::monostate, std::vector<uint16_t>, std::vector<uint32_t>>;

// An absent field means "retain", whereas monostate indices and remove explicitly request a
// structural deletion. These are owned CPU snapshots: neither the scene nor the upload path
// retains pointers into the script runtime's typed-array storage.
struct SceneModelShapeUpdate {
    bool remove { false };
    std::optional<std::vector<float>> vertex_buffer;
    std::optional<SceneModelIndices> index_buffer;
    std::optional<uint32_t> vertex_format;
    std::optional<std::string> material;
    std::optional<bool> vertex_dynamic;
    std::optional<bool> index_dynamic;
};

struct SceneModelDataUpdate {
    std::vector<SceneModelShapeUpdate> shapes;
    std::optional<SceneModelBounds> bounds;
};

class SceneModelData : NoCopy, NoMove {
public:
    struct Shape {
        std::shared_ptr<SceneMesh> geometry;
        std::string material;
        uint32_t vertex_format { 0 };
        bool vertex_dynamic { false };
        bool index_dynamic { false };
    };

    ~SceneModelData();
    static std::shared_ptr<SceneModelData> Create(const SceneModelDataUpdate& data,
                                                 std::string& error);
    bool ApplyData(const SceneModelDataUpdate& data, std::string& error);

    const std::vector<Shape>& Shapes() const { return m_shapes; }
    const SceneModelBounds& Bounds() const { return m_bounds; }
    uint32_t Token() const { return m_token; }

private:
    friend class SceneModelDataRegistry;
    SceneModelData() = default;
    void RefreshBounds();

    std::vector<Shape> m_shapes;
    std::optional<SceneModelBounds> m_declared_bounds;
    SceneModelBounds m_bounds;
    uint32_t m_token { 0 };
};

// A scene owns the script reference, and each model layer owns a shared resource reference.
// Releasing the script handle must not invalidate a live layer; collecting a JS wrapper must
// not substitute for explicit release. Weak lookup entries allow the final layer teardown to
// release geometry immediately, without a model-to-layer reference cycle.
class SceneModelDataRegistry : NoCopy, NoMove {
public:
    uint32_t Register(std::shared_ptr<SceneModelData> data);
    std::shared_ptr<SceneModelData> Find(uint32_t token) const;
    bool Release(uint32_t token);
    void Prune();

private:
    struct Entry {
        std::shared_ptr<SceneModelData> script_reference;
        std::weak_ptr<SceneModelData> resource;
    };
    std::unordered_map<uint32_t, Entry> m_entries;
    uint64_t m_next_token { 1 };
};

} // namespace wallpaper

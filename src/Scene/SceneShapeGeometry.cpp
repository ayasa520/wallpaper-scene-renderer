#include "Scene/SceneShapeGeometry.h"

#include <array>
#include <optional>
#include <string>

#include "Scene/SceneDestinationTarget.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"

namespace wallpaper
{
namespace
{
using ShapePoints = std::array<std::array<float, 2>, 4>;

std::optional<ShapePoints> ReadShapeMaterialPoints(SceneImageEffectLayer& layer) {
    // Only the first effect's first material participates in geometry setup. Read the same
    // current uniform storage used by script/material writes; effect names, shader filenames and
    // later passes do not select or merge these controls.
    if (layer.EffectCount() == 0 || layer.GetEffect(0)->nodes.empty()) return std::nullopt;
    const auto& material = *layer.GetEffect(0)->nodes.front().sceneNode->Mesh()->Material();
    ShapePoints points;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto alias = material.uniformAliases.find("point" + std::to_string(i));
        if (alias == material.uniformAliases.end()) return std::nullopt;
        const auto* value = material.FindUniformValue(alias->second);
        if (value == nullptr || value->size() < 2) return std::nullopt;
        points[i] = { (*value)[0], (*value)[1] };
    }
    return points;
}
} // namespace

void RebuildShapeLayerGeometry(SceneObject& owner) {
    auto& layer = *owner.ImageEffectLayer();
    const auto material_points = ReadShapeMaterialPoints(layer);
    const ShapePoints points = material_points.value_or(
        ShapePoints { { { 0.0f, 0.0f }, { 1.0f, 0.0f },
                        { 1.0f, 1.0f }, { 0.0f, 1.0f } } });
    // The owner fields are script storage; the retained card describes the last geometry setup.
    // Read the current owner size only when this construction/setup operation actually runs, not
    // on each property write or draw. This also makes a later child-boundary setup consume an
    // intervening size edit without treating that edit itself as a mesh or destination
    // invalidation.
    const auto size = owner.ImageRuntimeState()->size;
    layer.SetCardSize(size);
    std::array<float, 12> positions;
    std::array<float, 8> texcoords;
    Eigen::Vector3f bounds_min;
    Eigen::Vector3f bounds_max;
    for (size_t i = 0; i < points.size(); ++i) {
        // Keep all four supplied UVs and their exact positions, including points inside or
        // outside the unit domain. Enclosing them in an expanded axis-aligned rectangle changes
        // the authored triangles and coverage.
        const float x = size[0] * points[i][0] - size[0] * 0.5f;
        const float y = size[1] * (1.0f - points[i][1]) - size[1] * 0.5f;
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = 0.0f;
        texcoords[i * 2] = points[i][0];
        texcoords[i * 2 + 1] = points[i][1];
        const Eigen::Vector3f position { x, y, 0.0f };
        bounds_min = i == 0 ? position : bounds_min.cwiseMin(position).eval();
        bounds_max = i == 0 ? position : bounds_max.cwiseMax(position).eval();
    }

    SceneVertexArray vertices({ { WE_IN_POSITION.data(), VertexType::FLOAT3 },
                                { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 } }, 4);
    vertices.SetVertex(WE_IN_POSITION, positions);
    vertices.SetVertex(WE_IN_TEXCOORD, texcoords);
    const std::array<uint16_t, 6> triangles { 0, 2, 1, 0, 3, 2 };
    SceneIndexArray indices(2);
    indices.AssignHalf(0, triangles);
    SceneMesh mesh;
    mesh.AddVertexArray(std::move(vertices));
    mesh.AddIndexArray(std::move(indices));
    mesh.SetBounds(bounds_min, bounds_max);
    layer.FinalMesh().ChangeMeshDataFrom(mesh);
    layer.FinalMesh().SetDirty();
    // Construction has no registered owner draw handle yet. Store the canonical mesh only;
    // the first graph resolve consumes it after layer registration. Runtime setup separately
    // synchronizes already resolved draws once this geometry operation has completed.
    LOG_INFO("SceneShapeGeometrySetup: layer=%d parent=%d controls=%s card=[%.3f %.3f] "
             "points=[%.3f %.3f; %.3f %.3f; %.3f %.3f; %.3f %.3f]",
             owner.Id(), owner.ParentId(), material_points ? "material" : "ordinary-card",
             size[0], size[1], points[0][0], points[0][1], points[1][0], points[1][1],
             points[2][0], points[2][1], points[3][0], points[3][1]);
}

void RefreshShapeLayerResources(Scene& scene, SceneObject& owner) {
    // Destination/FBO setup runs before the material-controlled publication mesh is rebuilt. The
    // card's live dimensions shape its vertices; the independent destination sizing policy still
    // returns half the canvas.
    owner.ImageEffectLayer()->RefreshDestinationTargets(
        scene, ResolveShapeDestinationExtent({ scene.ortho[0], scene.ortho[1] }));
    RebuildShapeLayerGeometry(owner);
    owner.ImageEffectLayer()->SyncResolvedOutputMesh();
}
} // namespace wallpaper

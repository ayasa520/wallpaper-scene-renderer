#include "SceneModelData.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

#include "SpecTexs.hpp"
#include "Utils/Logging.h"

namespace wallpaper
{
namespace
{
constexpr std::array kAttributes {
    SceneModelVertexAttribute { "POSITION", "position", WE_IN_POSITION, 1, VertexType::FLOAT3 },
    SceneModelVertexAttribute { "NORMAL", "normal", WE_IN_NORMAL, 2, VertexType::FLOAT3 },
    SceneModelVertexAttribute { "TANGENT_SIGNED", "tangentSigned", WE_IN_TANGENT4, 4,
                                VertexType::FLOAT4 },
    SceneModelVertexAttribute { "UV", "uv", WE_IN_TEXCOORD, 8, VertexType::FLOAT2 },
    SceneModelVertexAttribute { "COLOR", "color", WE_IN_COLOR, 0x8000, VertexType::FLOAT4 },
};

size_t VertexStride(uint32_t format) {
    size_t stride = 0;
    uint32_t known = 0;
    for (const auto& attribute : kAttributes) {
        known |= attribute.mask;
        if ((format & attribute.mask) != 0) stride += SceneVertexArray::TypeCount(attribute.type);
    }
    return (format & ~known) == 0 ? stride : 0;
}

bool ValidateVertices(std::span<const float> vertices, uint32_t format, std::string& error) {
    const auto stride = VertexStride(format);
    if (stride == 0 || vertices.empty() || vertices.size() % stride != 0 ||
        vertices.size() / stride > std::numeric_limits<uint32_t>::max()) {
        error = "vertexBuffer must contain complete vertices in the declared packed format";
        return false;
    }
    if (! std::all_of(vertices.begin(), vertices.end(), [](float v) { return std::isfinite(v); })) {
        error = "vertexBuffer contains a non-finite component";
        return false;
    }
    return true;
}

bool ValidateIndices(const SceneModelIndices& indices, size_t vertex_count, std::string& error) {
    return std::visit([&](const auto& values) {
        using T = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            if (vertex_count % 3 != 0) {
                error = "non-indexed vertexBuffer must contain complete triangles";
                return false;
            }
        } else {
            if (values.empty() || values.size() % 3 != 0 ||
                values.size() > std::numeric_limits<uint32_t>::max()) {
                error = "indexBuffer must contain complete triangles";
                return false;
            }
            if (! std::all_of(values.begin(), values.end(),
                             [vertex_count](auto index) { return index < vertex_count; })) {
                error = "indexBuffer references a vertex outside vertexBuffer";
                return false;
            }
        }
        return true;
    }, indices);
}

void WriteIndices(SceneIndexArray& target, const SceneModelIndices& indices) {
    std::visit([&](const auto& values) {
        using T = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
            target.AssignHalf(0, values);
        } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
            target.Assign(0, values);
        }
    }, indices);
}

bool ValidBounds(const std::optional<SceneModelBounds>& bounds, std::string& error) {
    if (bounds && (! bounds->min.allFinite() || ! bounds->max.allFinite() ||
                   (bounds->min.array() > bounds->max.array()).any())) {
        error = "boundingBoxMins and boundingBoxMaxs must be finite and ordered";
        return false;
    }
    return true;
}

std::shared_ptr<SceneMesh> BuildGeometry(const SceneModelShapeUpdate& data, std::string& error) {
    if (! data.vertex_buffer || ! data.vertex_format || ! data.material || data.material->empty() ||
        data.remove) {
        error = "new shapes require vertexBuffer, vertexFormat and material";
        return nullptr;
    }
    if (! ValidateVertices(*data.vertex_buffer, *data.vertex_format, error)) return nullptr;
    const size_t vertex_count = data.vertex_buffer->size() / VertexStride(*data.vertex_format);
    const SceneModelIndices no_indices;
    const auto& indices = data.index_buffer ? *data.index_buffer : no_indices;
    if (! ValidateIndices(indices, vertex_count, error)) return nullptr;

    std::vector<SceneVertexArray::SceneVertexAttribute> attributes;
    for (const auto& attribute : kAttributes) {
        if ((*data.vertex_format & attribute.mask) != 0) {
            attributes.push_back({ .name = std::string(attribute.shader_name),
                                   .type = attribute.type, .padding = false });
        }
    }
    auto mesh = std::make_shared<SceneMesh>(data.vertex_dynamic.value_or(false) ||
                                          data.index_dynamic.value_or(false));
    SceneVertexArray vertices(attributes, vertex_count);
    vertices.SetVertexs(0, *data.vertex_buffer);
    mesh->AddVertexArray(std::move(vertices));
    std::visit([&](const auto& values) {
        using T = std::decay_t<decltype(values)>;
        if constexpr (! std::is_same_v<T, std::monostate>) {
            // SceneIndexArray capacity is measured in triangles of uint32 storage words. This
            // also retains enough aligned capacity for a single packed uint16 triangle and its
            // uploader's minimum allocation; only the logical triangle indices are drawn.
            SceneIndexArray array(values.size() / 3);
            WriteIndices(array, indices);
            mesh->SetIndexElementBytes(sizeof(typename T::value_type));
            mesh->AddIndexArray(std::move(array));
        }
    }, indices);
    return mesh;
}

bool IndexLayoutMatches(const SceneMesh& mesh, const SceneModelIndices& indices) {
    return std::visit([&](const auto& values) {
        using T = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return mesh.IndexCount() == 0;
        } else {
            return mesh.IndexCount() != 0 &&
                   mesh.IndexElementBytes() == sizeof(typename T::value_type) &&
                   mesh.LogicalIndexCount() == values.size();
        }
    }, indices);
}

bool NeedsGeometryRebuild(const SceneModelData::Shape& shape,
                           const SceneModelShapeUpdate& data) {
    const auto& mesh = *shape.geometry;
    return (data.vertex_format && *data.vertex_format != shape.vertex_format) ||
           (data.vertex_dynamic && *data.vertex_dynamic != shape.vertex_dynamic) ||
           (data.index_dynamic && *data.index_dynamic != shape.index_dynamic) ||
           (data.vertex_buffer && (! shape.vertex_dynamic ||
                data.vertex_buffer->size() != mesh.GetVertexArray(0).DataSize())) ||
           (data.index_buffer && (! IndexLayoutMatches(mesh, *data.index_buffer) ||
                (! shape.index_dynamic && mesh.IndexCount() != 0)));
}

SceneModelShapeUpdate CompleteReplacement(const SceneModelData::Shape& shape,
                                          const SceneModelShapeUpdate& data) {
    auto complete = data;
    const auto& mesh = *shape.geometry;
    // Structural replacement is an infrequent operation. Reconstruct a complete owned shape
    // from the current CPU payload before allocating new draw storage; omitted fields retain
    // their current values even when another field changes the layout, counts or mutability.
    // No pointer into either a JS view or the retired geometry survives this reconstruction.
    if (! complete.vertex_buffer) {
        const auto& vertices = mesh.GetVertexArray(0);
        complete.vertex_buffer.emplace(vertices.Data(), vertices.Data() + vertices.DataSize());
    }
    if (! complete.index_buffer) {
        if (mesh.IndexCount() == 0) {
            complete.index_buffer.emplace(std::monostate {});
        } else if (mesh.IndexElementBytes() == sizeof(uint16_t)) {
            std::vector<uint16_t> indices(mesh.LogicalIndexCount());
            std::memcpy(indices.data(), mesh.GetIndexArray(0).Data(),
                        indices.size() * sizeof(uint16_t));
            complete.index_buffer.emplace(std::move(indices));
        } else {
            const auto* indices = mesh.GetIndexArray(0).Data();
            complete.index_buffer.emplace(
                std::vector<uint32_t>(indices, indices + mesh.LogicalIndexCount()));
        }
    }
    if (! complete.vertex_format) complete.vertex_format = shape.vertex_format;
    if (! complete.material) complete.material = shape.material;
    if (! complete.vertex_dynamic) complete.vertex_dynamic = shape.vertex_dynamic;
    if (! complete.index_dynamic) complete.index_dynamic = shape.index_dynamic;
    return complete;
}

bool ValidateApplyStructure(const SceneModelData::Shape& shape,
                            const SceneModelShapeUpdate& data, bool apply_indices,
                            std::string& error) {
    if (data.remove || (data.material && *data.material != shape.material) ||
        (data.vertex_format && *data.vertex_format != shape.vertex_format) ||
        (data.vertex_dynamic && *data.vertex_dynamic != shape.vertex_dynamic) ||
        (data.index_dynamic && *data.index_dynamic != shape.index_dynamic)) {
        error = "applyData cannot change shape structure, material, format or dynamic flags";
        return false;
    }
    if (data.vertex_buffer && ! shape.vertex_dynamic) {
        error = "vertexBuffer was not created with isVertexBufferDynamic";
        return false;
    }
    if (apply_indices && ! shape.index_dynamic) {
        error = "indexBuffer was not created with isIndexBufferDynamic";
        return false;
    }
    return true;
}

bool ApplyShapePayload(SceneModelData::Shape& shape, const SceneModelShapeUpdate& data,
                        bool replace, bool& vertices_changed, std::string& error) {
    auto& mesh = *shape.geometry;
    const bool apply_indices = data.index_buffer &&
        ! (replace && mesh.IndexCount() == 0 &&
           std::holds_alternative<std::monostate>(*data.index_buffer));
    if (! ValidateApplyStructure(shape, data, apply_indices, error)) return false;
    if (data.vertex_buffer) {
        if (data.vertex_buffer->size() != mesh.GetVertexArray(0).DataSize()) {
            error = "applyData vertexBuffer length must match the retained buffer";
            return false;
        }
        if (! ValidateVertices(*data.vertex_buffer, shape.vertex_format, error)) return false;
        mesh.GetVertexArray(0).SetVertexs(0, *data.vertex_buffer);
        mesh.SetDirty();
        vertices_changed = true;
    }
    if (apply_indices) {
        if (std::holds_alternative<std::monostate>(*data.index_buffer) ||
            ! IndexLayoutMatches(mesh, *data.index_buffer)) {
            error = "applyData indexBuffer length and type must match the retained buffer";
            return false;
        }
        if (! ValidateIndices(*data.index_buffer, mesh.GetVertexArray(0).VertexCount(), error))
            return false;
        WriteIndices(mesh.GetIndexArray(0), *data.index_buffer);
        mesh.SetDirty();
    }
    return true;
}
} // namespace

std::span<const SceneModelVertexAttribute> ModelVertexAttributes() { return kAttributes; }

SceneModelData::~SceneModelData() {
    if (wallpaper::diagnostics::Options().trace_model_data) {
        LOG_INFO("SceneModelDataRelease: token=%u shapes=%zu", m_token, m_shapes.size());
    }
}

std::shared_ptr<SceneModelData> SceneModelData::Create(const SceneModelDataUpdate& data,
                                                     std::string& error) {
    if (data.shapes.empty()) {
        error = "shapes must not be empty";
        return nullptr;
    }
    if (! ValidBounds(data.bounds, error)) return nullptr;
    auto model = std::shared_ptr<SceneModelData>(new SceneModelData());
    model->m_shapes.reserve(data.shapes.size());
    for (const auto& shape : data.shapes) {
        auto geometry = BuildGeometry(shape, error);
        if (! geometry) return nullptr;
        model->m_shapes.push_back({ .geometry = std::move(geometry), .material = *shape.material,
                                    .vertex_format = *shape.vertex_format,
                                    .vertex_dynamic = shape.vertex_dynamic.value_or(false),
                                    .index_dynamic = shape.index_dynamic.value_or(false) });
    }
    model->m_declared_bounds = data.bounds;
    model->RefreshBounds();
    return model;
}

bool SceneModelData::ApplyData(const SceneModelDataUpdate& data, std::string& error) {
    return UpdateData(data, false, error);
}

bool SceneModelData::ReplaceData(const SceneModelDataUpdate& data, std::string& error) {
    return UpdateData(data, true, error);
}

bool SceneModelData::UpdateData(const SceneModelDataUpdate& data, bool replace,
                                std::string& error) {
    if (data.shapes.empty() || (! replace && data.shapes.size() > m_shapes.size())) {
        error = "applyData must address existing shapes";
        return false;
    }
    if (! ValidBounds(data.bounds, error)) return false;
    bool vertices_changed = false;
    bool succeeded = true;
    // JS extraction finishes before publication, so getters can re-enter without invalidating
    // a borrowed view or a shape reference. Resource checks then use that final current model.
    // Publication is ordered, not transactional: a later error retains earlier valid writes,
    // including a changed material on the failing shape. Deletions are applied only after this
    // first pass succeeds, keeping input indices stable throughout the operation.
    for (size_t i = 0; i < data.shapes.size(); ++i) {
        const auto& update = data.shapes[i];
        if (update.remove) {
            if (! replace || i >= m_shapes.size()) {
                error = "only replaceData can remove an existing shape";
                succeeded = false;
                break;
            }
            continue;
        }
        if (i == m_shapes.size()) {
            auto geometry = BuildGeometry(update, error);
            if (! geometry) {
                succeeded = false;
                break;
            }
            m_shapes.push_back({ .geometry = std::move(geometry), .material = *update.material,
                                  .vertex_format = *update.vertex_format,
                                  .vertex_dynamic = update.vertex_dynamic.value_or(false),
                                  .index_dynamic = update.index_dynamic.value_or(false) });
            ++m_structure_revision;
            vertices_changed = true;
            continue;
        }
        auto& shape = m_shapes[i];
        if (replace && update.material && *update.material != shape.material) {
            shape.material = *update.material;
            ++m_structure_revision;
        }
        if (replace && NeedsGeometryRebuild(shape, update)) {
            const auto complete = CompleteReplacement(shape, update);
            auto geometry = BuildGeometry(complete, error);
            if (! geometry) {
                succeeded = false;
                break;
            }
            shape.geometry = std::move(geometry);
            shape.vertex_format = *complete.vertex_format;
            shape.vertex_dynamic = *complete.vertex_dynamic;
            shape.index_dynamic = *complete.index_dynamic;
            ++m_structure_revision;
            vertices_changed = true;
        } else if (! ApplyShapePayload(shape, update, replace, vertices_changed, error)) {
            succeeded = false;
            break;
        }
    }
    if (succeeded && replace) {
        for (size_t i = data.shapes.size(); i-- > 0;) {
            if (! data.shapes[i].remove) continue;
            m_shapes.erase(m_shapes.begin() + static_cast<std::vector<Shape>::difference_type>(i));
            ++m_structure_revision;
            vertices_changed = true;
        }
    }
    if (succeeded && data.bounds) m_declared_bounds = data.bounds;
    if (vertices_changed || (succeeded && data.bounds)) RefreshBounds();
    return succeeded;
}

void SceneModelData::RefreshBounds() {
    if (m_declared_bounds) {
        m_bounds = *m_declared_bounds;
    } else {
        bool found = false;
        for (const auto& shape : m_shapes) {
            if ((shape.vertex_format & kAttributes.front().mask) == 0) continue;
            const auto& vertices = shape.geometry->GetVertexArray(0);
            for (size_t i = 0; i < vertices.VertexCount(); ++i) {
                const Eigen::Vector3f point(vertices.Data() + i * vertices.OneSize());
                if (! found) {
                    m_bounds = { point, point };
                    found = true;
                } else {
                    m_bounds.min = m_bounds.min.cwiseMin(point);
                    m_bounds.max = m_bounds.max.cwiseMax(point);
                }
            }
        }
        if (! found) m_bounds = {};
    }
    // The declared model box can enclose future deformations. Keep it on every shared payload
    // so all owners and drawing phases observe the same bounds independently of their material.
    for (auto& shape : m_shapes) shape.geometry->SetBounds(m_bounds.min, m_bounds.max);
}

uint32_t SceneModelDataRegistry::Register(std::shared_ptr<SceneModelData> data) {
    Prune();
    if (m_next_token > std::numeric_limits<uint32_t>::max()) return 0;
    const auto token = static_cast<uint32_t>(m_next_token++);
    data->m_token = token;
    m_entries.emplace(token, Entry { data, data });
    if (wallpaper::diagnostics::Options().trace_model_data) {
        LOG_INFO("SceneModelDataCreate: token=%u shapes=%zu", token, data->Shapes().size());
    }
    return token;
}

std::shared_ptr<SceneModelData> SceneModelDataRegistry::Find(uint32_t token) const {
    const auto found = m_entries.find(token);
    return found == m_entries.end() ? nullptr : found->second.resource.lock();
}

bool SceneModelDataRegistry::Release(uint32_t token) {
    const auto found = m_entries.find(token);
    if (found == m_entries.end() || ! found->second.script_reference) return false;
    found->second.script_reference.reset();
    if (wallpaper::diagnostics::Options().trace_model_data) {
        LOG_INFO("SceneModelDataDestroy: token=%u retained=%s", token,
                 found->second.resource.expired() ? "false" : "true");
    }
    if (found->second.resource.expired()) m_entries.erase(found);
    return true;
}

void SceneModelDataRegistry::Prune() {
    std::erase_if(m_entries, [](const auto& entry) { return entry.second.resource.expired(); });
}
} // namespace wallpaper

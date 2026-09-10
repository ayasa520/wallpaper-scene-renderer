#include "SceneModelData.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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

bool ValidateApply(const SceneModelData::Shape& shape, const SceneModelShapeUpdate& data,
                    std::string& error) {
    const auto& mesh = *shape.geometry;
    if (data.remove || (data.material && *data.material != shape.material) ||
        (data.vertex_format && *data.vertex_format != shape.vertex_format) ||
        (data.vertex_dynamic && *data.vertex_dynamic != shape.vertex_dynamic) ||
        (data.index_dynamic && *data.index_dynamic != shape.index_dynamic)) {
        error = "applyData cannot change shape structure, material, format or dynamic flags";
        return false;
    }
    if (data.vertex_buffer) {
        if (! shape.vertex_dynamic) {
            error = "vertexBuffer was not created with isVertexBufferDynamic";
            return false;
        }
        if (data.vertex_buffer->size() != mesh.GetVertexArray(0).DataSize()) {
            error = "applyData vertexBuffer length must match the retained buffer";
            return false;
        }
        if (! ValidateVertices(*data.vertex_buffer, shape.vertex_format, error)) return false;
    }
    if (data.index_buffer) {
        if (! shape.index_dynamic) {
            error = "indexBuffer was not created with isIndexBufferDynamic";
            return false;
        }
        const bool matches = std::visit([&](const auto& values) {
            using T = std::decay_t<decltype(values)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return false;
            } else {
                return mesh.IndexCount() != 0 &&
                       mesh.IndexElementBytes() == sizeof(typename T::value_type) &&
                       mesh.LogicalIndexCount() == values.size();
            }
        }, *data.index_buffer);
        if (! matches) {
            error = "applyData indexBuffer length and type must match the retained buffer";
            return false;
        }
        if (! ValidateIndices(*data.index_buffer, mesh.GetVertexArray(0).VertexCount(), error))
            return false;
    }
    return true;
}
} // namespace

std::span<const SceneModelVertexAttribute> ModelVertexAttributes() { return kAttributes; }

SceneModelData::~SceneModelData() {
    if (std::getenv("WESCENE_TRACE_MODEL_DATA") != nullptr) {
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
    if (data.shapes.empty() || data.shapes.size() > m_shapes.size()) {
        error = "applyData must address existing shapes";
        return false;
    }
    if (! ValidBounds(data.bounds, error)) return false;
    // Validate the owned snapshots before publishing any bytes. Property getters can re-enter
    // scripts during conversion, so validation belongs here, against the final current resource,
    // rather than being interleaved with reads of JS fields and mutations of shared geometry.
    for (size_t i = 0; i < data.shapes.size(); ++i) {
        if (! ValidateApply(m_shapes[i], data.shapes[i], error)) return false;
    }
    bool vertices_changed = false;
    for (size_t i = 0; i < data.shapes.size(); ++i) {
        const auto& update = data.shapes[i];
        auto& mesh = *m_shapes[i].geometry;
        if (update.vertex_buffer) {
            mesh.GetVertexArray(0).SetVertexs(0, *update.vertex_buffer);
            vertices_changed = true;
        }
        if (update.index_buffer) WriteIndices(mesh.GetIndexArray(0), *update.index_buffer);
        if (update.vertex_buffer || update.index_buffer) mesh.SetDirty();
    }
    if (data.bounds) m_declared_bounds = data.bounds;
    if (vertices_changed || data.bounds) RefreshBounds();
    return true;
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
    if (std::getenv("WESCENE_TRACE_MODEL_DATA") != nullptr) {
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
    if (std::getenv("WESCENE_TRACE_MODEL_DATA") != nullptr) {
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

#include "WPParticleRawGener.h"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "Core/Literals.hpp"
#include "SpecTexs.hpp"
#include "ParticleModify.h"
#include "ParticleSystem.h"

#include "Utils/Logging.h"

using namespace wallpaper;
using namespace Eigen;

namespace
{
const char* ParticleRendererName(ParticleRendererKind renderer) noexcept {
    switch (renderer) {
    case ParticleRendererKind::Sprite: return "sprite";
    case ParticleRendererKind::SpriteTrail: return "spritetrail";
    case ParticleRendererKind::Rope: return "rope";
    case ParticleRendererKind::RopeTrail: return "ropetrail";
    }
    return "unknown";
}

const char* ParticleExpansionName(ParticleExpansionMode expansion) noexcept {
    switch (expansion) {
    case ParticleExpansionMode::GeometryPoint: return "GeometryPoint";
    case ParticleExpansionMode::IndexedQuad: return "IndexedQuad";
    }
    return "unknown";
}

struct ParticleRenderDiagnostics {
    std::string_view object_name;
    std::string_view material_name;
    const char*      renderer_name;
    const char*      expansion_name;
};

/*
 * The particle generators write a fixed, known set of shader input attributes for every vertex of
 * every live particle each frame. Resolving those names through a string-keyed map per Write()
 * call showed up in whole-process profiles on emitter-heavy scenes, so the layout is resolved once
 * per GenGLData() into a dense enum-indexed table and the per-vertex path only performs array
 * indexing.
 */
enum class ParticleVertexAttr : usize {
    Position,
    PositionVec4,
    Color,
    TexCoordVec4,
    TexCoordC2,
    TexCoordVec4C1,
    TexCoordVec4C2,
    TexCoordVec4C3,
    TexCoordVec3C2,
    Count,
};

constexpr std::array<std::string_view, static_cast<usize>(ParticleVertexAttr::Count)>
    kParticleVertexAttrNames {
        WE_IN_POSITION,       WE_IN_POSITIONVEC4,   WE_IN_COLOR,
        WE_IN_TEXCOORDVEC4,   WE_IN_TEXCOORDC2,     WE_IN_TEXCOORDVEC4C1,
        WE_IN_TEXCOORDVEC4C2, WE_IN_TEXCOORDVEC4C3, WE_IN_TEXCOORDVEC3C2,
    };

constexpr std::string_view ParticleVertexAttrName(ParticleVertexAttr attr) noexcept {
    return kParticleVertexAttrNames[static_cast<usize>(attr)];
}

struct ParticleVertexLayout {
    struct Attribute {
        usize offset { 0 };
        // Zero width marks an attribute the current mesh layout does not contain.
        usize width { 0 };
    };

    usize stride { 0 };
    std::array<Attribute, static_cast<usize>(ParticleVertexAttr::Count)> attributes {};

    const Attribute& At(ParticleVertexAttr attr) const noexcept {
        return attributes[static_cast<usize>(attr)];
    }
};

std::optional<ParticleVertexLayout> ResolveParticleVertexLayout(const SceneVertexArray& vertices,
                                                                 const ParticleRenderDiagnostics& diagnostics) {
    ParticleVertexLayout layout;
    for (const auto& attribute : vertices.Attributes()) {
        const usize width = SceneVertexArray::TypeCount(attribute.type);
        for (usize index = 0; index < kParticleVertexAttrNames.size(); index++) {
            if (kParticleVertexAttrNames[index] == attribute.name) {
                layout.attributes[index] = ParticleVertexLayout::Attribute {
                    .offset = layout.stride,
                    .width  = width,
                };
                break;
            }
        }
        layout.stride += SceneVertexArray::RealAttributeSize(attribute);
    }
    if (layout.stride == 0 || layout.stride != vertices.OneSize()) {
        LOG_ERROR("particle vertex layout invalid object='%.*s' material='%.*s' renderer=%s "
                  "expansion=%s stride=%zu expected=%zu",
                  static_cast<int>(diagnostics.object_name.size()),
                  diagnostics.object_name.data(),
                  static_cast<int>(diagnostics.material_name.size()),
                  diagnostics.material_name.data(),
                  diagnostics.renderer_name,
                  diagnostics.expansion_name,
                  layout.stride,
                  vertices.OneSize());
        return std::nullopt;
    }
    return layout;
}

class ParticleVertexWriter {
public:
    ParticleVertexWriter(SceneVertexArray& vertices, ParticleVertexLayout layout,
                         ParticleRenderDiagnostics diagnostics)
        : vertices_(vertices),
          layout_(std::move(layout)),
          diagnostics_(diagnostics),
          storage_(layout_.stride, 0.0f),
          vertex_capacity_(vertices.OneSize() == 0
                               ? 0
                               : vertices.CapacitySize() / vertices.OneSize()) {}

    void BeginVertex() {
        std::fill(storage_.begin(), storage_.end(), 0.0f);
    }

    bool Write(ParticleVertexAttr attr, std::span<const float> values) {
        const auto& attribute = layout_.At(attr);
        if (attribute.width != values.size() ||
            attribute.offset + values.size() > storage_.size()) {
            const std::string_view name = ParticleVertexAttrName(attr);
            LOG_ERROR("particle vertex attribute mismatch object='%.*s' material='%.*s' "
                      "renderer=%s expansion=%s attribute='%.*s' values=%zu expected=%zu "
                      "stride=%zu",
                      static_cast<int>(diagnostics_.object_name.size()),
                      diagnostics_.object_name.data(),
                      static_cast<int>(diagnostics_.material_name.size()),
                      diagnostics_.material_name.data(),
                      diagnostics_.renderer_name,
                      diagnostics_.expansion_name,
                      static_cast<int>(name.size()),
                      name.data(),
                      values.size(),
                      attribute.width,
                      layout_.stride);
            return false;
        }
        std::copy(values.begin(), values.end(), storage_.begin() + attribute.offset);
        return true;
    }

    template<size_t N>
    bool Write(ParticleVertexAttr attr, const std::array<float, N>& values) {
        return Write(attr, std::span<const float>(values));
    }

    bool EnsureCapacity(size_t additional_vertices, std::string_view operation) const {
        if (next_vertex_ <= vertex_capacity_ &&
            additional_vertices <= vertex_capacity_ - next_vertex_) {
            return true;
        }
        LOG_ERROR("particle vertex capacity exceeded object='%.*s' material='%.*s' renderer=%s "
                  "expansion=%s operation='%.*s' written=%zu requested=%zu capacity=%zu",
                  static_cast<int>(diagnostics_.object_name.size()),
                  diagnostics_.object_name.data(),
                  static_cast<int>(diagnostics_.material_name.size()),
                  diagnostics_.material_name.data(),
                  diagnostics_.renderer_name,
                  diagnostics_.expansion_name,
                  static_cast<int>(operation.size()),
                  operation.data(),
                  next_vertex_,
                  additional_vertices,
                  vertex_capacity_);
        return false;
    }

    bool Commit() {
        if (! vertices_.SetVertexs(next_vertex_, storage_)) {
            LOG_ERROR("particle vertex commit failed object='%.*s' material='%.*s' renderer=%s "
                      "expansion=%s vertex=%zu capacity=%zu",
                      static_cast<int>(diagnostics_.object_name.size()),
                      diagnostics_.object_name.data(),
                      static_cast<int>(diagnostics_.material_name.size()),
                      diagnostics_.material_name.data(),
                      diagnostics_.renderer_name,
                      diagnostics_.expansion_name,
                      next_vertex_,
                      vertex_capacity_);
            return false;
        }
        next_vertex_++;
        return true;
    }

    size_t Written() const noexcept { return next_vertex_; }
    const ParticleRenderDiagnostics& Diagnostics() const noexcept { return diagnostics_; }

private:
    SceneVertexArray&          vertices_;
    ParticleVertexLayout       layout_;
    ParticleRenderDiagnostics diagnostics_;
    std::vector<float>         storage_;
    size_t                     vertex_capacity_ { 0 };
    size_t                     next_vertex_ { 0 };
};

Eigen::Vector3f RenderParticlePosition(const ParticleInstance& instance,
                                       const Eigen::Vector3f& position) {
    return instance.GetBoundedData().pos + position;
}

template<typename Fn>
bool ForEachLiveParticle(std::span<const std::unique_ptr<ParticleInstance>> instances, Fn&& fn) {
    for (const auto& instance : instances) {
        if (instance->IsNoLiveParticle()) continue;
        for (const auto& particle : instance->Particles()) {
            if (! ParticleModify::LifetimeOk(particle)) continue;
            if (! fn(*instance, particle)) return false;
        }
    }
    return true;
}

size_t CountLiveParticles(std::span<const std::unique_ptr<ParticleInstance>> instances) {
    size_t live_particle_count { 0 };
    for (const auto& instance : instances) {
        if (instance->IsNoLiveParticle()) continue;
        const auto particles = instance->Particles();
        live_particle_count += static_cast<size_t>(std::count_if(
            particles.begin(), particles.end(), [](const Particle& p) {
                return ParticleModify::LifetimeOk(p);
            }));
    }
    return live_particle_count;
}

struct SpriteParticleRenderData {
    Eigen::Vector3f     position;
    std::array<float, 4> color;
    std::array<float, 4> velocity_lifetime;
};

SpriteParticleRenderData PrepareSpriteParticleData(const ParticleInstance& instance,
                                                   const Particle& particle,
                                                   const ParticleRawGenSpecOp& specOp,
                                                   float alpha_multiplier) {
    float animation_lifetime = particle.lifetime;
    specOp(particle, { &animation_lifetime });
    const auto& render_velocity = ParticleModify::GetRenderVelocity(particle);
    return SpriteParticleRenderData {
        .position = RenderParticlePosition(instance, particle.position),
        .color = { particle.color[0],
                   particle.color[1],
                   particle.color[2],
                   particle.alpha * alpha_multiplier },
        .velocity_lifetime = { render_velocity[0],
                               render_velocity[1],
                               render_velocity[2],
                               animation_lifetime },
    };
}

bool WriteSpriteCommonAttributes(ParticleVertexWriter& writer,
                                 const SpriteParticleRenderData& data, bool thick_format) {
    if (! writer.Write(
            ParticleVertexAttr::Position,
            std::array { data.position[0], data.position[1], data.position[2] }) ||
        ! writer.Write(ParticleVertexAttr::Color, data.color)) {
        return false;
    }
    return ! thick_format || writer.Write(ParticleVertexAttr::TexCoordVec4C1, data.velocity_lifetime);
}

bool GenParticlePointData(std::span<const std::unique_ptr<ParticleInstance>> instances,
                          const ParticleRawGenSpecOp& specOp, bool thick_format,
                          ParticleVertexWriter& writer, float alpha_multiplier) {
    if (! writer.EnsureCapacity(CountLiveParticles(instances), "sprite-point")) return false;

    return ForEachLiveParticle(instances, [&](const ParticleInstance& instance,
                                               const Particle& particle) {
        const auto data =
            PrepareSpriteParticleData(instance, particle, specOp, alpha_multiplier);
        writer.BeginVertex();
        // Point-expanded materials receive simulation rotation and size directly; their geometry
        // stage creates the four authored corners after the vertex shader has run once.
        return WriteSpriteCommonAttributes(writer, data, thick_format) &&
            writer.Write(ParticleVertexAttr::TexCoordVec4,
                         std::array { particle.rotation[0],
                                      particle.rotation[1],
                                      particle.rotation[2],
                                      particle.size / 2.0f }) &&
            writer.Commit();
    });
}

bool GenParticleQuadData(std::span<const std::unique_ptr<ParticleInstance>> instances,
                         const ParticleRawGenSpecOp& specOp, bool thick_format,
                         ParticleVertexWriter& writer, float alpha_multiplier) {
    constexpr std::array<std::array<float, 2>, 4> kCorners {
        std::array { 0.0f, 1.0f },
        std::array { 1.0f, 1.0f },
        std::array { 1.0f, 0.0f },
        std::array { 0.0f, 0.0f },
    };

    const size_t live_particle_count = CountLiveParticles(instances);
    constexpr size_t kVerticesPerParticle = kCorners.size();
    if (live_particle_count > std::numeric_limits<size_t>::max() / kVerticesPerParticle) {
        const auto& diagnostics = writer.Diagnostics();
        LOG_ERROR("particle quad vertex count overflow object='%.*s' material='%.*s' "
                  "renderer=%s expansion=%s particles=%zu",
                  static_cast<int>(diagnostics.object_name.size()),
                  diagnostics.object_name.data(),
                  static_cast<int>(diagnostics.material_name.size()),
                  diagnostics.material_name.data(),
                  diagnostics.renderer_name,
                  diagnostics.expansion_name,
                  live_particle_count);
        return false;
    }
    const size_t required_vertices = live_particle_count * kVerticesPerParticle;
    if (! writer.EnsureCapacity(required_vertices, "sprite-indexed-quad")) return false;

    return ForEachLiveParticle(instances, [&](const ParticleInstance& instance,
                                               const Particle& particle) {
        const auto data =
            PrepareSpriteParticleData(instance, particle, specOp, alpha_multiplier);

        // A non-GS particle shader receives four invocations with identical simulation data. Only
        // the authored corner changes, and the rotation pair remains in TexCoordC2 exactly as the
        // original vertex shader declares it.
        for (const auto& corner : kCorners) {
            writer.BeginVertex();
            if (! WriteSpriteCommonAttributes(writer, data, thick_format) ||
                ! writer.Write(ParticleVertexAttr::TexCoordVec4,
                               std::array { corner[0],
                                            corner[1],
                                            particle.rotation[2],
                                            particle.size / 2.0f }) ||
                ! writer.Write(ParticleVertexAttr::TexCoordC2,
                               std::array { particle.rotation[0], particle.rotation[1] }) ||
                ! writer.Commit()) {
                return false;
            }
        }
        return true;
    });
}

class ParticleQuadIndexWriter {
public:
    ParticleQuadIndexWriter(SceneIndexArray& indices, ParticleRenderDiagnostics diagnostics)
        : indices_(indices), diagnostics_(diagnostics) {}

    bool SetQuadCount(size_t quad_count) {
        constexpr size_t kIndicesPerQuad = 6;
        constexpr size_t kVerticesPerQuad = 4;
        constexpr size_t kMaxQuadCount =
            (static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1) /
            kVerticesPerQuad;
        const size_t capacity = indices_.PackedUint16CapacityCount() / kIndicesPerQuad;
        if (quad_count > kMaxQuadCount || quad_count > capacity) {
            LOG_ERROR("particle quad index capacity exceeded object='%.*s' material='%.*s' "
                      "renderer=%s expansion=%s requested=%zu capacity=%zu uint16-limit=%zu",
                      static_cast<int>(diagnostics_.object_name.size()),
                      diagnostics_.object_name.data(),
                      static_cast<int>(diagnostics_.material_name.size()),
                      diagnostics_.material_name.data(),
                      diagnostics_.renderer_name,
                      diagnostics_.expansion_name,
                      quad_count,
                      capacity,
                      kMaxQuadCount);
            return false;
        }

        const size_t initialized_quad_count =
            indices_.PackedUint16DataCount() / kIndicesPerQuad;
        for (size_t quad = initialized_quad_count; quad < quad_count; quad++) {
            const uint16_t base = static_cast<uint16_t>(quad * kVerticesPerQuad);
            const std::array<uint16_t, kIndicesPerQuad> values {
                base,
                static_cast<uint16_t>(base + 1),
                static_cast<uint16_t>(base + 3),
                static_cast<uint16_t>(base + 1),
                static_cast<uint16_t>(base + 2),
                static_cast<uint16_t>(base + 3),
            };
            indices_.AssignHalf(quad * kIndicesPerQuad, values);
        }

        indices_.SetPackedUint16RenderDataCount(quad_count * kIndicesPerQuad);
        return true;
    }

private:
    SceneIndexArray&          indices_;
    ParticleRenderDiagnostics diagnostics_;
};

bool GenRopeParticleData(const ParticleInstance& instance, bool thick_format,
                         ParticleVertexWriter& writer, float alpha_multiplier) {
    /*
    attribute vec4 a_PositionVec4;
    attribute vec4 a_TexCoordVec4;
    attribute vec4 a_TexCoordVec4C1;

    #if THICKFORMAT
    attribute vec4 a_TexCoordVec4C2;
    attribute vec4 a_TexCoordVec4C3;
    #else
    attribute vec3 a_TexCoordVec3C2;
    #endif

    attribute vec4 a_Color;

    #define in_ParticleTrailLength (a_TexCoordVec4.w)
    #define in_ParticleTrailPosition (a_TexCoordVec4C1.w)
    */
    const auto particles = instance.Particles();
    std::vector<const Particle*> live_particles;
    live_particles.reserve(particles.size());
    for (const auto& particle : particles) {
        if (ParticleModify::LifetimeOk(particle)) {
            live_particles.push_back(&particle);
        }
    }
    if (live_particles.size() < 2) return true;
    std::sort(live_particles.begin(), live_particles.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->spawnSequence < rhs->spawnSequence;
    });

    const size_t alive_count  = live_particles.size();
    const size_t num_segments = alive_count - 1;
    const float  trail_length = static_cast<float>(alive_count);
    if (! writer.EnsureCapacity(num_segments, "rope-current-chain")) return false;

    for (size_t i = 0; i < num_segments; i++) {
        const auto& p0_src = i > 0 ? *live_particles[i - 1] : *live_particles[i];
        const auto& p1_src = *live_particles[i];
        const auto& p2_src = *live_particles[i + 1];
        const auto& p3_src = i + 2 < alive_count ? *live_particles[i + 2] : *live_particles[i + 1];

        const Vector3f p0 = RenderParticlePosition(instance, p0_src.position);
        const Vector3f p1 = RenderParticlePosition(instance, p1_src.position);
        const Vector3f p2 = RenderParticlePosition(instance, p2_src.position);
        const Vector3f p3 = RenderParticlePosition(instance, p3_src.position);

        /*
         * A rope segment is submitted as one point primitive.  The stock Wallpaper Engine
         * vertex and geometry shaders consume the four adjacent particle positions directly:
         * p0/p3 provide the Bezier tangents, while p1/p2 are the segment endpoints.  Keeping
         * this exact input contract moves ribbon expansion, interpolation, and subdivision into
         * genericropeparticle.geom; the CPU only uploads adjacent current particle positions and
         * never synthesizes ribbon vertices.
         */
        writer.BeginVertex();
        if (! writer.Write(
                ParticleVertexAttr::PositionVec4,
                std::array { p1[0], p1[1], p1[2], p1_src.size / 2.0f }) ||
            ! writer.Write(
                ParticleVertexAttr::TexCoordVec4,
                std::array { p2[0], p2[1], p2[2], trail_length }) ||
            ! writer.Write(
                ParticleVertexAttr::TexCoordVec4C1,
                std::array { p0[0], p0[1], p0[2], static_cast<float>(i) })) {
            return false;
        }
        if (thick_format) {
            if (! writer.Write(
                    ParticleVertexAttr::TexCoordVec4C2,
                    std::array { p3[0], p3[1], p3[2], p2_src.size / 2.0f }) ||
                ! writer.Write(
                    ParticleVertexAttr::TexCoordVec4C3,
                    std::array { p2_src.color[0], p2_src.color[1], p2_src.color[2],
                                 p2_src.alpha * alpha_multiplier })) {
                return false;
            }
        } else if (! writer.Write(
                       ParticleVertexAttr::TexCoordVec3C2,
                       std::array { p3[0], p3[1], p3[2] })) {
            return false;
        }
        if (! writer.Write(
                ParticleVertexAttr::Color,
                std::array { p1_src.color[0],
                             p1_src.color[1],
                             p1_src.color[2],
                             p1_src.alpha * alpha_multiplier }) ||
            ! writer.Commit()) {
            return false;
        }
    }

    return true;
}

bool GenRopeTrailSegments(const ParticleInstance& instance, const Particle& particle,
                          const ParticleTrail& trail, bool thick_format,
                          ParticleVertexWriter& writer, float alpha_multiplier) {
    const size_t trail_length = trail.Length();
    if (trail_length == 0) return true;

    if (! writer.EnsureCapacity(trail_length, "rope-history")) return false;

    const float size = particle.size / 2.0f;
    const auto point = [&](size_t point_index) {
        if (point_index == 0) return RenderParticlePosition(instance, particle.position);
        return RenderParticlePosition(instance, trail.At(trail_length - point_index));
    };

    // Each live head emits a fixed number of point primitives. Segment zero starts at the current
    // operator-complete head position, while following points walk newest-to-oldest history.
    for (size_t sample_index = 0; sample_index < trail_length; sample_index++) {
        const Vector3f start = point(sample_index);
        const Vector3f end   = point(sample_index + 1);
        const Vector3f before = point(sample_index == 0 ? 0 : sample_index - 1);
        const Vector3f after  = point(std::min(sample_index + 2, trail_length));

        writer.BeginVertex();
        if (! writer.Write(
                ParticleVertexAttr::PositionVec4,
                std::array { start[0], start[1], start[2], size }) ||
            ! writer.Write(
                ParticleVertexAttr::TexCoordVec4,
                std::array { end[0], end[1], end[2],
                             static_cast<float>(trail.sample_count) }) ||
            ! writer.Write(
                ParticleVertexAttr::TexCoordVec4C1,
                std::array { before[0], before[1], before[2],
                             static_cast<float>(sample_index) })) {
            return false;
        }
        if (thick_format) {
            if (! writer.Write(
                    ParticleVertexAttr::TexCoordVec4C2,
                    std::array { after[0], after[1], after[2], size }) ||
                ! writer.Write(
                    ParticleVertexAttr::TexCoordVec4C3,
                    std::array { particle.color[0], particle.color[1], particle.color[2],
                                 particle.alpha * alpha_multiplier })) {
                return false;
            }
        } else if (! writer.Write(
                       ParticleVertexAttr::TexCoordVec3C2,
                       std::array { after[0], after[1], after[2] })) {
            return false;
        }
        if (! writer.Write(
                ParticleVertexAttr::Color,
                std::array { particle.color[0], particle.color[1], particle.color[2],
                             particle.alpha * alpha_multiplier }) ||
            ! writer.Commit()) {
            return false;
        }
    }
    return true;
}

bool GenRopeTrailData(std::span<const std::unique_ptr<ParticleInstance>> instances,
                      bool thick_format, ParticleVertexWriter& writer,
                      float alpha_multiplier) {
    for (size_t instance_index = 0; instance_index < instances.size(); instance_index++) {
        const auto& instance = *instances[instance_index];
        if (instance.IsNoLiveParticle()) continue;
        const auto particles = instance.Particles();
        const auto trails    = instance.Trails();
        if (particles.size() != trails.size()) {
            const auto& diagnostics = writer.Diagnostics();
            LOG_ERROR("particle trail slot invariant violated object='%.*s' material='%.*s' "
                      "renderer=%s expansion=%s instance=%zu particles=%zu trails=%zu",
                      static_cast<int>(diagnostics.object_name.size()),
                      diagnostics.object_name.data(),
                      static_cast<int>(diagnostics.material_name.size()),
                      diagnostics.material_name.data(),
                      diagnostics.renderer_name,
                      diagnostics.expansion_name,
                      instance_index,
                      particles.size(),
                      trails.size());
            return false;
        }
        for (size_t slot = 0; slot < particles.size(); slot++) {
            if (! ParticleModify::LifetimeOk(particles[slot])) continue;
            if (! GenRopeTrailSegments(instance,
                                       particles[slot],
                                       trails[slot],
                                       thick_format,
                                       writer,
                                       alpha_multiplier)) {
                return false;
            }
        }
    }
    return true;
}

void ResetParticleOutput(SceneMesh& mesh) noexcept {
    if (mesh.VertexCount() > 0) mesh.GetVertexArray(0).ResetSize();
    if (mesh.IndexCount() > 0) {
        mesh.GetIndexArray(0).SetPackedUint16RenderDataCount(0);
    }
}

bool ValidateParticleMeshContract(const SceneMesh& mesh, const ParticleRenderPlan& plan,
                                  const ParticleRenderDiagnostics& diagnostics) {
    const bool shader_selection_valid =
        plan.IsRope() == (plan.shader_selection == ParticleShaderSelection::StockRope);
    const bool expansion_valid =
        ! plan.IsRope() || plan.expansion == ParticleExpansionMode::GeometryPoint;
    const MeshPrimitive expected_primitive =
        plan.IsRope() || plan.expansion == ParticleExpansionMode::GeometryPoint
        ? MeshPrimitive::POINT
        : MeshPrimitive::TRIANGLE;
    const size_t expected_index_count =
        ! plan.IsRope() && plan.expansion == ParticleExpansionMode::IndexedQuad ? 1 : 0;

    if (shader_selection_valid && expansion_valid && mesh.VertexCount() == 1 &&
        mesh.Primitive() == expected_primitive && mesh.IndexCount() == expected_index_count) {
        return true;
    }

    LOG_ERROR("particle render plan mismatch object='%.*s' material='%.*s' renderer=%s "
              "expansion=%s primitive=%u expected-primitive=%u vertices=%zu indices=%zu "
              "expected-indices=%zu shader-selection=%s",
              static_cast<int>(diagnostics.object_name.size()),
              diagnostics.object_name.data(),
              static_cast<int>(diagnostics.material_name.size()),
              diagnostics.material_name.data(),
              diagnostics.renderer_name,
              diagnostics.expansion_name,
              static_cast<unsigned>(mesh.Primitive()),
              static_cast<unsigned>(expected_primitive),
              mesh.VertexCount(),
              mesh.IndexCount(),
              expected_index_count,
              plan.shader_selection == ParticleShaderSelection::Authored ? "Authored"
                                                                          : "StockRope");
    return false;
}

} // namespace

void WPParticleRawGener::GenGLData(std::span<const std::unique_ptr<ParticleInstance>> instances,
                                   SceneMesh& mesh, ParticleRawGenSpecOp& specOp,
                                   const ParticleRenderPlan& plan,
                                   std::string_view object_name, float alpha_multiplier) {
    const auto* material = mesh.Material();
    const std::string_view material_name = material != nullptr
        ? std::string_view(material->name)
        : std::string_view("(unbound-particle-material)");
    const ParticleRenderDiagnostics diagnostics {
        .object_name    = object_name,
        .material_name  = material_name,
        .renderer_name  = ParticleRendererName(plan.renderer),
        .expansion_name = ParticleExpansionName(plan.expansion),
    };

    // Clear the complete live draw prefix before resolving this frame. Any layout, capacity, or
    // topology error therefore leaves the dynamic mesh with an explicit zero draw count rather
    // than exposing vertices or indices produced by an earlier frame.
    ResetParticleOutput(mesh);
    if (! ValidateParticleMeshContract(mesh, plan, diagnostics)) return;

    auto& sv = mesh.GetVertexArray(0);
    const auto layout = ResolveParticleVertexLayout(sv, diagnostics);
    if (! layout.has_value()) return;
    ParticleVertexWriter writer(sv, std::move(*layout), diagnostics);

    bool generated { true };
    switch (plan.renderer) {
    case ParticleRendererKind::Sprite:
    case ParticleRendererKind::SpriteTrail:
        generated = plan.expansion == ParticleExpansionMode::GeometryPoint
            ? GenParticlePointData(
                  instances, specOp, plan.thick_format, writer, alpha_multiplier)
            : GenParticleQuadData(
                  instances, specOp, plan.thick_format, writer, alpha_multiplier);
        break;
    case ParticleRendererKind::Rope:
        for (size_t instance_index = 0; instance_index < instances.size(); instance_index++) {
            const auto& instance = instances[instance_index];
            if (instance->IsNoLiveParticle()) continue;
            if (! GenRopeParticleData(
                    *instance, plan.thick_format, writer, alpha_multiplier)) {
                generated = false;
                break;
            }
        }
        break;
    case ParticleRendererKind::RopeTrail:
        generated =
            GenRopeTrailData(instances, plan.thick_format, writer, alpha_multiplier);
        break;
    }

    if (! generated) {
        ResetParticleOutput(mesh);
        return;
    }

    if (plan.expansion == ParticleExpansionMode::IndexedQuad) {
        constexpr size_t kVerticesPerQuad = 4;
        if (writer.Written() % kVerticesPerQuad != 0) {
            LOG_ERROR("particle quad vertex invariant violated object='%.*s' material='%.*s' "
                      "renderer=%s expansion=%s vertices=%zu",
                      static_cast<int>(diagnostics.object_name.size()),
                      diagnostics.object_name.data(),
                      static_cast<int>(diagnostics.material_name.size()),
                      diagnostics.material_name.data(),
                      diagnostics.renderer_name,
                      diagnostics.expansion_name,
                      writer.Written());
            ResetParticleOutput(mesh);
            return;
        }

        // Index activation happens only after all vertices have been generated successfully. This
        // ordering keeps the indexed draw prefix and the vertex prefix transactional for the frame.
        ParticleQuadIndexWriter index_writer(mesh.GetIndexArray(0), diagnostics);
        if (! index_writer.SetQuadCount(writer.Written() / kVerticesPerQuad)) {
            ResetParticleOutput(mesh);
        }
    }
}

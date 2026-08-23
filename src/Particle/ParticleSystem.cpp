#include "ParticleSystem.h"
#include "Core/Literals.hpp"
#include "Scene/Scene.h"
#include "Scene/SceneNode.h"
#include "ParticleModify.h"
#include "Scene/SceneMesh.h"
#include "Core/Random.hpp"

#include "Utils/Logging.h"

#include <algorithm>
#include <cmath>

using namespace wallpaper;

namespace
{
constexpr float kRuntimeSizeEpsilon = 0.000001f;
constexpr double kControlPointTransformDeterminantEpsilon = 0.000000001;

float SafeRuntimeSizeReference(float size) {
    return std::abs(size) > kRuntimeSizeEpsilon ? size : 1.0f;
}

Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& transform, const Eigen::Vector3d& point) {
    const Eigen::Vector4d transformed = transform * Eigen::Vector4d { point.x(), point.y(), point.z(), 1.0 };
    if (std::abs(transformed.w()) <= kControlPointTransformDeterminantEpsilon) {
        // Homogeneous scene transforms should keep w near one. Falling back to the un-divided xyz
        // vector prevents a malformed matrix from turning the linked control point into infinity.
        return transformed.head<3>();
    }
    return transformed.head<3>() / transformed.w();
}
} // namespace

void ParticleTrail::Reset() noexcept {
    head              = 0;
    sample_count      = 0;
    previous_position = Eigen::Vector3f::Zero();
}

void ParticleTrail::Initialize(const Eigen::Vector3f& position) noexcept {
    Reset();
    if (positions.empty()) return;
    std::fill(positions.begin(), positions.end(), position);
    sample_count      = 1;
    previous_position = position;
}

void ParticleTrail::Push(const Eigen::Vector3f& position) noexcept {
    if (positions.empty()) return;
    head            = static_cast<uint16_t>((static_cast<size_t>(head) + 1) % positions.size());
    positions[head] = position;
    sample_count    = static_cast<uint16_t>(
        std::min(static_cast<size_t>(sample_count) + 1, positions.size()));
}

Eigen::Vector3f ParticleTrail::At(size_t logical_index) const noexcept {
    const size_t length = Length();
    if (logical_index >= length) return Eigen::Vector3f::Zero();
    const size_t capacity = positions.size();
    const size_t oldest   = (static_cast<size_t>(head) + capacity + 1 - length) % capacity;
    return positions[(oldest + logical_index) % capacity];
}

void ParticleInstance::Refresh() {
    SetDeath(false);
    SetNoLiveParticle(false);
    GetBoundedData() = {};
    ParticlesVec().clear();
    TrailsVec().clear();
}

bool ParticleInstance::IsDeath() const { return m_is_death; }
void ParticleInstance::SetDeath(bool v) { m_is_death = v; };

bool ParticleInstance::IsNoLiveParticle() const { return m_no_live_particle; };
void ParticleInstance::SetNoLiveParticle(bool v) { m_no_live_particle = v; };

std::span<const Particle> ParticleInstance::Particles() const { return m_particles; };
std::vector<Particle>&    ParticleInstance::ParticlesVec() { return m_particles; };
std::span<const ParticleTrail> ParticleInstance::Trails() const { return m_trails; };
std::vector<ParticleTrail>&    ParticleInstance::TrailsVec() { return m_trails; };
std::vector<ParticleEmitRuntime>& ParticleInstance::EmitRuntimes() { return m_emit_runtimes; }
std::span<const ParticleEmitRuntime> ParticleInstance::EmitRuntimes() const {
    return m_emit_runtimes;
}

ParticleInstance::BoundedData& ParticleInstance::GetBoundedData() { return m_bounded_data; }
const ParticleInstance::BoundedData& ParticleInstance::GetBoundedData() const {
    return m_bounded_data;
}

ParticleSubSystem::ParticleSubSystem(ParticleSystem& p, std::shared_ptr<SceneMesh> sm,
                                     uint32_t maxcount, double rate, u32 maxcount_instance,
                                     double probability, SpawnType type,
                                     ParticleRenderPlan render_plan,
                                     ParticleRawGenSpecOp specOp, uint16_t trail_length,
                                     double trail_duration,
                                     std::function<void(float)> trail_uniform_update)
    : m_sys(p),
      m_mesh(std::move(sm)),
      m_genSpecOp(std::move(specOp)),
      m_render_plan(render_plan),
      m_maxcount(maxcount),
      m_rate(rate),
      m_time(0),
      m_trail_length(trail_length),
      m_trail_sample_interval(trail_length == 0
                                  ? 0.0
                                  : trail_duration / static_cast<double>(trail_length)),
      m_trail_uniform_update(std::move(trail_uniform_update)),
      m_maxcount_instance(maxcount_instance),
      m_probability(probability),
      m_spawn_type(type) {};

ParticleSubSystem::~ParticleSubSystem() = default;

void ParticleSubSystem::AddEmitter(ParticleEmittOp&& em, ParticleEmitterTiming timing) {
    m_emiters.emplace_back(std::move(em));
    m_emit_timings.push_back(std::move(timing));
}

void ParticleSubSystem::ResetInstanceEmitRuntimes(ParticleInstance& instance) {
    auto& runtimes = instance.EmitRuntimes();
    runtimes.resize(m_emit_timings.size());
    for (size_t i = 0; i < m_emit_timings.size(); ++i) {
        runtimes[i] = MakeParticleEmitRuntime(m_emit_timings[i]);
    }
}

void ParticleSubSystem::AddInitializer(ParticleInitOp&& ini) { m_initializers.emplace_back(ini); }

void ParticleSubSystem::AddOperator(ParticleOperatorOp&& op) { m_operators.emplace_back(op); }

std::span<const ParticleControlpoint> ParticleSubSystem::Controlpoints() const {
    return m_controlpoints;
}
std::span<ParticleControlpoint> ParticleSubSystem::Controlpoints() { return m_controlpoints; };

ParticleSubSystem::SpawnType ParticleSubSystem::Type() const { return m_spawn_type; }

u32 ParticleSubSystem::MaxInstanceCount() const { return m_maxcount_instance; };

void ParticleSubSystem::SetSceneNode(SceneNode* node) { m_node = node; }

void ParticleSubSystem::ApplyRuntimeColorOverrideToParticle(Particle& particle) const {
    if (!m_runtime_color_override.has_value()) return;

    const auto& color = *m_runtime_color_override;
    ParticleModify::InitColorOverride(
        particle, Eigen::Vector3f { color[0], color[1], color[2] });
}

void ParticleSubSystem::ApplyRuntimeColorOverrideToInstances() {
    for (auto& instance : m_instances) {
        if (!instance) continue;
        for (auto& particle : instance->ParticlesVec()) {
            ApplyRuntimeColorOverrideToParticle(particle);
        }
    }
}

void ParticleSubSystem::SetRuntimeColorOverride(const std::array<float, 3>& color) {
    m_runtime_color_override = color;
    // Re-evaluate existing particles from their preserved colorrandom endpoints. This is the same
    // HSV reference-delta operation used during cold initialization and therefore preserves the
    // authored gradient during live property edits.
    // Keep this update local to the subsystem. ParseParticleObj() clears color flags for nested
    // particle assets because their authored colorrandom ranges are independent; recursively
    // forwarding the live scene-layer color would erase that boundary again on the next edit.
    ApplyRuntimeColorOverrideToInstances();
    MarkMeshesDirty();
}

std::optional<std::array<float, 3>> ParticleSubSystem::RuntimeColorOverride() const {
    return m_runtime_color_override;
}

void ParticleSubSystem::SetRuntimeRateOverride(float rate) {
    if (!std::isfinite(rate)) return;

    // Wallpaper treats particle override rate as a non-negative simulation clock multiplier. The
    // same layer-level override is parsed into child emitters, so propagate live script/user edits
    // recursively to keep event-follow trails and their parent particles in the same time domain.
    const float normalized_rate = std::max(0.0f, rate);
    m_rate = normalized_rate;
    m_runtime_rate_override = normalized_rate;

    for (auto& child : m_children) {
        if (child) child->SetRuntimeRateOverride(normalized_rate);
    }
}

std::optional<float> ParticleSubSystem::RuntimeRateOverride() const {
    return m_runtime_rate_override;
}

void ParticleSubSystem::ApplyRuntimeSizeDeltaToParticle(Particle& particle,
                                                        float     size_delta) const {
    particle.init.size *= size_delta;
    particle.size *= size_delta;
}

void ParticleSubSystem::ApplyRuntimeSizeDeltaToInstances(float size_delta) {
    for (auto& instance : m_instances) {
        if (!instance) continue;
        for (auto& particle : instance->ParticlesVec()) {
            ApplyRuntimeSizeDeltaToParticle(particle, size_delta);
        }
    }
}

void ParticleSubSystem::ApplyRuntimeSizeOverrideToNewParticle(Particle& particle) const {
    if (!m_runtime_size_override.has_value()) return;
    if (std::abs(m_runtime_size_ratio - 1.0f) <= kRuntimeSizeEpsilon) return;

    // Newly emitted particles have just received the parse-time instanceoverride.size multiplier
    // from the initializer list. Apply the current live ratio exactly once before the particle is
    // marked old so future Reset() calls preserve the corrected initializer size.
    ApplyRuntimeSizeDeltaToParticle(particle, m_runtime_size_ratio);
}

void ParticleSubSystem::SetRuntimeSizeReference(float size) {
    m_runtime_size_reference = SafeRuntimeSizeReference(size);
    m_runtime_size_override  = size;
    m_runtime_size_ratio     = 1.0f;

    for (auto& child : m_children) {
        if (child) child->SetRuntimeSizeReference(size);
    }
}

void ParticleSubSystem::SetRuntimeSizeOverride(float size) {
    if (!m_runtime_size_reference.has_value()) {
        // Dynamic or legacy parse paths should seed the reference during ParseParticleObj(), but
        // falling back to the first runtime value keeps the subsystem stable if an older caller
        // wires size edits before the parser had a chance to record the cold multiplier.
        m_runtime_size_reference = SafeRuntimeSizeReference(size);
    }

    const float reference = SafeRuntimeSizeReference(*m_runtime_size_reference);
    const float next_ratio = size / reference;
    const float current_ratio =
        std::abs(m_runtime_size_ratio) > kRuntimeSizeEpsilon ? m_runtime_size_ratio : 1.0f;
    const float size_delta = next_ratio / current_ratio;

    m_runtime_size_override = size;
    m_runtime_size_ratio    = next_ratio;

    if (std::isfinite(size_delta) && std::abs(size_delta - 1.0f) > kRuntimeSizeEpsilon) {
        // Existing particles already contain the previous live multiplier in init.size. Scaling by
        // the ratio delta changes them to the new multiplier without rebuilding the emitter or
        // compounding the size on every frame.
        ApplyRuntimeSizeDeltaToInstances(size_delta);
        MarkMeshesDirty();
    }

    for (auto& child : m_children) {
        if (child) child->SetRuntimeSizeOverride(size);
    }
}

std::optional<float> ParticleSubSystem::RuntimeSizeOverride() const {
    return m_runtime_size_override;
}

void ParticleSubSystem::UpdateLinkedControlpoints() {
    bool has_linked_controlpoint = std::any_of(m_controlpoints.begin(), m_controlpoints.end(),
                                               [](const ParticleControlpoint& controlpoint) {
                                                   return controlpoint.link_mouse;
                                               });
    if (!has_linked_controlpoint) return;

    Eigen::Vector3d mouse_scene = m_sys.MouseScenePosition();
    Eigen::Matrix4d scene_to_particle_local = Eigen::Matrix4d::Identity();
    if (m_node != nullptr) {
        m_node->UpdateTrans();
        const auto model = m_node->ModelTrans();
        const double determinant = model.determinant();
        if (!std::isfinite(determinant) ||
            std::abs(determinant) <= kControlPointTransformDeterminantEpsilon) {
            // A linked control point must live in the same local coordinate space as particle
            // positions. If the node transform cannot be inverted, keep the previous offset instead
            // of applying forces from a nonsense location.
            LOG_ERROR("ParticleControlPoint: non-invertible node transform for linked mouse control point");
            return;
        }
        scene_to_particle_local = model.inverse();
    }

    for (auto& controlpoint : m_controlpoints) {
        if (!controlpoint.link_mouse) continue;
        if (controlpoint.worldspace) {
            // World-space control point offsets are authored in scene coordinates. Convert the final
            // world target back into particle-local space because emitters/operators compare against
            // Particle::position, which is generated before the SceneNode model transform is applied.
            controlpoint.offset =
                TransformPoint(scene_to_particle_local, mouse_scene + controlpoint.base_offset);
        } else {
            // Pointer-locked local control points still need the pointer converted through the full
            // inverse node transform. A plain mouse-origin subtraction misses layer scale and
            // rotation, which made scaled particle systems (for example fireflies) ignore the force
            // distance threshold.
            controlpoint.offset =
                controlpoint.base_offset + TransformPoint(scene_to_particle_local, mouse_scene);
        }
    }
}

Eigen::Vector3f ParticleSubSystem::ResolveEventAnchorPosition(
    const Eigen::Vector3f& parent_position) {
    if (m_node == nullptr) return parent_position;

    const Eigen::Matrix4d local_transform = m_node->GetLocalTrans();
    const Eigen::Matrix3d local_linear    = local_transform.block<3, 3>(0, 0);
    const double          determinant     = local_linear.determinant();
    if (!std::isfinite(determinant) ||
        std::abs(determinant) <= kControlPointTransformDeterminantEpsilon) {
        if (!m_logged_event_anchor_transform_error) {
            // Event-spawned children are anchored at a parent particle position, then their own
            // child transform is applied by the scene node. The anchor must therefore be expressed
            // in the inverse child basis; if that basis is singular, keep the old raw anchor and
            // log once so the broken authored transform can be diagnosed without flooding frames.
            LOG_ERROR("ParticleEventAnchor: non-invertible child transform for event particle");
            m_logged_event_anchor_transform_error = true;
        }
        return parent_position;
    }

    // Wallpaper's event child transform is applied around the spawned child system, not around the
    // parent particle that triggered it. Convert only the followed parent coordinate through the
    // inverse child linear basis so child scale/rotation enlarge and rotate the glow/trail itself
    // without pushing the glow away from the firefly center. The child translation remains authored
    // as the event offset and is intentionally left out of this inverse.
    return (local_linear.inverse() * parent_position.cast<double>()).cast<float>();
}

void ParticleSubSystem::AddChild(std::unique_ptr<ParticleSubSystem>&& child) {
    m_children.emplace_back(std::move(child));
}

void ParticleSubSystem::SynchronizeTrailSlots(ParticleInstance& instance) {
    if (m_trail_length == 0) return;
    auto& particles = instance.ParticlesVec();
    auto& trails    = instance.TrailsVec();
    trails.resize(particles.size());
    for (auto& trail : trails) {
        if (trail.positions.size() == m_trail_length) continue;
        trail.positions.assign(m_trail_length, Eigen::Vector3f::Zero());
        trail.Reset();
    }
}

void ParticleSubSystem::SampleTrailHistory(double frame_time) {
    if (m_trail_length == 0) return;

    const double delta = std::max(0.0, frame_time);
    size_t       sample_steps { 0 };
    double       remainder { 0.0 };
    if (m_trail_sample_interval > 0.0) {
        const double elapsed = m_trail_sample_accumulator + delta;
        const double total_steps = std::floor(elapsed / m_trail_sample_interval);
        remainder = elapsed - total_steps * m_trail_sample_interval;
        if (remainder < 0.0) remainder = 0.0;
        if (remainder >= m_trail_sample_interval) {
            remainder = std::fmod(remainder, m_trail_sample_interval);
        }
        m_trail_sample_accumulator = remainder;
        sample_steps = total_steps >= static_cast<double>(m_trail_length)
            ? m_trail_length
            : static_cast<size_t>(total_steps);
        if (m_trail_uniform_update) {
            m_trail_uniform_update(static_cast<float>(
                std::clamp(remainder / m_trail_sample_interval, 0.0, 1.0)));
        }
    } else {
        sample_steps = 1;
        if (m_trail_uniform_update) m_trail_uniform_update(1.0f);
    }

    // History is sampled only after every particle operator has finished. Positions remain in
    // simulation space; the bounded event anchor is applied once by the extraction helper.
    for (auto& instance_ptr : m_instances) {
        auto& instance  = *instance_ptr;
        auto& particles = instance.ParticlesVec();
        auto& trails    = instance.TrailsVec();
        if (trails.size() != particles.size()) {
            LOG_ERROR("particle trail slot invariant violated particles=%zu trails=%zu",
                      particles.size(),
                      trails.size());
            return;
        }
        for (size_t index = 0; index < particles.size(); index++) {
            const auto& particle = particles[index];
            if (! ParticleModify::LifetimeOk(particle)) continue;

            auto& trail = trails[index];
            if (trail.Length() == 0) {
                // Initialize the fixed ring from the operator-complete head on its first sample
                // pass. The topology is fully degenerate for the first extracted frame instead of
                // drawing back to the spawn position before the authored movement operator ran.
                trail.Initialize(particle.position);
                continue;
            }

            const Eigen::Vector3f previous = trail.previous_position;
            for (size_t sample = 0; sample < sample_steps; sample++) {
                const double age = remainder +
                    static_cast<double>(sample_steps - sample - 1) * m_trail_sample_interval;
                const double amount = delta > 0.0
                    ? std::clamp((delta - age) / delta, 0.0, 1.0)
                    : 1.0;
                trail.Push(previous +
                           (particle.position - previous) * static_cast<float>(amount));
            }
            trail.previous_position = particle.position;
        }
    }
}

ParticleInstance* ParticleSubSystem::QueryNewInstance() {
    if (Random::get(0.0, 1.0) <= m_probability) {
        for (auto& inst : m_instances) {
            if (inst->IsDeath() && inst->IsNoLiveParticle()) {
                inst->Refresh();
                ResetInstanceEmitRuntimes(*inst);
                return inst.get();
            }
        }
        if (m_instances.size() < m_maxcount_instance) {
            m_instances.emplace_back(std::make_unique<ParticleInstance>());
            ResetInstanceEmitRuntimes(*m_instances.back());
            return m_instances.back().get();
        }
    }
    return nullptr;
}

void ParticleSubSystem::MarkMeshesDirty() {
    if (m_mesh) m_mesh->SetDirty();
    for (auto& extra : m_extra_outputs) {
        if (extra.mesh) extra.mesh->SetDirty();
    }
}

void ParticleSubSystem::AddRenderOutput(std::shared_ptr<SceneMesh> mesh, ParticleRenderPlan plan,
                                        ParticleRawGenSpecOp spec_op) {
    if (! mesh) return;
    m_extra_outputs.push_back({ std::move(mesh), std::move(plan), std::move(spec_op) });
}

void ParticleSubSystem::Prewarm(double seconds) {
    if (! std::isfinite(seconds) || seconds <= 0.0) return;

    // Particle-system starttime is advanced with a fixed step chosen from the
    // authored maxcount: 0.2s when that count is at least 500, otherwise 0.05s.
    // Each iteration uses a full step, so the last tick can pass starttime.
    // Mesh upload stays off for the whole pass; the live clock is restored after.
    const double step  = m_maxcount >= 500u ? 0.2 : 0.05;
    const double saved = m_sys.scene.frameTime;
    const double saved_rate = m_rate;

    // starttime is advanced with raw steps. Layer rate is applied only on displayed ticks.
    m_rate              = 1.0;
    m_suppress_mesh_gen = true;
    double accumulated  = 0.0;
    while (accumulated < seconds) {
        m_sys.scene.frameTime = step;
        Emitt();
        accumulated += step;
    }
    m_suppress_mesh_gen   = false;
    m_sys.scene.frameTime = saved;
    m_rate                = saved_rate;
}

void ParticleSubSystem::Emitt() {
    double frameTime    = m_sys.scene.frameTime;
    double particleTime = frameTime * m_rate;
    m_time += particleTime;

    auto spawn_inst = [](ParticleInstance& inst, ParticleSubSystem& child, isize idx) {
        ParticleInstance* n_inst = child.QueryNewInstance();
        if (n_inst != nullptr) {
            n_inst->GetBoundedData() = {
                .parent       = &inst,
                .particle_idx = idx,
            };
        }
    };

    UpdateLinkedControlpoints();
    if (m_spawn_type == SpawnType::STATIC) {
        if (m_instances.empty()) {
            m_instances.emplace_back(std::make_unique<ParticleInstance>());
            ResetInstanceEmitRuntimes(*m_instances.back());
        }
    }

    for (auto& inst : m_instances) {
        assert(inst);

        auto& bounded_data = inst->GetBoundedData();

        bool type_has_death =
            m_spawn_type == SpawnType::EVENT_SPAWN || m_spawn_type == SpawnType::EVENT_FOLLOW;

        // bouded data and death
        if (bounded_data.parent != nullptr) {
            std::span particles = bounded_data.parent->Particles();
            if (bounded_data.particle_idx != -1 && bounded_data.particle_idx < particles.size()) {
                auto& p          = particles[bounded_data.particle_idx];
                const auto parent_pos = ParticleModify::GetPos(p);
                bounded_data.pos = ResolveEventAnchorPosition(parent_pos);
                // only update pos once when event_death
                if (m_spawn_type == SpawnType::EVENT_DEATH) {
                    bounded_data.particle_idx = -1;
                }

                // death if bounded particle death
                if (! inst->IsDeath() && type_has_death) {
                    bool cur_life_ok = ParticleModify::LifetimeOk(p);
                    inst->SetDeath(! cur_life_ok && bounded_data.pre_lifetime_ok);
                    bounded_data.pre_lifetime_ok = cur_life_ok;
                }
            }

            // death if parent death
            if (! inst->IsDeath() && type_has_death) {
                inst->SetDeath(bounded_data.parent->IsDeath());
            }
        }

        // clear when death if follow
        if (inst->IsDeath() && m_spawn_type == SpawnType::EVENT_FOLLOW) {
            inst->ParticlesVec().clear();
            inst->TrailsVec().clear();
        }

        if (! inst->IsDeath()) {
            auto& runtimes = inst->EmitRuntimes();
            if (runtimes.size() != m_emiters.size()) {
                ResetInstanceEmitRuntimes(*inst);
            }
            for (size_t i = 0; i < m_emiters.size(); ++i) {
                m_emiters[i](inst->ParticlesVec(),
                             m_initializers,
                             m_controlpoints,
                             m_maxcount,
                             particleTime,
                             m_time,
                             m_next_spawn_sequence,
                             runtimes[i]);
            }
        }

        SynchronizeTrailSlots(*inst);

        // event_death is always death after emitop
        if (m_spawn_type == SpawnType::EVENT_DEATH) inst->SetDeath(true);

        ParticleInfo info {
            .particles     = inst->ParticlesVec(),
            .controlpoints = m_controlpoints,
            .time          = m_time,
            .time_pass     = particleTime,
        };

        bool  has_live = false;
        isize i        = -1;
        for (auto& p : info.particles) {
            i++;

            if (ParticleModify::IsNew(p)) {
                if (m_trail_length != 0) {
                    auto& trail = inst->TrailsVec()[static_cast<size_t>(i)];
                    trail.Reset();
                }
                // new spawn
                for (auto& child : m_children) {
                    if (child->Type() == SpawnType::EVENT_FOLLOW ||
                        child->Type() == SpawnType::EVENT_SPAWN)
                        spawn_inst(*inst, *child, i);
                }
                ApplyRuntimeSizeOverrideToNewParticle(p);
                ApplyRuntimeColorOverrideToParticle(p);
            }

            ParticleModify::MarkOld(p);
            if (! ParticleModify::LifetimeOk(p)) {
                continue;
            }
            ParticleModify::Reset(p);
            ParticleModify::ChangeLifetime(p, -particleTime);

            if (! ParticleModify::LifetimeOk(p)) {
                // new dead
                for (auto& child : m_children) {
                    if (child->Type() == SpawnType::EVENT_DEATH) spawn_inst(*inst, *child, i);
                }
            } else {
                has_live = true;
            }
        }

        inst->SetNoLiveParticle(! has_live);

        std::for_each(m_operators.begin(), m_operators.end(),
                      [&info](ParticleOperatorOp& op) { op(info); });

        for (size_t particle_index = 0; particle_index < info.particles.size();
             particle_index++) {
            auto& particle = info.particles[particle_index];
            if (! particle.operatorDeleted) continue;

            particle.operatorDeleted = false;
            for (auto& child : m_children) {
                if (child->Type() == SpawnType::EVENT_DEATH) {
                    spawn_inst(*inst, *child, static_cast<isize>(particle_index));
                }
            }
        }
        has_live = std::any_of(
            info.particles.begin(), info.particles.end(), [](const Particle& particle) {
                return ParticleModify::LifetimeOk(particle);
            });
        inst->SetNoLiveParticle(! has_live);
    }

    SampleTrailHistory(frameTime);

    const std::string_view object_name = m_node != nullptr
        ? std::string_view(m_node->Name())
        : std::string_view("(unbound-particle-object)");
    if (! m_suppress_mesh_gen) {
        MarkMeshesDirty();
        m_sys.gener->GenGLData(m_instances, *m_mesh, m_genSpecOp, m_render_plan, object_name);
        for (auto& extra : m_extra_outputs) {
            if (! extra.mesh) continue;
            m_sys.gener->GenGLData(m_instances, *extra.mesh, extra.spec_op, extra.plan, object_name);
        }
    }

    for (auto& child : m_children) {
        const bool saved_suppress = child->m_suppress_mesh_gen;
        child->m_suppress_mesh_gen = m_suppress_mesh_gen;
        child->Emitt();
        child->m_suppress_mesh_gen = saved_suppress;
    }
}

void ParticleSystem::Emitt() {
    for (auto& el : subsystems) {
        el->Emitt();
    }
}

void ParticleSystem::SetMousePos(float x, float y) { m_mouse_pos = { x, y }; }

std::array<float, 2> ParticleSystem::MousePos() const { return m_mouse_pos; }

void ParticleSystem::MarkNeedsAudioSpectrum() { m_needs_audio_spectrum = true; }

bool ParticleSystem::NeedsAudioSpectrum() const { return m_needs_audio_spectrum; }

void ParticleSystem::SetAudioSpectrum(std::span<const float> left, std::span<const float> right) {
    if (left.size() < kParticleAudioBandCount || right.size() < kParticleAudioBandCount) {
        ClearAudioSpectrum();
        return;
    }
    for (size_t band = 0; band < kParticleAudioBandCount; band++) {
        const float left_value  = left[band];
        const float right_value = right[band];
        m_audio_left[band]  = std::isfinite(left_value) ? std::max(0.0f, left_value) : 0.0f;
        m_audio_right[band] = std::isfinite(right_value) ? std::max(0.0f, right_value) : 0.0f;
    }
    m_audio_spectrum_valid = true;
}

void ParticleSystem::ClearAudioSpectrum() { m_audio_spectrum_valid = false; }

double ParticleSystem::AudioResponseFactor(const ParticleAudioResponseParams& params) const {
    if (params.mode == 0) return 1.0;

    const size_t start = std::min<size_t>(params.frequency_start,
                                          kParticleAudioBandCount - 1);
    const size_t end = std::min<size_t>(params.frequency_end, kParticleAudioBandCount - 1);
    const size_t lower_band = std::min(start, end);
    const size_t upper_band = std::max(start, end);
    /*
     * Scan the authored closed interval for a peak. Stereo mode first averages the left/right
     * values of each individual band, then compares that band with the running peak. This is
     * different from averaging the whole interval or independently taking channel peaks. Unknown
     * nonzero modes leave the peak at zero and still pass it through the authored response curve.
     */
    double peak = 0.0;
    for (size_t band = lower_band; band <= upper_band; band++) {
        const double left = m_audio_spectrum_valid ? m_audio_left[band] : 0.0;
        const double right = m_audio_spectrum_valid ? m_audio_right[band] : 0.0;
        double band_value = 0.0;
        switch (params.mode) {
        case 1: band_value = left; break;
        case 2: band_value = right; break;
        case 3: band_value = (left + right) * 0.5; break;
        default: break;
        }
        peak = std::max(peak, band_value);
    }

    const double lower = static_cast<double>(params.bounds[0]);
    const double upper = static_cast<double>(params.bounds[1]);
    const double normalized = (peak - lower) / (upper - lower);
    const double t = std::isnan(normalized) ? 1.0 : std::clamp(normalized, 0.0, 1.0);
    const double smooth = t * t * (3.0 - 2.0 * t);
    const double powered = std::pow(smooth, static_cast<double>(params.exponent));
    return std::isnan(powered) ? 1.0 : std::clamp(powered, 0.0, 1.0);
}

Eigen::Vector3d ParticleSystem::MouseScenePosition() const {
    const SceneCamera* camera = scene.activeCamera;
    auto global_camera_it = scene.cameras.find("global");
    if (global_camera_it != scene.cameras.end() && global_camera_it->second) {
        camera = global_camera_it->second.get();
    }

    if (camera == nullptr) {
        return Eigen::Vector3d { m_mouse_pos[0] * scene.ortho[0],
                                 (1.0f - m_mouse_pos[1]) * scene.ortho[1],
                                 0.0 };
    }

    Eigen::Vector3d camera_pos = camera->GetPosition();
    double          left = camera_pos.x() - camera->Width() / 2.0;
    double          top  = camera_pos.y() + camera->Height() / 2.0;
    return Eigen::Vector3d { left + m_mouse_pos[0] * camera->Width(),
                             top - m_mouse_pos[1] * camera->Height(),
                             0.0 };
}
ParticleSystem::~ParticleSystem() = default;

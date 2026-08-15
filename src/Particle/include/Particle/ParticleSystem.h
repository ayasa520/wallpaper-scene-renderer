#pragma once
#include "ParticleEmitter.h"
#include "ParticleRenderPlan.h"
#include "Interface/IParticleRawGener.h"
#include "Core/NoCopyMove.hpp"
#include "Core/MapSet.hpp"

#include <memory>
#include <optional>
#include <functional>
#include <fstream>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace wallpaper
{

class SceneNode;

enum class ParticleAnimationMode
{
    SEQUENCE,
    RANDOMONE,
};

class ParticleSystem;

struct ParticleTrail {
    std::vector<Eigen::Vector3f> positions;
    uint16_t                     head { 0 }; // Newest ring element.
    uint16_t                     sample_count { 0 }; // Real samples, capped at positions.size().
    Eigen::Vector3f              previous_position { Eigen::Vector3f::Zero() };

    void            Reset() noexcept;
    void            Initialize(const Eigen::Vector3f& position) noexcept;
    void            Push(const Eigen::Vector3f& position) noexcept;
    // A zero sample count marks an unused slot. Once initialized, the complete fixed-size ring is
    // emitted so the geometry shader receives stable topology while sample_count tracks real age.
    size_t          Length() const noexcept { return sample_count == 0 ? 0 : positions.size(); }
    Eigen::Vector3f At(size_t logical_index) const noexcept;
};

class ParticleInstance : NoCopy, NoMove {
public:
    struct BoundedData {
        ParticleInstance* parent { nullptr };
        isize             particle_idx { -1 };

        bool            pre_lifetime_ok { true };
        Eigen::Vector3f pos { 0.0f, 0.0f, 0.0f };
    };

    void Refresh();

    bool IsDeath() const;
    void SetDeath(bool);

    bool IsNoLiveParticle() const;
    void SetNoLiveParticle(bool);

    std::span<const Particle>      Particles() const;
    std::vector<Particle>&         ParticlesVec();
    std::span<const ParticleTrail> Trails() const;
    std::vector<ParticleTrail>&    TrailsVec();

    BoundedData& GetBoundedData();
    const BoundedData& GetBoundedData() const;

private:
    bool                       m_is_death { false };
    bool                       m_no_live_particle { false };
    std::vector<Particle>      m_particles;
    std::vector<ParticleTrail> m_trails;
    BoundedData                m_bounded_data;
};

class ParticleSubSystem : NoCopy, NoMove {
public:
    enum class SpawnType
    {
        STATIC,
        EVENT_FOLLOW,
        EVENT_SPAWN,
        EVENT_DEATH,
    };

public:
    ParticleSubSystem(ParticleSystem& p, std::shared_ptr<SceneMesh> sm, uint32_t maxcount,
                      double rate, u32 maxcount_instance, double probability, SpawnType type,
                      ParticleRenderPlan render_plan, ParticleRawGenSpecOp specOp,
                      uint16_t trail_length = 0,
                      double trail_duration = 0.0,
                      std::function<void(float)> trail_uniform_update = {});
    ~ParticleSubSystem();

    void Emitt();

    ParticleInstance* QueryNewInstance();

    void AddEmitter(ParticleEmittOp&&);
    void AddInitializer(ParticleInitOp&&);
    void AddOperator(ParticleOperatorOp&&);

    void AddChild(std::unique_ptr<ParticleSubSystem>&&);

    std::span<const ParticleControlpoint> Controlpoints() const;
    std::span<ParticleControlpoint>       Controlpoints();

    SpawnType Type() const;
    u32       MaxInstanceCount() const;
    void      SetSceneNode(SceneNode* node);
    void      SetRuntimeColorOverride(const std::array<float, 3>& color);
    std::optional<std::array<float, 3>> RuntimeColorOverride() const;
    void      SetRuntimeRateOverride(float rate);
    std::optional<float> RuntimeRateOverride() const;
    void      SetRuntimeSizeReference(float size);
    void      SetRuntimeSizeOverride(float size);
    std::optional<float> RuntimeSizeOverride() const;

    struct TrajectorySample {
        uint64_t        spawn_sequence { 0 };
        float           lifetime_passed { 0.0f };
        Eigen::Vector3f position { Eigen::Vector3f::Zero() };
        Eigen::Vector3f velocity { Eigen::Vector3f::Zero() };
    };

    struct TrajectorySnapshot {
        int32_t                       layer_id { -1 };
        std::string                   layer_name;
        uint64_t                      frame_index { 0 };
        double                        simulation_time { 0.0 };
        size_t                        live_particle_count { 0 };
        Eigen::Vector3f               live_position_min { Eigen::Vector3f::Zero() };
        Eigen::Vector3f               live_position_max { Eigen::Vector3f::Zero() };
        std::vector<TrajectorySample> samples;
    };

    // Returns a bounded snapshot of post-operator simulation state. This is intentionally a
    // read-only diagnostics surface: callers can compare particle trajectories without coupling
    // capture code to the subsystem's slot-reuse and child-instance storage internals.
    TrajectorySnapshot CaptureTrajectorySnapshot(size_t sample_limit) const;

private:
    void UpdateLinkedControlpoints();
    Eigen::Vector3f ResolveEventAnchorPosition(const Eigen::Vector3f& parent_position);
    void ApplyRuntimeColorOverrideToParticle(Particle& particle) const;
    void ApplyRuntimeColorOverrideToInstances();
    void ApplyRuntimeSizeDeltaToParticle(Particle& particle, float size_delta) const;
    void ApplyRuntimeSizeDeltaToInstances(float size_delta);
    void ApplyRuntimeSizeOverrideToNewParticle(Particle& particle) const;
    void SynchronizeTrailSlots(ParticleInstance& instance);
    void SampleTrailHistory(double frame_time);
    void WriteTrajectoryDiagnostics();

    ParticleSystem&            m_sys;
    std::shared_ptr<SceneMesh> m_mesh;
    //	std::vector<std::unique_ptr<ParticleEmitter>> m_emiters;
    std::vector<ParticleEmittOp> m_emiters;

    // std::vector<Particle>           m_particles;
    std::vector<ParticleInitOp>     m_initializers;
    std::vector<ParticleOperatorOp> m_operators;

    std::array<ParticleControlpoint, 8> m_controlpoints;

    ParticleRawGenSpecOp m_genSpecOp;
    ParticleRenderPlan   m_render_plan;
    u32                  m_maxcount;
    // Wallpaper particle `instanceoverride.rate` scales the subsystem simulation clock, not just
    // the spawn count. Keep it mutable so audio/user scripts nested under instanceoverride can
    // speed up gravity, lifetime decay, and emitter timers after the particle system was parsed.
    double m_rate;
    // Keep the live rate override separate from the parsed particle clock so script init can
    // distinguish "no runtime value has been applied yet" from a parser fallback. Audio-reactive
    // rate scripts commonly capture init(value) as their base multiplier; returning parsed m_rate
    // here would seed them with the already-reduced cold value and shrink every update twice.
    std::optional<float>       m_runtime_rate_override;
    double                     m_time;
    uint64_t                   m_frame_index { 0 };
    uint64_t                   m_next_spawn_sequence { 0 };
    uint16_t                   m_trail_length { 0 };
    double                     m_trail_sample_interval { 0.0 };
    double                     m_trail_sample_accumulator { 0.0 };
    std::function<void(float)> m_trail_uniform_update;
    bool                       m_trajectory_diagnostics_initialized { false };
    std::ofstream              m_trajectory_diagnostics_stream;
    std::unordered_map<uint64_t, Eigen::Vector3f> m_trajectory_initial_positions;

    std::vector<std::unique_ptr<ParticleSubSystem>> m_children;
    std::vector<std::unique_ptr<ParticleInstance>>  m_instances;

    u32       m_maxcount_instance { 1 };
    double    m_probability { 1.0f };
    SpawnType m_spawn_type { SpawnType::STATIC };
    SceneNode* m_node { nullptr };
    bool       m_logged_event_anchor_transform_error { false };
    // Runtime particle color edits reuse each particle's authored random range and interpolation so
    // both future spawns and already alive particles receive Wallpaper Engine's HSV reference-delta
    // transform without losing colorrandom variation.
    std::optional<std::array<float, 3>> m_runtime_color_override;
    // Wallpaper particle `instanceoverride.size` is a multiplier baked into each particle's
    // initializer state. Remember the parse-time multiplier as a reference so live edits can scale
    // existing particles by the precise ratio instead of treating the property as an absolute pixel
    // size or compounding the multiplier every frame.
    std::optional<float> m_runtime_size_reference;
    std::optional<float> m_runtime_size_override;
    float                m_runtime_size_ratio { 1.0f };
};

class Scene;
class ParticleSystem : NoCopy, NoMove {
public:
    ParticleSystem(Scene& scene): scene(scene) {};
    ~ParticleSystem();

    void Emitt();
    void SetMousePos(float x, float y);
    std::array<float, 2> MousePos() const;
    Eigen::Vector3d MouseScenePosition() const;

    // Marked during parse when any emitter carries audioprocessingmode != 0, so the render loop
    // only queries the audio spectrum for scenes that actually consume it.
    void MarkNeedsAudioSpectrum();
    bool NeedsAudioSpectrum() const;

    // Per-frame audio loudness snapshot for audio-responsive emitters. Both spans must contain
    // kParticleAudioBandCount normalized band values; Clear marks the spectrum unavailable.
    void SetAudioSpectrum(std::span<const float> left, std::span<const float> right);
    void ClearAudioSpectrum();

    // Collect post-operator state for top-level particle layers. The layer filter is optional and
    // keeps diagnostics consumers independent from subsystem ordering in the parsed scene graph.
    std::vector<ParticleSubSystem::TrajectorySnapshot>
    CaptureTrajectorySnapshots(std::span<const int32_t> layer_ids, size_t sample_limit) const;

    // Inclusive-band peak, smoothstep(bounds), authored power, then saturation. Mode zero is an
    // explicit caller-side bypass; an unavailable spectrum is the same all-zero input consumed by
    // a valid silent source.
    double AudioResponseFactor(const ParticleAudioResponseParams& params) const;

    // Opt-in per-subsystem emit/operator/gen timings. Dormant unless the capture driver supplies
    // VIVID_PARTICLE_CPU_DIAGNOSTICS_PATH; the production hot path stays a single boolean check.
    bool EnsureCpuDiagnostics();
    void WriteCpuDiagnostics(int32_t layer_id, std::string_view layer_name, uint64_t frame_index,
                             double simulation_time, size_t live_count, size_t storage_count,
                             double emit_ms, double lifetime_ms, double operators_ms,
                             double trail_ms, double gen_ms, double total_ms);

    Scene& scene;

    std::vector<std::unique_ptr<ParticleSubSystem>> subsystems;
    std::unique_ptr<IParticleRawGener>              gener;

private:
    std::array<float, 2> m_mouse_pos { 0.5f, 0.5f };

    bool m_needs_audio_spectrum { false };
    bool m_audio_spectrum_valid { false };
    mutable bool m_audio_response_diagnostics_initialized { false };
    mutable uint64_t m_audio_response_diagnostics_sample { 0 };
    mutable std::ofstream m_audio_response_diagnostics_stream;
    bool m_cpu_diagnostics_initialized { false };
    bool m_cpu_diagnostics_enabled { false };
    std::ofstream m_cpu_diagnostics_stream;
    std::array<float, kParticleAudioBandCount> m_audio_left {};
    std::array<float, kParticleAudioBandCount> m_audio_right {};
};
} // namespace wallpaper

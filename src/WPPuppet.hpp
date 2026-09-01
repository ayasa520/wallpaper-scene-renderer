#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <span>
#include <limits>
#include <Eigen/Geometry>

#include "Core/Literals.hpp"

namespace wallpaper
{

class WPPuppetLayer;

// Axis-aligned bounds used by every puppet contract. Individual fields that use this type must
// state their coordinate space explicitly; keeping validity in the value prevents unrelated
// parser, pose and surface code from inventing separate sentinel conventions.
struct PuppetBounds3D {
    Eigen::Vector3f min { Eigen::Vector3f::Zero() };
    Eigen::Vector3f max { Eigen::Vector3f::Zero() };
    bool            valid { false };

    bool IsFiniteAndOrdered() const noexcept {
        return valid && min.allFinite() && max.allFinite() && (min.array() <= max.array()).all();
    }

    void Include(const Eigen::Vector3f& point) noexcept {
        if (!point.allFinite()) return;
        if (!valid) {
            min = point;
            max = point;
            valid = true;
            return;
        }
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    void Include(const PuppetBounds3D& bounds) noexcept {
        if (!bounds.IsFiniteAndOrdered()) return;
        Include(bounds.min);
        Include(bounds.max);
    }
};

enum class PuppetPoseDomain
{
    AuthoredEnvelope,
    RuntimeMutable,
};

struct PuppetPoseSnapshot {
    std::span<const Eigen::Affine3f> skinning;
    PuppetPoseDomain                  domain { PuppetPoseDomain::AuthoredEnvelope };
    uint64_t                          revision { 0 };
    uint64_t                          frame_serial { 0 };
};

class WPPuppet {
public:
    enum class PlayMode
    {
        Loop,
        Mirror,
        Single
    };
    struct Bone {
        std::string     name;
        Eigen::Affine3f transform { Eigen::Affine3f::Identity() };
        uint32_t        parent { 0xFFFFFFFFu };

        bool noParent() const { return parent == 0xFFFFFFFFu; }
        // prepared
        Eigen::Affine3f offset_trans { Eigen::Affine3f::Identity() };
        /*
        Eigen::Vector3f world_axis_x;
        Eigen::Vector3f world_axis_y;
        Eigen::Vector3f world_axis_z;
        */
    };
    struct Attachment {
        std::string     name;
        uint32_t        bone_index { 0xFFFFFFFFu };
        // Raw MDAT payload. Despite looking like an absolute affine matrix, this transform is
        // authored relative to `bone_index`; it is neither puppet-model space nor world space.
        Eigen::Affine3f transform { Eigen::Affine3f::Identity() };
        // Immutable runtime copy of the same bone-local locator. The complete attachment transform
        // is currentBoneModel * bind_transform * childLocal. Do not remove the bind bone chain with
        // inverse(bindBoneModel): doing so moves every locator attached to a non-root bone.
        Eigen::Affine3f bind_transform { Eigen::Affine3f::Identity() };
    };
    struct BoneFrame {
        Eigen::Vector3f position;
        Eigen::Vector3f angle;
        Eigen::Vector3f scale;

        // prepared
        Eigen::Quaterniond quaternion;
    };
    struct AnimTrans {
        std::vector<float>              extra_track;
        std::vector<float>              main_track;
        std::vector<std::vector<float>> tail_tracks;
    };
    struct BoneFrameCurve {
        std::vector<float> values;
    };
    struct AnimV4Event {
        float              time { 0.0f };
        uint32_t           flags { 0 };
        std::vector<float> values;
    };
    struct AnimEvent {
        uint32_t    time_value { 0 };
        std::string event_json;
    };
    enum class AuthoredBoundsSource
    {
        None,
        MdlaAabb,
        SampledFrames,
    };
    struct Animation {
        i32         id { 0 };
        u32         unk_after_id { 0 };
        double      fps;
        i32         length;
        PlayMode    mode;
        std::string name;

        struct BoneFrames {
            std::vector<BoneFrame> frames;
        };
        std::vector<BoneFrames> bframes_array;

        // Versioned MDLA payloads are retained in their semantic groups instead of being skipped
        // as anonymous bytes. The animation AABB is already expressed in puppet-local coordinates.
        std::optional<AnimTrans>        trans;
        std::vector<BoneFrameCurve>     blend_curves;
        std::vector<AnimV4Event>        v4_events;
        PuppetBounds3D                  authored_pose_bounds;
        AuthoredBoundsSource            authored_bounds_source { AuthoredBoundsSource::None };
        std::vector<BoneFrameCurve>     scalar_curves;
        std::vector<AnimEvent>          events;

        // prepared
        double max_time;
        double frame_time;
        struct InterpolationInfo {
            idx    frame_a;
            idx    frame_b;
            double t;
        };
        // Single-shot puppet animations must stop on their authored last frame instead of
        // wrapping back to frame zero like a looped idle layer.
        double            EndTime() const noexcept;
        InterpolationInfo getInterpolationInfo(double* cur_time) const;
    };

public:
    std::vector<Bone>      bones;
    std::vector<Attachment> attachments;
    std::vector<Animation> anims;

    std::span<const Eigen::Affine3f> genFrame(WPPuppetLayer&, double time) noexcept;
    void                             prepared();
    const Attachment*                FindAttachment(std::string_view name) const noexcept;
    uint32_t                         FindBoneIndex(std::string_view name) const noexcept;
    const Eigen::Affine3f&           BoneModelTransform(uint32_t index) const noexcept;

private:
    std::vector<Eigen::Affine3f> m_final_affines;
    std::vector<Eigen::Affine3f> m_bone_model_affines;
};

class WPPuppetLayer {
    friend class WPPuppet;

public:
    WPPuppetLayer();
    WPPuppetLayer(std::shared_ptr<WPPuppet>);
    ~WPPuppetLayer();

    bool hasPuppet() const { return static_cast<bool>(Runtime().puppet); };

    struct AnimationLayer {
        i32    id { 0 };
        double rate { 1.0f };
        double blend { 1.0f };
        bool   additive { false };
        bool   visible { true };
        double cur_time { 0.0f };
        bool   playing { true };
        // NotifyAnimationLayersAdvanced consumes this latch after it fires ended callbacks.
        // This lets single-shot layers report completion without pretending that they wrapped.
        bool   pending_ended_callback { false };
    };

    void prepared(std::span<const AnimationLayer>);
    // Runtime user properties can toggle an animation layer after the puppet has been prepared.
    // The layer list and animation pointers stay stable, but an already cached pose must be
    // invalidated whenever visibility or blend changes.
    void RefreshBlendState() noexcept;

    std::span<const Eigen::Affine3f> genFrame(double time) noexcept;
    PuppetPoseSnapshot AdvanceIfNeeded(double time, uint64_t frame_serial) noexcept;
    PuppetPoseSnapshot PoseSnapshot() const noexcept;
    std::span<const Eigen::Affine3f> SkinningMatrices() const noexcept {
        return Runtime().cached_skinning;
    }
    const WPPuppet* Puppet() const noexcept { return Runtime().puppet.get(); }
    usize AnimationLayerCount() const noexcept { return Runtime().layers.size(); }
    const AnimationLayer*            AnimationLayerState(usize index) const noexcept;
    AnimationLayer*                  AnimationLayerState(usize index) noexcept;
    const WPPuppet::Animation*       AnimationDefinition(usize index) const noexcept;
    bool SetLocalBoneTransform(usize index, const Eigen::Affine3f& transform) noexcept;
    PuppetPoseDomain PoseDomain() const noexcept;
    uint64_t PoseRevision() const noexcept;
    const void* RuntimeIdentity() const noexcept;

    void updateInterpolation(double time) noexcept;

private:
    struct Layer {
        AnimationLayer                         anim_layer;
        const WPPuppet::Animation*             anim { nullptr };
        WPPuppet::Animation::InterpolationInfo interp_info {};

        operator bool() const noexcept { return anim != nullptr; };
    };
    struct BoneOverride {
        bool            enabled { false };
        Eigen::Affine3f local_transform { Eigen::Affine3f::Identity() };
    };
    struct RuntimeState {
        std::vector<Layer>               layers;
        std::vector<BoneOverride>        bone_overrides;
        std::shared_ptr<WPPuppet>        puppet;
        std::span<const Eigen::Affine3f> cached_skinning {};
        uint64_t cached_frame_serial { std::numeric_limits<uint64_t>::max() };
        uint64_t pose_revision { 0 };
        PuppetPoseDomain domain { PuppetPoseDomain::AuthoredEnvelope };
    };

    RuntimeState& Runtime() noexcept { return *m_runtime; }
    const RuntimeState& Runtime() const noexcept { return *m_runtime; }
    void MarkRuntimePoseMutation() noexcept;

    std::shared_ptr<RuntimeState> m_runtime;
};

} // namespace wallpaper

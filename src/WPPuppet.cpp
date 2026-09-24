#include "WPPuppet.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include "Utils/Logging.h"

using namespace wallpaper;
using namespace Eigen;

static Quaterniond ToQuaternion(Vector3f euler) {
    const std::array<Vector3d, 3> axis { Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ() };
    return AngleAxis<double>(euler.z(), axis[2]) * AngleAxis<double>(euler.y(), axis[1]) *
           AngleAxis<double>(euler.x(), axis[0]);
};

static Quaternionf SpringRotation(const Vector3f& angles) {
    return AngleAxisf(angles.z(), Vector3f::UnitZ()) *
           AngleAxisf(angles.y(), Vector3f::UnitY()) *
           AngleAxisf(angles.x(), Vector3f::UnitX());
}

static Vector3f SpringAngles(const Matrix3f& rotation) {
    const float z = std::atan2(rotation(1, 0), rotation(0, 0));
    const float y = std::atan2(-rotation(2, 0),
                              std::sqrt(rotation(2, 1) * rotation(2, 1) +
                                        rotation(2, 2) * rotation(2, 2)));
    const float sine = std::sin(z);
    const float cosine = std::cos(z);
    const float x = std::atan2(sine * rotation(0, 2) - cosine * rotation(1, 2),
                              cosine * rotation(1, 1) - sine * rotation(0, 1));
    Vector3f angles {x, y, z};
    // Retain one signed Euler branch for the spring's restoring rotation. Negating these
    // components is a different operation from conjugating a combined-axis quaternion.
    constexpr float pi = std::numbers::pi_v<float>;
    for (auto& angle : angles) {
        angle = angle < 0.0f ? std::fmod(angle - pi, 2.0f * pi) + pi
                             : std::fmod(angle + pi, 2.0f * pi) - pi;
    }
    return angles;
}

void WPPuppet::prepared() {
    std::vector<Affine3f> combined_tran(bones.size());
    for (uint i = 0; i < bones.size(); i++) {
        auto& b = bones[i];
        if (!b.noParent() && b.parent >= i) {
            LOG_ERROR("puppet bone %u has invalid parent index %u during prepare, fallback to root", i, b.parent);
            b.parent = 0xFFFFFFFFu;
        }
        combined_tran[i] =
            (b.noParent() ? Affine3f::Identity() : combined_tran[b.parent]) * b.transform;

        b.offset_trans = combined_tran[i].inverse();
        /*
        b.world_axis_x = (b.offset_trans.linear() *
        Vector3f::UnitX()).normalized(); b.world_axis_y =
        (b.offset_trans.linear() * Vector3f::UnitY()).normalized();
        b.world_axis_z = (b.offset_trans.linear() *
        Vector3f::UnitZ()).normalized();
        */
    }
    for (auto& anim : anims) {
        anim.frame_time = 1.0f / anim.fps;
        anim.max_time   = anim.length / anim.fps;
        for (auto& b : anim.bframes_array) {
            for (auto& f : b.frames) {
                f.quaternion = ToQuaternion(f.angle);
            }
        }
    }

    for (auto& attachment : attachments) {
        if (attachment.bone_index >= combined_tran.size()) {
            LOG_ERROR("puppet attachment '%s' has invalid bone index %u",
                      attachment.name.c_str(),
                      attachment.bone_index);
            attachment.bind_transform = Affine3f::Identity();
            continue;
        }
        // MDAT stores the locator matrix in the owning bone's local space. Runtime attachment
        // evaluation already starts from the current model-space bone frame, so the stable bind
        // contract is currentBoneModel * authoredBoneLocal * childLocal. Converting the authored
        // matrix through inverse(bindBoneModel) here changes that local locator into an unrelated
        // space and displaces every attachment on non-root bones; keep the parsed matrix intact.
        attachment.bind_transform = attachment.transform;

        const auto& bind_bone_model = combined_tran[attachment.bone_index];
        const auto  bind_model      = bind_bone_model * attachment.bind_transform;
        LOG_INFO("ScenePuppetAttachmentBind: name='%s' bone=%u bone-local=[%.3f %.3f %.3f] "
                 "bind-bone-model=[%.3f %.3f %.3f] bind-model=[%.3f %.3f %.3f]",
                 attachment.name.c_str(),
                 attachment.bone_index,
                 attachment.bind_transform.translation().x(),
                 attachment.bind_transform.translation().y(),
                 attachment.bind_transform.translation().z(),
                 bind_bone_model.translation().x(),
                 bind_bone_model.translation().y(),
                 bind_bone_model.translation().z(),
                 bind_model.translation().x(),
                 bind_model.translation().y(),
                 bind_model.translation().z());
    }

    m_final_affines.resize(bones.size());
    m_bone_local_affines.resize(bones.size());
    m_bone_model_affines.resize(bones.size());
}

std::span<const Eigen::Affine3f> WPPuppet::genFrame(WPPuppetLayer& puppet_layer,
                                                    double time,
                                                    const Affine3f& world_from_model,
                                                    bool advance_simulation) noexcept {
    auto& runtime = puppet_layer.Runtime();

    puppet_layer.updateInterpolation(time);

    for (uint i = 0; i < m_final_affines.size(); i++) {
        const auto& bone   = bones[i];
        auto&       affine = m_final_affines[i];

        affine = Affine3f::Identity();
        const Affine3f parent =
            (bone.noParent() || bone.parent >= i) ? Affine3f::Identity() : m_final_affines[bone.parent];

        Vector3f trans { bone.transform.translation() };
        Vector3f scale;
        Matrix3f bind_rotation = bone.transform.linear();
        for (Eigen::Index axis = 0; axis < 3; ++axis) {
            scale[axis] = bind_rotation.col(axis).norm();
            bind_rotation.col(axis) /= scale[axis];
        }
        Quaterniond quat { bind_rotation.cast<double>() };
        const Quaterniond ident { Quaterniond::Identity() };

        for (auto& layer : runtime.layers) {
            auto& alayer = layer.anim_layer;
            if (layer.anim == nullptr || ! alayer.visible) continue;
            assert(i < layer.anim->bframes_array.size());
            if (i >= layer.anim->bframes_array.size()) continue;

            const auto& track = layer.anim->bframes_array[i];
            // A disabled bone track contributes nothing from this animation layer. Preserve
            // the pose accumulated from earlier layers, or the bind pose when none contributed;
            // its retained frame data must not replace any position, rotation or scale channel.
            if (!track.enabled) continue;

            auto&  info       = layer.interp_info;
            auto&  frame_base = track.frames[(usize)0];
            auto&  frame_a    = track.frames[(usize)info.frame_a];
            auto&  frame_b    = track.frames[(usize)info.frame_b];

            const double   blend         = alayer.blend;
            const float    t             = static_cast<float>(info.t);
            const Vector3f sampled_trans = frame_a.position * (1.0f - t) + frame_b.position * t;
            const Vector3f sampled_scale = frame_a.scale * (1.0f - t) + frame_b.scale * t;
            const Quaterniond sampled_quat = frame_a.quaternion.slerp(info.t, frame_b.quaternion);

            // The animation-layer additive flag is stored separately from blend. Later
            // non-additive layers attenuate earlier poses, while additive layers contribute only
            // the delta from frame zero. Applying layers in authored order is the same
            // composition: a normal layer interpolates toward its absolute pose and an additive
            // layer contributes only its delta. Treating every full-weight layer as additive
            // sums several complete locomotion rotations and twists articulated models.
            if (! alayer.additive) {
                trans = trans * static_cast<float>(1.0 - blend) +
                        sampled_trans * static_cast<float>(blend);
                scale = scale * static_cast<float>(1.0 - blend) +
                        sampled_scale * static_cast<float>(blend);
                quat = quat.slerp(blend, sampled_quat);
                continue;
            }

            trans += static_cast<float>(blend) * (sampled_trans - frame_base.position);
            scale += static_cast<float>(blend) * (sampled_scale - frame_base.scale);
            const Quaterniond rotation_delta =
                sampled_quat * frame_base.quaternion.conjugate();
            quat *= ident.slerp(blend, rotation_delta);
        }
        affine.pretranslate(trans);
        affine.rotate(quat.cast<float>());
        affine.scale(scale);
        if (bone.rotation_simulation) {
            const auto& simulation = *bone.rotation_simulation;
            auto& state = runtime.rotation_simulations[i];
            if (advance_simulation && state.initialized) {
                const float dt = static_cast<float>(time);
                const Affine3f current_world = world_from_model * parent * affine;
                const Affine3f current_inverse = current_world.inverse();
                const Quaternionf orientation = SpringRotation(state.angles);
                Vector3f predicted_tip = orientation * simulation.tip;
                const Vector3f previous_tip =
                    current_inverse * (state.previous_world * simulation.tip);
                if (simulation.gravity) {
                    const float scene_scale = world_from_model.linear().colwise().norm().mean();
                    const Vector3f gravity = (current_inverse.linear() * simulation.gravity->direction) *
                                             (simulation.gravity->mass * scene_scale);
                    const Vector3f direction = predicted_tip.normalized();
                    // Gravity contributes only perpendicular to the rotated bone tip. Express
                    // the authored world direction in this bone's current frame and account for
                    // owner scale before projecting it. Subtract that contribution from the
                    // predicted tip so alignment toward its history supplies the gravity turn;
                    // the same adjusted tip determines the existing motion-response weight.
                    predicted_tip -= gravity - direction * direction.dot(gravity);
                }
                const Vector3f predicted_direction = predicted_tip.normalized();
                const Vector3f previous_direction = previous_tip.normalized();
                const Quaternionf identity = Quaternionf::Identity();
                const Quaternionf alignment =
                    predicted_direction.dot(previous_direction) >
                            1.0f - std::numeric_limits<float>::epsilon()
                        ? identity
                        : Quaternionf::FromTwoVectors(predicted_direction, previous_direction);

                // Tip motion is measured in the current bone frame against the previous
                // displayed world pose, including script writes. Bound its per-step response
                // before converting the authored angular response from degrees to radians.
                constexpr float radians_per_degree = std::numbers::pi_v<float> / 180.0f;
                constexpr float max_tip_motion_per_second = 900.0f;
                const float motion = std::min((predicted_tip - previous_tip).norm(),
                                               dt * max_tip_motion_per_second);
                state.velocity *= identity.slerp(
                    std::min(motion * simulation.response * radians_per_degree, 1.0f), alignment);
                if (simulation.restoring_stiffness) {
                    // A rigid bone keeps its displaced orientation when tip motion stops.
                    // Apply the orientation-restoring increment only to springs, preserving
                    // the shared integration, limit correction and damping order below.
                    state.velocity *= identity.slerp(
                        std::min(*simulation.restoring_stiffness * radians_per_degree * dt, 1.0f),
                        SpringRotation(-state.angles));
                }

                // Quaternion velocity is a rotation increment at the reference 60 Hz step.
                // Its step rotation precedes the retained orientation; damping affects the
                // velocity for the following frame, not the pose just produced by this step.
                constexpr float reference_step = 1.0f / 60.0f;
                const Quaternionf step_rotation =
                    identity.slerp(std::min(dt / reference_step, 1.0f), state.velocity);
                state.angles = SpringAngles(step_rotation.toRotationMatrix() *
                                             orientation.toRotationMatrix());
                if (simulation.limits) {
                    const Vector3f unconstrained = state.angles;
                    state.angles = state.angles.cwiseMax(simulation.limits->minimum)
                                               .cwiseMin(simulation.limits->maximum);
                    // Limit the relative angular displacement, leaving the authored animation
                    // pose intact. Remove the clipped rotation from the velocity on the left
                    // before damping, so the next step retains the permitted response instead
                    // of repeatedly driving the bone through its angular boundary.
                    state.velocity = SpringRotation(unconstrained - state.angles).conjugate() *
                                     state.velocity;
                }
                state.velocity = state.velocity.slerp(std::min(dt * simulation.friction, 1.0f),
                                                       identity);
            }
            affine.rotate(SpringRotation(state.angles));
        }
        if (bone.translation_spring) {
            const auto& spring = *bone.translation_spring;
            auto& state = runtime.translation_springs[i];
            const Affine3f world_from_parent = world_from_model * parent;
            if (advance_simulation && state.initialized) {
                // Simulation state stays in world space. The previous displayed origin includes
                // script writes, while the current target comes from the authored/animated pose.
                // A drag therefore affects the next inertia step without replacing velocity or
                // turning the script's pose into a permanent animation override.
                const Vector3f origin = world_from_parent * affine.translation();
                const float dt = static_cast<float>(time);
                const Vector3f displaced = origin + state.displacement;
                const Vector3f corrected = state.displacement -
                    spring.response * (displaced - state.previous_world_origin);
                const Vector3f velocity = state.velocity - (dt * spring.stiffness) * corrected;
                state.displacement = corrected + dt * velocity;
                const float scene_scale = world_from_model.linear().colwise().norm().mean();
                const float distance_limit = scene_scale * spring.max_distance;
                const float distance = state.displacement.norm();
                if (distance_limit > 0.0f && distance > distance_limit) {
                    state.displacement *= distance_limit / distance;
                }
                // Position consumes the undamped velocity. The distance bound limits only
                // displacement; damping is applied afterwards to the retained velocity.
                state.velocity = velocity - std::min(dt * spring.friction, 1.0f) * velocity;
            }
            affine.translation() += world_from_parent.linear().inverse() * state.displacement;
        }
        if (i < runtime.bone_overrides.size() && runtime.bone_overrides[i].enabled) {
            affine = runtime.bone_overrides[i].local_transform;
        }
        // Keep the evaluated local frame before composing its parent. Script component
        // writes must start from these exact values: recovering local space by inverting
        // the model-space parent feeds multiplication roundoff back into every later write.
        // The model and skinning palettes still derive from this same evaluated pose.
        m_bone_local_affines[i] = affine;
        affine = parent * affine;
        m_bone_model_affines[i] = affine;
        if (bone.rotation_simulation) {
            auto& state = runtime.rotation_simulations[i];
            state.previous_world = world_from_model * affine;
            state.initialized = true;
        }
        if (bone.translation_spring) {
            auto& state = runtime.translation_springs[i];
            // Recomposition after a same-frame script write updates the displayed pose for
            // the next frame, but does not perform another integration step.
            state.previous_world_origin = world_from_model * affine.translation();
            state.initialized = true;
        }
    }

    for (uint i = 0; i < m_final_affines.size(); i++) {
        m_final_affines[i] *= bones[i].offset_trans.matrix();
    }
    return m_final_affines;
}

const WPPuppet::Attachment* WPPuppet::FindAttachment(std::string_view name) const noexcept {
    auto it = std::find_if(attachments.begin(), attachments.end(), [name](const auto& attachment) {
        return attachment.name == name;
    });
    return it == attachments.end() ? nullptr : std::addressof(*it);
}

uint32_t WPPuppet::FindBoneIndex(std::string_view name) const noexcept {
    for (uint32_t i = 0; i < bones.size(); ++i) {
        if (bones[i].name == name) return i;
    }
    return 0xFFFFFFFFu;
}

const Affine3f& WPPuppet::BoneLocalTransform(uint32_t index) const noexcept {
    return m_bone_local_affines[index];
}

const Affine3f& WPPuppet::BoneModelTransform(uint32_t index) const noexcept {
    static const Affine3f identity = Affine3f::Identity();
    if (index >= m_bone_model_affines.size()) return identity;
    return m_bone_model_affines[index];
}

static constexpr void genInterpolationInfo(WPPuppet::Animation::InterpolationInfo& info,
                                           double& cur, u32 length, double frame_time,
                                           double max_time) {
    cur          = std::fmod(cur, max_time);
    double _rate = cur / frame_time;

    info.frame_a = ((uint)_rate) % length;
    info.frame_b = (info.frame_a + 1) % length;
    info.t       = _rate - (double)info.frame_a;
}

double WPPuppet::Animation::EndTime() const noexcept {
    if (length <= 1 || frame_time <= std::numeric_limits<double>::epsilon()) return 0.0;
    return frame_time * static_cast<double>(length - 1);
}

WPPuppet::Animation::InterpolationInfo
WPPuppet::Animation::getInterpolationInfo(double* cur_time) const {
    InterpolationInfo _info;
    auto&             _cur_time = *cur_time;

    if (mode == PlayMode::Loop) {
        genInterpolationInfo(_info, _cur_time, (u32)length, frame_time, max_time);
    } else if (mode == PlayMode::Single) {
        // Clamp single-shot layers to the authored end frame so click-triggered animations
        // stay on their last pose instead of wrapping back to the start pose.
        _cur_time = std::clamp(_cur_time, 0.0, EndTime());
        const double frame_position = frame_time <= std::numeric_limits<double>::epsilon()
                                          ? 0.0
                                          : (_cur_time / frame_time);
        const auto last_frame = static_cast<idx>(std::max(length - 1, 0));
        _info.frame_a = std::min(static_cast<idx>(frame_position), last_frame);
        _info.frame_b = std::min(static_cast<idx>(_info.frame_a + 1), last_frame);
        _info.t = _info.frame_a == _info.frame_b
                      ? 0.0
                      : std::clamp(frame_position - static_cast<double>(_info.frame_a), 0.0, 1.0);
    } else if (mode == PlayMode::Mirror) {
        const auto _get_frame = [this](auto f) {
            return f >= length ? (length - 1) - (f - length) : f;
        };
        genInterpolationInfo(_info, _cur_time, (u32)length * 2, frame_time, max_time * 2.0f);
        _info.frame_a = _get_frame(_info.frame_a);
        _info.frame_b = _get_frame(_info.frame_b);
    }

    return _info;
}

void WPPuppetLayer::prepared(std::span<const AnimationLayer> alayers) {
    auto& runtime = Runtime();
    runtime.layers.resize(alayers.size());
    runtime.bone_overrides.assign(runtime.puppet != nullptr ? runtime.puppet->bones.size() : 0,
                                  BoneOverride {});
    runtime.translation_springs.assign(runtime.bone_overrides.size(), TranslationSpringState {});
    runtime.rotation_simulations.assign(runtime.bone_overrides.size(), RotationSimulationState {});
    runtime.cached_skinning     = {};
    runtime.cached_frame_serial = std::numeric_limits<uint64_t>::max();
    runtime.advanced_frame_serial = std::numeric_limits<uint64_t>::max();
    runtime.pose_revision       = 0;
    runtime.domain              = PuppetPoseDomain::AuthoredEnvelope;

    if (runtime.puppet == nullptr) return;
    if (std::any_of(runtime.puppet->bones.begin(), runtime.puppet->bones.end(),
                    [](const auto& bone) {
                        return bone.translation_spring.has_value() || bone.rotation_simulation.has_value();
                    })) {
        runtime.domain = PuppetPoseDomain::RuntimeMutable;
    }

    std::transform(
        alayers.rbegin(), alayers.rend(), runtime.layers.rbegin(),
        [&runtime](const auto& layer) {
            const auto& anims = runtime.puppet->anims;

            auto it = std::find_if(anims.begin(), anims.end(), [&layer](auto& a) {
                return layer.id == a.id;
            });
            const bool has_animation = it != anims.end();

            auto runtime_layer = layer;
            if (has_animation && it->mode == WPPuppet::PlayMode::Single) {
                // Wallpaper Engine uses single-shot puppet layers for event-driven motions such
                // as clicks, so they must start paused until the script explicitly plays them.
                runtime_layer.playing = false;
            }

            // Keep the animation definition even when the authored/user initial state is hidden.
            // User properties may enable the layer later, and dropping the pointer here would make
            // false->true toggles impossible without reparsing the entire puppet.
            return Layer {
                .anim_layer = runtime_layer,
                .anim       = has_animation ? std::addressof(*it) : nullptr,
            };
        });
    RefreshBlendState();
}

void WPPuppetLayer::RefreshBlendState() noexcept {
    auto& runtime = Runtime();
    // Blend and visibility are consumed directly in authored order by genFrame(). Invalidate the
    // cached palette so a script mutation is visible even before the next frame serial is issued.
    runtime.cached_skinning     = {};
    runtime.cached_frame_serial = std::numeric_limits<uint64_t>::max();
    runtime.pose_revision++;
}

std::span<const Eigen::Affine3f> WPPuppetLayer::genFrame(double time) noexcept {
    auto& runtime = Runtime();
    runtime.cached_skinning = runtime.puppet->genFrame(*this, time, Affine3f::Identity(), false);
    return runtime.cached_skinning;
}

PuppetPoseSnapshot WPPuppetLayer::AdvanceIfNeeded(double time,
                                                  uint64_t frame_serial,
                                                  const Affine3f& world_from_model) noexcept {
    auto& runtime = Runtime();
    if (!runtime.puppet) return {};
    // Pose invalidation is independent of clock advancement. Scripts can read, write and read
    // a bone repeatedly before drawing; those reads must all observe one simulation step.
    // An ancestor query may still carry the preceding render serial during script dispatch,
    // so only a newer serial advances time, never a query for an older frame.
    const bool advance = runtime.advanced_frame_serial == std::numeric_limits<uint64_t>::max() ||
                         frame_serial > runtime.advanced_frame_serial;
    if (advance) {
        for (usize i = 0; i < runtime.bone_overrides.size(); ++i) {
            if (runtime.puppet->bones[i].translation_spring ||
                runtime.puppet->bones[i].rotation_simulation) {
                runtime.bone_overrides[i].enabled = false;
            }
        }
        runtime.advanced_frame_serial = frame_serial;
    }
    if (advance || runtime.cached_frame_serial == std::numeric_limits<uint64_t>::max()) {
        runtime.cached_skinning = runtime.puppet->genFrame(
            *this, advance ? time : 0.0, world_from_model, advance);
        runtime.cached_frame_serial = runtime.advanced_frame_serial;
        runtime.pose_revision++;
    }
    return PoseSnapshot();
}

PuppetPoseSnapshot WPPuppetLayer::PoseSnapshot() const noexcept {
    const auto& runtime = Runtime();
    return PuppetPoseSnapshot {
        .skinning = runtime.cached_skinning,
        .domain = runtime.domain,
        .revision = runtime.pose_revision,
        .frame_serial = runtime.cached_frame_serial,
    };
}

void WPPuppetLayer::updateInterpolation(double time) noexcept {
    for (auto& layer : Runtime().layers) {
        if (layer) {
            double current_time = layer.anim_layer.cur_time;
            if (layer.anim_layer.playing) {
                current_time += time * layer.anim_layer.rate;
                if (layer.anim->mode == WPPuppet::PlayMode::Single) {
                    const double end_time = layer.anim->EndTime();
                    const bool reached_boundary =
                        current_time < 0.0 || current_time > end_time;
                    if (reached_boundary) {
                        // Stop single-shot layers at their terminal frame and arm a completion
                        // callback so scripts can replay them on the next explicit play().
                        current_time = layer.anim_layer.rate < 0.0 ? 0.0 : end_time;
                        layer.anim_layer.playing = false;
                        layer.anim_layer.pending_ended_callback = true;
                    }
                }
            }
            layer.interp_info = layer.anim->getInterpolationInfo(&current_time);
            layer.anim_layer.cur_time = current_time;
        }
    }
}

const WPPuppetLayer::AnimationLayer* WPPuppetLayer::AnimationLayerState(usize index) const noexcept {
    const auto& layers = Runtime().layers;
    if (index >= layers.size()) return nullptr;
    return std::addressof(layers[index].anim_layer);
}

WPPuppetLayer::AnimationLayer* WPPuppetLayer::AnimationLayerState(usize index) noexcept {
    auto& layers = Runtime().layers;
    if (index >= layers.size()) return nullptr;
    return std::addressof(layers[index].anim_layer);
}

const WPPuppet::Animation* WPPuppetLayer::AnimationDefinition(usize index) const noexcept {
    const auto& layers = Runtime().layers;
    if (index >= layers.size()) return nullptr;
    return layers[index].anim;
}

bool WPPuppetLayer::SetLocalBoneTransform(usize index, const Eigen::Affine3f& transform) noexcept {
    auto& runtime = Runtime();
    if (!runtime.puppet || index >= runtime.puppet->bones.size()) return false;
    if (index >= runtime.bone_overrides.size()) {
        runtime.bone_overrides.resize(runtime.puppet->bones.size());
    }

    runtime.bone_overrides[index].enabled = true;
    runtime.bone_overrides[index].local_transform = transform;
    MarkRuntimePoseMutation();
    return true;
}

bool WPPuppetLayer::ApplyBonePhysicsImpulse(usize index, const Eigen::Vector3f& impulse,
                                            const Eigen::Vector3f& angular_degrees) noexcept {
    auto& runtime = Runtime();
    if (!runtime.puppet || index >= runtime.puppet->bones.size() ||
        (!runtime.puppet->bones[index].translation_spring &&
         !runtime.puppet->bones[index].rotation_simulation)) return false;

    // Directional impulses accumulate in the simulation's world-space velocity. They do
    // not change the displayed pose or invalidate its current-frame cache: the next frame's
    // integration consumes the new velocity. Repeated calls add independent impulses rather
    // than replacing a pose or resetting the retained spring displacement.
    runtime.translation_springs[index].velocity += impulse;
    // Angular input is expressed in degrees and accumulates on the right of the retained
    // quaternion velocity. It leaves both the displayed pose and its current-frame cache
    // unchanged, just like directional input; the following update consumes the impulse.
    constexpr float radians_per_degree = std::numbers::pi_v<float> / 180.0f;
    runtime.rotation_simulations[index].velocity *= SpringRotation(angular_degrees * radians_per_degree);
    return true;
}

void WPPuppetLayer::ResetBonePhysicsSimulation(usize index) noexcept {
    auto& runtime = Runtime();
    if (!runtime.puppet || index >= runtime.puppet->bones.size()) return;

    // Reset only the retained simulation displacement and velocity. The displayed pose
    // and its world-space history belong to frame evaluation and must remain available:
    // the next integration step uses that history for its motion response. Invalidating
    // the pose cache or clearing initialization here would instead change the current
    // frame's pose or suppress the following frame's response to the previous pose.
    auto& state = runtime.translation_springs[index];
    state.displacement.setZero();
    state.velocity.setZero();
    auto& rotation = runtime.rotation_simulations[index];
    rotation.angles.setZero();
    rotation.velocity.setIdentity();
}

void WPPuppetLayer::MarkRuntimePoseMutation() noexcept {
    auto& runtime = Runtime();
    runtime.domain = PuppetPoseDomain::RuntimeMutable;
    runtime.pose_revision++;
    runtime.cached_skinning = {};
    runtime.cached_frame_serial = std::numeric_limits<uint64_t>::max();
}

PuppetPoseDomain WPPuppetLayer::PoseDomain() const noexcept {
    return Runtime().domain;
}

uint64_t WPPuppetLayer::PoseRevision() const noexcept {
    return Runtime().pose_revision;
}

const void* WPPuppetLayer::RuntimeIdentity() const noexcept {
    return m_runtime.get();
}

WPPuppetLayer::WPPuppetLayer(std::shared_ptr<WPPuppet> pup)
    : m_runtime(std::make_shared<RuntimeState>()) {
    m_runtime->puppet = std::move(pup);
}
WPPuppetLayer::WPPuppetLayer(): m_runtime(std::make_shared<RuntimeState>()) {}
WPPuppetLayer::~WPPuppetLayer() = default;

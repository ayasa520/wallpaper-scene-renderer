#pragma once
// Internal surface shared between the WPSceneScriptHost translation units. The script host is
// being split into cohesive files, and everything here is the state those units exchange: the
// Opaque runtime-state struct and the record types it owns. This header is deliberately not
// part of the public include tree.

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "Core/Literals.hpp"
#include "Scene/SceneTexture.h"
#include "WPDynamicValue.hpp"
#include "WPPropertyAnimation.hpp"
#include "WPSceneScriptHost.hpp"
#include "WPSceneScriptMedia.hpp"
#include "WPUserProperties.hpp"

extern "C" {
#include "quickjs.h"
}

namespace wallpaper
{

class Scene;
class SceneNode;

struct SceneRegistrationRange {
    std::size_t binding_start { 0 };
    std::size_t binding_end { std::numeric_limits<std::size_t>::max() };
    std::size_t animation_start { 0 };
    std::size_t animation_end { std::numeric_limits<std::size_t>::max() };
    std::size_t script_start { 0 };
    std::size_t script_end { std::numeric_limits<std::size_t>::max() };
};

struct TextureAnimationState {
    SpriteAnimation base_animation;
    SpriteAnimation animation;
    double          rate { 1.0 };
};

struct AnimationLayerRuntimeState {
    double               last_time { 0.0 };
    bool                 seen { false };
    std::vector<JSValue> ended_callbacks;
};

struct PropertyAnimationInstance {
    WPSceneScriptRegistration registration;
    uint32_t                  animation_id { 0 };
    WPPropertyAnimationState  state;
};

struct ScriptInstance {
    uint32_t                  instance_id { 0 };
    WPSceneScriptRegistration registration;
    WPDynamicValue            current_value;
    JSValue                   script_properties { JS_UNDEFINED };
    JSValue                   exports { JS_UNDEFINED };
    JSValue                   init_fn { JS_UNDEFINED };
    JSValue                   update_fn { JS_UNDEFINED };
    JSValue                   apply_user_properties_fn { JS_UNDEFINED };
    JSValue                   apply_general_settings_fn { JS_UNDEFINED };
    JSValue                   cursor_enter_fn { JS_UNDEFINED };
    JSValue                   cursor_leave_fn { JS_UNDEFINED };
    JSValue                   cursor_move_fn { JS_UNDEFINED };
    JSValue                   cursor_down_fn { JS_UNDEFINED };
    JSValue                   cursor_up_fn { JS_UNDEFINED };
    JSValue                   cursor_click_fn { JS_UNDEFINED };
    JSValue                   media_thumbnail_changed_fn { JS_UNDEFINED };
    JSValue                   media_properties_changed_fn { JS_UNDEFINED };
    JSValue                   media_playback_changed_fn { JS_UNDEFINED };
    JSValue                   destroy_fn { JS_UNDEFINED };
    JSValue                   resize_screen_fn { JS_UNDEFINED };
    bool                      initialized { false };
};

struct ScriptTimer {
    uint64_t id { 0 };
    uint32_t owner_instance_id { 0 };
    double   remaining_ms { 0.0 };
    double   interval_ms { 0.0 };
    bool     repeat { false };
    JSValue  callback { JS_UNDEFINED };
};

struct AudioBufferBinding {
    uint32_t resolution { 0 };
    JSValue  object { JS_UNDEFINED };
    JSValue  left { JS_UNDEFINED };
    JSValue  right { JS_UNDEFINED };
    JSValue  average { JS_UNDEFINED };
};

constexpr size_t kExternalAudioSampleCount        = 128;
constexpr size_t kSceneAudioChannelBandCount      = 64;
constexpr size_t kSceneAudioEnvelopeGroupCount    = 16;
constexpr size_t kSceneAudioBandsPerEnvelope      = 8;
constexpr float  kSceneAudioSignalThreshold       = 0.0001f;
constexpr float  kSceneAudioEnvelopeFloor         = 0.001f;
constexpr float  kSceneAudioGlobalEnvelopeRatio   = 0.333f;
constexpr float  kSceneAudioEnvelopeAttackRate    = 1.0f;
constexpr float  kSceneAudioEnvelopeReleaseRate   = -0.5f;
constexpr float  kSceneAudioTemporalSmoothingRate = 20.0f;
constexpr float  kSceneAudioSlewRate              = 40.0f;

struct ExternalAudioSpectrumCache {
    bool valid { false };
    std::array<std::vector<float>, 3> left;
    std::array<std::vector<float>, 3> right;
    std::array<std::vector<float>, 3> average;
};

struct ExternalSceneAudioState {
    bool has_snapshot { false };
    std::array<float, kExternalAudioSampleCount> latest_raw {};
    std::array<float, kSceneAudioEnvelopeGroupCount> envelope {};
    std::array<float, kExternalAudioSampleCount> smoothed {};
    std::array<float, kExternalAudioSampleCount> previous_output {};
    ExternalAudioSpectrumCache cache;
};

struct RuntimeState {
    JSRuntime* runtime { nullptr };
    JSContext* context { nullptr };
};

struct WPSceneScriptHost::Opaque {
    Scene*                                       scene { nullptr };
    RuntimeState                                 runtime;
    JSValue                                      shared { JS_UNDEFINED };
    JSValue                                      engine_base { JS_UNDEFINED };
    JSValue                                      console { JS_UNDEFINED };
    JSValue                                      input { JS_UNDEFINED };
    JSValue                                      general_settings_object { JS_UNDEFINED };
    JSValue                                      scene_object { JS_UNDEFINED };
    JSValue                                      native_bridge { JS_UNDEFINED };
    JSValue                                      user_properties_object { JS_UNDEFINED };
    uint32_t                                     next_instance_id { 1 };
    uint32_t                                     next_property_animation_id { 1 };
    uint64_t                                     next_timer_id { 1 };
    double                                       runtime_seconds { 0.0 };
    bool                                         initialized { false };
    bool                                         applying_user_properties { false };
    std::vector<WPSceneScriptRegistration>       property_bindings;
    std::vector<PropertyAnimationInstance>       property_animations;
    std::vector<std::unique_ptr<ScriptInstance>> instances;
    std::vector<ScriptTimer>                     timers;
    std::vector<AudioBufferBinding>              audio_buffers;
    ExternalSceneAudioState                      external_audio;
    std::vector<int32_t>                         pending_destroy_layer_ids;
    UserPropertyMap                              user_properties;
    UserPropertyMap                              dispatched_user_properties;
    std::unordered_set<std::string>              user_property_names;
    std::unordered_map<std::string, std::string> general_settings;
    std::unordered_map<std::string, std::string> dispatched_general_settings;
    std::unordered_set<std::string>              general_setting_names;
    nlohmann::json                               local_storage_global { nlohmann::json::object() };
    nlohmann::json                               local_storage_screen { nlohmann::json::object() };
    std::unordered_map<SceneNode*, std::unordered_map<usize, TextureAnimationState>> texture_states;
    std::unordered_map<SceneNode*, std::unordered_map<usize, AnimationLayerRuntimeState>>
                                 animation_layer_states;
    WPSceneScriptMediaState      media_state;
    WPSceneScriptMediaState      dispatched_media_state;
    std::unordered_set<uint32_t> hovered_instances;
    std::unordered_set<uint32_t> pressed_instances;
    // Layers whose script-driven color routing has already been reported. Animated colors change
    // every frame, so only the first application per layer is logged.
    std::unordered_set<int32_t>  color_apply_reported_layers;
    std::vector<SceneRegistrationRange> pending_scene_registration_ranges;
    // Physical wallpaper output extent (engine.screenResolution). Zero until the wallpaper
    // surface publishes its size; cursor screen coordinates then fall back to the scene canvas.
    std::array<int32_t, 2> screen_size { 0, 0 };
    // Diagnostic aid: instance id of the script update currently executing (0 outside updates).
    uint32_t current_running_instance { 0 };
};

// The textures and render targets a layer tree currently holds resident on the GPU.
struct LayerResidencyResources {
    std::unordered_set<std::string> static_textures;
    std::unordered_set<std::string> video_textures;
    std::unordered_set<std::string> render_targets;
};

// Defined in WPSceneScriptHostResidency.cpp: resource ownership bookkeeping used by dynamic layer
// destruction.
LayerResidencyResources CollectLayerResidencyResources(const Scene& scene, int32_t layer_id);
LayerResidencyResources CollectRetainedResidencyResources(
    const Scene& scene, const std::unordered_set<int32_t>& excluded_layers);
void QueueLayerResourceRelease(Scene& scene, int32_t layer_id,
                               const LayerResidencyResources& retained, const char* reason);

// Defined in WPSceneScriptHost.cpp: the registration and node-resolution helpers the
// residency unit calls back into.
SceneNode*             FindNodeById(WPSceneScriptHost::Opaque* opaque, int32_t node_id);
void                   RebindLayerRegistrations(WPSceneScriptHost::Opaque* opaque,
                                                int32_t layer_id, SceneNode* node);
SceneRegistrationRange CaptureSceneRegistrationRange(WPSceneScriptHost::Opaque* opaque);
bool SceneRegistrationRangeHasNewEntries(WPSceneScriptHost::Opaque* opaque,
                                         const SceneRegistrationRange& range);
void RegisterSceneRegistrationRange(WPSceneScriptHost::Opaque* opaque,
                                    const SceneRegistrationRange& range);
void EnsureTextureAnimationStatesForNode(WPSceneScriptHost::Opaque* opaque, SceneNode* node);

} // namespace wallpaper

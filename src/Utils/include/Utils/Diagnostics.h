#pragma once

#include <cstdint>
#include <optional>

namespace wallpaper::diagnostics
{

inline constexpr bool Enabled = WESCENE_ENABLE_DIAGNOSTICS;

// Capture inputs are immutable for a renderer process. Development builds read them once in
// Diagnostics.cpp; production never compiles that source. A constant empty configuration lets
// the compiler discard trace blocks and their input preparation without per-frame queries,
// initialization guards or a runtime setting capable of enabling diagnostics in a release.
struct Settings {
    bool trace_depth_attachments {};
    bool trace_draw_every_frame {};
    bool trace_effect_phases {};
    bool trace_effect_projection {};
    bool trace_eye_position {};
    bool trace_masked_draw {};
    bool trace_material_state {};
    bool trace_material_types {};
    bool trace_media_state {};
    bool trace_mesh_uploads {};
    bool trace_model_data {};
    bool trace_model_input {};
    bool trace_model_projection {};
    bool trace_object_draw {};
    bool trace_prelighting {};
    bool trace_puppet_vertex_input {};
    bool trace_reflection {};
    bool trace_render_commands {};
    bool trace_render_uniforms {};
    bool trace_scene_clear {};
    bool trace_scene_projection {};
    bool trace_shape_state {};
    bool trace_sound_mix {};
    bool trace_texture_uploads {};
    bool trace_text_background {};
    bool trace_text_color {};
    bool trace_text_depth {};
    bool trace_text_destination {};
    bool trace_text_glyph {};
    bool trace_text_source {};
    bool trace_user_bindings {};
    bool trace_volumetrics {};
    const char* transform_layer {};
    const char* render_target {};
    const char* material_uniform {};
    const char* present_after {};
    const char* frame_trace_directory {};
    const char* frame_trace_draws {};
    bool gpu_profile {};
    bool validation {};
    bool lockstep {};
    std::optional<double> fixed_dt;
    std::optional<double> fixed_epoch;
    std::optional<uint64_t> random_seed;
};

#if WESCENE_ENABLE_DIAGNOSTICS
const Settings& Options();
#else
inline constexpr Settings DisabledSettings {};
inline constexpr const Settings& Options() { return DisabledSettings; }
#endif

} // namespace wallpaper::diagnostics

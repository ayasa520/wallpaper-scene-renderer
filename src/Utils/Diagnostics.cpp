#include "Diagnostics.h"

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace wallpaper::diagnostics
{
namespace
{

std::optional<double> ReadDouble(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return std::nullopt;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text) return std::nullopt;
    return value;
}

std::optional<uint64_t> ReadSeed() {
    const char* text = std::getenv("WESCENE_RANDOM_SEED");
    if (text == nullptr || *text == '\0') return std::nullopt;
    const std::string_view input(text);
    uint64_t value = 0;
    if (std::from_chars(input.data(), input.data() + input.size(), value).ec != std::errc {})
        return std::nullopt;
    return value;
}

} // namespace

const Settings& Options() {
    static const Settings settings = [] {
        Settings result;
        // Most trace switches use presence, including an empty value. Lockstep/validation
        // require exactly "1", while sound-mix tracing accepts nonempty values except "0".
        // Preserve those harness inputs at this development-only boundary.
        result.trace_depth_attachments = std::getenv("WESCENE_TRACE_DEPTH_ATTACHMENTS") != nullptr;
        result.trace_draw_every_frame = std::getenv("WESCENE_TRACE_DRAW_EVERY_FRAME") != nullptr;
        result.trace_effect_phases = std::getenv("WESCENE_TRACE_EFFECT_PHASES") != nullptr;
        result.trace_effect_projection = std::getenv("WESCENE_TRACE_EFFECT_PROJECTION") != nullptr;
        result.trace_eye_position = std::getenv("WESCENE_TRACE_EYE_POSITION") != nullptr;
        result.trace_masked_draw = std::getenv("WESCENE_TRACE_MASKED_DRAW") != nullptr;
        result.trace_material_state = std::getenv("WESCENE_TRACE_MATERIAL_STATE") != nullptr;
        result.trace_material_types = std::getenv("WESCENE_TRACE_MATERIAL_TYPES") != nullptr;
        result.trace_media_state = std::getenv("WESCENE_TRACE_MEDIA_STATE") != nullptr;
        result.trace_mesh_uploads = std::getenv("WESCENE_TRACE_MESH_UPLOADS") != nullptr;
        result.trace_model_data = std::getenv("WESCENE_TRACE_MODEL_DATA") != nullptr;
        result.trace_model_input = std::getenv("WESCENE_TRACE_MODEL_INPUT") != nullptr;
        result.trace_model_projection = std::getenv("WESCENE_TRACE_MODEL_PROJECTION") != nullptr;
        result.trace_object_draw = std::getenv("WESCENE_TRACE_OBJECT_DRAW") != nullptr;
        result.trace_prelighting = std::getenv("WESCENE_TRACE_PRELIGHTING") != nullptr;
        result.trace_puppet_vertex_input = std::getenv("WESCENE_TRACE_PUPPET_VERTEX_INPUT") != nullptr;
        result.trace_reflection = std::getenv("WESCENE_TRACE_REFLECTION") != nullptr;
        result.trace_render_commands = std::getenv("WESCENE_TRACE_RENDER_COMMANDS") != nullptr;
        result.trace_render_uniforms = std::getenv("WESCENE_TRACE_RENDER_UNIFORMS") != nullptr;
        result.trace_scene_clear = std::getenv("WESCENE_TRACE_SCENE_CLEAR") != nullptr;
        result.trace_scene_projection = std::getenv("WESCENE_TRACE_SCENE_PROJECTION") != nullptr;
        result.trace_shape_state = std::getenv("WESCENE_TRACE_SHAPE_STATE") != nullptr;
        result.trace_texture_uploads = std::getenv("WESCENE_TRACE_TEXTURE_UPLOADS") != nullptr;
        result.trace_text_background = std::getenv("WESCENE_TRACE_TEXT_BACKGROUND") != nullptr;
        result.trace_text_color = std::getenv("WESCENE_TRACE_TEXT_COLOR") != nullptr;
        result.trace_text_depth = std::getenv("WESCENE_TRACE_TEXT_DEPTH") != nullptr;
        result.trace_text_destination = std::getenv("WESCENE_TRACE_TEXT_DESTINATION") != nullptr;
        result.trace_text_glyph = std::getenv("WESCENE_TRACE_TEXT_GLYPH") != nullptr;
        result.trace_text_source = std::getenv("WESCENE_TRACE_TEXT_SOURCE") != nullptr;
        result.trace_user_bindings = std::getenv("WESCENE_TRACE_USER_BINDINGS") != nullptr;
        result.trace_volumetrics = std::getenv("WESCENE_TRACE_VOLUMETRICS") != nullptr;
        result.transform_layer = std::getenv("WESCENE_TRACE_TRANSFORM_LAYER");
        result.render_target = std::getenv("WESCENE_TRACE_RENDER_TARGET");
        result.material_uniform = std::getenv("WESCENE_TRACE_MATERIAL_UNIFORM");
        result.present_after = std::getenv("WESCENE_TRACE_PRESENT_AFTER");
        result.frame_trace_directory = std::getenv("WESCENE_DUMP_FRAME_TRACE");
        result.frame_trace_draws = std::getenv("WESCENE_DUMP_FRAME_TRACE_DRAWS");
        result.gpu_profile = std::getenv("VIVID_GPU_PROFILE") != nullptr;
        if (const char* value = std::getenv("WESCENE_TRACE_SOUND_MIX"))
            result.trace_sound_mix = *value != '\0' && std::string_view(value) != "0";
        if (const char* value = std::getenv("WESCENE_VK_VALIDATION"))
            result.validation = std::string_view(value) == "1";
        if (const char* value = std::getenv("WESCENE_LOCKSTEP"))
            result.lockstep = std::string_view(value) == "1";
        result.fixed_dt = ReadDouble("WESCENE_FIXED_DT");
        result.fixed_epoch = ReadDouble("WESCENE_FIXED_EPOCH");
        result.random_seed = ReadSeed();
        return result;
    }();
    return settings;
}

} // namespace wallpaper::diagnostics

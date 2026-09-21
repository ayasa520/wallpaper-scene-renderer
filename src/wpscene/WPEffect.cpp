#include "wpscene/WPEffect.h"
#include "wpscene/WPEffectInput.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#include "Fs/VFS.h"
#include "Scene/SceneRenderTarget.h"
#include "Utils/Logging.h"
#include "WPJson.hpp"

using namespace wallpaper::wpscene;

namespace
{

int32_t ReadFboInt32(const nlohmann::json& json, const char* name, int32_t default_value) {
    const auto value = json.find(name);
    if (value == json.end() || !value->is_number()) return default_value;
    constexpr auto minimum = std::numeric_limits<int32_t>::min();
    constexpr auto maximum = std::numeric_limits<int32_t>::max();
    // Numeric admission precedes narrowing to the authored byte/word fields. Integral JSON
    // doubles are valid integers here, while booleans, strings, fractional numbers and values
    // outside signed 32-bit range do not replace the property's default.
    if (value->is_number_unsigned()) {
        const auto integer = value->get<uint64_t>();
        return integer <= maximum ? static_cast<int32_t>(integer) : default_value;
    }
    if (value->is_number_integer()) {
        const auto integer = value->get<int64_t>();
        return integer >= minimum && integer <= maximum
            ? static_cast<int32_t>(integer) : default_value;
    }
    const auto number = value->get<double>();
    return number >= minimum && number <= maximum && std::trunc(number) == number
        ? static_cast<int32_t>(number) : default_value;
}

std::string NormalizeMaterialResourceName(std::string name) {
    std::replace(name.begin(), name.end(), '\\', '/');
    // Resource identity preserves the leading slash pair. Interior separator runs collapse
    // without resolving dot segments or changing letter case, so lookup uses the same name
    // that selected the material resource rather than a filesystem canonical path.
    for (auto slash = name.find('/'); slash != std::string::npos;
         slash = name.find('/', slash + 1)) {
        if (slash == 0) continue;
        const auto end = name.find_first_not_of('/', slash + 1);
        const auto count = (end == std::string::npos ? name.size() : end) - slash - 1;
        name.erase(slash + 1, count);
    }
    return name;
}

void ReadVisibleBinding(const nlohmann::json& json, wallpaper::VisibleBinding* binding) {
    if (! json.is_object()) return;

    GET_JSON_NAME_VALUE_NOWARN(json, "value", binding->value);
    if (! json.contains("user") || json.at("user").is_null()) return;

    const auto& user = json.at("user");
    if (user.is_string()) {
        GET_JSON_VALUE(user, binding->user.name);
        return;
    }
    if (! user.is_object()) return;

    GET_JSON_NAME_VALUE_NOWARN(user, "name", binding->user.name);
    GET_JSON_NAME_VALUE_NOWARN(user, "condition", binding->user.condition);
}

} // namespace

bool WPEffectCommand::FromJson(const nlohmann::json& json) {
    GET_JSON_NAME_VALUE(json, "command", command);
    GET_JSON_NAME_VALUE_NOWARN(json, "target", target);
    GET_JSON_NAME_VALUE_NOWARN(json, "source", source);
    GET_JSON_NAME_VALUE_NOWARN(json, "compose", compose);
    return true;
}

bool WPEffectFbo::FromJson(const nlohmann::json& json) {
    const auto authored_name = json.find("name");
    const auto authored_format = json.find("format");
    // These members identify a framebuffer declaration, rather than a dynamic property.
    // Reject the entire record before it can allocate a target or enter a named-clear prefix.
    if (authored_name == json.end() || !authored_name->is_string() ||
        authored_format == json.end() || !authored_format->is_string()) return false;
    name = authored_name->get<std::string>();
    format = authored_format->get<std::string>();

    // The stored byte is the divisor used by target allocation, including integer wraparound
    // and zero. Parsing must not replace an admitted value with a different resource scale.
    scale = static_cast<uint8_t>(ReadFboInt32(json, "scale", 1));
    width = static_cast<uint16_t>(ReadFboInt32(json, "width", -1));
    height = static_cast<uint16_t>(ReadFboInt32(json, "height", -1));
    fit = static_cast<uint16_t>(ReadFboInt32(json, "fit", -1));
    GET_JSON_NAME_VALUE_NOWARN(json, "unique", unique);
    const auto clear = json.find("clear");
    if (clear != json.end() && clear->is_string()) {
        const char* token = clear->get_ref<const std::string&>().c_str();
        clear_on_setup = *token == '\0';
        // Numeric conversion and component traversal have different boundaries: a number may
        // end before the token does, while only literal spaces advance to the next component.
        // Retain partial prefixes with zero remaining components. Empty strings and reaching
        // the fourth component request setup clearing; shorter prefixes only supply a color.
        if (*token != '\0') {
            for (std::size_t index = 0; index < clear_color.size(); ++index) {
                clear_color[index] = static_cast<float>(std::strtod(token, nullptr));
                if (index + 1 == clear_color.size()) {
                    clear_on_setup = true;
                    break;
                }
                token = std::strchr(token, ' ');
                if (token == nullptr) break;
                while (*token == ' ') ++token;
            }
        }
    }
    return true;
}

std::array<int32_t, 2> WPEffectFbo::ResolveReferenceExtent(std::array<float, 2> source_size) const {
    return wallpaper::ResolveEffectRenderTargetReferenceExtent(source_size, { width, height }, fit);
}

// The blacklist belongs to the shared effect parser, not to WPImageObject. Text effects and image
// effects must make the same safety decision from the same data so both paths stay behaviorally
// identical after the text primitive split.
const std::unordered_set<std::string> WPImageEffect::BLACKLISTED_WORKSHOP_EFFECTS =
{
    // Keep this quarantine list evidence-driven: a blocked effect disappears before the scene
    // parser can build render passes or visibility bindings for it. Audio Responsive
    // Oscilloscope (2799421411) must remain enabled because it is the authored source of the
    // user-controlled music halo in scene 3585875739, and hiding it here makes the `osc` toggle
    // impossible to observe even when audio samples are flowing correctly.
};

bool WPImageEffect::IsEffectBlacklisted(const std::string& filePath) {
    std::filesystem::path path(filePath);
    // Workshop effect paths encode the item id in the grandparent directory. Keep the path walk
    // local to the effect parser so object types only receive a parsed visibility result.
    if (path.has_parent_path()) {
        path = path.parent_path();
        if(path.has_parent_path()) {
            std::string effectId = path.parent_path().filename().string();
            std::string parentPath = path.parent_path().string();
            return WPImageEffect::BLACKLISTED_WORKSHOP_EFFECTS.find(effectId) != WPImageEffect::BLACKLISTED_WORKSHOP_EFFECTS.end();
        }
    }
    return false;
}

std::unordered_set<std::string> WPImageEffect::FeedbackFboNames() const {
    // A normal transient effect FBO only needs to survive until its final reader in the same frame.
    // Feedback simulations are different: they sample one of their own FBOs before any pass has
    // written that FBO in the current frame, so the sampler intentionally reads last frame's state.
    // Cursor ripple is the canonical case: pass 0 reads `_rt_EightBuffer2`, pass 1 writes it back
    // after diffusion, and the next frame starts from that stored wave field. Detecting the
    // read-before-write contract from the authored pass order keeps this generic and avoids
    // hard-coding cursor-ripple resource names in the renderer.
    std::unordered_set<std::string> fbo_names;
    for (const auto& fbo : fbos) {
        if (! fbo.name.empty()) fbo_names.insert(fbo.name);
    }

    std::unordered_set<std::string> written_this_frame;
    std::unordered_set<std::string> feedback_fbos;

    auto read_fbo = [&](const std::string& name) {
        if (fbo_names.count(name) == 0) return;
        if (written_this_frame.count(name) == 0) feedback_fbos.insert(name);
    };

    auto write_fbo = [&](const std::string& name) {
        if (fbo_names.count(name) != 0) written_this_frame.insert(name);
    };

    auto apply_commands_at = [&](int32_t afterpos) {
        for (const auto& command : commands) {
            if (command.afterpos != afterpos) continue;
            if (command.command == "swap") {
                // Swap exchanges future references without copying either image. Both images can
                // become the next frame's feedback input, so their contents must survive
                // temporary-target lifetime release.
                if (fbo_names.contains(command.source) && fbo_names.contains(command.target)) {
                    feedback_fbos.insert(command.source);
                    feedback_fbos.insert(command.target);
                }
            } else if (command.command == "copy") {
                read_fbo(command.source);
                write_fbo(command.target);
            }
        }
    };

    for (std::size_t pass_index = 0; pass_index < passes.size(); pass_index++) {
        apply_commands_at(static_cast<int32_t>(pass_index));
        const auto& pass = passes[pass_index];
        for (const auto& binding : pass.bind) {
            read_fbo(binding.name);
        }
        write_fbo(pass.target);
    }
    apply_commands_at(static_cast<int32_t>(passes.size()));

    return feedback_fbos;
}

bool WPImageEffect::FromJson(const nlohmann::json& json, fs::VFS& vfs) {
    std::string filePath;
    GET_JSON_NAME_VALUE(json, "file", filePath);
    if(this->IsEffectBlacklisted(filePath)) {
        // A blacklist entry rejects the effect rather than changing its visibility. Log the exact
        // resource path so the missing effect has an explicit parse-time cause.
        LOG_INFO("SceneEffectBlacklist: file='%s' rejected=true", filePath.c_str());
        return false;
    }
	GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
    nlohmann::json jEffect;
    if(!PARSE_JSON(fs::GetFileContent(vfs, "/assets/" + filePath), jEffect))
        return false;
    if(!FromFileJson(jEffect, vfs, json))
        return false;
    // Parse-time mechanism marker for log-driven tooling: which effect resources a scene uses.
    LOG_INFO("SceneEffectParsed: id=%d file='%s' passes=%zu", id, filePath.c_str(), passes.size());

    // Parse the shared effect resource and pass overrides first, then apply the scene instance
    // property table. Visibility records execution state but never prevents the resource, its
    // pass materials, or its framebuffer declarations from being parsed.
    // The selection name also belongs to this instance. A missing name stays empty even when
    // the shared resource has a display title, and authored value wrappers use the same string
    // property reader as other scene properties.
    GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
    if (json.contains("visible")) {
        visible_json = json.at("visible");
        if (visible_json.is_object()) {
            ReadVisibleBinding(visible_json, &visible_binding);
            visible = visible_binding.value;
        } else {
            GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
            visible_binding.value = visible;
        }
    }
    return true;
}

bool WPImageEffect::FromFileJson(const nlohmann::json& json, fs::VFS& vfs,
                                const nlohmann::json& instance) {
	GET_JSON_NAME_VALUE_NOWARN(json, "version", version);
    const nlohmann::json no_input;
    const auto combo_input = instance.find("combos");
    const auto& combos = combo_input == instance.end() ? no_input : *combo_input;
    const auto pass_input = instance.find("passes");
    const auto& pass_overrides = pass_input == instance.end() ? no_input : *pass_input;
    if(json.contains("fbos")) {
        for(auto& jF:json.at("fbos")) {
            if (!MatchesEffectConditions(jF, combos)) continue;
            WPEffectFbo fbo;
            if (fbo.FromJson(jF)) fbos.push_back(std::move(fbo));
        }
    }
    if(json.contains("passes")) {
        const auto& jEPasses = json.at("passes");
        const nlohmann::json no_override;
        std::size_t input_index = 0;
        for(const auto& jP:jEPasses) {
            // Overrides retain their authored resource positions, including rejected records
            // and commands. Public material records and command positions instead compact to
            // the admitted sequence. Resolve the original index before the condition gate,
            // which must run before any material loading or command interpretation.
            const auto& material_input = pass_overrides.is_array() && input_index < pass_overrides.size()
                ? pass_overrides[input_index] : no_override;
            ++input_index;
            if (!MatchesEffectConditions(jP, combos)) continue;
            if(!jP.contains("material")) {
                if(jP.contains("command")) {
                    WPEffectCommand cmd;
                    cmd.FromJson(jP);
                    cmd.afterpos = passes.size();
                    commands.push_back(cmd);
                    material_records.emplace_back(std::nullopt);
                    continue;
                }
                LOG_ERROR("no material in effect pass");
                return false;
            }
            std::string matPath;
            GET_JSON_NAME_VALUE(jP, "material", matPath);
            matPath = NormalizeMaterialResourceName(std::move(matPath));
            nlohmann::json jMat;
            if(!PARSE_JSON(fs::GetFileContent(vfs, "/assets/" + matPath), jMat))
                return false;
            WPMaterial material;
            material.FromJson(jMat, material_input);
            material_records.emplace_back(matPath);
            materials.push_back(std::move(material));
            WPMaterialPass pass;
            pass.FromJson(jP, combos);
            passes.push_back(std::move(pass));
        }
    } else {
        LOG_ERROR("no passes in effect file");
        return false;
    }

    const auto functions = json.find("functions");
    if (functions != json.end() && functions->is_object()) {
        for (const auto& [name, function] : functions->items()) {
            if (name.empty() || !function.is_object()) continue;
            const auto action = function.find("action");
            const auto targets = function.find("fbos");
            if (action == function.end() || !action->is_string() || *action != "clear" ||
                targets == function.end() || !targets->is_array()) continue;
            std::size_t count = 0;
            for (const auto& target : *targets) {
                if (!target.is_string()) continue;
                const auto& target_name = target.get_ref<const std::string&>();
                if (target_name.empty()) continue;
                if (std::any_of(fbos.begin(), fbos.end(), [&](const auto& fbo) {
                        return fbo.name == target_name;
                    })) ++count;
            }
            // Functions select a prefix by the number of admitted references, including
            // duplicates. The reference values do not reorder that prefix. Admit only a
            // nonempty prefix contained in this effect's target records.
            if (count != 0 && count <= fbos.size()) clear_functions.emplace(name, count);
        }
    }
    return true;
}

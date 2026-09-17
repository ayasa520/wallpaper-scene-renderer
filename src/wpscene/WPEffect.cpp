#include "wpscene/WPEffect.h"

#include <cmath>
#include <filesystem>

#include "Fs/VFS.h"
#include "Scene/SceneRenderTarget.h"
#include "Utils/Logging.h"
#include "WPJson.hpp"

using namespace wallpaper::wpscene;

namespace
{

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
    GET_JSON_NAME_VALUE(json, "name", name);
    GET_JSON_NAME_VALUE(json, "format", format);

    GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
    GET_JSON_NAME_VALUE_NOWARN(json, "fit", fit);
    GET_JSON_NAME_VALUE_NOWARN(json, "unique", unique);
    if(scale == 0) { 
        LOG_ERROR("fbo scale can't be 0");
        scale = 1;
    }
    return true;
}

std::array<int32_t, 2> WPEffectFbo::ResolveSize(std::array<float, 2> source_size) const {
    return wallpaper::ResolveEffectRenderTargetExtent(source_size, scale, fit);
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
    if(!FromFileJson(jEffect, vfs, json.value("passes", nlohmann::json())))
        return false;
    // Parse-time mechanism marker for log-driven tooling: which effect resources a scene uses.
    LOG_INFO("SceneEffectParsed: id=%d file='%s' passes=%zu", id, filePath.c_str(), passes.size());

    // Parse the shared effect resource and pass overrides first, then apply the scene instance
    // property table. Visibility records execution state but never prevents the resource, its
    // pass materials, or its framebuffer declarations from being parsed.
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
                                const nlohmann::json& pass_overrides) {
	GET_JSON_NAME_VALUE_NOWARN(json, "version", version);
    GET_JSON_NAME_VALUE(json, "name", name);
    if(json.contains("fbos")) {
        for(auto& jF:json.at("fbos")) {
            WPEffectFbo fbo;
            fbo.FromJson(jF);
            fbos.push_back(std::move(fbo));
        }
    }
    if(json.contains("passes")) {
        const auto& jEPasses = json.at("passes");
        const nlohmann::json no_override;
        std::size_t input_index = 0;
        for(const auto& jP:jEPasses) {
            // Instance entries address the authored effect-pass array, including positions
            // occupied by commands. Resolve this index before filtering commands into their
            // execution list. The resource entry owns routing; only the matching instance
            // object is merged with the material resource before typed interpretation.
            const auto& instance = pass_overrides.is_array() && input_index < pass_overrides.size()
                ? pass_overrides[input_index] : no_override;
            ++input_index;
            if(!jP.contains("material")) {
                if(jP.contains("command")) {
                    WPEffectCommand cmd;
                    cmd.FromJson(jP);
                    cmd.afterpos = passes.size();
                    commands.push_back(cmd);
                    continue;
                }
                LOG_ERROR("no material in effect pass");
                return false;
            }
            std::string matPath;
            GET_JSON_NAME_VALUE(jP, "material", matPath);
            nlohmann::json jMat;
            if(!PARSE_JSON(fs::GetFileContent(vfs, "/assets/" + matPath), jMat))
                return false;
            WPMaterial material;
            material.FromJson(jMat, instance);
            materials.push_back(std::move(material));
            WPMaterialPass pass;
            pass.FromJson(jP);
            passes.push_back(std::move(pass));
        }
    } else {
        LOG_ERROR("no passes in effect file");
        return false;
    }
    return true;
}

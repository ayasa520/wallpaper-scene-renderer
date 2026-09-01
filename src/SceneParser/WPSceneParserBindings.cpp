#include "WPSceneParser.hpp"
#include "WPSceneParserShared.hpp"

// Script, user-property, property-animation, and effect-visibility binding registration for
// parsed scenes. Object-level registration runs after materialization so effect targets already
// exist and visibility updates never need to rebuild their pass/FBO topology.

#include "Utils/Logging.h"
#include "Core/StringHelper.hpp"
#include "Scene/SceneImageEffectLayer.h"
#include "WPJson.hpp"
#include "WPDynamicValue.hpp"
#include "WPPropertyAnimation.hpp"
#include "WPUserSetting.hpp"
#include "WPSoundParser.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

using namespace wallpaper;

std::optional<WPDynamicValue> ParsePropertyBaseValue(const nlohmann::json& property_json,
                                                     WPDynamicValue::Type  hint) {
    if (! property_json.is_object() || ! property_json.contains("value")) return std::nullopt;
    return WPDynamicValue::FromJsonLiteral(property_json.at("value"), hint);
}

void RegisterScenePropertyAnimationBinding(ParseContext& context, const nlohmann::json& object_json,
                                           std::string_view     property_name,
                                           WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains(property_name)) {
        return;
    }

    const auto& property_json = object_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    // Property animation registrations still target SceneNode-backed values. Camera layers keep a
    // node too, but they must dispatch through the camera target so origin/zoom keyframes update
    // the active SceneCamera instead of only the invisible layer node.
    const bool camera_registration =
        IsCameraLayerObjectJson(object_json) && IsCameraLayerRuntimeProperty(property_name);
    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    WPPropertyAnimationDefinition animation_definition;
    if (! ParsePropertyAnimationDefinition(property_json, hint, animation_definition)) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint)) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->propertyAnimationRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind =
            camera_registration ? WPSceneScriptTargetKind::Camera : WPSceneScriptTargetKind::Layer,
        .target_index = 0,
        .value_type   = hint,
        .base_value   = ParsePropertyBaseValue(property_json, hint).value_or(setting.value),
        .animation =
            std::make_shared<WPPropertyAnimationDefinition>(std::move(animation_definition)),
        .setting = std::move(setting),
    });
    if (IsTextLayerObjectJson(object_json)) {
        const auto& registration = context.scene->propertyAnimationRegistrations.back();
        LogTextLayerRegistration("register-property-animation",
                                 registration.object_id,
                                 registration.object_name,
                                 registration.property_name,
                                 registration.value_type,
                                 registration.setting,
                                 registration.base_value);
    }
    if (camera_registration) {
        LOG_INFO("SceneCameraLayerRegister: layer=%d property='%.*s' kind=animation target=camera",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
}

void RegisterSceneScriptBinding(ParseContext& context, const nlohmann::json& object_json,
                                std::string_view property_name, WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains(property_name)) {
        return;
    }

    const auto& property_json = object_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("script")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasScript()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    // Script bindings still dispatch through a SceneNode-backed target. Camera layers use the
    // camera target so parser/runtime scripts see normal layer objects while writes are routed to
    // the active SceneCamera state.
    const bool camera_registration =
        IsCameraLayerObjectJson(object_json) && IsCameraLayerRuntimeProperty(property_name);
    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    context.scene->scriptRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind =
            camera_registration ? WPSceneScriptTargetKind::Camera : WPSceneScriptTargetKind::Layer,
        .target_index = 0,
        .value_type   = hint,
        .base_value   = setting.value,
        .setting      = std::move(setting),
    });
    if (IsTextLayerObjectJson(object_json)) {
        const auto& registration = context.scene->scriptRegistrations.back();
        LogTextLayerRegistration("register-script-binding",
                                 registration.object_id,
                                 registration.object_name,
                                 registration.property_name,
                                 registration.value_type,
                                 registration.setting,
                                 registration.base_value);
    }
    if (camera_registration) {
        LOG_INFO("SceneCameraLayerRegister: layer=%d property='%.*s' kind=script target=camera",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
}

void RegisterScenePropertyBinding(ParseContext& context, const nlohmann::json& object_json,
                                  std::string_view property_name, WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains(property_name)) {
        return;
    }

    const auto& property_json = object_json.at(property_name);
    if (! property_json.is_object()) return;
    if (property_json.contains("script") || property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    // Sound objects are mounted as SoundManager streams and intentionally have no SceneNode in
    // context.object_nodes. Treat volume as a first-class runtime binding anyway so live user
    // property edits follow the same value path as the parse-time WPSoundParser::MountStream call.
    const bool sound_volume_binding =
        property_name == "volume" && context.scene != nullptr &&
        context.scene->GetLayerSoundHandle(object_id).has_value();
    const bool camera_registration =
        IsCameraLayerObjectJson(object_json) && IsCameraLayerRuntimeProperty(property_name);
    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end() && ! sound_volume_binding) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = sound_volume_binding ? nullptr : object_node_it->second.get(),
        .target_kind   = sound_volume_binding
                             ? WPSceneScriptTargetKind::Sound
                             : (camera_registration ? WPSceneScriptTargetKind::Camera
                                                    : WPSceneScriptTargetKind::Layer),
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });
    if (sound_volume_binding) {
        LOG_INFO("SceneSoundRegister: layer=%d property='%.*s' kind=user target=sound",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
    if (IsTextLayerObjectJson(object_json)) {
        const auto& registration = context.scene->bindingRegistrations.back();
        LogTextLayerRegistration("register-user-binding",
                                 registration.object_id,
                                 registration.object_name,
                                 registration.property_name,
                                 registration.value_type,
                                 registration.setting,
                                 registration.base_value);
    }
    if (camera_registration) {
        LOG_INFO("SceneCameraLayerRegister: layer=%d property='%.*s' kind=user target=camera",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
}

void RegisterSceneParticleOverridePropertyBinding(ParseContext&         context,
                                                  const nlohmann::json& object_json,
                                                  std::string_view      property_name,
                                                  WPDynamicValue::Type  hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains("particle") ||
        object_json.at("particle").is_null() || ! object_json.contains("instanceoverride") ||
        ! object_json.at("instanceoverride").is_object()) {
        return;
    }

    const auto& override_json = object_json.at("instanceoverride");
    if (! override_json.contains(property_name)) return;

    const auto& property_json = override_json.at(property_name);
    if (! property_json.is_object()) return;
    if (property_json.contains("script") || property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    // Particle authoring stores trail/sprite overrides under `instanceoverride`, not beside
    // ordinary layer material or image-size properties. Registering the nested value as a layer
    // target lets the runtime dispatch reuse the existing user-property pipeline while
    // ApplyLayerPropertyValue() routes `colorn` and scalar `size` to ParticleSubSystem.
    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind   = WPSceneScriptTargetKind::Layer,
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = ParsePropertyBaseValue(property_json, hint).value_or(setting.value),
        .setting       = std::move(setting),
    });

}

void RegisterSceneParticleOverrideScriptBinding(ParseContext&         context,
                                                const nlohmann::json& object_json,
                                                std::string_view      property_name,
                                                WPDynamicValue::Type  hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains("particle") ||
        object_json.at("particle").is_null() || ! object_json.contains("instanceoverride") ||
        ! object_json.at("instanceoverride").is_object()) {
        return;
    }

    const auto& override_json = object_json.at("instanceoverride");
    if (! override_json.contains(property_name)) return;

    const auto& property_json = override_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("script")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasScript()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    // Particle override scripts are authored under instanceoverride, but the script runtime only
    // knows how to dispatch persistent scripts to concrete scene targets. Register the nested
    // value as a layer target with the bare override name so ApplyParticlePropertyValue() can
    // forward it to ParticleSubSystem instead of losing audio-reactive particle clocks after parse.
    context.scene->scriptRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind   = WPSceneScriptTargetKind::Layer,
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });

}

void RegisterEffectVisibilityBinding(ParseContext& context, const nlohmann::json& object_json,
                                     const nlohmann::json& effect_json,
                                     uint32_t              authored_effect_index) {
    if (! object_json.is_object() || ! effect_json.is_object() ||
        ! effect_json.contains("visible") || ! effect_json.at("visible").is_object()) {
        return;
    }

    int32_t object_id { 0 };
    GET_JSON_NAME_VALUE_NOWARN(object_json, "id", object_id);
    if (object_id == 0 || context.object_nodes.count(object_id) == 0 || context.scene == nullptr) {
        return;
    }

    int32_t     effect_id { 0 };
    std::string authored_effect_name;
    GET_JSON_NAME_VALUE_NOWARN(effect_json, "id", effect_id);
    GET_JSON_NAME_VALUE_NOWARN(effect_json, "name", authored_effect_name);

    SceneImageEffect* effect =
        effect_id != 0 ? context.scene->FindImageEffectById(object_id, effect_id)
                       : context.scene->FindImageEffect(object_id, authored_effect_index);
    if (effect == nullptr) return;

    const auto&       visible_json = effect_json.at("visible");
    const std::string effect_name =
        ! authored_effect_name.empty() ? authored_effect_name : effect->EffectName();
    WPUserSetting setting;
    if (! ParseUserSetting(visible_json, setting, WPDynamicValue::Type::Boolean)) return;

    // The target points at the fully materialized SceneImageEffect. Runtime dispatch changes only
    // its local execution bit; all pass nodes and named render targets remain attached.
    WPSceneScriptRegistration registration {
        .object_id     = object_id,
        .object_name   = effect_name,
        .property_name = "visible",
        .node          = context.object_nodes.at(object_id).get(),
        .target_kind   = WPSceneScriptTargetKind::Effect,
        .target_index  = effect->EffectIndex(),
        .target_id     = effect_id,
        .value_type    = WPDynamicValue::Type::Boolean,
        .base_value    = ParsePropertyBaseValue(visible_json, WPDynamicValue::Type::Boolean)
                             .value_or(setting.value),
        .setting       = std::move(setting),
    };

    const char* registration_kind = nullptr;
    if (visible_json.contains("animation") && ! visible_json.at("animation").is_null()) {
        WPPropertyAnimationDefinition animation_definition;
        if (! ParsePropertyAnimationDefinition(
                visible_json, WPDynamicValue::Type::Boolean, animation_definition)) {
            return;
        }
        registration.animation =
            std::make_shared<WPPropertyAnimationDefinition>(std::move(animation_definition));
        context.scene->propertyAnimationRegistrations.push_back(std::move(registration));
        registration_kind = "animation";
    } else if (registration.setting.hasScript()) {
        context.scene->scriptRegistrations.push_back(std::move(registration));
        registration_kind = "script";
    } else if (registration.setting.hasUserBinding()) {
        context.scene->bindingRegistrations.push_back(std::move(registration));
        registration_kind = "user";
    }

    if (registration_kind != nullptr) {
        LOG_INFO("SceneVisibilityEffectRegister: layer=%d effect-id=%d effect-index=%u name='%s' "
                 "kind=%s initial-visible=%s",
                 object_id,
                 effect_id,
                 effect->EffectIndex(),
                 effect_name.c_str(),
                 registration_kind,
                 effect->LocalVisible() ? "true" : "false");
    }
}

void RegisterEffectVisibilityBindings(ParseContext& context, const nlohmann::json& object_json) {
    if (! object_json.is_object() || ! object_json.contains("effects") ||
        ! object_json.at("effects").is_array()) {
        return;
    }

    uint32_t effect_index = 0;
    for (const auto& effect_json : object_json.at("effects")) {
        RegisterEffectVisibilityBinding(context, object_json, effect_json, effect_index);
        effect_index++;
    }
}

void RegisterAnimationLayerSceneScriptBinding(ParseContext&         context,
                                              const nlohmann::json& object_json,
                                              const nlohmann::json& layer_json,
                                              uint32_t layer_index, std::string_view property_name,
                                              WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! layer_json.is_object() ||
        ! layer_json.contains(property_name)) {
        return;
    }

    const auto& property_json = layer_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("script")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    if (context.object_nodes.count(object_id) == 0) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasScript()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->scriptRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = context.object_nodes.at(object_id).get(),
        .target_kind   = WPSceneScriptTargetKind::AnimationLayer,
        .target_index  = layer_index,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });
}

void RegisterAnimationLayerPropertyBinding(ParseContext& context, const nlohmann::json& object_json,
                                           const nlohmann::json& layer_json, uint32_t layer_index,
                                           std::string_view     property_name,
                                           WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! layer_json.is_object() ||
        ! layer_json.contains(property_name)) {
        return;
    }

    const auto& property_json = layer_json.at(property_name);
    if (! property_json.is_object()) return;
    // A script on an animation-layer property can initialize timing or shared state while the same
    // property still declares a user binding for live visibility/blend control. Only authored
    // property animations own the value path exclusively; scripts and direct user bindings must
    // coexist so runtime project-property edits are applied after script initialization.
    if (property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    if (context.object_nodes.count(object_id) == 0) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = context.object_nodes.at(object_id).get(),
        .target_kind   = WPSceneScriptTargetKind::AnimationLayer,
        .target_index  = layer_index,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });

    LOG_INFO("SceneAnimationLayerRegister: layer=%d animation-index=%u property='%.*s' kind=user "
             "target=animationLayer script-present=%s",
             object_id,
             layer_index,
             static_cast<int>(property_name.size()),
             property_name.data(),
             property_json.contains("script") ? "true" : "false");
}

void RegisterSceneGeneralPropertyBinding(ParseContext& context, const nlohmann::json& general_json,
                                         std::string_view     property_name,
                                         WPDynamicValue::Type hint) {
    if (context.scene == nullptr || ! general_json.is_object() ||
        ! general_json.contains(property_name)) {
        return;
    }

    const auto& property_json = general_json.at(property_name);
    if (! property_json.is_object()) return;
    // General properties do not have SceneNode owners. Keep authored animations out of the direct
    // binding table, but allow user-bound values to drive global runtime state such as camera
    // parallax through the same dispatch path as layer and effect properties.
    if (property_json.contains("animation")) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = 0,
        .object_name   = "scene.general",
        .property_name = std::string(property_name),
        .node          = nullptr,
        .target_kind   = WPSceneScriptTargetKind::Scene,
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = ParsePropertyBaseValue(property_json, hint).value_or(setting.value),
        .setting       = std::move(setting),
    });

    LOG_INFO("SceneGeneralRegister: property='%.*s' kind=user target=scene.general",
             static_cast<int>(property_name.size()),
             property_name.data());
}

void RegisterSceneScripts(ParseContext& context, const nlohmann::json& json) {
    if (json.contains("general") && json.at("general").is_object()) {
        const auto& general_json = json.at("general");
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "clearcolor", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "ambientcolor", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "skylightcolor", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloom", WPDynamicValue::Type::Boolean);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloomstrength", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloomthreshold", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloomtint", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallax", WPDynamicValue::Type::Boolean);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallaxamount", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallaxdelay", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallaxmouseinfluence", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashake", WPDynamicValue::Type::Boolean);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashakeamplitude", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashakeroughness", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashakespeed", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "fov", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "nearz", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "farz", WPDynamicValue::Type::Float);
    }

    if (! json.contains("objects") || ! json.at("objects").is_array()) return;

    for (const auto& object_json : json.at("objects")) {
        RegisterScenePropertyBinding(
            context, object_json, "visible", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(
            context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
        if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
            (object_json.contains("text") && ! object_json.at("text").is_null())) {
            RegisterScenePropertyBinding(
                context, object_json, "size", WPDynamicValue::Type::Float2);
        }
        RegisterScenePropertyBinding(context, object_json, "text", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(context, object_json, "font", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
        // Particle controls are nested below instanceoverride, so the generic top-level scans
        // never see them. Register the nested override names here so user-property edits and
        // persistent scripts can reach the live particle subsystem.
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "colorn", WPDynamicValue::Type::Float3);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "color", WPDynamicValue::Type::Float3);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "size", WPDynamicValue::Type::Float);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "rate", WPDynamicValue::Type::Float);
        RegisterSceneParticleOverrideScriptBinding(
            context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterSceneParticleOverrideScriptBinding(
            context, object_json, "rate", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(
            context, object_json, "brightness", WPDynamicValue::Type::Float);
        // Sound-layer volume is authored beside visual layer properties, but its runtime target is
        // the mounted audio stream instead of a SceneNode material.
        RegisterScenePropertyBinding(context, object_json, "volume", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(
            context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(
            context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(
            context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(
            context, object_json, "pointsize", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
        RegisterScenePropertyBinding(
            context, object_json, "horizontalalign", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(
            context, object_json, "verticalalign", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(
            context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
        RegisterScenePropertyBinding(
            context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "visible", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "origin", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "angles", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "scale", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
        if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
            (object_json.contains("text") && ! object_json.at("text").is_null())) {
            RegisterScenePropertyAnimationBinding(
                context, object_json, "size", WPDynamicValue::Type::Float2);
        }
        RegisterScenePropertyAnimationBinding(
            context, object_json, "color", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "brightness", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "pointsize", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "padding", WPDynamicValue::Type::Int32);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "maxrows", WPDynamicValue::Type::Int32);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "maxwidth", WPDynamicValue::Type::Float);

        RegisterSceneScriptBinding(context, object_json, "visible", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(
            context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
        if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
            (object_json.contains("text") && ! object_json.at("text").is_null())) {
            RegisterSceneScriptBinding(context, object_json, "size", WPDynamicValue::Type::Float2);
        }
        RegisterSceneScriptBinding(context, object_json, "text", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(context, object_json, "font", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "brightness", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(
            context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(
            context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(
            context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "pointsize", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
        RegisterSceneScriptBinding(
            context, object_json, "horizontalalign", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(
            context, object_json, "verticalalign", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(
            context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
        RegisterSceneScriptBinding(
            context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);

        if (object_json.contains("light") && ! object_json.at("light").is_null()) {
            // Official lighting docs animate Intensity (and optionally Radius) on the light
            // object itself. These are not material uniforms, so they are scanned only for
            // authored lights and applied to SceneLight each frame.
            RegisterScenePropertyBinding(
                context, object_json, "intensity", WPDynamicValue::Type::Float);
            RegisterScenePropertyBinding(
                context, object_json, "radius", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "intensity", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "radius", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(
                context, object_json, "intensity", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(context, object_json, "radius", WPDynamicValue::Type::Float);
        }

        if (IsCameraLayerObjectJson(object_json)) {
            // Camera zoom/fov are not normal drawable layer properties, so they are scanned only
            // for authored camera layers and routed through the camera target kind registered
            // above. Origin/visible are already part of the shared layer scan and are retargeted
            // by RegisterScene*Binding when the object is a camera layer.
            RegisterScenePropertyBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
            RegisterScenePropertyBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "zoom", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "fov", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
        }

        RegisterEffectVisibilityBindings(context, object_json);

        if (! object_json.contains("animationlayers") ||
            ! object_json.at("animationlayers").is_array()) {
            continue;
        }

        uint32_t layer_index = 0;
        for (const auto& animation_layer_json : object_json.at("animationlayers")) {
            RegisterAnimationLayerPropertyBinding(context,
                                                  object_json,
                                                  animation_layer_json,
                                                  layer_index,
                                                  "visible",
                                                  WPDynamicValue::Type::Boolean);
            RegisterAnimationLayerPropertyBinding(context,
                                                  object_json,
                                                  animation_layer_json,
                                                  layer_index,
                                                  "rate",
                                                  WPDynamicValue::Type::Float);
            RegisterAnimationLayerPropertyBinding(context,
                                                  object_json,
                                                  animation_layer_json,
                                                  layer_index,
                                                  "blend",
                                                  WPDynamicValue::Type::Float);
            RegisterAnimationLayerSceneScriptBinding(context,
                                                     object_json,
                                                     animation_layer_json,
                                                     layer_index,
                                                     "visible",
                                                     WPDynamicValue::Type::Boolean);
            RegisterAnimationLayerSceneScriptBinding(context,
                                                     object_json,
                                                     animation_layer_json,
                                                     layer_index,
                                                     "rate",
                                                     WPDynamicValue::Type::Float);
            RegisterAnimationLayerSceneScriptBinding(context,
                                                     object_json,
                                                     animation_layer_json,
                                                     layer_index,
                                                     "blend",
                                                     WPDynamicValue::Type::Float);
            layer_index++;
        }
    }
}

void RegisterSceneScriptsForObject(ParseContext& context, const nlohmann::json& object_json) {
    RegisterScenePropertyBinding(context, object_json, "visible", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(
        context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
    if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
        (object_json.contains("text") && ! object_json.at("text").is_null())) {
        RegisterScenePropertyBinding(context, object_json, "size", WPDynamicValue::Type::Float2);
    }
    RegisterScenePropertyBinding(context, object_json, "text", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "font", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
    // Dynamic layer materialization uses this per-object registration path, so particle
    // instanceoverride bindings must be added here as well as during the initial full-scene scan.
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "colorn", WPDynamicValue::Type::Float3);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "color", WPDynamicValue::Type::Float3);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "size", WPDynamicValue::Type::Float);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "rate", WPDynamicValue::Type::Float);
    RegisterSceneParticleOverrideScriptBinding(
        context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterSceneParticleOverrideScriptBinding(
        context, object_json, "rate", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(context, object_json, "brightness", WPDynamicValue::Type::Float);
    // Dynamic materialization reuses the same registration helper, so keep sound volume in this
    // per-object path as well as the initial full-scene scan.
    RegisterScenePropertyBinding(context, object_json, "volume", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(
        context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(
        context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(
        context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "pointsize", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
    RegisterScenePropertyBinding(
        context, object_json, "horizontalalign", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(
        context, object_json, "verticalalign", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
    RegisterScenePropertyBinding(context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "visible", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "origin", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "angles", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "scale", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
    if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
        (object_json.contains("text") && ! object_json.at("text").is_null())) {
        RegisterScenePropertyAnimationBinding(
            context, object_json, "size", WPDynamicValue::Type::Float2);
    }
    RegisterScenePropertyAnimationBinding(
        context, object_json, "color", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "brightness", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "pointsize", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "padding", WPDynamicValue::Type::Int32);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "maxrows", WPDynamicValue::Type::Int32);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "maxwidth", WPDynamicValue::Type::Float);

    RegisterSceneScriptBinding(context, object_json, "visible", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
    if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
        (object_json.contains("text") && ! object_json.at("text").is_null())) {
        RegisterSceneScriptBinding(context, object_json, "size", WPDynamicValue::Type::Float2);
    }
    RegisterSceneScriptBinding(context, object_json, "text", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "font", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(context, object_json, "brightness", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(
        context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(
        context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(
        context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "pointsize", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
    RegisterSceneScriptBinding(
        context, object_json, "horizontalalign", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "verticalalign", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
    RegisterSceneScriptBinding(context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);

    if (object_json.contains("light") && ! object_json.at("light").is_null()) {
        RegisterScenePropertyBinding(
            context, object_json, "intensity", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "radius", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "intensity", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "radius", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "intensity", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "radius", WPDynamicValue::Type::Float);
    }

    if (IsCameraLayerObjectJson(object_json)) {
        // Dynamic camera layers need the same camera-only zoom/fov registration as scene-load
        // camera layers; otherwise scripts that create or re-materialize camera assets would keep
        // origin live but leave zoom frozen at the authored parse value.
        RegisterScenePropertyBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "zoom", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "fov", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
    }

    RegisterEffectVisibilityBindings(context, object_json);

    if (! object_json.contains("animationlayers") ||
        ! object_json.at("animationlayers").is_array()) {
        return;
    }

    uint32_t layer_index = 0;
    for (const auto& animation_layer_json : object_json.at("animationlayers")) {
        RegisterAnimationLayerPropertyBinding(context,
                                              object_json,
                                              animation_layer_json,
                                              layer_index,
                                              "visible",
                                              WPDynamicValue::Type::Boolean);
        RegisterAnimationLayerPropertyBinding(context,
                                              object_json,
                                              animation_layer_json,
                                              layer_index,
                                              "rate",
                                              WPDynamicValue::Type::Float);
        RegisterAnimationLayerPropertyBinding(context,
                                              object_json,
                                              animation_layer_json,
                                              layer_index,
                                              "blend",
                                              WPDynamicValue::Type::Float);
        RegisterAnimationLayerSceneScriptBinding(context,
                                                 object_json,
                                                 animation_layer_json,
                                                 layer_index,
                                                 "visible",
                                                 WPDynamicValue::Type::Boolean);
        RegisterAnimationLayerSceneScriptBinding(context,
                                                 object_json,
                                                 animation_layer_json,
                                                 layer_index,
                                                 "rate",
                                                 WPDynamicValue::Type::Float);
        RegisterAnimationLayerSceneScriptBinding(context,
                                                 object_json,
                                                 animation_layer_json,
                                                 layer_index,
                                                 "blend",
                                                 WPDynamicValue::Type::Float);
        layer_index++;
    }
}

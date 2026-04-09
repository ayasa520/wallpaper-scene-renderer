#include "WPUserSetting.hpp"

#include <nlohmann/json.hpp>

#include "WPJson.hpp"

namespace wallpaper
{
namespace
{

std::optional<UserPropertyBinding> ParseUserBinding(const nlohmann::json& json) {
    if (! json.is_object() || ! json.contains("user") || json.at("user").is_null()) {
        return std::nullopt;
    }

    UserPropertyBinding binding;
    const auto& user = json.at("user");
    if (user.is_string()) {
        binding.name = user.get<std::string>();
    } else if (user.is_object()) {
        if (user.contains("name") && user.at("name").is_string()) {
            binding.name = user.at("name").get<std::string>();
        }
        if (user.contains("condition") && user.at("condition").is_string()) {
            binding.condition = user.at("condition").get<std::string>();
        }
    }

    if (binding.empty()) return std::nullopt;
    return binding;
}

const nlohmann::json* ResolveInitialValueNode(const nlohmann::json& json) {
    if (! json.is_object()) return &json;
    if (json.contains("value")) return &json.at("value");

    if (json.contains("animation") && json.at("animation").is_object()) {
        const auto& animation = json.at("animation");
        bool        start_paused { false };
        if (animation.contains("options") && animation.at("options").is_object() &&
            animation.at("options").contains("startpaused") &&
            animation.at("options").at("startpaused").is_boolean()) {
            start_paused = animation.at("options").at("startpaused").get<bool>();
        }

        if (start_paused && animation.contains("c0") && animation.at("c0").is_array() &&
            ! animation.at("c0").empty() && animation.at("c0").front().is_object() &&
            animation.at("c0").front().contains("value")) {
            return &animation.at("c0").front().at("value");
        }
    }

    return &json;
}

template<typename T>
bool ParseLegacyValue(const nlohmann::json& json, T& out_value) {
    return GetJsonValue(
        __SHORT_FILE__, __FUNCTION__, __LINE__, json, out_value, false, "", false);
}

bool ParseLegacyValue(const nlohmann::json& json, WPDynamicValue::Type hint, WPDynamicValue& out_value) {
    switch (hint) {
        case WPDynamicValue::Type::Boolean: {
            bool value = false;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Int32: {
            int32_t value = 0;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::UInt32: {
            uint32_t value = 0;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Float: {
            float value = 0.0f;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Double: {
            double value = 0.0;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::String: {
            std::string value;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(std::move(value));
            return true;
        }
        case WPDynamicValue::Type::FloatVector: {
            std::vector<float> value;
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(std::move(value));
            return true;
        }
        case WPDynamicValue::Type::Int3: {
            std::array<int32_t, 3> value {};
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Float2: {
            std::array<float, 2> value {};
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Float3: {
            std::array<float, 3> value {};
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Float4: {
            std::array<float, 4> value {};
            if (!ParseLegacyValue(json, value)) return false;
            out_value = WPDynamicValue(value);
            return true;
        }
        case WPDynamicValue::Type::Null:
            return false;
    }

    return false;
}

} // namespace

WPDynamicValue WPUserSetting::evaluate(const UserPropertyMap*           user_properties,
                                       WPScriptRuntime*                 runtime,
                                       const WPScriptEvaluationContext& base_context) const {
    (void)runtime;
    (void)base_context;

    WPDynamicValue resolved = value;

    if (property.has_value()) {
        if (const auto* user_value = LookupUserProperty(user_properties, property->name)) {
            if (property->condition.empty() ||
                MatchesUserPropertyCondition(*user_value, property->condition)) {
                if (const auto override_value =
                        WPDynamicValue::FromUserPropertyValue(*user_value, value.type());
                    override_value.has_value()) {
                    resolved = *override_value;
                }
            }
        }
    }

    return resolved;
}

bool ParseUserSetting(const nlohmann::json& json, WPUserSetting& setting, WPDynamicValue::Type hint) {
    setting = {};

    const auto* value_node = ResolveInitialValueNode(json);
    if (value_node == nullptr) return false;

    std::optional<WPDynamicValue> parsed_value;
    if (hint == WPDynamicValue::Type::Null) {
        parsed_value = WPDynamicValue::FromJsonLiteral(*value_node, hint);
    } else {
        WPDynamicValue legacy_value;
        if (ParseLegacyValue(json, hint, legacy_value)) {
            parsed_value = legacy_value;
        } else {
            parsed_value = WPDynamicValue::FromJsonLiteral(*value_node, hint);
        }
    }
    if (! parsed_value.has_value()) return false;

    setting.value = *parsed_value;

    if (json.is_object()) {
        setting.property = ParseUserBinding(json);

        if (json.contains("script") && json.at("script").is_string()) {
            setting.script = json.at("script").get<std::string>();
        }

        if (json.contains("scriptproperties") && json.at("scriptproperties").is_object()) {
            for (const auto& [name, property_json] : json.at("scriptproperties").items()) {
                auto nested = std::make_shared<WPUserSetting>();
                if (ParseUserSetting(property_json, *nested)) {
                    setting.script_properties.emplace(name, std::move(nested));
                }
            }
        }
    }

    return true;
}

bool ParseUserSetting(const nlohmann::json& json, WPUserSetting& setting) {
    return ParseUserSetting(json, setting, WPDynamicValue::Type::Null);
}

} // namespace wallpaper

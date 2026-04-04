#include "WPJson.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <optional>
#include <sstream>
#include <type_traits>

#include "Utils/Identity.hpp"
#include "Utils/String.h"

namespace wallpaper
{
namespace
{
thread_local const UserPropertyMap* g_json_user_properties = nullptr;

std::optional<UserPropertyBinding> ResolveUserPropertyBinding(const nlohmann::json& json) {
    if (! json.is_object() || ! json.contains("user") || json.at("user").is_null()) {
        return std::nullopt;
    }

    UserPropertyBinding binding;
    const auto&         user = json.at("user");
    if (user.is_string()) {
        GET_JSON_VALUE_NOWARN(user, binding.name);
    } else if (user.is_object()) {
        GET_JSON_NAME_VALUE_NOWARN(user, "name", binding.name);
        GET_JSON_NAME_VALUE_NOWARN(user, "condition", binding.condition);
    }

    if (binding.name.empty()) return std::nullopt;
    return binding;
}

const nlohmann::json* ResolveAnimatedInitialValue(const nlohmann::json& json) {
    if (! json.is_object() || ! json.contains("animation")) return nullptr;

    const auto& animation = json.at("animation");
    if (! animation.is_object()) return nullptr;

    bool start_paused { false };
    if (animation.contains("options") && animation.at("options").is_object()) {
        GET_JSON_NAME_VALUE_NOWARN(animation.at("options"), "startpaused", start_paused);
    }
    if (! start_paused || ! animation.contains("c0")) return nullptr;

    const auto& c0 = animation.at("c0");
    if (! c0.is_array() || c0.empty() || ! c0.front().is_object() ||
        ! c0.front().contains("value")) {
        return nullptr;
    }

    return &c0.front().at("value");
}

const nlohmann::json& ResolvePropertyValueNode(const nlohmann::json& json) {
    if (const auto* animated = ResolveAnimatedInitialValue(json)) return *animated;
    if (json.is_object() && json.contains("value")) return json.at("value");
    return json;
}

template<typename T>
bool TryParseNumber(std::string_view text, T& value);

template<>
bool TryParseNumber<float>(std::string_view text, float& value) {
    char* endptr = nullptr;
    value        = std::strtof(std::string(text).c_str(), &endptr);
    return endptr != nullptr && *endptr == '\0';
}

template<>
bool TryParseNumber<double>(std::string_view text, double& value) {
    char* endptr = nullptr;
    value        = std::strtod(std::string(text).c_str(), &endptr);
    return endptr != nullptr && *endptr == '\0';
}

template<>
bool TryParseNumber<int32_t>(std::string_view text, int32_t& value) {
    char* endptr = nullptr;
    value        = (int32_t)std::strtol(std::string(text).c_str(), &endptr, 10);
    return endptr != nullptr && *endptr == '\0';
}

template<>
bool TryParseNumber<uint32_t>(std::string_view text, uint32_t& value) {
    char* endptr = nullptr;
    value        = (uint32_t)std::strtoul(std::string(text).c_str(), &endptr, 10);
    return endptr != nullptr && *endptr == '\0';
}

std::string ShaderValueToString(const ShaderValue& value) {
    std::ostringstream out;
    for (size_t i = 0; i < value.size(); i++) {
        if (i != 0) out << ' ';
        out << value[i];
    }
    return out.str();
}

template<typename T>
bool TryConvertUserPropertyValue(const UserPropertyValue& property, T& value) {
    if constexpr (std::is_same_v<T, bool>) {
        value = IsUserPropertyTruthy(property);
        return true;
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (const auto* string_value = std::get_if<std::string>(&property)) {
            value = *string_value;
            return true;
        }
        if (const auto* shader_value = std::get_if<ShaderValue>(&property)) {
            value = ShaderValueToString(*shader_value);
            return true;
        }
        return false;
    } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                         std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
        if (const auto* shader_value = std::get_if<ShaderValue>(&property)) {
            if (shader_value->size() == 0) return false;
            value = (T)(*shader_value)[0];
            return true;
        }
        if (const auto* string_value = std::get_if<std::string>(&property)) {
            return TryParseNumber<T>(TrimString(*string_value), value);
        }
        return false;
    } else {
        return false;
    }
}

template<typename T>
bool TryConvertUserPropertyValue(const UserPropertyValue& property, std::vector<T>& value) {
    if (const auto* shader_value = std::get_if<ShaderValue>(&property)) {
        value.resize(shader_value->size());
        for (size_t i = 0; i < shader_value->size(); i++) {
            value[i] = (T)(*shader_value)[i];
        }
        return true;
    }
    if (const auto* string_value = std::get_if<std::string>(&property)) {
        return utils::StrToArray::Convert(*string_value, value);
    }
    return false;
}

template<typename T, std::size_t N>
bool TryConvertUserPropertyValue(const UserPropertyValue& property, std::array<T, N>& value) {
    if (const auto* shader_value = std::get_if<ShaderValue>(&property)) {
        if (shader_value->size() != N) return false;
        for (size_t i = 0; i < N; i++) {
            value[i] = (T)(*shader_value)[i];
        }
        return true;
    }
    if (const auto* string_value = std::get_if<std::string>(&property)) {
        return utils::StrToArray::Convert(*string_value, value);
    }
    return false;
}

template<typename T>
bool TryGetUserPropertyOverride(const nlohmann::json& json, T& value) {
    if (g_json_user_properties == nullptr) return false;

    const auto binding = ResolveUserPropertyBinding(json);
    if (! binding.has_value()) return false;

    const auto* property = LookupUserProperty(g_json_user_properties, binding->name);
    if (property == nullptr) return false;

    if (! binding->condition.empty() &&
        ! MatchesUserPropertyCondition(*property, binding->condition)) {
        return false;
    }

    return TryConvertUserPropertyValue(*property, value);
}
} // namespace

bool ParseJson(const char* file, const char* func, int line, const std::string& source,
               nlohmann::json& result) {
    try {
        result = nlohmann::json::parse(source);
    } catch (nlohmann::json::parse_error& e) {
        WallpaperLog(LOGLEVEL_ERROR, file, line, "parse json(%s), %s", func, e.what());
        return false;
    }
    return true;
}

template<typename T>
inline bool _GetJsonValue(const nlohmann::json&                  json,
                          typename utils::is_std_array<T>::type& value) {
    if (TryGetUserPropertyOverride(json, value)) return true;

    using Tv          = typename T::value_type;
    const auto& njson = ResolvePropertyValueNode(json);
    if (njson.is_number()) {
        value = { njson.get<Tv>() };
        return true;
    }

    std::string strvalue = njson.get<std::string>();
    return utils::StrToArray::Convert(strvalue, value);
}

template<typename T>
inline bool _GetJsonValue(const nlohmann::json& json, T& value) {
    if (TryGetUserPropertyOverride(json, value)) return true;

    value = ResolvePropertyValueNode(json).get<T>();
    return true;
}

template<typename T>
inline bool _GetJsonValue(const char* file, const char* func, int line, const nlohmann::json& json,
                          T& value, bool warn, const char* name) {
    (void)warn;

    using njson = nlohmann::json;
    std::string nameinfo;
    if (name != nullptr) nameinfo = std::string("(key: ") + name + ")";
    try {
        return _GetJsonValue<T>(json, value);
    } catch (const njson::type_error& e) {
        WallpaperLog(LOGLEVEL_INFO,
                     file,
                     line,
                     "%s %s at %s\n%s",
                     e.what(),
                     nameinfo.c_str(),
                     func,
                     json.dump(4).c_str());
    } catch (const std::invalid_argument& e) {
        WallpaperLog(LOGLEVEL_ERROR, file, line, "%s %s at %s", e.what(), nameinfo.c_str(), func);
    } catch (const std::out_of_range& e) {
        WallpaperLog(LOGLEVEL_ERROR, file, line, "%s %s at %s", e.what(), nameinfo.c_str(), func);
    } catch (const utils::StrToArray::WrongSizeExp& e) {
        WallpaperLog(LOGLEVEL_ERROR, file, line, "%s %s at %s", e.what(), nameinfo.c_str(), func);
    }
    return false;
}

template<typename T>
typename JsonTemplateTypeCheck<T>::type
GetJsonValue(const char* file, const char* func, int line, const nlohmann::json& json, T& value,
             bool has_name, std::string_view name_view, bool warn) {
    std::string name { name_view };
    if (has_name) {
        if (! json.contains(name)) {
            if (warn)
                WallpaperLog(LOGLEVEL_INFO,
                             "",
                             0,
                             "read json \"%s\" not a key at %s(%s:%d)",
                             name.data(),
                             func,
                             file,
                             line);
            return false;
        } else if (json.at(name).is_null()) {
            if (warn)
                WallpaperLog(LOGLEVEL_INFO,
                             "",
                             0,
                             "read json \"%s\" is null at %s(%s:%d)",
                             name.data(),
                             func,
                             file,
                             line);
            return false;
        }
    }
    return _GetJsonValue<T>(file,
                            func,
                            line,
                            has_name ? json.at(name) : json,
                            value,
                            warn,
                            name.empty() ? nullptr : name.c_str());
}

#define T_IMPL_GET_JSON(TYPE)                                                            \
    template JsonTemplateTypeCheck<TYPE>::type GetJsonValue<TYPE>(const char*,           \
                                                                  const char*,           \
                                                                  int,                   \
                                                                  const nlohmann::json&, \
                                                                  TYPE&,                 \
                                                                  bool,                  \
                                                                  std::string_view,      \
                                                                  bool);

T_IMPL_GET_JSON(bool);
T_IMPL_GET_JSON(int32_t);
T_IMPL_GET_JSON(uint32_t);
T_IMPL_GET_JSON(float);
T_IMPL_GET_JSON(double);
T_IMPL_GET_JSON(std::string);
T_IMPL_GET_JSON(std::vector<float>);

template<std::size_t N>
using iarray = std::array<int, N>;
T_IMPL_GET_JSON(iarray<3>);

template<std::size_t N>
using farray = std::array<float, N>;
T_IMPL_GET_JSON(farray<2>);
T_IMPL_GET_JSON(farray<3>);

ScopedJsonUserProperties::ScopedJsonUserProperties(const UserPropertyMap* properties)
    : m_previous(g_json_user_properties) {
    g_json_user_properties = properties;
}

ScopedJsonUserProperties::~ScopedJsonUserProperties() {
    g_json_user_properties = m_previous;
}
} // namespace wallpaper

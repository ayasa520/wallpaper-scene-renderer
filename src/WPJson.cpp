#include "WPJson.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "Utils/Identity.hpp"
#include "Utils/Diagnostics.h"
#include "Utils/String.h"
#include "WPDynamicValue.hpp"

namespace wallpaper
{
bool ReadJsonLiteralBoolean(const nlohmann::json& json, std::string_view name,
                            bool default_value) {
    // Literal object flags do not register dynamic properties or coerce strings/numbers.
    // An absent or differently typed entry retains the field's schema default.
    const auto value = json.find(name);
    return value != json.end() && value->is_boolean() ? value->get<bool>() : default_value;
}

namespace
{
thread_local const UserPropertyMap* g_json_user_properties = nullptr;

template<typename T>
struct IsStdVector : std::false_type {};

template<typename T, typename Allocator>
struct IsStdVector<std::vector<T, Allocator>> : std::true_type {};

template<typename T>
bool TryParseNumber(std::string_view text, T& value);

std::string ShortenForLog(std::string_view text, size_t max_length);
std::string ShaderValueToString(const ShaderValue& value);

std::string DescribeUserPropertyValue(const UserPropertyValue& value) {
    if (const auto* shader_value = std::get_if<ShaderValue>(&value)) {
        return std::string("shader(") + ShaderValueToString(*shader_value) + ")";
    }
    return std::string("string(\"") + ShortenForLog(std::get<std::string>(value), 160) + "\")";
}

std::string ShortenForLog(std::string_view text, size_t max_length = 160) {
    if (text.size() <= max_length) return std::string(text);
    return std::string(text.substr(0, max_length)) + "...";
}

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

const nlohmann::json& ResolvePropertyValueNode(const nlohmann::json& json) {
    // Property parsing reads the authored value; the script source is registered separately with
    // the persistent scene host. That host owns init(), user-property callbacks and frame
    // updates. Running those callbacks here would use an incomplete scene/shared state and
    // replace valid base values with derived results before the real script instance has even
    // been initialized. Property timelines also register separately and apply their complete
    // sampled value through that host. A paused timeline must retain its authored base here:
    // substituting c0 collapses vector dimensions before material binding registration and
    // discards the base components needed by relative animation.
    if (json.is_object() && json.contains("value")) return json.at("value");
    return json;
}

bool TryReadJsonNumber(const nlohmann::json& json, double& value) {
    try {
        if (json.is_number()) {
            value = json.get<double>();
            return true;
        }
        if (json.is_boolean()) {
            value = json.get<bool>() ? 1.0 : 0.0;
            return true;
        }
        if (json.is_string()) {
            const auto text = TrimString(json.get<std::string>());
            if (text.empty()) return false;

            char* endptr = nullptr;
            value        = std::strtod(text.c_str(), &endptr);
            return endptr != nullptr && *endptr == '\0';
        }
    } catch (const nlohmann::json::exception&) {
    }
    return false;
}

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
        if (shader_value->size() == 1 && !value.empty()) {
            std::fill(value.begin(), value.end(), (T)(*shader_value)[0]);
            return true;
        }
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

bool TryConvertUserPropertyValue(const UserPropertyValue& property, std::array<float, 2>& value) {
    // A float2 property consumes its own component width, even when the bound color contains
    // three components. Use the same conversion as live user-property updates so loading a scene
    // does not reject a value that the persistent host accepts later. This also keeps numeric
    // broadcasting distinct from a one-component string, whose missing second component is zero
    // in the string reader.
    const auto converted =
        WPDynamicValue::FromUserPropertyValue(property, WPDynamicValue::Type::Float2);
    return converted.has_value() && converted->tryGet(&value);
}

template<typename T, std::size_t N>
bool TryConvertUserPropertyValue(const UserPropertyValue& property, std::array<T, N>& value) {
    if (const auto* shader_value = std::get_if<ShaderValue>(&property)) {
        if (shader_value->size() == 1) {
            value.fill((T)(*shader_value)[0]);
            return true;
        }
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
    const auto* property_entry = FindUserPropertyEntry(g_json_user_properties, binding->name);
    if (property == nullptr || property_entry == nullptr) return false;

    if (! binding->condition.empty()) {
        // Wallpaper Engine uses object-form `user` bindings with `condition` as a branch gate, not
        // as a direct value source. For example, a combo value "0" can mean "Open" while the actual
        // property value is the authored JSON `value=true`; converting that raw "0" string to bool
        // would incorrectly disable camera parallax. When the branch matches, fall through to the
        // authored value node. When it does not match, consume the property with an inactive zero
        // value so the authored branch value cannot leak into the wrong selector state.
        if (MatchesUserPropertyCondition(*property_entry, binding->condition)) {
            return false;
        }
        value = T {};
        return true;
    }

    const bool converted = TryConvertUserPropertyValue(*property, value);
    if constexpr (std::is_same_v<T, std::array<float, 2>>) {
        if (converted && wallpaper::diagnostics::Options().trace_user_bindings) {
            LOG_INFO("SceneUserPropertyBinding: property='%s' type=float2 source=%s resolved=[%.6f %.6f]",
                     binding->name.c_str(),
                     DescribeUserPropertyValue(property_entry->value).c_str(),
                     value[0],
                     value[1]);
        }
    }
    if (!converted) {
        LOG_ERROR("SceneScript: failed to convert direct user binding '%s' raw=%s condition='%s'",
                  binding->name.c_str(),
                  DescribeUserPropertyValue(property_entry->value).c_str(),
                  binding->condition.c_str());
    }
    return converted;
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

    const auto& njson = ResolvePropertyValueNode(json);

    using Tv = typename T::value_type;
    if constexpr (std::is_same_v<T, std::array<float, 2>> ||
                  std::is_same_v<T, std::array<float, 3>> ||
                  std::is_same_v<T, std::array<float, 4>>) {
        if (njson.is_number() || njson.is_string()) return ReadJsonFloatVectorValue(njson, value);
    }
    if (njson.is_number()) {
        value = { njson.get<Tv>() };
        return true;
    }

    if (njson.is_array()) {
        if constexpr (IsStdVector<T>::value) {
            value.clear();
            value.reserve(njson.size());
            for (const auto& item : njson) {
                double component = 0.0;
                if (! TryReadJsonNumber(item, component)) return false;
                value.push_back((Tv)component);
            }
            return true;
        } else {
            if (njson.size() != std::tuple_size_v<T>) return false;

            size_t index = 0;
            for (const auto& item : njson) {
                double component = 0.0;
                if (! TryReadJsonNumber(item, component)) return false;
                value[index++] = (Tv)component;
            }
            return true;
        }
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
        LOG_INFO("%s %s at %s\n%s", e.what(), nameinfo.c_str(), func, json.dump(4).c_str());
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
                LOG_INFO("read json \"%s\" not a key at %s(%s:%d)",
                             name.data(),
                             func,
                             file,
                             line);
            return false;
        } else if (json.at(name).is_null()) {
            if (warn)
                LOG_INFO("read json \"%s\" is null at %s(%s:%d)",
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
T_IMPL_GET_JSON(farray<4>);

bool ReadJsonFloatVectorValue(const nlohmann::json& json, std::span<float> value) {
    // Float2/Float3/Float4 properties share one decoding contract: numbers broadcast to the
    // declared width, while strings initialize independent components separated by ASCII spaces.
    // Zero the destination before reading a string so an omitted component remains zero; stop at
    // the declared width rather than inferring a new property type from the number of tokens.
    // Cold object/material parsing and typed dynamic values must use this same conversion.
    if (json.is_number()) {
        std::fill(value.begin(), value.end(), json.get<float>());
        return true;
    }
    if (!json.is_string()) return false;

    const auto& text = json.get_ref<const std::string&>();
    std::fill(value.begin(), value.end(), 0.0f);
    size_t position { 0 };
    for (auto& component : value) {
        component = static_cast<float>(std::strtod(text.c_str() + position, nullptr));
        const auto separator = text.find(' ', position);
        if (separator == std::string::npos) break;
        position = text.find_first_not_of(' ', separator);
        if (position == std::string::npos) break;
    }
    return true;
}

ScopedJsonUserProperties::ScopedJsonUserProperties(const UserPropertyMap* properties)
    : m_previous(g_json_user_properties) {
    g_json_user_properties = properties;
}

ScopedJsonUserProperties::~ScopedJsonUserProperties() {
    g_json_user_properties = m_previous;
}
} // namespace wallpaper

#include "WPJson.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#include "Utils/Identity.hpp"
#include "Utils/String.h"

namespace wallpaper
{
namespace
{
thread_local const UserPropertyMap* g_json_user_properties = nullptr;
thread_local const nlohmann::json*  g_json_scene_root      = nullptr;

struct StaticCanvasSize {
    double x { 0.0 };
    double y { 0.0 };
    bool   valid { false };
};

enum class StaticValueKind {
    Number,
    Boolean,
    VectorString,
    NumberArray,
};

struct StaticValueState {
    std::vector<double> values;
    StaticValueKind     kind { StaticValueKind::Number };
};

template<typename T>
bool TryParseNumber(std::string_view text, T& value);

std::string ShaderValueToString(const ShaderValue& value);

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

std::optional<nlohmann::json> TryResolveUserPropertyOverrideJson(const nlohmann::json& json) {
    if (g_json_user_properties == nullptr) return std::nullopt;

    const auto binding = ResolveUserPropertyBinding(json);
    if (! binding.has_value()) return std::nullopt;

    const auto* property = LookupUserProperty(g_json_user_properties, binding->name);
    if (property == nullptr) return std::nullopt;

    if (! binding->condition.empty() &&
        ! MatchesUserPropertyCondition(*property, binding->condition)) {
        return std::nullopt;
    }

    if (const auto* shader_value = std::get_if<ShaderValue>(property)) {
        if (shader_value->size() == 1) return (*shader_value)[0];
        return ShaderValueToString(*shader_value);
    }

    return std::get<std::string>(*property);
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

std::optional<StaticCanvasSize> TryReadCanvasSize(const nlohmann::json& root) {
    if (! root.is_object() || ! root.contains("general") || ! root.at("general").is_object()) {
        return std::nullopt;
    }

    const auto& general = root.at("general");
    if (! general.contains("orthogonalprojection") ||
        ! general.at("orthogonalprojection").is_object()) {
        return std::nullopt;
    }

    const auto& ortho = general.at("orthogonalprojection");
    double      width = 0.0;
    double      height = 0.0;
    if (! ortho.contains("width") || ! ortho.contains("height") ||
        ! TryReadJsonNumber(ortho.at("width"), width) ||
        ! TryReadJsonNumber(ortho.at("height"), height)) {
        return std::nullopt;
    }

    return StaticCanvasSize { width, height, true };
}

bool TryParseNumberVectorString(const std::string& source, std::vector<double>& out) {
    std::vector<float> values;
    if (! utils::StrToArray::Convert(source, values)) return false;

    out.resize(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        out[i] = values[i];
    }
    return true;
}

bool TryReadStaticValueState(const nlohmann::json& node, StaticValueState& state) {
    const auto& value_node = ResolvePropertyValueNode(node);

    if (value_node.is_number()) {
        state.kind   = StaticValueKind::Number;
        state.values = { value_node.get<double>() };
        return true;
    }

    if (value_node.is_boolean()) {
        state.kind   = StaticValueKind::Boolean;
        state.values = { value_node.get<bool>() ? 1.0 : 0.0 };
        return true;
    }

    if (value_node.is_array()) {
        std::vector<double> values;
        values.reserve(value_node.size());
        for (const auto& item : value_node) {
            double component = 0.0;
            if (! TryReadJsonNumber(item, component)) return false;
            values.push_back(component);
        }
        if (values.empty()) return false;
        state.kind   = StaticValueKind::NumberArray;
        state.values = std::move(values);
        return true;
    }

    if (value_node.is_string()) {
        std::vector<double> values;
        if (! TryParseNumberVectorString(value_node.get<std::string>(), values) || values.empty()) {
            return false;
        }
        state.kind   = StaticValueKind::VectorString;
        state.values = std::move(values);
        return true;
    }

    return false;
}

std::string FormatStaticVectorString(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(5);
    for (size_t i = 0; i < values.size(); i++) {
        if (i != 0) out << ' ';
        out << values[i];
    }
    return out.str();
}

nlohmann::json SerializeStaticValueState(const StaticValueState& state) {
    switch (state.kind) {
        case StaticValueKind::Boolean:
            return std::abs(state.values.front()) > 0.0001;
        case StaticValueKind::VectorString:
            return FormatStaticVectorString(state.values);
        case StaticValueKind::NumberArray:
            return nlohmann::json(state.values);
        case StaticValueKind::Number:
        default:
            return state.values.front();
    }
}

std::string StripScriptComments(std::string_view source) {
    enum class State {
        Normal,
        Slash,
        LineComment,
        BlockComment,
        SingleQuote,
        DoubleQuote,
        Template,
    };

    std::string output;
    output.reserve(source.size());

    State state = State::Normal;
    for (size_t i = 0; i < source.size(); i++) {
        const char ch = source[i];

        switch (state) {
            case State::Normal:
                if (ch == '/') {
                    state = State::Slash;
                } else {
                    output.push_back(ch);
                    if (ch == '\'') state = State::SingleQuote;
                    else if (ch == '"') state = State::DoubleQuote;
                    else if (ch == '`') state = State::Template;
                }
                break;
            case State::Slash:
                if (ch == '/') {
                    state = State::LineComment;
                } else if (ch == '*') {
                    state = State::BlockComment;
                } else {
                    output.push_back('/');
                    output.push_back(ch);
                    state = State::Normal;
                    if (ch == '\'') state = State::SingleQuote;
                    else if (ch == '"') state = State::DoubleQuote;
                    else if (ch == '`') state = State::Template;
                }
                break;
            case State::LineComment:
                if (ch == '\n') {
                    output.push_back('\n');
                    state = State::Normal;
                }
                break;
            case State::BlockComment:
                if (ch == '*' && i + 1 < source.size() && source[i + 1] == '/') {
                    i++;
                    state = State::Normal;
                } else if (ch == '\n') {
                    output.push_back('\n');
                }
                break;
            case State::SingleQuote:
                output.push_back(ch);
                if (ch == '\\' && i + 1 < source.size()) {
                    output.push_back(source[++i]);
                } else if (ch == '\'') {
                    state = State::Normal;
                }
                break;
            case State::DoubleQuote:
                output.push_back(ch);
                if (ch == '\\' && i + 1 < source.size()) {
                    output.push_back(source[++i]);
                } else if (ch == '"') {
                    state = State::Normal;
                }
                break;
            case State::Template:
                output.push_back(ch);
                if (ch == '\\' && i + 1 < source.size()) {
                    output.push_back(source[++i]);
                } else if (ch == '`') {
                    state = State::Normal;
                }
                break;
        }
    }

    if (state == State::Slash) output.push_back('/');
    return output;
}

class StaticExpressionParser {
public:
    StaticExpressionParser(std::string_view expression,
                           const std::unordered_map<std::string, double>& script_properties,
                           const StaticCanvasSize&                        canvas_size,
                           const StaticValueState&                        current_value)
        : m_expression(expression),
          m_script_properties(script_properties),
          m_canvas_size(canvas_size),
          m_current_value(current_value) {}

    std::optional<double> Parse() {
        auto value = ParseAddSub();
        SkipSpace();
        if (! value.has_value() || m_pos != m_expression.size()) return std::nullopt;
        return value;
    }

private:
    std::optional<double> ParseAddSub() {
        auto lhs = ParseMulDiv();
        while (lhs.has_value()) {
            SkipSpace();
            if (Consume('+')) {
                auto rhs = ParseMulDiv();
                if (! rhs.has_value()) return std::nullopt;
                *lhs += *rhs;
            } else if (Consume('-')) {
                auto rhs = ParseMulDiv();
                if (! rhs.has_value()) return std::nullopt;
                *lhs -= *rhs;
            } else {
                break;
            }
        }
        return lhs;
    }

    std::optional<double> ParseMulDiv() {
        auto lhs = ParseUnary();
        while (lhs.has_value()) {
            SkipSpace();
            if (Consume('*')) {
                auto rhs = ParseUnary();
                if (! rhs.has_value()) return std::nullopt;
                *lhs *= *rhs;
            } else if (Consume('/')) {
                auto rhs = ParseUnary();
                if (! rhs.has_value() || std::abs(*rhs) < 0.000001) return std::nullopt;
                *lhs /= *rhs;
            } else {
                break;
            }
        }
        return lhs;
    }

    std::optional<double> ParseUnary() {
        SkipSpace();
        if (Consume('+')) return ParseUnary();
        if (Consume('-')) {
            auto value = ParseUnary();
            if (! value.has_value()) return std::nullopt;
            return -*value;
        }
        return ParsePrimary();
    }

    std::optional<double> ParsePrimary() {
        SkipSpace();
        if (Consume('(')) {
            auto value = ParseAddSub();
            SkipSpace();
            if (! value.has_value() || ! Consume(')')) return std::nullopt;
            return value;
        }

        if (const auto number = ParseNumber(); number.has_value()) return number;
        if (const auto identifier = ParseIdentifierPath(); identifier.has_value()) {
            return ResolveIdentifier(*identifier);
        }
        return std::nullopt;
    }

    std::optional<double> ParseNumber() {
        SkipSpace();
        const auto remaining = std::string(m_expression.substr(m_pos));
        if (remaining.empty()) return std::nullopt;

        char*  endptr = nullptr;
        double value = std::strtod(remaining.c_str(), &endptr);
        if (endptr == nullptr || endptr == remaining.c_str()) return std::nullopt;

        m_pos += static_cast<size_t>(endptr - remaining.c_str());
        return value;
    }

    std::optional<std::string> ParseIdentifierPath() {
        SkipSpace();
        if (m_pos >= m_expression.size() || ! IsIdentifierStart(m_expression[m_pos])) {
            return std::nullopt;
        }

        const auto start = m_pos;
        m_pos++;
        while (m_pos < m_expression.size() && IsIdentifierChar(m_expression[m_pos])) {
            m_pos++;
        }

        while (m_pos < m_expression.size() && m_expression[m_pos] == '.') {
            m_pos++;
            if (m_pos >= m_expression.size() || ! IsIdentifierStart(m_expression[m_pos])) {
                return std::nullopt;
            }
            m_pos++;
            while (m_pos < m_expression.size() && IsIdentifierChar(m_expression[m_pos])) {
                m_pos++;
            }
        }

        return std::string(m_expression.substr(start, m_pos - start));
    }

    std::optional<double> ResolveIdentifier(const std::string& identifier) const {
        if (identifier == "true") return 1.0;
        if (identifier == "false") return 0.0;

        if (identifier == "engine.canvasSize.x") return m_canvas_size.x;
        if (identifier == "engine.canvasSize.y") return m_canvas_size.y;

        if (identifier == "value") {
            if (m_current_value.values.size() != 1) return std::nullopt;
            return m_current_value.values[0];
        }

        if (identifier == "value.x") return ResolveValueComponent(0);
        if (identifier == "value.y") return ResolveValueComponent(1);
        if (identifier == "value.z") return ResolveValueComponent(2);
        if (identifier == "value.w") return ResolveValueComponent(3);

        constexpr std::string_view prefix = "scriptProperties.";
        if (identifier.rfind(prefix.data(), 0) == 0) {
            const auto it = m_script_properties.find(identifier.substr(prefix.size()));
            if (it == m_script_properties.end()) return std::nullopt;
            return it->second;
        }

        return std::nullopt;
    }

    std::optional<double> ResolveValueComponent(size_t index) const {
        if (index >= m_current_value.values.size()) return std::nullopt;
        return m_current_value.values[index];
    }

    void SkipSpace() {
        while (m_pos < m_expression.size() &&
               std::isspace(static_cast<unsigned char>(m_expression[m_pos]))) {
            m_pos++;
        }
    }

    bool Consume(char ch) {
        SkipSpace();
        if (m_pos >= m_expression.size() || m_expression[m_pos] != ch) return false;
        m_pos++;
        return true;
    }

    static bool IsIdentifierStart(char ch) {
        return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
    }

    static bool IsIdentifierChar(char ch) {
        return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
    }

    std::string_view                          m_expression;
    const std::unordered_map<std::string, double>& m_script_properties;
    const StaticCanvasSize&                   m_canvas_size;
    const StaticValueState&                   m_current_value;
    size_t                                    m_pos { 0 };
};

bool TryResolveScriptPropertyNumber(const nlohmann::json& json, double& value) {
    if (const auto overridden = TryResolveUserPropertyOverrideJson(json); overridden.has_value()) {
        return TryReadJsonNumber(*overridden, value);
    }

    return TryReadJsonNumber(ResolvePropertyValueNode(json), value);
}

bool ApplyScriptAssignment(std::string_view target, double result, StaticValueState& value) {
    if (target.empty()) {
        if (value.values.size() != 1) return false;
        value.values[0] = result;
        return true;
    }

    size_t index = 0;
    if (target == "x") index = 0;
    else if (target == "y") index = 1;
    else if (target == "z") index = 2;
    else if (target == "w") index = 3;
    else return false;

    if (index >= value.values.size()) return false;
    value.values[index] = result;
    return true;
}

std::optional<nlohmann::json> TryResolveStaticScriptValueNode(const nlohmann::json& node) {
    if (! node.is_object() || ! node.contains("script") || ! node.contains("scriptproperties")) {
        return std::nullopt;
    }
    if (ResolveUserPropertyBinding(node).has_value()) return std::nullopt;
    if (g_json_scene_root == nullptr) return std::nullopt;

    const auto canvas_size = TryReadCanvasSize(*g_json_scene_root);
    if (! canvas_size.has_value() || ! canvas_size->valid || ! node.at("script").is_string() ||
        ! node.at("scriptproperties").is_object()) {
        return std::nullopt;
    }

    StaticValueState value_state;
    if (! TryReadStaticValueState(node, value_state) || value_state.values.empty()) {
        return std::nullopt;
    }

    std::unordered_map<std::string, double> script_properties;
    for (const auto& [name, property_node] : node.at("scriptproperties").items()) {
        double property_value = 0.0;
        if (! TryResolveScriptPropertyNumber(property_node, property_value)) return std::nullopt;
        script_properties.emplace(name, property_value);
    }

    const auto stripped_script = StripScriptComments(node.at("script").get<std::string>());
    const std::regex assignment_regex(R"(\bvalue(?:\.(x|y|z|w))?\s*=\s*([^;]+);)");

    bool updated = false;
    for (std::sregex_iterator it(stripped_script.begin(), stripped_script.end(), assignment_regex),
         end;
         it != end;
         ++it) {
        const auto target = (*it)[1].matched ? (*it)[1].str() : std::string {};
        const auto expression = (*it)[2].str();

        StaticExpressionParser parser(expression, script_properties, *canvas_size, value_state);
        const auto evaluated = parser.Parse();
        if (! evaluated.has_value()) return std::nullopt;
        if (! ApplyScriptAssignment(target, *evaluated, value_state)) return std::nullopt;
        updated = true;
    }

    if (! updated) return std::nullopt;
    return SerializeStaticValueState(value_state);
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

    if (const auto scripted = TryResolveStaticScriptValueNode(json); scripted.has_value()) {
        const auto& njson = *scripted;
        if (njson.is_number()) {
            value = { njson.get<typename T::value_type>() };
            return true;
        }

        std::string strvalue = njson.get<std::string>();
        return utils::StrToArray::Convert(strvalue, value);
    }

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

    if (const auto scripted = TryResolveStaticScriptValueNode(json); scripted.has_value()) {
        value = scripted->get<T>();
        return true;
    }

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

ScopedJsonUserProperties::ScopedJsonUserProperties(const UserPropertyMap* properties,
                                                   const nlohmann::json*  root)
    : m_previous(g_json_user_properties), m_previous_root(g_json_scene_root) {
    g_json_user_properties = properties;
    g_json_scene_root      = root;
}

ScopedJsonUserProperties::~ScopedJsonUserProperties() {
    g_json_user_properties = m_previous;
    g_json_scene_root      = m_previous_root;
}
} // namespace wallpaper

#include "WPScriptRuntime.hpp"

#include <cmath>
#include <sstream>
#include <utility>

#include "Utils/Logging.h"

extern "C" {
#include "quickjs.h"
}

namespace wallpaper
{
namespace
{

struct RuntimeState {
    JSRuntime* runtime { nullptr };
    JSContext* context { nullptr };
};

std::string StripScriptModuleSyntax(std::string source) {
    std::string::size_type pos = 0;
    while ((pos = source.find("'use strict';", pos)) != std::string::npos) {
        source.erase(pos, 13);
    }

    pos = 0;
    while ((pos = source.find("\"use strict\";", pos)) != std::string::npos) {
        source.erase(pos, 13);
    }

    pos = 0;
    while ((pos = source.find("export ", pos)) != std::string::npos) {
        source.erase(pos, 7);
    }

    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t");
        const auto trimmed =
            first == std::string::npos ? std::string_view {} : std::string_view(line).substr(first);
        if (trimmed.starts_with("import ")) {
            continue;
        }
        output << line << '\n';
    }

    return output.str();
}

void LogQuickJSException(JSContext* context, const char* stage) {
    JSValue exception = JS_GetException(context);

    std::string message = "<unknown>";
    if (const char* text = JS_ToCString(context, exception)) {
        message = text;
        JS_FreeCString(context, text);
    }

    std::string stack;
    JSValue stack_value = JS_GetPropertyStr(context, exception, "stack");
    if (!JS_IsException(stack_value) && !JS_IsUndefined(stack_value)) {
        if (const char* text = JS_ToCString(context, stack_value)) {
            stack = text;
            JS_FreeCString(context, text);
        }
    }
    JS_FreeValue(context, stack_value);

    if (stack.empty()) {
        LOG_ERROR("QuickJS %s failed: %s", stage, message.c_str());
    } else {
        LOG_ERROR("QuickJS %s failed: %s\n%s", stage, message.c_str(), stack.c_str());
    }

    JS_FreeValue(context, exception);
}

JSValue NumericVectorToJS(JSContext* context, const std::vector<double>& values) {
    static constexpr const char* names[] = { "x", "y", "z", "w" };

    if (values.size() <= 1) {
        return JS_NewFloat64(context, values.empty() ? 0.0 : values.front());
    }

    if (values.size() <= 4) {
        JSValue object = JS_NewObject(context);
        for (size_t index = 0; index < values.size(); index++) {
            JS_SetPropertyStr(context, object, names[index], JS_NewFloat64(context, values[index]));
        }
        return object;
    }

    JSValue array = JS_NewArray(context);
    for (uint32_t index = 0; index < values.size(); index++) {
        JS_SetPropertyUint32(context,
                             array,
                             index,
                             JS_NewFloat64(context, values[static_cast<size_t>(index)]));
    }
    return array;
}

JSValue ScriptValueToJS(JSContext* context, const WPScriptValue& value) {
    switch (value.shape) {
        case WPScriptValueShape::Number:
            return JS_NewFloat64(context,
                                 value.numeric_values.empty() ? 0.0 : value.numeric_values.front());
        case WPScriptValueShape::Boolean:
            return JS_NewBool(context, value.boolean_value);
        case WPScriptValueShape::String:
            return JS_NewStringLen(
                context, value.string_value.c_str(), value.string_value.size());
        case WPScriptValueShape::NumberArray:
        case WPScriptValueShape::VectorString:
            return NumericVectorToJS(context, value.numeric_values);
    }

    return JS_UNDEFINED;
}

bool ReadJSNumber(JSContext* context, JSValueConst value, double* out_value) {
    if (!out_value)
        return false;

    if (JS_IsException(value) || JS_IsUndefined(value) || JS_IsNull(value) || JS_IsObject(value))
        return false;

    if (JS_ToFloat64(context, out_value, value) != 0)
        return false;

    return std::isfinite(*out_value);
}

bool ReadJSString(JSContext* context, JSValueConst value, std::string* out_value) {
    if (!out_value)
        return false;

    const char* text = JS_ToCString(context, value);
    if (!text)
        return false;

    *out_value = text;
    JS_FreeCString(context, text);
    return true;
}

bool ReadJSObjectNumber(JSContext* context,
                        JSValueConst value,
                        const char*  property_name,
                        double*      out_value) {
    JSValue property = JS_GetPropertyStr(context, value, property_name);
    const bool ok = !JS_IsException(property) && !JS_IsUndefined(property) &&
        ReadJSNumber(context, property, out_value);
    JS_FreeValue(context, property);
    return ok;
}

bool ReadJSArrayLength(JSContext* context, JSValueConst value, uint32_t* out_length) {
    JSValue length_value = JS_GetPropertyStr(context, value, "length");
    const bool ok = !JS_IsException(length_value) &&
        JS_ToUint32(context, out_length, length_value) == 0;
    JS_FreeValue(context, length_value);
    return ok;
}

bool ReadJSNumericVector(JSContext* context,
                         JSValueConst value,
                         size_t       expected_size,
                         std::vector<double>* out_values) {
    if (!out_values)
        return false;

    out_values->clear();
    out_values->reserve(expected_size);

    if (expected_size <= 1) {
        double number = 0.0;
        if (!ReadJSNumber(context, value, &number))
            return false;
        out_values->push_back(number);
        return true;
    }

    static constexpr const char* names[] = { "x", "y", "z", "w" };
    if (expected_size <= 4 && JS_IsObject(value) && !JS_IsArray(value)) {
        for (size_t index = 0; index < expected_size; index++) {
            double component = 0.0;
            if (!ReadJSObjectNumber(context, value, names[index], &component))
                return false;
            out_values->push_back(component);
        }
        return true;
    }

    double scalar = 0.0;
    if (ReadJSNumber(context, value, &scalar)) {
        out_values->assign(expected_size, scalar);
        return true;
    }

    if (!JS_IsArray(value))
        return false;

    uint32_t length = 0;
    if (!ReadJSArrayLength(context, value, &length) || length < expected_size)
        return false;

    for (uint32_t index = 0; index < expected_size; index++) {
        JSValue item = JS_GetPropertyUint32(context, value, index);
        double number = 0.0;
        const bool ok = !JS_IsException(item) && ReadJSNumber(context, item, &number);
        JS_FreeValue(context, item);
        if (!ok)
            return false;
        out_values->push_back(number);
    }

    return true;
}

std::optional<WPScriptValue> ScriptValueFromJS(JSContext*                 context,
                                               JSValueConst               value,
                                               const WPScriptValue&       hint) {
    switch (hint.shape) {
        case WPScriptValueShape::Number: {
            double number = 0.0;
            if (!ReadJSNumber(context, value, &number))
                return std::nullopt;
            return WPScriptValue::Number(number);
        }
        case WPScriptValueShape::Boolean: {
            const int result = JS_ToBool(context, value);
            if (result < 0)
                return std::nullopt;
            return WPScriptValue::Boolean(result != 0);
        }
        case WPScriptValueShape::String: {
            std::string text;
            if (!ReadJSString(context, value, &text))
                return std::nullopt;
            return WPScriptValue::String(std::move(text));
        }
        case WPScriptValueShape::NumberArray:
        case WPScriptValueShape::VectorString: {
            std::vector<double> values;
            if (!ReadJSNumericVector(context, value, hint.numeric_values.size(), &values))
                return std::nullopt;

            if (hint.shape == WPScriptValueShape::NumberArray)
                return WPScriptValue::NumberArray(std::move(values));

            return WPScriptValue::VectorString(std::move(values));
        }
    }

    return std::nullopt;
}

std::string BuildWrappedScript(std::string_view script_source) {
    const std::string body = StripScriptModuleSyntax(std::string(script_source));

    std::ostringstream wrapper;
    wrapper << "(function() {\n"
            << "  const __props = globalThis.__scriptProps;\n"
            << "  const __initialValue = globalThis.__currentValue;\n"
            << "  const shared = (globalThis.shared && typeof globalThis.shared === 'object')\n"
            << "    ? globalThis.shared\n"
            << "    : (globalThis.shared = {});\n"
            << "  const console = (globalThis.console && typeof globalThis.console === 'object')\n"
            << "    ? globalThis.console\n"
            << "    : (globalThis.console = {\n"
            << "        log() {},\n"
            << "        info() {},\n"
            << "        warn() {},\n"
            << "        error() {}\n"
            << "      });\n"
            << "  const WEMath = (globalThis.WEMath && typeof globalThis.WEMath === 'object')\n"
            << "    ? globalThis.WEMath\n"
            << "    : (globalThis.WEMath = {\n"
            << "        mix(a, b, t) {\n"
            << "          const lerp = (x, y, alpha) => Number(x) + (Number(y) - Number(x)) * Number(alpha);\n"
            << "          if (Array.isArray(a) && Array.isArray(b)) {\n"
            << "            return a.map((value, index) => lerp(value, b[index] ?? value, t));\n"
            << "          }\n"
            << "          if (a && b && typeof a === 'object' && typeof b === 'object') {\n"
            << "            const out = Array.isArray(a) ? [] : {};\n"
            << "            for (const key of Object.keys(a)) {\n"
            << "              out[key] = lerp(a[key], b[key] ?? a[key], t);\n"
            << "            }\n"
            << "            return out;\n"
            << "          }\n"
            << "          return lerp(a, b, t);\n"
            << "        },\n"
            << "        clamp(value, minValue, maxValue) {\n"
            << "          return Math.min(Math.max(Number(value), Number(minValue)), Number(maxValue));\n"
            << "        }\n"
            << "      });\n"
            << "  const engine = (globalThis.engine && typeof globalThis.engine === 'object')\n"
            << "    ? globalThis.engine\n"
            << "    : {};\n"
            << "  if (typeof engine.isRunningInEditor !== 'function') engine.isRunningInEditor = () => false;\n"
            << "  if (typeof engine.registerAudioBuffers !== 'function') {\n"
            << "    engine.registerAudioBuffers = (resolution) => ({\n"
            << "      average: Array.from({ length: Math.max(0, Number(resolution) || 0) }, () => 0)\n"
            << "    });\n"
            << "  }\n"
            << "  if (typeof engine.setTimeout !== 'function') engine.setTimeout = () => 0;\n"
            << "  if (typeof engine.clearTimeout !== 'function') engine.clearTimeout = () => {};\n"
            << "  if (typeof engine.frametime !== 'number') engine.frametime = 1 / 60;\n"
            << "  if (engine.AUDIO_RESOLUTION_16 === undefined) engine.AUDIO_RESOLUTION_16 = 16;\n"
            << "  if (!engine.userProperties || typeof engine.userProperties !== 'object') engine.userProperties = {};\n"
            << "  globalThis.engine = engine;\n"
            << "  const thisObject = (globalThis.thisObject && typeof globalThis.thisObject === 'object')\n"
            << "    ? globalThis.thisObject\n"
            << "    : (globalThis.thisObject = {});\n"
            << "  const thisLayer = (globalThis.thisLayer && typeof globalThis.thisLayer === 'object')\n"
            << "    ? globalThis.thisLayer\n"
            << "    : (globalThis.thisLayer = {});\n"
            << "  function createScriptProperties() {\n"
            << "    const applyOption = function(opts) {\n"
            << "      if (opts && opts.name !== undefined && !Object.prototype.hasOwnProperty.call(__props, opts.name)) {\n"
            << "        __props[opts.name] = opts.value;\n"
            << "      }\n"
            << "      return builder;\n"
            << "    };\n"
            << "    const target = {\n"
            << "      finish() { return __props; }\n"
            << "    };\n"
            << "    const builder = new Proxy(target, {\n"
            << "      get(obj, prop, receiver) {\n"
            << "        if (prop in obj) return Reflect.get(obj, prop, receiver);\n"
            << "        if (typeof prop === 'string' && prop.startsWith('add')) return applyOption;\n"
            << "        return undefined;\n"
            << "      }\n"
            << "    });\n"
            << "    return builder;\n"
            << "  }\n"
            << body << "\n"
            << "  let __result = __initialValue;\n"
            << "  if (typeof init === 'function') {\n"
            << "    const __initResult = init();\n"
            << "    if (__initResult !== undefined) __result = __initResult;\n"
            << "  }\n"
            << "  if (typeof update === 'function') {\n"
            << "    const __updatedResult = update(__result);\n"
            << "    if (__updatedResult !== undefined) __result = __updatedResult;\n"
            << "  }\n"
            << "  return __result;\n"
            << "})();\n";
    return wrapper.str();
}

void InstallGlobalValue(JSContext* context, JSValueConst global, const char* name, JSValue value) {
    JS_SetPropertyStr(context, global, name, value);
}

} // namespace

struct WPScriptRuntime::Opaque {
    RuntimeState state;
};

WPScriptValue WPScriptValue::Number(double value) {
    WPScriptValue result;
    result.shape = WPScriptValueShape::Number;
    result.numeric_values = { value };
    return result;
}

WPScriptValue WPScriptValue::Boolean(bool value) {
    WPScriptValue result;
    result.shape = WPScriptValueShape::Boolean;
    result.boolean_value = value;
    return result;
}

WPScriptValue WPScriptValue::String(std::string value) {
    WPScriptValue result;
    result.shape = WPScriptValueShape::String;
    result.string_value = std::move(value);
    return result;
}

WPScriptValue WPScriptValue::NumberArray(std::vector<double> values) {
    WPScriptValue result;
    result.shape = WPScriptValueShape::NumberArray;
    result.numeric_values = std::move(values);
    return result;
}

WPScriptValue WPScriptValue::VectorString(std::vector<double> values) {
    WPScriptValue result;
    result.shape = WPScriptValueShape::VectorString;
    result.numeric_values = std::move(values);
    return result;
}

bool WPScriptValue::isNumericVector() const noexcept {
    return shape == WPScriptValueShape::NumberArray || shape == WPScriptValueShape::VectorString;
}

WPScriptRuntime::WPScriptRuntime()
    : m_impl(new Opaque()) {
    m_impl->state.runtime = JS_NewRuntime();
    if (!m_impl->state.runtime) {
        LOG_ERROR("failed to create QuickJS runtime");
        return;
    }

    m_impl->state.context = JS_NewContext(m_impl->state.runtime);
    if (!m_impl->state.context) {
        LOG_ERROR("failed to create QuickJS context");
        JS_FreeRuntime(m_impl->state.runtime);
        m_impl->state.runtime = nullptr;
        return;
    }
}

WPScriptRuntime::~WPScriptRuntime() {
    if (!m_impl)
        return;

    if (m_impl->state.context)
        JS_FreeContext(m_impl->state.context);
    if (m_impl->state.runtime)
        JS_FreeRuntime(m_impl->state.runtime);

    delete m_impl;
}

bool WPScriptRuntime::isReady() const noexcept {
    return m_impl && m_impl->state.runtime && m_impl->state.context;
}

std::optional<WPScriptValue> WPScriptRuntime::evaluate(std::string_view                  script_source,
                                                       const WPScriptValue&             current_value,
                                                       const WPScriptEvaluationContext& context) {
    if (!isReady())
        return std::nullopt;

    JSContext* js_context = m_impl->state.context;

    JSValue script_props = JS_NewObject(js_context);
    for (const auto& [name, value] : context.script_properties) {
        JS_SetPropertyStr(js_context, script_props, name.c_str(), ScriptValueToJS(js_context, value));
    }

    JSValue engine = JS_NewObject(js_context);
    JSValue canvas_size = JS_NewObject(js_context);
    JS_SetPropertyStr(js_context, canvas_size, "x", JS_NewFloat64(js_context, context.canvas_size[0]));
    JS_SetPropertyStr(js_context, canvas_size, "y", JS_NewFloat64(js_context, context.canvas_size[1]));
    JS_SetPropertyStr(js_context, engine, "canvasSize", canvas_size);

    JSValue global = JS_GetGlobalObject(js_context);
    InstallGlobalValue(js_context, global, "__scriptProps", JS_DupValue(js_context, script_props));
    InstallGlobalValue(js_context, global, "__currentValue", ScriptValueToJS(js_context, current_value));
    InstallGlobalValue(js_context, global, "engine", engine);

    const std::string wrapped = BuildWrappedScript(script_source);
    JSValue result = JS_Eval(js_context,
                             wrapped.c_str(),
                             wrapped.size(),
                             "<scene-script>",
                             JS_EVAL_TYPE_GLOBAL);

    InstallGlobalValue(js_context, global, "__scriptProps", JS_UNDEFINED);
    InstallGlobalValue(js_context, global, "__currentValue", JS_UNDEFINED);
    InstallGlobalValue(js_context, global, "engine", JS_UNDEFINED);
    JS_FreeValue(js_context, global);
    JS_FreeValue(js_context, script_props);

    if (JS_IsException(result)) {
        LogQuickJSException(js_context, "evaluation");
        JS_FreeValue(js_context, result);
        return std::nullopt;
    }

    auto converted = ScriptValueFromJS(js_context, result, current_value);
    JS_FreeValue(js_context, result);
    if (!converted.has_value()) {
        LOG_ERROR("failed to convert QuickJS result back to native script value");
        return std::nullopt;
    }

    return converted;
}

} // namespace wallpaper

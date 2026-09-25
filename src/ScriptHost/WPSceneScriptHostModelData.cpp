#include "WPSceneScriptHostShared.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "Fs/VFS.h"
#include "Scene/Scene.h"
#include "Scene/SceneModelData.h"
#include "Utils/Logging.h"

namespace wallpaper
{
namespace
{
class ModelJSValue {
public:
    ModelJSValue(JSContext* context, JSValue value) : m_context(context), m_value(value) {}
    ~ModelJSValue() { JS_FreeValue(m_context, m_value); }
    ModelJSValue(const ModelJSValue&) = delete;
    ModelJSValue& operator=(const ModelJSValue&) = delete;
    JSValue Get() const { return m_value; }
    JSValue Release() { return std::exchange(m_value, JS_UNDEFINED); }

private:
    JSContext* m_context;
    JSValue m_value;
};

bool ModelTypeError(JSContext* context, const char* message) {
    JS_ThrowTypeError(context, "%s", message);
    return false;
}

template<typename T>
bool CopyTypedView(JSContext* context, JSValueConst value, JSTypedArrayEnum type,
                   std::vector<T>& output) {
    if (JS_GetTypedArrayType(value) != type) {
        return ModelTypeError(context, "ModelData buffer has the wrong typed-array type");
    }
    size_t offset = 0;
    size_t bytes = 0;
    size_t element_bytes = 0;
    ModelJSValue backing(context,
                         JS_GetTypedArrayBuffer(context, value, &offset, &bytes, &element_bytes));
    if (JS_IsException(backing.Get())) return false;
    size_t backing_bytes = 0;
    const auto* data = JS_GetArrayBuffer(context, &backing_bytes, backing.Get());
    if (data == nullptr) {
        return JS_HasException(context)
                   ? false : ModelTypeError(context, "ModelData buffer has no readable storage");
    }
    if (element_bytes != sizeof(T) || offset > backing_bytes || bytes > backing_bytes - offset ||
        bytes % sizeof(T) != 0) {
        return ModelTypeError(context, "ModelData typed-array view is outside its backing buffer");
    }
    // Copy this exact view before reading any other JS property. A later getter may detach or
    // replace the buffer, or re-enter model APIs; retaining the JS object alone would not make a
    // borrowed backing-store pointer safe across those calls or the subsequent Vulkan upload.
    output.resize(bytes / sizeof(T));
    if (bytes != 0) std::memcpy(output.data(), data + offset, bytes);
    return true;
}

bool ReadModelFormat(JSContext* context, JSValueConst value, uint32_t& format) {
    if (! JS_IsArray(value)) {
        return ModelTypeError(context, "vertexFormat must be an array of IModelData constants");
    }
    ModelJSValue length(context, JS_GetPropertyStr(context, value, "length"));
    uint32_t count = 0;
    if (JS_IsException(length.Get()) || JS_ToUint32(context, &count, length.Get()) < 0) return false;
    format = 0;
    for (uint32_t i = 0; i < count; ++i) {
        ModelJSValue item(context, JS_GetPropertyUint32(context, value, i));
        if (JS_IsException(item.Get())) return false;
        if (! JS_IsString(item.Get())) {
            return ModelTypeError(context, "vertexFormat entries must be IModelData strings");
        }
        size_t size = 0;
        const char* chars = JS_ToCStringLen(context, &size, item.Get());
        if (chars == nullptr) return false;
        const std::string_view name(chars, size);
        uint32_t mask = 0;
        for (const auto& attribute : ModelVertexAttributes()) {
            if (name == attribute.name) mask = attribute.mask;
        }
        JS_FreeCString(context, chars);
        if (mask == 0) return ModelTypeError(context, "unknown vertexFormat attribute");
        format |= mask;
    }
    return format != 0 || ModelTypeError(context, "vertexFormat must not be empty");
}

bool ReadModelBool(JSContext* context, JSValueConst shape, const char* name,
                   std::optional<bool>& result) {
    ModelJSValue value(context, JS_GetPropertyStr(context, shape, name));
    if (JS_IsException(value.Get())) return false;
    if (JS_IsBool(value.Get())) result = JS_ToBool(context, value.Get()) != 0;
    return true;
}

bool ReadModelShape(JSContext* context, JSValueConst value, SceneModelShapeUpdate& shape) {
    if (JS_IsNull(value)) {
        shape.remove = true;
        return true;
    }
    if (! JS_IsObject(value) || JS_IsArray(value)) {
        return ModelTypeError(context, "each ModelData shape must be an object");
    }
    {
        ModelJSValue buffer(context, JS_GetPropertyStr(context, value, "vertexBuffer"));
        if (JS_IsException(buffer.Get())) return false;
        if (! JS_IsUndefined(buffer.Get())) {
            shape.vertex_buffer.emplace();
            if (! CopyTypedView(context, buffer.Get(), JS_TYPED_ARRAY_FLOAT32,
                                *shape.vertex_buffer)) return false;
        }
    }
    {
        ModelJSValue buffer(context, JS_GetPropertyStr(context, value, "indexBuffer"));
        if (JS_IsException(buffer.Get())) return false;
        if (JS_IsNull(buffer.Get())) {
            shape.index_buffer.emplace(std::monostate {});
        } else if (! JS_IsUndefined(buffer.Get())) {
            const auto type = JS_GetTypedArrayType(buffer.Get());
            if (type == JS_TYPED_ARRAY_UINT16) {
                std::vector<uint16_t> indices;
                if (! CopyTypedView(context, buffer.Get(), JS_TYPED_ARRAY_UINT16, indices))
                    return false;
                shape.index_buffer.emplace(std::move(indices));
            } else if (type == JS_TYPED_ARRAY_UINT32) {
                std::vector<uint32_t> indices;
                if (! CopyTypedView(context, buffer.Get(), JS_TYPED_ARRAY_UINT32, indices))
                    return false;
                shape.index_buffer.emplace(std::move(indices));
            } else {
                return ModelTypeError(context, "indexBuffer must be Uint16Array or Uint32Array");
            }
        }
    }
    {
        ModelJSValue format(context, JS_GetPropertyStr(context, value, "vertexFormat"));
        if (JS_IsException(format.Get())) return false;
        if (! JS_IsUndefined(format.Get())) {
            shape.vertex_format.emplace();
            if (! ReadModelFormat(context, format.Get(), *shape.vertex_format)) return false;
        }
    }
    {
        ModelJSValue material(context, JS_GetPropertyStr(context, value, "material"));
        if (JS_IsException(material.Get())) return false;
        if (! JS_IsUndefined(material.Get())) {
            auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
            shape.material = ResolveSceneScriptAssetFile(opaque->scene, context, material.Get());
            if (! shape.material) {
                if (JS_HasException(context)) return false;
                return ModelTypeError(context, "material must be a registered asset handle");
            }
            if (! opaque->scene->vfs->Contains("/assets/" + *shape.material)) {
                return ModelTypeError(context, "ModelData material asset does not exist");
            }
        }
    }
    return ReadModelBool(context, value, "isVertexBufferDynamic", shape.vertex_dynamic) &&
           ReadModelBool(context, value, "isIndexBufferDynamic", shape.index_dynamic);
}

bool ReadModelVector(JSContext* context, JSValueConst value, Eigen::Vector3f& result) {
    if (! JS_IsObject(value)) return ModelTypeError(context, "ModelData bounds must be Vec3 values");
    constexpr std::array names { "x", "y", "z" };
    for (size_t i = 0; i < names.size(); ++i) {
        ModelJSValue component(context, JS_GetPropertyStr(context, value, names[i]));
        if (JS_IsException(component.Get())) return false;
        double number = 0.0;
        if (! JS_IsNumber(component.Get())) {
            return ModelTypeError(context, "ModelData bounds components must be numbers");
        }
        if (JS_ToFloat64(context, &number, component.Get()) < 0) return false;
        result[static_cast<Eigen::Index>(i)] = static_cast<float>(number);
    }
    return true;
}

bool ReadModelUpdate(JSContext* context, JSValueConst value, SceneModelDataUpdate& data) {
    if (! JS_IsObject(value) || JS_IsArray(value)) {
        return ModelTypeError(context, "ModelData expects a shape object or { shapes: [...] }");
    }
    ModelJSValue shapes(context, JS_GetPropertyStr(context, value, "shapes"));
    if (JS_IsException(shapes.Get())) return false;
    if (JS_IsUndefined(shapes.Get())) {
        data.shapes.emplace_back();
        if (! ReadModelShape(context, value, data.shapes.back())) return false;
    } else {
        if (! JS_IsArray(shapes.Get())) {
            return ModelTypeError(context, "ModelData shapes must be an array");
        }
        ModelJSValue length(context, JS_GetPropertyStr(context, shapes.Get(), "length"));
        uint32_t count = 0;
        if (JS_IsException(length.Get()) || JS_ToUint32(context, &count, length.Get()) < 0)
            return false;
        if (count == 0) return ModelTypeError(context, "ModelData shapes must not be empty");
        data.shapes.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            ModelJSValue shape(context, JS_GetPropertyUint32(context, shapes.Get(), i));
            if (JS_IsException(shape.Get()) || ! ReadModelShape(context, shape.Get(), data.shapes[i]))
                return false;
        }
    }
    ModelJSValue mins(context, JS_GetPropertyStr(context, value, "boundingBoxMins"));
    if (JS_IsException(mins.Get())) return false;
    ModelJSValue maxs(context, JS_GetPropertyStr(context, value, "boundingBoxMaxs"));
    if (JS_IsException(maxs.Get())) return false;
    if (! JS_IsUndefined(mins.Get()) || ! JS_IsUndefined(maxs.Get())) {
        data.bounds.emplace();
        if (! ReadModelVector(context, mins.Get(), data.bounds->min) ||
            ! ReadModelVector(context, maxs.Get(), data.bounds->max)) return false;
    }
    return true;
}

const uint32_t* ModelToken(JSContext* context, JSValueConst value) {
    const auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
    return static_cast<const uint32_t*>(JS_GetOpaque2(context, value, opaque->model_data_class));
}

void FinalizeModelHandle(JSRuntime*, JSValue value) {
    // The wrapper contains only an identity. Explicit destroyModelData and model owners release
    // the scene resource; JS garbage collection only releases this small token allocation.
    delete static_cast<uint32_t*>(JS_GetOpaque(value, JS_GetClassID(value)));
}

JSValue ModelConstructor(JSContext* context, JSValueConst, int, JSValueConst*) {
    return JS_ThrowTypeError(context, "create IModelData with thisScene.createModelData");
}

JSValue ModelToConfigString(JSContext* context, JSValueConst receiver, int, JSValueConst*) {
    const auto* token = ModelToken(context, receiver);
    return token != nullptr ? JS_NewUint32(context, *token) : JS_EXCEPTION;
}

JSValue NativeCreateModelData(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
    auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
    if (opaque->execution_phase == SceneScriptExecutionPhase::GlobalEvaluation) {
        return JS_ThrowTypeError(context, "createModelData is not available during global evaluation");
    }
    if (argc < 1) return JS_ThrowTypeError(context, "createModelData requires a configuration");
    try {
        SceneModelDataUpdate data;
        if (! ReadModelUpdate(context, argv[0], data)) return JS_EXCEPTION;
        std::string error;
        auto model = SceneModelData::Create(data, error);
        if (! model) return JS_ThrowTypeError(context, "createModelData: %s", error.c_str());
        ModelJSValue handle(context, JS_NewObjectClass(context, opaque->model_data_class));
        if (JS_IsException(handle.Get())) return JS_EXCEPTION;
        auto token = std::make_unique<uint32_t>(0);
        *token = opaque->scene->modelData.Register(std::move(model));
        if (*token == 0) return JS_ThrowRangeError(context, "ModelData token range exhausted");
        if (JS_DefinePropertyValueStr(context, handle.Get(), "__modelDataToken",
                                      JS_NewUint32(context, *token), 0) < 0) {
            opaque->scene->modelData.Release(*token);
            return JS_EXCEPTION;
        }
        JS_SetOpaque(handle.Get(), token.release());
        return handle.Release();
    } catch (const std::bad_alloc&) {
        return JS_ThrowOutOfMemory(context);
    }
}

JSValue NativeDestroyModelData(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
    auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
    if (opaque->execution_phase == SceneScriptExecutionPhase::GlobalEvaluation) {
        return JS_ThrowTypeError(context, "destroyModelData is not available during global evaluation");
    }
    if (argc < 1) return JS_ThrowTypeError(context, "destroyModelData requires a model handle");
    uint32_t token = 0;
    if (JS_IsNumber(argv[0])) {
        double number = 0.0;
        if (JS_ToFloat64(context, &number, argv[0]) < 0) return JS_EXCEPTION;
        if (! std::isfinite(number) || number <= 0.0 ||
            number > std::numeric_limits<uint32_t>::max() || std::floor(number) != number) {
            return JS_ThrowTypeError(context, "destroyModelData requires a valid model token");
        }
        token = static_cast<uint32_t>(number);
    } else {
        const auto* handle_token = ModelToken(context, argv[0]);
        if (handle_token == nullptr) return JS_EXCEPTION;
        token = *handle_token;
    }
    if (! opaque->scene->modelData.Release(token)) {
        return JS_ThrowTypeError(context, "destroyModelData: model handle is already released");
    }
    return JS_UNDEFINED;
}

JSValue NativeApplyModelData(JSContext* context, JSValueConst receiver, int argc,
                              JSValueConst* argv) {
    auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
    const auto* token = ModelToken(context, receiver);
    if (token == nullptr) return JS_EXCEPTION;
    auto model = opaque->scene->modelData.Find(*token);
    if (! model) return JS_ThrowTypeError(context, "applyData: model resource has been released");
    if (argc < 1) return JS_ThrowTypeError(context, "applyData requires shape updates");
    try {
        const bool trace = wallpaper::diagnostics::Options().trace_model_data;
        const auto start = trace ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point {};
        SceneModelDataUpdate data;
        if (! ReadModelUpdate(context, argv[0], data)) return JS_EXCEPTION;
        const auto copied = trace ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point {};
        std::string error;
        if (! model->ApplyData(data, error)) {
            return JS_ThrowTypeError(context, "applyData: %s", error.c_str());
        }
        if (trace) {
            const auto applied = std::chrono::steady_clock::now();
            const auto micros = [](auto duration) {
                return std::chrono::duration<double, std::micro>(duration).count();
            };
            LOG_INFO("SceneModelDataApply: token=%u shapes=%zu copy-us=%.3f apply-us=%.3f",
                     *token, data.shapes.size(), micros(copied - start), micros(applied - copied));
        }
        return JS_UNDEFINED;
    } catch (const std::bad_alloc&) {
        return JS_ThrowOutOfMemory(context);
    }
}

JSValue NativeReplaceModelData(JSContext* context, JSValueConst receiver, int argc,
                               JSValueConst* argv) {
    auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
    if (opaque->execution_phase == SceneScriptExecutionPhase::Update) {
        return JS_ThrowTypeError(context, "replaceData is not available in an update callback");
    }
    const auto* token = ModelToken(context, receiver);
    if (token == nullptr) return JS_EXCEPTION;
    auto model = opaque->scene->modelData.Find(*token);
    if (! model) return JS_ThrowTypeError(context, "replaceData: model resource has been released");
    if (argc < 1) return JS_ThrowTypeError(context, "replaceData requires shape updates");
    try {
        SceneModelDataUpdate data;
        if (! ReadModelUpdate(context, argv[0], data)) return JS_EXCEPTION;
        // A material callback can replace the very resource it is evaluating. Publish CPU data
        // now, but retire/re-register draw materials only outside JS dispatch. Queue before
        // publication so a later shape error still refreshes earlier structural changes. The
        // frame drain compares owner revisions, making unchanged/payload-only requests cheap.
        opaque->pending_model_refresh_tokens.insert(*token);
        std::string error;
        const bool succeeded = model->ReplaceData(data, error);
        if (wallpaper::diagnostics::Options().trace_model_data) {
            LOG_INFO("SceneModelDataReplace: token=%u shapes=%zu revision=%llu success=%s",
                     *token, model->Shapes().size(),
                     static_cast<unsigned long long>(model->StructureRevision()),
                     succeeded ? "true" : "false");
        }
        if (! succeeded) return JS_ThrowTypeError(context, "replaceData: %s", error.c_str());
        return JS_UNDEFINED;
    } catch (const std::bad_alloc&) {
        return JS_ThrowOutOfMemory(context);
    }
}
} // namespace

void RegisterSceneModelDataBindings(WPSceneScriptHost::Opaque& opaque) {
    auto* context = opaque.runtime.context;
    JS_NewClassID(opaque.runtime.runtime, &opaque.model_data_class);
    const JSClassDef definition { .class_name = "IModelData", .finalizer = FinalizeModelHandle };
    JS_NewClass(opaque.runtime.runtime, opaque.model_data_class, &definition);
    ModelJSValue prototype(context, JS_NewObject(context));
    JS_SetPropertyStr(context, prototype.Get(), "toConfigString",
                       JS_NewCFunction(context, ModelToConfigString, "toConfigString", 0));
    JS_SetPropertyStr(context, prototype.Get(), "applyData",
                       JS_NewCFunction(context, NativeApplyModelData, "applyData", 1));
    JS_SetPropertyStr(context, prototype.Get(), "replaceData",
                       JS_NewCFunction(context, NativeReplaceModelData, "replaceData", 1));
    ModelJSValue constructor(context,
                             JS_NewCFunction2(context, ModelConstructor, "IModelData", 0,
                                              JS_CFUNC_constructor, 0));
    JS_SetConstructor(context, constructor.Get(), prototype.Get());
    JS_SetClassProto(context, opaque.model_data_class, JS_DupValue(context, prototype.Get()));
    for (const auto& attribute : ModelVertexAttributes()) {
        JS_DefinePropertyValueStr(context, constructor.Get(), attribute.constant,
                                   JS_NewString(context, attribute.name), JS_PROP_ENUMERABLE);
    }
    ModelJSValue global(context, JS_GetGlobalObject(context));
    JS_SetPropertyStr(context, global.Get(), "IModelData", constructor.Release());
    JS_SetPropertyStr(context, opaque.native_bridge, "createModelData",
                       JS_NewCFunction(context, NativeCreateModelData, "createModelData", 1));
    JS_SetPropertyStr(context, opaque.native_bridge, "destroyModelData",
                       JS_NewCFunction(context, NativeDestroyModelData, "destroyModelData", 1));
}
} // namespace wallpaper

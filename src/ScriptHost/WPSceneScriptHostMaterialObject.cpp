#include "WPSceneScriptHostShared.hpp"

#include <memory>
#include <utility>

namespace wallpaper
{
namespace
{

class MaterialJSValue {
public:
    MaterialJSValue(JSContext* context, JSValue value): m_context(context), m_value(value) {}
    ~MaterialJSValue() { JS_FreeValue(m_context, m_value); }
    MaterialJSValue(const MaterialJSValue&) = delete;
    MaterialJSValue& operator=(const MaterialJSValue&) = delete;
    JSValueConst Get() const { return m_value; }
    JSValue Release() { return std::exchange(m_value, JS_UNDEFINED); }

private:
    JSContext* m_context;
    JSValue m_value;
};

void FreeMaterialDescriptor(JSRuntime* runtime, const JSPropertyDescriptor& descriptor) {
    JS_FreeValueRT(runtime, descriptor.value);
    JS_FreeValueRT(runtime, descriptor.getter);
    JS_FreeValueRT(runtime, descriptor.setter);
}

struct MaterialDescriptor {
    explicit MaterialDescriptor(JSContext* context): runtime(JS_GetRuntime(context)) {}
    ~MaterialDescriptor() { FreeMaterialDescriptor(runtime, value); }
    JSPropertyDescriptor Release() {
        return std::exchange(value, JSPropertyDescriptor { 0, JS_UNDEFINED, JS_UNDEFINED, JS_UNDEFINED });
    }
    JSRuntime* runtime;
    JSPropertyDescriptor value { 0, JS_UNDEFINED, JS_UNDEFINED, JS_UNDEFINED };
};

struct MaterialProperty {
    JSPropertyDescriptor descriptor { 0, JS_UNDEFINED, JS_UNDEFINED, JS_UNDEFINED };
    bool native { false };
};

struct MaterialObject {
    explicit MaterialObject(JSRuntime* runtime): runtime(runtime) {}
    ~MaterialObject() {
        for (const auto& [name, property] : properties) {
            JS_FreeAtomRT(runtime, name);
            FreeMaterialDescriptor(runtime, property.descriptor);
        }
        JS_FreeValueRT(runtime, names);
        JS_FreeValueRT(runtime, read);
        JS_FreeValueRT(runtime, write);
    }

    JSRuntime* runtime;
    // This ordinary object owns only the key order, including integer indices and symbols.
    // Property attributes and callback membership belong to the separate descriptor map:
    // sealing a material must not prevent an existing descriptor from keeping a live value.
    JSValue names { JS_UNDEFINED };
    JSValue read { JS_UNDEFINED };
    JSValue write { JS_UNDEFINED };
    std::unordered_map<JSAtom, MaterialProperty> properties;
};

MaterialObject* GetMaterialObject(JSContext* context, JSValueConst object) {
    const auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
    return static_cast<MaterialObject*>(JS_GetOpaque(object, opaque->material_object_class));
}

void FinalizeMaterialObject(JSRuntime*, JSValue object) {
    delete static_cast<MaterialObject*>(JS_GetOpaque(object, JS_GetClassID(object)));
}

void MarkMaterialObject(JSRuntime* runtime, JSValueConst object, JS_MarkFunc* mark) {
    const auto* material = static_cast<MaterialObject*>(JS_GetOpaque(object, JS_GetClassID(object)));
    if (material == nullptr) return;
    JS_MarkValue(runtime, material->names, mark);
    JS_MarkValue(runtime, material->read, mark);
    JS_MarkValue(runtime, material->write, mark);
    for (const auto& [name, property] : material->properties) {
        JS_MarkValue(runtime, property.descriptor.value, mark);
        JS_MarkValue(runtime, property.descriptor.getter, mark);
        JS_MarkValue(runtime, property.descriptor.setter, mark);
    }
}

int ReadMaterialOwnProperty(JSContext* context, JSPropertyDescriptor* descriptor,
                            JSValueConst object, JSAtom name) {
    auto* material = GetMaterialObject(context, object);
    const auto found = material->properties.find(name);
    if (found == material->properties.end()) return 0;
    if (descriptor == nullptr) return 1;

    const auto& property = found->second;
    *descriptor = { property.descriptor.flags, JS_UNDEFINED,
                    JS_DupValue(context, property.descriptor.getter),
                    JS_DupValue(context, property.descriptor.setter) };
    if (property.native) {
        MaterialJSValue key(context, JS_AtomToValue(context, name));
        JSValueConst argument = key.Get();
        descriptor->value = JS_Call(context, material->read, JS_UNDEFINED, 1, &argument);
    } else {
        descriptor->value = JS_DupValue(context, property.descriptor.value);
    }
    if (! JS_IsException(descriptor->value)) return 1;
    FreeMaterialDescriptor(JS_GetRuntime(context), *descriptor);
    *descriptor = { 0, JS_UNDEFINED, JS_UNDEFINED, JS_UNDEFINED };
    return -1;
}

int ReadMaterialOwnNames(JSContext* context, JSPropertyEnum** names, uint32_t* count,
                         JSValueConst object) {
    return JS_GetOwnPropertyNames(context, names, count, GetMaterialObject(context, object)->names,
                                 JS_GPN_STRING_MASK | JS_GPN_SYMBOL_MASK);
}

int DeleteMaterialOwnProperty(JSContext* context, JSValueConst object, JSAtom name) {
    auto* material = GetMaterialObject(context, object);
    const auto found = material->properties.find(name);
    if (found == material->properties.end()) return 1;
    if (!(found->second.descriptor.flags & JS_PROP_CONFIGURABLE)) return 0;
    if (JS_DeleteProperty(context, material->names, name, 0) < 0) return -1;
    FreeMaterialDescriptor(material->runtime, found->second.descriptor);
    material->properties.erase(found);
    JS_FreeAtom(context, name);
    return 1;
}

int WriteMaterialNativeProperty(JSContext* context, MaterialObject& material,
                                JSAtom name, JSValueConst value) {
    MaterialJSValue key(context, JS_AtomToValue(context, name));
    JSValueConst arguments[] { key.Get(), value };
    MaterialJSValue result(context, JS_Call(context, material.write, JS_UNDEFINED, 2, arguments));
    return JS_IsException(result.Get()) ? -1 : JS_ToBool(context, result.Get());
}

int DefineMaterialOwnProperty(JSContext* context, JSValueConst object, JSAtom name,
                              JSValueConst value, JSValueConst getter, JSValueConst setter,
                              int flags) {
    try {
        auto* material = GetMaterialObject(context, object);
        auto found = material->properties.find(name);
        const bool exists = found != material->properties.end();
        const bool native = exists && found->second.native;
        constexpr int descriptor_fields = JS_PROP_HAS_VALUE | JS_PROP_HAS_WRITABLE |
            JS_PROP_HAS_CONFIGURABLE | JS_PROP_HAS_ENUMERABLE | JS_PROP_HAS_GET | JS_PROP_HAS_SET;
        if (native && (found->second.descriptor.flags & JS_PROP_WRITABLE) &&
            (flags & descriptor_fields) == JS_PROP_HAS_VALUE) {
            // Ordinary uniform writes do not change the descriptor. They can go straight
            // to the retained callback; other receivers are handled by the VM's ordinary
            // data-property assignment and therefore get their own property instead.
            return WriteMaterialNativeProperty(context, *material, name, value);
        }

        MaterialDescriptor before(context);
        if (ReadMaterialOwnProperty(context, &before.value, object, name) < 0) return -1;

        // Validate changes against a fresh descriptor snapshot using the VM's ordinary
        // property rules. A native readonly value can change through another material
        // handle, so a permanent frozen data slot would impose the wrong value invariant.
        // The temporary object also validates accessor replacement and missing attributes
        // without duplicating the language's descriptor conversion and rejection rules.
        MaterialJSValue prospective(context, JS_NewObjectProto(context, JS_NULL));
        if (JS_IsException(prospective.Get())) return -1;
        if (exists) {
            const auto& current = before.value;
            int initial_flags = (current.flags & JS_PROP_C_W_E) |
                JS_PROP_HAS_CONFIGURABLE | JS_PROP_HAS_ENUMERABLE;
            initial_flags |= (current.flags & JS_PROP_GETSET)
                ? JS_PROP_HAS_GET | JS_PROP_HAS_SET : JS_PROP_HAS_VALUE | JS_PROP_HAS_WRITABLE;
            if (JS_DefineProperty(context, prospective.Get(), name, current.value,
                                  current.getter, current.setter, initial_flags) < 0) return -1;
        } else if (JS_IsExtensible(context, object) == 0) {
            if (JS_PreventExtensions(context, prospective.Get()) < 0) return -1;
        }
        const int defined = JS_DefineProperty(context, prospective.Get(), name, value,
                                              getter, setter, flags);
        if (defined <= 0) return defined;
        MaterialDescriptor after(context);
        if (JS_GetOwnProperty(context, &after.value, prospective.Get(), name) < 0) return -1;
        const bool keep_native = native && !(after.value.flags & JS_PROP_GETSET);
        if (keep_native && (before.value.flags & (JS_PROP_CONFIGURABLE | JS_PROP_WRITABLE))) {
            const int written = WriteMaterialNativeProperty(context, *material, name, after.value.value);
            if (written <= 0) return written;
        }

        // Only a successful descriptor change can detach a native callback. Deletion and
        // JavaScript accessor replacement remove that binding permanently for this object;
        // a later data definition is an ordinary script property, not a uniform alias.
        if (! exists) {
            found = material->properties.try_emplace(name).first;
            JS_DupAtom(context, name);
            if (JS_SetProperty(context, material->names, name, JS_TRUE) < 0) {
                material->properties.erase(found);
                JS_FreeAtom(context, name);
                return -1;
            }
        } else {
            found = material->properties.find(name);
            FreeMaterialDescriptor(material->runtime, found->second.descriptor);
        }
        if (keep_native) {
            JS_FreeValue(context, after.value.value);
            after.value.value = JS_UNDEFINED;
        }
        found->second = { after.Release(), keep_native };
        return 1;
    } catch (const std::bad_alloc&) {
        JS_ThrowOutOfMemory(context);
        return -1;
    }
}

bool IsMaterialFrozen(JSContext* context, JSValueConst object, const MaterialObject& material) {
    if (JS_IsExtensible(context, object)) return false;
    for (const auto& [name, property] : material.properties) {
        if (property.descriptor.flags & JS_PROP_CONFIGURABLE) return false;
        if (! property.native && (property.descriptor.flags & JS_PROP_WRITABLE)) return false;
    }
    return true;
}

enum class MaterialIntegrity { Seal, Freeze, IsFrozen };

JSValue MaterialIntegrityOperation(JSContext* context, JSValueConst receiver, int argc,
                                   JSValueConst* arguments, int operation, JSValue* original) {
    auto* material = argc > 0 ? GetMaterialObject(context, arguments[0]) : nullptr;
    if (material == nullptr) return JS_Call(context, original[0], receiver, argc, arguments);
    const auto kind = static_cast<MaterialIntegrity>(operation);
    const bool frozen = IsMaterialFrozen(context, arguments[0], *material);
    if (kind == MaterialIntegrity::IsFrozen) return JS_NewBool(context, frozen);
    if (kind == MaterialIntegrity::Freeze && frozen) return JS_DupValue(context, arguments[0]);

    // Material callbacks are exposed as data-shaped descriptors but remain callbacks for
    // integrity queries. Sealing a callback-only object can already satisfy isFrozen while
    // its setters remain enabled. A subsequent freeze preserves that established state;
    // a first freeze makes data/native properties readonly without invoking their setters.
    // Getters stay live in either state. Ordinary objects use their original builtins above.
    if (JS_PreventExtensions(context, arguments[0]) < 0) return JS_EXCEPTION;
    for (auto& [name, property] : material->properties) {
        property.descriptor.flags &= ~JS_PROP_CONFIGURABLE;
        if (kind == MaterialIntegrity::Freeze) property.descriptor.flags &= ~JS_PROP_WRITABLE;
    }
    return JS_DupValue(context, arguments[0]);
}

JSValue CreateMaterialObject(JSContext* context, JSValueConst, int, JSValueConst* arguments) {
    try {
        const auto* opaque = static_cast<WPSceneScriptHost::Opaque*>(JS_GetContextOpaque(context));
        MaterialJSValue object(context, JS_NewObjectClass(context, opaque->material_object_class));
        if (JS_IsException(object.Get())) return JS_EXCEPTION;
        auto material = std::make_unique<MaterialObject>(JS_GetRuntime(context));
        material->names = JS_NewObjectProto(context, JS_NULL);
        if (JS_IsException(material->names)) return JS_EXCEPTION;
        material->read = JS_DupValue(context, arguments[1]);
        material->write = JS_DupValue(context, arguments[2]);
        auto* state = material.get();
        JS_SetOpaque(object.Get(), material.release());
        MaterialJSValue length(context, JS_GetPropertyStr(context, arguments[0], "length"));
        uint32_t count = 0;
        if (JS_ToUint32(context, &count, length.Get()) < 0) return JS_EXCEPTION;
        for (uint32_t index = 0; index < count; ++index) {
            MaterialJSValue key(context, JS_GetPropertyUint32(context, arguments[0], index));
            const JSAtom name = JS_ValueToAtom(context, key.Get());
            if (name == JS_ATOM_NULL) return JS_EXCEPTION;
            const auto [found, added] = state->properties.try_emplace(name);
            if (added) {
                found->second.native = true;
                found->second.descriptor.flags = JS_PROP_C_W_E;
                if (JS_SetProperty(context, state->names, name, JS_TRUE) < 0) return JS_EXCEPTION;
            } else {
                JS_FreeAtom(context, name);
            }
        }
        return object.Release();
    } catch (const std::bad_alloc&) {
        return JS_ThrowOutOfMemory(context);
    }
}

JSClassExoticMethods material_object_methods {
    .get_own_property = ReadMaterialOwnProperty,
    .get_own_property_names = ReadMaterialOwnNames,
    .delete_property = DeleteMaterialOwnProperty,
    .define_own_property = DefineMaterialOwnProperty,
};

} // namespace

void RegisterSceneMaterialObjectBindings(WPSceneScriptHost::Opaque& opaque) {
    auto* context = opaque.runtime.context;
    JS_NewClassID(opaque.runtime.runtime, &opaque.material_object_class);
    const JSClassDef definition { .class_name = "Object", .finalizer = FinalizeMaterialObject,
                                 .gc_mark = MarkMaterialObject, .exotic = &material_object_methods };
    JS_NewClass(opaque.runtime.runtime, opaque.material_object_class, &definition);
    MaterialJSValue global(context, JS_GetGlobalObject(context));
    MaterialJSValue object(context, JS_GetPropertyStr(context, global.Get(), "Object"));
    JS_SetClassProto(context, opaque.material_object_class,
                     JS_GetPropertyStr(context, object.Get(), "prototype"));
    for (const auto& [name, operation] : {
             std::pair { "seal", MaterialIntegrity::Seal },
             std::pair { "freeze", MaterialIntegrity::Freeze },
             std::pair { "isFrozen", MaterialIntegrity::IsFrozen } }) {
        MaterialJSValue original(context, JS_GetPropertyStr(context, object.Get(), name));
        JSValue saved = original.Get();
        JSValue method = JS_NewCFunctionData(context, MaterialIntegrityOperation, 1,
                                             static_cast<int>(operation), 1, &saved);
        JS_DefinePropertyValueStr(context, method, "name", JS_NewString(context, name), JS_PROP_CONFIGURABLE);
        JS_SetPropertyStr(context, object.Get(), name, method);
    }
    JS_SetPropertyStr(context, opaque.native_bridge, "createMaterialObject",
                       JS_NewCFunction(context, CreateMaterialObject, "createMaterialObject", 3));
}

} // namespace wallpaper

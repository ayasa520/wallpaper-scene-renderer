#include "WPSceneParser.hpp"
#include "WPSceneParserShared.hpp"
#include "WPJson.hpp"
#include "WPUserProperties.hpp"

#include "Utils/String.h"
#include "Utils/Logging.h"
#include "Utils/Algorism.h"
#include "Utils/Eigen.h"
#include "Core/Visitors.hpp"
#include "Core/StringHelper.hpp"
#include "Core/ArrayHelper.hpp"
#include "SpecTexs.hpp"
#include "VulkanRender/Msaa.hpp"
#include "Scene/ShadowAtlas.hpp"
#include "Scene/LightingV1.hpp"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneTexture.h"

#include "WPShaderParser.hpp"
#include "WPTexImageParser.hpp"
#include "WPSyntheticImageParser.hpp"
#include "WPParticleParser.hpp"
#include "WPSoundParser.hpp"
#include "WPMdlParser.hpp"
#include "WPPropertyAnimation.hpp"
#include "WPSceneScriptMedia.hpp"
#include "WPTextLayer.hpp"
#include "WPUserSetting.hpp"
#include "WPImageAlignment.hpp"

#include "Particle/ParticleRenderPlan.h"
#include "Particle/WPParticleRawGener.h"
#include "Particle/ParticleSystem.h"

#include "WPShaderValueUpdater.hpp"
#include "wpscene/WPImageObject.h"
#include "wpscene/WPParallaxDepth.hpp"
#include "wpscene/WPParticleObject.h"
#include "wpscene/WPSoundObject.h"
#include "wpscene/WPLightObject.hpp"
#include "wpscene/WPScene.h"

#include "Fs/VFS.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <random>
#include <cmath>
#include <functional>
#include <regex>
#include <optional>
#include <type_traits>
#include <variant>
#include <limits>
#include <cstring>
#include <span>
#include <Eigen/Dense>

using namespace wallpaper;
using namespace Eigen;

// Cameras remain per authored layer because they carry that layer's transform. Render targets do
// not: destination targets intern globally by their generated string, and effect FBOs intern by
// authored name. Keeping target identity out of EffectCameraName is what lets repeated language
// branches share the same backing images while every object remains fully materialized.
std::string EffectCameraName(int32_t layer_id) {
    return "__hanabi_effect_camera_" + std::to_string(layer_id);
}

// A layer destination target is never created smaller than 4x4 pixels; the same clamped extent
// is both the interned name and the backing image size.
constexpr int32_t kMinDestinationRenderTargetExtent = 4;

int32_t ClampDestinationRenderTargetExtent(int32_t extent) {
    return std::max(kMinDestinationRenderTargetExtent, extent);
}

std::string SceneDestinationRenderTargetBaseName(int32_t width, int32_t height, char suffix) {
    // Destination render targets are interned by name, and the name is only the pixel size plus
    // the suffix. Every destination target is a fixed-size image: a layer whose source is the
    // output framebuffer takes that framebuffer's pixel size at load, so two layers that resolve
    // to the same pixel size legitimately share one backing image. There is no separate
    // screen-following identity.
    return "sc." + std::to_string(ClampDestinationRenderTargetExtent(width)) + "." +
           std::to_string(ClampDestinationRenderTargetExtent(height)) + "." + suffix;
}

std::array<std::string, 2> SceneDestinationRenderTargetNames(
    const Scene& scene, int32_t parent_id, int32_t width, int32_t height) {
    std::array<std::string, 2> names {
        SceneDestinationRenderTargetBaseName(width, height, 'b'),
        SceneDestinationRenderTargetBaseName(width, height, 'n'),
    };
    std::array<uint32_t, 2> ancestor_collisions {};
    std::unordered_set<int32_t> visited;

    // Destination names start as `sc.W.H.b|n`. Walk only the current object's parent chain and
    // append the number of ancestors whose destination-slot name exactly equals that base. Only
    // ancestors flagged passthrough take part in the comparison; every other ancestor is walked
    // through without contributing. Siblings therefore intern the same named RT (the multilingual
    // case), while a nested same-sized effect under a passthrough parent does not alias the target
    // that parent still uses. Compare exact names rather than inventing a layer id or a global
    // occurrence counter.
    while (parent_id != 0 && visited.insert(parent_id).second) {
        const auto* parent = scene.FindSceneObject(parent_id);
        if (parent != nullptr && parent->Passthrough()) {
            if (const auto* effect_layer = scene.FindImageEffectLayer(parent_id)) {
                if (effect_layer->FirstTarget() == names[0]) {
                    ancestor_collisions[0]++;
                    ancestor_collisions[1]++;
                }
            }
        }

        parent_id = parent != nullptr ? parent->ParentId() : 0;
    }

    for (size_t index = 0; index < names.size(); index++) {
        if (ancestor_collisions[index] != 0) {
            names[index] += std::to_string(ancestor_collisions[index]);
        }
    }
    return names;
}

// A fullscreen layer samples the output framebuffer, so its effect targets take that framebuffer's
// pixel size. The renderer hands the parser its live output extent at load; a scene parsed without
// a live output (tests, tooling) falls back to the authored canvas.
std::array<float, 2> OutputFramebufferEffectTargetSize(const ParseContext& context) {
    if (context.scene != nullptr) {
        const auto& extent = context.scene->physicalOutputExtent;
        if (extent[0] > 0u && extent[1] > 0u) {
            return { static_cast<float>(extent[0]), static_cast<float>(extent[1]) };
        }
    }
    return { static_cast<float>(std::max(1, context.ortho_w)),
             static_cast<float>(std::max(1, context.ortho_h)) };
}

std::string EffectFboRenderTargetName(const wpscene::WPEffectFbo& fbo, int32_t effect_id) {
    if (! fbo.unique) return fbo.name;
    return fbo.name + "_" + std::to_string(effect_id);
}

const SceneRenderTarget& InternNamedRenderTarget(Scene& scene, const std::string& name,
                                                  SceneRenderTarget target) {
    // Named-RT lookup keys only the string. A hit returns the existing resource without applying
    // the later caller's dimensions. `try_emplace` makes that first-registration rule explicit
    // and prevents a later hidden language branch from silently replacing the descriptor shared
    // by an earlier branch.
    const auto [it, inserted] = scene.renderTargets.try_emplace(name, target);
    if (! inserted &&
        (it->second.width != target.width || it->second.height != target.height ||
         it->second.ContentWidth() != target.ContentWidth() ||
         it->second.ContentHeight() != target.ContentHeight())) {
        LOG_INFO("SceneNamedRenderTargetIntern: name='%s' first-size=%dx%d first-map=%dx%d "
                 "ignored-size=%dx%d ignored-map=%dx%d",
                 name.c_str(),
                 it->second.width,
                 it->second.height,
                 it->second.ContentWidth(),
                 it->second.ContentHeight(),
                 target.width,
                 target.height,
                 target.ContentWidth(),
                 target.ContentHeight());
    }
    return it->second;
}

uint32_t HashParticleFrameU32(uint32_t seed, uint32_t bits) {
    seed ^= bits + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
    return seed;
}

uint32_t HashParticleFrameFloat(uint32_t seed, float value) {
    uint32_t bits { 0 };
    std::memcpy(&bits, &value, sizeof(bits));
    return HashParticleFrameU32(seed, bits);
}

float RandomParticleFrameLifetime(const Particle& p, float sprite_frame_count_value) {
    const auto sprite_frame_count =
        static_cast<uint32_t>(std::max(1.0f, std::round(sprite_frame_count_value)));
    if (sprite_frame_count <= 1u) return 0.0f;

    // The shader only receives one float named "lifetime", so randomframe encodes a stable
    // atlas cell into that float. Hash spawn-time identity rather than live velocity, which
    // changes every operator tick.
    uint32_t seed = 2166136261u;
    seed          = HashParticleFrameFloat(seed, p.init.lifetime);
    seed          = HashParticleFrameFloat(seed, p.init.size);
    seed          = HashParticleFrameFloat(seed, p.init.color.x());
    seed          = HashParticleFrameFloat(seed, p.init.color.y());
    seed          = HashParticleFrameFloat(seed, p.init.color.z());
    seed          = HashParticleFrameU32(seed, static_cast<uint32_t>(p.spawnSequence));
    seed          = HashParticleFrameU32(seed, static_cast<uint32_t>(p.spawnSequence >> 32));

    const uint32_t frame = seed % sprite_frame_count;
    return (static_cast<float>(frame) + 0.5f) / static_cast<float>(sprite_frame_count);
}

std::string DescribeIndexVec(const std::vector<usize>& values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); i++) {
        if (i != 0) oss << ", ";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::optional<std::string_view> TextureFormatShaderDefine(TextureFormat format) {
    // Wallpaper Engine's stock shader headers use TEXnFORMAT preprocessor symbols to choose the
    // correct channel layout for compressed normal maps, R/RG masks, and light-gradient textures.
    // The old path only populated TEX0FORMAT, which left later slots such as normal maps
    // (TEX1FORMAT), PBR gradients (TEX4FORMAT), and fur/detail masks (TEX8FORMAT) to compile as
    // the implicit RGBA default under DXC. Keep the policy here at the material/texture binding
    // boundary: the shader remains authored WE code, while the parser supplies the real texture
    // metadata for every bound slot.
    switch (format) {
    case TextureFormat::RGBA8: return "FORMAT_RGBA8888";
    case TextureFormat::RGB8: return "FORMAT_RGB888";
    case TextureFormat::BC1: return "FORMAT_DXT1";
    case TextureFormat::BC2: return "FORMAT_DXT3";
    case TextureFormat::BC3: return "FORMAT_DXT5";
    case TextureFormat::RG8: return "FORMAT_RG88";
    case TextureFormat::R8: return "FORMAT_R8";
    }
    return std::nullopt;
}

void SetTextureFormatShaderDefine(WPShaderInfo& shader_info, usize slot, TextureFormat format) {
    const auto define = TextureFormatShaderDefine(format);
    if (! define.has_value()) return;
    shader_info.combos["TEX" + std::to_string(slot) + "FORMAT"] = std::string(*define);
}

// Text-layer bindings span authored JSON, runtime state, and script registration data. Keeping
// this predicate centralized avoids duplicating the identification rule across parser entry points.
bool IsTextLayerObjectJson(const nlohmann::json& object_json) {
    return object_json.contains("text") && ! object_json.at("text").is_null();
}

const char* DynamicValueTypeName(WPDynamicValue::Type hint) {
    switch (hint) {
    case WPDynamicValue::Type::Null: return "null";
    case WPDynamicValue::Type::Boolean: return "bool";
    case WPDynamicValue::Type::Int32: return "int32";
    case WPDynamicValue::Type::UInt32: return "uint32";
    case WPDynamicValue::Type::Float: return "float";
    case WPDynamicValue::Type::Double: return "double";
    case WPDynamicValue::Type::String: return "string";
    case WPDynamicValue::Type::FloatVector: return "floatVector";
    case WPDynamicValue::Type::Int3: return "int3";
    case WPDynamicValue::Type::Float2: return "float2";
    case WPDynamicValue::Type::Float3: return "float3";
    case WPDynamicValue::Type::Float4: return "float4";
    }
    return "unknown";
}

bool IsParserOpacityUniformName(std::string_view uniform_name) {
    return uniform_name == "alpha" || uniform_name == "g_Alpha" ||
           uniform_name == "g_UserAlpha";
}

float ClampParserOpacityScalar(float opacity) {
    if (! std::isfinite(opacity)) return 0.0f;
    return std::clamp(opacity, 0.0f, 1.0f);
}

std::array<i32, 4> ResolvePaddedSpriteSheetResolution(const ImageHeader& texh,
                                                       const SpriteFrame& frame) {
    const auto physical_width  = texh.width > 0 ? texh.width : texh.mapWidth;
    const auto physical_height = texh.height > 0 ? texh.height : texh.mapHeight;
    auto       content_width   = texh.mapWidth > 0 ? texh.mapWidth : physical_width;
    auto       content_height  = texh.mapHeight > 0 ? texh.mapHeight : physical_height;

    const auto frame_width = static_cast<i32>(std::lround(frame.width));
    if (frame_width > 0) content_width -= content_width % frame_width;
    const auto frame_height = static_cast<i32>(std::lround(frame.height));
    if (frame_height > 0) content_height -= content_height % frame_height;

    return { physical_width, physical_height, content_width, content_height };
}

ShaderValue ClampParserOpacityUniformValue(std::string_view uniform_name,
                                           const ShaderValue& value) {
    if (! IsParserOpacityUniformName(uniform_name) || value.size() == 0) return value;

    // Parser-time material constants can be consumed before the script host advances animations.
    // Clamp only normalized opacity uniforms here so cold-start alpha writes match the runtime
    // registration boundary without flattening intentionally overshooting non-opacity curves.
    ShaderValue clamped = value;
    clamped[0]          = ClampParserOpacityScalar(clamped[0]);
    return clamped;
}

void LogTextLayerRegistration(const char* event_name, int32_t object_id,
                              const std::string& object_name, std::string_view property_name,
                              WPDynamicValue::Type hint, const WPUserSetting& setting,
                              const std::optional<WPDynamicValue>& base_value) {}

bool IsZeroParallaxDepth(const std::array<float, 2>& depth) {
    return std::abs(depth[0]) <= 1e-6f && std::abs(depth[1]) <= 1e-6f;
}

std::string NormalizeParallaxPeerName(std::string_view name) {
    std::string normalized;
    normalized.reserve(name.size());
    for (unsigned char ch : name) {
        if (std::isalnum(ch)) normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

std::array<float, 2> ImageObjectParallaxDepth(const wpscene::WPImageObject& object) {
    return { object.parallaxDepth[0], object.parallaxDepth[1] };
}

std::array<float, 2> TextObjectParallaxDepth(const wpscene::WPTextObject& object) {
    return { object.parallaxDepth[0], object.parallaxDepth[1] };
}

bool LayerUsesRoutedParent(int32_t parent_id, std::string_view attachment) {
    return parent_id != 0 && attachment.empty();
}

struct WPEmptyObject {
    int32_t              id { 0 };
    std::string          name;
    std::array<float, 3> origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> angles { 0.0f, 0.0f, 0.0f };
    std::array<float, 2> parallaxDepth { wpscene::kDefaultParallaxDepth };
    bool                 parallaxDepthAuthored { false };
    bool                 visible { true };
    VisibleBinding       visible_binding;
    int32_t              parent { 0 };
    std::string          attachment;
    bool                 is_camera_layer { false };
    std::string          camera_name;
    std::string          camera_path;
    float                fov { 50.0f };
    float                zoom { 1.0f };

    bool FromJson(const nlohmann::json& json, fs::VFS&) {
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE_NOWARN(json, "origin", origin);
        GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
        GET_JSON_NAME_VALUE_NOWARN(json, "angles", angles);
        parallaxDepthAuthored =
            json.contains("parallaxDepth") && ! json.at("parallaxDepth").is_null();
        GET_JSON_NAME_VALUE_NOWARN(json, "parallaxDepth", parallaxDepth);
        GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
        if (json.contains("visible")) {
            GET_JSON_NAME_VALUE_NOWARN(json.at("visible"), "value", visible_binding.value);
            if (json.at("visible").contains("user") && ! json.at("visible").at("user").is_null()) {
                const auto& user = json.at("visible").at("user");
                if (user.is_string()) {
                    GET_JSON_VALUE(user, visible_binding.user.name);
                } else if (user.is_object()) {
                    GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                    GET_JSON_NAME_VALUE_NOWARN(user, "condition", visible_binding.user.condition);
                }
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
        GET_JSON_NAME_VALUE_NOWARN(json, "attachment", attachment);
        if (json.contains("camera") && json.at("camera").is_string()) {
            // Camera assets do not have an image/particle/text discriminator, so they otherwise
            // arrive here as empty layers. Preserve their camera marker and authored projection
            // values so ParseEmptyObj can register a runtime camera target instead of a harmless
            // transform-only placeholder.
            GET_JSON_NAME_VALUE_NOWARN(json, "camera", camera_name);
            is_camera_layer = true;
        }
        if (json.contains("path") && json.at("path").is_string()) {
            GET_JSON_NAME_VALUE_NOWARN(json, "path", camera_path);
            if (camera_path.find("scripts/camera_paths_") == 0) {
                is_camera_layer = true;
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "fov", fov);
        GET_JSON_NAME_VALUE_NOWARN(json, "zoom", zoom);
        return true;
    }
};

struct WPShapeObject {
    int32_t                             id { 0 };
    std::string                         name;
    std::string                         shape;
    std::array<float, 3>                origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3>                scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3>                angles { 0.0f, 0.0f, 0.0f };
    std::array<float, 2>                size { 0.0f, 0.0f };
    std::array<float, 2>                parallaxDepth { wpscene::kDefaultParallaxDepth };
    bool                                parallaxDepthAuthored { false };
    std::array<float, 3>                color { 1.0f, 1.0f, 1.0f };
    float                               alpha { 1.0f };
    float                               brightness { 1.0f };
    bool                                visible { true };
    bool                                has_size { false };
    VisibleBinding                      visible_binding;
    int32_t                             parent { 0 };
    std::string                         attachment;
    std::vector<wpscene::WPImageEffect> effects;

    bool FromJson(const nlohmann::json& json, fs::VFS& vfs) {
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE_NOWARN(json, "shape", shape);
        GET_JSON_NAME_VALUE_NOWARN(json, "origin", origin);
        GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
        GET_JSON_NAME_VALUE_NOWARN(json, "angles", angles);
        if (json.contains("size") && ! json.at("size").is_null()) {
            GET_JSON_NAME_VALUE_NOWARN(json, "size", size);
            has_size = size[0] > 0.0f && size[1] > 0.0f;
        }
        parallaxDepthAuthored =
            json.contains("parallaxDepth") && ! json.at("parallaxDepth").is_null();
        GET_JSON_NAME_VALUE_NOWARN(json, "parallaxDepth", parallaxDepth);
        GET_JSON_NAME_VALUE_NOWARN(json, "color", color);
        GET_JSON_NAME_VALUE_NOWARN(json, "alpha", alpha);
        GET_JSON_NAME_VALUE_NOWARN(json, "brightness", brightness);

        if (json.contains("visible") && ! json.at("visible").is_null()) {
            const auto& visible_json = json.at("visible");
            if (visible_json.is_boolean()) {
                GET_JSON_VALUE_NOWARN(visible_json, visible);
                visible_binding.value = visible;
            } else if (visible_json.is_object()) {
                // Shape layers can be driven by the same user/script visibility contracts as image
                // layers. Preserve the authored fallback and binding instead of treating the whole
                // object as a boolean, otherwise direct-draw effects would materialize with the
                // wrong initial visibility and runtime toggles would have no stable target.
                GET_JSON_NAME_VALUE_NOWARN(visible_json, "value", visible_binding.value);
                visible = visible_binding.value;
                if (visible_json.contains("user") && ! visible_json.at("user").is_null()) {
                    const auto& user = visible_json.at("user");
                    if (user.is_string()) {
                        GET_JSON_VALUE(user, visible_binding.user.name);
                    } else if (user.is_object()) {
                        GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                        GET_JSON_NAME_VALUE_NOWARN(
                            user, "condition", visible_binding.user.condition);
                    }
                }
            }
        }

        GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
        GET_JSON_NAME_VALUE_NOWARN(json, "attachment", attachment);

        if (json.contains("effects") && json.at("effects").is_array()) {
            for (const auto& effect_json : json.at("effects")) {
                wpscene::WPImageEffect effect;
                if (effect.FromJson(effect_json, vfs)) {
                    effects.push_back(std::move(effect));
                } else {
                    LOG_ERROR(
                        "ShapeObject: effect parse failed, layer=%d name='%s'", id, name.c_str());
                }
            }
        }
        return true;
    }
};

// WPModelObject's definition lives in the shared header so both parser units see
// the same type; the JSON loader stays here beside the other FromJson loaders.
bool WPModelObject::FromJson(const nlohmann::json& json, fs::VFS&) {
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE_NOWARN(json, "origin", origin);
        GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
        GET_JSON_NAME_VALUE_NOWARN(json, "angles", angles);
        GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
        if (json.contains("visible") && json.at("visible").is_object()) {
            GET_JSON_NAME_VALUE_NOWARN(json.at("visible"), "value", visible_binding.value);
            if (json.at("visible").contains("user") && ! json.at("visible").at("user").is_null()) {
                const auto& user = json.at("visible").at("user");
                if (user.is_string()) {
                    GET_JSON_VALUE(user, visible_binding.user.name);
                } else if (user.is_object()) {
                    GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                    GET_JSON_NAME_VALUE_NOWARN(user, "condition", visible_binding.user.condition);
                }
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
        GET_JSON_NAME_VALUE_NOWARN(json, "attachment", attachment);
        GET_JSON_NAME_VALUE_NOWARN(json, "model", model);
        GET_JSON_NAME_VALUE_NOWARN(json, "skin", skin);
        GET_JSON_NAME_VALUE_NOWARN(json, "reflected", reflected);
        GET_JSON_NAME_VALUE_NOWARN(json, "castshadow", castshadow);
        if (json.contains("animationlayers") && json.at("animationlayers").is_array()) {
            for (const auto& animation_json : json.at("animationlayers")) {
                WPPuppetLayer::AnimationLayer layer;
                GET_JSON_NAME_VALUE(animation_json, "animation", layer.id);
                GET_JSON_NAME_VALUE(animation_json, "blend", layer.blend);
                GET_JSON_NAME_VALUE(animation_json, "rate", layer.rate);
                GET_JSON_NAME_VALUE_NOWARN(animation_json, "additive", layer.additive);
                GET_JSON_NAME_VALUE_NOWARN(animation_json, "visible", layer.visible);
                animation_layers.push_back(layer);
            }
        }
        return ! model.empty();
}


using WPObjectVar =
    std::variant<wpscene::WPImageObject, wpscene::WPParticleObject, wpscene::WPSoundObject,
                 wpscene::WPLightObject, wpscene::WPTextObject, WPModelObject, WPShapeObject,
                 WPEmptyObject>;


void ApplyNodeOwnerParallaxFallback(ParseContext& context, int32_t owner_id,
                                    const std::array<float, 2>& depth, SceneNode* anchor,
                                    bool suppress_model_parallax = false) {
    if (context.scene == nullptr || context.shader_updater == nullptr) return;

    auto apply_to_node = [&context, &depth, anchor, suppress_model_parallax](SceneNode* node) {
        if (node == nullptr) return;
        if (! node->Camera().empty()) return;
        auto* node_data = context.shader_updater->GetNodeData(node);
        if (node_data == nullptr) return;

        // Only camera-facing/world-facing nodes should receive this repaired parallax contract.
        // Effect source nodes render inside private effect cameras, so moving them here would bake
        // the same mouse offset into the offscreen texture and then apply it again at composition.
        node_data->SetParallaxContract(depth, anchor, suppress_model_parallax);
    };

    // The layer's registered draw handles live on its identity object. The list also contains
    // detached effect-source nodes, which the former nodeOwners scan never held; they own the
    // private effect camera and stay excluded by apply_to_node's camera check either way.
    for (auto* node : context.scene->GetLayerRuntimeNodes(owner_id)) {
        apply_to_node(node);
    }

    // The layer's effect passes and final composite are drawing phases rather than registered
    // node identities, so they are reached through the owning effect layer instead of nodeOwners.
    // Effect pass nodes carry no SceneNode camera (their camera is a pass override), so they keep
    // receiving the repaired contract exactly like they did through the former registration;
    // detached source nodes own the private effect camera and stay excluded by the camera check.
    if (auto* effect_layer = context.scene->FindImageEffectLayer(owner_id);
        effect_layer != nullptr) {
        for (std::size_t i = 0; i < effect_layer->EffectCount(); i++) {
            const auto& effect = effect_layer->GetEffect(i);
            if (! effect) continue;
            for (const auto& effect_node : effect->nodes) {
                apply_to_node(effect_node.sceneNode.get());
            }
        }
        if (effect_layer->HasFinalComposite()) {
            apply_to_node(&effect_layer->FinalNode());
        }
    }
}

void ApplyMissingImageParallaxFallbacks(ParseContext&                   context,
                                        const std::vector<WPObjectVar>& objects) {
    std::unordered_map<std::string, int32_t> explicit_parallax_peer_by_name;

    for (const auto& object : objects) {
        std::visit(visitor::overload {
                       [&explicit_parallax_peer_by_name](const wpscene::WPImageObject& image) {
                           if (! image.parallaxDepthAuthored ||
                               IsZeroParallaxDepth(ImageObjectParallaxDepth(image))) {
                               return;
                           }
                           const auto key = NormalizeParallaxPeerName(image.name);
                           if (! key.empty()) explicit_parallax_peer_by_name.emplace(key, image.id);
                       },
                       [&explicit_parallax_peer_by_name](const WPEmptyObject& empty) {
                           if (! empty.parallaxDepthAuthored ||
                               IsZeroParallaxDepth(empty.parallaxDepth)) {
                               return;
                           }
                           const auto key = NormalizeParallaxPeerName(empty.name);
                           if (! key.empty()) explicit_parallax_peer_by_name.emplace(key, empty.id);
                       },
                       [](const auto&) {
                       },
                   },
                   object);
    }

    for (const auto& object : objects) {
        const auto* image = std::get_if<wpscene::WPImageObject>(&object);
        if (image == nullptr || image->parallaxDepthAuthored || image->parent != 0) continue;

        const auto key     = NormalizeParallaxPeerName(image->name);
        auto       peer_it = explicit_parallax_peer_by_name.find(key);
        if (peer_it != explicit_parallax_peer_by_name.end() && peer_it->second != image->id) {
            auto peer_node_it = context.object_nodes.find(peer_it->second);
            if (peer_node_it != context.object_nodes.end() && peer_node_it->second) {
                // Some WE projects split a character into a static-looking root image plus an
                // explicitly-parallaxed detail/effect group with the same normalized name. The root
                // layer has no authored parent, but visually it must inherit the detail group's
                // parallax offset so the pieces stay locked together.
                ApplyNodeOwnerParallaxFallback(
                    context, image->id, { 0.0f, 0.0f }, peer_node_it->second.get());
                continue;
            }
        }

        const bool is_compose_layer         = image->image == "models/util/composelayer.json";
        const bool has_authored_descendants = context.dependent_parent_ids.count(image->id) != 0;
        const bool is_compose_container =
            is_compose_layer && (! image->effects.empty() || has_authored_descendants);
        if (is_compose_container) {
            // A composition source subtracts its parallax while rendering routed children into a
            // private target. Its final writer must restore the resolved default in scene space;
            // suppressing that offset makes an omitted root and child cancel to a static result.
            ApplyNodeOwnerParallaxFallback(
                context, image->id, wpscene::kDefaultParallaxDepth, nullptr, false);
        }
    }
}

namespace
{

constexpr std::string_view kSyntheticDirectDrawShapeTextureName {
    "__hanabi_shape_directdraw_transparent_source"
};

bool UsesShaderColorBlendMode(int32_t color_blend_mode) {
    return color_blend_mode >= 1 && color_blend_mode <= 30;
}

struct FinalShaderCapabilityEntry {
    std::string_view shader;
    FinalOutputCapability capability;
};

constexpr std::array<FinalShaderCapabilityEntry, 5> kFinalShaderCapabilities {{
    // Scroll evaluates UV/time in the authored layer projection. Running its visible raster in a
    // source-sized private target quantizes repeated pixel art before the layer is scaled to the
    // display. The capability is attached to the shader contract itself, never to a wallpaper,
    // texture name, or nearest-sampler heuristic.
    { "effects/scroll", FinalOutputCapability::SceneAuthoredWriter },
    // X-Ray unprojects the normalized desktop cursor through the inverse authored layer MVP and
    // divides the resulting local position by the source texture extent. A private effect-camera
    // pass uses a 2x2 clip-space helper mesh whose XY MVP is identity, collapsing cursor movement
    // to roughly one source texel before the neutral publication pass. Keep the authored final
    // shader in scene space so its matrix, pointer coordinates, and visible layer geometry remain
    // one coherent projection contract.
    { "effects/xray", FinalOutputCapability::SceneAuthoredWriter },
    // Vertex-mode distortions displace `a_Position` before multiplying by g_MVP, and the
    // displacement is authored in layer pixels: skew adds `g_Texture0Resolution.zw * g_Left` (and
    // friends), foliagesway adds `g_Strength * 100`, transform rotates/offsets the raw position.
    // That arithmetic is correct on the authored final pass: a +/-(W/2, H/2) pixel card with the
    // unscaled layer MVP. The private effect-camera pass instead draws the 2x2 clip-space helper
    // card, where a 0.27 skew on a 244 px box moves the left edge ~66 clip units off-target and
    // the whole layer vanishes. Keep these shaders in scene space.
    { "effects/skew", FinalOutputCapability::SceneAuthoredWriter },
    { "effects/foliagesway", FinalOutputCapability::SceneAuthoredWriter },
    { "effects/transform", FinalOutputCapability::SceneAuthoredWriter },
}};

FinalOutputCapability ResolveFinalShaderCapability(std::string_view shader) {
    const auto it = std::find_if(kFinalShaderCapabilities.begin(),
                                 kFinalShaderCapabilities.end(),
                                 [shader](const auto& entry) { return entry.shader == shader; });
    return it == kFinalShaderCapabilities.end()
        ? FinalOutputCapability::PrivateThenPublish
        : it->capability;
}

bool IsCurrentEffectWriterTarget(std::string_view target) {
    return target == SpecTex_Default || sstart_with(target, WE_EFFECT_PPONG_PREFIX_B);
}

BlendMode ResolveObjectFinalBlend(BlendMode authored_blend, int32_t color_blend_mode) {
    return color_blend_mode == 31 ? BlendMode::Additive : authored_blend;
}

void EnsureSystemTextureRegistered(Scene& scene, std::string_view texture_key) {
    auto* synthetic_parser = AsSyntheticImageParser(scene.imageParser.get());
    if (synthetic_parser == nullptr) return;

    const std::string key(texture_key);
    if (scene.textures.count(key) != 0) return;

    synthetic_parser->RegisterImage(key, CreateSceneScriptSolidImage(texture_key, { 0, 0, 0, 0 }));
    scene.textures[key] = SceneTexture {
        .url = key,
        .sample =
            TextureSample {
                .wrapS     = TextureWrap::CLAMP_TO_EDGE,
                .wrapT     = TextureWrap::CLAMP_TO_EDGE,
                .magFilter = TextureFilter::LINEAR,
                .minFilter = TextureFilter::LINEAR,
            },
        .format    = TextureFormat::RGBA8,
        .isVideo   = false,
        .isSprite  = false,
        .width     = 1,
        .height    = 1,
        .mapWidth     = 1,
        .mapHeight    = 1,
        .mipmapCount  = 1,
    };
    scene.dirtyImportedTextureKeys.insert(key);
}

bool ResolveObjectVisibility(bool raw_visible, const VisibleBinding& binding,
                             const UserPropertyMap* user_properties) {
    if (! binding.hasUserBinding()) return raw_visible;
    return EvaluateVisibleBinding(binding, user_properties);
}

bool EffectVisibilityCanChangeAtRuntime(const wpscene::WPImageEffect& effect) {
    if (! effect.visible_json.is_object()) return false;

    // These are the three runtime producers consumed by WPSceneParserBindings. A plain object with
    // only `value` is static, while any non-null producer requires a stable effect target and a
    // conditional execution route for the lifetime of the already-built graph.
    for (std::string_view producer : { "user", "script", "animation" }) {
        const std::string key(producer);
        if (effect.visible_json.contains(key) && ! effect.visible_json.at(key).is_null()) {
            return true;
        }
    }
    return false;
}

bool ResolveEffectVisibility(const wpscene::WPImageEffect& effect,
                             const UserPropertyMap*        user_properties) {
    return ResolveObjectVisibility(effect.visible, effect.visible_binding, user_properties);
}

// Shared with WPSceneParserBindings.cpp (declared in WPSceneParserShared.hpp).
} // namespace

bool IsCameraLayerObjectJson(const nlohmann::json& object_json) {
    if (! object_json.is_object()) return false;
    if (object_json.contains("camera") && object_json.at("camera").is_string()) return true;
    if (! object_json.contains("path") || ! object_json.at("path").is_string()) return false;

    // Camera assets can be serialized as otherwise-empty objects that only carry a camera path.
    // Treat those as camera layers too, because their zoom/origin properties still drive the
    // active view even when the path file itself contains no authored points.
    const auto path = object_json.at("path").get<std::string>();
    return path.find("scripts/camera_paths_") == 0;
}

bool IsCameraLayerRuntimeProperty(std::string_view property_name) {
    return property_name == "visible" || property_name == "origin" || property_name == "angles" ||
           property_name == "zoom" || property_name == "fov";
}

namespace
{

void PopulateGlobalBaseUniforms(ParseContext& context, const Scene& scene) {
    auto& gb                   = context.global_base_uniforms;
    gb["g_ViewUp"]             = std::array { 0.0f, 1.0f, 0.0f };
    gb["g_ViewRight"]          = std::array { 1.0f, 0.0f, 0.0f };
    gb["g_ViewForward"]        = std::array { 0.0f, 0.0f, -1.0f };
    gb["g_TexelSize"]          = std::array { 1.0f / 1920.0f, 1.0f / 1080.0f };
    gb["g_TexelSizeHalf"]      = std::array { 1.0f / 1920.0f / 2.0f, 1.0f / 1080.0f / 2.0f };
    gb["g_LightAmbientColor"]  = scene.ambientColor;
    gb["g_LightSkylightColor"] = scene.skylightColor;
    gb["g_NormalModelMatrix"]  = ShaderValue::fromMatrix(Matrix4f::Identity());

    if (context.ortho_w > 0 && context.ortho_h > 0) {
        gb["g_TexelSize"]     = std::array { 1.0f / static_cast<float>(context.ortho_w),
                                             1.0f / static_cast<float>(context.ortho_h) };
        gb["g_TexelSizeHalf"] = std::array { 1.0f / static_cast<float>(context.ortho_w) / 2.0f,
                                             1.0f / static_cast<float>(context.ortho_h) / 2.0f };
    }
}

void CollectSceneNodeRefs(const std::shared_ptr<SceneNode>&                           node,
                          std::unordered_map<SceneNode*, std::shared_ptr<SceneNode>>& refs) {
    if (! node) return;
    refs[node.get()] = node;
    for (const auto& child : node->GetChildren()) {
        CollectSceneNodeRefs(child, refs);
    }
}

int32_t AllocateDynamicLayerId(const Scene& scene) {
    int32_t max_id = 0;
    for (const auto layer_id : scene.layerOrder) {
        max_id = std::max(max_id, layer_id);
    }
    // Every registered layer-node slot lives on a SceneObject, so scanning the identity map covers
    // the former layerNodes keys (and any identity-bearing id beyond them, which only makes the
    // allocated id safer against reuse).
    for (const auto& [layer_id, _] : scene.sceneObjects) {
        (void)_;
        max_id = std::max(max_id, layer_id);
    }
    // Runtime-node records live on SceneObjects, so the identity scan above already covers every
    // id the former objectRuntimeNodes key scan contributed.
    return max_id + 1;
}

// mapRate < 1.0
void GenCardMesh(SceneMesh& mesh, const std::array<uint16_t, 2> size,
                 const std::array<float, 2> mapRate = { 1.0f, 1.0f }) {
    float left   = -(size[0] / 2.0f);
    float right  = size[0] / 2.0f;
    float bottom = -(size[1] / 2.0f);
    float top    = size[1] / 2.0f;
    float z      = 0.0f;

    float tw = mapRate[0], th = mapRate[1];

    // clang-format off
	const std::array pos = {
		left, bottom, z,
		left,  top, z,
		right, bottom, z,
		right,  top, z,
	};
	const std::array texCoord = {
		0.0f, th,
		0.0f, 0.0f,
		tw, th,
		tw, 0.0f,
	};
    // clang-format on

    SceneVertexArray vertex(
        {
            { WE_IN_POSITION.data(), VertexType::FLOAT3 },
            { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 },
        },
        4);
    vertex.SetVertex(WE_IN_POSITION, pos);
    vertex.SetVertex(WE_IN_TEXCOORD, texCoord);
    mesh.AddVertexArray(std::move(vertex));
}

void GenCardMeshWithTexCoordBounds(SceneMesh& mesh, const std::array<float, 2>& size,
                                   const std::array<float, 4>& texcoord_bounds) {
    const float width  = std::max(1.0f, size[0]);
    const float height = std::max(1.0f, size[1]);
    const float min_u  = texcoord_bounds[0];
    const float min_v  = texcoord_bounds[1];
    const float max_u  = texcoord_bounds[2];
    const float max_v  = texcoord_bounds[3];
    const float z      = 0.0f;

    auto local_x = [width](float u) {
        return (u - 0.5f) * width;
    };
    auto local_y = [height](float v) {
        return (0.5f - v) * height;
    };

    // The final writer may need to cover UVs outside [0, 1], but the effect shader still expects
    // its authored domain to be the original layer size. Expanding positions from UV bounds keeps
    // those two contracts independent: shader math stays stable, while the final quad no longer
    // clips generated DIRECTDRAW pixels at the canonical card edge.
    const std::array pos = {
        local_x(min_u), local_y(max_v), z,
        local_x(min_u), local_y(min_v), z,
        local_x(max_u), local_y(max_v), z,
        local_x(max_u), local_y(min_v), z,
    };
    const std::array texCoord = {
        min_u, max_v,
        min_u, min_v,
        max_u, max_v,
        max_u, min_v,
    };

    SceneVertexArray vertex(
        {
            { WE_IN_POSITION.data(), VertexType::FLOAT3 },
            { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 },
        },
        4);
    vertex.SetVertex(WE_IN_POSITION, pos);
    vertex.SetVertex(WE_IN_TEXCOORD, texCoord);
    mesh.AddVertexArray(std::move(vertex));
}

void ParseSpecTexName(std::string& name, const wpscene::WPMaterial& wpmat, const Scene* scene,
                      const WPShaderInfo& sinfo) {
    if (IsSpecTex(name)) {
        if (name == "_rt_FullFrameBuffer") {
            name = SpecTex_Default;
            if (wpmat.shader == "genericimage2" && ! exists(sinfo.combos, "BLENDMODE")) name = "";
            /*
            if(wpmat.shader == "genericparticle") {
                name = "_rt_ParticleRefract";
            }
            */
        } else if (sstart_with(name, WE_IMAGE_LAYER_COMPOSITE_PREFIX)) {
            LOG_INFO("link tex \"%s\"", name.c_str());
            int         wpid { -1 };
            std::regex  reImgId { R"(_rt_imageLayerComposite_([0-9]+))" };
            std::smatch match;
            if (std::regex_search(name, match, reImgId)) {
                STRTONUM(std::string(match[1]), wpid);
            }
            name = GenLinkTex((u32)wpid);
        } else if (name == SpecTex_DefaultPingPong) {
        } else if (sstart_with(name, WE_MIP_MAPPED_FRAME_BUFFER)) {
        } else if (sstart_with(name, WE_EFFECT_PPONG_PREFIX)) {
        } else if (sstart_with(name, WE_HALF_COMPO_BUFFER_PREFIX)) {
        } else if (sstart_with(name, WE_QUARTER_COMPO_BUFFER_PREFIX)) {
        } else if (sstart_with(name, WE_FULL_COMPO_BUFFER_PREFIX)) {
        } else if (name == "_rt_shadowAtlas") {
        } else if (name == kModelReflectionTargetName) {
            // 3D model reflection targets are registered lazily by model materialization, but
            // first-party grid materials may reference the sampler while their own model is not
            // reflected. Accept the name here as a model/runtime target instead of treating it as a
            // missing 2D texture.
        } else if (scene != nullptr && scene->renderTargets.count(name) != 0) {
            // Effect-local feedback buffers such as `_rt_EightBuffer1_<effect-layer-address>` are
            // registered dynamically from the authored FBO table. They still use Wallpaper Engine's
            // `_rt_` prefix, so the generic special-texture parser sees them, but the scene render
            // target table is the authoritative contract for whether they are valid runtime FBOs.
        } else {
            LOG_ERROR("unknown tex \"%s\"", name.c_str());
        }
    }
}

void ApplyKnownShaderSourceFixes(std::string_view shader_name, ShaderType stage,
                                 std::string& source) {
    // Wallpaper Engine's stock `genericropeparticle.vert` expands the ribbon with
    // `position += right * uvs.x * 2.0 - 1.0;`, which offsets the whole rope instead
    // of scaling `right` symmetrically around the centerline.
    if (shader_name == "genericropeparticle" && stage == ShaderType::VERTEX) {
        const std::string_view broken = "position += right * uvs.x * 2.0 - 1.0;";
        const auto             pos    = source.find(broken);
        if (pos != std::string::npos) {
            source.replace(pos, broken.size(), "position += right * (uvs.x * 2.0 - 1.0);");
        }
    }

    // Stock volumetricsback.frag is empty. The engine writes window Z into a
    // depth RT and volumetricsfront samples it with texSample2DBackBuffer.
    // This renderer exposes RTs as color images, so publish the same gl_FragCoord.z into .r.
    // The Steam copy is CRLF (`void main() {\r\n}`), so a LF-only search never matched and the
    // empty main left glOutColor at 0 — every ray then had length 0.
    if (shader_name == "volumetricsback" && stage == ShaderType::FRAGMENT &&
        source.find("gl_FragCoord") == std::string::npos) {
        source = "void main() {\n"
                 "\tgl_FragColor = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, 1.0);\n"
                 "}\n";
        LOG_INFO("SceneVolumetrics: rewrote volumetricsback.frag to publish window Z");
    }

}

bool IsMaterialRuntimeRenderTarget(const Scene* scene, const std::string& name) {
    // Wallpaper Engine effect FBOs are runtime render targets even when their authored names do not
    // use the `_rt_` prefix. Checking the scene table keeps names like `blur_start_2_<addr>` on the
    // render-target path instead of probing `/assets/materials/<name>.tex` and logging false VFS
    // errors.
    return scene != nullptr && scene->renderTargets.count(name) != 0;
}

void RegisterSceneTextureFromHeader(Scene& scene, const std::string& name,
                                    const ImageHeader& header) {
    if (scene.textures.count(name) != 0) return;

    SceneTexture texture;
    texture.sample    = header.sample;
    texture.url       = name;
    texture.format    = header.format;
    texture.isVideo   = header.isVideoTexture;
    texture.width     = header.width;
    texture.height    = header.height;
    texture.mapWidth  = header.mapWidth;
    texture.mapHeight = header.mapHeight;
    texture.mipmapCount   = header.mipmapCount;
    texture.mipmap_larger = header.mipmap_larger;
    if (header.isSprite) {
        texture.isSprite   = true;
        texture.spriteAnim = header.spriteAnim;
    }
    scene.textures[name] = std::move(texture);
}

// LoadMaterial is shared with WPSceneParserPostFx.cpp (declared in
// WPSceneParserShared.hpp), so it needs external linkage; the anonymous
// namespace resumes right after it.
} // namespace

std::optional<MaterialLoadResult>
LoadMaterial(fs::VFS& vfs, const wpscene::WPMaterial& wpmat, Scene* pScene, SceneNode* pNode,
             SceneMaterial* pMaterial, WPShaderValueData* pSvData,
             const UserPropertyMap* user_properties,
             WPShaderInfo*          pWPShaderInfo,
             GeometryStagePolicy geometry_stage) {
    bool geometry_stage_loaded { false };

    auto& svData   = *pSvData;
    auto& material = *pMaterial;

    std::unique_ptr<WPShaderInfo> upWPShaderInfo(nullptr);
    if (pWPShaderInfo == nullptr) {
        upWPShaderInfo = std::make_unique<WPShaderInfo>();
        pWPShaderInfo  = upWPShaderInfo.get();
    }

    // Scene materials compile their HDR shader variant only when the wallpaper actually renders
    // into an HDR swapchain (display HDR). Ultra post-processing on a standard-range output keeps
    // the LDR material variant: the HDR variant's g_Brightness multiply and CombineLighting
    // overbright term would wash out lit surfaces that the standard-range chain then blooms.
    // Vivid currently always renders standard-range, so the combo stays off.

    SceneMaterialCustomShader materialShader;

    auto& shader = materialShader.shader;
    shader       = std::make_shared<SceneShader>();
    shader->name = wpmat.shader;

    std::string shaderPath("/assets/shaders/" + wpmat.shader);

    std::vector<WPShaderUnit> sd_units;
    sd_units.push_back(WPShaderUnit {
        .stage           = ShaderType::VERTEX,
        .src             = fs::GetFileContent(vfs, shaderPath + ".vert"),
        .preprocess_info = {},
        .debug_name      = wpmat.shader + ".vert",
    });
    if (geometry_stage != GeometryStagePolicy::Disabled) {
        const auto geometry_path = shaderPath + ".geom";
        const bool geometry_source_exists = vfs.Contains(geometry_path);
        if (! geometry_source_exists && geometry_stage == GeometryStagePolicy::Required) {
            LOG_ERROR("material '%s' required geometry shader source missing shader='%s' path='%s'",
                      wpmat.shader.c_str(),
                      wpmat.shader.c_str(),
                      geometry_path.c_str());
            return std::nullopt;
        }
        if (geometry_source_exists) {
            sd_units.push_back(WPShaderUnit {
                .stage           = ShaderType::GEOMETRY,
                .src             = fs::GetFileContent(vfs, geometry_path),
                .preprocess_info = {},
                .debug_name      = wpmat.shader + ".geom",
            });
            // GS_ENABLED describes a stage that was successfully materialized. It is deliberately
            // set after the VFS check so combo state and mesh topology cannot claim a geometry ABI
            // that the selected material does not actually provide.
            pWPShaderInfo->combos["GS_ENABLED"] = "1";
            geometry_stage_loaded = true;
        }
    }
    sd_units.push_back(WPShaderUnit {
        .stage           = ShaderType::FRAGMENT,
        .src             = fs::GetFileContent(vfs, shaderPath + ".frag"),
        .preprocess_info = {},
        .debug_name      = wpmat.shader + ".frag",
    });

    for (auto& unit : sd_units) {
        ApplyKnownShaderSourceFixes(wpmat.shader, unit.stage, unit.src);
    }

    bool lighting_v1 { false };
    if (pScene != nullptr) {
        const auto lighting_desc = LightingDescFromScene(*pScene);
        for (auto& unit : sd_units) {
            if (ExpandRequireLightingV1(unit.src, lighting_desc)) lighting_v1 = true;
        }
        if (lighting_v1) {
            LOG_INFO("SceneLightingExpand: shader='%s' point=%d spot=%d directional=%d tube=%d "
                     "shadows=%s",
                     wpmat.shader.c_str(),
                     lighting_desc.point,
                     lighting_desc.spot,
                     lighting_desc.directional,
                     lighting_desc.tube,
                     lighting_desc.shadows ? "true" : "false");
        }
    }

    auto textures = wpmat.textures;
    if (wpmat.usertextures.size() > textures.size()) {
        textures.resize(wpmat.usertextures.size());
    }
    for (usize i = 0; i < wpmat.usertextures.size(); i++) {
        const auto& binding = wpmat.usertextures[i];
        if (binding.empty()) continue;
        if (binding.type == "system") {
            if (binding.name == "$mediaThumbnail") {
                textures[i] = std::string(WP_SCENE_SCRIPT_MEDIA_THUMBNAIL_TEXTURE);
                EnsureSystemTextureRegistered(*pScene, WP_SCENE_SCRIPT_MEDIA_THUMBNAIL_TEXTURE);
            } else if (binding.name == "$mediaPreviousThumbnail") {
                textures[i] = std::string(WP_SCENE_SCRIPT_MEDIA_PREVIOUS_THUMBNAIL_TEXTURE);
                EnsureSystemTextureRegistered(*pScene,
                                              WP_SCENE_SCRIPT_MEDIA_PREVIOUS_THUMBNAIL_TEXTURE);
            }
            continue;
        }

        const auto* property = LookupUserPropertyString(user_properties, binding.name);
        if (property == nullptr || property->empty()) continue;

        textures[i] = *property;
    }

    std::vector<WPShaderTexInfo>                 texinfos;
    std::unordered_map<std::string, ImageHeader> texHeaders;
    // The shader parser uses this effective texture list to decide whether a texture-driven combo
    // may expose a sampler branch. User texture bindings must be resolved before this point so a
    // real selected texture still enables its combo, while an empty optional mask slot stays off.
    for (const auto& el : textures) {
        if (el.empty()) {
            texinfos.push_back({ false });
        } else if (! IsSpecTex(el) && ! IsMaterialRuntimeRenderTarget(pScene, el)) {
            const auto& texh = pScene->imageParser->ParseHeader(el);
            texHeaders[el]   = texh;
            if (texh.extraHeader.count("compo1") == 0) {
                texinfos.push_back({ false });
                continue;
            }
            const auto compo_flag = [&texh](const char* key) {
                const auto it = texh.extraHeader.find(key);
                return it != texh.extraHeader.end() && it->second.val != 0;
            };
            texinfos.push_back({ true,
                                 {
                                     compo_flag("compo1"),
                                     compo_flag("compo2"),
                                     compo_flag("compo3"),
                                     compo_flag("compo4"),
                                 } });
        } else {
            // Runtime render targets should expose sampler metadata to the shader preprocessor just
            // like `_rt_` textures. Their exact dimensions are resolved below from
            // SceneRenderTarget, so no material header lookup is needed here.
            WPShaderTexInfo texinfo { .enabled = true };
            if (pScene != nullptr) {
                const auto rt_it = pScene->renderTargets.find(el);
                if (rt_it != pScene->renderTargets.end()) {
                    // Render-target sampling conventions travel with the bound texture slot. This
                    // keeps screen-space reflection correction generic and avoids changing the
                    // producer pass viewport, which would also affect geometry and culling.
                    texinfo.screenSpaceSampleYFlip = rt_it->second.screenSpaceSampleYFlip;
                }
            }
            texinfos.push_back(texinfo);
        }
    }

    for (auto& unit : sd_units) {
        unit.src = WPShaderParser::PreShaderSrc(vfs, unit.src, pWPShaderInfo, texinfos);
    }

    shader->default_uniforms = pWPShaderInfo->svs;

    for (const auto& el : wpmat.combos) {
        pWPShaderInfo->combos[el.first] = std::to_string(el.second);
    }

    if (lighting_v1 && pScene != nullptr) {
        if (pScene->shadows.quality != 0 && SceneHasShadowLights(*pScene)) {
            pWPShaderInfo->combos["LIGHTS_SHADOW_MAPPING"]         = "1";
            pWPShaderInfo->combos["LIGHTS_SHADOW_MAPPING_QUALITY"] =
                std::to_string(pScene->shadows.quality);
        }
    }

    // ALPHATOCOVERAGE is a compile-time combo injected when the material blending
    // mode is alphatocoverage. Live msaa changes do not recompile materials;
    // rasterizer A2C is a separate draw-state bit and follows the current sample
    // count. Shaders may contain an unused #if ALPHATOCOVERAGE block without
    // this blending mode; those stay combo-off so 2D translucent layers are not
    // coverage-tested.
    const bool blending_alpha_to_coverage = wpmat.blending == "alphatocoverage";
    if (blending_alpha_to_coverage) {
        pWPShaderInfo->combos["ALPHATOCOVERAGE"] = "1";
    }

    if (pWPShaderInfo->defTexs.size() > 0) {
        for (auto& t : pWPShaderInfo->defTexs) {
            if (textures.size() > t.first) {
                if (! textures.at(t.first).empty()) continue;
            } else {
                textures.resize(t.first + 1);
            }
            textures[t.first] = t.second;
        }
    }

    for (usize i = 0; i < textures.size(); i++) {
        std::string name = textures.at(i);
        if (name == "_alias_lightCookie") name = "cookie/flashlight1";
        ParseSpecTexName(name, wpmat, pScene, *pWPShaderInfo);
        material.textures.push_back(name);
        material.defines.push_back("g_Texture" + std::to_string(i));
        if (name.empty()) {
            continue;
        }

        std::array<i32, 4> resolution {};
        if (IsSpecTex(name) || IsMaterialRuntimeRenderTarget(pScene, name)) {
            if (IsSpecLinkTex(name)) {
                svData.renderTargets.push_back({ i, name });
            } else if (pScene->renderTargets.count(name) == 0) {
                LOG_ERROR("%s not found in render targes", name.c_str());
            } else {
                svData.renderTargets.push_back({ i, name });
                const auto& rt = pScene->renderTargets.at(name);
                // Runtime render targets may keep a larger physical allocation than the logical
                // content they currently store. Forwarding the authored content extent through
                // `.zw` preserves Wallpaper Engine's original "sample area" contract for effects
                // that distinguish between allocated size and meaningful image size.
                resolution = rt.ResolutionVector();
            }
        } else {
            const ImageHeader& texh = texHeaders.count(name) == 0
                                          ? pScene->imageParser->ParseHeader(name)
                                          : texHeaders.at(name);
            SetTextureFormatShaderDefine(*pWPShaderInfo, i, texh.format);
            if (texh.mipmap_larger) {
                resolution = { texh.width, texh.height, texh.mapWidth, texh.mapHeight };
            } else {
                resolution = { texh.mapWidth, texh.mapHeight, texh.mapWidth, texh.mapHeight };
            }
            // Parse-time values use the authored header. After GPU upload the
            // shader updater overwrites g_TextureNResolution from
            // EffectiveImportedTextureResolution() so half/auto follow the
            // bind-path GPU extent.

            RegisterSceneTextureFromHeader(*pScene, name, texh);
            if ((pScene->textures.at(name)).isSprite) {
                material.hasSprite = true;
                const auto& f1     = texh.spriteAnim.GetCurFrame();
                if (wpmat.shader == "genericparticle" || wpmat.shader == "genericropeparticle") {
                    pWPShaderInfo->combos["SPRITESHEET"] = "1";
                    pWPShaderInfo->combos["THICKFORMAT"] = "1";
                    if (algorism::IsPowOfTwo((u32)texh.width) &&
                        algorism::IsPowOfTwo((u32)texh.height)) {
                        pWPShaderInfo->combos["SPRITESHEETBLENDNPOT"] = "1";
                        resolution = ResolvePaddedSpriteSheetResolution(texh, f1);
                    }
                    materialShader.constValues["g_RenderVar1"] = std::array {
                        f1.xAxis[0], f1.yAxis[1], (float)(texh.spriteAnim.numFrames()), f1.rate
                    };
                }
            }
        }
        if (! resolution.empty()) {
            const std::string gResolution = WE_GLTEX_RESOLUTION_NAMES[i];

            materialShader.constValues[gResolution] = array_cast<float>(resolution);
        }
    }
    if (exists(pWPShaderInfo->combos, "LIGHTING")) {
        // pWPShaderInfo->combos["PRELIGHTING"] =
        // pWPShaderInfo->combos.at("LIGHTING");
    }

    if (! WPShaderParser::CompileToSpv(
            pScene->scene_id, sd_units, shader->codes, vfs, pWPShaderInfo, texinfos)) {
        return std::nullopt;
    }

    material.blenmode = ParseBlendMode(wpmat.blending);
    material.alpha_to_coverage = blending_alpha_to_coverage;

    const auto& fragment_unit = sd_units.back();
    assert(fragment_unit.stage == ShaderType::FRAGMENT);
    for (uint i = 0; i < material.textures.size(); i++) {
        if (! exists(fragment_unit.preprocess_info.active_tex_slots, i)) {
            material.textures[i].clear();
        }
    }

    for (const auto& el : pWPShaderInfo->baseConstSvs) {
        materialShader.constValues[el.first] = el.second;
    }
    material.customShader = materialShader;
    material.name         = wpmat.shader;
    // Store the material-name to GLSL-uniform alias table on the live SceneMaterial. Runtime WE
    // scripts can then write properties such as `thisObject.getMaterial(0).raythreshold` and have
    // the script bridge resolve them to the actual shader uniform (`g_Threshold`) that this parse
    // pass discovered from the shader metadata comments.
    material.uniformAliases = pWPShaderInfo->alias;

    return MaterialLoadResult { .geometry_stage_loaded = geometry_stage_loaded };
}

namespace
{

bool ConfigureEffectFinalComposite(ParseContext& context, SceneImageEffectLayer& effect_layer,
                                   std::string_view initial_source, int32_t owner_layer_id,
                                   std::string_view         owner_name,
                                   int32_t                  color_blend_mode,
                                   const WPShaderValueData* final_transform_data = nullptr,
                                   const wpscene::WPMaterial* direct_material    = nullptr,
                                   int32_t                    direct_effect_id   = 0) {
    auto& vfs = *context.vfs;

    wpscene::WPMaterial composite_source;
    if (direct_material != nullptr) {
        // A single authored effect pass with no private buffers is the layer's on-screen
        // writer: the pass material draws the final quad at output resolution while sampling
        // the private source. Routing it through a second ping-pong target instead would cap
        // procedural detail (orbit lines, atmosphere rims) at that target's resolution.
        composite_source = *direct_material;
    } else {
        nlohmann::json composite_json;
        if (! PARSE_JSON(fs::GetFileContent(vfs, "/assets/materials/util/effectpassthrough.json"),
                         composite_json) ||
            ! composite_source.FromJson(composite_json)) {
            LOG_ERROR(
                "SceneEffectFinalComposite: layer=%d name='%.*s' failed to load passthrough material",
                owner_layer_id,
                static_cast<int>(owner_name.size()),
                owner_name.data());
            return false;
        }
    }

    if (composite_source.textures.empty()) composite_source.textures.resize(1);
    if (direct_material == nullptr || composite_source.textures[0].empty()) {
        composite_source.textures[0] = std::string(initial_source);
    }
    if (direct_material == nullptr) {
        // Modes 1..30 are framebuffer-aware shader blend equations. Modes 0 and 31 use the neutral
        // shader variant; mode 31 is expressed by the final fixed-function additive state instead.
        composite_source.combos["BLENDMODE"] =
            UsesShaderColorBlendMode(color_blend_mode) ? color_blend_mode : 0;
    }

    WPShaderInfo composite_shader_info;
    composite_shader_info.baseConstSvs = context.global_base_uniforms;
    if (direct_material != nullptr) {
        composite_shader_info.baseConstSvs["g_EffectTextureProjectionMatrix"] =
            ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
        composite_shader_info.baseConstSvs["g_EffectTextureProjectionMatrixInverse"] =
            ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
    } else {
        // The source object/effect chain has already baked authored color, alpha, brightness, and
        // effect output into the ping-pong texture. The final composite must therefore be a neutral
        // sampler: it applies the layer's final mesh and blend state, but it must not tint or fade
        // the resolved texture a second time.
        composite_shader_info.baseConstSvs["g_Color4"] =
            std::array<float, 4> { 1.0f, 1.0f, 1.0f, 1.0f };
        composite_shader_info.baseConstSvs["g_Color"] = std::array<float, 3> { 1.0f, 1.0f, 1.0f };
        composite_shader_info.baseConstSvs["g_Alpha"]      = 1.0f;
        composite_shader_info.baseConstSvs["g_UserAlpha"]  = 1.0f;
        composite_shader_info.baseConstSvs["g_Brightness"] = 1.0f;
    }

    SceneMaterial     composite_material;
    WPShaderValueData composite_data;
    if (! LoadMaterial(vfs,
                       composite_source,
                       context.scene.get(),
                       &effect_layer.FinalNode(),
                       &composite_material,
                       &composite_data,
                       context.user_properties,
                       &composite_shader_info)) {
        LOG_ERROR("SceneEffectFinalComposite: layer=%d name='%.*s' material compile failed",
                  owner_layer_id,
                  static_cast<int>(owner_name.size()),
                  owner_name.data());
        return false;
    }
    if (direct_material != nullptr) {
        LoadConstvalue(composite_material, composite_source, composite_shader_info);
        LoadUserShaderValue(
            composite_material, composite_source, composite_shader_info, context.user_properties);
    }
    if (final_transform_data != nullptr) {
        // The fallback final composite is the screen-space writer used when the last authored
        // effect in a layer is hidden. It is not an authored child node, so copying the full parent
        // transform binding would multiply route matrices twice; only the parallax contract is
        // mirrored from the visible world node so hidden-effect fallbacks keep moving with their
        // compose/text layer.
        composite_data.CopyParallaxContractFrom(*final_transform_data);
    }

    auto composite_mesh = std::make_shared<SceneMesh>();
    composite_mesh->AddMaterial(std::move(composite_material));

    auto& final_node = effect_layer.FinalNode();
    final_node.SetName(owner_name.empty()
                           ? std::string("__hanabi_effect_final_composite")
                           : std::string(owner_name) + "::__hanabi_effect_final_composite");
    final_node.ID() = owner_layer_id;
    final_node.SetCamera(std::string());
    final_node.AddMesh(composite_mesh);

    context.shader_updater->SetNodeData(&final_node, composite_data);
    if (direct_material != nullptr) {
        RegisterUserShaderValueBindings(context,
                                        composite_source,
                                        composite_shader_info,
                                        &final_node,
                                        owner_layer_id,
                                        owner_name);
        RegisterConstantShaderValueBindings(context,
                                            composite_source,
                                            composite_shader_info,
                                            &final_node,
                                            owner_layer_id,
                                            owner_name,
                                            direct_effect_id,
                                            0,
                                            0);
        LOG_INFO("SceneEffectDirectFinalDraw: layer=%d name='%.*s' effect-id=%d shader='%s' "
                 "source='%.*s'",
                 owner_layer_id,
                 static_cast<int>(owner_name.size()),
                 owner_name.data(),
                 direct_effect_id,
                 composite_source.shader.c_str(),
                 static_cast<int>(initial_source.size()),
                 initial_source.data());
    }
    // The final composite is a drawing phase of the owning layer, not a second layer identity:
    // it stays out of sceneGraph/nodeOwners. Its node id (set above) is the back-reference the
    // render graph uses to resolve the owning layer for visibility and residency decisions.
    effect_layer.SetFinalCompositeSource(std::string(initial_source));
    return true;
}

SceneImageEffectLayer::HiddenFinalCompositePolicy
ResolveHiddenFinalCompositePolicy(const Scene& scene, const wpscene::WPImageObject& image) {
    // This is a source-contract decision rather than a shader or layer-name decision. Normal image
    // layers have a meaningful pre-effect source, so disabling the final effect should reveal that
    // source. Passthrough compose helpers are source-less routing layers; when their final effect is
    // hidden they should publish nothing instead of preserving a helper render target that may only
    // contain previous framebuffer contents.
    //
    // The flag is an object-level identity property, so the registered SceneObject is the
    // authoritative source; the authored config is only the bootstrap value for paths that
    // materialize before identity registration.
    const auto* scene_object = scene.FindSceneObject(image.id);
    const bool  passthrough =
        scene_object != nullptr ? scene_object->Passthrough() : image.config.passthrough;
    return passthrough
        ? SceneImageEffectLayer::HiddenFinalCompositePolicy::SuppressOutput
        : SceneImageEffectLayer::HiddenFinalCompositePolicy::PreserveSource;
}

SceneImageEffectLayer::SourcePolicy ResolveImageEffectSourcePolicy(
    bool is_compose_layer, const wpscene::WPImageObject& image) {
    if (!is_compose_layer) return SceneImageEffectLayer::SourcePolicy::OwnerNode;
    // copybackground composition layers seed their private source from the owner framebuffer
    // material before routed children are accumulated. Ordinary composition layers instead author
    // their source exclusively through routed children; drawing the owner framebuffer again would
    // composite the already-rendered scene a second time and brighten the complete wallpaper.
    return image.copybackground ? SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren
                                : SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly;
}

std::string_view ImageEffectSourcePolicyName(SceneImageEffectLayer::SourcePolicy policy) {
    switch (policy) {
    case SceneImageEffectLayer::SourcePolicy::OwnerNode: return "owner-node";
    case SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren:
        return "owner-node-and-proxy-children";
    case SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly: return "proxy-children-only";
    }
    return "unknown";
}

struct ImageEffectCameraClipRange {
    float near_clip { -1.0f };
    float far_clip { 1.0f };
};

PuppetBounds3D TransformPuppetBounds(const PuppetBounds3D& bounds,
                                     const Eigen::Affine3f& transform) {
    PuppetBounds3D result;
    if (!bounds.IsFiniteAndOrdered()) return result;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                result.Include(transform * Eigen::Vector3f {
                    x == 0 ? bounds.min.x() : bounds.max.x(),
                    y == 0 ? bounds.min.y() : bounds.max.y(),
                    z == 0 ? bounds.min.z() : bounds.max.z(),
                });
            }
        }
    }
    return result;
}

PuppetSurfaceProjection BuildPuppetSurfaceProjection(
    const wpscene::WPImageObject& image,
    const WPMdl& mdl,
    const SceneMesh& puppet_mesh,
    std::string camera_name,
    std::string target_name,
    const std::array<float, 2>& effect_source_size) {
    PuppetSurfaceProjection projection;
    projection.authored_layer_size = { image.size[0], image.size[1] };
    projection.geometry_transform = puppet_mesh.GeometryTransform();
    projection.camera_name = std::move(camera_name);
    projection.target_name = std::move(target_name);
    projection.render_density = std::max(
        effect_source_size[0] / std::max(image.size[0], 1.0f),
        effect_source_size[1] / std::max(image.size[1], 1.0f));
    projection.asset_bounds = mdl.asset_bounds;
    projection.authored_pose_bounds = mdl.authored_pose_bounds.IsFiniteAndOrdered()
                                          ? mdl.authored_pose_bounds
                                          : mdl.asset_bounds;

    PuppetBounds3D authored_surface = TransformPuppetBounds(
        projection.authored_pose_bounds, projection.geometry_transform);
    authored_surface.Include(Eigen::Vector3f {
        -image.size[0] * 0.5f, -image.size[1] * 0.5f, 0.0f
    });
    authored_surface.Include(Eigen::Vector3f {
        image.size[0] * 0.5f, image.size[1] * 0.5f, 0.0f
    });
    // One physical target pixel protects raster-edge precision while keeping the guard tied to the
    // actual render density. The resulting surface remains immutable for authored animation.
    const float guard = 1.0f / std::max(projection.render_density,
                                        std::numeric_limits<float>::epsilon());
    projection.surface_bounds = authored_surface;
    projection.surface_bounds.min -= Eigen::Vector3f::Constant(guard);
    projection.surface_bounds.max += Eigen::Vector3f::Constant(guard);
    return projection;
}

TextureSample ResolvePrimaryMaterialSampler(const Scene& scene, const SceneMaterial& material) {
    if (material.textures.empty() || material.textures.front().empty()) return {};
    const auto& texture_name = material.textures.front();
    if (const auto texture_it = scene.textures.find(texture_name);
        texture_it != scene.textures.end()) {
        return texture_it->second.sample;
    }
    if (const auto target_it = scene.renderTargets.find(texture_name);
        target_it != scene.renderTargets.end()) {
        return target_it->second.sample;
    }
    return {};
}

bool PrimaryMaterialTextureIsSprite(const Scene& scene, const SceneMaterial& material) {
    if (material.textures.empty() || material.textures.front().empty()) return false;
    const auto texture_it = scene.textures.find(material.textures.front());
    return texture_it != scene.textures.end() && texture_it->second.isSprite;
}

ImageEffectCameraClipRange ResolveImageEffectCameraClipRange(bool has_animated_puppet_mesh) {
    if (! has_animated_puppet_mesh) return {};

    // Animated puppet meshes are still rendered by 2D image-effect cameras, but their bone
    // animation is not limited to the flat source-card z range. Keep ordinary non-puppet effects on
    // the tight range and widen only puppet-capable layer-surface paths that need authored z to
    // survive the final synthetic writer.
    return { -1024.0f, 1024.0f };
}

void LoadAlignment(SceneNode& node, std::string_view align, Vector2f size) {
    // Alignment changes where the centered quad is drawn relative to the authored origin. Store it
    // as a local mesh offset instead of mutating translation, because translation is the pivot that
    // Wallpaper Engine scripts read and rotate around.
    node.SetAlignmentOffset(ResolveImageAlignmentOffset(align, size));
}

std::shared_ptr<SceneNode> FindParentNode(ParseContext& context, int32_t parent_id) {
    auto it = context.object_nodes.find(parent_id);
    return it == context.object_nodes.end() ? nullptr : it->second;
}

// Render-order proxy routing is derived from authored parent bindings at query time
// (Scene::IsRenderOrderProxyNode / RenderOrderProxyChildrenOf), so no separate parse-time proxy
// node table is required.

struct AttachmentBinding {
    uint32_t        bone_index { 0xFFFFFFFFu };
    Eigen::Affine3f transform { Eigen::Affine3f::Identity() };
};

std::optional<AttachmentBinding> ResolveAttachmentBinding(const ParseContext& context,
                                                          int32_t             parent_id,
                                                          std::string_view    attachment) {
    auto pit = context.object_puppets.find(parent_id);
    if (pit == context.object_puppets.end() || pit->second == nullptr) return std::nullopt;

    const auto& puppet = *pit->second;
    if (const auto* named_attachment = puppet.FindAttachment(attachment)) {
        return AttachmentBinding {
            .bone_index = named_attachment->bone_index,
            .transform  = named_attachment->bind_transform,
        };
    }

    auto bone_index = puppet.FindBoneIndex(attachment);
    if (bone_index == 0xFFFFFFFFu) return std::nullopt;
    return AttachmentBinding {
        .bone_index = bone_index,
        .transform  = Eigen::Affine3f::Identity(),
    };
}

// Shared with WPSceneParserModel.cpp (declared in WPSceneParserShared.hpp).
} // namespace

bool ConfigureBoneAttachment(ParseContext& context, int32_t parent_id, std::string_view attachment,
                             const Eigen::Affine3f& local_transform, std::string_view object_kind,
                             std::string_view object_name, WPShaderValueData& node_data) {
    if (parent_id == 0 || attachment.empty()) return false;

    auto parent_node = FindParentNode(context, parent_id);
    if (! parent_node) {
        LOG_ERROR("parent id %d for %s '%s' not found while resolving attachment '%s'",
                  (int)parent_id,
                  std::string(object_kind).c_str(),
                  std::string(object_name).c_str(),
                  std::string(attachment).c_str());
        return false;
    }

    auto attachment_binding = ResolveAttachmentBinding(context, parent_id, attachment);
    if (! attachment_binding.has_value()) {
        LOG_ERROR("attachment '%s' not found for %s '%s'",
                  std::string(attachment).c_str(),
                  std::string(object_kind).c_str(),
                  std::string(object_name).c_str());
        return false;
    }

    node_data.AttachToBone(parent_node.get(),
                           attachment_binding->bone_index,
                           attachment_binding->transform,
                           local_transform);
    return true;
}

void AttachNodeToScene(ParseContext& context, const std::shared_ptr<SceneNode>& node,
                       int32_t parent_id, const std::string& object_name,
                       WPShaderValueData* node_data) {
    if (parent_id == 0) {
        context.scene->sceneGraph->AppendChild(node);
        return;
    }

    auto parent = FindParentNode(context, parent_id);
    if (! parent) {
        LOG_ERROR("parent id %d for object '%s' not found, attaching to scene root",
                  (int)parent_id,
                  object_name.c_str());
        context.scene->sceneGraph->AppendChild(node);
        return;
    }

    parent->AppendChild(node);
    if (node_data != nullptr) {
        node_data->SetParallaxAnchor(parent.get());
    }
}

namespace
{

bool ShouldInheritParentParallax(ParseContext& context, const SceneNode& parent,
                                 const WPShaderValueData& node_data) {
    if (context.shader_updater == nullptr) return true;
    if (! node_data.parallaxDepthAuthored) {
        // Wallpaper Engine stores its numeric default as an omitted field. A root layer with that
        // field still resolves to the scene default depth, but a child layer uses the parent
        // parallax contract instead of becoming a separate depth-1 camera source. This check must
        // precede the numeric comparison because both omitted and explicitly authored `1 1` carry
        // the same resolved float values.
        return true;
    }
    if (IsZeroParallaxDepth(node_data.parallaxDepth)) return true;

    const auto* parent_data = context.shader_updater->GetNodeData(&parent);
    if (parent_data == nullptr) return true;

    // Inherited transform and inherited camera-parallax are separate contracts in Wallpaper
    // Engine. Authored relay layers can be positioned relative to a parent while still carrying
    // their own parallaxDepth. Keep the old parent-anchor behavior for repaired/suppressed
    // containers and zero-depth parents, but let explicit child depth survive when both sides
    // authored their own non-zero parallax values.
    if (parent_data->suppress_model_parallax || parent_data->IsBoneAttached()) return true;
    if (parent_data->parallax_anchor != nullptr) return true;
    if (IsZeroParallaxDepth(parent_data->parallaxDepth)) return true;

    return false;
}

enum class ParentTransformBindingContract
{
    None,
    InheritAuthoredParent,
};

enum class ParentParallaxAnchorContract
{
    None,
    InheritWhenCompatible,
    ForceAuthoredParent,
};

struct ParentTransformContract {
    int32_t                         parent_id { 0 };
    ParentTransformBindingContract  transform_binding {
        ParentTransformBindingContract::None
    };
    ParentParallaxAnchorContract    parallax_anchor {
        ParentParallaxAnchorContract::None
    };

    static ParentTransformContract InheritAuthoredParentTransform(int32_t parent_id) {
        return { .parent_id         = parent_id,
                 .transform_binding = ParentTransformBindingContract::InheritAuthoredParent,
                 .parallax_anchor   = ParentParallaxAnchorContract::InheritWhenCompatible };
    }

    static ParentTransformContract RoutedEffectWorldTransform(int32_t parent_id) {
        return { .parent_id         = parent_id,
                 .transform_binding = ParentTransformBindingContract::InheritAuthoredParent,
                 .parallax_anchor   = ParentParallaxAnchorContract::ForceAuthoredParent };
    }

    static ParentTransformContract ParallaxAnchorWhenCompatible(int32_t parent_id) {
        return { .parent_id       = parent_id,
                 .parallax_anchor = ParentParallaxAnchorContract::InheritWhenCompatible };
    }

    static ParentTransformContract ForceParallaxAnchor(int32_t parent_id) {
        return { .parent_id       = parent_id,
                 .parallax_anchor = ParentParallaxAnchorContract::ForceAuthoredParent };
    }
};

void ApplyParentTransformContract(ParseContext& context, const ParentTransformContract& contract,
                                  WPShaderValueData& node_data) {
    if (contract.parent_id == 0) return;
    auto parent = FindParentNode(context, contract.parent_id);
    if (! parent) return;

    const bool inherit_parent_parallax =
        contract.parallax_anchor == ParentParallaxAnchorContract::ForceAuthoredParent ||
        (contract.parallax_anchor == ParentParallaxAnchorContract::InheritWhenCompatible &&
         ShouldInheritParentParallax(context, *parent, node_data));

    if (contract.transform_binding ==
        ParentTransformBindingContract::InheritAuthoredParent) {
        node_data.InheritParentTransform(parent.get(), inherit_parent_parallax);
        return;
    }

    if (inherit_parent_parallax) {
        node_data.SetParallaxAnchor(parent.get());
    }
}

// Shared with WPSceneParserParticle.cpp (declared in WPSceneParserShared.hpp).
} // namespace

void ConfigureInheritedParentBinding(ParseContext& context, int32_t parent_id,
                                     WPShaderValueData& node_data) {
    ApplyParentTransformContract(
        context, ParentTransformContract::InheritAuthoredParentTransform(parent_id), node_data);
}

namespace
{

void ConfigureRoutedEffectWorldParentBinding(ParseContext& context, int32_t parent_id,
                                             WPShaderValueData& node_data) {
    // Effect-backed image layers split into a routed world node and private effect nodes. The world
    // node is the scene-space writer that ResolveEffect()/UpdateUniforms use to synchronize the
    // final output, so it must keep the authored parent as its camera-parallax anchor even when both
    // parent and child authored non-zero parallaxDepth. Authored effect materials still keep their
    // own child parallax contract; this binding only preserves the parent component that would
    // otherwise be lost before the final writer is synchronized.
    ApplyParentTransformContract(
        context, ParentTransformContract::RoutedEffectWorldTransform(parent_id), node_data);
}

struct EffectWriterTransformContract {
    std::array<float, 2>       parallax_depth { 0.0f, 0.0f };
    bool                       parallax_depth_authored { true };
    SceneImageEffectLayer*     projection_layer { nullptr };
    bool                       binds_puppet_surface { false };
    ParentTransformContract    parent_transform {};
    bool suppress_own_model_parallax { false };
};

void ApplyEffectWriterTransformContract(ParseContext& context,
                                        const EffectWriterTransformContract& contract,
                                        WPShaderValueData& data) {
    data.SetParallaxContract(contract.parallax_depth,
                             nullptr,
                             false,
                             contract.parallax_depth_authored);

    if (contract.projection_layer != nullptr) {
        data.SetEffectTextureProjection(&contract.projection_layer->FinalNode(),
                                        &contract.projection_layer->FinalMesh());
        if (contract.binds_puppet_surface) {
            data.SetPuppetSurface(contract.projection_layer,
                                  &contract.projection_layer->FinalMesh());
        }
    }

    ApplyParentTransformContract(context, contract.parent_transform, data);

    if (contract.suppress_own_model_parallax) {
        data.SuppressOwnModelParallax();
    }
}

WPShaderValueData BuildEffectWriterTransformData(ParseContext& context,
                                                 const EffectWriterTransformContract& contract) {
    WPShaderValueData data;
    ApplyEffectWriterTransformContract(context, contract, data);
    return data;
}

EffectWriterTransformContract BuildImageEffectFinalCompositeContract(
    const wpscene::WPImageObject& image, bool uses_routed_parent) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = ImageObjectParallaxDepth(image);
    contract.parallax_depth_authored = image.parallaxDepthAuthored;

    if (image.parent != 0 && ! image.attachment.empty()) {
        // A bone-attached image is positioned by the parent puppet before this neutral final
        // composite is drawn. SyncResolvedNodeToWorld() copies that already-parallaxed world
        // matrix into the detached final node, while final-composite data deliberately does not
        // retain the bone binding itself. Applying the child's parsed parallaxDepth again here
        // therefore adds an independent camera translation after the parent translation. Missing
        // parallaxDepth now resolves to the normal scene default, so without this explicit
        // attachment contract a head, eye, or other attached artwork drifts away from its bone.
        // Match the direct-image path: attachments get exactly the parent puppet's parallax.
        contract.suppress_own_model_parallax = true;
    }

    if (uses_routed_parent) {
        // Effect-backed image world nodes deliberately force the authored parent as their parallax
        // anchor because their physical SceneNode is detached and routed only for render order. The
        // private final composite is the actual visible writer, so it must mirror that same anchor
        // rather than re-running compatibility selection against its own synthetic node data. This
        // keeps parented media covers and other routed effect layers moving with their authored
        // parent while the route matrix continues to supply only the raw transform hierarchy.
        contract.parent_transform = ParentTransformContract::ForceParallaxAnchor(image.parent);
    }
    return contract;
}

enum class ImageEffectWriterRole
{
    AuthoredEffectProjection,
    LayerSurfaceProxy,
};

struct ImageEffectMaterialTopology {
    bool                  uses_routed_parent { false };
    ImageEffectWriterRole writer_role { ImageEffectWriterRole::AuthoredEffectProjection };

    bool NeedsLayerSurfaceParentParallax() const {
        return uses_routed_parent && writer_role == ImageEffectWriterRole::LayerSurfaceProxy;
    }
};

ImageEffectWriterRole ResolveImageEffectWriterRole(bool is_compose_layer) {
    if (is_compose_layer) return ImageEffectWriterRole::LayerSurfaceProxy;
    return ImageEffectWriterRole::AuthoredEffectProjection;
}

EffectWriterTransformContract BuildTextEffectFinalCompositeContract(
    const wpscene::WPTextObject& text) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = TextObjectParallaxDepth(text);
    contract.parallax_depth_authored = text.parallaxDepthAuthored;
    if (LayerUsesRoutedParent(text.parent, text.attachment)) {
        // The route matrix contains the authored parent transform but never shader-time mouse
        // parallax. Match the normal text-node contract here: zero-depth text inherits the closest
        // compatible parent parallax anchor, while text with an independent authored depth keeps its
        // own offset. The final composite then applies exactly one parallax contract instead of
        // suppressing the only offset available to effect-backed weekday/date labels.
        contract.parent_transform =
            ParentTransformContract::ParallaxAnchorWhenCompatible(text.parent);
    }
    return contract;
}

EffectWriterTransformContract BuildImageEffectMaterialContract(
    const wpscene::WPImageObject& image, SceneImageEffectLayer& effect_layer,
    const ImageEffectMaterialTopology& topology, bool private_layer_surface_writer) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = ImageObjectParallaxDepth(image);
    contract.parallax_depth_authored = image.parallaxDepthAuthored;
    contract.projection_layer        = &effect_layer;
    contract.binds_puppet_surface    = private_layer_surface_writer;

    if (private_layer_surface_writer) {
        // An animated puppet surface writer rasterizes the skinned mesh through the layer-local
        // source camera, then the neutral final composite places that resolved texture in scene
        // space. Camera parallax belongs exclusively to that final scene-space placement. Applying
        // it during the private puppet draw as well shifts the body inside its texture, while a
        // bone-attached child receives the parent's parallax only through its attachment transform;
        // the two pieces therefore separate as the pointer moves. Keep the private rasterization
        // local so the final composite and every attachment observe one shared parallax transform.
        contract.suppress_own_model_parallax = true;
    }

    if (topology.NeedsLayerSurfaceParentParallax()) {
        // This decision is based on render topology, not on authored parallax values. Compose
        // layers have no authored effect projection that should own a separate child-space parallax
        // result; their resolved screen writer is the layer image itself, merely routed through the
        // image-effect path. Match the no-effect
        // image-layer contract by inheriting the authored parent's parallax anchor, so virtual
        // render-order parents keep routed visual layers locked together.
        //
        // Detached chains with real authored effects intentionally do not enter this branch. Their
        // final writer belongs to the effect pipeline, and the world route matrix already supplies
        // the parent transform. Re-anchoring that authored final writer to the parent changes the
        // effect-chain projection contract and moves layers such as the lantern media cover and
        // audio rings away from their authored local center.
        contract.parent_transform = ParentTransformContract::ForceParallaxAnchor(image.parent);
    }
    return contract;
}

EffectWriterTransformContract BuildTextEffectMaterialContract(
    const wpscene::WPTextObject& text, SceneImageEffectLayer& effect_layer) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = TextObjectParallaxDepth(text);
    contract.parallax_depth_authored = text.parallaxDepthAuthored;
    contract.projection_layer        = &effect_layer;
    if (LayerUsesRoutedParent(text.parent, text.attachment)) {
        // Text effect nodes start as private bridge passes, but ResolveEffect() may turn the last
        // authored effect node into the visible scene-space writer. Parent-routed text already
        // receives the visual parent chain, including parent camera parallax, through the render
        // graph route matrix. Suppressing this node's own model parallax prevents two failure modes:
        // non-zero child text depths drifting inside zero-parallax HUD groups, and zero-depth
        // effect-backed date labels receiving an extra copy of their moving parent's parallax.
        contract.suppress_own_model_parallax = true;
    }
    return contract;
}

// Shared with WPSceneParserModel.cpp (declared in WPSceneParserShared.hpp).
} // namespace

void RegisterLayerSceneState(ParseContext& context, int32_t layer_id, int32_t parent_id,
                             std::string_view attachment, bool visible) {
    if (context.scene == nullptr || layer_id == 0) return;
    context.scene->SetLayerParentBinding(layer_id, parent_id, std::string(attachment));
    context.scene->SetLayerLocalVisibility(layer_id, visible);
}

namespace
{

void RegisterLogicalImageLayer(ParseContext& context, const wpscene::WPImageObject& wpimgobj) {
    auto node = std::make_shared<SceneNode>(Vector3f(wpimgobj.origin.data()),
                                            Vector3f(wpimgobj.scale.data()),
                                            Vector3f(wpimgobj.angles.data()),
                                            wpimgobj.name);
    LoadAlignment(*node, wpimgobj.alignment, { wpimgobj.size[0], wpimgobj.size[1] });
    node->ID() = wpimgobj.id;

    WPShaderValueData node_data;
    node_data.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };
    node_data.parallaxDepthAuthored = wpimgobj.parallaxDepthAuthored;
    ConfigureBoneAttachment(context,
                            wpimgobj.parent,
                            wpimgobj.attachment,
                            Eigen::Affine3f(node->GetLocalTrans().cast<float>()),
                            "image layer",
                            wpimgobj.name,
                            node_data);

    if (LayerUsesRoutedParent(wpimgobj.parent, wpimgobj.attachment)) {
        ConfigureInheritedParentBinding(context, wpimgobj.parent, node_data);
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, wpimgobj.parent, wpimgobj.name, &node_data);
    }

    context.object_nodes[wpimgobj.id] = node;
    context.scene->EnsureSceneObject(wpimgobj.id)
        .SetImageRuntimeState(Scene::ImageLayerRuntimeState {
            .size      = wpimgobj.size,
            .alignment = wpimgobj.alignment,
        });
    context.scene->AddLayerRuntimeNode(wpimgobj.id, node.get());
    context.shader_updater->SetNodeData(node.get(), node_data);
    RegisterLayerSceneState(
        context, wpimgobj.id, wpimgobj.parent, wpimgobj.attachment, wpimgobj.visible);
    context.scene->ApplyLayerVisibility(wpimgobj.id);

    LOG_INFO("SceneObjectMaterialize: mode=image-logical-only id=%d name='%s' image='%s' "
             "fullscreen=%s autosize=%s projectlayer=%s effects=%zu dependency-source=%s",
             wpimgobj.id,
             wpimgobj.name.c_str(),
             wpimgobj.image.c_str(),
             wpimgobj.fullscreen ? "true" : "false",
             wpimgobj.autosize ? "true" : "false",
             wpimgobj.projectlayer ? "true" : "false",
             wpimgobj.effects.size(),
             context.scene != nullptr &&
                     context.scene->IsLayerOffscreenDependencySource(wpimgobj.id)
                 ? "true"
                 : "false");
}

// Shared with WPSceneParserParticle.cpp (declared in WPSceneParserShared.hpp).
} // namespace

namespace
{

struct ResolvedUserShaderValueBinding {
    std::string        user_property_name;
    std::string        material_value_name;
    std::string        gl_uniform_name;
    const ShaderValue* property { nullptr };
    bool               legacy_reversed { false };
};

enum class MaterialValueUniformResolutionKind
{
    ExactAlias,
    UniformName,
    UniformSuffix,
    NormalizedAlias,
    AmbiguousNormalizedAlias,
    Unresolved,
};

struct MaterialValueUniformResolution {
    std::string                        uniform_name;
    std::string                        matched_alias;
    MaterialValueUniformResolutionKind kind {
        MaterialValueUniformResolutionKind::Unresolved
    };

    bool resolved() const noexcept {
        return kind != MaterialValueUniformResolutionKind::Unresolved &&
               kind != MaterialValueUniformResolutionKind::AmbiguousNormalizedAlias;
    }
};

const char* MaterialValueUniformResolutionKindName(MaterialValueUniformResolutionKind kind) {
    switch (kind) {
    case MaterialValueUniformResolutionKind::ExactAlias: return "exact-alias";
    case MaterialValueUniformResolutionKind::UniformName: return "uniform-name";
    case MaterialValueUniformResolutionKind::UniformSuffix: return "uniform-suffix";
    case MaterialValueUniformResolutionKind::NormalizedAlias: return "normalized-alias";
    case MaterialValueUniformResolutionKind::AmbiguousNormalizedAlias:
        return "ambiguous-normalized-alias";
    case MaterialValueUniformResolutionKind::Unresolved: return "unresolved";
    }
    return "unknown";
}

bool IsDirectMaterialValueResolution(MaterialValueUniformResolutionKind kind) {
    return kind == MaterialValueUniformResolutionKind::ExactAlias ||
           kind == MaterialValueUniformResolutionKind::UniformName ||
           kind == MaterialValueUniformResolutionKind::UniformSuffix;
}

std::string NormalizeMaterialValueAlias(std::string_view name) {
    std::string normalized;
    int         parenthetical_depth = 0;
    for (unsigned char raw_ch : name) {
        const char ch = static_cast<char>(raw_ch);
        if (ch == '(') {
            parenthetical_depth++;
            continue;
        }
        if (ch == ')') {
            if (parenthetical_depth > 0) parenthetical_depth--;
            continue;
        }
        if (parenthetical_depth > 0) continue;

        if (std::isalnum(raw_ch)) {
            normalized.push_back(static_cast<char>(std::tolower(raw_ch)));
        }
    }

    // Wallpaper Engine sometimes serializes constants by editor label ("Texture parallax depth")
    // while the shader metadata only exposes numbered material keys ("4textureParallaxDepth").
    // Dropping only leading digits lets those two forms meet without treating unrelated numeric
    // suffixes as equivalent.
    const auto first_non_digit =
        std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch) {
            return ! std::isdigit(ch);
        });
    normalized.erase(normalized.begin(), first_non_digit);
    return normalized;
}

MaterialValueUniformResolution
ResolveMaterialValueUniform(const WPShaderInfo& info, std::string_view material_value_name,
                            bool allow_normalized_alias) {
    const std::string material_value_key(material_value_name);
    if (const auto alias_it = info.alias.find(material_value_key); alias_it != info.alias.end()) {
        return {
            .uniform_name  = alias_it->second,
            .matched_alias = alias_it->first,
            .kind          = MaterialValueUniformResolutionKind::ExactAlias,
        };
    }

    for (const auto& [alias_name, uniform_name] : info.alias) {
        if (uniform_name == material_value_key) {
            return {
                .uniform_name  = uniform_name,
                .matched_alias = alias_name,
                .kind          = MaterialValueUniformResolutionKind::UniformName,
            };
        }

        // Some shader metadata stores material aliases like `color1`, while the parsed GLSL
        // uniform is named `g_Color1`. Keep this suffix match so user-facing project properties
        // can still target old stock shaders whose material JSON uses the shorter alias instead
        // of the final GLSL symbol.
        if (uniform_name.size() > 2 && uniform_name.substr(2) == material_value_key) {
            return {
                .uniform_name  = uniform_name,
                .matched_alias = alias_name,
                .kind          = MaterialValueUniformResolutionKind::UniformSuffix,
            };
        }
    }

    if (! allow_normalized_alias) {
        return {
            .uniform_name = material_value_key,
            .kind         = MaterialValueUniformResolutionKind::Unresolved,
        };
    }

    const auto normalized_key = NormalizeMaterialValueAlias(material_value_key);
    if (normalized_key.empty()) {
        return {
            .uniform_name = material_value_key,
            .kind         = MaterialValueUniformResolutionKind::Unresolved,
        };
    }

    std::optional<MaterialValueUniformResolution> candidate;
    for (const auto& [alias_name, uniform_name] : info.alias) {
        if (NormalizeMaterialValueAlias(alias_name) != normalized_key) continue;

        if (candidate.has_value() && candidate->uniform_name != uniform_name) {
            return {
                .uniform_name  = material_value_key,
                .matched_alias = alias_name,
                .kind = MaterialValueUniformResolutionKind::AmbiguousNormalizedAlias,
            };
        }

        candidate = MaterialValueUniformResolution {
            .uniform_name  = uniform_name,
            .matched_alias = alias_name,
            .kind          = MaterialValueUniformResolutionKind::NormalizedAlias,
        };
    }

    if (candidate.has_value()) return *candidate;
    return {
        .uniform_name = material_value_key,
        .kind         = MaterialValueUniformResolutionKind::Unresolved,
    };
}

std::string ResolveMaterialValueUniformName(const WPShaderInfo& info,
                                            const std::string&  material_value_name) {
    const auto resolution = ResolveMaterialValueUniform(info, material_value_name, true);
    return resolution.resolved() ? resolution.uniform_name : material_value_name;
}

void ApplyResolvedConstvalue(SceneMaterial& material, const std::string& material_value_name,
                             const std::vector<float>&                 value,
                             const MaterialValueUniformResolution&     resolution) {
    if (! resolution.resolved()) return;
    if (resolution.kind == MaterialValueUniformResolutionKind::NormalizedAlias) {
        LOG_INFO("ShaderValueAliasFallback: material-value='%s' alias='%s' uniform='%s'",
                 material_value_name.c_str(),
                 resolution.matched_alias.c_str(),
                 resolution.uniform_name.c_str());
    }

    material.customShader.constValues[resolution.uniform_name] =
        ClampParserOpacityUniformValue(resolution.uniform_name, ShaderValue(value));
}

// Shared with WPSceneParserModel.cpp (declared in WPSceneParserShared.hpp).
} // namespace

void LoadConstvalue(SceneMaterial& material, const wpscene::WPMaterial& wpmat,
                    const WPShaderInfo& info) {
    // Apply exact authored material keys before display-name fallbacks. Some Wallpaper Engine
    // projects serialize both forms in one pass; the display-name value is the editor-visible
    // override and must be allowed to replace the internal default key deterministically.
    std::unordered_set<std::string> exact_uniform_names;
    for (const auto& [name, value] : wpmat.constantshadervalues) {
        const auto resolution = ResolveMaterialValueUniform(info, name, false);
        if (! resolution.resolved()) continue;
        ApplyResolvedConstvalue(material, name, value, resolution);
        exact_uniform_names.insert(resolution.uniform_name);
    }

    for (const auto& [name, value] : wpmat.constantshadervalues) {
        const auto direct_resolution = ResolveMaterialValueUniform(info, name, false);
        if (direct_resolution.resolved() &&
            IsDirectMaterialValueResolution(direct_resolution.kind)) {
            continue;
        }

        const auto resolution = ResolveMaterialValueUniform(info, name, true);
        if (resolution.resolved()) {
            // Model importer leftovers ("Alpha", "Color") normalize onto the same uniforms as the
            // authored lowercase keys. An exact authored key is the value the editor exported, so
            // a normalized fallback may fill gaps but never override it; planet atmosphere shells
            // authored as {Alpha: 1, alpha: 0.25} must stay translucent.
            if (exact_uniform_names.count(resolution.uniform_name) != 0) {
                LOG_INFO("ShaderValueAliasSkip: material-value='%s' uniform='%s' "
                         "reason=exact-key-owns-uniform",
                         name.c_str(),
                         resolution.uniform_name.c_str());
                continue;
            }
            ApplyResolvedConstvalue(material, name, value, resolution);
            continue;
        }

        LOG_WARN("ShaderValue: material-value='%s' skipped reason=%s",
                 name.c_str(),
                 MaterialValueUniformResolutionKindName(resolution.kind));
    }
}

namespace
{

std::vector<ResolvedUserShaderValueBinding>
ResolveUserShaderValueBindings(const wpscene::WPMaterial& wpmat, const WPShaderInfo& info,
                               const UserPropertyMap* user_properties, bool log_missing) {
    std::vector<ResolvedUserShaderValueBinding> bindings;
    if (user_properties == nullptr) return bindings;

    bindings.reserve(wpmat.usershadervalues.size());
    for (const auto& us : wpmat.usershadervalues) {
        // Wallpaper Engine writes `usershadervalues` as
        // `{ "<project user property>": "<shader material value>" }`. Eagle Flag is a compact
        // example: `schemecolor -> color1`, `flagcolor1 -> color2`, and `flagcolor2 -> color3`.
        // Looking up the value side as a user property misses the authored colors and leaves the
        // shader on its black/white defaults, which makes the red and green flag regions vanish.
        std::string user_property_name  = us.first;
        std::string material_value_name = us.second;
        bool        legacy_reversed     = false;
        const auto* property = LookupUserPropertyShaderValue(user_properties, user_property_name);
        if (property == nullptr) {
            // Older local builds interpreted the mapping in the opposite direction. This fallback
            // keeps any locally-authored scenes that accidentally depended on that reversed
            // behavior visible, while logging the mismatch so the material JSON can be fixed.
            const auto* legacy_property =
                LookupUserPropertyShaderValue(user_properties, material_value_name);
            if (legacy_property != nullptr) {
                legacy_reversed = true;
                std::swap(user_property_name, material_value_name);
                property = legacy_property;
            } else {
                if (log_missing) {
                    LOG_INFO("UserShaderValue: property '%s' not provided for material value '%s'",
                             user_property_name.c_str(),
                             material_value_name.c_str());
                }
                continue;
            }
        }

        const auto gl_uniform_name = ResolveMaterialValueUniformName(info, material_value_name);
        bindings.push_back(ResolvedUserShaderValueBinding {
            .user_property_name  = std::move(user_property_name),
            .material_value_name = std::move(material_value_name),
            .gl_uniform_name     = gl_uniform_name,
            .property            = property,
            .legacy_reversed     = legacy_reversed,
        });
    }

    return bindings;
}

WPDynamicValue::Type DynamicTypeForShaderValue(const ShaderValue& value) {
    switch (value.size()) {
    case 2: return WPDynamicValue::Type::Float2;
    case 3: return WPDynamicValue::Type::Float3;
    case 4: return WPDynamicValue::Type::Float4;
    case 1: return WPDynamicValue::Type::Float;
    default: return WPDynamicValue::Type::FloatVector;
    }
}

bool SceneMaterialHasUniform(const SceneMaterial& material, std::string_view uniform_name) {
    const std::string uniform_key(uniform_name);
    if (material.customShader.constValues.count(uniform_key) != 0) return true;
    return material.customShader.shader != nullptr &&
           material.customShader.shader->default_uniforms.count(uniform_key) != 0;
}

// Shared with WPSceneParserModel.cpp (declared in WPSceneParserShared.hpp).
} // namespace

void RegisterUserShaderValueBindings(ParseContext& context, const wpscene::WPMaterial& wpmat,
                                     const WPShaderInfo& info, SceneNode* node, int32_t object_id,
                                     std::string_view object_name) {
    if (context.scene == nullptr || node == nullptr || node->Mesh() == nullptr ||
        node->Mesh()->Material() == nullptr) {
        return;
    }

    for (const auto& binding :
         ResolveUserShaderValueBindings(wpmat, info, context.user_properties, false)) {
        if (binding.property == nullptr) continue;

        const auto value_type = DynamicTypeForShaderValue(*binding.property);
        auto       base_value =
            WPDynamicValue::FromUserPropertyValue(UserPropertyValue(*binding.property), value_type)
                .value_or(WPDynamicValue {});

        WPUserSetting setting;
        setting.value    = base_value;
        setting.property = UserPropertyBinding {
            .name      = binding.user_property_name,
            .condition = {},
        };

        // `usershadervalues` bindings are not layer properties: they write directly into the
        // material uniform map. Registering them after the material has been attached makes 2D
        // layers and 3D model chunks share the same live-update contract: the dispatcher resolves
        // node->Mesh()->Material() immediately and writes the same GLSL uniform that the cold parse
        // resolved from shader metadata.
        context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
            .object_id     = object_id,
            .object_name   = std::string(object_name),
            .property_name = binding.gl_uniform_name,
            .node          = node,
            .target_kind   = WPSceneScriptTargetKind::MaterialUniform,
            .target_index  = 0,
            .value_type    = value_type,
            .base_value    = base_value,
            .setting       = std::move(setting),
        });

        LOG_INFO("UserShaderValueRegister: layer=%d name='%.*s' user-property='%s' "
                 "material-value='%s' uniform='%s' components=%zu legacy-reversed=%s",
                 object_id,
                 static_cast<int>(object_name.size()),
                 object_name.data(),
                 binding.user_property_name.c_str(),
                 binding.material_value_name.c_str(),
                 binding.gl_uniform_name.c_str(),
                 binding.property->size(),
                 binding.legacy_reversed ? "true" : "false");
    }
}

// Shared with WPSceneParserModel.cpp (declared in WPSceneParserShared.hpp): model chunk
// materials carry the same authored constant bindings as effect pass materials, so their
// script/user/animation constants register through this one dispatcher.
void RegisterConstantShaderValueBindings(ParseContext& context, const wpscene::WPMaterial& wpmat,
                                         const WPShaderInfo& info, SceneNode* node,
                                         int32_t object_id, std::string_view object_name,
                                         int32_t effect_id, int32_t effect_index,
                                         usize material_index) {
    if (context.scene == nullptr || node == nullptr || node->Mesh() == nullptr ||
        node->Mesh()->Material() == nullptr) {
        return;
    }

    for (const auto& [material_value_name, binding] : wpmat.constantshadervaluebindings) {
        const auto& setting       = binding.setting;
        const bool  has_animation = binding.animation != nullptr && binding.animation->valid();
        if (! setting.hasUserBinding() && ! setting.hasScript() && ! has_animation) continue;

        const auto  resolution      = ResolveMaterialValueUniform(info, material_value_name, true);
        const auto& gl_uniform_name = resolution.uniform_name;
        if (! SceneMaterialHasUniform(*node->Mesh()->Material(), gl_uniform_name)) {
            LOG_INFO("ConstantShaderValueRegister: layer=%d effect-id=%d effect-index=%d "
                     "material-index=%zu material-value='%s' unresolved uniform='%s' reason=%s",
                     object_id,
                     effect_id,
                     effect_index,
                     material_index,
                     material_value_name.c_str(),
                     gl_uniform_name.c_str(),
                     MaterialValueUniformResolutionKindName(resolution.kind));
            continue;
        }

        // Effect pass constants are parsed into SceneMaterial::constValues for cold start, but
        // dynamic constants also need a live target on the concrete pass node. User bindings and
        // scripts both reuse the MaterialUniform dispatcher so album-art color scripts can update
        // Gradient Color uniforms without rebuilding the post-process chain.
        WPSceneScriptRegistration registration {
            .object_id     = object_id,
            .object_name   = std::string(object_name),
            .property_name = gl_uniform_name,
            .node          = node,
            .target_kind   = WPSceneScriptTargetKind::MaterialUniform,
            .target_index  = static_cast<uint32_t>(material_index),
            .target_id     = effect_id,
            .value_type    = setting.value.type(),
            .base_value    = setting.value,
            .setting       = setting,
        };

        std::string registration_kind;
        if (has_animation) {
            // Material-uniform animations are registered against the same target descriptor as
            // their sibling script/user binding. This keeps thisObject.getAnimation() resolvable
            // for effect scripts that replay cover-transition timelines during media changes.
            auto animation_registration      = registration;
            animation_registration.animation = binding.animation;
            context.scene->propertyAnimationRegistrations.push_back(
                std::move(animation_registration));
            registration_kind = "animation";
        }
        if (setting.hasScript()) {
            context.scene->scriptRegistrations.push_back(registration);
            registration_kind += registration_kind.empty() ? "script" : "+script";
        } else if (setting.hasUserBinding()) {
            context.scene->bindingRegistrations.push_back(registration);
            registration_kind += registration_kind.empty() ? "user" : "+user";
        }

        LOG_INFO("ConstantShaderValueRegister: layer=%d name='%.*s' effect-id=%d "
                 "effect-index=%d material-index=%zu kind=%s user-property='%s' "
                 "material-value='%s' uniform='%s' value-type=%s",
                 object_id,
                 static_cast<int>(object_name.size()),
                 object_name.data(),
                 effect_id,
                 effect_index,
                 material_index,
                 registration_kind.c_str(),
                 setting.property.has_value() ? setting.property->name.c_str() : "",
                 material_value_name.c_str(),
                 gl_uniform_name.c_str(),
                 DynamicValueTypeName(setting.value.type()));
    }
}

void LoadUserShaderValue(SceneMaterial& material, const wpscene::WPMaterial& wpmat,
                         const WPShaderInfo& info, const UserPropertyMap* user_properties) {
    for (const auto& binding : ResolveUserShaderValueBindings(wpmat, info, user_properties, true)) {
        if (binding.legacy_reversed) {
            LOG_INFO("UserShaderValue: legacy reversed mapping user-property '%s' -> material "
                     "value '%s'",
                     binding.user_property_name.c_str(),
                     binding.material_value_name.c_str());
        }

        LOG_INFO("UserShaderValue: property '%s' -> material value '%s' -> uniform '%s' (%zu)",
                 binding.user_property_name.c_str(),
                 binding.material_value_name.c_str(),
                 binding.gl_uniform_name.c_str(),
                 binding.property != nullptr ? binding.property->size() : 0);
        if (binding.property != nullptr)
            material.customShader.constValues[binding.gl_uniform_name] =
                ClampParserOpacityUniformValue(binding.gl_uniform_name, *binding.property);
    }
}

namespace
{

// parse

bool ParseModelCameraPathKeyframe(const nlohmann::json& json, Scene::CameraPathKeyframe& keyframe) {
    if (! json.is_object()) return false;
    GET_JSON_NAME_VALUE_NOWARN(json, "timestamp", keyframe.timestamp);
    GET_JSON_NAME_VALUE_NOWARN(json, "eye", keyframe.eye);
    GET_JSON_NAME_VALUE_NOWARN(json, "center", keyframe.center);
    GET_JSON_NAME_VALUE_NOWARN(json, "up", keyframe.up);
    return true;
}

void LoadModelCameraPaths(ParseContext& context, const wpscene::WPSceneCamera& authored_camera) {
    auto& scene = *context.scene;
    scene.modelCameraPathSegments.clear();
    scene.modelCameraPathEnabled       = false;
    scene.activeModelCameraPathSegment = -1;

    if (authored_camera.paths.empty()) return;

    for (const auto& relative_path : authored_camera.paths) {
        const std::string asset_path = "/assets/" + relative_path;
        nlohmann::json    camera_path_json;
        if (! context.vfs->Contains(asset_path) ||
            ! PARSE_JSON(fs::GetFileContent(*context.vfs, asset_path), camera_path_json)) {
            LOG_ERROR("Scene3DModelCameraPath: failed to read path='%s'", relative_path.c_str());
            continue;
        }
        if (! camera_path_json.is_object() || ! camera_path_json.contains("paths") ||
            ! camera_path_json.at("paths").is_array()) {
            LOG_ERROR("Scene3DModelCameraPath: path file has no paths array path='%s'",
                      relative_path.c_str());
            continue;
        }

        for (const auto& segment_json : camera_path_json.at("paths")) {
            if (! segment_json.is_object()) continue;

            Scene::CameraPathSegment segment;
            GET_JSON_NAME_VALUE_NOWARN(segment_json, "name", segment.name);
            GET_JSON_NAME_VALUE_NOWARN(segment_json, "duration", segment.duration);
            if (segment_json.contains("transforms") && segment_json.at("transforms").is_array()) {
                for (const auto& keyframe_json : segment_json.at("transforms")) {
                    Scene::CameraPathKeyframe keyframe;
                    if (ParseModelCameraPathKeyframe(keyframe_json, keyframe)) {
                        segment.keyframes.push_back(keyframe);
                    }
                }
            }

            if (segment.duration <= 0.0 && ! segment.keyframes.empty()) {
                segment.duration = segment.keyframes.back().timestamp;
            }
            if (segment.duration <= 0.0 || segment.keyframes.empty()) {
                LOG_ERROR("Scene3DModelCameraPath: ignore empty segment path='%s' duration=%.3f "
                          "keyframes=%zu",
                          relative_path.c_str(),
                          segment.duration,
                          segment.keyframes.size());
                continue;
            }

            LOG_INFO("Scene3DModelCameraPath: segment parsed path='%s' index=%zu duration=%.3f "
                     "keyframes=%zu",
                     relative_path.c_str(),
                     scene.modelCameraPathSegments.size(),
                     segment.duration,
                     segment.keyframes.size());
            scene.modelCameraPathSegments.push_back(std::move(segment));
        }
    }

    scene.modelCameraPathEnabled = ! scene.modelCameraPathSegments.empty();
    if (scene.modelCameraPathEnabled) {
        const auto& first     = scene.modelCameraPathSegments.front().keyframes.front();
        auto        camera_it = scene.cameras.find(std::string(kSceneModelPerspectiveCameraName));
        if (camera_it != scene.cameras.end() && camera_it->second) {
            // Seed frame zero on the model-only camera. The legacy 2D `global_perspective` camera
            // is intentionally not touched here, because 2D particle scenes depend on its old
            // screen center transform.
            camera_it->second->SetExplicitView(
                Vector3d(first.eye[0], first.eye[1], first.eye[2]),
                Vector3d(first.center[0], first.center[1], first.center[2]),
                Vector3d(first.up[0], first.up[1], first.up[2]));
        }
        LOG_INFO("Scene3DModelCameraPath: enabled segments=%zu first-eye=[%.3f, %.3f, %.3f] "
                 "first-center=[%.3f, %.3f, %.3f]",
                 scene.modelCameraPathSegments.size(),
                 first.eye[0],
                 first.eye[1],
                 first.eye[2],
                 first.center[0],
                 first.center[1],
                 first.center[2]);
    }
}

void ParseCamera(ParseContext& context, const wpscene::WPScene& scene_config) {
    auto&       scene   = *context.scene;
    const auto& general = scene_config.general;
    // effect camera
    scene.cameras["effect"]    = std::make_shared<SceneCamera>(2, 2, -1.0f, 1.0f);
    context.effect_camera_node = std::make_shared<SceneNode>(); // at 0,0,0
    scene.cameras.at("effect")->AttatchNode(context.effect_camera_node);
    scene.sceneGraph->AppendChild(context.effect_camera_node);

    // global camera
    scene.cameras["global"] = std::make_shared<SceneCamera>((context.ortho_w / (i32)general.zoom),
                                                            (context.ortho_h / (i32)general.zoom),
                                                            -5000.0f,
                                                            5000.0f);
    scene.activeCamera      = scene.cameras.at("global").get();
    Vector3f cori { (float)context.ortho_w / 2.0f, (float)context.ortho_h / 2.0f, 0 },
        cscale { 1.0f, 1.0f, 1.0f }, cangle(Vector3f::Zero());

    context.global_camera_node = std::make_shared<SceneNode>(cori, cscale, cangle);
    scene.activeCamera->AttatchNode(context.global_camera_node);
    scene.sceneGraph->AppendChild(context.global_camera_node);
    scene.defaultGlobalCameraNode = context.global_camera_node;
    scene.defaultGlobalCameraZoom = std::max(0.0001f, general.zoom);

    scene.cameras["global_perspective"] =
        std::make_shared<SceneCamera>((float)context.ortho_w / (float)context.ortho_h,
                                      general.nearz,
                                      general.farz,
                                      algorism::ResolvePerspectiveFov(scene.perspectiveOverrideFov,
                                                                      context.ortho_h));

    Vector3f cperori                       = cori;
    cperori[2]                             = 1000.0f;
    context.global_perspective_camera_node = std::make_shared<SceneNode>(cperori, cscale, cangle);
    scene.cameras["global_perspective"]->AttatchNode(context.global_perspective_camera_node);
    scene.sceneGraph->AppendChild(context.global_perspective_camera_node);

    const Vector3d eye(
        scene_config.camera.eye[0], scene_config.camera.eye[1], scene_config.camera.eye[2]);
    const Vector3d center(scene_config.camera.center[0],
                          scene_config.camera.center[1],
                          scene_config.camera.center[2]);
    const Vector3d up(
        scene_config.camera.up[0], scene_config.camera.up[1], scene_config.camera.up[2]);
    scene.cameras[std::string(kSceneModelPerspectiveCameraName)] = std::make_shared<SceneCamera>(
        (float)context.ortho_w / (float)context.ortho_h, general.nearz, general.farz, general.fov);
    auto model_camera_node = std::make_shared<SceneNode>();
    // 3D model support must not reuse `global_perspective`: existing 2D particle systems and
    // camera-layer scenes already depend on that camera's historical centered-at-screen transform.
    // The authored scene camera is therefore installed under a model-only name and consumed only by
    // WPModelObject materialization and model camera-path playback.
    scene.cameras[std::string(kSceneModelPerspectiveCameraName)]->AttatchNode(model_camera_node);
    scene.cameras[std::string(kSceneModelPerspectiveCameraName)]->SetExplicitView(eye, center, up);
    scene.modelPerspectiveCameraName = std::string(kSceneModelPerspectiveCameraName);
    scene.sceneGraph->AppendChild(model_camera_node);
    LoadModelCameraPaths(context, scene_config.camera);

    if (! general.isOrtho) {
        // A scene without an orthogonal projection renders every layer kind through the one
        // authored perspective view: image quads, text, and models share the same eye and
        // projection instead of splitting between a canvas-sized orthographic camera and a
        // model-only camera. A camera layer, when present, re-targets this view per frame.
        scene.activeCamera = scene.cameras.at(std::string(kSceneModelPerspectiveCameraName)).get();
        LOG_INFO("ScenePerspectiveView: 3d scene routes all layers through camera='%s' "
                 "fov=%.3f near=%.5f far=%.1f eye=[%.5f, %.5f, %.5f] center=[%.5f, %.5f, %.5f]",
                 scene.modelPerspectiveCameraName.c_str(),
                 general.fov,
                 general.nearz,
                 general.farz,
                 eye.x(),
                 eye.y(),
                 eye.z(),
                 center.x(),
                 center.y(),
                 center.z());
    }
}

void InitContext(ParseContext& context, fs::VFS& vfs, wpscene::WPScene& sc,
                 std::string_view scene_id) {
    context.scene     = std::make_shared<Scene>();
    context.vfs       = &vfs;
    auto& scene       = *context.scene;
    scene.imageParser = std::make_unique<WPSyntheticImageParser>(
        std::make_unique<WPTexImageParser>(&vfs, std::string(scene_id)));
    scene.paritileSys->gener = std::make_unique<WPParticleRawGener>();
    scene.shaderValueUpdater = std::make_unique<WPShaderValueUpdater>(&scene);
    GenCardMesh(scene.default_effect_mesh, { 2, 2 });
    context.shader_updater = static_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());

    scene.clearColor                   = sc.general.clearcolor;
    scene.ambientColor                 = sc.general.ambientcolor;
    scene.skylightColor                = sc.general.skylightcolor;
    scene.bloom.enabled                = sc.general.bloom;
    scene.bloom.strength               = sc.general.bloomstrength;
    scene.bloom.threshold              = sc.general.bloomthreshold;
    scene.bloom.tint                   = sc.general.bloomtint;
    scene.bloom.hdr                    = sc.general.hdr;
    scene.bloom.hdrStrength            = sc.general.bloomhdrstrength;
    scene.bloom.hdrThreshold           = sc.general.bloomhdrthreshold;
    scene.bloom.hdrScatter             = sc.general.bloomhdrscatter;
    scene.bloom.hdrFeather             = sc.general.bloomhdrfeather;
    scene.bloom.hdrIterations          = sc.general.bloomhdriterations;
    scene.cameraParallax               = sc.general.cameraparallax;
    scene.cameraParallaxAmount         = sc.general.cameraparallaxamount;
    scene.cameraParallaxDelay          = sc.general.cameraparallaxdelay;
    scene.cameraParallaxMouseInfluence = sc.general.cameraparallaxmouseinfluence;
    scene.cameraOrthographic           = sc.general.isOrtho;
    scene.perspectiveOverrideFov       = sc.general.perspectiveoverridefov;
    scene.cameraShake                  = sc.general.camerashake;
    scene.cameraShakeAmplitude         = sc.general.camerashakeamplitude;
    scene.cameraShakeRoughness         = sc.general.camerashakeroughness;
    scene.cameraShakeSpeed             = sc.general.camerashakespeed;
    scene.ortho[0]                     = sc.general.orthogonalprojection.width;
    scene.ortho[1]                     = sc.general.orthogonalprojection.height;
    context.ortho_w                    = scene.ortho[0];
    context.ortho_h                    = scene.ortho[1];

    PopulateGlobalBaseUniforms(context, scene);

    {
        WPCameraParallax cam_para;
        cam_para.enable         = sc.general.cameraparallax;
        cam_para.amount         = sc.general.cameraparallaxamount;
        cam_para.delay          = sc.general.cameraparallaxdelay;
        cam_para.mouseinfluence = sc.general.cameraparallaxmouseinfluence;
        context.shader_updater->SetCameraParallax(cam_para);
    }
    if (scene.cameraShake) {
        LOG_INFO("SceneCameraShake: enabled=true amplitude=%.3f roughness=%.3f speed=%.3f "
                 "ortho=%s height=%d",
                 scene.cameraShakeAmplitude,
                 scene.cameraShakeRoughness,
                 scene.cameraShakeSpeed,
                 scene.cameraOrthographic ? "true" : "false",
                 scene.ortho[1]);
    }
}

void ParseImageObj(ParseContext& context, wpscene::WPImageObject& img_obj) {
    auto& wpimgobj = img_obj;

    auto& vfs = *context.vfs;

    const auto register_logical_only_layer = [&]() {
        RegisterLogicalImageLayer(context, wpimgobj);
    };

    const int32_t count_eff = static_cast<int32_t>(wpimgobj.effects.size());
    const bool hasAuthoredEffect = count_eff > 0;
    bool       isCompose         = (wpimgobj.image == "models/util/composelayer.json");
    const bool isProjectLayer =
        wpimgobj.projectlayer || wpimgobj.image == "models/util/projectlayer.json";
    const bool is_offscreen_dependency_source =
        context.scene != nullptr &&
        context.scene->IsLayerOffscreenDependencySource(wpimgobj.id);
    const bool has_shader_color_blend = UsesShaderColorBlendMode(wpimgobj.colorBlendMode);
    // Wallpaper Engine `dependencies` expose a layer through `_rt_imageLayerComposite_<id>`
    // even when the source layer has no authored effects. Such layers still need a private source
    // render target because the visible consumer samples that source while the layer itself remains
    // hidden in the main scene. Treating dependency-only image layers as effect-backed sources lets
    // the existing effect camera/ping-pong path materialize the raw image or mask without drawing
    // it directly into `_rt_default`.
    bool hasEffect =
        hasAuthoredEffect || has_shader_color_blend || is_offscreen_dependency_source;
    // Detached effect world nodes still need to inherit the parent transform even though they
    // cannot become real scene-graph children of that parent. SceneScript/property-animation
    // also needs a dedicated logical/world node for image layers with effects, otherwise runtime
    // transform updates move the offscreen source quad out of its effect camera and the final
    // output turns blank.
    const bool uses_routed_parent = LayerUsesRoutedParent(wpimgobj.parent, wpimgobj.attachment);
    bool use_detached_effect_world_node = hasEffect && ! isCompose;
    const std::array<float, 2> effect_source_size =
        wpimgobj.effectSourceSize[0] > 0.0f && wpimgobj.effectSourceSize[1] > 0.0f
            ? wpimgobj.effectSourceSize
            : wpimgobj.size;
    // Effect chains allocate fixed-size private targets. A fullscreen layer's source texture is
    // the output framebuffer, so its targets take that framebuffer's pixel size at load. Layers
    // whose effect source size is already a resolved pixel extent (direct-draw shapes) use it
    // as-is. Otherwise, perspective scenes rasterize effect chains at canvas density: a unit-space
    // layer's authored size would allocate a private target of a few texels, and procedural
    // detail such as orbit lines or atmosphere rims cannot survive that sampling. The private
    // target keeps the authored aspect ratio, because effect shaders derive their own aspect
    // correction from the target resolution; only the pixel density rises toward the canvas.
    const std::array<float, 2> effect_target_resolution = [&] {
        if (wpimgobj.fullscreen) {
            return OutputFramebufferEffectTargetSize(context);
        }
        if (wpimgobj.effectSourceSizeIsPixelExtent) {
            return effect_source_size;
        }
        if (context.scene == nullptr || context.scene->cameraOrthographic) {
            return effect_source_size;
        }
        const float source_w = std::max(effect_source_size[0], 1.0f);
        const float source_h = std::max(effect_source_size[1], 1.0f);
        const float density  = std::max(
            1.0f,
            std::min(static_cast<float>(context.ortho_w) / source_w,
                      static_cast<float>(context.ortho_h) / source_h));
        return std::array<float, 2> { std::ceil(source_w * density),
                                      std::ceil(source_h * density) };
    }();
    // skip no effect fullscreen layer
    if (! hasEffect && wpimgobj.fullscreen) {
        register_logical_only_layer();
        return;
    }

    const bool hasAuthoredPuppet = ! wpimgobj.puppet.empty();
    // No-effect compose/project layers are logical framebuffer helpers. Drawing them as regular
    // image meshes can sample `_rt_default` and write it back through the scene camera, which
    // applies a second projection to the already-composited frame on non-authored output aspects.
    if (! hasEffect && (isCompose || isProjectLayer) && ! is_offscreen_dependency_source) {
        register_logical_only_layer();
        return;
    }

    std::unique_ptr<WPMdl> puppet;
    if (hasAuthoredPuppet) {
        puppet = std::make_unique<WPMdl>();
        if (! WPMdlParser::Parse(wpimgobj.puppet, vfs, *puppet)) {
            LOG_ERROR("parse puppet failed: %s", wpimgobj.puppet.c_str());
            puppet = nullptr;
        } else if (puppet->kind == WPMdl::MeshKind::Puppet &&
                   (puppet->puppet == nullptr || puppet->puppet->bones.empty())) {
            LOG_ERROR("puppet has no bones: %s", wpimgobj.puppet.c_str());
            puppet = nullptr;
        }
    }
    if (puppet != nullptr) {
        // Puppet clipping masks are independent imported textures referenced by MDLV0022+
        // metadata rather than by the visible material JSON. Register them with the same scene
        // texture contract as authored material slots so structural dynamic preparation can stage
        // their bytes before the masked mesh becomes executable.
        for (const auto& mask : puppet->masks) {
            if (context.scene->textures.count(mask.material) != 0) continue;
            const auto& header = context.scene->imageParser->ParseHeader(mask.material);
            RegisterSceneTextureFromHeader(*context.scene, mask.material, header);
        }
    }
    const bool hasAnimatedPuppetMesh =
        puppet != nullptr && puppet->kind == WPMdl::MeshKind::Puppet && puppet->puppet != nullptr;
    const bool hasStaticImageMesh =
        puppet != nullptr && puppet->kind == WPMdl::MeshKind::StaticImage;

    // wpimgobj.origin[1] = context.ortho_h - wpimgobj.origin[1];
    auto spWorldNode = std::make_shared<SceneNode>(Vector3f(wpimgobj.origin.data()),
                                                   Vector3f(wpimgobj.scale.data()),
                                                   Vector3f(wpimgobj.angles.data()),
                                                   wpimgobj.name);
    LoadAlignment(*spWorldNode, wpimgobj.alignment, { wpimgobj.size[0], wpimgobj.size[1] });
    spWorldNode->ID() = wpimgobj.id;
    auto spImgNode = use_detached_effect_world_node ? std::make_shared<SceneNode>() : spWorldNode;
    if (use_detached_effect_world_node) {
        // The detached node is this layer's private-camera source phase, not a second layer
        // identity. Name it like the other phase nodes so the authored name stays unique to the
        // world node and graph logs distinguish the two.
        spImgNode->SetName(wpimgobj.name + "::__hanabi_effect_source");
    }
    spImgNode->ID() = wpimgobj.id;

    SceneMaterial     material;
    WPShaderValueData svData;
    WPShaderValueData worldNodeData;
    TextureSample     source_sampler;
    std::string       primary_source_texture;
    WPPuppetLayer     shared_puppet_pose;
    if (hasAnimatedPuppetMesh) {
        shared_puppet_pose = WPPuppetLayer(puppet->puppet);
        shared_puppet_pose.prepared(wpimgobj.puppet_layers);
    }

    ShaderValueMap baseConstSvs = context.global_base_uniforms;
    const float    initial_alpha = ClampParserOpacityScalar(wpimgobj.alpha);
    WPShaderInfo   shaderInfo;
    {
        if (! hasEffect) {
            svData.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };
            svData.parallaxDepthAuthored = wpimgobj.parallaxDepthAuthored;
            if (hasAnimatedPuppetMesh) {
                WPMdlParser::AddPuppetShaderInfo(shaderInfo, *puppet);
            }
        }

        baseConstSvs["g_Color4"] = std::array<float, 4> {
            wpimgobj.color[0], wpimgobj.color[1], wpimgobj.color[2], initial_alpha
        };
        baseConstSvs["g_Color"] =
            std::array<float, 3> { wpimgobj.color[0], wpimgobj.color[1], wpimgobj.color[2] };
        baseConstSvs["g_Alpha"]      = initial_alpha;
        baseConstSvs["g_UserAlpha"]  = initial_alpha;
        baseConstSvs["g_Brightness"] = wpimgobj.brightness;

        shaderInfo.baseConstSvs = baseConstSvs;

        if (! LoadMaterial(vfs,
                           wpimgobj.material,
                           context.scene.get(),
                           spImgNode.get(),
                           &material,
                           &svData,
                           context.user_properties,
                           &shaderInfo)) {
            LOG_ERROR("load imageobj '%s' material faild", wpimgobj.name.c_str());
            return;
        };
        LoadConstvalue(material, wpimgobj.material, shaderInfo);
        LoadUserShaderValue(material, wpimgobj.material, shaderInfo, context.user_properties);
        source_sampler = ResolvePrimaryMaterialSampler(*context.scene, material);
        if (!material.textures.empty()) primary_source_texture = material.textures.front();
    }

    // mesh
    SceneMesh effct_final_mesh {};
    auto      spMesh = std::make_shared<SceneMesh>();
    auto&     mesh   = *spMesh;

    {
        const bool primary_texture_is_sprite =
            PrimaryMaterialTextureIsSprite(*context.scene, material);
        std::array<float, 2> mapRate { 1.0f, 1.0f };
        if (! wpimgobj.nopadding &&
            exists(material.customShader.constValues, WE_GLTEX_RESOLUTION_NAMES[0])) {
            const auto& r = material.customShader.constValues.at(WE_GLTEX_RESOLUTION_NAMES[0]);
            const std::array<float, 2> padded_map_rate { r[2] / r[0], r[3] / r[1] };

            /*
             * Ordinary padded images sample the card UV directly, so their base coordinates must
             * be cropped to the logical content extent. Sprite frames already encode rotation and
             * translation normalized against the physical mip0 atlas. Cropping the card before
             * that frame transform applies the same content-to-physical ratio twice and stretches
             * a partial frame over the authored layer. Keep sprite cards in frame-local [0, 1]
             * space while preserving the physical/content resolution contract for shader users.
             */
            if (primary_texture_is_sprite) {
                if (padded_map_rate[0] != 1.0f || padded_map_rate[1] != 1.0f) {
                    LOG_INFO("SceneImageCardUvContract: layer=%d name='%s' texture='%s' "
                             "physical=[%.0f %.0f] content=[%.0f %.0f] "
                             "skipped-card-uv=[%.4f %.4f] "
                             "final-card-uv=[1.0000 1.0000] owner=sprite-frame",
                             wpimgobj.id,
                             wpimgobj.name.c_str(),
                             primary_source_texture.c_str(),
                             r[0],
                             r[1],
                             r[2],
                             r[3],
                             padded_map_rate[0],
                             padded_map_rate[1]);
                }
            } else {
                mapRate = padded_map_rate;
            }
        }

        if (hasAnimatedPuppetMesh) {
            if (hasEffect) {
                // Effects operate on a rectangular layer-local source, then the original image
                // material is appended as the authoritative puppet surface writer. Reusing that
                // material is required because puppet images may carry extra textures and shader
                // semantics such as iris movement and blink masks that a neutral passthrough cannot
                // reconstruct from the resolved color texture alone.
                GenCardMesh(
                    mesh, { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] }, mapRate);
                WPMdlParser::GenPuppetMesh(effct_final_mesh, *puppet);

                wpscene::WPImageEffect puppet_effect;
                wpscene::WPMaterial    puppet_mat;
                puppet_mat             = wpimgobj.material;
                puppet_mat.textures[0] = "";
                WPMdlParser::AddPuppetMatInfo(puppet_mat, *puppet);
                puppet_effect.materials.push_back(std::move(puppet_mat));
                wpimgobj.effects.push_back(std::move(puppet_effect));
            } else {
                svData.puppet_layer = shared_puppet_pose;
                WPMdlParser::GenPuppetMesh(mesh, *puppet);
            }
        } else if (hasStaticImageMesh) {
            if (hasEffect) {
                // Static image-puppet meshes authored in the puppet slot are final-layer shape
                // masks, not animated sources. The effect chain still needs a normal layer-sized
                // source card so filters sample the full media texture, then the resolved writer
                // uses the authored mesh to clip/crop the final visible image without enabling
                // skinning uniforms.
                GenCardMesh(
                    mesh, { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] }, mapRate);
                WPMdlParser::GenPuppetMesh(effct_final_mesh, *puppet);
            } else {
                // No-effect static image puppets can draw the authored mesh directly. This keeps
                // the exported crop geometry and UVs while avoiding any WPPuppet runtime state,
                // which does not exist for flag-9 static image mesh files.
                WPMdlParser::GenPuppetMesh(mesh, *puppet);
            }
        } else {
            if (hasAuthoredPuppet) {
                // Keep this diagnostic tied to the geometry fallback point. The parser error above
                // explains why the authored puppet was unusable; this line records the rendering
                // consequence before the rectangular card hides the real cause in visual output.
                LOG_INFO("ImagePuppetFallback: layer=%d name='%s' puppet='%s' using rectangular "
                         "card mesh",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         wpimgobj.puppet.c_str());
            }
            const auto source_mesh_size = wpimgobj.size;
            GenCardMesh(
                mesh, { (uint16_t)source_mesh_size[0], (uint16_t)source_mesh_size[1] }, mapRate);
            if (wpimgobj.effectFinalTexCoordBoundsEnabled) {
                GenCardMeshWithTexCoordBounds(
                    effct_final_mesh, wpimgobj.size, wpimgobj.effectFinalTexCoordBounds);
            } else {
                GenCardMesh(effct_final_mesh,
                            { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] });
            }
        }
    }
    // material blendmode for last step to use
    auto imgBlendMode = ResolveObjectFinalBlend(material.blenmode, wpimgobj.colorBlendMode);
    // disable img material blend, as it's the first effect node now
    if (hasEffect) {
        material.blenmode = BlendMode::Normal;
    } else {
        material.blenmode = imgBlendMode;
    }
    mesh.AddMaterial(std::move(material));
    spImgNode->AddMesh(spMesh);
    RegisterUserShaderValueBindings(
        context, wpimgobj.material, shaderInfo, spImgNode.get(), wpimgobj.id, wpimgobj.name);

    if (hasAnimatedPuppetMesh) {
        svData.puppet_layer = shared_puppet_pose;
    }

    ConfigureBoneAttachment(context,
                            wpimgobj.parent,
                            wpimgobj.attachment,
                            Eigen::Affine3f(spWorldNode->GetLocalTrans().cast<float>()),
                            "object",
                            wpimgobj.name,
                            svData);

    worldNodeData               = svData;
    worldNodeData.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };
    worldNodeData.parallaxDepthAuthored = wpimgobj.parallaxDepthAuthored;

    if (hasEffect) {
        auto& scene = *context.scene;
        FinalOutputCapability final_shader_capability =
            wpimgobj.config.finalOutputCapability;
        std::string effect_camera_name = EffectCameraName(wpimgobj.id);
        const auto  effect_camera_clip = ResolveImageEffectCameraClipRange(hasAnimatedPuppetMesh);
        // set camera to attatch effect
        if (isCompose) {
            const int32_t source_camera_width =
                std::max<int32_t>(1, static_cast<int32_t>(std::lround(effect_source_size[0])));
            const int32_t source_camera_height =
                std::max<int32_t>(1, static_cast<int32_t>(std::lround(effect_source_size[1])));
            scene.cameras[effect_camera_name] = std::make_shared<SceneCamera>(
                source_camera_width,
                source_camera_height,
                effect_camera_clip.near_clip,
                effect_camera_clip.far_clip);
            scene.cameras.at(effect_camera_name)->AttatchNode(spWorldNode);
            LOG_INFO("SceneCompositionLayerSourceCamera: layer=%d name='%s' camera='%s' "
                     "size=[%d, %d] source-target=[%.3f, %.3f] near=%.3f far=%.3f "
                     "animated-puppet=%s",
                     wpimgobj.id,
                     wpimgobj.name.c_str(),
                     effect_camera_name.c_str(),
                     source_camera_width,
                     source_camera_height,
                     effect_source_size[0],
                     effect_source_size[1],
                     effect_camera_clip.near_clip,
                     effect_camera_clip.far_clip,
                     hasAnimatedPuppetMesh ? "true" : "false");
        } else {
            // Keep the effect camera extents in display units. The render target
            // resolution below may still be reduced independently.
            i32 w                   = (i32)wpimgobj.size[0];
            i32 h                   = (i32)wpimgobj.size[1];
            scene.cameras[effect_camera_name] = std::make_shared<SceneCamera>(
                w, h, effect_camera_clip.near_clip, effect_camera_clip.far_clip);
            scene.cameras.at(effect_camera_name)->AttatchNode(context.effect_camera_node);
            LOG_INFO("SceneImageEffectSourceCamera: layer=%d name='%s' camera='%s' "
                     "size=[%d, %d] near=%.3f far=%.3f animated-puppet=%s",
                     wpimgobj.id,
                     wpimgobj.name.c_str(),
                     effect_camera_name.c_str(),
                     w,
                     h,
                     effect_camera_clip.near_clip,
                     effect_camera_clip.far_clip,
                     hasAnimatedPuppetMesh ? "true" : "false");
        }
        spImgNode->SetCamera(effect_camera_name);
        const int32_t effect_target_width = ClampDestinationRenderTargetExtent(
            static_cast<int32_t>(std::lround(effect_target_resolution[0])));
        const int32_t effect_target_height = ClampDestinationRenderTargetExtent(
            static_cast<int32_t>(std::lround(effect_target_resolution[1])));
        const auto effect_destination_names = SceneDestinationRenderTargetNames(
            scene, wpimgobj.parent, effect_target_width, effect_target_height);
        const std::string& effect_ppong_a = effect_destination_names[0];
        const std::string& effect_ppong_b = effect_destination_names[1];
        // set image effect
        // Compose layers keep their source node in the normal scene tree, but their final authored
        // effect pass is still a detached render-graph node. Give the effect layer a world node
        // even when the source node is not detached so final output can inherit virtual parent
        // transforms from render-order proxy groups instead of drawing at the compose layer's local
        // coordinates.
        auto* effect_world_node =
            (use_detached_effect_world_node || isCompose) ? spWorldNode.get() : nullptr;
        auto imgEffectLayer = std::make_shared<SceneImageEffectLayer>(
            effect_world_node, wpimgobj.size[0], wpimgobj.size[1], effect_ppong_a, effect_ppong_b);
        {
            // Fullscreen image-effect layers are postprocess-style framebuffer passes. Remember
            // that authored shape here so ResolveEffect() can keep their final shader on the
            // effect-camera fullscreen quad instead of projecting the 2x2 utility mesh through the
            // active scene camera.
            imgEffectLayer->SetFullscreen(wpimgobj.fullscreen);
            imgEffectLayer->SetFinalBlend(imgBlendMode);
            imgEffectLayer->SetCopyBackground(wpimgobj.copybackground);
            const auto source_policy = ResolveImageEffectSourcePolicy(isCompose, wpimgobj);
            imgEffectLayer->SetSourceContributionPolicy(source_policy);
            if (isCompose) {
                LOG_INFO("SceneCompositionLayerSourcePolicy: layer=%d name='%s' "
                         "copybackground=%s policy=%.*s",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         wpimgobj.copybackground ? "true" : "false",
                         static_cast<int>(ImageEffectSourcePolicyName(source_policy).size()),
                         ImageEffectSourcePolicyName(source_policy).data());
            }
            imgEffectLayer->SetHiddenFinalCompositePolicy(
                ResolveHiddenFinalCompositePolicy(*context.scene, wpimgobj));
            imgEffectLayer->SourceMesh().ChangeMeshDataFrom(mesh);
            imgEffectLayer->FinalMesh().ChangeMeshDataFrom(effct_final_mesh);
            imgEffectLayer->FinalNode().CopyTrans(use_detached_effect_world_node ? *spWorldNode
                                                                                 : *spImgNode);
            if (! use_detached_effect_world_node && ! isCompose) {
                spImgNode->CopyTrans(SceneNode());
            }
            // The owning SceneObject owns the effect bridge; the private effect camera is a pure
            // projection resource with no back-reference. The bridge records that camera's name so
            // draw-time consumers can match SceneNode::Camera() against it and geometry updates
            // and destroy reach the camera through the layer.
            scene.EnsureSceneObject(wpimgobj.id).SetImageEffectLayer(imgEffectLayer);
            imgEffectLayer->SetBridgeCameraName(effect_camera_name);
            imgEffectLayer->AddRuntimeCameraName(effect_camera_name);
        }
        if (hasAnimatedPuppetMesh && puppet->asset_bounds.IsFiniteAndOrdered()) {
            const std::string puppet_surface_camera = effect_camera_name + "__puppet_surface_camera";
            const std::string puppet_surface_target =
                "_rt_puppet_surface_" + effect_camera_name;
            imgEffectLayer->SetPuppetSurfaceProjection(BuildPuppetSurfaceProjection(
                wpimgobj,
                *puppet,
                effct_final_mesh,
                puppet_surface_camera,
                puppet_surface_target,
                effect_source_size));
            imgEffectLayer->SetLayerSurfaceCamera(puppet_surface_camera);

            const auto* projection = imgEffectLayer->GetPuppetSurfaceProjection();
            if (projection != nullptr) {
                scene.cameras[puppet_surface_camera] = std::make_shared<SceneCamera>(
                    1, 1, effect_camera_clip.near_clip, effect_camera_clip.far_clip);
                scene.cameras.at(puppet_surface_camera)->AttatchNode(context.effect_camera_node);
                scene.cameras.at(puppet_surface_camera)->SetOrthographicViewRect(
                    projection->surface_bounds.min.x(),
                    projection->surface_bounds.max.x(),
                    projection->surface_bounds.min.y(),
                    projection->surface_bounds.max.y());
                imgEffectLayer->AddRuntimeCameraName(puppet_surface_camera);

                scene.renderTargets[puppet_surface_target] = SceneRenderTarget {
                    .width = projection->target_extent[0],
                    .height = projection->target_extent[1],
                    .mapWidth = projection->target_extent[0],
                    .mapHeight = projection->target_extent[1],
                    .allowReuse = true,
                    .sample = source_sampler,
                };
                imgEffectLayer->AddRuntimeRenderTargetName(puppet_surface_target);
            }
        }
        // set renderTarget for ping-pong operate. Destination targets are fixed-size images even
        // for fullscreen layers: they keep the output framebuffer size resolved at load instead of
        // following later output resizes, so the interned name always describes the backing image.
        {
            SceneRenderTarget pingpong_a_target {
                .width      = effect_target_width,
                .height     = effect_target_height,
                .mapWidth   = effect_target_width,
                .mapHeight  = effect_target_height,
                .allowReuse = true,
                .sample     = source_sampler,
            };
            InternNamedRenderTarget(scene, effect_ppong_a, pingpong_a_target);

            SceneRenderTarget pingpong_b_target = pingpong_a_target;
            // Intermediate effect output is a separate sampling contract. Point-preserving source
            // passes read ping-pong A with the authored source sampler; generic downstream filters
            // read ping-pong B linearly unless their own FBO contract says otherwise.
            pingpong_b_target.sample = TextureSample {
                .wrapS = TextureWrap::CLAMP_TO_EDGE,
                .wrapT = TextureWrap::CLAMP_TO_EDGE,
                .magFilter = TextureFilter::LINEAR,
                .minFilter = TextureFilter::LINEAR,
            };
            InternNamedRenderTarget(scene, effect_ppong_b, pingpong_b_target);
            imgEffectLayer->AddRuntimeRenderTargetName(effect_ppong_a);
            imgEffectLayer->AddRuntimeRenderTargetName(effect_ppong_b);
            if (wpimgobj.fullscreen || wpimgobj.effectSourceSizeIsPixelExtent) {
                LOG_INFO("SceneEffectPingPongTargetResolve: layer=%d name='%s' "
                         "pingpong-a='%s' pingpong-b='%s' authored-size=[%.3f, %.3f] "
                         "target=%dx%d fullscreen=%s",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         effect_ppong_a.c_str(),
                         effect_ppong_b.c_str(),
                         effect_source_size[0],
                         effect_source_size[1],
                         effect_target_width,
                         effect_target_height,
                         wpimgobj.fullscreen ? "true" : "false");
            }
        }
        // A single static-visible effect pass with no private buffers or commands is the layer's
        // own on-screen writer: its material draws the final quad at output resolution while
        // sampling the private source target. Multi-pass chains, dependency sources, puppets,
        // compose helpers, and runtime-toggled effects keep the private ping-pong route.
        const wpscene::WPImageEffect* direct_final_effect = nullptr;
        wpscene::WPMaterial           direct_final_material;
        if (hasAuthoredEffect && ! has_shader_color_blend && ! is_offscreen_dependency_source &&
            ! isCompose && ! hasAnimatedPuppetMesh && wpimgobj.effects.size() == 1) {
            const auto& candidate = wpimgobj.effects.front();
            const bool candidate_initial_visible =
                ResolveEffectVisibility(candidate, context.user_properties);
            bool eligible = candidate_initial_visible &&
                            ! EffectVisibilityCanChangeAtRuntime(candidate) &&
                            candidate.materials.size() == 1 && candidate.fbos.empty() &&
                            candidate.commands.empty() && candidate.passes.size() <= 1;
            if (eligible) {
                direct_final_material = candidate.materials.front();
                if (! candidate.passes.empty()) {
                    const auto& direct_pass = candidate.passes.front();
                    if (! direct_pass.target.empty()) {
                        eligible = false;
                    } else {
                        for (const auto& bind : direct_pass.bind) {
                            if (bind.name != "previous") {
                                eligible = false;
                                break;
                            }
                        }
                    }
                    if (eligible) {
                        direct_final_material.MergePass(direct_pass);
                        for (const auto& bind : direct_pass.bind) {
                            if (direct_final_material.textures.size() <=
                                static_cast<usize>(bind.index)) {
                                direct_final_material.textures.resize(
                                    static_cast<usize>(bind.index) + 1);
                            }
                            direct_final_material.textures[static_cast<usize>(bind.index)] =
                                effect_ppong_a;
                        }
                    }
                }
            }
            if (eligible) direct_final_effect = &candidate;
        }

        if (hasAuthoredEffect || has_shader_color_blend) {
            // Dependency-only sources intentionally stop at the first ping-pong target so
            // `_rt_imageLayerComposite_<id>` samples the raw source texture. Every real authored
            // chain and every framebuffer-aware color blend instead uses Wallpaper Engine's
            // independent final passthrough publisher.
            const auto finalCompositeTransformData = BuildEffectWriterTransformData(
                context,
                BuildImageEffectFinalCompositeContract(wpimgobj,
                                                       uses_routed_parent));
            ConfigureEffectFinalComposite(context,
                                          *imgEffectLayer,
                                          effect_ppong_a,
                                          wpimgobj.id,
                                          wpimgobj.name,
                                          wpimgobj.colorBlendMode,
                                          &finalCompositeTransformData,
                                          direct_final_effect != nullptr ? &direct_final_material
                                                                         : nullptr,
                                          direct_final_effect != nullptr ? direct_final_effect->id
                                                                         : 0);
        }
        int32_t i_eff = -1;
        for (const auto& wpeffobj : wpimgobj.effects) {
            i_eff++;
            if (&wpeffobj == direct_final_effect) {
                // The single authored pass already draws as the layer's final on-screen writer;
                // no private pass nodes exist for this effect.
                continue;
            }
            std::shared_ptr<SceneImageEffect> imgEffect = std::make_shared<SceneImageEffect>();
            imgEffect->SetIdentity(
                wpimgobj.id, wpeffobj.id, static_cast<uint32_t>(i_eff), wpeffobj.name);
            const bool effect_initial_visible =
                ResolveEffectVisibility(wpeffobj, context.user_properties);
            const bool effect_runtime_visibility =
                EffectVisibilityCanChangeAtRuntime(wpeffobj);
            imgEffect->SetRuntimeVisibilityContract(effect_runtime_visibility);
            const ImageEffectMaterialTopology effect_material_topology {
                .uses_routed_parent = uses_routed_parent,
                .writer_role        = ResolveImageEffectWriterRole(isCompose),
            };

            // this will be replace when resolve, use here to get rt info
            const std::string inRT { effect_ppong_a };

            // FBO name map and effect command. `unique` is scoped to the authored effect id;
            // non-unique FBOs retain their JSON name and therefore intern across layers.
            const auto  feedback_fbos = wpeffobj.FeedbackFboNames();

            std::unordered_map<std::string, std::string> fboMap;
            {
                fboMap["previous"] = inRT;
                for (usize i = 0; i < wpeffobj.fbos.size(); i++) {
                    const auto& wpfbo  = wpeffobj.fbos.at(i);
                    const std::string rtname =
                        EffectFboRenderTargetName(wpfbo, wpeffobj.id);
                    // Effect FBOs are fixed images derived from the layer's effect target size
                    // (divided by the authored scale, or fitted). A fullscreen layer therefore gets
                    // framebuffer-sized FBOs at load without a screen binding, matching its
                    // fixed destination targets above.
                    const auto  fbo_size = wpfbo.ResolveSize(effect_target_resolution);
                    const bool  persistent_feedback_fbo =
                        feedback_fbos.count(wpfbo.name) != 0;
                    SceneRenderTarget fbo_target {
                        .width      = fbo_size[0],
                        .height     = fbo_size[1],
                        .mapWidth   = fbo_size[0],
                        .mapHeight  = fbo_size[1],
                        .allowReuse = ! persistent_feedback_fbo,
                    };
                    // `fit`-sized feedback buffers are authored simulation textures, not
                    // display-space framebuffers. Their resolved descriptor enters the same global
                    // name table and the first registration owns the size.
                    InternNamedRenderTarget(scene, rtname, fbo_target);
                    if (wpfbo.fit > 0 || persistent_feedback_fbo) {
                        LOG_INFO("SceneEffectFboResolve: layer=%d effect-id=%d effect='%s' "
                                 "fbo='%s' target='%s' size=%dx%d scale=%u fit=%u "
                                 "persistent-feedback=%s",
                                 wpimgobj.id,
                                 wpeffobj.id,
                                 wpeffobj.name.c_str(),
                                 wpfbo.name.c_str(),
                                 rtname.c_str(),
                                 fbo_size[0],
                                 fbo_size[1],
                                 wpfbo.scale,
                                 wpfbo.fit,
                                 persistent_feedback_fbo ? "true" : "false");
                    }
                    imgEffectLayer->AddRuntimeRenderTargetName(rtname);
                    fboMap[wpfbo.name] = rtname;
                }
            }
            // load! effect commands
            {
                for (const auto& el : wpeffobj.commands) {
                    if (el.command != "copy") {
                        LOG_ERROR("Unknown effect command: %s", el.command.c_str());
                        continue;
                    }
                    if (fboMap.count(el.target) + fboMap.count(el.source) < 2) {
                        LOG_ERROR("Unknown effect command dst or src: %s %s",
                                  el.target.c_str(),
                                  el.source.c_str());
                        continue;
                    }
                    const auto resolved_dst = fboMap[el.target];
                    const auto resolved_src = fboMap[el.source];
                    imgEffect->commands.push_back({ .cmd          = SceneImageEffect::CmdType::Copy,
                                                    .authored_dst = resolved_dst,
                                                    .authored_src = resolved_src,
                                                    .dst          = resolved_dst,
                                                    .src          = resolved_src,
                                                    .afterpos     = el.afterpos });
                }
            }

            bool eff_mat_ok { true };

            for (usize i_mat = 0; i_mat < wpeffobj.materials.size(); i_mat++) {
                wpscene::WPMaterial wpmat = wpeffobj.materials.at(i_mat);
                std::string         matOutRT { WE_EFFECT_PPONG_PREFIX_B };
                if (wpeffobj.passes.size() > i_mat) {
                    const auto& wppass = wpeffobj.passes.at(i_mat);
                    wpmat.MergePass(wppass);
                    // Set rendertarget, in and out
                    for (const auto& el : wppass.bind) {
                        if (fboMap.count(el.name) == 0) {
                            LOG_ERROR("fbo %s not found", el.name.c_str());
                            continue;
                        }
                        if (wpmat.textures.size() <= (usize)el.index)
                            wpmat.textures.resize((usize)el.index + 1);
                        wpmat.textures[(usize)el.index] = fboMap[el.name];
                    }
                    if (! wppass.target.empty()) {
                        if (fboMap.count(wppass.target) == 0) {
                            LOG_ERROR("fbo %s not found", wppass.target.c_str());
                        } else {
                            matOutRT = fboMap.at(wppass.target);
                        }
                    }
                }
                if (wpmat.textures.size() == 0) wpmat.textures.resize(1);
                if (wpmat.textures.at(0).empty()) {
                    wpmat.textures[0] = inRT;
                }
                auto spEffNode = std::make_shared<SceneNode>();
                // Effect passes are drawing phases of the authored layer, not second layer
                // identities: they stay out of nodeOwners, the node id is the back-reference the
                // layer resolvers use (NodeLayerId and friends), and the phase name keeps the
                // authored name unique to the world node.
                spEffNode->ID() = wpimgobj.id;
                spEffNode->SetName(wpimgobj.name + "::__hanabi_effect_pass_" +
                                   std::to_string(i_eff) + "_" + std::to_string(i_mat));
                ShaderValueMap effectBaseConstSvs = baseConstSvs;
                WPShaderInfo wpEffShaderInfo;
                wpEffShaderInfo.baseConstSvs = std::move(effectBaseConstSvs);
                wpEffShaderInfo.baseConstSvs["g_EffectTextureProjectionMatrix"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
                wpEffShaderInfo.baseConstSvs["g_EffectTextureProjectionMatrixInverse"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
                SceneMaterial     material;
                WPShaderValueData svData;
                if (! LoadMaterial(vfs,
                                   wpmat,
                                   context.scene.get(),
                                   spEffNode.get(),
                                   &material,
                                   &svData,
                                   context.user_properties,
                                   &wpEffShaderInfo)) {
                    LOG_ERROR(
                        "SceneEffectLoad: layer=%d effect='%s' material-index=%zu load failed",
                        wpimgobj.id,
                        wpeffobj.name.c_str(),
                        i_mat);
                    eff_mat_ok = false;
                    break;
                }

                if (IsCurrentEffectWriterTarget(matOutRT)) {
                    final_shader_capability = ResolveFinalShaderCapability(wpmat.shader);
                }

                // load glname from alias and load to constvalue
                LoadConstvalue(material, wpmat, wpEffShaderInfo);
                LoadUserShaderValue(material, wpmat, wpEffShaderInfo, context.user_properties);
                auto spMesh = std::make_shared<SceneMesh>();
                {
                    ApplyEffectWriterTransformContract(
                        context,
                        BuildImageEffectMaterialContract(wpimgobj,
                                                         *imgEffectLayer,
                                                         effect_material_topology,
                                                         hasAnimatedPuppetMesh && wpmat.use_puppet),
                        svData);
                    if (hasAnimatedPuppetMesh && wpmat.use_puppet) {
                        svData.puppet_layer = shared_puppet_pose;
                    }
                }
                const auto authored_textures = material.textures;
                spMesh->AddMaterial(std::move(material));
                spEffNode->AddMesh(spMesh);
                RegisterUserShaderValueBindings(
                    context, wpmat, wpEffShaderInfo, spEffNode.get(), wpimgobj.id, wpimgobj.name);
                RegisterConstantShaderValueBindings(context,
                                                    wpmat,
                                                    wpEffShaderInfo,
                                                    spEffNode.get(),
                                                    wpimgobj.id,
                                                    wpimgobj.name,
                                                    wpeffobj.id,
                                                    i_eff,
                                                    i_mat);

                context.shader_updater->SetNodeData(spEffNode.get(), svData);
                imgEffect->nodes.push_back({ .authored_output = matOutRT,
                                             .output = matOutRT,
                                             .authored_textures = authored_textures,
                                             .sceneNode = spEffNode,
                                             .private_final_output_uses_layer_surface =
                                                 hasAnimatedPuppetMesh && wpmat.use_puppet });
            }

            if (eff_mat_ok) {
                // Set the resolved instance bit only after every pass node exists so all nodes
                // receive the same initial gate. This changes execution state only; the effect and
                // every render target above remain resident exactly as they were materialized.
                imgEffect->SetLocalVisible(effect_initial_visible);
                if (! wpeffobj.visible_json.is_null()) {
                    LOG_INFO("SceneEffectVisibilityResolve: layer=%d effect-id=%d effect-index=%d "
                             "name='%s' authored=%s initial=%s runtime=%s",
                             wpimgobj.id,
                             wpeffobj.id,
                             i_eff,
                             wpeffobj.name.c_str(),
                             wpeffobj.visible ? "true" : "false",
                             effect_initial_visible ? "true" : "false",
                             effect_runtime_visibility ? "true" : "false");
                }
                imgEffectLayer->AddEffect(imgEffect);
            } else {
                LOG_ERROR("effect \'%s\' failed to load", wpeffobj.name.c_str());
            }
        }

        // Capability priority is structural. Dependency sources must remain private, animated
        // puppets must keep their private skinned surface, and only then may the last authored
        // shader publish directly in scene space.
        if (is_offscreen_dependency_source) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivateDependency);
        } else if (hasAnimatedPuppetMesh) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivatePuppetSurface);
        } else if (wpimgobj.config.finalOutputCapability ==
                   FinalOutputCapability::SceneAuthoredWriter) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::SceneAuthoredWriter);
        } else {
            imgEffectLayer->SetFinalOutputCapability(final_shader_capability);
        }
        LOG_INFO("SceneEffectOutputCapability: layer=%d name='%s' capability=%.*s "
                 "dependency=%s puppet=%s source-policy=%.*s",
                 wpimgobj.id,
                 wpimgobj.name.c_str(),
                 static_cast<int>(FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).size()),
                 FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).data(),
                 is_offscreen_dependency_source ? "true" : "false",
                 hasAnimatedPuppetMesh ? "true" : "false",
                 static_cast<int>(ImageEffectSourcePolicyName(
                     imgEffectLayer->SourceContributionPolicy()).size()),
                 ImageEffectSourcePolicyName(
                     imgEffectLayer->SourceContributionPolicy()).data());

        int32_t source_width = 0;
        int32_t source_height = 0;
        if (const auto source_texture_it = scene.textures.find(primary_source_texture);
            source_texture_it != scene.textures.end()) {
            source_width = source_texture_it->second.width;
            source_height = source_texture_it->second.height;
        } else if (const auto source_target_it = scene.renderTargets.find(primary_source_texture);
                   source_target_it != scene.renderTargets.end()) {
            source_width = source_target_it->second.ContentWidth();
            source_height = source_target_it->second.ContentHeight();
        }
        const auto& pingpong_a = scene.renderTargets.at(effect_ppong_a);
        const auto& pingpong_b = scene.renderTargets.at(effect_ppong_b);
        const auto& display_target = scene.renderTargets.at(SpecTex_Default.data());
        LOG_INFO("SceneEffectTextureContract: layer=%d name='%s' source='%s' "
                 "source-size=[%d %d] authored-layer-size=[%.3f %.3f] "
                 "source-sampler=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "private-target-a='%s' private-target-a-size=[%d %d] "
                 "intermediate-sampler-a=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "private-target-b='%s' private-target-b-size=[%d %d] "
                 "intermediate-sampler-b=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "display-target='%.*s' display-target-size=[%d %d] output-policy=%.*s",
                 wpimgobj.id,
                 wpimgobj.name.c_str(),
                 primary_source_texture.c_str(),
                 source_width,
                 source_height,
                 wpimgobj.size[0],
                 wpimgobj.size[1],
                 static_cast<int>(TextureWrapName(source_sampler.wrapS).size()),
                 TextureWrapName(source_sampler.wrapS).data(),
                 static_cast<int>(TextureWrapName(source_sampler.wrapT).size()),
                 TextureWrapName(source_sampler.wrapT).data(),
                 static_cast<int>(TextureFilterName(source_sampler.magFilter).size()),
                 TextureFilterName(source_sampler.magFilter).data(),
                 static_cast<int>(TextureFilterName(source_sampler.minFilter).size()),
                 TextureFilterName(source_sampler.minFilter).data(),
                 effect_ppong_a.c_str(),
                 pingpong_a.width,
                 pingpong_a.height,
                 static_cast<int>(TextureWrapName(pingpong_a.sample.wrapS).size()),
                 TextureWrapName(pingpong_a.sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(pingpong_a.sample.wrapT).size()),
                 TextureWrapName(pingpong_a.sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(pingpong_a.sample.magFilter).size()),
                 TextureFilterName(pingpong_a.sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(pingpong_a.sample.minFilter).size()),
                 TextureFilterName(pingpong_a.sample.minFilter).data(),
                 effect_ppong_b.c_str(),
                 pingpong_b.width,
                 pingpong_b.height,
                 static_cast<int>(TextureWrapName(pingpong_b.sample.wrapS).size()),
                 TextureWrapName(pingpong_b.sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(pingpong_b.sample.wrapT).size()),
                 TextureWrapName(pingpong_b.sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(pingpong_b.sample.magFilter).size()),
                 TextureFilterName(pingpong_b.sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(pingpong_b.sample.minFilter).size()),
                 TextureFilterName(pingpong_b.sample.minFilter).data(),
                 static_cast<int>(SpecTex_Default.size()),
                 SpecTex_Default.data(),
                 display_target.width,
                 display_target.height,
                 static_cast<int>(FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).size()),
                 FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).data());
    }
    if (uses_routed_parent) {
        if (hasEffect) {
            ConfigureRoutedEffectWorldParentBinding(context, wpimgobj.parent, worldNodeData);
        } else {
            ConfigureInheritedParentBinding(context, wpimgobj.parent, svData);
        }
        context.scene->sceneGraph->AppendChild(spWorldNode);
    } else {
        AttachNodeToScene(context, spWorldNode, wpimgobj.parent, wpimgobj.name, &svData);
    }
    context.object_nodes[wpimgobj.id] = spWorldNode;
    context.scene->EnsureSceneObject(wpimgobj.id)
        .SetImageRuntimeState(Scene::ImageLayerRuntimeState {
            .size      = wpimgobj.size,
            .alignment = wpimgobj.alignment,
        });
    context.scene->AddLayerRuntimeNode(wpimgobj.id, spWorldNode.get());
    if (hasAnimatedPuppetMesh) {
        context.object_puppets[wpimgobj.id] = puppet->puppet.get();
    }
    // Effect-backed image layers usually use a detached source node plus a separate world node:
    // the source node keeps `svData` so it can render into the private effect camera, while the
    // world node keeps `worldNodeData` so authored parent transforms and parent-anchored parallax
    // match Wallpaper Engine's scene hierarchy. Compose layers are the exception because their
    // image node and world node are the same object. In that case, register the inherited
    // world-node data on the shared node; otherwise a child compose layer with its own
    // `parallaxDepth` would ignore a zero-parallax parent and incorrectly drift with the cursor.
    context.shader_updater->SetNodeData(
        spImgNode.get(),
        spImgNode.get() == spWorldNode.get() && hasEffect ? worldNodeData : svData);
    if (spImgNode.get() != spWorldNode.get()) {
        context.shader_updater->SetNodeData(spWorldNode.get(), worldNodeData);
        // The detached source node is a drawing phase owned by the effect bridge, not a second
        // layer identity: it never enters sceneGraph, stays out of nodeOwners, and its node id
        // (set above) is the back-reference layer resolvers use. The render graph emits its draw
        // when the world node is visited at the authored order position.
        if (auto* source_bridge = context.scene->FindImageEffectLayer(wpimgobj.id)) {
            source_bridge->AddDetachedSourceNode(spImgNode);
        }
        context.scene->AddLayerRuntimeNode(wpimgobj.id, spImgNode.get());
    }
    RegisterLayerSceneState(
        context, wpimgobj.id, wpimgobj.parent, wpimgobj.attachment, wpimgobj.visible);
    context.scene->ApplyLayerVisibility(wpimgobj.id);
}

void ParseTextObj(ParseContext& context, wpscene::WPTextObject& text_obj) {
    TextLayerRenderContract render_contract;
    render_contract.has_materialized_authored_effects = ! text_obj.effects.empty();
    render_contract.uses_shader_color_blend_bridge =
        UsesShaderColorBlendMode(text_obj.colorBlendMode);

    // This immutable contract is resolved before materialization. Text rasterization, logical-box
    // preservation, camera/target sizing, glyph placement, and final publication must all agree on
    // the same bridge decision; consulting effects or blend mode again downstream recreates the
    // crop/offset mismatch this contract exists to prevent.

    std::shared_ptr<SceneTextPrimitive> primitive;
    std::string                         error;
    if (! BuildSceneTextPrimitive(
            *context.vfs,
            text_obj,
            render_contract,
            0,
            context.scene->textRenderScale,
            &primitive,
            &error)) {
        LOG_ERROR("build text primitive '%s' failed: %s", text_obj.name.c_str(), error.c_str());
        return;
    }

    const bool has_effect = render_contract.RequiresBridge();
    auto       spWorldNode = std::make_shared<SceneNode>(Vector3f(text_obj.origin.data()),
                                                         Vector3f(text_obj.scale.data()),
                                                         Vector3f(text_obj.angles.data()),
                                                         text_obj.name);
    spWorldNode->ID()      = text_obj.id;
    auto spTextNode        = has_effect ? std::make_shared<SceneNode>() : spWorldNode;
    if (has_effect) {
        // Same phase-naming contract as image layers: the bridged text node is the layer's
        // private-camera source phase, so the authored name stays unique to the world node.
        spTextNode->SetName(text_obj.name + "::__hanabi_effect_source");
    }
    spTextNode->ID() = text_obj.id;
    spTextNode->AddText(primitive);

    WPShaderValueData worldNodeData;
    worldNodeData.parallaxDepth = { text_obj.parallaxDepth[0], text_obj.parallaxDepth[1] };
    worldNodeData.parallaxDepthAuthored = text_obj.parallaxDepthAuthored;
    ConfigureBoneAttachment(context,
                            text_obj.parent,
                            text_obj.attachment,
                            Eigen::Affine3f(spWorldNode->GetLocalTrans().cast<float>()),
                            "text object",
                            text_obj.name,
                            worldNodeData);

    if (has_effect) {
        auto&             scene       = *context.scene;
        const std::string camera_name = EffectCameraName(text_obj.id);
        primitive->bridge.camera_name = camera_name;
        primitive->bridge.bridge_backing_extent = {
            static_cast<uint32_t>(ClampDestinationRenderTargetExtent(
                static_cast<int32_t>(std::lround(primitive->VisibleDisplaySize()[0])))),
            static_cast<uint32_t>(ClampDestinationRenderTargetExtent(
                static_cast<int32_t>(std::lround(primitive->VisibleDisplaySize()[1])))),
        };
        const auto bridge_destination_names = SceneDestinationRenderTargetNames(
            scene,
            text_obj.parent,
            static_cast<int32_t>(primitive->bridge.bridge_backing_extent[0]),
            static_cast<int32_t>(primitive->bridge.bridge_backing_extent[1]));
        primitive->bridge.pingpong_a = bridge_destination_names[0];
        primitive->bridge.pingpong_b = bridge_destination_names[1];
        primitive->bridge.render_targets.push_back(
            TextBridgeRenderTarget { .name = primitive->bridge.pingpong_a, .scale = 1 });
        primitive->bridge.render_targets.push_back(
            TextBridgeRenderTarget { .name = primitive->bridge.pingpong_b, .scale = 1 });

        const auto display_size = primitive->VisibleDisplaySize();
        SceneMesh  effect_final_mesh {};
        RebuildTextPrimitiveVisibleMesh(&effect_final_mesh, *primitive);

        scene.cameras[camera_name] = std::make_shared<SceneCamera>(
            std::max(1, static_cast<int32_t>(std::lround(display_size[0]))),
            std::max(1, static_cast<int32_t>(std::lround(display_size[1]))),
            -1.0f,
            1.0f);
        scene.cameras.at(camera_name)->AttatchNode(context.effect_camera_node);
        // Effect-backed text draws the canonical glyph primitive into an isolated source target
        // before authored image effects sample it. Keep that source node on an explicit identity
        // shader-data contract instead of relying on the visible world node's parallax/attachment
        // data: the world node is only the final composited output transform, while this node must
        // fill the bridge camera exactly in local text space.
        WPShaderValueData text_source_node_data;
        context.shader_updater->SetNodeData(spTextNode.get(), text_source_node_data);
        spTextNode->SetCamera(camera_name);

        auto imgEffectLayer = std::make_shared<SceneImageEffectLayer>(spWorldNode.get(),
                                                                      display_size[0],
                                                                      display_size[1],
                                                                      primitive->bridge.pingpong_a,
                                                                      primitive->bridge.pingpong_b);
        imgEffectLayer->SetFinalBlend(
            ResolveObjectFinalBlend(BlendMode::Translucent, text_obj.colorBlendMode));
        imgEffectLayer->FinalMesh().ChangeMeshDataFrom(effect_final_mesh);
        imgEffectLayer->FinalNode().CopyTrans(*spWorldNode);
        // Same contract as image layers: the object owns the bridge, and the bridge records the
        // camera it materialized; the camera itself carries no back-reference.
        scene.EnsureSceneObject(text_obj.id).SetImageEffectLayer(imgEffectLayer);
        imgEffectLayer->SetBridgeCameraName(camera_name);
        imgEffectLayer->AddRuntimeCameraName(camera_name);

        SceneRenderTarget text_pingpong_target {
            .width = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[0]),
            .height = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[1]),
            .mapWidth = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[0]),
            .mapHeight = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[1]),
            // Text targets keep the authored letter box for both the physical backing and the
            // logical effect grid. Persisting the cache entry replaces the old image in place.
            .allowReuse = false,
        };
        InternNamedRenderTarget(scene, primitive->bridge.pingpong_a, text_pingpong_target);
        InternNamedRenderTarget(scene, primitive->bridge.pingpong_b, text_pingpong_target);
        imgEffectLayer->AddRuntimeRenderTargetName(primitive->bridge.pingpong_a);
        imgEffectLayer->AddRuntimeRenderTargetName(primitive->bridge.pingpong_b);
        const auto source_size = primitive->VisibleSourceSize();
        const auto& bridge_camera = *scene.cameras.at(camera_name);
        const auto& bridge_target = scene.renderTargets.at(primitive->bridge.pingpong_a);
        LOG_INFO("SceneTextBridgeContract: layer=%d name='%s' camera='%s' "
                 "camera-size=[%.3f %.3f] target='%s' target-size=[%d %d] "
                 "logical-display=[%.3f %.3f] logical-source=[%.3f %.3f] "
                 "glyph-display=[%.3f %.3f] glyph-source=[%.3f %.3f] "
                 "glyph-offset=[%.3f %.3f] display-offset=[%.3f %.3f] "
                 "source-crop=[%.3f %.3f %.3f %.3f] "
                 "final-mesh-bounds=[%.3f %.3f]-[%.3f %.3f]",
                 text_obj.id,
                 text_obj.name.c_str(),
                 camera_name.c_str(),
                 bridge_camera.Width(),
                 bridge_camera.Height(),
                 primitive->bridge.pingpong_a.c_str(),
                 bridge_target.width,
                 bridge_target.height,
                 display_size[0],
                 display_size[1],
                 source_size[0],
                 source_size[1],
                 primitive->layout.glyph_display_size[0],
                 primitive->layout.glyph_display_size[1],
                 primitive->layout.glyph_source_size[0],
                 primitive->layout.glyph_source_size[1],
                 primitive->layout.glyph_offset[0],
                 primitive->layout.glyph_offset[1],
                 primitive->layout.visible_display_offset[0],
                 primitive->layout.visible_display_offset[1],
                 primitive->layout.glyph_source_crop[0],
                 primitive->layout.glyph_source_crop[1],
                 primitive->layout.glyph_source_crop[2],
                 primitive->layout.glyph_source_crop[3],
                 -display_size[0] * 0.5f,
                 -display_size[1] * 0.5f,
                 display_size[0] * 0.5f,
                 display_size[1] * 0.5f);
        const auto finalCompositeTransformData = BuildEffectWriterTransformData(
            context, BuildTextEffectFinalCompositeContract(text_obj));
        ConfigureEffectFinalComposite(context,
                                      *imgEffectLayer,
                                      primitive->bridge.pingpong_a,
                                      text_obj.id,
                                      text_obj.name,
                                      text_obj.colorBlendMode,
                                      &finalCompositeTransformData);

        const std::string in_rt        = primitive->bridge.pingpong_a;
        int32_t           effect_index = -1;
        // Same final-writer contract as image layers: the shader of the last pass that writes the
        // chain output decides whether that pass may draw in scene space or must stay private.
        FinalOutputCapability final_shader_capability = FinalOutputCapability::PrivateThenPublish;
        for (const auto& wp_effect : text_obj.effects) {
            effect_index++;
            std::shared_ptr<SceneImageEffect> img_effect = std::make_shared<SceneImageEffect>();
            img_effect->SetIdentity(
                text_obj.id, wp_effect.id, static_cast<uint32_t>(effect_index), wp_effect.name);
            const bool effect_initial_visible =
                ResolveEffectVisibility(wp_effect, context.user_properties);
            const bool effect_runtime_visibility =
                EffectVisibilityCanChangeAtRuntime(wp_effect);
            img_effect->SetRuntimeVisibilityContract(effect_runtime_visibility);
            std::unordered_map<std::string, std::string> fbo_map;
            fbo_map["previous"] = in_rt;
            const auto feedback_fbos = wp_effect.FeedbackFboNames();

            for (const auto& wp_fbo : wp_effect.fbos) {
                const std::string rt_name =
                    EffectFboRenderTargetName(wp_fbo, wp_effect.id);
                const auto fbo_size = wp_fbo.ResolveSize(primitive->VisibleDisplaySize());
                const bool        persistent_feedback_fbo =
                    feedback_fbos.count(wp_fbo.name) != 0;
                SceneRenderTarget fbo_target {
                    .width      = fbo_size[0],
                    .height     = fbo_size[1],
                    .mapWidth   = fbo_size[0],
                    .mapHeight  = fbo_size[1],
                    // Text effect targets participate in the same global name table as image
                    // effects. Persistent allocation also preserves feedback contents across
                    // frames when the authored command stream reads its previous output.
                    .allowReuse = false,
                };
                InternNamedRenderTarget(scene, rt_name, fbo_target);
                if (wp_fbo.fit > 0 || persistent_feedback_fbo) {
                    LOG_INFO("SceneTextEffectFboResolve: layer=%d effect-id=%d effect='%s' "
                             "fbo='%s' target='%s' size=%dx%d scale=%u fit=%u "
                             "persistent-feedback=%s",
                             text_obj.id,
                             wp_effect.id,
                             wp_effect.name.c_str(),
                             wp_fbo.name.c_str(),
                             rt_name.c_str(),
                             fbo_size[0],
                             fbo_size[1],
                             wp_fbo.scale,
                             wp_fbo.fit,
                             persistent_feedback_fbo ? "true" : "false");
                }
                imgEffectLayer->AddRuntimeRenderTargetName(rt_name);
                primitive->bridge.render_targets.push_back(TextBridgeRenderTarget {
                    .name = rt_name,
                    .scale = std::max<uint32_t>(1u, wp_fbo.scale),
                    .fit = wp_fbo.fit,
                    .persistent_feedback = persistent_feedback_fbo,
                });
                fbo_map[wp_fbo.name] = rt_name;
            }

            for (const auto& command : wp_effect.commands) {
                if (command.command != "copy") continue;
                if (fbo_map.count(command.target) == 0 || fbo_map.count(command.source) == 0)
                    continue;
                const auto resolved_dst = fbo_map.at(command.target);
                const auto resolved_src = fbo_map.at(command.source);
                img_effect->commands.push_back({ .cmd          = SceneImageEffect::CmdType::Copy,
                                                 .authored_dst = resolved_dst,
                                                 .authored_src = resolved_src,
                                                 .dst          = resolved_dst,
                                                 .src          = resolved_src,
                                                 .afterpos     = command.afterpos });
            }

            bool effect_materials_ok = true;
            for (usize material_index = 0; material_index < wp_effect.materials.size();
                 material_index++) {
                wpscene::WPMaterial material_source = wp_effect.materials.at(material_index);
                std::string         material_output { WE_EFFECT_PPONG_PREFIX_B };
                if (wp_effect.passes.size() > material_index) {
                    const auto& wp_pass = wp_effect.passes.at(material_index);
                    material_source.MergePass(wp_pass);
                    for (const auto& bind : wp_pass.bind) {
                        if (fbo_map.count(bind.name) == 0) continue;
                        if (material_source.textures.size() <= static_cast<usize>(bind.index)) {
                            material_source.textures.resize(static_cast<usize>(bind.index) + 1);
                        }
                        material_source.textures[static_cast<usize>(bind.index)] =
                            fbo_map.at(bind.name);
                    }
                    if (! wp_pass.target.empty() && fbo_map.count(wp_pass.target) != 0) {
                        material_output = fbo_map.at(wp_pass.target);
                    }
                }
                if (material_source.textures.empty()) material_source.textures.resize(1);
                if (material_source.textures[0].empty()) material_source.textures[0] = in_rt;

                auto spEffectNode = std::make_shared<SceneNode>();
                // Same phase contract as image effect passes: no nodeOwners registration, the
                // node id back-references the owning layer, and the name marks the pass.
                spEffectNode->ID() = text_obj.id;
                spEffectNode->SetName(text_obj.name + "::__hanabi_effect_pass_" +
                                      std::to_string(effect_index) + "_" +
                                      std::to_string(material_index));
                WPShaderInfo effect_shader_info;
                effect_shader_info.baseConstSvs = context.global_base_uniforms;
                effect_shader_info.baseConstSvs["g_EffectTextureProjectionMatrix"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
                effect_shader_info.baseConstSvs["g_EffectTextureProjectionMatrixInverse"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());

                SceneMaterial     effect_material;
                WPShaderValueData effect_node_data;
                if (! LoadMaterial(*context.vfs,
                                   material_source,
                                   context.scene.get(),
                                   spEffectNode.get(),
                                   &effect_material,
                                   &effect_node_data,
                                   context.user_properties,
                                   &effect_shader_info)) {
                    effect_materials_ok = false;
                    break;
                }
                if (IsCurrentEffectWriterTarget(material_output)) {
                    final_shader_capability = ResolveFinalShaderCapability(material_source.shader);
                }
                LoadConstvalue(effect_material, material_source, effect_shader_info);
                LoadUserShaderValue(
                    effect_material, material_source, effect_shader_info, context.user_properties);

                // Text effect passes can receive a new bridge quad whenever the shaped text bounds
                // change, for example on a minute rollover from "12:47" to "12:48". Treating these
                // effect meshes as dynamic from graph construction keeps the already-compiled
                // CustomShaderPass hot and lets the renderer upload the changed quad through the
                // same dynamic-buffer path used by particle-like geometry, instead of destroying
                // and rebuilding the shader pass for every text size change.
                auto spMesh                             = std::make_shared<SceneMesh>(true);
                ApplyEffectWriterTransformContract(
                    context,
                    BuildTextEffectMaterialContract(text_obj, *imgEffectLayer),
                    effect_node_data);
                const auto authored_textures = effect_material.textures;
                spMesh->AddMaterial(std::move(effect_material));
                spEffectNode->AddMesh(spMesh);
                RegisterUserShaderValueBindings(context,
                                                material_source,
                                                effect_shader_info,
                                                spEffectNode.get(),
                                                text_obj.id,
                                                text_obj.name);
                // Text layers build their effect materials through this separate bridge path rather
                // than the image-layer effect parser. Register constant-shader scripts here as well
                // so media-driven opacity fades on dynamic song-title/artist text receive playback
                // events and can reveal the updated text layer over the authored placeholder layer.
                RegisterConstantShaderValueBindings(context,
                                                    material_source,
                                                    effect_shader_info,
                                                    spEffectNode.get(),
                                                    text_obj.id,
                                                    text_obj.name,
                                                    wp_effect.id,
                                                    effect_index,
                                                    material_index);
                context.shader_updater->SetNodeData(spEffectNode.get(), effect_node_data);
                img_effect->nodes.push_back({ .authored_output = material_output,
                                              .output = material_output,
                                              .authored_textures = authored_textures,
                                              .sceneNode = spEffectNode });
            }

            if (effect_materials_ok) {
                img_effect->SetLocalVisible(effect_initial_visible);
                if (! wp_effect.visible_json.is_null()) {
                    LOG_INFO("SceneEffectVisibilityResolve: layer=%d effect-id=%d effect-index=%d "
                             "name='%s' authored=%s initial=%s runtime=%s",
                             text_obj.id,
                             wp_effect.id,
                             effect_index,
                             wp_effect.name.c_str(),
                             wp_effect.visible ? "true" : "false",
                             effect_initial_visible ? "true" : "false",
                             effect_runtime_visibility ? "true" : "false");
                }
                imgEffectLayer->AddEffect(img_effect);
            }
        }

        // Mirror the image-layer priority: a dependency source must stay sampleable, otherwise the
        // final writer shader decides. ResolveFinalOutputCapability() still forces the private
        // route for chains with runtime-toggled effects and for composition-source routing.
        if (scene.IsLayerOffscreenDependencySource(text_obj.id)) {
            imgEffectLayer->SetFinalOutputCapability(FinalOutputCapability::PrivateDependency);
        } else {
            imgEffectLayer->SetFinalOutputCapability(final_shader_capability);
        }
        LOG_INFO("SceneTextEffectOutputCapability: layer=%d name='%s' capability=%.*s "
                 "dependency=%s",
                 text_obj.id,
                 text_obj.name.c_str(),
                 static_cast<int>(FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).size()),
                 FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).data(),
                 scene.IsLayerOffscreenDependencySource(text_obj.id) ? "true" : "false");
    }

    if (LayerUsesRoutedParent(text_obj.parent, text_obj.attachment)) {
        ConfigureInheritedParentBinding(context, text_obj.parent, worldNodeData);
        context.scene->sceneGraph->AppendChild(spWorldNode);
    } else {
        AttachNodeToScene(context, spWorldNode, text_obj.parent, text_obj.name, &worldNodeData);
    }

    context.object_nodes[text_obj.id] = spWorldNode;
    context.scene->AddLayerRuntimeNode(text_obj.id, spWorldNode.get());
    context.shader_updater->SetNodeData(spWorldNode.get(), worldNodeData);

    if (spTextNode.get() != spWorldNode.get()) {
        // Same phase contract as the image source node: bridge-owned, never in sceneGraph, no
        // nodeOwners registration, and the node id (set above) back-references the owning layer.
        if (auto* source_bridge = context.scene->FindImageEffectLayer(text_obj.id)) {
            source_bridge->AddDetachedSourceNode(spTextNode);
        }
        context.scene->AddLayerRuntimeNode(text_obj.id, spTextNode.get());
    }

    context.scene->SetTextLayerState(text_obj.id,
                                     TextLayerRuntimeState {
                                         .object            = text_obj,
                                         .primitive         = primitive,
                                         .render_contract   = render_contract,
                                         .applied_alignment = ResolveTextLayerSceneAlignment(text_obj),
                                     });

    ApplyTextLayerNodePlacement(spWorldNode.get(),
                                *context.scene->FindTextLayerState(text_obj.id),
                                text_obj.origin);

    RegisterLayerSceneState(
        context, text_obj.id, text_obj.parent, text_obj.attachment, text_obj.visible);
    context.scene->ApplyLayerVisibility(text_obj.id);
}

void ParseLightObj(ParseContext& context, wpscene::WPLightObject& light_obj) {
    auto node = std::make_shared<SceneNode>(Vector3f(light_obj.origin.data()),
                                            Vector3f(light_obj.scale.data()),
                                            Vector3f(light_obj.angles.data()),
                                            light_obj.name);
    node->ID() = light_obj.id;
    LOG_INFO("SceneLightParsed: layer=%d name='%s' type='%s' radius=%.3f intensity=%.3f",
             light_obj.id,
             light_obj.name.c_str(),
             light_obj.light.c_str(),
             light_obj.radius,
             light_obj.intensity);

    context.scene->lights.emplace_back(std::make_unique<SceneLight>(
        Vector3f(light_obj.color.data()), light_obj.radius, light_obj.intensity));

    auto& light = *(context.scene->lights.back());
    // Workshop scene.pkg and editor assets write prefixed tokens (lpoint/lspot/
    // ldirectional/ltube). Unpacked defaultprojects and older scene.json keep the
    // unprefixed names. Accept both.
    if (light_obj.light == "lpoint" || light_obj.light == "point") {
        light.setType(SceneLightType::Point);
    } else if (light_obj.light == "lspot" || light_obj.light == "spot") {
        light.setType(SceneLightType::Spot);
    } else if (light_obj.light == "ldirectional" || light_obj.light == "directional") {
        light.setType(SceneLightType::Directional);
    } else if (light_obj.light == "ltube" || light_obj.light == "tube") {
        light.setType(SceneLightType::Tube);
    } else {
        light.setType(SceneLightType::Other);
    }
    light.setCastVolumetrics(light_obj.castvolumetrics);
    light.setDensity(light_obj.density);
    light.setVolumetricsExponent(light_obj.volumetricsexponent);
    light.setInnerCone(light_obj.innercone);
    light.setOuterCone(light_obj.outercone);
    light.setCastsShadows(light_obj.castshadows);
    light.setExponent(light_obj.exponent);
    light.setCascadeDistances(light_obj.cascadedistance0, light_obj.cascadedistance1,
                              light_obj.cascadedistance2);
    if (light_obj.usecookie && ! light_obj.cookie.empty()) {
        light.setCookie(light_obj.cookie);
    }
    context.scene->AddLayerRuntimeLight(light_obj.id, &light);
    light.setNode(node);

    if (LayerUsesRoutedParent(light_obj.parent, {})) {
        // A parented light composes the authored ancestor chain exactly like parented image and
        // model layers; its shader-facing world transform is published per frame from the routed
        // resolution. Physically nesting it would only apply the immediate parent's local
        // transform because group ancestors are root-owned routed layers.
        WPShaderValueData light_data;
        ConfigureInheritedParentBinding(context, light_obj.parent, light_data);
        context.scene->sceneGraph->AppendChild(node);
        context.shader_updater->SetNodeData(node.get(), light_data);
    } else {
        AttachNodeToScene(context, node, light_obj.parent, light_obj.name);
    }
    context.object_nodes[light_obj.id] = node;
    context.scene->AddLayerRuntimeNode(light_obj.id, node.get());
    RegisterLayerSceneState(context, light_obj.id, light_obj.parent, {}, light_obj.visible);
    context.scene->ApplyLayerVisibility(light_obj.id);
}

void ParseEmptyObj(ParseContext& context, WPEmptyObject& empty_obj) {
    // 2D camera layers are authored around the centered canvas view, so their node gains the
    // canvas half-size. Perspective scenes keep the authored origin verbatim: the camera layer's
    // world translation is the eye position of the scene view.
    const auto node_origin =
        empty_obj.is_camera_layer && context.scene->cameraOrthographic
            ? context.scene->ResolveCameraLayerNodeTranslation(empty_obj.origin)
            : Vector3f(empty_obj.origin.data());
    auto node  = std::make_shared<SceneNode>(node_origin,
                                             Vector3f(empty_obj.scale.data()),
                                             Vector3f(empty_obj.angles.data()),
                                             empty_obj.name);
    node->ID() = empty_obj.id;

    WPShaderValueData svData;
    svData.parallaxDepth = empty_obj.parallaxDepth;
    svData.parallaxDepthAuthored = empty_obj.parallaxDepthAuthored;
    ConfigureBoneAttachment(context,
                            empty_obj.parent,
                            empty_obj.attachment,
                            Eigen::Affine3f(node->GetLocalTrans().cast<float>()),
                            "object",
                            empty_obj.name,
                            svData);

    if (LayerUsesRoutedParent(empty_obj.parent, empty_obj.attachment)) {
        ConfigureInheritedParentBinding(context, empty_obj.parent, svData);
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, empty_obj.parent, empty_obj.name, &svData);
    }
    context.object_nodes[empty_obj.id] = node;
    context.scene->AddLayerRuntimeNode(empty_obj.id, node.get());
    context.shader_updater->SetNodeData(node.get(), svData);
    RegisterLayerSceneState(
        context, empty_obj.id, empty_obj.parent, empty_obj.attachment, empty_obj.visible);
    context.scene->ApplyLayerVisibility(empty_obj.id);

    if (empty_obj.is_camera_layer) {
        Scene::CameraLayerRuntimeState camera_layer;
        // Wallpaper Engine writes "default" for the normal scene camera. Hanabi's matching
        // orthographic camera is named "global", so normalize the authored token once and keep
        // the rest of the runtime path name-based for future camera targets.
        camera_layer.camera_name =
            empty_obj.camera_name.empty() || empty_obj.camera_name == "default"
                ? "global"
                : empty_obj.camera_name;
        camera_layer.node                    = node;
        camera_layer.origin                  = empty_obj.origin;
        camera_layer.angles                  = empty_obj.angles;
        camera_layer.zoom                    = empty_obj.zoom;
        camera_layer.fov                     = empty_obj.fov;
        const bool first_camera_registration =
            context.scene->FindCameraLayerState(empty_obj.id) == nullptr;
        context.scene->SetCameraLayerState(empty_obj.id, camera_layer);
        if (first_camera_registration) context.scene->cameraLayerOrder.push_back(empty_obj.id);
        context.scene->UpdateActiveCameraLayer();

        size_t camera_path_count = 0;
        if (! empty_obj.camera_path.empty()) {
            nlohmann::json    path_json;
            const std::string asset_path = "/assets/" + empty_obj.camera_path;
            if (context.vfs != nullptr && context.vfs->Contains(asset_path) &&
                PARSE_JSON(fs::GetFileContent(*context.vfs, asset_path), path_json) &&
                path_json.contains("paths") && path_json.at("paths").is_array()) {
                camera_path_count = path_json.at("paths").size();
            }
        }
        LOG_INFO("SceneCameraLayerParsed: id=%d name='%s' camera='%s' origin=[%.3f, %.3f, %.3f] "
                 "zoom=%.3f fov=%.3f visible=%s path='%s' path-count=%zu",
                 empty_obj.id,
                 empty_obj.name.c_str(),
                 camera_layer.camera_name.c_str(),
                 empty_obj.origin[0],
                 empty_obj.origin[1],
                 empty_obj.origin[2],
                 empty_obj.zoom,
                 empty_obj.fov,
                 context.scene->IsLayerVisible(empty_obj.id) ? "true" : "false",
                 empty_obj.camera_path.c_str(),
                 camera_path_count);
    }
}

bool ShapeEffectRequestsDirectDraw(const WPShapeObject& shape_obj) {
    // Wallpaper Engine marks shader-authored shape output with the DIRECTDRAW combo on the effect
    // chain. Ask the parsed effect model for that shader contract instead of inferring it from the
    // outer shape geometry string; the geometry name only says what primitive the editor displayed,
    // while DIRECTDRAW is the actual render-path switch that means no image/model source exists.
    return std::any_of(shape_obj.effects.begin(), shape_obj.effects.end(), [](const auto& effect) {
        return effect.HasEnabledCombo("DIRECTDRAW");
    });
}

std::string ToLowerAscii(std::string_view text) {
    std::string lower;
    lower.reserve(text.size());
    for (unsigned char ch : text) {
        lower.push_back(static_cast<char>(std::tolower(ch)));
    }
    return lower;
}

bool EffectLooksLikeLightshafts(const wpscene::WPImageEffect& effect) {
    auto matches = [](std::string_view text) {
        const std::string lower = ToLowerAscii(text);
        return lower.find("lightshafts") != std::string::npos ||
               lower.find("light shafts") != std::string::npos;
    };

    if (matches(effect.name)) return true;
    for (const auto& material : effect.materials) {
        if (matches(material.shader)) return true;
    }
    return false;
}

void MergeLightshaftPointValues(
    const std::unordered_map<std::string, std::vector<float>>& values,
    std::array<std::optional<std::array<float, 2>>, 4>&        points) {
    for (const auto& [name, value] : values) {
        if (value.size() < 2 || ! std::isfinite(value[0]) || ! std::isfinite(value[1])) continue;

        const std::string lower = ToLowerAscii(name);
        for (usize i = 0; i < points.size(); i++) {
            const std::string point_name = "point" + std::to_string(i);
            if (lower == point_name || lower == "g_" + point_name) {
                points[i] = std::array<float, 2> { value[0], value[1] };
                break;
            }
        }
    }
}

std::optional<std::array<float, 4>>
ResolveLightshaftsDirectDrawFinalTexCoordBounds(const WPShapeObject& shape_obj) {
    for (const auto& effect : shape_obj.effects) {
        if (! EffectLooksLikeLightshafts(effect)) continue;

        std::array<std::optional<std::array<float, 2>>, 4> points;
        for (usize material_index = 0; material_index < effect.materials.size(); material_index++) {
            MergeLightshaftPointValues(effect.materials[material_index].constantshadervalues,
                                       points);
            if (material_index < effect.passes.size()) {
                MergeLightshaftPointValues(effect.passes[material_index].constantshadervalues,
                                           points);
            }
        }

        if (! std::all_of(points.begin(), points.end(), [](const auto& point) {
                return point.has_value();
            })) {
            continue;
        }

        std::array<float, 4> bounds { 0.0f, 0.0f, 1.0f, 1.0f };
        for (const auto& point : points) {
            bounds[0] = std::min(bounds[0], point->at(0));
            bounds[1] = std::min(bounds[1], point->at(1));
            bounds[2] = std::max(bounds[2], point->at(0));
            bounds[3] = std::max(bounds[3], point->at(1));
        }

        const bool expands_default_quad = bounds[0] < 0.0f || bounds[1] < 0.0f ||
                                          bounds[2] > 1.0f || bounds[3] > 1.0f;
        if (expands_default_quad) return bounds;
    }

    return std::nullopt;
}

std::array<float, 2> ResolveImplicitDirectDrawShapeVisualSize(const ParseContext& context) {
    const float scene_width  = static_cast<float>(std::max(1, context.ortho_w));
    const float scene_height = static_cast<float>(std::max(1, context.ortho_h));
    const float short_edge   = std::min(scene_width, scene_height);
    return { short_edge, short_edge };
}

struct DirectDrawShapeMetrics {
    std::array<float, 2> visual_size { 1.0f, 1.0f };
    std::array<float, 2> effect_source_size { 0.0f, 0.0f };
    bool                 effect_source_size_is_pixel_extent { false };
    bool                 final_texcoord_bounds_enabled { false };
    std::array<float, 4> final_texcoord_bounds { 0.0f, 0.0f, 1.0f, 1.0f };
    const char*          visual_policy { "authored-size" };
    const char*          effect_source_policy { "layer-size" };
    const char*          final_uv_policy { "default" };
};

// Shape effect chains run in a destination target that is half the canvas in each dimension
// (integer halves), independent of the shape's own card size. The value is already a pixel
// extent, so it bypasses perspective density scaling; the destination intern applies the common
// minimum-extent clamp.
std::array<float, 2> ResolveDirectDrawShapeEffectTargetSize(const ParseContext& context) {
    return { static_cast<float>(context.ortho_w / 2), static_cast<float>(context.ortho_h / 2) };
}

DirectDrawShapeMetrics ResolveDirectDrawShapeMetrics(const ParseContext& context,
                                                     const WPShapeObject& shape_obj) {
    DirectDrawShapeMetrics metrics;
    // Shapes keep two separate size contracts. The intermediate effect buffers are always the
    // fixed half-canvas shape destination, whether or not the shape carries an authored size;
    // the authored size (or the implicit short-edge square) only shapes the visible card. This
    // preserves world transforms without forcing the layer through the fullscreen postprocess
    // path.
    metrics.effect_source_size                 = ResolveDirectDrawShapeEffectTargetSize(context);
    metrics.effect_source_size_is_pixel_extent = true;
    metrics.effect_source_policy               = "half-canvas";
    if (shape_obj.has_size) {
        metrics.visual_size = shape_obj.size;
    } else {
        metrics.visual_size   = ResolveImplicitDirectDrawShapeVisualSize(context);
        metrics.visual_policy = "implicit-short-edge-square";
    }

    if (auto bounds = ResolveLightshaftsDirectDrawFinalTexCoordBounds(shape_obj);
        bounds.has_value()) {
        metrics.final_texcoord_bounds_enabled = true;
        metrics.final_texcoord_bounds         = *bounds;
        metrics.final_uv_policy               = "lightshafts-control-points";
    }

    return metrics;
}

void ParseShapeObj(ParseContext& context, WPShapeObject& shape_obj) {

    const bool direct_draw_shape = ShapeEffectRequestsDirectDraw(shape_obj);
    if (! direct_draw_shape) {
        // Unsupported or effect-less shape layers still need to behave like transform containers.
        // Registering a normal empty object preserves parent bindings, scripts, and child ordering
        // while making the missing drawable path explicit in the log instead of silently dropping
        // the authored layer.
        LOG_INFO("SceneShapeObjectFallback: id=%d name='%s' shape='%s' effects=%zu",
                 shape_obj.id,
                 shape_obj.name.c_str(),
                 shape_obj.shape.c_str(),
                 shape_obj.effects.size());

        WPEmptyObject empty_obj;
        empty_obj.id              = shape_obj.id;
        empty_obj.name            = shape_obj.name;
        empty_obj.origin          = shape_obj.origin;
        empty_obj.scale           = shape_obj.scale;
        empty_obj.angles          = shape_obj.angles;
        empty_obj.parallaxDepth   = shape_obj.parallaxDepth;
        empty_obj.parallaxDepthAuthored = shape_obj.parallaxDepthAuthored;
        empty_obj.visible         = shape_obj.visible;
        empty_obj.visible_binding = shape_obj.visible_binding;
        empty_obj.parent          = shape_obj.parent;
        empty_obj.attachment      = shape_obj.attachment;
        ParseEmptyObj(context, empty_obj);
        return;
    }

    auto& vfs = *context.vfs;

    wpscene::WPMaterial transparent_source_material;
    nlohmann::json      transparent_source_json;
    if (! PARSE_JSON(fs::GetFileContent(vfs, "/assets/materials/util/effectpassthrough.json"),
                     transparent_source_json) ||
        ! transparent_source_material.FromJson(transparent_source_json)) {
        LOG_ERROR(
            "SceneShapeDirectDraw: layer=%d name='%s' failed to load transparent source material",
            shape_obj.id,
            shape_obj.name.c_str());
        return;
    }

    EnsureSystemTextureRegistered(*context.scene, kSyntheticDirectDrawShapeTextureName);
    if (transparent_source_material.textures.empty()) {
        transparent_source_material.textures.resize(1);
    }
    transparent_source_material.textures[0] = std::string(kSyntheticDirectDrawShapeTextureName);
    // Direct-draw shape effects author their visible pixels inside the effect shader and leave
    // untouched areas with alpha zero. Treat the final synthetic source as additive for every
    // shape direct-draw layer: the shader output is authored as generated contribution over the
    // existing scene, and translucent alpha compositing would multiply the destination by
    // `1 - alpha`, causing rays and other generated highlights to darken the wallpaper instead of
    // adding energy. The source texture itself is transparent, so additive blending keeps empty
    // regions neutral while preserving the intended brightening behavior.
    const std::string_view direct_draw_final_blend = "additive";
    transparent_source_material.blending           = std::string(direct_draw_final_blend);

    const DirectDrawShapeMetrics metrics = ResolveDirectDrawShapeMetrics(context, shape_obj);

    // Shape direct-draw layers have no image asset because the effect shader owns the visible
    // pixels (`DIRECTDRAW=1`). Synthesize a fully transparent image source only to reuse the
    // established image-effect camera, ping-pong render targets, visibility contracts, and final
    // composite path; the authored shape string stays metadata, not a render-path discriminator.
    wpscene::WPImageObject image_obj;
    image_obj.id               = shape_obj.id;
    image_obj.name             = shape_obj.name;
    image_obj.origin           = shape_obj.origin;
    image_obj.scale            = shape_obj.scale;
    image_obj.angles           = shape_obj.angles;
    image_obj.size             = metrics.visual_size;
    image_obj.parallaxDepth    = shape_obj.parallaxDepth;
    image_obj.parallaxDepthAuthored = shape_obj.parallaxDepthAuthored;
    image_obj.color            = shape_obj.color;
    image_obj.alpha            = shape_obj.alpha;
    image_obj.brightness       = shape_obj.brightness;
    image_obj.visible          = shape_obj.visible;
    image_obj.visible_binding  = shape_obj.visible_binding;
    image_obj.image            = "__hanabi_shape_directdraw";
    image_obj.parent           = shape_obj.parent;
    image_obj.attachment       = shape_obj.attachment;
    image_obj.effectSourceSize              = metrics.effect_source_size;
    image_obj.effectSourceSizeIsPixelExtent = metrics.effect_source_size_is_pixel_extent;
    image_obj.effectFinalTexCoordBoundsEnabled = metrics.final_texcoord_bounds_enabled;
    image_obj.effectFinalTexCoordBounds        = metrics.final_texcoord_bounds;
    image_obj.material         = std::move(transparent_source_material);
    image_obj.effects          = std::move(shape_obj.effects);
    image_obj.nopadding        = true;
    image_obj.config.finalOutputCapability = FinalOutputCapability::SceneAuthoredWriter;

    LOG_INFO("SceneShapeDirectDraw: materialize layer=%d name='%s' shape='%s' effects=%zu "
             "visual-size=[%.3f, %.3f] visual-policy=%s effect-source-policy=%s "
             "effect-source-size=[%.3f, %.3f] authored-size=%s final-uv-policy=%s "
             "final-uv-bounds=[%.3f, %.3f, %.3f, %.3f] transparent-texture='%.*s' "
             "final-blend='%.*s'",
             image_obj.id,
             image_obj.name.c_str(),
             shape_obj.shape.c_str(),
             image_obj.effects.size(),
             image_obj.size[0],
             image_obj.size[1],
             metrics.visual_policy,
             metrics.effect_source_policy,
             image_obj.effectSourceSize[0],
             image_obj.effectSourceSize[1],
             shape_obj.has_size ? "true" : "false",
             metrics.final_uv_policy,
             metrics.final_texcoord_bounds[0],
             metrics.final_texcoord_bounds[1],
             metrics.final_texcoord_bounds[2],
             metrics.final_texcoord_bounds[3],
             static_cast<int>(kSyntheticDirectDrawShapeTextureName.size()),
             kSyntheticDirectDrawShapeTextureName.data(),
             static_cast<int>(direct_draw_final_blend.size()),
             direct_draw_final_blend.data());

    ParseImageObj(context, image_obj);
}

template<typename T>
void AddWPObject(std::vector<WPObjectVar>& objs, const nlohmann::json& json_obj, fs::VFS& vfs,
                 const UserPropertyMap* user_properties) {
    T wpobj;
    if (! wpobj.FromJson(json_obj, vfs)) {
        LOG_ERROR("parse scene object failed, name: %s", wpobj.name.c_str());
        return;
    }

    wpobj.visible =
        ResolveObjectVisibility(wpobj.visible, wpobj.visible_binding, user_properties);
    if constexpr (std::is_same_v<T, wpscene::WPImageObject>) {
        // This image parse log intentionally mirrors the runtime geometry fields that decide
        // whether utility layers become drawable. It makes project-layer regressions visible in
        // run logs before the render graph has a chance to create a framebuffer feedback pass.
        LOG_INFO("SceneObjectParsed: source=scene-load kind=image id=%d name='%s' "
                 "origin=[%.3f, %.3f, %.3f] size=[%.3f, %.3f] fullscreen=%s autosize=%s "
                 "projectlayer=%s image='%s' effects=%zu",
                 wpobj.id,
                 wpobj.name.c_str(),
                 wpobj.origin[0],
                 wpobj.origin[1],
                 wpobj.origin[2],
                 wpobj.size[0],
                 wpobj.size[1],
                 wpobj.fullscreen ? "true" : "false",
                 wpobj.autosize ? "true" : "false",
                 wpobj.projectlayer ? "true" : "false",
                 wpobj.image.c_str(),
                 wpobj.effects.size());
    } else if constexpr (std::is_same_v<T, WPShapeObject>) {
        // Shape/direct-draw layers do not carry an image model, so this parser log is the only
        // early proof that the layer was classified as drawable effect content rather than as a
        // transform-only empty object.
        LOG_INFO("SceneObjectParsed: source=scene-load kind=shape id=%d name='%s' "
                 "origin=[%.3f, %.3f, %.3f] size=[%.3f, %.3f] has-size=%s shape='%s' effects=%zu",
                 wpobj.id,
                 wpobj.name.c_str(),
                 wpobj.origin[0],
                 wpobj.origin[1],
                 wpobj.origin[2],
                 wpobj.size[0],
                 wpobj.size[1],
                 wpobj.has_size ? "true" : "false",
                 wpobj.shape.c_str(),
                 wpobj.effects.size());
    } else if constexpr (std::is_same_v<T, WPModelObject>) {
        // Model objects are intentionally logged before materialization because a missing model
        // parse would otherwise look like an empty scene: the object has no image/particle/text
        // discriminator, so this line proves the 3D-specific dispatch path claimed it.
        LOG_INFO("SceneObjectParsed: source=scene-load kind=model id=%d name='%s' model='%s' "
                 "origin=[%.3f, %.3f, %.3f] reflected=%s skin=%d",
                 wpobj.id,
                 wpobj.name.c_str(),
                 wpobj.model.c_str(),
                 wpobj.origin[0],
                 wpobj.origin[1],
                 wpobj.origin[2],
                 wpobj.reflected ? "true" : "false",
                 wpobj.skin);
    }
    objs.push_back(wpobj);
}

std::optional<int32_t> GetObjectId(const WPObjectVar& obj) {
    return std::visit(
        visitor::overload {
            [](const auto& value) -> std::optional<int32_t> {
                return value.id;
            },
        },
        obj);
}

std::string GetObjectName(const WPObjectVar& obj) {
    return std::visit(
        [](const auto& value) {
            return value.name;
        },
        obj);
}

// Registers one scene.json object as the authored SceneObject identity: id, kind, name, authored
// transform, effect count, and the passthrough flag. Behavior-facing fields (local visibility,
// parent binding, image runtime state) are intentionally not written here; those keep flowing
// through the existing registration points so runtime semantics stay exactly as before.
template <typename ValueT>
void FillSceneObjectIdentityFor(Scene& scene, const ValueT& value) {
    if (value.id == 0) return;
    auto& object = scene.EnsureSceneObject(value.id);
    object.SetName(value.name);
    if constexpr (std::is_same_v<ValueT, wpscene::WPSoundObject>) {
        object.SetKind(SceneObjectKind::Sound);
        return;
    } else {
        object.SetAuthoredTransform(value.origin, value.scale, value.angles);
        if constexpr (std::is_same_v<ValueT, wpscene::WPImageObject>) {
            object.SetKind(SceneObjectKind::Image);
            object.SetEffectCount(static_cast<int32_t>(value.effects.size()));
            object.SetPassthrough(value.config.passthrough);
        } else if constexpr (std::is_same_v<ValueT, wpscene::WPParticleObject>) {
            object.SetKind(SceneObjectKind::Particle);
        } else if constexpr (std::is_same_v<ValueT, wpscene::WPTextObject>) {
            object.SetKind(SceneObjectKind::Text);
            object.SetEffectCount(static_cast<int32_t>(value.effects.size()));
        } else if constexpr (std::is_same_v<ValueT, wpscene::WPLightObject>) {
            object.SetKind(SceneObjectKind::Light);
        } else if constexpr (std::is_same_v<ValueT, WPModelObject>) {
            object.SetKind(SceneObjectKind::Model);
        } else if constexpr (std::is_same_v<ValueT, WPShapeObject>) {
            object.SetKind(SceneObjectKind::Shape);
            object.SetEffectCount(static_cast<int32_t>(value.effects.size()));
        } else if constexpr (std::is_same_v<ValueT, WPEmptyObject>) {
            object.SetKind(value.is_camera_layer ? SceneObjectKind::Camera
                                                 : SceneObjectKind::Empty);
        }
    }
}

void FillSceneObjectIdentity(Scene& scene, const WPObjectVar& obj) {
    std::visit(
        [&scene](const auto& value) {
            FillSceneObjectIdentityFor(scene, value);
        },
        obj);
}

bool InitDynamicParseContext(ParseContext& context, Scene& scene,
                             const UserPropertyMap* user_properties) {
    if (scene.shaderValueUpdater == nullptr || scene.vfs == nullptr) return false;

    auto* shader_updater = dynamic_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());
    if (shader_updater == nullptr) return false;

    context.scene           = std::shared_ptr<Scene>(&scene, [](Scene*) {
    });
    context.shader_updater  = shader_updater;
    context.vfs             = scene.vfs.get();
    context.user_properties = user_properties;
    context.ortho_w         = scene.ortho[0];
    context.ortho_h         = scene.ortho[1];
    PopulateGlobalBaseUniforms(context, scene);

    auto effect_it      = scene.cameras.find("effect");
    auto global_it      = scene.cameras.find("global");
    auto perspective_it = scene.cameras.find("global_perspective");
    if (effect_it != scene.cameras.end())
        context.effect_camera_node = effect_it->second->GetAttachedNode();
    if (global_it != scene.cameras.end())
        context.global_camera_node = global_it->second->GetAttachedNode();
    if (perspective_it != scene.cameras.end()) {
        context.global_perspective_camera_node = perspective_it->second->GetAttachedNode();
    }

    std::unordered_map<SceneNode*, std::shared_ptr<SceneNode>> shared_nodes;
    CollectSceneNodeRefs(scene.sceneGraph, shared_nodes);
    for (const auto& [layer_id, object] : scene.sceneObjects) {
        if (object == nullptr || ! object->HasLayerNodeSlot()) continue;
        // A registered slot may hold a null handle; downstream lookups tolerate that exactly like
        // the former map's null entries did.
        SceneNode* node    = object->LayerNode();
        auto       node_it = shared_nodes.find(node);
        if (node_it != shared_nodes.end()) {
            context.object_nodes[layer_id] = node_it->second;
        }

        if (const auto* node_data = shader_updater->GetNodeData(node);
            node_data != nullptr && node_data->puppet_layer.Puppet() != nullptr) {
            context.object_puppets[layer_id] = node_data->puppet_layer.Puppet();
        }
    }
    return true;
}

bool ParseDynamicSceneObject(ParseContext& context, const nlohmann::json& object_json,
                             const UserPropertyMap* user_properties, int32_t* out_layer_id) {
    const auto resolve_visibility = [&](auto& object) {
        object.visible =
            ResolveObjectVisibility(object.visible, object.visible_binding, user_properties);
    };

    if (object_json.contains("image") && ! object_json.at("image").is_null()) {
        wpscene::WPImageObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);

        FillSceneObjectIdentityFor(*context.scene, object);
        ParseImageObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("particle") && ! object_json.at("particle").is_null()) {
        wpscene::WPParticleObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        FillSceneObjectIdentityFor(*context.scene, object);
        ParseParticleObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("light") && ! object_json.at("light").is_null()) {
        wpscene::WPLightObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        FillSceneObjectIdentityFor(*context.scene, object);
        ParseLightObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("sound") && ! object_json.at("sound").is_null()) {
        wpscene::WPSoundObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        if (context.scene->soundManager == nullptr) return false;
        const auto sound_handle =
            WPSoundParser::Parse(object, *context.vfs, *context.scene->soundManager);
        if (sound_handle == 0) return false;
        FillSceneObjectIdentityFor(*context.scene, object);
        context.scene->SetLayerSoundHandle(object.id, sound_handle);
        if (out_layer_id) *out_layer_id = object.id;
        return true;
    }

    if (object_json.contains("text") && ! object_json.at("text").is_null()) {
        wpscene::WPTextObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        FillSceneObjectIdentityFor(*context.scene, object);
        ParseTextObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("model") && ! object_json.at("model").is_null()) {
        WPModelObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        FillSceneObjectIdentityFor(*context.scene, object);
        ParseModelObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    WPEmptyObject object;
    if (! object.FromJson(object_json, *context.vfs)) return false;
    resolve_visibility(object);
    FillSceneObjectIdentityFor(*context.scene, object);
    ParseEmptyObj(context, object);
    context.scene->SetLayerLocalVisibility(object.id, object.visible);
    context.scene->ApplyLayerVisibility(object.id);
    if (out_layer_id) *out_layer_id = object.id;
    return context.object_nodes.count(object.id) != 0;
}
} // namespace


bool wallpaper::CreateDynamicSceneLayer(
    Scene& scene, const nlohmann::json& object_json, const UserPropertyMap* user_properties,
    std::vector<WPSceneScriptRegistration>* out_binding_registrations,
    std::vector<WPSceneScriptRegistration>* out_script_registrations,
    std::vector<WPSceneScriptRegistration>* out_property_animation_registrations,
    std::string* out_initial_config_json, int32_t* out_layer_id) {
    if (! object_json.is_object()) return false;

    ParseContext context {};
    if (! InitDynamicParseContext(context, scene, user_properties)) return false;

    nlohmann::json normalized_object_json = object_json;
    int32_t        layer_id               = 0;
    GET_JSON_NAME_VALUE_NOWARN(normalized_object_json, "id", layer_id);
    if (layer_id <= 0 || scene.HasLayerNodeSlot(layer_id) ||
        ! scene.GetLayerRuntimeNodes(layer_id).empty()) {
        layer_id                     = AllocateDynamicLayerId(scene);
        normalized_object_json["id"] = layer_id;
    }

    if (! ParseDynamicSceneObject(context, normalized_object_json, user_properties, &layer_id)) {
        return false;
    }

    auto       node_it = context.object_nodes.find(layer_id);
    SceneNode* layer_node =
        node_it != context.object_nodes.end() && node_it->second ? node_it->second.get() : nullptr;
    const bool has_sound_runtime = scene.GetLayerSoundHandle(layer_id).has_value();
    if (layer_node == nullptr && ! has_sound_runtime) return false;

    const auto binding_start            = scene.bindingRegistrations.size();
    const auto property_animation_start = scene.propertyAnimationRegistrations.size();
    const auto script_start             = scene.scriptRegistrations.size();
    RegisterSceneScriptsForObject(context, normalized_object_json);

    scene.layerOrder.push_back(layer_id);
    scene.SetLayerNode(layer_id, layer_node);
    scene.SetLayerInitialConfigJson(layer_id, normalized_object_json.dump());
    std::string layer_name = layer_node != nullptr
                                 ? layer_node->Name()
                                 : normalized_object_json.value("name", std::string {});
    if (! layer_name.empty()) {
        scene.layerNameToId.emplace(layer_name, layer_id);
    }

    if (out_binding_registrations != nullptr) {
        out_binding_registrations->assign(scene.bindingRegistrations.begin() + binding_start,
                                          scene.bindingRegistrations.end());
    }
    if (out_property_animation_registrations != nullptr) {
        out_property_animation_registrations->assign(scene.propertyAnimationRegistrations.begin() +
                                                         property_animation_start,
                                                     scene.propertyAnimationRegistrations.end());
    }
    if (out_script_registrations != nullptr) {
        out_script_registrations->assign(scene.scriptRegistrations.begin() + script_start,
                                         scene.scriptRegistrations.end());
    }
    if (out_initial_config_json != nullptr) {
        // The record was stored just above, so the pointer is always valid here.
        *out_initial_config_json = *scene.GetLayerInitialConfigJson(layer_id);
    }
    if (out_layer_id != nullptr) {
        *out_layer_id = layer_id;
    }
    return true;
}

std::shared_ptr<Scene> WPSceneParser::Parse(std::string_view scene_id, const std::string& buf,
                                            fs::VFS& vfs, audio::SoundManager& sm,
                                            const UserPropertyMap*  user_properties,
                                            double                  text_render_scale,
                                            std::array<uint32_t, 2> output_extent) {
    nlohmann::json json;
    if (! PARSE_JSON(buf, json)) return nullptr;

    ScopedJsonUserProperties json_user_scope(user_properties, &json);

    wpscene::WPScene sc;
    sc.FromJson(json);
    //	LOG_INFO(nlohmann::json(sc).dump(4));

    ParseContext context {};
    context.user_properties = user_properties;

    std::vector<WPObjectVar>                 wp_objs;
    std::unordered_map<int32_t, std::string> initial_layer_config_json_by_id;
    std::unordered_set<int32_t>              dependency_source_ids;

    for (auto& obj : json.at("objects")) {
        if (obj.contains("dependencies") && obj.at("dependencies").is_array()) {
            for (const auto& dependency : obj.at("dependencies")) {
                int32_t dependency_id = 0;
                GET_JSON_VALUE_NOWARN(dependency, dependency_id);
                if (dependency_id != 0) dependency_source_ids.insert(dependency_id);
            }
        }
    }

    bool has_3d_models = false;
    for (auto& obj : json.at("objects")) {
        int32_t     object_id = 0;
        GET_JSON_NAME_VALUE_NOWARN(obj, "id", object_id);

        // MSAA enablement: any non-null objects[].model counts, including hidden layers. An
        // image path of models/foo.json does not.
        if (obj.contains("model") && ! obj.at("model").is_null()) {
            has_3d_models = true;
        }

        if (object_id != 0) {
            // Keep the exact authored JSON for originalOrigin and dynamic script queries. Object
            // visibility never changes membership in the authored list, so every id below also
            // receives a concrete SceneObject.
            initial_layer_config_json_by_id[object_id] = obj.dump();
        }

        if (obj.contains("image") && ! obj.at("image").is_null()) {
            AddWPObject<wpscene::WPImageObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("particle") && ! obj.at("particle").is_null()) {
            AddWPObject<wpscene::WPParticleObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("sound") && ! obj.at("sound").is_null()) {
            AddWPObject<wpscene::WPSoundObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("light") && ! obj.at("light").is_null()) {
            AddWPObject<wpscene::WPLightObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("text") && ! obj.at("text").is_null()) {
            AddWPObject<wpscene::WPTextObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("model") && ! obj.at("model").is_null()) {
            // Scene-level 3D models have their own parser/materialization path. Classify them
            // before the generic empty fallback so a model-only layer cannot silently become a
            // transform placeholder.
            AddWPObject<WPModelObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("shape") && ! obj.at("shape").is_null()) {
            // Wallpaper Engine uses shape objects for direct-draw effects such as Light Shafts.
            // They have no `image` field, so they must be classified before the generic empty
            // object fallback or their effect chain never reaches the render graph.
            AddWPObject<WPShapeObject>(wp_objs, obj, vfs, user_properties);
        } else {
            AddWPObject<WPEmptyObject>(wp_objs, obj, vfs, user_properties);
        }
    }

    for (const auto& obj : wp_objs) {
        std::visit(visitor::overload {
                       [&context](const wpscene::WPImageObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const wpscene::WPParticleObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const wpscene::WPLightObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const wpscene::WPTextObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const WPModelObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const WPShapeObject& obj) {
                           // Shape effect layers participate in the same inherited-parent ordering
                           // as images; otherwise a parented direct-draw light shaft would resolve
                           // outside the transform that authored its final screen position.
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const WPEmptyObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [](const wpscene::WPSoundObject&) {
                       },
                   },
                   obj);
    }

    if (sc.general.orthogonalprojection.auto_) {
        i32 w = 0, h = 0;
        for (auto& obj : wp_objs) {
            auto*                       img            = std::get_if<wpscene::WPImageObject>(&obj);
            auto*                       shape          = std::get_if<WPShapeObject>(&obj);
            const std::array<float, 2>* candidate_size = nullptr;
            if (img != nullptr) {
                candidate_size = &img->size;
            } else if (shape != nullptr && shape->has_size) {
                // Authored shape sizes can define the projection when no larger image layer exists;
                // implicit full-screen shape sizes are resolved later from the final projection and
                // must not feed this auto-projection bootstrap loop.
                candidate_size = &shape->size;
            }
            if (candidate_size == nullptr) continue;
            i32 size = (i32)(candidate_size->at(0) * candidate_size->at(1));
            if (size > w * h) {
                w = (i32)candidate_size->at(0);
                h = (i32)candidate_size->at(1);
            }
        }
        sc.general.orthogonalprojection.width  = w;
        sc.general.orthogonalprojection.height = h;
    }

    InitContext(context, vfs, sc, scene_id);
    for (const auto& obj : wp_objs) {
        // Every authored object becomes exactly one SceneObject in parse order before per-type
        // materialization runs. Visibility only controls draw execution.
        FillSceneObjectIdentity(*context.scene, obj);
    }
    context.scene->has3dModels  = has_3d_models;
    context.scene->soundManager = &sm;
    // The output framebuffer already exists when a scene loads. Record its extent before objects
    // are materialized so fullscreen layers can size their effect targets from it; the renderer
    // refreshes the same field with the identical extent when it frames the first output.
    context.scene->physicalOutputExtent = output_extent;
    // Text atlases and effect ping-pong stay in the authored letter box. Desktop density is
    // applied when those results are composited, not by rebuilding glyphs at the output scale.
    (void)text_render_scale;
    context.scene->textRenderScale = 1.0;
    for (const auto dependency_source_id : dependency_source_ids) {
        context.scene->MarkLayerOffscreenDependencySource(dependency_source_id);
    }
    if (user_properties) {
        context.scene->userProperties = *user_properties;
    } else {
        context.scene->userProperties.clear();
    }
    ParseCamera(context, sc);

    {
        context.scene->renderTargets[SpecTex_Default.data()] = {
            .width     = context.ortho_w,
            .height    = context.ortho_h,
            .mapWidth  = context.ortho_w,
            .mapHeight = context.ortho_h,
            .bind      = { .enable = true, .screen = true },
        };
        // Stable compose snapshot for `_rt_default` self-writes. Screen-bound so a live output
        // resize keeps the snapshot the same extent as the compose target it mirrors.
        context.scene->renderTargets[SpecTex_DefaultPingPong.data()] = {
            .width     = context.ortho_w,
            .height    = context.ortho_h,
            .mapWidth  = context.ortho_w,
            .mapHeight = context.ortho_h,
            .bind      = { .enable = true, .screen = true },
        };
        ConfigureSceneMsaa(*context.scene);
        context.scene->renderTargets[WE_MIP_MAPPED_FRAME_BUFFER.data()] = {
            .width      = context.ortho_w,
            .height     = context.ortho_h,
            .mapWidth   = context.ortho_w,
            .mapHeight  = context.ortho_h,
            .has_mipmap = true,
            .bind       = { .enable = true, .name = SpecTex_Default.data() }
        };
        context.scene->renderTargets[std::string(SpecTex_ShadowAtlas)] = {
            .width           = 2,
            .height          = 2,
            .mapWidth        = 2,
            .mapHeight       = 2,
            .allowReuse      = false,
            .comparisonDepth = true,
        };
    }
    context.scene->scene_id = scene_id;
    // Scene Bloom owns a synthetic shader and its cache keys include the scene id, so it must be
    // built only after the parse context has reached the same identity state as authored shaders.
    ConfigureSceneBloomPass(context);

    context.scene->lighting = {};
    for (const auto& obj : wp_objs) {
        const auto* light_obj = std::get_if<wpscene::WPLightObject>(&obj);
        if (light_obj == nullptr) continue;
        const auto& token = light_obj->light;
        if (token == "lpoint" || token == "point") {
            context.scene->lighting.point++;
            context.scene->lighting.point_shadow.push_back(light_obj->castshadows ? 1 : 0);
        } else if (token == "lspot" || token == "spot") {
            context.scene->lighting.spot++;
            context.scene->lighting.spot_shadow.push_back(light_obj->castshadows ? 1 : 0);
            context.scene->lighting.spot_cookie.push_back(
                (light_obj->usecookie && ! light_obj->cookie.empty()) ? 1 : 0);
        } else if (token == "ldirectional" || token == "directional") {
            context.scene->lighting.directional++;
            context.scene->lighting.directional_shadow.push_back(light_obj->castshadows ? 1 : 0);
        } else if (token == "ltube" || token == "tube") {
            context.scene->lighting.tube++;
        }
    }

    for (WPObjectVar& obj : wp_objs) {
        std::visit(visitor::overload {
                       [&context](wpscene::WPImageObject& obj) {
                           ParseImageObj(context, obj);
                       },
                       [&context](wpscene::WPParticleObject& obj) {
                           ParseParticleObj(context, obj);
                       },
                       [&context, &sm](wpscene::WPSoundObject& obj) {
                           context.scene->SetLayerSoundHandle(
                               obj.id, WPSoundParser::Parse(obj, *context.vfs, sm));
                       },
                       [&context](wpscene::WPLightObject& obj) {
                           ParseLightObj(context, obj);
                       },
                       [&context](wpscene::WPTextObject& obj) {
                           ParseTextObj(context, obj);
                       },
                       [&context](WPModelObject& obj) {
                           ParseModelObj(context, obj);
                       },
                       [&context](WPShapeObject& obj) {
                           ParseShapeObj(context, obj);
                       },
                       [&context](WPEmptyObject& obj) {
                           ParseEmptyObj(context, obj);
                       },
                   },
                   obj);
    }

    ApplyMissingImageParallaxFallbacks(context, wp_objs);

    RegisterSceneScripts(context, json);

    context.scene->layerOrder.clear();
    context.scene->ClearAllLayerNodeSlots();
    context.scene->ClearAllLayerInitialConfigJson();
    context.scene->layerNameToId.clear();
    for (const auto& obj : wp_objs) {
        const auto object_id = GetObjectId(obj);
        if (! object_id.has_value()) continue;

        auto node_it = context.object_nodes.find(*object_id);
        context.scene->layerOrder.push_back(*object_id);
        context.scene->SetLayerNode(*object_id,
                                    node_it != context.object_nodes.end() && node_it->second
                                        ? node_it->second.get()
                                        : nullptr);
        if (auto config_it = initial_layer_config_json_by_id.find(*object_id);
            config_it != initial_layer_config_json_by_id.end()) {
            context.scene->SetLayerInitialConfigJson(*object_id, config_it->second);
        }

        const auto node_name = node_it != context.object_nodes.end() && node_it->second
                                   ? node_it->second->Name()
                                   : GetObjectName(obj);
        if (! node_name.empty()) {
            // Scene scripts resolve getLayer(name) to one layer per name: the earliest authored
            // object wins when several layers share a name. First-write registration keeps that
            // contract; a last-write map would silently retarget script writes (planet radius,
            // origins) onto later same-named HUD helper layers.
            context.scene->layerNameToId.emplace(node_name, *object_id);
        }
    }

    context.scene->ApplyAllLayerVisibility();

    ConfigureSceneVolumetricsImpl(*context.scene, *context.vfs);

    return context.scene;
}

std::shared_ptr<Scene> WPSceneParser::Parse(std::string_view scene_id, const std::string& buf,
                                            fs::VFS& vfs, audio::SoundManager& sm) {
    return Parse(scene_id, buf, vfs, sm, nullptr);
}

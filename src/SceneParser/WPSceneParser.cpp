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
#include "Scene/SceneDestinationTarget.h"
#include "Scene/SceneShapeGeometry.h"
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

struct ImageDestinationExtent {
    std::array<int32_t, 2> extent { 0, 0 };
    // 'n' when the layer's source texture is point-sampled, otherwise 'b'.
    char                   suffix { 'b' };
    const char*            policy { "" };
    bool                   uses_card_size { false };
};

// The pixel extent of an effect-backed image layer's destination targets and effect FBOs.
//
// It is the content size of the layer's first material texture: one frame for a sprite sheet,
// the output framebuffer for a fullscreen layer (its texture is the framebuffer itself), the
// content rectangle for a texture or a named render target. Two cases use the card size
// (ceil of the authored size) instead: a layer without a texture, and a passthrough or solid
// layer that is not fullscreen and not instanced. Shapes arrive with an already resolved pixel
// extent. The value is not scaled by the scene camera or the canvas density.
ImageDestinationExtent ResolveImageDestinationExtent(const ParseContext&           context,
                                                     const wpscene::WPImageObject& image,
                                                     const SceneMaterial&          material,
                                                     const std::array<float, 2>&   card_size) {
    ImageDestinationExtent result;
    const auto use_card = [&](const char* policy) {
        result.extent = ResolveCardDestinationExtent(card_size);
        result.policy = policy;
        result.uses_card_size = true;
        return result;
    };
    if (image.fullscreen) {
        const auto output = OutputFramebufferEffectTargetSize(context);
        result.extent     = { static_cast<int32_t>(std::lround(output[0])),
                              static_cast<int32_t>(std::lround(output[1])) };
        result.policy     = "output-framebuffer";
        return result;
    }
    if (material.textures.empty() || material.Texture(0).empty() ||
        context.scene == nullptr) {
        return use_card("card-no-texture");
    }
    const auto& texture_name = material.Texture(0);
    const bool  card_sized_helper =
        (image.config.passthrough || image.solidlayer) && ! image.instanced;

    if (const auto texture_it = context.scene->textures.find(texture_name);
        texture_it != context.scene->textures.end()) {
        const auto& texture = texture_it->second;
        result.suffix = texture.sample.magFilter == TextureFilter::NEAREST ? 'n' : 'b';
        if (texture.isSprite) {
            const auto& frames = texture.spriteAnim.Frames();
            if (! frames.empty() && frames.front().width > 0.0f && frames.front().height > 0.0f) {
                result.extent = { static_cast<int32_t>(std::lround(frames.front().width)),
                                  static_cast<int32_t>(std::lround(frames.front().height)) };
            } else {
                result.extent = { texture.mapWidth, texture.mapHeight };
            }
            result.policy = "sprite-frame";
        } else if (card_sized_helper) {
            return use_card("card-passthrough");
        } else {
            result.extent = { texture.mapWidth, texture.mapHeight };
            result.policy = "texture-content";
        }
    } else if (const auto target_it = context.scene->renderTargets.find(texture_name);
               target_it != context.scene->renderTargets.end()) {
        const auto& target = target_it->second;
        result.suffix = target.sample.magFilter == TextureFilter::NEAREST ? 'n' : 'b';
        if (card_sized_helper) return use_card("card-passthrough");
        result.extent = { target.ContentWidth(), target.ContentHeight() };
        result.policy = "render-target-content";
    } else {
        return use_card("card-unresolved-texture");
    }
    if (result.extent[0] <= 0 || result.extent[1] <= 0) return use_card("card-empty-texture");
    return result;
}

std::string EffectFboRenderTargetName(const wpscene::WPEffectFbo& fbo, int32_t effect_id) {
    if (! fbo.unique) return fbo.name;
    return fbo.name + "_" + std::to_string(effect_id);
}

void LoadEffectCommands(const wpscene::WPImageEffect& source, SceneImageEffect& effect,
                        const std::unordered_map<std::string, std::string>& declared_fbos) {
    // Command indices are looked up exclusively in the effect's declared FBO array. The
    // material-only `previous` alias is inserted by callers after this step; an undeclared
    // command name must retain the -1 value.
    const auto resolve_fbo = [&](const std::string& name) -> std::optional<std::string> {
        const auto found = declared_fbos.find(name);
        if (found == declared_fbos.end()) return std::nullopt;
        return found->second;
    };
    for (const auto& [_, target] : declared_fbos) effect.RegisterFbo(target);
    for (const auto& command : source.commands) {
        if (command.command != "copy" && command.command != "swap") {
            LOG_ERROR("Unknown effect command: %s", command.command.c_str());
            continue;
        }
        effect.commands.push_back({ .cmd = command.command == "swap"
                                        ? SceneImageEffect::CmdType::Swap
                                        : SceneImageEffect::CmdType::Copy,
                                    .authored_dst = resolve_fbo(command.target),
                                    .authored_src = resolve_fbo(command.source),
                                    .afterpos = command.afterpos,
                                    .advances_composition = command.compose });
    }
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
        // This field is a literal boolean, not a script/user property. Missing and non-boolean
        // values both retain the default inclusion in the reflected-owner list.
        reflected = ReadJsonLiteralBoolean(json, "reflected", true);
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




namespace
{

bool UsesShaderColorBlendMode(int32_t color_blend_mode) {
    return color_blend_mode >= 1 && color_blend_mode <= 30;
}

BlendMode ResolveObjectFinalBlend(BlendMode authored_blend, int32_t color_blend_mode) {
    return color_blend_mode == 31 ? BlendMode::Additive : authored_blend;
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
        } else if (IsImageLayerCompositeTex(name)) {
            // The private slot is registered under its complete authored name and sampled through
            // ordinary named texture lookup. Rewriting it to a layer id loses the physical slot
            // and can make the consumer sample a later screen publication instead of the private
            // result.
        } else if (name == SpecTex_DefaultPingPong) {
        } else if (sstart_with(name, WE_MIP_MAPPED_FRAME_BUFFER)) {
        } else if (sstart_with(name, WE_EFFECT_PPONG_PREFIX)) {
        } else if (sstart_with(name, WE_HALF_COMPO_BUFFER_PREFIX)) {
        } else if (sstart_with(name, WE_QUARTER_COMPO_BUFFER_PREFIX)) {
        } else if (sstart_with(name, WE_FULL_COMPO_BUFFER_PREFIX)) {
        } else if (name == "_rt_shadowAtlas") {
        } else if (name == kModelReflectionTargetName) {
            // The name can come from a shader's default texture, before model setup has
            // classified its active samplers. Only that classification allocates the target; an
            // unused authored slot does not.
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
LoadMaterial(fs::VFS& vfs, const wpscene::WPMaterial& wpmat, Scene* pScene,
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
            // Keep the authored input intact when the system property has no value. The live
            // handle is resolved by every texture consumer, after effect aliases have selected
            // their current targets.
            material.systemTextureBindings.emplace(
                i, pScene->GetSystemTextureBinding(binding.name));
            if (std::getenv("WESCENE_TRACE_MEDIA_STATE") != nullptr) {
                LOG_INFO("SceneMaterialSystemTexture: shader='%s' slot=%zu property='%s' "
                         "authored='%s' override='%s'",
                         wpmat.shader.c_str(), i, binding.name.c_str(), textures[i].c_str(),
                         material.systemTextureBindings.at(i)->c_str());
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
            if (pScene->renderTargets.count(name) != 0) {
                const auto& rt = pScene->renderTargets.at(name);
                // Runtime render targets may keep a larger physical allocation than the logical
                // content they currently store. Forwarding the authored content extent through
                // `.zw` preserves Wallpaper Engine's original "sample area" contract for effects
                // that distinguish between allocated size and meaningful image size.
                resolution = rt.ResolutionVector();
            } else if (name == kModelReflectionTargetName) {
                // Receiver detection follows compilation. Seed its screen-sized resolution
                // without allocating a target for every material mentioning the name. The
                // live updater reads the registered receiver target after model setup.
                resolution = pScene->renderTargets.at(std::string(SpecTex_Default))
                                 .ResolutionVector();
            } else if (!IsImageLayerCompositeTex(name)) {
                LOG_ERROR("%s not found in render targets", name.c_str());
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
            material.systemTextureBindings.erase(i);
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

const char* EffectPublicationMaterialPath(uint32_t scene_version) {
    // The scene's top-level JSON version selects the utility program for private publication,
    // using the unsigned version >= 3 boundary. This decision is independent of the authored
    // source shader, its lighting combos and the renderer's graphics capabilities.
    return scene_version >= 3 ? "/assets/materials/util/effectpassthrough_4.json"
                              : "/assets/materials/util/effectpassthrough.json";
}

bool LoadEffectPublicationSource(ParseContext& context, wpscene::WPMaterial& material) {
    nlohmann::json json;
    return PARSE_JSON(fs::GetFileContent(*context.vfs,
                                        EffectPublicationMaterialPath(context.scene->authoredVersion)),
                      json) && material.FromJson(json);
}

void SetNeutralPublicationModulation(ShaderValueMap& values) {
    // Publication uses neutral color and alpha when sampling the resolved slot. It preserves
    // those pixels without applying the authored source's tint, opacity or brightness again.
    values["g_Color4"] = std::array<float, 4> { 1.0f, 1.0f, 1.0f, 1.0f };
    values["g_Color"] = std::array<float, 3> { 1.0f, 1.0f, 1.0f };
    values["g_Alpha"] = 1.0f;
    values["g_UserAlpha"] = 1.0f;
    values["g_Brightness"] = 1.0f;
}

bool ConfigureEffectFinalComposite(ParseContext& context, SceneImageEffectLayer& effect_layer,
                                   std::string_view initial_source, int32_t owner_layer_id,
                                   std::string_view         owner_name,
                                   int32_t                  color_blend_mode,
                                   const WPShaderValueData* final_transform_data = nullptr,
                                   const WPMdl* puppet = nullptr) {
    auto& vfs = *context.vfs;

    wpscene::WPMaterial composite_source;
    if (!LoadEffectPublicationSource(context, composite_source)) {
        LOG_ERROR(
            "SceneEffectFinalComposite: layer=%d name='%.*s' failed to load passthrough material",
            owner_layer_id,
            static_cast<int>(owner_name.size()),
            owner_name.data());
        return false;
    }

    if (composite_source.textures.empty()) composite_source.textures.resize(1);
    composite_source.textures[0] = std::string(initial_source);
    // Modes 1..30 are framebuffer-aware shader blend equations. Modes 0 and 31 use the neutral
    // shader variant; mode 31 is expressed by the final fixed-function additive state instead.
    composite_source.combos["BLENDMODE"] =
        UsesShaderColorBlendMode(color_blend_mode) ? color_blend_mode : 0;
    if (puppet != nullptr) {
        // The restored destination receives the imported mesh with this utility program. Skinning
        // samples the owner's current pose at uniform upload; it does not require another local
        // render target or a second animation runtime on this drawing phase.
        WPMdlParser::AddPuppetMatInfo(composite_source, *puppet);
        composite_source.combos["LIGHTING"] = 0;
        composite_source.combos["REFLECTION"] = 0;
    }

    WPShaderInfo composite_shader_info;
    composite_shader_info.baseConstSvs = context.global_base_uniforms;
    SetNeutralPublicationModulation(composite_shader_info.baseConstSvs);

    SceneMaterial     composite_material;
    WPShaderValueData composite_data;
    if (! LoadMaterial(vfs,
                       composite_source,
                       context.scene.get(),
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
    if (final_transform_data != nullptr) {
        // Publication evaluates the owner's raw transform through the layer binding below.
        // Its destination displacement remains phase-specific: inheriting the owner's parent
        // binding here would apply the hierarchy twice when resolving this draw handle.
        composite_data.CopyParallaxContractFrom(*final_transform_data);
    }
    composite_data.SetEffectLayerProjection(&effect_layer);

    auto composite_mesh = std::make_shared<SceneMesh>();
    composite_mesh->AddMaterial(std::move(composite_material));

    auto& publication = effect_layer.FinalCompositeDraw();
    publication.SetName(std::string(owner_name) + "::publication");
    publication.SetMesh(std::move(composite_mesh));
    context.shader_updater->SetNodeData(&publication, composite_data);
    // Publication stores only its raster resources and projection selection. Its owner reference
    // supplies identity, visibility and placement; no SceneNode is created or registered here.
    effect_layer.SetFinalCompositeSource(std::string(initial_source));
    return true;
}

std::string_view ImageEffectSourcePolicyName(SceneImageEffectLayer::SourcePolicy policy) {
    switch (policy) {
    case SceneImageEffectLayer::SourcePolicy::None: return "none";
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

TextureSample ResolvePrimaryMaterialSampler(const Scene& scene, const SceneMaterial& material) {
    if (material.textures.empty() || material.Texture(0).empty()) return {};
    const auto& texture_name = material.Texture(0);
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
    if (material.textures.empty() || material.Texture(0).empty()) return false;
    const auto texture_it = scene.textures.find(material.Texture(0));
    return texture_it != scene.textures.end() && texture_it->second.isSprite;
}

void MergeImageSourceProgramBindings(SceneMaterial& material, const SceneMaterial& variant) {
    // Both executables address the same authored bindings. Install additional reflected
    // slots once, before script registration, then leave their live values untouched during
    // program switches. Per-program default uniforms remain owned by each compiled shader.
    material.customShader.constValues.insert(variant.customShader.constValues.begin(),
                                             variant.customShader.constValues.end());
    material.uniformAliases.insert(variant.uniformAliases.begin(), variant.uniformAliases.end());
    material.systemTextureBindings.insert(variant.systemTextureBindings.begin(),
                                           variant.systemTextureBindings.end());
    for (size_t slot = 0; slot < variant.textures.size(); ++slot) {
        if (slot >= material.textures.size()) {
            material.textures.push_back(variant.textures[slot]);
            material.defines.push_back(variant.defines[slot]);
        } else if (material.textures[slot].empty()) {
            material.textures[slot] = variant.textures[slot];
        }
    }
}

bool LoadImageDirectPuppetSource(
    ParseContext& context, const wpscene::WPImageObject& image, const WPMdl& puppet,
    const WPShaderInfo& source_info, SceneMaterial& material,
    SceneImageEffectLayer::DirectPuppetSource& result) {
    // Direct skinning uses the original image material, including authored lighting and
    // reflection. The parser supplies only owners eligible for this branch; live visible-effect
    // state decides whether it executes before private publication has been selected for the
    // owner.
    WPShaderInfo skinned_info = source_info;
    WPMdlParser::AddPuppetShaderInfo(skinned_info, puppet);
    SceneMaterial skinned_material;
    WPShaderValueData skinned_values;
    if (!LoadMaterial(*context.vfs, image.material, context.scene.get(), &skinned_material,
                      &skinned_values, context.user_properties, &skinned_info)) {
        LOG_ERROR("SceneImageDirectPuppetMaterial: layer=%d name='%s' shader='%s' load failed",
                  image.id, image.name.c_str(), image.material.shader.c_str());
        return false;
    }
    result.ordinary_shader = material.customShader.shader;
    result.skinned_shader = skinned_material.customShader.shader;
    MergeImageSourceProgramBindings(material, skinned_material);
    LOG_INFO("SceneImageDirectPuppetMaterial: layer=%d name='%s' shader='%s' chunk-info=0x%x",
             image.id, image.name.c_str(), image.material.shader.c_str(), puppet.chunk_info);
    return true;
}

bool LoadImagePrelightingSource(
    ParseContext& context, const wpscene::WPImageObject& image, const WPMdl* puppet,
    const WPShaderInfo& source_info, SceneMaterial& material,
    std::optional<SceneImageEffectLayer::PrelightingSource>& result) {
    const auto combo_enabled = [&](std::string_view name) {
        const auto it = source_info.combos.find(std::string(name));
        return it != source_info.combos.end() && it->second != "0";
    };
    const bool lighting = combo_enabled("LIGHTING");
    const bool reflection = combo_enabled("REFLECTION");
    if ((!lighting && !reflection) || material.textures.empty() ||
        material.Texture(0).empty()) return true;

    // Derive the source variant from the complete authored material, preserving its feature
    // combos and property inputs. This prepares an executable; visible-effect/private/sprite
    // state selects it at graph construction, so a runtime visibility change never reloads a
    // material over script edits.
    auto authored = image.material;
    authored.combos["LIGHTING"] = lighting ? 1 : 0;
    authored.combos["REFLECTION"] = reflection ? 1 : 0;
    authored.combos["PRELIGHTING"] = 1;
    const bool skinned = puppet != nullptr && puppet->HasImageSkinning();
    if (skinned) {
        WPMdlParser::AddPuppetMatInfo(authored, *puppet);
        authored.combos["PRELIGHTINGDUALVERTEX"] = 1;
    }
    WPShaderInfo prelighting_info;
    prelighting_info.baseConstSvs = source_info.baseConstSvs;
    SceneMaterial prelighting_material;
    WPShaderValueData prelighting_values;
    if (!LoadMaterial(*context.vfs, authored, context.scene.get(), &prelighting_material,
                       &prelighting_values, context.user_properties, &prelighting_info)) {
        LOG_ERROR("SceneImagePrelightingMaterial: layer=%d name='%s' shader='%s' load failed",
                  image.id, image.name.c_str(), authored.shader.c_str());
        return false;
    }

    const auto& scene = *context.scene;
    const auto& texture_name = material.Texture(0);
    std::array<int32_t, 2> allocation;
    std::array<float, 2> content;
    bool sprite = false;
    if (const auto it = scene.textures.find(texture_name); it != scene.textures.end()) {
        const auto& texture = it->second;
        allocation = {texture.width, texture.height};
        sprite = texture.isSprite;
        if (sprite) {
            const auto& frame = texture.spriteAnim.Frames().front();
            content = {frame.width, frame.height};
        } else {
            content = {static_cast<float>(texture.mapWidth),
                       static_cast<float>(texture.mapHeight)};
        }
    } else {
        const auto& texture = scene.renderTargets.at(texture_name);
        allocation = {texture.width, texture.height};
        content = {static_cast<float>(texture.ContentWidth()),
                   static_cast<float>(texture.ContentHeight())};
    }

    auto source_mesh = std::make_shared<SceneMesh>();
    if (skinned) {
        // The primary position and tangent basis are skinned for lighting; the unskinned
        // auxiliary position rasterizes into the source texture. Do not apply the display-card
        // calibration or the animated publication envelope here.
        WPMdlParser::GenPuppetMesh(*source_mesh, *puppet);
    } else {
        // Private source geometry uses the physical texture extent and full UVs. Logical content
        // determines its half-size I translation; target extent independently determines
        // projection.
        GenCardMesh(*source_mesh, {static_cast<uint16_t>(allocation[0]),
                                   static_cast<uint16_t>(allocation[1])});
    }
    result = SceneImageEffectLayer::PrelightingSource {
        .ordinary_shader = material.customShader.shader,
        .prelighting_shader = prelighting_material.customShader.shader,
        .mesh = std::move(source_mesh),
        .content_size = content,
        .sprite = sprite,
        .instanced = image.instanced,
    };

    MergeImageSourceProgramBindings(material, prelighting_material);
    LOG_INFO("SceneImagePrelightingMaterial: layer=%d name='%s' shader='%s' lighting=%s "
             "reflection=%s dual-position=%s allocation=[%d %d] content=[%.3f %.3f]",
             image.id, image.name.c_str(), authored.shader.c_str(), lighting ? "true" : "false",
             reflection ? "true" : "false", skinned ? "true" : "false",
             allocation[0], allocation[1], content[0], content[1]);
    return true;
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
                             std::string_view object_kind,
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
                           attachment_binding->transform);
    return true;
}

void AttachNodeToScene(ParseContext& context, const std::shared_ptr<SceneNode>& node,
                       int32_t parent_id, const std::string& object_name) {
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

}

void ConfigureInheritedParentBinding(ParseContext& context, int32_t parent_id,
                                     WPShaderValueData& node_data) {
    if (auto parent = FindParentNode(context, parent_id)) {
        node_data.InheritParentTransform(parent.get());
    }
}

namespace
{

struct EffectWriterTransformContract {
    std::array<float, 2>       parallax_depth { 0.0f, 0.0f };
    SceneImageEffectLayer*     projection_layer { nullptr };
    bool suppress_own_model_parallax { false };
};

void ApplyEffectWriterTransformContract(ParseContext& context,
                                        const EffectWriterTransformContract& contract,
                                        WPShaderValueData& data) {
    data.SetParallaxContract(contract.parallax_depth);

    if (contract.projection_layer != nullptr) {
        data.SetEffectLayerProjection(contract.projection_layer);
    }


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
    const wpscene::WPImageObject& image) {
    EffectWriterTransformContract contract;
    contract.parallax_depth = ImageObjectParallaxDepth(image);
    return contract;
}

EffectWriterTransformContract BuildTextEffectFinalCompositeContract(
    const wpscene::WPTextObject& text) {
    EffectWriterTransformContract contract;
    contract.parallax_depth = TextObjectParallaxDepth(text);
    return contract;
}

EffectWriterTransformContract BuildImageEffectMaterialContract(
    const wpscene::WPImageObject& image, SceneImageEffectLayer& effect_layer) {
    // The authored last material and a layer blend stage publish the same object. Both must
    // use the same parent/attachment parallax contract on the raw object transform. Intermediate
    // effect draws select local projection separately and do not apply scene parallax.
    auto contract = BuildImageEffectFinalCompositeContract(image);
    contract.projection_layer        = &effect_layer;
    return contract;
}

EffectWriterTransformContract BuildTextEffectMaterialContract(
    const wpscene::WPTextObject& text, SceneImageEffectLayer& effect_layer) {
    // Text re-layout synchronizes an unmodified object transform. The material that publishes
    // the shaped card therefore owns the same single parallax application as direct text or
    // the layer blend stage, including the authored parent anchor.
    auto contract = BuildTextEffectFinalCompositeContract(text);
    contract.projection_layer        = &effect_layer;
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
        // the material resource directly and writes the same GLSL uniform that the cold parse
        // resolved from shader metadata.
        context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
            .object_id     = object_id,
            .object_name   = std::string(object_name),
            .property_name = binding.gl_uniform_name,
            .material      = node->Mesh()->Material(),
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
        // dynamic constants also need a live target on the concrete material. User bindings and
        // scripts both reuse the MaterialUniform dispatcher so album-art color scripts can update
        // Gradient Color uniforms without rebuilding the post-process chain.
        WPSceneScriptRegistration registration {
            .object_id     = object_id,
            .object_name   = std::string(object_name),
            .property_name = gl_uniform_name,
            .material      = node->Mesh()->Material(),
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
                                      algorism::ResolvePerspectiveFov(
                                          scene.generalProjection.perspectiveOverrideFov,
                                          context.ortho_h));

    Vector3f cperori                       = cori;
    cperori[2]                             = 1000.0f;
    context.global_perspective_camera_node = std::make_shared<SceneNode>(cperori, cscale, cangle);
    scene.cameras["global_perspective"]->AttatchNode(context.global_perspective_camera_node);
    scene.sceneGraph->AppendChild(context.global_perspective_camera_node);

    scene.authoredCameraPose = {
        .eye = scene_config.camera.eye,
        .center = scene_config.camera.center,
        .up = scene_config.camera.up,
    };
    const auto& pose = scene.authoredCameraPose;
    const Vector3d eye(pose.eye[0], pose.eye[1], pose.eye[2]);
    const Vector3d center(pose.center[0], pose.center[1], pose.center[2]);
    const Vector3d up(pose.up[0], pose.up[1], pose.up[2]);
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

    scene.authoredVersion              = sc.version;
    LOG_INFO("SceneEffectPublicationMaterial: scene-version=%u path='%s'",
             scene.authoredVersion,
             EffectPublicationMaterialPath(scene.authoredVersion));
    scene.clearEnabled                 = sc.general.clearenabled;
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
    scene.generalProjection = {
        .fov = sc.general.fov,
        .perspectiveOverrideFov = sc.general.perspectiveoverridefov,
        .nearClip = sc.general.nearz,
        .farClip = sc.general.farz,
    };
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

namespace
{
// Effect retention, FBO allocation, material loading and script registration are shared resource
// responsibilities. The optional source target belongs to the owner's draw contract: images
// supply a ping-pong source, while shape's source-slot -1 leaves authored material inputs intact.
void LoadLayerEffects(ParseContext& context, SceneImageEffectLayer& layer,
                      const std::vector<wpscene::WPImageEffect>& effects,
                      const std::array<float, 2>& target_resolution,
                      const ShaderValueMap& base_uniforms,
                      const EffectWriterTransformContract& transform_contract,
                      std::optional<std::string_view> source_target,
                      const WPPuppetLayer* puppet_pose = nullptr) {
    auto& scene = *context.scene;
    const auto layer_id = layer.Owner().Id();
    const auto& layer_name = layer.Owner().Name();
    const auto visibility_policy = layer.UsesShapeDraw()
        ? SceneImageEffect::VisibilityPolicy::OwnerOnly
        : SceneImageEffect::VisibilityPolicy::Instance;
    for (usize effect_index = 0; effect_index < effects.size(); ++effect_index) {
        const auto& authored_effect = effects[effect_index];
        auto effect = std::make_shared<SceneImageEffect>(visibility_policy);
        effect->SetIdentity(layer_id, authored_effect.id, static_cast<uint32_t>(effect_index),
                            authored_effect.name);
        const auto feedback_fbos = authored_effect.FeedbackFboNames();
        std::unordered_map<std::string, std::string> fbo_map;
        for (const auto& authored_fbo : authored_effect.fbos) {
            const auto name = EffectFboRenderTargetName(authored_fbo, authored_effect.id);
            const auto size = authored_fbo.ResolveSize(target_resolution);
            const bool feedback = feedback_fbos.contains(authored_fbo.name);
            const SceneRenderTarget target {
                .width = size[0],
                .height = size[1],
                .mapWidth = size[0],
                .mapHeight = size[1],
                .allowReuse = !feedback,
            };
            InternNamedRenderTarget(scene, name, target);
            if (authored_fbo.fit > 0 || feedback) {
                LOG_INFO("SceneEffectFboResolve: layer=%d effect-id=%d effect='%s' "
                         "fbo='%s' target='%s' size=%dx%d scale=%u fit=%u persistent-feedback=%s",
                         layer_id, authored_effect.id, authored_effect.name.c_str(),
                         authored_fbo.name.c_str(), name.c_str(), size[0], size[1],
                         authored_fbo.scale, authored_fbo.fit, feedback ? "true" : "false");
            }
            layer.AddEffectRenderTarget(name, authored_fbo.scale, authored_fbo.fit);
            fbo_map[authored_fbo.name] = name;
        }

        LoadEffectCommands(authored_effect, *effect, fbo_map);
        if (source_target) fbo_map.try_emplace("previous", *source_target);

        bool materials_loaded = true;
        for (usize material_index = 0; material_index < authored_effect.materials.size();
             ++material_index) {
            auto material_source = authored_effect.materials[material_index];
            std::string output(source_target ? WE_EFFECT_PPONG_PREFIX_B : SpecTex_Default);
            std::vector<usize> fbo_texture_slots;
            bool output_is_fbo = false;
            if (material_index < authored_effect.passes.size()) {
                const auto& pass = authored_effect.passes[material_index];
                material_source.MergePass(pass);
                for (const auto& binding : pass.bind) {
                    const auto target = fbo_map.find(binding.name);
                    if (target == fbo_map.end()) {
                        // A negative binding index is ignored for source-slot -1; in particular,
                        // "previous" must not overwrite an authored texture with a shape target.
                        if (source_target) LOG_ERROR("fbo %s not found", binding.name.c_str());
                        continue;
                    }
                    const auto slot = static_cast<usize>(binding.index);
                    if (material_source.textures.size() <= slot) {
                        material_source.textures.resize(slot + 1);
                    }
                    material_source.textures[slot] = target->second;
                    if (effect->IsDeclaredFbo(target->second)) fbo_texture_slots.push_back(slot);
                }
                if (!pass.target.empty()) {
                    const auto target = fbo_map.find(pass.target);
                    if (target != fbo_map.end()) {
                        output = target->second;
                        output_is_fbo = effect->IsDeclaredFbo(output);
                    } else if (source_target) {
                        LOG_ERROR("fbo %s not found", pass.target.c_str());
                    }
                }
            }
            if (source_target) {
                if (material_source.textures.empty()) material_source.textures.resize(1);
                if (material_source.textures[0].empty()) {
                    material_source.textures[0] = *source_target;
                }
            }

            auto node = std::make_shared<SceneNode>();
            // Material nodes are private draw handles. Their id refers to the canonical owner,
            // but they do not enter the authored hierarchy or allocate another script identity.
            node->ID() = layer_id;
            node->SetName(layer_name + "::__hanabi_effect_pass_" +
                          std::to_string(effect_index) + "_" + std::to_string(material_index));
            WPShaderInfo shader_info;
            shader_info.baseConstSvs = base_uniforms;
            shader_info.baseConstSvs["g_EffectTextureProjectionMatrix"] =
                ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
            shader_info.baseConstSvs["g_EffectTextureProjectionMatrixInverse"] =
                ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
            SceneMaterial material;
            WPShaderValueData node_data;
            if (!LoadMaterial(*context.vfs, material_source, &scene, &material, &node_data,
                              context.user_properties, &shader_info)) {
                LOG_ERROR("SceneEffectLoad: layer=%d effect='%s' material-index=%zu load failed",
                          layer_id, authored_effect.name.c_str(), material_index);
                materials_loaded = false;
                break;
            }
            LoadConstvalue(material, material_source, shader_info);
            LoadUserShaderValue(material, material_source, shader_info, context.user_properties);
            ApplyEffectWriterTransformContract(context, transform_contract, node_data);
            if (puppet_pose != nullptr && material_source.use_puppet) {
                node_data.puppet_layer = *puppet_pose;
            }

            const auto authored_textures = material.textures;
            const auto authored_blend = material.blenmode;
            auto mesh = std::make_shared<SceneMesh>();
            mesh->AddMaterial(std::move(material));
            node->AddMesh(mesh);
            RegisterUserShaderValueBindings(
                context, material_source, shader_info, node.get(), layer_id, layer_name);
            RegisterConstantShaderValueBindings(
                context, material_source, shader_info, node.get(), layer_id, layer_name,
                authored_effect.id, static_cast<int32_t>(effect_index), material_index);
            context.shader_updater->SetNodeData(node.get(), node_data);
            const bool final_material = material_index + 1 == authored_effect.materials.size();
            effect->nodes.push_back({
                .authored_output = output,
                .output = output,
                .authored_textures = authored_textures,
                .fbo_texture_slots = std::move(fbo_texture_slots),
                .output_is_fbo = output_is_fbo,
                .sceneNode = node,
                .advances_composition = material_index < authored_effect.passes.size() &&
                    authored_effect.passes[material_index].compose,
                .is_final_material = final_material,
                .authored_blend = authored_blend,
            });
            if (layer.UsesShapeDraw()) {
                LOG_INFO("SceneShapeMaterialRetained: layer=%d effect=%d effect-index=%zu "
                         "material=%zu final-material=%s input='%s' explicit-fbo=%s output='%s'",
                         layer_id, authored_effect.id, effect_index, material_index,
                         final_material ? "true" : "false",
                         authored_textures.empty() ? "" : authored_textures[0].c_str(),
                         output_is_fbo ? "true" : "false", output.c_str());
            }
        }
        if (!materials_loaded) {
            LOG_ERROR("effect '%s' failed to load", authored_effect.name.c_str());
            continue;
        }
        const bool visible = ResolveEffectVisibility(authored_effect, context.user_properties);
        effect->SetLocalVisible(visible);
        if (!authored_effect.visible_json.is_null()) {
            LOG_INFO("SceneEffectVisibilityResolve: layer=%d effect-id=%d effect-index=%zu "
                     "name='%s' authored=%s initial=%s runtime=%s",
                     layer_id, authored_effect.id, effect_index, authored_effect.name.c_str(),
                     authored_effect.visible ? "true" : "false", visible ? "true" : "false",
                     EffectVisibilityCanChangeAtRuntime(authored_effect) ? "true" : "false");
        }
        layer.AddEffect(effect);
    }
}
} // namespace

void ParseImageObj(ParseContext& context, wpscene::WPImageObject& img_obj) {
    auto& wpimgobj = img_obj;

    auto& vfs = *context.vfs;

    // Runtime image properties exist before materialization, including empty compositions.
    // Composition resources read this owner record throughout parsing and after script updates,
    // so both paths retain one value for the copybackground property.
    context.scene->EnsureSceneObject(wpimgobj.id)
        .SetImageRuntimeState(Scene::ImageLayerRuntimeState {
            .size = wpimgobj.size,
            .alignment = wpimgobj.alignment,
            .copy_background = wpimgobj.copybackground,
        });

    const int32_t count_eff = static_cast<int32_t>(wpimgobj.effects.size());
    const bool hasAuthoredEffect = count_eff > 0;
    const bool isCompose = wpimgobj.config.passthrough && !wpimgobj.fullscreen;
    const bool is_offscreen_dependency_source =
        context.scene != nullptr &&
        context.scene->IsLayerOffscreenDependencySource(wpimgobj.id);
    const bool has_shader_color_blend = UsesShaderColorBlendMode(wpimgobj.colorBlendMode);
    // Wallpaper Engine `dependencies` expose a layer through `_rt_imageLayerComposite_<id>_a`
    // even when the source layer has no authored effects. Such layers still need a private source
    // render target because a consumer samples that source independently of the owner's
    // visibility. Treating dependency-only image layers as effect-backed sources lets the
    // existing effect camera/ping-pong path materialize the raw image or mask. Publication
    // separately follows the owner's current visibility, whether its enclosing destination is the
    // scene or a composition. A composition can gain its first child after parsing. Retain its
    // material, mesh and bridge descriptors even while empty, so that transition does not reparse
    // the owner or reset script-written properties. An empty non-private zero-step owner still
    // emits no graph passes. The scene target table only declares resources; GPU images are
    // queried by the emitted draw passes.
    bool hasEffect = hasAuthoredEffect || has_shader_color_blend ||
        is_offscreen_dependency_source || wpimgobj.config.passthrough;
    const bool uses_routed_parent = LayerUsesRoutedParent(wpimgobj.parent, wpimgobj.attachment);
    // Card/compose-camera size is independent of the source texture's destination extent.
    const std::array<float, 2> effect_source_size = wpimgobj.size;
    const bool hasAuthoredPuppet = ! wpimgobj.puppet.empty();
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

    // One object retains the authored transform, mesh, pose and script identity. The source
    // pass selects its local raster matrix without allocating or mutating another SceneNode.
    auto spImgNode = std::make_shared<SceneNode>(Vector3f(wpimgobj.origin.data()),
                                                Vector3f(wpimgobj.scale.data()),
                                                Vector3f(wpimgobj.angles.data()),
                                                wpimgobj.name);
    LoadAlignment(*spImgNode, wpimgobj.alignment, { wpimgobj.size[0], wpimgobj.size[1] });
    spImgNode->ID() = wpimgobj.id;

    SceneMaterial     material;
    WPShaderValueData svData;
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
        if (!material.textures.empty()) primary_source_texture = material.Texture(0);
    }
    std::optional<SceneImageEffectLayer::DirectPuppetSource> direct_puppet_source;
    if (hasEffect && puppet && puppet->HasImageSkinning() &&
        !puppet->HasImagePrivateChunk() &&
        !PrimaryMaterialTextureIsSprite(*context.scene, material) &&
        !is_offscreen_dependency_source && !has_shader_color_blend &&
        !wpimgobj.config.passthrough) {
        direct_puppet_source.emplace();
        if (!LoadImageDirectPuppetSource(context, wpimgobj, *puppet, shaderInfo,
                                         material, *direct_puppet_source)) return;
    }
    std::optional<SceneImageEffectLayer::PrelightingSource> prelighting_source;
    if (hasEffect && !LoadImagePrelightingSource(context, wpimgobj, puppet.get(), shaderInfo,
                                                 material, prelighting_source)) return;
    // Destination targets and effect FBOs are fixed-size images sized from the source texture
    // content (or the card for texture-less / passthrough helpers), never from the scene camera.
    const ImageDestinationExtent destination_extent =
        ResolveImageDestinationExtent(context, wpimgobj, material, effect_source_size);
    // Destination filtering follows the source unless the layer explicitly disables
    // interpolation. Addressing belongs to the destination itself, not to the source file;
    // include both settings in the intern key and use the same sampler for both slots.
    const TextureSample destination_sampler = DestinationRenderTargetSampler(
        wpimgobj.nointerpolation || destination_extent.suffix == 'n', wpimgobj.clampuvs);
    // Destination setup clamps the selected source extent before either the destination pair or
    // the authored effect FBOs use it.
    const std::array<float, 2> effect_target_resolution {
        static_cast<float>(ClampDestinationRenderTargetExtent(destination_extent.extent[0])),
        static_cast<float>(ClampDestinationRenderTargetExtent(destination_extent.extent[1])),
    };

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
                // The source material already evaluates authored image behavior before the effect
                // sequence. Publication samples that resolved result with a utility material and
                // the original skinned mesh; repeating the source shader would process iris
                // controls, lighting and tint twice. Keep source and publication programs
                // separate while sharing the same owner and immutable puppet pose.
                GenCardMesh(
                    mesh, { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] }, mapRate);
                WPMdlParser::GenPuppetMesh(effct_final_mesh, *puppet);
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
            GenCardMesh(effct_final_mesh,
                        { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] });
        }
    }
    // A passthrough source cleared to transparent publishes with translucent blending.
    // colorBlendMode 31 takes precedence over that rule; other images retain their authored
    // destination blend independently of the source-pass override below.
    const auto authored_destination_blend =
        ResolveObjectFinalBlend(material.blenmode, wpimgobj.colorBlendMode);
    const auto transparent_destination_blend =
        ResolveObjectFinalBlend(BlendMode::Translucent, wpimgobj.colorBlendMode);
    const auto imgBlendMode = wpimgobj.config.passthrough && !wpimgobj.copybackground
        ? transparent_destination_blend : authored_destination_blend;
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
                            "object",
                            wpimgobj.name,
                            svData);

    svData.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };


    if (hasEffect) {
        auto& scene = *context.scene;
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
            scene.cameras.at(effect_camera_name)->AttatchNode(spImgNode);
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
        const int32_t effect_target_width = ClampDestinationRenderTargetExtent(
            static_cast<int32_t>(std::lround(effect_target_resolution[0])));
        const int32_t effect_target_height = ClampDestinationRenderTargetExtent(
            static_cast<int32_t>(std::lround(effect_target_resolution[1])));
        // Allocate the puppet's potential private branch before its materials are bound. Direct
        // drawing can omit every source/publication draw for initially hidden effects; resident
        // resources allow a later visibility change to promote the owner. Composition membership
        // and the separate hidden-output policy do not set this flag.
        const bool private_destination_output = is_offscreen_dependency_source ||
            hasAnimatedPuppetMesh || has_shader_color_blend;
        const SceneRenderTarget destination_target {
            .width = effect_target_width,
            .height = effect_target_height,
            .mapWidth = effect_target_width,
            .mapHeight = effect_target_height,
            .allowReuse = true,
            .sample = destination_sampler,
        };
        const auto effect_destination_names = ResolveSceneDestinationRenderTargets(
            scene, wpimgobj.id, wpimgobj.parent, private_destination_output, destination_target);
        const std::string& effect_ppong_a = effect_destination_names[0];
        const std::string& effect_ppong_b = effect_destination_names[1];
        // Source and destination are drawing phases of the same authored object. Keep its
        // identity here; the parser registers the canonical LayerNode after parent resolution,
        // and uniform evaluation reads its current transform through that owner at draw time.
        auto imgEffectLayer = std::make_shared<SceneImageEffectLayer>(
            scene.EnsureSceneObject(wpimgobj.id), wpimgobj.size[0], wpimgobj.size[1],
            effect_ppong_a, effect_ppong_b);
        imgEffectLayer->SetDestinationUsesCardSize(destination_extent.uses_card_size);
        {
            // Fullscreen image-effect layers are postprocess-style framebuffer passes. Remember
            // that authored shape here so ResolveEffect() can keep their final shader on the
            // effect-camera fullscreen quad instead of projecting the 2x2 utility mesh through the
            // active scene camera.
            if (wpimgobj.fullscreen) {
                // Fullscreen geometry stays in the effect camera's 2x2 raster domain, while
                // effect matrices scale by the source framebuffer's content pixels resolved
                // during this resource setup. Do not replace the incoming scene camera or
                // resize the raster card to compensate for that distinct matrix extent.
                imgEffectLayer->SetFullscreenTextureSize({
                    static_cast<float>(destination_extent.extent[0]),
                    static_cast<float>(destination_extent.extent[1]),
                });
            }
            imgEffectLayer->SetFinalBlend(authored_destination_blend);
            imgEffectLayer->SetTransparentCompositionBlend(transparent_destination_blend);
            const auto source_policy = imgEffectLayer->SourceContributionPolicy();
            if (isCompose) {
                LOG_INFO("SceneCompositionLayerSourcePolicy: layer=%d name='%s' "
                         "copybackground=%s policy=%.*s",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         wpimgobj.copybackground ? "true" : "false",
                         static_cast<int>(ImageEffectSourcePolicyName(source_policy).size()),
                         ImageEffectSourcePolicyName(source_policy).data());
            }
            imgEffectLayer->SourceMesh().ChangeMeshDataFrom(mesh);
            if (direct_puppet_source) {
                imgEffectLayer->SetDirectPuppetSource(std::move(*direct_puppet_source));
            }
            if (prelighting_source) {
                imgEffectLayer->SetPrelightingSource(std::move(*prelighting_source));
                svData.SetEffectLayerProjection(imgEffectLayer.get());
            }
            imgEffectLayer->FinalMesh().ChangeMeshDataFrom(effct_final_mesh);
            // Draw the authored image mesh when all effects are hidden. Imported meshes retain
            // their crop geometry and bone attributes; ordinary cards retain source-file UVs
            // instead of a resolved effect texture's full UVs.
            imgEffectLayer->SetDirectDrawMesh(hasStaticImageMesh || hasAnimatedPuppetMesh
                ? imgEffectLayer->FinalMesh() : imgEffectLayer->SourceMesh());
            // The owning SceneObject owns the effect bridge; the private effect camera is a pure
            // projection resource with no back-reference. The bridge records that camera's name so
            // source passes select it explicitly. It never becomes an inherited object camera;
            // geometry updates and destruction reach the projection resource through its owner.
            scene.EnsureSceneObject(wpimgobj.id).SetImageEffectLayer(imgEffectLayer);
            scene.EnsureSceneObject(wpimgobj.id).SetResourceSetupCallback(
                [](Scene& setup_scene, SceneObject& owner) {
                    owner.ImageEffectLayer()->RefreshDestinationTargets(setup_scene);
                });
            imgEffectLayer->SetBridgeCameraName(effect_camera_name);
            imgEffectLayer->AddRuntimeCameraName(effect_camera_name);
        }
        // Destination targets keep the output size resolved at load for fullscreen layers.
        // Only a new layer layout selects another shared extent or recreates its private slot.
        {
            imgEffectLayer->AddRuntimeRenderTargetName(effect_ppong_a);
            imgEffectLayer->AddRuntimeRenderTargetName(effect_ppong_b);
            LOG_INFO("SceneEffectPingPongTargetResolve: layer=%d name='%s' "
                     "pingpong-a='%s' pingpong-b='%s' authored-size=[%.3f, %.3f] "
                     "target=%dx%d policy=%s filter=%s wrap=%s texture='%s' fullscreen=%s",
                     wpimgobj.id,
                     wpimgobj.name.c_str(),
                     effect_ppong_a.c_str(),
                     effect_ppong_b.c_str(),
                     effect_source_size[0],
                     effect_source_size[1],
                     effect_target_width,
                     effect_target_height,
                     destination_extent.policy,
                     wpimgobj.nointerpolation || destination_extent.suffix == 'n' ? "point" : "linear",
                     wpimgobj.clampuvs ? "clamp" : "repeat",
                     primary_source_texture.c_str(),
                     wpimgobj.fullscreen ? "true" : "false");
        }
        {
            // A zero-effect composition still publishes its child source, and a dependency source
            // can also be visible in the enclosing destination. Materialize this draw for every
            // bridge; the graph's live visibility/phase gate decides whether it runs. Sampling
            // slot zero here does not replace the private texture used by readers.
            const auto finalCompositeTransformData = BuildEffectWriterTransformData(
                context,
                BuildImageEffectFinalCompositeContract(wpimgobj));
            ConfigureEffectFinalComposite(context,
                                          *imgEffectLayer,
                                          effect_ppong_a,
                                          wpimgobj.id,
                                          wpimgobj.name,
                                          wpimgobj.colorBlendMode,
                                          &finalCompositeTransformData,
                                          hasAnimatedPuppetMesh ? puppet.get() : nullptr);
        }
        LoadLayerEffects(context, *imgEffectLayer, wpimgobj.effects, effect_target_resolution,
                         baseConstSvs, BuildImageEffectMaterialContract(wpimgobj, *imgEffectLayer),
                         effect_ppong_a, hasAnimatedPuppetMesh ? &shared_puppet_pose : nullptr);

        // The final authored material draws the layer card into the restored destination. Its
        // shader name does not change that rule: custom vertex effects and cursor unprojection
        // need the same layer geometry and object-inclusive MVP as stock effects. Dependencies,
        // skinned surfaces and framebuffer color blending have an additional publication stage
        // and retain their corresponding resource contract.
        if (is_offscreen_dependency_source) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivateDependency);
        } else if (hasAnimatedPuppetMesh) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivatePuppetPublication);
        } else if (has_shader_color_blend) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivateThenPublish);
        } else {
            imgEffectLayer->SetFinalOutputCapability(FinalOutputCapability::SceneAuthoredWriter);
        }
        imgEffectLayer->RefreshPuppetPublicationState();
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
        ConfigureInheritedParentBinding(context, wpimgobj.parent, svData);
        context.scene->sceneGraph->AppendChild(spImgNode);
    } else {
        AttachNodeToScene(context, spImgNode, wpimgobj.parent, wpimgobj.name);
    }
    context.object_nodes[wpimgobj.id] = spImgNode;
    context.scene->AddLayerRuntimeNode(wpimgobj.id, spImgNode.get());
    if (hasAnimatedPuppetMesh) {
        context.object_puppets[wpimgobj.id] = puppet->puppet.get();
    }
    context.shader_updater->SetNodeData(spImgNode.get(), svData);
    RegisterLayerSceneState(
        context, wpimgobj.id, wpimgobj.parent, wpimgobj.attachment, wpimgobj.visible);
    context.scene->ApplyLayerVisibility(wpimgobj.id);
}

void ParseTextObj(ParseContext& context, wpscene::WPTextObject& text_obj) {
    TextLayerRenderContract render_contract;
    render_contract.has_materialized_authored_effects = ! text_obj.effects.empty();
    render_contract.has_visible_authored_effects = std::any_of(
        text_obj.effects.begin(), text_obj.effects.end(), [&](const auto& effect) {
            return ResolveEffectVisibility(effect, context.user_properties);
        });
    render_contract.uses_private_dependency_bridge =
        context.scene->IsLayerOffscreenDependencySource(text_obj.id);
    render_contract.uses_shader_color_blend_bridge =
        UsesShaderColorBlendMode(text_obj.colorBlendMode);

    // Bridge ownership is resolved before materialization and remains stable. The visible-effect
    // part of the contract can later change padding and destination extent without discarding the
    // authored effect materials or creating a second text representation.

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
    auto       spTextNode = std::make_shared<SceneNode>(Vector3f(text_obj.origin.data()),
                                                         Vector3f(text_obj.scale.data()),
                                                         Vector3f(text_obj.angles.data()),
                                                         text_obj.name);
    spTextNode->ID() = text_obj.id;
    spTextNode->AddText(primitive);

    WPShaderValueData svData;
    svData.parallaxDepth = { text_obj.parallaxDepth[0], text_obj.parallaxDepth[1] };

    ConfigureBoneAttachment(context,
                            text_obj.parent,
                            text_obj.attachment,
                            "text object",
                            text_obj.name,
                            svData);

    if (has_effect) {
        auto&             scene       = *context.scene;
        const std::string camera_name = EffectCameraName(text_obj.id);
        primitive->bridge.camera_name = camera_name;
        const auto destination_extent = ResolveTextDestinationExtent(primitive->VisibleDisplaySize());
        primitive->bridge.bridge_backing_extent = {
            static_cast<uint32_t>(destination_extent[0]),
            static_cast<uint32_t>(destination_extent[1]),
        };
        const auto destination_sampler = DestinationRenderTargetSampler(
            text_obj.nointerpolation, text_obj.clampuvs);
        const SceneRenderTarget text_pingpong_target {
            .width = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[0]),
            .height = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[1]),
            .mapWidth = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[0]),
            .mapHeight = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[1]),
            .allowReuse = false,
            .sample = destination_sampler,
        };
        const auto bridge_destination_names = ResolveSceneDestinationRenderTargets(
            scene,
            text_obj.id,
            text_obj.parent,
            render_contract.uses_private_dependency_bridge ||
                render_contract.uses_shader_color_blend_bridge,
            text_pingpong_target);
        primitive->bridge.pingpong_a = bridge_destination_names[0];
        primitive->bridge.pingpong_b = bridge_destination_names[1];

        const std::array<float, 2> effect_target_resolution {
            static_cast<float>(text_pingpong_target.width),
            static_cast<float>(text_pingpong_target.height),
        };

        const auto display_size = primitive->VisibleDisplaySize();
        SceneMesh  effect_final_mesh {};
        RebuildTextPrimitiveVisibleMesh(&effect_final_mesh, *primitive);

        scene.cameras[camera_name] = std::make_shared<SceneCamera>(
            std::max(1, static_cast<int32_t>(std::lround(display_size[0]))),
            std::max(1, static_cast<int32_t>(std::lround(display_size[1]))),
            -1.0f,
            1.0f);
        scene.cameras.at(camera_name)->AttatchNode(context.effect_camera_node);

        auto imgEffectLayer = std::make_shared<SceneImageEffectLayer>(scene.EnsureSceneObject(text_obj.id),
                                                                      display_size[0],
                                                                      display_size[1],
                                                                      primitive->bridge.pingpong_a,
                                                                      primitive->bridge.pingpong_b);
        imgEffectLayer->SetFinalBlend(
            ResolveObjectFinalBlend(BlendMode::Translucent, text_obj.colorBlendMode));
        imgEffectLayer->FinalMesh().ChangeMeshDataFrom(effect_final_mesh);
        // Same contract as image layers: the object owns the bridge, and the bridge records the
        // camera it materialized; the camera itself carries no back-reference.
        scene.EnsureSceneObject(text_obj.id).SetImageEffectLayer(imgEffectLayer);
        imgEffectLayer->SetBridgeCameraName(camera_name);
        imgEffectLayer->AddRuntimeCameraName(camera_name);

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
        for (const auto& wp_effect : text_obj.effects) {
            effect_index++;
            std::shared_ptr<SceneImageEffect> img_effect = std::make_shared<SceneImageEffect>();
            img_effect->SetIdentity(
                text_obj.id, wp_effect.id, static_cast<uint32_t>(effect_index), wp_effect.name);
            const bool effect_initial_visible =
                ResolveEffectVisibility(wp_effect, context.user_properties);
            const bool effect_runtime_visibility =
                EffectVisibilityCanChangeAtRuntime(wp_effect);
            std::unordered_map<std::string, std::string> fbo_map;
            const auto feedback_fbos = wp_effect.FeedbackFboNames();

            for (const auto& wp_fbo : wp_effect.fbos) {
                const std::string rt_name =
                    EffectFboRenderTargetName(wp_fbo, wp_effect.id);
                const auto fbo_size = wp_fbo.ResolveSize(effect_target_resolution);
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
                imgEffectLayer->AddEffectRenderTarget(rt_name, wp_fbo.scale, wp_fbo.fit);
                fbo_map[wp_fbo.name] = rt_name;
            }

            LoadEffectCommands(wp_effect, *img_effect, fbo_map);
            fbo_map.try_emplace("previous", in_rt);

            bool effect_materials_ok = true;
            for (usize material_index = 0; material_index < wp_effect.materials.size();
                 material_index++) {
                wpscene::WPMaterial material_source = wp_effect.materials.at(material_index);
                std::string         material_output { WE_EFFECT_PPONG_PREFIX_B };
                std::vector<usize>   fbo_texture_slots;
                bool                output_is_fbo = false;
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
                        if (img_effect->IsDeclaredFbo(fbo_map.at(bind.name))) {
                            fbo_texture_slots.push_back(static_cast<usize>(bind.index));
                        }
                    }
                    if (! wp_pass.target.empty() && fbo_map.count(wp_pass.target) != 0) {
                        material_output = fbo_map.at(wp_pass.target);
                        output_is_fbo = img_effect->IsDeclaredFbo(material_output);
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
                                   &effect_material,
                                   &effect_node_data,
                                   context.user_properties,
                                   &effect_shader_info)) {
                    effect_materials_ok = false;
                    break;
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
                effect_node_data.text_color_owner = spTextNode.get();
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
                                              .fbo_texture_slots = std::move(fbo_texture_slots),
                                              .output_is_fbo = output_is_fbo,
                                              .sceneNode = spEffectNode,
                                              .advances_composition =
                                                  material_index < wp_effect.passes.size() &&
                                                  wp_effect.passes[material_index].compose });
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

        // Text uses the same final-pass selection as images. A normal effect publishes the
        // shaped card with its object transform; framebuffer color blending retains the separate
        // blend stage, and dependency sources remain sampleable by their consuming layers.
        if (scene.IsLayerOffscreenDependencySource(text_obj.id)) {
            imgEffectLayer->SetFinalOutputCapability(FinalOutputCapability::PrivateDependency);
        } else if (render_contract.uses_shader_color_blend_bridge) {
            imgEffectLayer->SetFinalOutputCapability(FinalOutputCapability::PrivateThenPublish);
        } else {
            imgEffectLayer->SetFinalOutputCapability(FinalOutputCapability::SceneAuthoredWriter);
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
        ConfigureInheritedParentBinding(context, text_obj.parent, svData);
        context.scene->sceneGraph->AppendChild(spTextNode);
    } else {
        AttachNodeToScene(context, spTextNode, text_obj.parent, text_obj.name);
    }

    context.object_nodes[text_obj.id] = spTextNode;
    context.scene->AddLayerRuntimeNode(text_obj.id, spTextNode.get());
    context.shader_updater->SetNodeData(spTextNode.get(), svData);

    context.scene->SetTextLayerState(text_obj.id,
                                     TextLayerRuntimeState {
                                         .object            = text_obj,
                                         .primitive         = primitive,
                                         .render_contract   = render_contract,
                                         .applied_alignment = ResolveTextLayerSceneAlignment(text_obj),
                                     });
    if (has_effect) {
        context.scene->EnsureSceneObject(text_obj.id).SetResourceSetupCallback(
            RefreshTextLayerResources);
    }

    ApplyTextLayerNodePlacement(spTextNode.get(),
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

    ConfigureBoneAttachment(context,
                            empty_obj.parent,
                            empty_obj.attachment,
                            "object",
                            empty_obj.name,
                            svData);

    if (LayerUsesRoutedParent(empty_obj.parent, empty_obj.attachment)) {
        ConfigureInheritedParentBinding(context, empty_obj.parent, svData);
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, empty_obj.parent, empty_obj.name);
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

void PrepareShapeEffectMaterials(WPShapeObject& shape_obj) {
    // Shape output is selected by the owning layer type. Each material-bearing effect pass gets
    // DIRECTDRAW=1 after its authored override is read, even when the file omits the combo or
    // explicitly writes zero. Keep this in the shape parser: the common material merge then
    // applies this pass override over material defaults without changing image or text effects.
    // The parsed pass list contains only material entries, so command markers are unaffected.
    for (auto& effect : shape_obj.effects) {
        for (usize pass_index = 0; pass_index < effect.passes.size(); ++pass_index) {
            auto& combos = effect.passes[pass_index].combos;
            const auto authored = combos.find("DIRECTDRAW");
            LOG_INFO("SceneShapeMaterialPreparation: layer=%d effect-id=%d pass=%zu "
                     "authored-directdraw=%s authored-value=%d final-directdraw=1",
                     shape_obj.id,
                     effect.id,
                     pass_index,
                     authored != combos.end() ? "present" : "absent",
                     authored != combos.end() ? authored->second : 0);
            combos["DIRECTDRAW"] = 1;
        }
    }
}


std::array<float, 2> ResolveImplicitDirectDrawShapeVisualSize(const ParseContext& context) {
    // The default card uses scene ortho height for both axes, including portrait canvases;
    // choosing the shorter edge changes the shape's visible extent.
    const float scene_height = static_cast<float>(context.ortho_h);
    return { scene_height, scene_height };
}

struct DirectDrawShapeMetrics {
    std::array<float, 2> visual_size { 1.0f, 1.0f };
    std::array<float, 2> effect_source_size { 0.0f, 0.0f };
    const char*          visual_policy { "authored-size" };
    const char*          effect_source_policy { "layer-size" };
};

// Shape effect chains run in a destination target that is half the canvas in each dimension
// (integer halves), independent of the shape's own card size. The value is already a pixel
// extent, so it bypasses perspective density scaling; the destination intern applies the common
// minimum-extent clamp.
std::array<float, 2> ResolveDirectDrawShapeEffectTargetSize(const ParseContext& context) {
    const auto extent = ResolveShapeDestinationExtent({ context.ortho_w, context.ortho_h });
    return { static_cast<float>(extent[0]), static_cast<float>(extent[1]) };
}

DirectDrawShapeMetrics ResolveDirectDrawShapeMetrics(const ParseContext& context,
                                                     const WPShapeObject& shape_obj) {
    DirectDrawShapeMetrics metrics;
    // Shapes keep two separate size contracts. The intermediate effect buffers are always the
    // fixed half-canvas shape destination, whether or not the shape carries an authored size;
    // the authored size (or the implicit canvas-height square) only shapes the visible card. This
    // preserves world transforms without forcing the layer through the fullscreen postprocess
    // path.
    metrics.effect_source_size                 = ResolveDirectDrawShapeEffectTargetSize(context);
    metrics.effect_source_policy               = "half-canvas";
    if (shape_obj.has_size) {
        metrics.visual_size = shape_obj.size;
    } else {
        metrics.visual_size   = ResolveImplicitDirectDrawShapeVisualSize(context);
        metrics.visual_policy = "implicit-canvas-height";
    }

    return metrics;
}

void ParseShapeObj(ParseContext& context, WPShapeObject& shape_obj) {
    // The owner is a transform/script identity, not an image source. Materialize it once for
    // both empty and drawable shapes; the effect resource layer below owns the actual draws.
    WPEmptyObject transform;
    transform.id = shape_obj.id;
    transform.name = shape_obj.name;
    transform.origin = shape_obj.origin;
    transform.scale = shape_obj.scale;
    transform.angles = shape_obj.angles;
    transform.parallaxDepth = shape_obj.parallaxDepth;
    transform.parallaxDepthAuthored = shape_obj.parallaxDepthAuthored;
    transform.visible = shape_obj.visible;
    transform.visible_binding = shape_obj.visible_binding;
    transform.parent = shape_obj.parent;
    transform.attachment = shape_obj.attachment;
    ParseEmptyObj(context, transform);

    auto& scene = *context.scene;
    auto& owner = scene.EnsureSceneObject(shape_obj.id);
    const auto metrics = ResolveDirectDrawShapeMetrics(context, shape_obj);
    owner.SetImageRuntimeState(Scene::ImageLayerRuntimeState { .size = metrics.visual_size });
    // Construction/property decoding owns these values even when there is no drawable effect. Do
    // not clamp the stored scalar or manufacture a source material to make the script getter
    // work; property setters copy the owner fields and have no material-update callback. Authored
    // effect controls remain independent below.
    owner.SetModulationState({ shape_obj.color, shape_obj.alpha, shape_obj.brightness });
    LOG_INFO("SceneShapeOwnerState: layer=%d color=[%.6f %.6f %.6f] alpha=%.6f "
             "brightness=%.6f size=[%.3f %.3f] material-controls=independent",
             shape_obj.id, shape_obj.color[0], shape_obj.color[1], shape_obj.color[2],
             shape_obj.alpha, shape_obj.brightness, metrics.visual_size[0], metrics.visual_size[1]);

    if (shape_obj.effects.empty()) {
        LOG_INFO("SceneShapeEmpty: id=%d name='%s' shape='%s' effects=0",
                 shape_obj.id, shape_obj.name.c_str(), shape_obj.shape.c_str());
        return;
    }

    PrepareShapeEffectMaterials(shape_obj);
    const auto target_width =
        ClampDestinationRenderTargetExtent(static_cast<int32_t>(metrics.effect_source_size[0]));
    const auto target_height =
        ClampDestinationRenderTargetExtent(static_cast<int32_t>(metrics.effect_source_size[1]));
    const bool private_destination = scene.IsLayerOffscreenDependencySource(shape_obj.id);
    const SceneRenderTarget destination {
        .width = target_width,
        .height = target_height,
        .mapWidth = target_width,
        .mapHeight = target_height,
        .allowReuse = true,
        .sample = DestinationRenderTargetSampler(false, true),
    };
    const auto targets = ResolveSceneDestinationRenderTargets(
        scene, shape_obj.id, shape_obj.parent, private_destination, destination);
    auto layer = std::make_shared<SceneImageEffectLayer>(
        owner, metrics.visual_size[0], metrics.visual_size[1], targets[0], targets[1]);
    owner.SetImageEffectLayer(layer);
    layer->AddRuntimeRenderTargetName(targets[0]);
    layer->AddRuntimeRenderTargetName(targets[1]);
    layer->SetFinalOutputCapability(private_destination ? FinalOutputCapability::PrivateDependency
                                                       : FinalOutputCapability::SceneAuthoredWriter);
    // The common record dispatcher applies shape's additive state only to the retained final
    // material. There is no source image shader or utility-publication material to stand in for
    // this draw.
    layer->SetFinalBlend(BlendMode::Additive);
    owner.SetResourceSetupCallback(RefreshShapeLayerResources);

    // Shape's material loader receives the material resource and its instance override, not an
    // image source's modulation constants. Preserve shader defaults and authored material values
    // instead of baking unrelated owner fields into every effect. This does not introduce a new
    // engine-global color staging operation.
    const EffectWriterTransformContract transform_contract {
        .parallax_depth = shape_obj.parallaxDepth,
        .projection_layer = layer.get(),
    };
    LoadLayerEffects(context, *layer, shape_obj.effects,
                     { static_cast<float>(target_width), static_cast<float>(target_height) },
                     context.global_base_uniforms, transform_contract, std::nullopt);

    // Geometry uses the first retained effect's first material. Selecting the physical last
    // effect for execution must never truncate that list or change this independent setup.
    RebuildShapeLayerGeometry(owner);
    LOG_INFO("SceneShapeDirectDraw: materialize layer=%d name='%s' shape='%s' effects=%zu "
             "retained-effects=%zu visual-size=[%.3f %.3f] visual-policy=%s "
             "effect-source-policy=%s target=%dx%d source-slot=-1 publication=false",
             shape_obj.id, shape_obj.name.c_str(), shape_obj.shape.c_str(),
             shape_obj.effects.size(), layer->EffectCount(), metrics.visual_size[0],
             metrics.visual_size[1], metrics.visual_policy, metrics.effect_source_policy,
             target_width, target_height);
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

// Registers authored identity and ancestry before materialization. A composition's resource
// requirements include its authored children, including children appearing later in scene order.
// Parent ids therefore belong to this prepass; runtime node/material handles and resolved
// visibility remain in their respective registrars.
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
        if constexpr (requires { value.reflected; }) {
            object.SetReflected(value.reflected);
        }
        if constexpr (std::is_same_v<ValueT, wpscene::WPLightObject>) {
            scene.SetLayerParentBinding(value.id, value.parent, {});
        } else {
            scene.SetLayerParentBinding(value.id, value.parent, value.attachment);
        }
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

    if (object_json.contains("shape") && object_json.at("shape").is_string()) {
        // Runtime creation uses the ordinary shape parser so the canonical owner, retained
        // effects and resource-setup callback exist before property scripts are registered.
        // Subsequent size writes and child-boundary setup then share the same owner state and
        // geometry lifecycle as shapes materialized during initial scene loading.
        WPShapeObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        FillSceneObjectIdentityFor(*context.scene, object);
        ParseShapeObj(context, object);
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

    // Material parsing registers its own user values, timelines and scripts before the owner's
    // property scan. Capture the full creation range here so the host receives every descriptor
    // produced by materialization, including effect passes and model chunks.
    const auto binding_start            = scene.bindingRegistrations.size();
    const auto property_animation_start = scene.propertyAnimationRegistrations.size();
    const auto script_start             = scene.scriptRegistrations.size();
    if (! ParseDynamicSceneObject(context, normalized_object_json, user_properties, &layer_id)) {
        return false;
    }

    auto       node_it = context.object_nodes.find(layer_id);
    SceneNode* layer_node =
        node_it != context.object_nodes.end() && node_it->second ? node_it->second.get() : nullptr;
    const bool has_sound_runtime = scene.GetLayerSoundHandle(layer_id).has_value();
    if (layer_node == nullptr && ! has_sound_runtime) return false;

    scene.layerOrder.push_back(layer_id);
    scene.SetLayerNode(layer_id, layer_node);
    scene.SetLayerInitialConfigJson(layer_id, normalized_object_json.dump());
    RegisterSceneScriptsForObject(context, normalized_object_json);
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

    ScopedJsonUserProperties json_user_scope(user_properties);

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

    // Script registrations resolve the completed authored-object table, including live transforms
    // and initial JSON, rather than retaining parser-owned drawing-node addresses.
    RegisterSceneScripts(context, json);
    context.scene->ApplyAllLayerVisibility();

    ConfigureSceneVolumetricsImpl(*context.scene, *context.vfs);

    return context.scene;
}

std::shared_ptr<Scene> WPSceneParser::Parse(std::string_view scene_id, const std::string& buf,
                                            fs::VFS& vfs, audio::SoundManager& sm) {
    return Parse(scene_id, buf, vfs, sm, nullptr);
}

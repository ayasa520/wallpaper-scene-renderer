#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>

#include "SceneShader.h"
#include "Type.hpp"

namespace wallpaper
{

struct SceneMaterialCustomShader {
    std::shared_ptr<SceneShader> shader;
    ShaderValues                 constValues;
};

enum class SceneCullMode
{
    None,
    Back,
    Front,
};

enum class SceneModelColorLoadMode
{
    DontCare,
    Load,
    Clear,
};

struct SceneModelRenderState {
    // This optional state is attached only by scene-level 3D model materialization. Keeping it out
    // of ordinary image/effect/text materials prevents the model render-policy defaults from
    // changing the historical 2D scene path.
    // Model passes can target either the main scene buffer, which already has the ordinary pre-pass
    // clear contract, or private offscreen buffers that are sampled later by another material. The
    // color load mode makes that ownership explicit: the first offscreen producer clears to
    // transparent, later producers load and composite, and legacy main-target first passes can keep
    // the historical custom-shader load behavior.
    SceneModelColorLoadMode colorLoadMode { SceneModelColorLoadMode::DontCare };
    bool          depthTest { true };
    bool          depthWrite { true };
    // Scene depth is reversed (near = 1, far = 0): the nearest-wins test is always GREATER
    // against a buffer cleared to 0, so there is no per-material compare mode.
    float         depthClear { 0.0f };
};

struct SceneMaterial {
public:
    SceneMaterial()                     = default;
    SceneMaterial(const SceneMaterial&) = default;
    SceneMaterial(SceneMaterial&& o)
        : name(std::move(o.name)),
          textures(std::move(o.textures)),
          systemTextureBindings(std::move(o.systemTextureBindings)),
          defines(std::move(o.defines)),
          uniformAliases(std::move(o.uniformAliases)),
          hasSprite(o.hasSprite),
          customShader(std::move(o.customShader)),
          blenmode(o.blenmode),
          cullMode(o.cullMode),
          modelRenderState(o.modelRenderState),
          alpha_to_coverage(o.alpha_to_coverage) {};

    std::string              name;
    std::vector<std::string> textures;
    // The authored input and the live system property are separate values. An empty property
    // leaves the authored input selected; it does not bind an empty image. Shared property
    // handles also keep copied materials in sync without rewriting effect ping-pong templates or
    // retaining material pointers.
    Map<usize, std::shared_ptr<const std::string>> systemTextureBindings;
    std::vector<std::string> defines;

    const std::string& Texture(usize slot) const {
        const auto binding = systemTextureBindings.find(slot);
        if (binding != systemTextureBindings.end() && !binding->second->empty()) {
            return *binding->second;
        }
        return textures[slot];
    }

    bool SamplesTexture(std::string_view key) const {
        for (usize slot = 0; slot < textures.size(); ++slot) {
            if (Texture(slot) == key) return true;
        }
        return false;
    }

    // Wallpaper Engine scripts address shader controls through authored material names such as
    // `raythreshold`, while the compiled shader consumes GLSL uniforms such as `g_Threshold`.
    // Keeping the parser alias table on the runtime material lets script proxies resolve those
    // authored names without depending on project-specific shader source at assignment time.
    Map<std::string, std::string> uniformAliases;

    const ShaderValue* FindUniformValue(std::string_view uniform_name) const {
        const auto key = std::string(uniform_name);
        const auto value = customShader.constValues.find(key);
        if (value != customShader.constValues.end()) return std::addressof(value->second);
        if (customShader.shader == nullptr) return nullptr;
        const auto initial = customShader.shader->default_uniforms.find(key);
        return initial != customShader.shader->default_uniforms.end()
            ? std::addressof(initial->second) : nullptr;
    }

    bool hasSprite { false };

    SceneMaterialCustomShader customShader;
    BlendMode                 blenmode { BlendMode::Disable };
    // Culling belongs to every shader material, independently of model-only depth/attachment
    // state. Programmatic materials start two-sided; authored material loading installs the
    // parsed selection. Keeping one value also lets live effect writes and pipeline residency
    // observe the same state instead of maintaining separate 2D and model copies.
    SceneCullMode             cullMode { SceneCullMode::None };
    std::optional<SceneModelRenderState> modelRenderState;
    // True when the material blending mode is alphatocoverage. The ALPHATOCOVERAGE
    // combo is fixed at material compile from that blending value; rasterizer A2C
    // is enabled only while the compose target is multisampled.
    bool                      alpha_to_coverage { false };
};
} // namespace wallpaper

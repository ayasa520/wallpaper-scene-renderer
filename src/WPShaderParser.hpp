#pragma once

#include <span>
#include <string_view>
#include "Scene/Scene.h"
#include "Scene/SceneShader.h"
#include "Type.hpp"

namespace wallpaper
{
namespace fs
{
class VFS;
}
using Combos = Map<std::string, std::string>;

// ui material name to gl uniform name
using WPAliasValueDict = Map<std::string, std::string>;

using WPDefaultTexs = std::vector<std::pair<i32, std::string>>;

struct WPShaderInfo {
    Combos           combos;
    ShaderValueMap   svs;
    ShaderValueMap   baseConstSvs;
    WPAliasValueDict alias;
    // Derived from the selected compiled program, not serialized pre-shader metadata. Raw
    // aliases/defaults remain reusable across combos; only this view publishes live controls.
    WPAliasValueDict activeMaterialAliases;
    // Material descriptors retain their declared shader type independently of default values.
    // A scalar-authored vec3 still owns three script/timeline channels, including on cache hits.
    Map<std::string, std::string> materialTypes;
    // Keep scalar units with the declaration metadata, including an empty conversion. A
    // later declaration or cached shader must preserve its own unconverted descriptor rather
    // than inherit an earlier stage's units while the type and alias maps are merged.
    Map<std::string, std::string> materialConversions;
    WPDefaultTexs    defTexs;

    size_t MaterialValueComponents(std::string_view uniform_name) const;
};

struct WPPreprocessorInfo {
    Map<std::string, std::string> input; // name to line
    Map<std::string, std::string> output;
    Map<std::string, std::string> uniforms; // non-sampler uniform name to WE type + optional array suffix

    Set<uint> active_tex_slots;
};

struct WPShaderTexInfo {
    bool                enabled { false };
    // One slot per authored mask component in shader declaration order:
    // metallic, roughness, reflection, emissive.
    std::array<bool, 4> composEnabled { false, false, false, false };
    // Some runtime render targets are meant to be sampled with screen-space UVs reconstructed by
    // the authored shader. The shader preparer adjusts only calls that sample the affected
    // g_TextureN slot, so ordinary textures and unrelated 2D effect targets keep their old UV path.
    bool                screenSpaceSampleYFlip { false };
};

struct WPShaderUnit {
    ShaderType         stage;
    std::string        src;
    WPPreprocessorInfo preprocess_info;
    std::string        debug_name;
};

class WPShaderParser {
public:
    static std::string PreShaderSrc(fs::VFS&, const std::string& src, WPShaderInfo* pWPShaderInfo,
                                    const std::vector<WPShaderTexInfo>& texs);

    static bool CompileToSpv(std::string_view         scene_id, std::span<WPShaderUnit>,
                             std::vector<ShaderCode>& spvs, fs::VFS&, WPShaderInfo*,
                             std::span<const WPShaderTexInfo>);

    // Inspect compiled descriptor usage, including slots introduced by shader defaults.
    // Preprocessor declarations alone do not establish that a texture survives optimization.
    static bool ReflectTextureSlots(std::span<const ShaderCode>, Set<uint>& slots);

    // Query actual uniform-member use in any compiled stage without changing buffer layouts.
    static bool ReflectUniforms(std::span<const ShaderCode>, Set<std::string>& uniforms);
};
} // namespace wallpaper

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "WPUserProperties.hpp"
#include "wpscene/WPMaterial.h"

namespace wallpaper
{
namespace fs
{
class VFS;
}

namespace wpscene
{

// Wallpaper effects are authored resources shared by image layers and first-class text
// primitives. Keeping the declarations in this neutral header prevents WPTextObject from pulling
// in WPImageObject just to describe its optional effect chain.
class WPEffectCommand {
public:
    bool        FromJson(const nlohmann::json&);
    std::string command;
    std::string target;
    std::string source;

    int32_t afterpos { 0 }; // 0 for begin, start from 1
};

class WPEffectFbo {
public:
    bool        FromJson(const nlohmann::json&);
    std::array<int32_t, 2> ResolveSize(std::array<float, 2> source_size) const;
    std::string name;
    std::string format;
    uint32_t    scale { 1 };
    // Wallpaper Engine effect FBOs can use either `scale` (divide the source size) or `fit`
    // (fit the source aspect into a fixed longest edge). Cursor ripple uses `fit=512` for its
    // simulation buffers; treating that as scale=1 makes the propagation texel step several times
    // too small and the authored wavefront appears not to spread.
    uint32_t    fit { 0 };
};

class WPImageEffect {
private:
    static const std::unordered_set<std::string> BLACKLISTED_WORKSHOP_EFFECTS;
    bool IsEffectBlacklisted(const std::string& filePath);
public:
    bool                         FromJson(const nlohmann::json&, fs::VFS& vfs);
    bool                         FromFileJson(const nlohmann::json&, fs::VFS& vfs);
    // Returns true when a parsed effect enables a shader combo either through scene pass overrides
    // or through the resolved material, giving object parsers one semantic query for feature gates.
    bool                         HasEnabledCombo(const std::string& combo_name) const;
    std::unordered_set<std::string> FeedbackFboNames() const;
    int32_t                      id { 0 };
    std::string                  name;
    bool                         visible { true };
    // Keep the authored visible property intact so the parser can build one visibility contract
    // for user bindings, scripts, and animations. A bool-only field is not enough because
    // visible=false with a script still has to materialize an effect runtime target.
    nlohmann::json               visible_json;
    VisibleBinding               visible_binding;
    int32_t                      version;
    std::vector<WPMaterial>      materials;
    std::vector<WPMaterialPass>  passes;
    std::vector<WPEffectCommand> commands;
    std::vector<WPEffectFbo>     fbos;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPEffectFbo, name, scale, fit);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPImageEffect, name, visible, passes, fbos, materials);

} // namespace wpscene
} // namespace wallpaper

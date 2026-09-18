#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
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
    bool        compose { false };

    int32_t afterpos { 0 }; // 0 for begin, start from 1
};

class WPEffectFbo {
public:
    bool        FromJson(const nlohmann::json&);
    std::array<int32_t, 2> ResolveReferenceExtent(std::array<float, 2> source_size) const;
    std::string name;
    std::string format;
    uint8_t     scale { 1 };
    uint16_t    width { std::numeric_limits<uint16_t>::max() };
    uint16_t    height { std::numeric_limits<uint16_t>::max() };
    uint16_t    fit { std::numeric_limits<uint16_t>::max() };
    bool        unique { false };
    // A partial clear string still supplies values to named functions. Setup clearing is a
    // separate authored decision, so preserve it independently of the parsed color components.
    std::array<float, 4> clear_color {};
    bool                clear_on_setup { false };
};

class WPImageEffect {
private:
    static const std::unordered_set<std::string> BLACKLISTED_WORKSHOP_EFFECTS;
    bool IsEffectBlacklisted(const std::string& filePath);
public:
    bool                         FromJson(const nlohmann::json&, fs::VFS& vfs);
    bool                         FromFileJson(const nlohmann::json&, fs::VFS& vfs,
                                              const nlohmann::json& instance);
    std::unordered_set<std::string> FeedbackFboNames() const;
    int32_t                      id { 0 };
    std::string                  name;
    bool                         visible { true };
    // Instance visibility belongs to the scene effect entry, not to the shared effect resource.
    // Preserve the complete property object after loading the resource so parser/runtime binding
    // code can apply the same boolean, user, script, or animation value without changing the
    // material/pass/FBO topology that was already constructed.
    nlohmann::json               visible_json;
    VisibleBinding               visible_binding;
    int32_t                      version;
    // Public material selectors address every admitted pass record, including commands.
    // A populated entry stores the resource name of the next dense material; a command
    // retains a null entry without manufacturing a shader material or a render node.
    std::vector<std::optional<std::string>> material_records;
    std::vector<WPMaterial>      materials;
    std::vector<WPMaterialPass>  passes;
    std::vector<WPEffectCommand> commands;
    std::vector<WPEffectFbo>     fbos;
    std::unordered_map<std::string, std::size_t> clear_functions;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPEffectFbo, name, scale, width, height, fit, unique, clear_color, clear_on_setup);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPImageEffect, name, visible, passes, fbos, materials, clear_functions);

} // namespace wpscene
} // namespace wallpaper

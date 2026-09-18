#pragma once
#include "WPJson.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace wallpaper
{
namespace wpscene
{

class WPMaterialPassBindItem {
public:
    bool        FromJson(const nlohmann::json&, const nlohmann::json& combos);
    std::string name;
    int32_t     index;
};

class WPUserTextureBinding {
public:
    bool        FromJson(const nlohmann::json&);
    bool        empty() const noexcept { return name.empty(); }
    std::string name;
    std::string type;
    bool        keepaspect { false };
};

class WPMaterialPass {
public:
    bool                                                FromJson(const nlohmann::json&,
                                                                 const nlohmann::json& combos);
    // Effect routing belongs to the effect resource. Material input is resolved separately,
    // before WPMaterial interprets shader, raster state, textures and property declarations.
    std::string                                         target;
    std::vector<WPMaterialPassBindItem>                 bind;
    // A composed pass completes one destination step inside the effect. Subsequent passes
    // sample that result through `previous`, independently of the number of material passes.
    bool                                                compose { false };
};

class WPMaterial {
public:
    bool                                                FromJson(const nlohmann::json&,
                                                                 const nlohmann::json& pass_override = nlohmann::json());
    // Static material state starts opaque and back-face culled, with depth testing/writing
    // enabled. Each draw owner separately decides which material states apply to its pass.
    std::string                                         blending { "normal" };
    std::string                                         cullmode { "normal" };
    std::string                                         alphawriting { "default" };
    std::string                                         shader;
    std::string                                         depthtest { "enabled" };
    std::string                                         depthwrite { "enabled" };
    std::vector<std::string>                            textures;
    std::vector<WPUserTextureBinding>                   usertextures;
    nlohmann::json                                     usertexturereference;
    std::unordered_map<std::string, int32_t>            combos;
    // Cold uniform decoding and dynamic registration consume the same raw record, after the
    // material has loaded its shader metadata. Neither may infer a type from JSON value length.
    std::unordered_map<std::string, nlohmann::json>      constantshadervalues;
    std::unordered_map<std::string, std::string>        usershadervalues;

    bool use_puppet { false };
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPMaterialPassBindItem, name, index);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPUserTextureBinding, name, type, keepaspect);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPMaterialPass, bind, target, compose);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPMaterial, blending, shader, textures, usertextures, combos,
                                   constantshadervalues, usershadervalues, usertexturereference);
} // namespace wpscene
} // namespace wallpaper

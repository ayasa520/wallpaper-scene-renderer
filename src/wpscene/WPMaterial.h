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
    bool        FromJson(const nlohmann::json&);
    std::string name;
    int32_t     index;
};

class WPUserTextureBinding {
public:
    bool        FromJson(const nlohmann::json&);
    bool        empty() const noexcept { return name.empty(); }
    std::string name;
    std::string type;
};

class WPMaterialPass {
public:
    bool                                                FromJson(const nlohmann::json&);
    void                                                Update(const WPMaterialPass&);
    std::vector<std::string>                            textures;
    std::vector<WPUserTextureBinding>                   usertextures;
    std::unordered_map<std::string, int32_t>            combos;
    // The shader declaration owns each material property's type. Preserve the complete JSON
    // until that declaration is available: a numeric scalar can initialize a vector, and the
    // same object can retain a user binding, script, and independently sampled timeline. Pass
    // overrides replace that one authored record rather than merging a stale typed sidecar.
    std::unordered_map<std::string, nlohmann::json>      constantshadervalues;
    std::unordered_map<std::string, std::string>        usershadervalues;
    std::string                                         target;
    std::vector<WPMaterialPassBindItem>                 bind;
    // A composed pass completes one destination step inside the effect. Subsequent passes
    // sample that result through `previous`, independently of the number of material passes.
    bool                                                compose { false };
};

class WPMaterial {
public:
    bool                                                FromJson(const nlohmann::json&);
    void                                                MergePass(const WPMaterialPass&);
    std::string                                         blending { "translucent" };
    std::string                                         cullmode { "nocull" };
    std::string                                         shader;
    std::string                                         depthtest { "disabled" };
    std::string                                         depthwrite { "disabled" };
    // Authored-state bits are deliberately stored beside the parsed material. Scene-level 3D
    // models resolve omitted render-state fields through a separate model policy, while 2D
    // image/effect materials must keep the old default strings when the source omitted a field.
    bool                                                blendingAuthored { false };
    bool                                                cullmodeAuthored { false };
    bool                                                depthtestAuthored { false };
    bool                                                depthwriteAuthored { false };
    std::vector<std::string>                            textures;
    std::vector<WPUserTextureBinding>                   usertextures;
    std::unordered_map<std::string, int32_t>            combos;
    // Cold uniform decoding and dynamic registration consume the same raw record, after the
    // material has loaded its shader metadata. Neither may infer a type from JSON value length.
    std::unordered_map<std::string, nlohmann::json>      constantshadervalues;
    std::unordered_map<std::string, std::string>        usershadervalues;

    bool use_puppet { false };
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPMaterialPassBindItem, name, index);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPUserTextureBinding, name, type);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPMaterialPass, bind, target, textures, usertextures, combos,
                                   constantshadervalues, usershadervalues, compose);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPMaterial, blending, shader, textures, usertextures, combos,
                                   constantshadervalues, usershadervalues);
} // namespace wpscene
} // namespace wallpaper

#pragma once
#include <string>
#include <vector>
#include <memory>
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

struct SceneMaterial {
public:
    SceneMaterial()                     = default;
    SceneMaterial(const SceneMaterial&) = default;
    SceneMaterial(SceneMaterial&& o)
        : name(std::move(o.name)),
          textures(std::move(o.textures)),
          defines(std::move(o.defines)),
          uniformAliases(std::move(o.uniformAliases)),
          hasSprite(o.hasSprite),
          customShader(std::move(o.customShader)),
          blenmode(o.blenmode) {};

    std::string              name;
    std::vector<std::string> textures;
    std::vector<std::string> defines;

    // Wallpaper Engine scripts address shader controls through authored material names such as
    // `raythreshold`, while the compiled shader consumes GLSL uniforms such as `g_Threshold`.
    // Keeping the parser alias table on the runtime material lets script proxies resolve those
    // authored names without depending on project-specific shader source at assignment time.
    Map<std::string, std::string> uniformAliases;

    bool hasSprite { false };

    SceneMaterialCustomShader customShader;
    BlendMode                 blenmode { BlendMode::Disable };
};
} // namespace wallpaper

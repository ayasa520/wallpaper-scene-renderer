#pragma once
#include "SpriteAnimation.hpp"
#include <string>
#include <vector>
#include "Type.hpp"

namespace wallpaper
{

struct SceneTexture {
    std::string     url;
    TextureSample   sample;
    bool            isVideo { false };
    bool            isSprite { false };
    i32             width { 0 };
    i32             height { 0 };
    i32             mapWidth { 0 };
    i32             mapHeight { 0 };
    SpriteAnimation spriteAnim;
};
} // namespace wallpaper

#pragma once
#include "SpriteAnimation.hpp"
#include <string>
#include <vector>
#include "Type.hpp"

namespace wallpaper
{

enum class VideoTexturePlaybackState {
    Playing,
    Paused,
    Stopped,
};

struct VideoTextureRuntimeState {
    double   currentTime { 0.0 };
    double   duration { 0.0 };
    double   rate { 1.0 };
    bool     isPlaying { false };
};

struct SceneTexture {
    std::string     url;
    TextureSample   sample;
    // Keep the authored texture format next to the runtime texture record. Shader diagnostics and
    // material binding decisions need the same channel-layout metadata that WPSceneParser already
    // reads from the .tex header; storing it here avoids re-parsing images on the render thread.
    TextureFormat   format { TextureFormat::RGBA8 };
    bool            isVideo { false };
    bool            isSprite { false };
    i32             width { 0 };
    i32             height { 0 };
    i32             mapWidth { 0 };
    i32             mapHeight { 0 };
    i32             mipmapCount { 0 };
    bool            mipmap_larger { false };
    // Actual uploaded GPU extent after optional mip0 drop. 0 means not uploaded yet.
    i32             gpuWidth { 0 };
    i32             gpuHeight { 0 };
    SpriteAnimation spriteAnim;
};
} // namespace wallpaper

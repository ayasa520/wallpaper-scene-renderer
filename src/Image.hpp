#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

#include "Core/Literals.hpp"
#include "Type.hpp"
#include "SpriteAnimation.hpp"
#include "Scene/SceneTexture.h"
#include "Core/NoCopyMove.hpp"

namespace wallpaper
{

union ImageExtra {
    int32_t val { 0 };
    char    str[125];
};

typedef std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> ImageDataPtr;

struct ImageData {
    i32          width { 0 };
    i32          height { 0 };
    isize        size { 0 };
    ImageDataPtr data {};
    ImageData() = default;
};

struct ImageHeader {
    // these two size is not for tex, just come from we
    // using Slot's size for tex
    i32 width { 0 };
    i32 height { 0 };
    i32 mapWidth { 0 };
    i32 mapHeight { 0 };

    bool mipmap_larger { false };
    bool mipmap_pow2 { false };
    bool isVideoTexture { false };

    ImageType     type { ImageType::UNKNOWN };
    TextureFormat format { TextureFormat::RGBA8 };
    i32           count { 0 };
    // Authored mip chain length of the first slot. Official texture-resolution
    // only drops mip0 when this is greater than 1.
    i32           mipmapCount { 0 };

    bool          isSprite { false };
    TextureSample sample;

    SpriteAnimation spriteAnim;
    // for specific property
    std::unordered_map<std::string, ImageExtra> extraHeader;
};

// slot is one singal image
struct Image : NoCopy, NoMove {
    struct Slot {
        i32 width { 0 };
        i32 height { 0 };

        std::vector<ImageData> mipmaps;

        operator bool() { return width * height * std::ssize(mipmaps) > 0; }
    };
    ImageHeader       header;
    std::vector<Slot> slots;
    std::string       key;
    // Content producers advance this revision whenever the pixels associated with a stable key
    // change. GPU preparation policy has an independent lifetime and must never overwrite it.
    uint64_t revision { 0 };
    // Identifies the texture-resolution policy used to prepare the current mip chain. Texture
    // caches compare this independently from content revision so either dimension can invalidate.
    uint64_t textureResolutionEpoch { 0 };
};

} // namespace wallpaper

#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Type.hpp"
#include "WPJson.hpp"
#include "WPUserProperties.hpp"
#include "WPMaterial.h"
#include "WPPuppet.hpp"
#include "wpscene/WPEffect.h"
#include "wpscene/WPParallaxDepth.hpp"

namespace wallpaper
{
namespace fs
{
class VFS;
}

namespace wpscene
{

class WPImageObject {
public:
    struct Config {
        // Model-asset marker (`models/util/composelayer.json`, `projectlayer.json`,
        // `fullscreenlayer.json`): the layer is a routing helper whose destination stays active
        // while its children draw. Read from the model JSON, not from the scene object.
        bool passthrough { false };
    };
    bool                       FromJson(const nlohmann::json&, fs::VFS&);
    int32_t                    id { 0 };
    std::string                name;
    std::array<float, 3>       origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3>       scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3>       angles { 0.0f, 0.0f, 0.0f };
    std::array<float, 2>       size { 2.0f, 2.0f };
    // Fullscreen image layers never parse a scene transform, so they remain screen-space. Every
    // other image resolves the scene default in FromJson before reading the field.
    std::array<float, 2>       parallaxDepth { kScreenSpaceParallaxDepth };
    // Presence distinguishes an omitted default from an explicitly authored value, including zero.
    // The parser uses that distinction to inherit parent parallax for child layers.
    bool                       parallaxDepthAuthored { false };
    std::array<float, 3>       color { 1.0f, 1.0f, 1.0f };
    int32_t                    colorBlendMode { 0 };
    float                      alpha { 1.0f };
    float                      brightness { 1.0f };
    bool                       fullscreen { false };
    bool                       autosize { false };
    // Wallpaper Engine's `models/util/projectlayer.json` stores this marker in the model asset,
    // not on the scene object itself. Keeping it on the parsed image object lets the scene parser
    // distinguish logical framebuffer helper layers from normal drawable image layers.
    bool                       projectlayer { false };
    // Model-asset markers of `models/util/solidlayer.json` and its instanced variant. Together
    // with passthrough they decide whether a textured, non-fullscreen layer sizes its effect
    // destination from the card instead of the texture content.
    bool                       solidlayer { false };
    bool                       instanced { false };
    bool                       nopadding { false };
    bool                       visible { true };
    VisibleBinding             visible_binding;
    std::string                image;
    int32_t                    parent { 0 };
    std::string                attachment;
    std::string                alignment { "center" };
    // The shared constructor enables copybackground before property parsing. An omitted key
    // therefore copies the background; only an explicit false selects a transparent composition
    // source. Child count and dependency status do not alter it.
    bool                       copybackground { true };
    bool                       nointerpolation { false };
    bool                       clampuvs { true };
    WPMaterial                 material;
    std::vector<WPImageEffect> effects;
    Config                     config;

    std::string                                puppet;
    std::vector<WPPuppetLayer::AnimationLayer> puppet_layers;
};

// Image objects now depend on the neutral WPEffect model instead of owning those declarations.
// Text objects include the same effect header directly, which removes the previous text->image
// header dependency while preserving the authored effect JSON shape.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WPImageObject, name, origin, angles, scale, size, visible,
                                   material, effects);

} // namespace wpscene
} // namespace wallpaper

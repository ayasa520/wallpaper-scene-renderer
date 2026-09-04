#pragma once
#include <array>
#include <cstdint>
#include <string>

#include "Scene.h"

namespace wallpaper
{

// Destination render targets (the per-layer images an effect chain draws into) are interned by
// name, and the name is only the pixel size plus a suffix. The same rules serve parse time and
// every later re-layout, so a layer whose extent changes at runtime interns a new name and leaves
// the previous image untouched.

// A layer destination target is never created smaller than 4x4 pixels; the same clamped extent
// is both the interned name and the backing image size.
constexpr int32_t kMinDestinationRenderTargetExtent = 4;

int32_t ClampDestinationRenderTargetExtent(int32_t extent);

std::string SceneDestinationRenderTargetBaseName(int32_t width, int32_t height, char suffix);

// `suffix` is the layer's own destination suffix ('b' bilinear source, 'n' point-sampled source);
// the second name of the pair carries the other suffix so the ping-pong partner stays distinct.
std::array<std::string, 2> SceneDestinationRenderTargetNames(
    const Scene& scene, int32_t parent_id, int32_t width, int32_t height, char suffix = 'b');

// Named-RT lookup keys only the string. A hit returns the existing resource without applying the
// later caller's dimensions; a miss registers `target` under `name`.
const SceneRenderTarget& InternNamedRenderTarget(Scene& scene, const std::string& name,
                                                 SceneRenderTarget target);

} // namespace wallpaper

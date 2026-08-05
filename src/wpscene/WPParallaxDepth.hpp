#pragma once

#include <array>

namespace wallpaper
{
namespace wpscene
{

// Wallpaper Engine omits `parallaxDepth` when a layer uses its 1.0 default. Field presence is
// retained separately: an omitted child inherits its parent parallax contract, while an explicit
// value (including zero) owns an independent depth.
inline constexpr std::array<float, 2> kDefaultParallaxDepth { 1.0f, 1.0f };

// Fullscreen layers have no world transform for camera parallax to displace.
inline constexpr std::array<float, 2> kScreenSpaceParallaxDepth { 0.0f, 0.0f };

} // namespace wpscene
} // namespace wallpaper

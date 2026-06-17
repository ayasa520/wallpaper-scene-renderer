#pragma once

#include <cstdint>
#include <functional>

namespace wallpaper::vulkan
{

using OffscreenFrameReleaseCallback = std::function<bool(std::uint32_t slot_index)>;

} // namespace wallpaper::vulkan

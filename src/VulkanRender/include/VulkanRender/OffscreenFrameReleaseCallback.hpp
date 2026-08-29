#pragma once

#include <cstdint>
#include <functional>

namespace wallpaper::vulkan
{

using OffscreenFrameReleaseCallback = std::function<bool(std::uint32_t slot_index)>;

/*
 * Fired after drawFrameOffscreen() publishes a slot into the exported
 * swapchain ring. Transport plumbing for the out-of-process producer: the IPC
 * relay thread blocks on this instead of polling eatFrame() at a fixed rate.
 * Runs on the render thread, so implementations must only signal and return.
 */
using OffscreenFrameReadyCallback = std::function<void(std::uint32_t slot_index)>;

} // namespace wallpaper::vulkan

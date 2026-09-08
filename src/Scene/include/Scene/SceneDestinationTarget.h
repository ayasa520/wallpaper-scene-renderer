#pragma once
#include <array>
#include <cstdint>
#include <string>

#include "Scene.h"

namespace wallpaper
{

// Shared destination targets are interned by pixel size, sampler flags and slot index. Private
// publication instead owns slot zero by layer id. Resolve both through the same entry point at
// parse time and re-layout so resource lifetime follows the selected naming contract.

// A layer destination target is never created smaller than 4x4 pixels; the same clamped extent
// is both the interned name and the backing image size.
constexpr int32_t kMinDestinationRenderTargetExtent = 4;

int32_t ClampDestinationRenderTargetExtent(int32_t extent);

std::array<int32_t, 2> ResolveCardDestinationExtent(std::array<float, 2> card_size);

std::array<int32_t, 2> ResolveTextDestinationExtent(std::array<float, 2> layout_size);

std::array<int32_t, 2> ResolveShapeDestinationExtent(std::array<int32_t, 2> canvas_size);

TextureSample DestinationRenderTargetSampler(bool point_sampled, bool clamp_uvs);

std::string SceneDestinationRenderTargetBaseName(int32_t width, int32_t height,
                                                 const TextureSample& sampler, size_t slot);

// With private output, slot zero is a newly created owner target; its stable name must never hit
// the shared-name intern table and retain an earlier layout size. Slot one remains shared.
// Replacing a private target invalidates its GPU resource explicitly.
std::array<std::string, 2> ResolveSceneDestinationRenderTargets(
    Scene& scene, int32_t layer_id, int32_t parent_id, bool private_output,
    const SceneRenderTarget& target);

// Named-RT lookup keys only the string. A hit returns the existing resource without applying the
// later caller's dimensions; a miss registers `target` under `name`.
const SceneRenderTarget& InternNamedRenderTarget(Scene& scene, const std::string& name,
                                                 SceneRenderTarget target);

} // namespace wallpaper

#pragma once
#include "SceneTexture.h"
#include "Core/Literals.hpp"
#include <array>
#include <cstdint>

namespace wallpaper
{

// Authored effect FBO sizing is shared by initial materialization and later resource setup.
// Keep the calculation independent of parser objects and of the image/text geometry owner.
std::array<int32_t, 2> ResolveEffectRenderTargetReferenceExtent(
    std::array<float, 2> source_extent, std::array<uint16_t, 2> authored_extent, uint16_t fit);

struct SceneRenderTarget {
    struct Bind {
        bool        enable { false };
        std::string name {};
        bool        screen { false };
        double      scale { 1.0 };
    };

    i32           width { 0 };
    i32           height { 0 };
    i32           mapWidth { 0 };
    i32           mapHeight { 0 };
    bool          allowReuse { false };
    bool          withDepth { false };
    // Depth-only comparison target sampled with texSample2DCompare.
    bool          comparisonDepth { false };
    // Model reflection buffers are sampled by authored shaders that derive UVs from clip-space
    // screen coordinates rather than ordinary mesh texture coordinates. This opt-in sampling
    // convention lets material-bound slots fix that Y direction at shader preparation time without
    // changing how the producer pass renders or how legacy 2D effect targets are sampled.
    bool          screenSpaceSampleYFlip { false };
    bool          has_mipmap { false };
    uint          mipmap_level { 1 };
    // 1 = no MSAA. Compose color uses 2/4/8 when the host anti-aliasing tier is x2/x4/x8.
    int           sample_count { 1 };
    TextureSample sample { TextureWrap::CLAMP_TO_EDGE,
                           TextureWrap::CLAMP_TO_EDGE,
                           TextureFilter::LINEAR,
                           TextureFilter::LINEAR };
    Bind          bind {};

    // Reference dimensions describe the last resource setup before integer scaling. The first
    // creator owns the divisor, even when another layer later acquires and resizes this name.
    // Ordinary unscaled targets start with identical reference and physical extents.
    std::array<i32, 2> reference_extent { width, height };
    uint32_t          resolution_divisor { 1 };
    // A logical resize discards the old backing contents even when the divided size is unchanged.
    // Carry that decision into the allocation key instead of relying only on physical dimensions.
    uint64_t          allocation_revision { 0 };

    static SceneRenderTarget FromReferenceExtent(std::array<i32, 2> extent, uint32_t divisor);
    bool ResizeReferenceExtent(std::array<i32, 2> extent);

    [[nodiscard]] i32 ContentWidth() const {
        // Render targets can expose a logical content rectangle that is smaller than their
        // allocated Vulkan image. This remains a generic render-target capability for authored
        // effect chains that intentionally decouple sampled content size from backing allocation.
        return mapWidth > 0 ? mapWidth : width;
    }

    [[nodiscard]] i32 ContentHeight() const {
        // Keep the fallback symmetric with ContentWidth() so targets that never set
        // mapWidth/mapHeight continue to behave exactly as before.
        return mapHeight > 0 ? mapHeight : height;
    }

    [[nodiscard]] std::array<i32, 4> ResolutionVector() const {
        return { width, height, ContentWidth(), ContentHeight() };
    }
};
} // namespace wallpaper

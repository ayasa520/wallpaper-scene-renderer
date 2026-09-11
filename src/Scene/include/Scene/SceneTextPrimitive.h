#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Image.hpp"
#include "SceneDraw.h"
#include "SceneMesh.h"
#include "wpscene/WPTextObject.h"

namespace wallpaper
{

struct TextGlyphRun {
    uint32_t             page_index { 0 };
    std::array<float, 4> source_rect { 0.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 4> atlas_rect { 0.0f, 0.0f, 0.0f, 0.0f };
};

struct TextGlyphAtlasPage {
    std::string            texture_key;
    std::shared_ptr<Image> image;
    std::array<float, 2>   source_size { 0.0f, 0.0f };
};

struct TextLayoutResult {
    // The canonical logical box always tracks the authored text rectangle after shaping. This is
    // the box used by opaque-background rendering and by exact-size text bridges.
    std::array<float, 2> logical_size { 0.0f, 0.0f };
    std::array<float, 2> logical_source_size { 0.0f, 0.0f };

    // Glyph-only text can expose cropped glyph bounds as visible geometry, but placement still
    // belongs to `logical_size`. The cropped visible quad starts from its measured local offset from
    // that logical rectangle; WPTextLayer resolves the final mesh center from these crop metrics and
    // the authored alignment/origin.
    std::array<float, 2> glyph_display_size { 0.0f, 0.0f };
    std::array<float, 2> glyph_source_size { 0.0f, 0.0f };
    std::array<float, 2> glyph_offset { 0.0f, 0.0f };
    std::array<float, 4> glyph_source_crop { 0.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 2> visible_display_size { 0.0f, 0.0f };
    std::array<float, 2> visible_source_size { 0.0f, 0.0f };
    std::array<float, 2> visible_display_offset { 0.0f, 0.0f };

    // These values describe independent renderer contracts. Point-size conversion changes
    // authored glyph geometry, while backing density only changes atlas resolution; neither object
    // scale nor the scene camera is folded into either value.
    float point_size_authoring_units { 0.0f };
    float backing_density { 1.0f };

    std::vector<TextGlyphAtlasPage> glyph_pages;
    std::vector<TextGlyphRun>       glyph_runs;
};

struct TextSourceBridge {
    std::string camera_name;
    std::string pingpong_a;
    std::string pingpong_b;

    // The initializer is an owner-bound raster phase, not a second text layer. Keep its
    // material and identity across layout replacement; each graph invocation owns its GPU
    // bindings while the primitive refreshes the shared, current-size source card.
    std::shared_ptr<SceneDrawPhase> framebuffer_source;

    // The destination extent of the current shaped text box (clamped to the destination minimum).
    // Re-layout selects a new pair by name when this extent changes; earlier entries retain their
    // original size in the scene's intern table.
    std::array<uint32_t, 2> bridge_backing_extent { 1u, 1u };
};

struct TextLayerRenderContract {
    bool has_materialized_authored_effects { false };
    // Resource ownership is stable while visibility changes. Padding instead follows the visible
    // effect count. An all-hidden ordinary chain draws the glyph primitive directly while its
    // bridge resources remain resident; independent private publication still needs padding.
    bool has_visible_authored_effects { false };
    bool uses_private_dependency_bridge { false };
    bool uses_shader_color_blend_bridge { false };

    [[nodiscard]] bool RequiresBridge() const {
        return has_materialized_authored_effects || uses_private_dependency_bridge ||
               uses_shader_color_blend_bridge;
    }
    [[nodiscard]] bool UsesEffectPadding() const {
        return has_visible_authored_effects || uses_private_dependency_bridge ||
               uses_shader_color_blend_bridge;
    }
};

class SceneTextPrimitive {
public:
    struct GlyphPageRenderable {
        uint32_t                 page_index { 0 };
        std::string              texture_key;
        std::array<float, 2>     source_size { 0.0f, 0.0f };
        std::shared_ptr<SceneMesh> mesh;
    };

    // The primitive is scene-owned state: authored text properties, canonical layout, atlas page
    // meshes, material data, and optional bridge metadata all live here so parser, runtime, and
    // render graph consume one final representation instead of synthetic image-layer sidecars.
    wpscene::WPTextObject object;
    TextLayerRenderContract render_contract;
    TextLayoutResult      layout;
    TextSourceBridge      bridge;
    std::shared_ptr<SceneMesh> background_mesh;
    std::vector<GlyphPageRenderable> glyph_pages;
    uint32_t              atlas_version { 0 };

    // Direct background material selection belongs to the text owner, not to a glyph layout or
    // a render-graph pass. Select it only when a direct opaque background is actually submitted;
    // private source clears and pipeline preparation must leave an unused selection untouched.
    // Layout replacement carries this value forward, while a newly created owner starts empty.
    std::optional<bool> direct_background_depth_test;

    // Standard-range text uses the current owner RGB and alpha. Foreground brightness is
    // an HDR modulation and remains separate from this color on the renderer's standard-range
    // output. Glyph draws and authored effect uniforms consume this same source state.
    [[nodiscard]] std::array<float, 4> ForegroundColor() const {
        return { object.color[0], object.color[1], object.color[2], object.alpha };
    }

    // Background brightness, like foreground brightness, is an HDR-only modulation. Both direct
    // background draws and private source clears consume the same current standard-range color.
    [[nodiscard]] std::array<float, 4> BackgroundColor() const {
        return { object.backgroundcolor[0], object.backgroundcolor[1], object.backgroundcolor[2],
                 object.alpha };
    }

    [[nodiscard]] std::array<float, 2> VisibleDisplaySize() const { return layout.visible_display_size; }
    [[nodiscard]] std::array<float, 2> VisibleSourceSize() const { return layout.visible_source_size; }
    [[nodiscard]] std::array<float, 2> VisibleDisplayOffset() const { return layout.visible_display_offset; }
    [[nodiscard]] std::array<float, 2> BackgroundLocalOffset() const {
        return object.opaquebackground
            ? std::array<float, 2> { 0.0f, 0.0f }
            : std::array<float, 2> { -layout.visible_display_offset[0], -layout.visible_display_offset[1] };
    }
};

} // namespace wallpaper

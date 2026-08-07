#pragma once

namespace wallpaper
{

enum class ParticleRendererKind {
    Sprite,
    SpriteTrail,
    Rope,
    RopeTrail,
};

enum class ParticleShaderSelection {
    Authored,
    StockRope,
};

enum class ParticleExpansionMode {
    GeometryPoint,
    IndexedQuad,
};

// This value is the resolved particle rendering ABI shared by scene parsing, mesh construction,
// and per-frame vertex generation. Keeping the decision in one typed object prevents those three
// layers from independently inferring shader ownership, topology, or history semantics.
struct ParticleRenderPlan {
    ParticleRendererKind    renderer { ParticleRendererKind::Sprite };
    ParticleShaderSelection shader_selection { ParticleShaderSelection::Authored };
    ParticleExpansionMode   expansion { ParticleExpansionMode::GeometryPoint };
    bool                    thick_format { false };

    constexpr bool IsRope() const noexcept {
        return renderer == ParticleRendererKind::Rope ||
            renderer == ParticleRendererKind::RopeTrail;
    }

    constexpr bool IsTrail() const noexcept {
        return renderer == ParticleRendererKind::SpriteTrail ||
            renderer == ParticleRendererKind::RopeTrail;
    }

    constexpr bool UsesHistory() const noexcept {
        return renderer == ParticleRendererKind::RopeTrail;
    }
};

} // namespace wallpaper

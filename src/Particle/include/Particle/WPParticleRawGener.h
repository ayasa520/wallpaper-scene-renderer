#pragma once
#include "Interface/IParticleRawGener.h"
#include <functional>

namespace wallpaper
{

class WPParticleRawGener final : public IParticleRawGener {
public:
    void GenGLData(std::span<const std::unique_ptr<ParticleInstance>>, SceneMesh&,
                   ParticleRawGenSpecOp&, const ParticleRenderPlan&,
                   std::string_view object_name, float alpha_multiplier) override;
};

} // namespace wallpaper

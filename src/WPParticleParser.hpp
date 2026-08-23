#pragma once
#include "Particle/ParticleEmitter.h"
#include "wpscene/WPParticleObject.h"

namespace wallpaper
{
class WPParticleParser {
public:
    static ParticleInitOp     genParticleInitOp(const nlohmann::json&,
                                                ParticleAudioResponseFactor audio_factor = {});
    static ParticleOperatorOp genParticleOperatorOp(const nlohmann::json&,
                                                    const wpscene::ParticleInstanceoverride&,
                                                    ParticleAudioResponseFactor audio_factor = {});
    static ParticleEmittOp genParticleEmittOp(const wpscene::Emitter&,
                                              ParticleAudioResponseFactor audio_rate_factor,
                                              ParticleEmitterTiming& timing);
    static ParticleInitOp  genOverrideInitOp(const wpscene::ParticleInstanceoverride&);
};
} // namespace wallpaper

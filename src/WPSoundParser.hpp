#pragma once

#include <cstdint>
#include <memory>

namespace wallpaper 
{

namespace audio {
class SoundManager;
class ScenePlaybackState;
using SoundHandle = uint32_t;
}
namespace fs { class VFS; }
namespace wpscene { class WPSoundObject; }
class WPSoundParser {
public:
    static audio::SoundHandle Parse(const wpscene::WPSoundObject&, fs::VFS&, audio::SoundManager&,
                                    const std::shared_ptr<audio::ScenePlaybackState>&);
};
}

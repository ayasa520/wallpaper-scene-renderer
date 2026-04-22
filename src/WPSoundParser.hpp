#pragma once

#include <cstdint>

namespace wallpaper 
{

namespace audio {
class SoundManager;
using SoundHandle = uint32_t;
}
namespace fs { class VFS; }
namespace wpscene { class WPSoundObject; }
class WPSoundParser {
public:
    static audio::SoundHandle Parse(const wpscene::WPSoundObject&, fs::VFS&, audio::SoundManager&);
};
}

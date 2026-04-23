#include "WPSoundParser.hpp"
#include "Audio/SoundManager.h"
#include "Core/Random.hpp"
#include "Fs/VFS.h"
#include "wpscene/WPSoundObject.h"
#include "Utils/Logging.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <limits>

using namespace wallpaper;

enum class PlaybackMode
{
    Single,
    Random,
    Loop
};

static PlaybackMode ToPlaybackMode(std::string_view s) {
    if (s == "single")
        return PlaybackMode::Single;
    else if (s == "loop")
        return PlaybackMode::Loop;
    else if (s == "random")
        return PlaybackMode::Random;
    return PlaybackMode::Loop;
};

class WPSoundStream : public audio::SoundStream {
public:
    struct Config {
        float        maxtime { 10.0f };
        float        mintime { 0.0f };
        float        volume { 1.0f };
        PlaybackMode mode { PlaybackMode::Loop };
    };
    WPSoundStream(const std::vector<std::string>& paths, fs::VFS& vfs, Config c)
        : vfs(vfs), m_config(c), m_soundPaths(paths) {};
    virtual ~WPSoundStream() = default;

    uint64_t NextPcmData(void* pData, uint32_t frameCount) override {
        if (m_soundPaths.empty()) return 0;

        if (m_config.mode == PlaybackMode::Random && ! m_curActive) {
            // Wallpaper Engine random sound layers are ambient emitters: they stay mounted, wait a
            // randomized min/max interval, and then play one randomly selected authored sound.
            // Returning silent frames during the wait prevents SoundManager from marking the
            // channel ended while no clip is currently active.
            if (! m_randomDelayScheduled) ScheduleRandomDelay();
            if (m_randomDelayFramesRemaining > 0) return DrainRandomDelay(pData, frameCount);
            m_randomDelayScheduled = false;
        }

        // first
        if (! m_curActive) {
            Switch();
        }
        if (! m_curActive) return 0;

        // A "single" sound layer is an event sound effect in Wallpaper Engine.  Returning zero at
        // EOF lets SoundManager mark the channel as ended; the next script play() call resets the
        // stream and plays the effect exactly once again.
        uint64_t frameReads = m_curActive->NextPcmData(pData, frameCount);
        if (frameReads == 0) {
            if (m_config.mode == PlaybackMode::Single) {
                LOG_INFO("SceneSoundEnd: mode='single' paths=%zu", m_soundPaths.size());
                m_curActive.reset();
                return 0;
            }

            if (m_config.mode == PlaybackMode::Random) {
                // A finished random clip must not immediately chain into another clip.  Schedule
                // the authored ambient delay and emit silence so the channel remains mounted.
                m_curActive.reset();
                ScheduleRandomDelay();
                if (m_randomDelayFramesRemaining > 0) return DrainRandomDelay(pData, frameCount);
                m_randomDelayScheduled = false;
            }

            // Loop and random sound layers keep producing audio by selecting the next authored
            // asset whenever the active decoder reaches EOF.
            Switch();
            if (! m_curActive) return 0;
            frameReads = m_curActive->NextPcmData(pData, frameCount);
        }
        if (frameReads < frameCount) {
            // The mixer consumes a fixed-size scratch buffer regardless of how many frames the
            // decoder returned.  Clearing the unread tail prevents stale samples after short
            // one-shot and random clips finish near the end of a callback.
            FillSilence(static_cast<float*>(pData) + frameReads * m_desc.channels,
                        frameCount - static_cast<uint32_t>(frameReads));
        }
        // volume
        {
            float*     pData_float = static_cast<float*>(pData);
            const auto num         = frameReads * m_desc.channels;
            for (uint i = 0; i < num; i++, pData_float++) {
                (*pData_float) *= m_config.volume;
            }
        }
        return frameReads;
    };
    void PassDesc(const Desc& d) override { m_desc = d; }
    void Reset() override {
        m_curActive.reset();
        // Use an invalid sentinel instead of the last index so random mode can tell the first
        // selection from a real previous clip, while loop mode still wraps to index zero naturally.
        m_curIndex = std::numeric_limits<uint32_t>::max();
        m_randomDelayFramesRemaining = 0;
        m_randomDelayScheduled       = false;
    }
    void Switch() {
        if (m_soundPaths.empty()) return;

        const std::string path = m_soundPaths[LoopIndex()];
        auto              stream = vfs.Open("/assets/" + path);
        if (! stream) {
            // Keep missing packaged assets visible in run.log with the authored path. Without this
            // log a failed music selection only looks like silence from the script side.
            LOG_ERROR("SceneSoundSwitch: asset-open-failed path='%s'", path.c_str());
            m_curActive.reset();
            return;
        }

        m_curActive = audio::CreateSoundStream(std::move(stream), m_desc);
        if (! m_curActive) {
            // Decoder creation is deferred until the stream is actually played, so the switch log
            // carries the sound asset path that the generic SoundManager layer cannot know.
            LOG_ERROR("SceneSoundSwitch: decoder-create-failed path='%s' channels=%u sample-rate=%u",
                      path.c_str(),
                      m_desc.channels,
                      m_desc.sampleRate);
            return;
        }

        LOG_INFO("SceneSoundSwitch: path='%s' channels=%u sample-rate=%u",
                 path.c_str(),
                 m_desc.channels,
                 m_desc.sampleRate);
    }
    uint32_t LoopIndex() {
        if (m_config.mode == PlaybackMode::Random) {
            if (m_soundPaths.size() == 1) {
                m_curIndex = 0;
                return m_curIndex;
            }

            if (m_curIndex >= m_soundPaths.size()) {
                // There is no previous clip on the first random selection, so every authored asset
                // should be eligible.
                m_curIndex = Random::get<uint32_t>(0, static_cast<uint32_t>(m_soundPaths.size() - 1));
                return m_curIndex;
            }

            // Avoid immediate repeats by drawing from the N-1 other entries and remapping the
            // compact random range around the current index.
            uint32_t next = Random::get<uint32_t>(0, static_cast<uint32_t>(m_soundPaths.size() - 2));
            if (next >= m_curIndex) next++;
            m_curIndex = next;
            return m_curIndex;
        }

        m_curIndex++;
        if (m_curIndex == m_soundPaths.size()) m_curIndex = 0;
        return m_curIndex;
    }
    void ScheduleRandomDelay() {
        const float lower = std::max(0.0f, std::min(m_config.mintime, m_config.maxtime));
        const float upper = std::max(lower, std::max(m_config.mintime, m_config.maxtime));
        const double delaySeconds =
            Random::get<double>(static_cast<double>(lower), static_cast<double>(upper));

        m_randomDelayFramesRemaining =
            static_cast<uint64_t>(delaySeconds * static_cast<double>(m_desc.sampleRate));
        m_randomDelayScheduled = true;
        LOG_INFO("SceneSoundRandomDelay: delay=%.3f frames=%llu",
                 delaySeconds,
                 static_cast<unsigned long long>(m_randomDelayFramesRemaining));
    }
    void FillSilence(float* pData, uint32_t frameCount) {
        if (pData == nullptr || frameCount == 0 || m_desc.channels == 0) return;
        std::fill_n(pData, static_cast<size_t>(frameCount) * m_desc.channels, 0.0f);
    }
    uint64_t DrainRandomDelay(void* pData, uint32_t frameCount) {
        FillSilence(static_cast<float*>(pData), frameCount);
        const auto drained = std::min<uint64_t>(m_randomDelayFramesRemaining, frameCount);
        m_randomDelayFramesRemaining -= drained;
        return frameCount;
    }

private:
    fs::VFS& vfs;
    Config   m_config;
    Desc     m_desc;
    uint32_t m_curIndex { std::numeric_limits<uint32_t>::max() };
    uint64_t m_randomDelayFramesRemaining { 0 };
    bool     m_randomDelayScheduled { false };

    const std::vector<std::string> m_soundPaths;
    std::unique_ptr<SoundStream>   m_curActive;
};

audio::SoundHandle WPSoundParser::Parse(const wpscene::WPSoundObject& obj, fs::VFS& vfs,
                                        audio::SoundManager& sm) {
    WPSoundStream::Config config { .maxtime = obj.maxtime,
                                   .mintime = obj.mintime,
                                   .volume  = 1.0f,
                                   .mode    = ToPlaybackMode(obj.playbackmode) };

    auto ss = std::make_unique<WPSoundStream>(obj.sound, vfs, config);
    // startsilent is a playback gate, not a visibility gate.  Sound layers must still be mounted so
    // thisScene.getLayer(name) can resolve them, while autoplay stays disabled until script calls
    // play() on the selected layer.
    const bool autoplay = obj.visible && ! obj.startsilent;
    const auto handle   = sm.MountStream(std::move(ss), obj.volume, autoplay);
    LOG_INFO("SceneSoundMount: layer=%d name='%s' handle=%u sounds=%zu volume=%.3f visible=%s "
             "startsilent=%s autoplay=%s playbackmode='%s'",
             obj.id,
             obj.name.c_str(),
             handle,
             obj.sound.size(),
             obj.volume,
             obj.visible ? "true" : "false",
             obj.startsilent ? "true" : "false",
             autoplay ? "true" : "false",
             obj.playbackmode.c_str());
    return handle;
}

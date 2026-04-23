#include "Audio/SoundManager.h"
#include "miniaudio-wrapper.hpp"
#include "Fs/IBinaryStream.h"
#include "Core/Literals.hpp"
#include "Utils/Logging.h"

#include <algorithm>
#include <unordered_map>

using namespace wallpaper;
using namespace wallpaper::audio;

namespace
{

SoundStream::Desc ToSSDesc(const miniaudio::DeviceDesc& d) {
    return { .channels = d.phyChannels, .sampleRate = d.sampleRate };
}
miniaudio::DeviceDesc ToSSDesc(const SoundStream::Desc& d) {
    return { .phyChannels = d.channels, .sampleRate = d.sampleRate };
}

} // namespace

class Channel_Impl : public miniaudio::Channel {
public:
    Channel_Impl(std::unique_ptr<SoundStream>&& ss, float volume, bool autoplay)
        : m_ss(std::move(ss)), m_playing(autoplay), m_volume(std::clamp(volume, 0.0f, 1.0f)) {}
    virtual ~Channel_Impl() = default;

    ma_uint64 NextPcmData(void* pData, ma_uint32 frameCount) override {
        if (!m_playing || m_ended) return 0;
        const ma_uint64 frames_read = m_ss->NextPcmData(pData, frameCount);
        if (frames_read == 0) MarkEnded();
        return frames_read;
    }
    void PassDeviceDesc(const miniaudio::DeviceDesc& desc) override {
        m_ss->PassDesc(ToSSDesc(desc));
    }
    bool IsPlaying() const override { return m_playing && !m_ended; }
    bool IsEnded() const override { return m_ended; }
    bool ShouldRemove() const override { return m_detached; }
    float Volume() const override { return m_volume; }

    void Play() {
        if (m_ended) {
            m_ss->Reset();
            m_ended = false;
        }
        m_playing = true;
    }

    void Pause() { m_playing = false; }

    void Stop() {
        m_ss->Reset();
        m_playing = false;
        m_ended = false;
    }

    void MarkEnded() {
        m_playing = false;
        m_ended = true;
    }

    void SetVolume(float volume) { m_volume = std::clamp(volume, 0.0f, 1.0f); }
    void Detach() {
        m_playing  = false;
        m_detached = true;
    }

private:
    miniaudio::DeviceDesc        m_desc;
    std::unique_ptr<SoundStream> m_ss;
    bool                         m_playing { true };
    bool                         m_ended { false };
    bool                         m_detached { false };
    float                        m_volume { 1.0f };
};

struct BStreamWrapper {
    std::shared_ptr<wallpaper::fs::IBinaryStream> stream;
    size_t                                        Read(void* pBufferOut, size_t bytesToRead) {
        if (! stream) return 0;
        size_t reads = stream->Read(pBufferOut, bytesToRead);
        // LOG_INFO("r:%u, %u",bytesToRead, reads);
        return reads;
    }
    bool Seek(idx offset, ma_seek_origin origin) {
        if (! stream) return false;
        bool result { false };
        switch (origin) {
        case ma_seek_origin_start: result = stream->SeekSet(offset); break;
        case ma_seek_origin_current: result = stream->SeekCur(offset); break;
        case ma_seek_origin_end: result = stream->SeekEnd(offset); break;
        }
        // LOG_INFO("s:%u, %d",offset, result);
        return result;
    }
};

template<typename T>
class SoundStream_impl : public SoundStream {
public:
    SoundStream_impl(std::unique_ptr<T>&& ss): m_ss(std::move(ss)) {}
    virtual ~SoundStream_impl() {}

    uint64_t NextPcmData(void* pData, uint32_t frameCount) override {
        return m_ss->NextPcmData(pData, frameCount);
    }
    void PassDesc(const Desc&) override {}
    void Reset() override { m_ss->Reset(); }

private:
    std::unique_ptr<T> m_ss;
};

std::unique_ptr<SoundStream>
wallpaper::audio::CreateSoundStream(std::shared_ptr<wallpaper::fs::IBinaryStream> stream,
                                    const SoundStream::Desc&                      desc) {
    if (! stream) {
        // Audio assets are often opened lazily by scene scripts.  Returning nullptr instead of
        // constructing a decoder over a null stream lets the caller log the authored asset path and
        // prevents a silent null dereference inside miniaudio callbacks.
        LOG_ERROR("CreateSoundStream: stream is null");
        return nullptr;
    }

    BStreamWrapper sw { stream };
    auto           decoder = std::make_unique<miniaudio::Decoder<BStreamWrapper>>(std::move(sw));
    if (! decoder->Init(ToSSDesc(desc))) {
        // Propagate decoder initialization failure to the sound-layer wrapper.  The wrapper owns
        // the Wallpaper Engine asset path, so it can emit a more useful per-layer diagnostic after
        // this generic audio layer reports the codec failure.
        return nullptr;
    }
    return std::make_unique<SoundStream_impl<miniaudio::Decoder<BStreamWrapper>>>(
        std::move(decoder));
}

class SoundManager::impl : NoCopy, NoMove {
public:
    impl(): device() {};
    ~impl() = default;
    miniaudio::Device                                      device {};
    SoundHandle                                            next_handle { 1 };
    std::unordered_map<SoundHandle, std::shared_ptr<Channel_Impl>> channels;
};

SoundManager::SoundManager(): pImpl(std::make_unique<impl>()) {}
SoundManager::~SoundManager() {}

SoundHandle SoundManager::MountStream(std::unique_ptr<SoundStream>&& ss, float volume, bool autoplay) {
    if (! ss) {
        // Dynamic and authored sound layers share this entry point.  A zero handle keeps callers
        // from registering a script-visible layer whose backing audio stream cannot ever play.
        LOG_ERROR("SoundManager::MountStream failed: null sound stream");
        return 0;
    }

    const SoundHandle handle = pImpl->next_handle++;
    auto channel = std::make_shared<Channel_Impl>(std::move(ss), volume, autoplay);
    pImpl->channels.emplace(handle, channel);
    pImpl->device.MountChannel(std::move(channel));
    return handle;
}

void SoundManager::Test(std::shared_ptr<fs::IBinaryStream> stream) {
    BStreamWrapper sw { stream };
    auto           decoder = std::make_unique<miniaudio::Decoder<BStreamWrapper>>(std::move(sw));
}
bool SoundManager::Init() {
    if (Muted()) {
        LOG_INFO("muted, not init sound device");
        return false;
    }
    return pImpl->device.Init({});
}
bool SoundManager::IsInited() const { return pImpl->device.IsInited(); }
void SoundManager::Play() { pImpl->device.Start(); }
void SoundManager::Pause() { pImpl->device.Stop(); }

bool SoundManager::UnmountStream(SoundHandle handle) {
    auto it = pImpl->channels.find(handle);
    if (it == pImpl->channels.end() || !it->second) return false;
    it->second->Detach();
    pImpl->channels.erase(it);
    return true;
}

bool SoundManager::Play(SoundHandle handle) {
    auto it = pImpl->channels.find(handle);
    if (it == pImpl->channels.end() || !it->second) return false;
    it->second->Play();
    return true;
}

bool SoundManager::Pause(SoundHandle handle) {
    auto it = pImpl->channels.find(handle);
    if (it == pImpl->channels.end() || !it->second) return false;
    it->second->Pause();
    return true;
}

bool SoundManager::Stop(SoundHandle handle) {
    auto it = pImpl->channels.find(handle);
    if (it == pImpl->channels.end() || !it->second) return false;
    it->second->Stop();
    return true;
}

bool SoundManager::IsPlaying(SoundHandle handle) const {
    auto it = pImpl->channels.find(handle);
    return it != pImpl->channels.end() && it->second && it->second->IsPlaying();
}

float SoundManager::StreamVolume(SoundHandle handle) const {
    auto it = pImpl->channels.find(handle);
    return it != pImpl->channels.end() && it->second ? it->second->Volume() : 0.0f;
}

bool SoundManager::SetStreamVolume(SoundHandle handle, float volume) {
    auto it = pImpl->channels.find(handle);
    if (it == pImpl->channels.end() || !it->second) return false;
    it->second->SetVolume(volume);
    return true;
}

void SoundManager::UnMountAll() {
    for (auto& [handle, channel] : pImpl->channels) {
        (void)handle;
        if (channel) channel->Detach();
    }
    pImpl->channels.clear();
    pImpl->device.UnmountAll();
}
float SoundManager::Volume() const { return pImpl->device.Volume(); }

bool SoundManager::Muted() const { return pImpl->device.Muted(); }
void SoundManager::SetMuted(bool v) {
    pImpl->device.SetMuted(v);
    // Keep mounted streams alive across mute/unmute. Recreate the device only
    // when audio was never initialized, such as scenes loaded while muted.
    if (! Muted() && ! pImpl->device.IsInited()) {
        Init();
    }
}
void SoundManager::SetVolume(float v) { pImpl->device.SetVolume(v); }
void SoundManager::GetSpectrum(uint32_t resolution,
                               std::vector<float>* left,
                               std::vector<float>* right,
                               std::vector<float>* average) const {
    pImpl->device.GetSpectrum(resolution, left, right, average);
}

#pragma once

#include "ThreadTimer.hpp"
#include <deque>

namespace wallpaper
{
class FrameTimer : NoCopy, NoMove {
    constexpr static usize FRAMETIME_QUEUE_SIZE { 5 };

public:
    FrameTimer(std::function<void()> callback = {});
    ~FrameTimer();

    // call brefore run
    void SetCallback(const std::function<void()>&);

    void Run();
    void Stop();
    void RequestFrame();

    u16    RequiredFps() const;
    bool   Running() const;
    double FrameTime() const;
    double IdeaTime() const;

    void SetRequiredFps(u16);

    // only used with one render
    void FrameBegin();
    void FrameEnd();

private:
    void AddFrametime(std::chrono::microseconds);
    void UpdateFrametime();

    std::function<void()>                 m_callback;
    std::deque<std::chrono::microseconds> m_frametime_queue;

    u16                                    m_req_fps;
    std::atomic<std::chrono::microseconds> m_frametime;
    std::atomic<std::chrono::microseconds> m_ideatime;
    // A frame request remains outstanding from the moment it is posted until the render thread
    // finishes that draw. Keeping a single global gate here lets periodic ticks, startup draws,
    // and explicit producer requests share the same latest-frame scheduling contract instead of
    // building a FIFO queue of stale frames ahead of pointer input.
    std::atomic<bool>                      m_frame_outstanding { false };

    ThreadTimer m_timer;

    // out of time thread
    std::chrono::time_point<std::chrono::steady_clock> m_clock;
};
} // namespace wallpaper

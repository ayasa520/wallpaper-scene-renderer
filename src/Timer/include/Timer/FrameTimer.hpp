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
    // Returns false when a draw is already outstanding and the request was therefore dropped.
    bool RequestFrame();

    // Externally driven timers never start the periodic thread and never re-request a draw
    // after a slow frame: the embedding decides when every draw happens. Used by lockstep
    // golden-frame capture so the number of draws between two captured frames is exact.
    void SetExternallyDriven(bool);
    bool ExternallyDriven() const;

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
    std::atomic<bool>                      m_externally_driven { false };

    ThreadTimer m_timer;

    // out of time thread
    std::chrono::time_point<std::chrono::steady_clock> m_clock;
};
} // namespace wallpaper

#include "FrameTimer.hpp"
#include "Utils//Logging.h"

#include <numeric>

using namespace wallpaper;
using micros = std::chrono::microseconds;
using namespace std::chrono;

FrameTimer::FrameTimer(std::function<void()> cb)
    : m_callback(cb), m_timer([this]() { RequestFrame(); }) {
    SetRequiredFps(15);
}

FrameTimer::~FrameTimer() {};

u16 FrameTimer::RequiredFps() const { return m_req_fps; }

double FrameTimer::FrameTime() const {
    return duration_cast<duration<double>>(m_frametime.load()).count();
}

double FrameTimer::IdeaTime() const {
    auto frametime = m_frametime.load();
    auto ideatime  = m_ideatime.load();
    auto time      = frametime > ideatime ? frametime : ideatime;
    return duration_cast<duration<double>>(time).count();
}

void FrameTimer::UpdateFrametime() {
    m_frametime.store(std::accumulate(m_frametime_queue.begin(),
                                      m_frametime_queue.end(),
                                      duration_cast<microseconds>(0s)) /
                      m_frametime_queue.size());
}

void FrameTimer::SetRequiredFps(u16 value) {
    m_req_fps = value;
    // Preserve the fractional part of rates such as 60 Hz. The old millisecond division produced
    // a 16 ms interval (62.5 Hz), which caused uneven request phases against the producer's exact
    // 16.667 ms frame clock and increased both missed swaps and visible pointer jitter.
    const auto ideatime = microseconds(
        (microseconds::period::den + static_cast<uint64_t>(m_req_fps) - 1u) /
        static_cast<uint64_t>(m_req_fps));
    m_ideatime = ideatime;
    m_timer.SetInterval(ideatime);
    for (usize i = 0; i < FrameTimer::FRAMETIME_QUEUE_SIZE; i++) {
        AddFrametime(ideatime);
    }
    UpdateFrametime();
}

void FrameTimer::AddFrametime(micros t) {
    m_frametime_queue.push_back(t);
    while (m_frametime_queue.size() > FrameTimer::FRAMETIME_QUEUE_SIZE) {
        m_frametime_queue.pop_front();
    }
}

void FrameTimer::FrameBegin() { m_clock = steady_clock::now(); }
void FrameTimer::FrameEnd() {
    auto now = steady_clock::now();
    const auto elapsed = duration_cast<microseconds>(now - m_clock);
    AddFrametime(elapsed);
    UpdateFrametime();

    m_frame_outstanding.store(false, std::memory_order_release);
    if (Running() && elapsed >= m_ideatime.load()) {
        // When rendering itself is slower than the target interval, waiting for another timer tick
        // inserts avoidable idle time. Queue the next frame immediately, but only after releasing
        // the outstanding gate so pointer messages already waiting on the render looper stay ahead
        // of the new draw. RequestFrame's compare-exchange also resolves a concurrent timer tick
        // without ever allowing two queued/in-flight draws.
        RequestFrame();
    }
}

void FrameTimer::RequestFrame() {
    if (! m_callback) return;

    bool expected = false;
    if (! m_frame_outstanding.compare_exchange_strong(expected,
                                                       true,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
        return;
    }
    m_callback();
}

void FrameTimer::SetCallback(const std::function<void()>& cb) {
    if (! Running()) m_callback = cb;
}
void FrameTimer::Run() { m_timer.Start(); }
void FrameTimer::Stop() { m_timer.Stop(); }
bool FrameTimer::Running() const { return m_timer.Running(); }

#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unistd.h>
#include <utility>
#include "Core/NoCopyMove.hpp"
#include "FrameSequenceTrace.hpp"

namespace wallpaper
{

template<typename T>
class TripleSwapchain : NoCopy, NoMove {
public:
    virtual ~TripleSwapchain() { clearAcquireFds(); }

    T* eatFrame() {
        std::lock_guard<std::mutex> lk(stateMutex());
        if (! dirty().exchange(false)) return nullptr;
        presented() = ready().exchange(presented());
        m_frame_trace.presented();
        return presented();
    }
    void renderFrame() {
        std::lock_guard<std::mutex> lk(stateMutex());
        inprogress()          = ready().exchange(inprogress());
        m_frame_trace.published();
        dirty().exchange(true);
    }
    T* getInprogress() { return inprogress(); }

    // 1-based publish number of the frame most recently returned by eatFrame(); 0 before the
    // first eat or when diagnostics are disabled. Consecutive recorded values differ by
    // exactly one when no frame was skipped.
    uint64_t presentedSequence() {
        return m_frame_trace.presentedSequence(stateMutex());
    }
    // Total number of renderFrame() calls so far.
    uint64_t publishedCount() {
        return m_frame_trace.publishedCount(stateMutex());
    }

    /*
     * Explicit-sync stash for the exported offscreen ring: the render thread
     * stores each slot's acquire sync-file right before renderFrame() makes
     * the slot visible, and the IPC relay takes it together with eatFrame().
     * A pipelined producer publishes frames whose GPU work is still running,
     * so the consumer must wait this fence instead of a pre-signaled one.
     */
    void storeAcquireFd(std::size_t slot, int fd) {
        if (slot >= m_acquire_fds.size()) {
            if (fd >= 0) ::close(fd);
            return;
        }
        const int previous = m_acquire_fds[slot].exchange(fd);
        if (previous >= 0) ::close(previous);
    }
    int takeAcquireFd(std::size_t slot) {
        if (slot >= m_acquire_fds.size()) return -1;
        return m_acquire_fds[slot].exchange(-1);
    }
    void clearAcquireFds() {
        for (auto& fd : m_acquire_fds) {
            const int previous = fd.exchange(-1);
            if (previous >= 0) ::close(previous);
        }
    }

    virtual uint width() const  = 0;
    virtual uint height() const = 0;

protected:
    TripleSwapchain() = default;

    virtual std::atomic<T*>& presented()  = 0;
    virtual std::atomic<T*>& ready()      = 0;
    virtual std::atomic<T*>& inprogress() = 0;

    void resetDirty() { m_dirty.store(false); }
    std::mutex& stateMutex() { return m_state_mutex; }

private:
    std::atomic<bool>& dirty() { return m_dirty; };
    std::atomic<bool>  m_dirty { false };
    std::mutex         m_state_mutex;
    std::array<std::atomic<int>, 3> m_acquire_fds { -1, -1, -1 };
    [[no_unique_address]] diagnostics::FrameSequenceTrace m_frame_trace;
};

} // namespace wallpaper

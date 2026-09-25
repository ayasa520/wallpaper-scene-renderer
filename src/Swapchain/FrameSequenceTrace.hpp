#pragma once

#include "Utils/Diagnostics.h"

#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>

namespace wallpaper::diagnostics
{

// These numbers observe which published frame occupies each ring role. The caller holds
// the ring's state mutex while changing roles; readers take that same mutex. They are capture
// metadata, independent of the acquire fences and transport sequences used for synchronization.
class FrameSequenceRecorder {
public:
    void presented() { std::swap(m_presented, m_ready); }
    void published() {
        m_inprogress = ++m_total;
        std::swap(m_ready, m_inprogress);
    }
    uint64_t presentedSequence(std::mutex& mutex) const {
        std::lock_guard lock(mutex);
        return m_presented;
    }
    uint64_t publishedCount(std::mutex& mutex) const {
        std::lock_guard lock(mutex);
        return m_total;
    }

private:
    uint64_t m_total { 0 };
    uint64_t m_ready { 0 };
    uint64_t m_inprogress { 0 };
    uint64_t m_presented { 0 };
};

class DisabledFrameSequenceTrace {
public:
    static constexpr void presented() {}
    static constexpr void published() {}
    static constexpr uint64_t presentedSequence(std::mutex&) { return 0; }
    static constexpr uint64_t publishedCount(std::mutex&) { return 0; }
};

using FrameSequenceTrace = std::conditional_t<Enabled,
                                              FrameSequenceRecorder, DisabledFrameSequenceTrace>;

} // namespace wallpaper::diagnostics

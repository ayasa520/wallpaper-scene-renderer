#pragma once
#include <atomic>
#include <mutex>
#include "Core/NoCopyMove.hpp"

namespace wallpaper
{

template<typename T>
class TripleSwapchain : NoCopy, NoMove {
public:
    virtual ~TripleSwapchain() = default;

    T* eatFrame() {
        std::lock_guard<std::mutex> lk(stateMutex());
        if (! dirty().exchange(false)) return nullptr;
        presented() = ready().exchange(presented());
        return presented();
    }
    void renderFrame() {
        std::lock_guard<std::mutex> lk(stateMutex());
        inprogress() = ready().exchange(inprogress());
        dirty().exchange(true);
    }
    T* getInprogress() { return inprogress(); }

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
};

} // namespace wallpaper

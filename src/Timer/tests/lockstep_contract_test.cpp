// Contracts the deterministic (lockstep) golden-frame capture relies on. These are checked
// without a GPU: the frame timer's request gate, the exported swapchain's publish numbering,
// and the determinism helpers that pin the frame step and random seeds.
//
// Each check states the harness assumption it protects, so a failure here explains which part
// of the capture protocol would silently break before any golden image differs.

#undef NDEBUG

#include "Core/Determinism.hpp"
#include "Swapchain/TripleSwapchain.hpp"
#include "Timer/FrameTimer.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace wallpaper;

namespace
{

// Harness assumption: with an externally driven timer, the number of callbacks equals the
// number of accepted RequestFrame() calls, and a slow frame never queues an extra draw.
void CheckExternallyDrivenTimer() {
    std::atomic<int> callbacks { 0 };
    FrameTimer       timer([&]() { ++callbacks; });
    timer.SetRequiredFps(1000);
    timer.SetExternallyDriven(true);
    timer.Run();
    assert(! timer.Running() && "externally driven timer must not start its periodic thread");

    assert(timer.RequestFrame() && "first request is accepted");
    assert(callbacks.load() == 1);
    assert(! timer.RequestFrame() && "a request while a draw is outstanding is dropped");
    assert(callbacks.load() == 1);

    // A draw that takes longer than the 1 ms ideal interval would re-request itself when the
    // periodic timer runs; externally driven timers leave that decision to the embedding.
    timer.FrameBegin();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.FrameEnd();
    assert(callbacks.load() == 1 && "slow frame must not re-request when externally driven");
    assert(timer.RequestFrame() && "gate is released after FrameEnd");
    assert(callbacks.load() == 2);
    timer.FrameBegin();
    timer.FrameEnd();
    std::puts("ok: externally driven frame timer");
}

// Sanity check of the production contract the lockstep switch replaces: the periodic timer
// does post draws on its own.
void CheckPeriodicTimerStillRuns() {
    std::atomic<int> callbacks { 0 };
    FrameTimer       timer([&]() { ++callbacks; });
    timer.SetRequiredFps(200);
    timer.Run();
    assert(timer.Running());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (callbacks.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    timer.Stop();
    assert(callbacks.load() >= 1 && "periodic timer posts draws without requests");
    std::puts("ok: periodic frame timer");
}

struct Slot {
    int id { 0 };
};

class TestSwapchain : public TripleSwapchain<Slot> {
public:
    TestSwapchain() {
        m_presented.store(&m_slots[0]);
        m_ready.store(&m_slots[1]);
        m_inprogress.store(&m_slots[2]);
    }
    uint width() const override { return 1; }
    uint height() const override { return 1; }

protected:
    std::atomic<Slot*>& presented() override { return m_presented; }
    std::atomic<Slot*>& ready() override { return m_ready; }
    std::atomic<Slot*>& inprogress() override { return m_inprogress; }

private:
    Slot               m_slots[3] { { 0 }, { 1 }, { 2 } };
    std::atomic<Slot*> m_presented { nullptr };
    std::atomic<Slot*> m_ready { nullptr };
    std::atomic<Slot*> m_inprogress { nullptr };
};

// Harness assumption: presentedSequence() of consecutive eats differs by exactly one when no
// draw was skipped, and exposes the gap when the latest-wins ring overwrote an unread frame.
void CheckSwapchainPublishNumbering() {
    TestSwapchain swapchain;
    assert(swapchain.eatFrame() == nullptr && "nothing published yet");
    assert(swapchain.presentedSequence() == 0);

    swapchain.renderFrame();
    assert(swapchain.eatFrame() != nullptr);
    assert(swapchain.presentedSequence() == 1);
    assert(swapchain.eatFrame() == nullptr && "a frame is eaten once");

    swapchain.renderFrame();
    swapchain.renderFrame(); // overwrites the unread frame 2
    assert(swapchain.publishedCount() == 3);
    assert(swapchain.eatFrame() != nullptr);
    assert(swapchain.presentedSequence() == 3 && "latest-wins skip is visible as a gap");

    swapchain.renderFrame();
    assert(swapchain.eatFrame() != nullptr);
    assert(swapchain.presentedSequence() == 4);
    std::puts("ok: swapchain publish numbering");
}

// Harness assumption: the environment pins the step, the seed replays the same random sequence
// on the calling thread, and the production path is untouched when nothing is set.
void CheckDeterminismHelpers() {
    setenv("WESCENE_LOCKSTEP", "1", 1);
    setenv("WESCENE_FIXED_DT", "0.02", 1);
    setenv("WESCENE_RANDOM_SEED", "12345", 1);
    setenv("WESCENE_FIXED_EPOCH", "1700000000", 1);

    assert(determinism::LockstepEnabled());
    assert(determinism::FixedFrameTime(1.0 / 30.0) == 0.02);
    assert(determinism::RandomSeed().has_value() && *determinism::RandomSeed() == 12345);
    assert(determinism::FixedEpoch().has_value() && *determinism::FixedEpoch() == 1700000000.0);

    determinism::SeedThreadRandom();
    const auto a0 = Random::get(0.0f, 1.0f);
    const auto a1 = Random::get(0.0f, 1.0f);
    const auto a2 = Random::get<uint32_t>(0, 1000);
    determinism::SeedThreadRandom();
    assert(Random::get(0.0f, 1.0f) == a0);
    assert(Random::get(0.0f, 1.0f) == a1);
    assert(Random::get<uint32_t>(0, 1000) == a2 && "reseeding replays the same sequence");

    // Another thread has its own engine; seeding there must produce the same sequence too.
    float b0 = -1.0f;
    std::thread([&]() {
        determinism::SeedThreadRandom();
        b0 = Random::get(0.0f, 1.0f);
    }).join();
    assert(b0 == a0 && "per-thread seeding gives every thread the same start");
    std::puts("ok: determinism helpers");
}

} // namespace

int main() {
    CheckExternallyDrivenTimer();
    CheckPeriodicTimerStillRuns();
    CheckSwapchainPublishNumbering();
    CheckDeterminismHelpers();
    std::puts("lockstep_contract_test: all checks passed");
    return 0;
}

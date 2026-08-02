#include "ParticleParallelExecutor.h"

#include <algorithm>

using namespace wallpaper;

ParticleParallelExecutor& ParticleParallelExecutor::Instance() {
    static ParticleParallelExecutor executor;
    return executor;
}

ParticleParallelExecutor::ParticleParallelExecutor() {
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    const size_t worker_count = hardware_threads > 1 ? static_cast<size_t>(hardware_threads - 1) : 0;
    m_workers.reserve(worker_count);
    for (size_t worker_index = 0; worker_index < worker_count; worker_index++) {
        m_workers.emplace_back([this, worker_index] { WorkerMain(worker_index); });
    }
}

ParticleParallelExecutor::~ParticleParallelExecutor() {
    {
        std::lock_guard<std::mutex> lock { m_state_mutex };
        m_stopping = true;
        m_generation++;
    }
    m_work_available.notify_all();
    for (auto& worker : m_workers) {
        if (worker.joinable()) worker.join();
    }
}

void ParticleParallelExecutor::ParallelFor(
    size_t item_count, const std::function<void(size_t begin, size_t end)>& operation) {
    if (item_count == 0) return;
    if (m_workers.empty() || item_count == 1) {
        operation(0, item_count);
        return;
    }

    std::unique_lock<std::mutex> dispatch_lock { m_dispatch_mutex };
    const size_t participant_count = std::min(item_count, m_workers.size() + 1);
    const size_t active_worker_count = participant_count - 1;

    {
        std::lock_guard<std::mutex> state_lock { m_state_mutex };
        m_item_count           = item_count;
        m_active_worker_count  = active_worker_count;
        m_pending_worker_count = active_worker_count;
        m_operation            = operation;
        m_generation++;
    }
    m_work_available.notify_all();

    // The render thread owns one chunk instead of sleeping immediately. Together with the worker
    // count chosen above this uses at most hardware_concurrency() participants and leaves no extra
    // oversubscribed coordinator thread during the expensive particle operator.
    const size_t main_begin = (active_worker_count * item_count) / participant_count;
    operation(main_begin, item_count);

    {
        std::unique_lock<std::mutex> state_lock { m_state_mutex };
        m_work_finished.wait(state_lock, [this] { return m_pending_worker_count == 0; });
        m_operation = {};
    }
}

void ParticleParallelExecutor::WorkerMain(size_t worker_index) {
    uint64_t observed_generation { 0 };
    while (true) {
        std::function<void(size_t, size_t)> operation;
        size_t                             item_count { 0 };
        size_t                             active_worker_count { 0 };
        {
            std::unique_lock<std::mutex> lock { m_state_mutex };
            m_work_available.wait(lock, [this, observed_generation] {
                return m_stopping || m_generation != observed_generation;
            });
            if (m_stopping) return;

            observed_generation = m_generation;
            active_worker_count = m_active_worker_count;
            if (worker_index >= active_worker_count) continue;
            operation  = m_operation;
            item_count = m_item_count;
        }

        const size_t participant_count = active_worker_count + 1;
        const size_t begin = (worker_index * item_count) / participant_count;
        const size_t end   = ((worker_index + 1) * item_count) / participant_count;
        operation(begin, end);

        {
            std::lock_guard<std::mutex> lock { m_state_mutex };
            if (--m_pending_worker_count == 0) m_work_finished.notify_one();
        }
    }
}

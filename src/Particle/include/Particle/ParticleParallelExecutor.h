#pragma once

#include "Core/NoCopyMove.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace wallpaper
{

class ParticleParallelExecutor : NoCopy, NoMove {
public:
    static ParticleParallelExecutor& Instance();

    void ParallelFor(size_t item_count,
                     const std::function<void(size_t begin, size_t end)>& operation);

private:
    ParticleParallelExecutor();
    ~ParticleParallelExecutor();

    void WorkerMain(size_t worker_index);

    // Multiple scene render threads may reach the shared executor. A dispatch remains synchronous,
    // so serializing callers here keeps the single job-generation state compact while each job
    // still uses every available CPU participant internally.
    std::mutex m_dispatch_mutex;

    std::mutex              m_state_mutex;
    std::condition_variable m_work_available;
    std::condition_variable m_work_finished;
    bool                    m_stopping { false };
    uint64_t                m_generation { 0 };
    size_t                  m_active_worker_count { 0 };
    size_t                  m_pending_worker_count { 0 };
    size_t                  m_item_count { 0 };
    std::function<void(size_t, size_t)> m_operation;
    std::vector<std::thread>            m_workers;
};

} // namespace wallpaper

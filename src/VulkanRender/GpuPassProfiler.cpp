#include "GpuPassProfiler.hpp"

#include "Utils/Logging.h"
#include "Vulkan/Device.hpp"
#include "VulkanPass.hpp"

#include <algorithm>
#include <utility>

using namespace wallpaper::vulkan;

bool GpuPassRecorder::active(const Device& device) {
    if (!m_checked) {
        m_checked = true;
        m_enabled = wallpaper::diagnostics::Options().gpu_profile;
        if (m_enabled) {
            m_period_ns = device.limits().timestampPeriod;
            m_capacity = 4096;
            const VkQueryPoolCreateInfo pool_ci {
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = m_capacity,
                .pipelineStatistics = 0,
            };
            if (device.handle().CreateQueryPool(pool_ci, m_pool) != VK_SUCCESS ||
                m_period_ns <= 0.0) {
                LOG_INFO("GpuPassProfiler: query pool unavailable, profiling disabled");
                m_enabled = false;
            } else {
                LOG_INFO("GpuPassProfiler: enabled timestamp-period=%.3fns capacity=%u",
                         m_period_ns, m_capacity);
            }
        }
    }
    return m_enabled;
}

void GpuPassRecorder::collect(const Device& device) {
    if (!active(device) || !m_pending) return;
    m_pending = false;
    if (m_used < 2) return;

    std::vector<uint64_t> ticks(m_used, 0);
    // The caller has drained the previous offscreen frame fence. These reads therefore
    // consume completed queries and cannot delay the current frame on unfinished GPU work.
    const VkResult result = device.handle().GetQueryPoolResults(
        *m_pool, 0, m_used, ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) return;

    for (uint32_t i = 1; i < m_used; i++) {
        const double delta_ms = static_cast<double>(ticks[i] - ticks[i - 1]) * m_period_ns / 1e6;
        m_accum_ms[m_pending_names[i - 1]] += delta_ms;
    }
    m_total_ms += static_cast<double>(ticks[m_used - 1] - ticks[0]) * m_period_ns / 1e6;
    m_frames++;
    if (m_frames >= 240) report();
}

void GpuPassRecorder::beginFrame(const Device& device, vvk::CommandBuffer& command) {
    if (!active(device)) return;
    m_used = 0;
    m_pending_names.clear();
    command.ResetQueryPool(*m_pool, 0, m_capacity);
    command.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *m_pool, m_used++);
}

void GpuPassRecorder::stamp(vvk::CommandBuffer& command, std::string name) {
    command.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *m_pool, m_used++);
    m_pending_names.emplace_back(std::move(name));
}

void GpuPassRecorder::afterUploads(vvk::CommandBuffer& command) {
    if (m_enabled && m_used < m_capacity) stamp(command, "__uploads");
}

void GpuPassRecorder::afterPass(vvk::CommandBuffer& command, const VulkanPass& pass) {
    // Pass identities may allocate strings. Obtain them only inside the enabled recorder,
    // after checking query capacity, so an inactive recorder never pays that cost either.
    if (m_enabled && m_used < m_capacity) stamp(command, pass.profileName());
}

void GpuPassRecorder::submitted() {
    if (m_enabled && m_used >= 2) m_pending = true;
}

void GpuPassRecorder::reset() {
    // Query pools are device-owned and must be released before the device is destroyed.
    m_pool.reset();
    m_pending = false;
}

void GpuPassRecorder::report() {
    if (m_frames == 0) return;
    std::vector<std::pair<std::string, double>> entries(m_accum_ms.begin(), m_accum_ms.end());
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    LOG_INFO("GpuPassProfiler: frames=%u gpu-total=%.2fms/frame passes=%zu",
             m_frames, m_total_ms / m_frames, entries.size());
    const std::size_t top_count = std::min<std::size_t>(entries.size(), 40);
    for (std::size_t i = 0; i < top_count; i++) {
        LOG_INFO("GpuPassProfiler: %6.3fms/frame %s",
                 entries[i].second / m_frames, entries[i].first.c_str());
    }
    m_accum_ms.clear();
    m_total_ms = 0.0;
    m_frames = 0;
}

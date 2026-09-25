#pragma once

#include "Utils/Diagnostics.h"
#include "vvk/vulkan_wrapper.hpp"

#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace wallpaper::vulkan
{

class Device;
class VulkanPass;

// This recorder owns all query state and pass-name formatting. The render loop supplies
// existing objects to a few lifecycle hooks; it does not prepare diagnostic arguments or
// branch on profiling state. CMake links the implementation only in diagnostic builds.
class GpuPassRecorder {
public:
    void collect(const Device&);
    void beginFrame(const Device&, vvk::CommandBuffer&);
    void afterUploads(vvk::CommandBuffer&);
    void afterPass(vvk::CommandBuffer&, const VulkanPass&);
    void submitted();
    void reset();

private:
    bool active(const Device&);
    void stamp(vvk::CommandBuffer&, std::string name);
    void report();

    bool                                    m_checked { false };
    bool                                    m_enabled { false };
    vvk::QueryPool                          m_pool;
    uint32_t                                m_capacity { 0 };
    uint32_t                                m_used { 0 };
    bool                                    m_pending { false };
    double                                  m_period_ns { 0.0 };
    std::vector<std::string>                m_pending_names;
    std::unordered_map<std::string, double> m_accum_ms;
    double                                  m_total_ms { 0.0 };
    uint32_t                                m_frames { 0 };
};

class DisabledGpuPassProfiler {
public:
    static constexpr void collect(const Device&) {}
    static constexpr void beginFrame(const Device&, vvk::CommandBuffer&) {}
    static constexpr void afterUploads(vvk::CommandBuffer&) {}
    static constexpr void afterPass(vvk::CommandBuffer&, const VulkanPass&) {}
    static constexpr void submitted() {}
    static constexpr void reset() {}
};

using GpuPassProfiler = std::conditional_t<wallpaper::diagnostics::Enabled,
                                         GpuPassRecorder, DisabledGpuPassProfiler>;

} // namespace wallpaper::vulkan

#pragma once

#include "Swapchain/ExSwapchain.hpp"
#include "Device.hpp"
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace wallpaper
{
namespace vulkan
{

struct VulkanExHandle : NoCopy {
    ExHandle          handle;
    ExImageParameters image;

    VulkanExHandle()  = default;
    ~VulkanExHandle() = default;
    VulkanExHandle(VulkanExHandle&& o) noexcept: handle(o.handle), image(std::move(o.image)) {}
    VulkanExHandle& operator=(VulkanExHandle&& o) noexcept {
        handle = o.handle;
        image  = std::move(o.image);
        return *this;
    }
};
struct VulkanExHandleSemaphore {
    ExHandle    handle;
    VkSemaphore semaphore;
};

class VulkanExSwapchainRenderLock {
public:
    VulkanExSwapchainRenderLock() = default;
    virtual ~VulkanExSwapchainRenderLock() = default;
};

class VulkanExSwapchain : public ExSwapchain {
    using atomic_ = std::atomic<ExHandle*>;

public:
    VulkanExSwapchain(std::array<VulkanExHandle, 3> handles, VkExtent2D ext)
        : m_active(std::make_unique<Generation>(std::move(handles), ext)) {
        activateGeneration(*m_active);
    }
    virtual ~VulkanExSwapchain() = default;

    uint width() const override {
        std::lock_guard<std::mutex> lk(m_generation_mutex);
        return m_active ? m_active->extent.width : 0;
    }
    uint height() const override {
        std::lock_guard<std::mutex> lk(m_generation_mutex);
        return m_active ? m_active->extent.height : 0;
    }

    std::vector<ExHandle> handlesSnapshot() const {
        std::lock_guard<std::mutex> lk(m_generation_mutex);
        if (!m_active) return {};

        std::vector<ExHandle> snapshot;
        snapshot.reserve(m_active->handles.size());
        for (const auto& h : m_active->handles) {
            snapshot.push_back(h.handle);
        }
        return snapshot;
    }

    std::unique_ptr<VulkanExSwapchainRenderLock> acquireRenderLock() {
        return std::make_unique<ScopedRenderLock>(m_generation_mutex);
    }

    ExImageParameters& GetInprogressImage() {
        auto* active = m_active.get();
        return active->handles.at((usize)(*inprogress()).id()).image;
    }

    bool Reconfigure(const Device& device,
                     uint          w,
                     uint          h,
                     VkImageTiling tiling,
                     ExternalFrameExportMode export_mode,
                     uint32_t export_drm_fourcc = 0,
                     std::span<const uint64_t> export_drm_modifiers = {},
                     ExternalFrameMemoryPreference memory_preference =
                         ExternalFrameMemoryPreference::Default) {
        auto next_handles =
            CreateExHandles(device, w, h, tiling, export_mode, export_drm_fourcc,
                            export_drm_modifiers, memory_preference);
        if (!next_handles.has_value()) return false;

        std::lock_guard<std::mutex> generation_lk(m_generation_mutex);
        std::lock_guard<std::mutex> state_lk(stateMutex());
        /*
         * The display protocol may still hold fds from a blacklisted generation
         * while the renderer is negotiating a new modifier.  Keep old Vulkan
         * images alive for the lifetime of this swapchain instead of releasing
         * them immediately after publication switches to the next generation.
         */
        if (m_active) m_retired.push_back(std::move(m_active));
        m_active = std::make_unique<Generation>(std::move(next_handles.value()),
                                                VkExtent2D { w, h });
        activateGeneration(*m_active);
        resetDirty();
        return true;
    }

    constexpr VkFormat format() const { return VK_FORMAT_R8G8B8A8_UNORM; };

protected:
    atomic_& presented() override { return m_presented; };
    atomic_& ready() override { return m_ready; };
    atomic_& inprogress() override { return m_inprogress; };

private:
    class ScopedRenderLock : public VulkanExSwapchainRenderLock {
    public:
        explicit ScopedRenderLock(std::mutex& mutex): m_lock(mutex) {}

    private:
        std::unique_lock<std::mutex> m_lock;
    };

    struct Generation {
        Generation(std::array<VulkanExHandle, 3> next_handles, VkExtent2D next_extent)
            : handles(std::move(next_handles)), extent(next_extent) {
            int index = 0;
            for (auto& h : handles) {
                auto& handle  = h.handle;
                handle        = ExHandle(index++);
                handle.handle_type = h.image.handle_type;
                handle.width  = (i32)h.image.extent.width;
                handle.height = (i32)h.image.extent.height;
                handle.fd     = h.image.fd;
                handle.size   = h.image.mem_reqs.size;
                handle.drm_fourcc = h.image.drm_fourcc;
                handle.drm_modifier = h.image.drm_modifier;
                handle.n_planes = h.image.n_planes;
                handle.premultiplied = h.image.premultiplied;
                handle.planes = h.image.planes;
            }
        }

        std::array<VulkanExHandle, 3> handles;
        VkExtent2D                    extent;
    };

    static std::optional<std::array<VulkanExHandle, 3>>
    CreateExHandles(const Device& device,
                    uint          w,
                    uint          h,
                    VkImageTiling tiling,
                    ExternalFrameExportMode export_mode,
                    uint32_t export_drm_fourcc,
                    std::span<const uint64_t> export_drm_modifiers,
                    ExternalFrameMemoryPreference memory_preference) {
        std::array<VulkanExHandle, 3> handles;
        for (auto& handle : handles) {
            if (auto rv = device.tex_cache().CreateExTex(
                    w,
                    h,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    tiling,
                    export_mode,
                    export_drm_fourcc,
                    export_drm_modifiers,
                    memory_preference);
                rv.has_value())
                handle.image = std::move(rv.value());
            else
                return std::nullopt;
        }
        return handles;
    }

    void activateGeneration(Generation& generation) {
        m_presented  = &generation.handles[0].handle;
        m_ready      = &generation.handles[1].handle;
        m_inprogress = &generation.handles[2].handle;
    }

    mutable std::mutex                m_generation_mutex;
    std::unique_ptr<Generation>       m_active;
    std::vector<std::unique_ptr<Generation>> m_retired;
    atomic_                       m_presented { nullptr };
    atomic_                       m_ready { nullptr };
    atomic_                       m_inprogress { nullptr };
};

inline std::unique_ptr<VulkanExSwapchain> CreateExSwapchain(const Device& device,
                                                            uint          w,
                                                            uint          h,
                                                            VkImageTiling tiling,
                                                            ExternalFrameExportMode export_mode,
                                                            uint32_t export_drm_fourcc = 0,
                                                            std::span<const uint64_t> export_drm_modifiers = {},
                                                            ExternalFrameMemoryPreference memory_preference =
                                                                ExternalFrameMemoryPreference::Default) {
    std::array<VulkanExHandle, 3> handles;
    for (auto& handle : handles) {
        if (auto rv = device.tex_cache().CreateExTex(
                w,
                h,
                VK_FORMAT_R8G8B8A8_UNORM,
                tiling,
                export_mode,
                export_drm_fourcc,
                export_drm_modifiers,
                memory_preference);
            rv.has_value())
            handle.image = std::move(rv.value());
        else
            return nullptr;
    }
    /*
    VulkanExHandleSemaphore handle_sem;
    {
        vk::SemaphoreCreateInfo info;
        vk::ExportSemaphoreCreateInfo esci { vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
        info.setPNext(&esci);
        VK_CHECK_RESULT_ACT(return nullptr, device.handle().createSemaphore(&info, nullptr,
    &handle_sem.semaphore)); vk::SemaphoreGetFdInfoKHR fd_info; fd_info.semaphore =
    handle_sem.semaphore; fd_info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
        VK_CHECK_RESULT_ACT(return nullptr, device.handle().getSemaphoreFdKHR(&fd_info,
    &handle_sem.handle.fd));
    }*/
    return std::make_unique<VulkanExSwapchain>(std::move(handles), VkExtent2D { w, h });
}

} // namespace vulkan
} // namespace wallpaper

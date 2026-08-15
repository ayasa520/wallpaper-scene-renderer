#include "VulkanRender.hpp"
#include <cstdio>
#include <cstdlib>
#include <typeinfo>

#include "Utils/Logging.h"
#include "RenderGraph/RenderGraph.hpp"
#include "Scene/Scene.h"
#include "Interface/IImageParser.h"
#include "Interface/IShaderValueUpdater.h"

#include "Utils/Algorism.h"

#include "Vulkan/Device.hpp"
#include "Vulkan/TextureCache.hpp"
#include "Vulkan/Swapchain.hpp"
#include "Vulkan/VideoTextureCache.hpp"
#include "Vulkan/VulkanExSwapchain.hpp"

#include "VulkanPass.hpp"
#include "PrePass.hpp"
#include "FinPass.hpp"
#include "CopyPass.hpp"
#include "Resource.hpp"

#include "Core/ArrayHelper.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if ENABLE_RENDERDOC_API
#    include "RenderDoc.h"
#endif

using namespace wallpaper::vulkan;

constexpr uint64_t vk_wait_time { 10u * 1000u * 1000000u };
constexpr uint32_t vk_command_num { 1 };
constexpr std::size_t kDeferredPrepareMaxPassesPerFrame { 96 };
constexpr double      kDeferredPrepareFrameBudgetMs { 2.0 };

constexpr std::array base_inst_exts {
    Extension { false, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME },
};
constexpr std::array base_device_exts {
    Extension { false, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME },
    Extension { false, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME },
    Extension { false, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME },
    Extension { true, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME }
};

namespace
{

std::mutex& VulkanInitMutex() {
    static std::mutex mutex;
    return mutex;
}

std::string MakeResidencyInstanceKey(
    const VulkanPass& pass, std::unordered_map<std::string, std::size_t>& occurrence_counts) {
    const auto base_key = pass.residencyKey();
    if (base_key.empty()) return {};

    const auto instance = occurrence_counts[base_key]++;
    return base_key + "|instance=" + std::to_string(instance);
}

void DestroyPassOnce(VulkanPass* pass, const Device& device, RenderingResources& resources,
                     std::unordered_set<VulkanPass*>& destroyed) {
    if (pass == nullptr || !destroyed.insert(pass).second) return;
    pass->destory(device, resources);
}

const char* ExternalMemoryPreferenceName(wallpaper::ExternalFrameMemoryPreference preference) {
    switch (preference) {
    case wallpaper::ExternalFrameMemoryPreference::HostVisible: return "host-visible";
    case wallpaper::ExternalFrameMemoryPreference::DeviceLocal: return "device-local";
    case wallpaper::ExternalFrameMemoryPreference::Default:
    default: return "default";
    }
}

constexpr uint32_t kGpuTimestampBegin         = 0;
constexpr uint32_t kGpuTimestampAfterBuffers  = 1;
constexpr uint32_t kGpuTimestampAfterTextures = 2;
constexpr uint32_t kGpuTimestampEnd           = 3;
constexpr uint32_t kGpuTimestampPassBase      = 4;
constexpr uint32_t kGpuPassCapacity           = 256;
constexpr uint32_t kGpuTimestampQueryCount    = kGpuTimestampPassBase + kGpuPassCapacity;
constexpr uint32_t kGpuPipelineStatCount      = 8;
constexpr VkQueryPipelineStatisticFlags kGpuPipelineStatFlags =
    VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
    VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
    VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
    VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
constexpr uint32_t kGpuPassCategoryCount = 8;

struct GpuPassDiagRecord {
    GpuPassCategory category { GpuPassCategory::Other };
    int32_t         layer_id { -1 };
    uint32_t        draws { 0 };
    uint32_t        vertices { 0 };
    uint32_t        indices { 0 };
    const char*     primitive { "none" };
    const char*     blend { "none" };
    std::string     key;
};

template <typename Fn>
double TimeMs(Fn&& fn) {
    const auto started = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
        .count();
}

double TimestampDeltaMs(uint64_t begin, uint64_t end, float period_ns) {
    if (end < begin) return 0.0;
    return static_cast<double>(end - begin) * static_cast<double>(period_ns) / 1.0e6;
}

std::string CsvEscape(std::string_view value) {
    if (value.find_first_of(",\"\n\r") == std::string_view::npos) return std::string(value);
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

std::string GpuPassCsvPath(std::string_view frame_path) {
    const auto slash = frame_path.find_last_of("/\\");
    if (slash == std::string_view::npos) return "gpu-pass.csv";
    return std::string(frame_path.substr(0, slash + 1)) + "gpu-pass.csv";
}

} // namespace

struct VulkanRender::Impl {
    Impl()  = default;
    ~Impl() = default;

    bool init(RenderInitInfo);
    void destroy();

    void drawFrame(Scene&);
    void setPaused(bool paused);
    void setOffscreenFrameReleaseCallback(OffscreenFrameReleaseCallback callback);
    bool reconfigureOffscreenExport(uint32_t width,
                                    uint32_t height,
                                    TexTiling tiling,
                                    ExternalFrameExportMode export_mode,
                                    uint32_t export_drm_fourcc,
                                    const std::vector<uint64_t>& export_drm_modifiers,
                                    ExternalFrameMemoryPreference memory_preference);

    bool CreateRenderingResource(RenderingResources&);
    void DestroyRenderingResource(RenderingResources&);

    void clearLastRenderGraph(bool clear_scene_caches);
    void clearRenderGraphResources();
    void releasePendingSceneResources(Scene&);
    void compileRenderGraph(Scene&, rg::RenderGraph&, bool refresh_resources_only);
    void warmupRenderGraphPipelines(Scene&, rg::RenderGraph&);
    void refreshImportedTextures(Scene&);
    void UpdateCameraFillMode(Scene&, wallpaper::FillMode);

    bool initRes();
    void drawFrameSwapchain();
    void drawFrameOffscreen();
    void processDeferredGraphPreparation(Scene&);
    void setRenderTargetSize(Scene&, rg::RenderGraph&);
    bool isDeviceFaultResult(VkResult) const;
    bool checkVkResult(VkResult, const char* operation);
    void abandonDeviceOwnedResourcesAfterFault();
    void ensureGpuFrameDiagnostics();
    void finishGpuFrameDiagnostics(double cpu_update_ms, double cpu_record_upload_ms,
                                   double cpu_execute_ms, double cpu_fence_ms, double cpu_frame_ms);

    Instance                m_instance;
    std::unique_ptr<Device> m_device;

    std::unique_ptr<PrePass> m_prepass { nullptr };
    std::unique_ptr<FinPass> m_finpass { nullptr };

    std::unique_ptr<FinPass> m_testpass { nullptr };
    ReDrawCB                 m_redraw_cb;

    std::unique_ptr<StagingBuffer> m_vertex_buf { nullptr };
    std::unique_ptr<StagingBuffer> m_dyn_buf { nullptr };

    vvk::CommandBuffers m_cmds;
    vvk::CommandBuffer  m_render_cmd;

    bool m_with_surface { false };
    bool m_inited { false };
    bool m_pass_loaded { false };
    bool m_device_faulted { false };
    bool m_device_fault_log_emitted { false };
    std::deque<std::size_t> m_deferred_prepare_indices;
    std::unordered_set<std::size_t> m_deferred_waiting_indices_logged;

    std::unique_ptr<VulkanExSwapchain> m_ex_swapchain;
    RenderingResources                 m_rendering_resources;
    OffscreenFrameReleaseCallback      m_offscreen_frame_release_cb;

    std::vector<VulkanPass*> m_passes;
    std::vector<std::shared_ptr<rg::Pass>> m_compiled_pass_refs;

    bool            m_gpu_diag_initialized { false };
    bool            m_gpu_diag_enabled { false };
    bool            m_pass_diag_enabled { false };
    bool            m_timestamps_ok { false };
    bool            m_pipeline_stats_ok { false };
    std::ofstream   m_gpu_diag_stream;
    std::ofstream   m_gpu_pass_diag_stream;
    vvk::QueryPool  m_timestamp_pool;
    vvk::QueryPool  m_pipeline_stats_pool;
    float           m_timestamp_period_ns { 1.0f };
    uint64_t        m_gpu_diag_frame { 0 };
    FrameDrawStats  m_frame_draw_stats {};
    StagingBuffer::FrameStats m_dyn_stats_before_upload {};
    std::vector<GpuPassDiagRecord> m_gpu_diag_passes;

};

VulkanRender::VulkanRender(): pImpl(std::make_unique<Impl>()) {}
VulkanRender::~VulkanRender() { pImpl->destroy(); };

bool VulkanRender::inited() const { return pImpl->m_inited; }

bool VulkanRender::init(RenderInitInfo info) { return pImpl->init(info); }
void VulkanRender::destroy() { pImpl->destroy(); }
void VulkanRender::drawFrame(Scene& scene) { pImpl->drawFrame(scene); };
void VulkanRender::setPaused(bool paused) { pImpl->setPaused(paused); };
void VulkanRender::setOffscreenFrameReleaseCallback(OffscreenFrameReleaseCallback callback) {
    pImpl->setOffscreenFrameReleaseCallback(std::move(callback));
};
bool VulkanRender::reconfigureOffscreenExport(
    uint32_t width,
    uint32_t height,
    TexTiling tiling,
    ExternalFrameExportMode export_mode,
    uint32_t export_drm_fourcc,
    const std::vector<uint64_t>& export_drm_modifiers,
    ExternalFrameMemoryPreference memory_preference) {
    return pImpl->reconfigureOffscreenExport(width,
                                             height,
                                             tiling,
                                             export_mode,
                                             export_drm_fourcc,
                                             export_drm_modifiers,
                                             memory_preference);
};
void VulkanRender::clearLastRenderGraph(bool clear_scene_caches) {
    pImpl->clearLastRenderGraph(clear_scene_caches);
};
void VulkanRender::clearRenderGraphResources() { pImpl->clearRenderGraphResources(); };
void VulkanRender::compileRenderGraph(Scene& scene, rg::RenderGraph& rg, bool refresh_resources_only) {
    pImpl->compileRenderGraph(scene, rg, refresh_resources_only);
};
void VulkanRender::warmupRenderGraphPipelines(Scene& scene, rg::RenderGraph& rg) {
    pImpl->warmupRenderGraphPipelines(scene, rg);
};
void VulkanRender::refreshImportedTextures(Scene& scene) {
    pImpl->refreshImportedTextures(scene);
};
void VulkanRender::UpdateCameraFillMode(Scene& scene, wallpaper::FillMode fill) {
    pImpl->UpdateCameraFillMode(scene, fill);
};

wallpaper::ExSwapchain* VulkanRender::exSwapchain() const { return pImpl->m_ex_swapchain.get(); };

bool VulkanRender::Impl::init(RenderInitInfo info) {
    if (m_inited) return true;

    /*
     * Independent-display mode can start several scene renderers from the same
     * process. NVIDIA's Vulkan loader/ICD path has shown crashes when multiple
     * threads create instances and devices at the exact same time, so serialize
     * the one-time Vulkan bootstrap for this backend. The lock is deliberately
     * held only during init(); steady-state rendering remains fully parallel.
     */
    std::lock_guard<std::mutex> vulkan_init_lock(VulkanInitMutex());
    if (m_inited) return true;

    m_redraw_cb = info.redraw_callback;
    VkExtent2D extent { info.width, info.height };
    if (extent.width * extent.height < 500 * 500) {
        LOG_ERROR("too small swapchain image size: %dx%d", extent.width, extent.height);
    } else {
        LOG_INFO("set swapchain image size: %dx%d", extent.width, extent.height);
    }

    std::vector<Extension> inst_exts { base_inst_exts.begin(), base_inst_exts.end() };
    std::vector<Extension> device_exts { base_device_exts.begin(), base_device_exts.end() };

    if (! info.offscreen) {
        std::transform(info.surface_info.instanceExts.begin(),
                       info.surface_info.instanceExts.end(),
                       std::back_inserter(inst_exts),
                       [](const auto& s) {
                           return Extension { true, s.c_str() };
                       });
        device_exts.push_back({ true, VK_KHR_SWAPCHAIN_EXTENSION_NAME });
    }

    std::vector<InstanceLayer> inst_layers;
    // valid layer
    if (info.enable_valid_layer) {
        inst_layers.push_back({ true, VALIDATION_LAYER_NAME });
        LOG_INFO("vulkan valid layer \"%s\" enabled", VALIDATION_LAYER_NAME.data());
    }

    if (! Instance::Create(m_instance, inst_exts, inst_layers)) {
        LOG_ERROR("init vulkan failed");
        return false;
    }
    if (! info.offscreen) {
        VkSurfaceKHR surface;
        VVK_CHECK_ACT(
            {
                LOG_ERROR("create vulkan surface failed");
                return false;
            },
            info.surface_info.createSurfaceOp(*m_instance.inst(), &surface));
        m_instance.setSurface(VkSurfaceKHR(surface));
        m_with_surface = true;
    }
    {
        auto surface   = *m_instance.surface();
        auto check_gpu = [&device_exts, surface](const vvk::PhysicalDevice& gpu) {
            return Device::CheckGPU(gpu, device_exts, surface);
        };
        const auto preference =
            info.device_preference == wallpaper::VulkanDevicePreference::PreferIntegrated
                ? PhysicalDevicePreference::PreferIntegrated
                : (info.device_preference == wallpaper::VulkanDevicePreference::PreferDiscrete
                       ? PhysicalDevicePreference::PreferDiscrete
                       : PhysicalDevicePreference::Default);
        if (! m_instance.ChoosePhysicalDevice(check_gpu, info.uuid, preference)) return false;
    }

    {
        m_device = std::make_unique<Device>();
        const VideoTextureDecoderSettings video_texture_settings {
            .decoder_route =
                info.video_texture_decoder_route == wallpaper::VideoTextureDecoderRoute::Va
                    ? VideoTextureDecoderRoute::Va
                    : VideoTextureDecoderRoute::Nvidia,
            .render_node = info.render_node,
        };
        if (! Device::Create(m_instance, device_exts, extent, *m_device, video_texture_settings)) {
            LOG_ERROR("init vulkan device failed");
            return false;
        }
    }

    if (info.offscreen) {
        if (info.ex_swapchain_factory) {
            const RenderInitInfo::ExSwapchainHandles handles {
                .instance = *m_instance.inst(),
                .physical_device = *m_instance.gpu(),
                .device = *m_device->handle(),
                .graphics_queue = *m_device->graphics_queue().handle,
                .graphics_queue_family = m_device->graphics_queue().family_index,
                .renderer_device = m_device.get(),
            };
            m_ex_swapchain = info.ex_swapchain_factory(handles);
            if (!m_ex_swapchain) {
                LOG_ERROR("external offscreen swapchain factory returned null");
                return false;
            }
        } else {
            m_ex_swapchain = CreateExSwapchain(*m_device,
                                               extent.width,
                                               extent.height,
                                               (info.offscreen_tiling == TexTiling::OPTIMAL
                                                    ? VK_IMAGE_TILING_OPTIMAL
                                                    : VK_IMAGE_TILING_LINEAR),
                                               info.export_mode,
                                               info.export_drm_fourcc,
                                               info.export_drm_modifiers,
                                               info.export_memory_preference);
        }
        m_with_surface = false;
    }

    if (! initRes()) return false;
    ;

    m_inited = true;
    return m_inited;
}

bool VulkanRender::Impl::isDeviceFaultResult(VkResult result) const {
    return result == VK_ERROR_DEVICE_LOST || result == VK_TIMEOUT;
}

bool VulkanRender::Impl::checkVkResult(VkResult result, const char* operation) {
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) return true;

    const char* operation_name = operation ? operation : "unknown operation";
    if (isDeviceFaultResult(result)) {
        m_device_faulted = true;
        m_pass_loaded = false;
        if (!m_device_fault_log_emitted) {
            // Device loss is terminal for this VulkanRender instance.  Continuing to submit frames
            // only repeats VK_ERROR_DEVICE_LOST, and destroying every pipeline after an NVIDIA Xid
            // can enter driver teardown paths that have appeared in the Arsenal crash stacks.
            LOG_ERROR("HanabiScene Vulkan: device became unhealthy during %s (%s); "
                      "suppressing future frame submissions and abandoning deep Vulkan teardown",
                      operation_name,
                      vvk::ToString(result));
            m_device_fault_log_emitted = true;
        }
        return false;
    }

    LOG_ERROR("HanabiScene Vulkan: %s failed with %s", operation_name, vvk::ToString(result));
    return false;
}

bool VulkanRender::Impl::initRes() {
    m_prepass = std::make_unique<PrePass>(PrePass::Desc {});
    m_finpass = std::make_unique<FinPass>(FinPass::Desc {});
    if (m_with_surface) {
        m_finpass->setPresentFormat(m_device->swapchain().format());
        m_finpass->setPresentQueueIndex(m_device->present_queue().family_index);
        m_finpass->setPresentLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    } else {
        m_finpass->setPresentFormat(m_ex_swapchain->format());
        m_finpass->setPresentLayout(VK_IMAGE_LAYOUT_GENERAL);
        m_finpass->setPresentQueueIndex(VK_QUEUE_FAMILY_EXTERNAL);
    }
    /*
    m_testpass = std::make_unique<FinPass>(FinPass::Desc{});
    m_testpass->setPresentFormat(m_ex_swapchain->format());
    m_testpass->setPresentQueueIndex(m_device->graphics_queue().family_index);
    m_testpass->setPresentLayout(vk::ImageLayout::ePresentSrcKHR);
    */

    m_vertex_buf = std::make_unique<StagingBuffer>(*m_device,
                                                   2 * 1024 * 1024,
                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m_dyn_buf    = std::make_unique<StagingBuffer>(*m_device,
                                                2 * 1024 * 1024,
                                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (! m_vertex_buf->allocate()) return false;
    if (! m_dyn_buf->allocate()) return false;
    {
        auto& pool = m_device->cmd_pool();
        VVK_CHECK_BOOL_RE(pool.Allocate(vk_command_num, VK_COMMAND_BUFFER_LEVEL_PRIMARY, m_cmds));
        m_render_cmd = vvk::CommandBuffer(m_cmds[0], m_device->handle().Dispatch());
    }
    if (! CreateRenderingResource(m_rendering_resources)) return false;

#if ENABLE_RENDERDOC_API
    load_renderdoc_api();
#endif
    return true;
}

void VulkanRender::Impl::abandonDeviceOwnedResourcesAfterFault() {
    // Once the GPU has reported timeout/device-lost, Vulkan object destruction is no longer a safe
    // cleanup mechanism on the observed NVIDIA path.  This routine intentionally abandons wrapper
    // ownership so process shutdown or backend replacement does not call back into driver destroy
    // functions with a broken device.  The leaked objects are bounded to this renderer instance and
    // are preferable to a SIGSEGV while switching away from the failed scene.
    m_rendering_resources.sem_swap_wait_image.abandon();
    m_rendering_resources.sem_swap_finish.abandon();
    m_rendering_resources.fence_frame.abandon();
    m_rendering_resources.command.abandon();
    for (auto& [_, image] : m_rendering_resources.model_depth_images) {
        image.sampler.abandon();
        image.view.abandon();
        image.handle.abandon();
    }
    m_rendering_resources.model_depth_images.clear();
    m_rendering_resources.masked_draw_attachments.abandon();
    if (m_rendering_resources.pipeline_cache) {
        m_rendering_resources.pipeline_cache->abandon();
    }
    if (m_device) {
        m_device->tex_cache().CancelDeferredGraphActivation();
    }
    m_rendering_resources.vertex_buf = nullptr;
    m_rendering_resources.dyn_buf = nullptr;
    m_deferred_prepare_indices.clear();
    m_deferred_waiting_indices_logged.clear();

    m_render_cmd.abandon();
    m_cmds.abandon();
    m_timestamp_pool.abandon();
    m_pipeline_stats_pool.abandon();
    m_gpu_diag_stream.close();
    m_gpu_pass_diag_stream.close();
    m_rendering_resources.frame_draw_stats = nullptr;
    m_compiled_pass_refs.clear();
    m_passes.clear();
    (void)m_prepass.release();
    (void)m_finpass.release();
    (void)m_testpass.release();
    (void)m_vertex_buf.release();
    (void)m_dyn_buf.release();
    (void)m_ex_swapchain.release();
    (void)m_device.release();
    m_instance.Abandon();
    m_inited = false;
    m_pass_loaded = false;
}

void VulkanRender::Impl::destroy() {
    if (! m_inited && !m_device_faulted) return;
    if (m_device_faulted) {
        abandonDeviceOwnedResourcesAfterFault();
        return;
    }
    if (m_device && m_device->handle()) {
        if (!checkVkResult(m_device->handle().WaitIdle(), "device wait idle before destroy")) {
            abandonDeviceOwnedResourcesAfterFault();
            return;
        }

        // res
        for (auto& p : m_passes) {
            p->destory(*m_device, m_rendering_resources);
        }
        m_compiled_pass_refs.clear();
        m_passes.clear();
        m_deferred_prepare_indices.clear();
        m_deferred_waiting_indices_logged.clear();
        m_device->tex_cache().CancelDeferredGraphActivation();
        if (m_rendering_resources.pipeline_cache) {
            m_rendering_resources.pipeline_cache->clear();
        }
        m_rendering_resources.pipeline_cache.reset();
        m_rendering_resources.model_depth_images.clear();
        m_rendering_resources.masked_draw_attachments.clear();
        m_vertex_buf->destroy();
        m_dyn_buf->destroy();

        m_rendering_resources.command.reset();
        m_render_cmd.reset();
        m_cmds.reset();
        m_timestamp_pool.reset();
        m_pipeline_stats_pool.reset();
        m_gpu_diag_stream.close();
        m_gpu_pass_diag_stream.close();
        m_rendering_resources.frame_draw_stats = nullptr;

        m_device->Destroy();
    }
    m_rendering_resources.sem_swap_wait_image.reset();
    m_rendering_resources.sem_swap_finish.reset();
    m_rendering_resources.fence_frame.reset();
    m_rendering_resources.vertex_buf = nullptr;
    m_rendering_resources.dyn_buf = nullptr;
    m_prepass.reset();
    m_finpass.reset();
    m_testpass.reset();
    m_vertex_buf.reset();
    m_dyn_buf.reset();
    m_ex_swapchain.reset();
    m_device.reset();
    m_instance.Destroy();
    m_with_surface = false;
    m_inited = false;
    m_pass_loaded = false;
}

bool VulkanRender::Impl::CreateRenderingResource(RenderingResources& rr) {
    rr.command = m_render_cmd;
    VVK_CHECK_BOOL_RE(m_device->handle().CreateFence(
        VkFenceCreateInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        },
        rr.fence_frame));

    rr.fence_frame.Reset();

    if (m_with_surface) {
        VkSemaphoreCreateInfo ci { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                   .pNext = nullptr };
        VVK_CHECK_BOOL_RE(m_device->handle().CreateSemaphore(ci, rr.sem_swap_finish));
        VVK_CHECK_BOOL_RE(m_device->handle().CreateSemaphore(ci, rr.sem_swap_wait_image));
    }

    rr.vertex_buf = m_vertex_buf.get();
    rr.dyn_buf    = m_dyn_buf.get();
    rr.pipeline_cache = std::make_shared<GraphicsPipelineStateCache>();
    return true;
}

void VulkanRender::Impl::DestroyRenderingResource(RenderingResources& rr) {}

void VulkanRender::Impl::ensureGpuFrameDiagnostics() {
    if (m_gpu_diag_initialized) return;
    m_gpu_diag_initialized = true;

    const char* path = std::getenv("VIVID_GPU_FRAME_DIAGNOSTICS_PATH");
    if (path == nullptr || path[0] == '\0') return;

    m_gpu_diag_stream.open(path, std::ios::out | std::ios::trunc);
    if (! m_gpu_diag_stream) {
        LOG_ERROR("GpuFrameDiagnostics: open failed path='%s'", path);
        return;
    }
    m_gpu_diag_stream
        << "frame,cpu_update_ms,cpu_record_upload_ms,cpu_execute_ms,cpu_fence_ms,cpu_frame_ms,"
           "gpu_buffer_upload_ms,gpu_tex_upload_ms,gpu_draw_ms,gpu_total_ms,"
           "gpu_particle_ms,gpu_video_ms,gpu_effect_ms,gpu_copy_ms,gpu_clear_ms,gpu_text_ms,"
           "gpu_composite_ms,gpu_other_ms,"
           "vertex_writes,vertex_mark_dirty,vertex_memcpy_bytes,vertex_ranges,vertex_flush_bytes,"
           "dyn_writes,dyn_mark_dirty,dyn_memcpy_bytes,dyn_ranges,dyn_flush_bytes,"
           "dyn_writes_after_upload,dyn_mark_dirty_after_upload,dyn_memcpy_after_upload,"
           "shader_draws,shader_vertices,shader_indices,prepared_passes,"
           "ia_vertices,ia_primitives,vs_invocations,gs_invocations,gs_primitives,"
           "clip_invocations,clip_primitives,fs_invocations,"
           "particle_draws,particle_vertices\n";

    const char* pass_env = std::getenv("VIVID_GPU_PASS_DIAGNOSTICS");
    m_pass_diag_enabled =
        pass_env == nullptr || pass_env[0] == '\0' ||
        (pass_env[0] != '0' && pass_env[0] != 'n' && pass_env[0] != 'N' && pass_env[0] != 'f' &&
         pass_env[0] != 'F');
    if (m_pass_diag_enabled) {
        const auto pass_path = GpuPassCsvPath(path);
        m_gpu_pass_diag_stream.open(pass_path, std::ios::out | std::ios::trunc);
        if (m_gpu_pass_diag_stream) {
            m_gpu_pass_diag_stream
                << "frame,pass_index,category,layer_id,primitive,blend,draws,vertices,indices,"
                   "gpu_ms,ia_vertices,ia_primitives,vs_invocations,gs_invocations,gs_primitives,"
                   "clip_invocations,clip_primitives,fs_invocations,key\n";
        } else {
            LOG_ERROR("GpuFrameDiagnostics: open pass csv failed path='%s'", pass_path.c_str());
        }
    }

    m_vertex_buf->setCollectFrameStats(true);
    m_dyn_buf->setCollectFrameStats(true);
    m_rendering_resources.frame_draw_stats = &m_frame_draw_stats;
    m_gpu_diag_passes.reserve(kGpuPassCapacity);

    const auto families = m_device->gpu().GetQueueFamilyProperties();
    const auto family   = m_device->graphics_queue().family_index;
    if (family < families.size() && families[family].timestampValidBits > 0) {
        VkQueryPoolCreateInfo ci {
            .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType  = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kGpuTimestampQueryCount,
        };
        if (m_device->handle().CreateQueryPool(ci, m_timestamp_pool) == VK_SUCCESS) {
            m_timestamp_period_ns = m_device->limits().timestampPeriod;
            m_timestamps_ok       = true;
        } else {
            LOG_ERROR("GpuFrameDiagnostics: timestamp query pool create failed");
        }
    }

    const auto gpu_features = m_device->gpu().GetFeatures();
    if (m_pass_diag_enabled && gpu_features.pipelineStatisticsQuery) {
        VkQueryPoolCreateInfo ci {
            .sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType          = VK_QUERY_TYPE_PIPELINE_STATISTICS,
            .queryCount         = kGpuPassCapacity,
            .pipelineStatistics = kGpuPipelineStatFlags,
        };
        if (m_device->handle().CreateQueryPool(ci, m_pipeline_stats_pool) == VK_SUCCESS) {
            m_pipeline_stats_ok = true;
        } else {
            LOG_ERROR("GpuFrameDiagnostics: pipeline statistics query pool create failed");
        }
    }

    m_gpu_diag_enabled = true;
    LOG_INFO("GpuFrameDiagnostics: recording path='%s' timestamps=%d pass_diag=%d "
             "pipeline_stats=%d period_ns=%f",
             path,
             m_timestamps_ok ? 1 : 0,
             m_pass_diag_enabled ? 1 : 0,
             m_pipeline_stats_ok ? 1 : 0,
             static_cast<double>(m_timestamp_period_ns));
}

void VulkanRender::Impl::finishGpuFrameDiagnostics(double cpu_update_ms, double cpu_record_upload_ms,
                                                   double cpu_execute_ms, double cpu_fence_ms,
                                                   double cpu_frame_ms) {
    if (! m_gpu_diag_enabled || ! m_gpu_diag_stream) return;

    const uint32_t pass_count =
        static_cast<uint32_t>(std::min(m_gpu_diag_passes.size(), static_cast<size_t>(kGpuPassCapacity)));
    const uint32_t stamp_count =
        m_pass_diag_enabled ? kGpuTimestampPassBase + pass_count : (kGpuTimestampEnd + 1);

    double gpu_buffer_upload_ms = 0.0;
    double gpu_tex_upload_ms    = 0.0;
    double gpu_draw_ms          = 0.0;
    double gpu_total_ms         = 0.0;
    std::array<double, kGpuPassCategoryCount> category_ms {};
    std::vector<double> pass_gpu_ms(pass_count, 0.0);
    std::vector<uint64_t> stamps(stamp_count, 0);
    if (m_timestamps_ok && m_timestamp_pool && stamp_count > 0) {
        const VkResult res = m_device->handle().GetQueryPoolResults(
            *m_timestamp_pool,
            0,
            stamp_count,
            stamps.size() * sizeof(uint64_t),
            stamps.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (res == VK_SUCCESS) {
            gpu_buffer_upload_ms =
                TimestampDeltaMs(stamps[kGpuTimestampBegin], stamps[kGpuTimestampAfterBuffers],
                                 m_timestamp_period_ns);
            gpu_tex_upload_ms = TimestampDeltaMs(
                stamps[kGpuTimestampAfterBuffers], stamps[kGpuTimestampAfterTextures],
                m_timestamp_period_ns);
            const uint64_t draw_end = stamps[kGpuTimestampEnd];
            gpu_draw_ms = TimestampDeltaMs(stamps[kGpuTimestampAfterTextures], draw_end,
                                           m_timestamp_period_ns);
            gpu_total_ms = TimestampDeltaMs(stamps[kGpuTimestampBegin], draw_end,
                                            m_timestamp_period_ns);
            uint64_t prev = stamps[kGpuTimestampAfterTextures];
            for (uint32_t i = 0; i < pass_count; i++) {
                const uint64_t cur = stamps[kGpuTimestampPassBase + i];
                const double ms = TimestampDeltaMs(prev, cur, m_timestamp_period_ns);
                pass_gpu_ms[i] = ms;
                const auto cat = static_cast<uint32_t>(m_gpu_diag_passes[i].category);
                if (cat < kGpuPassCategoryCount) category_ms[cat] += ms;
                prev = cur;
            }
        }
    }

    std::array<uint64_t, kGpuPipelineStatCount> totals {};
    std::vector<uint64_t> pass_stats(static_cast<size_t>(pass_count) * kGpuPipelineStatCount, 0);
    if (m_pipeline_stats_ok && m_pipeline_stats_pool && pass_count > 0) {
        const VkResult res = m_device->handle().GetQueryPoolResults(
            *m_pipeline_stats_pool,
            0,
            pass_count,
            pass_stats.size() * sizeof(uint64_t),
            pass_stats.data(),
            sizeof(uint64_t) * kGpuPipelineStatCount,
            VK_QUERY_RESULT_64_BIT);
        if (res == VK_SUCCESS) {
            for (uint32_t i = 0; i < pass_count; i++) {
                for (uint32_t s = 0; s < kGpuPipelineStatCount; s++) {
                    totals[s] += pass_stats[i * kGpuPipelineStatCount + s];
                }
            }
        }
    }

    uint32_t particle_draws    = 0;
    uint32_t particle_vertices = 0;
    for (uint32_t i = 0; i < pass_count; i++) {
        const auto& rec = m_gpu_diag_passes[i];
        if (rec.category != GpuPassCategory::Particle) continue;
        particle_draws += rec.draws;
        particle_vertices += rec.vertices;
    }

    const auto& vertex = m_vertex_buf->frameStats();
    const auto& dyn    = m_dyn_buf->frameStats();
    const auto& before = m_dyn_stats_before_upload;
    const auto  sub_u32 = [](uint32_t later, uint32_t earlier) {
        return later > earlier ? later - earlier : 0u;
    };
    const auto sub_u64 = [](uint64_t later, uint64_t earlier) {
        return later > earlier ? later - earlier : 0ull;
    };

    auto category = [&](GpuPassCategory c) {
        return category_ms[static_cast<uint32_t>(c)];
    };

    m_gpu_diag_stream << m_gpu_diag_frame << ',' << cpu_update_ms << ',' << cpu_record_upload_ms
                      << ',' << cpu_execute_ms << ',' << cpu_fence_ms << ',' << cpu_frame_ms << ','
                      << gpu_buffer_upload_ms << ',' << gpu_tex_upload_ms << ',' << gpu_draw_ms
                      << ',' << gpu_total_ms << ','
                      << category(GpuPassCategory::Particle) << ','
                      << category(GpuPassCategory::Video) << ','
                      << category(GpuPassCategory::Effect) << ','
                      << category(GpuPassCategory::Copy) << ','
                      << category(GpuPassCategory::Clear) << ','
                      << category(GpuPassCategory::Text) << ','
                      << category(GpuPassCategory::Composite) << ','
                      << category(GpuPassCategory::Other) << ','
                      << vertex.write_calls << ','
                      << vertex.mark_dirty_calls << ',' << vertex.memcpy_bytes << ','
                      << vertex.dirty_range_count << ',' << vertex.flush_bytes << ','
                      << dyn.write_calls << ',' << dyn.mark_dirty_calls << ',' << dyn.memcpy_bytes
                      << ',' << dyn.dirty_range_count << ',' << dyn.flush_bytes << ','
                      << sub_u32(dyn.write_calls, before.write_calls) << ','
                      << sub_u32(dyn.mark_dirty_calls, before.mark_dirty_calls) << ','
                      << sub_u64(dyn.memcpy_bytes, before.memcpy_bytes) << ','
                      << m_frame_draw_stats.shader_draws << ','
                      << m_frame_draw_stats.shader_vertices << ','
                      << m_frame_draw_stats.shader_indices << ','
                      << m_frame_draw_stats.prepared_passes << ','
                      << totals[0] << ',' << totals[1] << ',' << totals[2] << ',' << totals[3] << ','
                      << totals[4] << ',' << totals[5] << ',' << totals[6] << ',' << totals[7] << ','
                      << particle_draws << ',' << particle_vertices << '\n';

    if (m_gpu_pass_diag_stream) {
        for (uint32_t i = 0; i < pass_count; i++) {
            const auto& rec = m_gpu_diag_passes[i];
            const uint64_t* stats = pass_stats.data() + static_cast<size_t>(i) * kGpuPipelineStatCount;
            m_gpu_pass_diag_stream
                << m_gpu_diag_frame << ',' << i << ',' << GpuPassCategoryName(rec.category) << ','
                << rec.layer_id << ',' << rec.primitive << ',' << rec.blend << ','
                << rec.draws << ',' << rec.vertices << ',' << rec.indices << ','
                << pass_gpu_ms[i] << ','
                << stats[0] << ',' << stats[1] << ',' << stats[2] << ',' << stats[3] << ','
                << stats[4] << ',' << stats[5] << ',' << stats[6] << ',' << stats[7] << ','
                << CsvEscape(rec.key) << '\n';
        }
    }

    if ((m_gpu_diag_frame % 30) == 0) {
        m_gpu_diag_stream.flush();
        if (m_gpu_pass_diag_stream) m_gpu_pass_diag_stream.flush();
    }
    m_gpu_diag_frame++;
}

// VulkanExSwapchain* VulkanRender::exSwapchain() const { return m_ex_swapchain.get(); }

void VulkanRender::Impl::drawFrame(Scene& scene) {
    if (m_device_faulted) return;
    if (! (m_inited && m_pass_loaded)) return;

    // The QuickJS host records getVideoTexture().play()/pause() decisions on Scene before the
    // renderer polls GStreamer. Applying them here keeps hidden authored videos from decoding
    // while prepared passes can still reuse the last uploaded frame when they are invisible.
    m_device->video_tex_cache().ApplyPlaybackStates(scene.videoTexturePaused,
                                                    scene.videoTextureStopped,
                                                    scene.videoTextureRates);
    // setCurrentTime() requests are one-shot decoder commands, so the video cache consumes and
    // removes only the requests whose concrete GStreamer pipeline already exists.
    m_device->video_tex_cache().ApplySeekRequests(scene.videoTextureSeekRequests);
    m_device->video_tex_cache().Poll();
    m_device->video_tex_cache().PublishRuntimeStates(
        scene.videoTextureRuntimeStates, scene.videoTextureRuntimeStateRequests);
    processDeferredGraphPreparation(scene);

    // Opt-in GPU memory attribution dump for diagnosing device-memory regressions against the
    // reference implementation. VMA's JSON stats list every allocation with its size and pool, so
    // one periodic snapshot in a real desktop session shows exactly which textures/buffers own
    // the footprint. Enabled only via environment variable to keep steady-state frames free of
    // stats-collection cost.
    {
        static const bool  dump_vma_stats = std::getenv("VIVID_VMA_STATS") != nullptr;
        static std::size_t vma_stats_frame_counter { 0 };
        if (dump_vma_stats && m_device && (vma_stats_frame_counter++ % 600 == 0)) {
            char* stats_json = nullptr;
            vmaBuildStatsString(m_device->vma_allocator(), &stats_json, VK_TRUE);
            if (stats_json != nullptr) {
                if (FILE* out = std::fopen("/tmp/vivid-vma-stats.json", "w")) {
                    std::fputs(stats_json, out);
                    std::fclose(out);
                    LOG_INFO("VMA stats dumped to /tmp/vivid-vma-stats.json usage=%.1fMiB",
                             m_device->GetUsage() / 1024.0 / 1024.0);
                }
                vmaFreeStatsString(m_device->vma_allocator(), stats_json);
            }
        }
    }

#if ENABLE_RENDERDOC_API
    if (rdoc_api)
        rdoc_api->StartFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE((VkInstance)m_instance.inst()), NULL);
#endif

    if (m_instance.offscreen()) {
        drawFrameOffscreen();
    } else {
        drawFrameSwapchain();
    }

    if (m_redraw_cb) m_redraw_cb();

#if ENABLE_RENDERDOC_API
    if (rdoc_api)
        rdoc_api->EndFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE((VkInstance)m_instance.inst()), NULL);
#endif
}

void VulkanRender::Impl::setPaused(bool paused) {
    if (m_device == nullptr) return;
    m_device->video_tex_cache().SetGlobalPaused(paused);
}

void VulkanRender::Impl::setOffscreenFrameReleaseCallback(
    OffscreenFrameReleaseCallback callback) {
    m_offscreen_frame_release_cb = std::move(callback);
}

bool VulkanRender::Impl::reconfigureOffscreenExport(
    uint32_t width,
    uint32_t height,
    TexTiling tiling,
    ExternalFrameExportMode export_mode,
    uint32_t export_drm_fourcc,
    const std::vector<uint64_t>& export_drm_modifiers,
    ExternalFrameMemoryPreference memory_preference) {
    if (!m_inited || !m_device || !m_ex_swapchain || !m_instance.offscreen())
        return false;

    const bool ok = m_ex_swapchain->Reconfigure(*m_device,
                                                width,
                                                height,
                                                tiling == TexTiling::OPTIMAL
                                                    ? VK_IMAGE_TILING_OPTIMAL
                                                    : VK_IMAGE_TILING_LINEAR,
                                                export_mode,
                                                export_drm_fourcc,
                                                export_drm_modifiers,
                                                memory_preference);
    LOG_INFO("HanabiScene Vulkan: offscreen export reconfigure %s size=%ux%u "
             "fourcc=0x%08x modifier-count=%zu first-modifier=0x%016llx memory=%s",
             ok ? "succeeded" : "failed",
             width,
             height,
             export_drm_fourcc,
             export_drm_modifiers.size(),
             export_drm_modifiers.empty()
                 ? 0ull
                 : static_cast<unsigned long long>(export_drm_modifiers.front()),
             ExternalMemoryPreferenceName(memory_preference));
    return ok;
}

void VulkanRender::Impl::refreshImportedTextures(Scene& scene) {
    if (!m_device) return;

    for (const auto& key : scene.dirtyImportedTextureKeys) {
        scene.DropParsedImageCache(key);
        auto image = scene.ParseImageBlockingCached(key);
        if (!image) continue;
        m_device->tex_cache().CreateTex(*image);
        scene.DropParsedImageCache(key);
    }
    scene.dirtyImportedTextureKeys.clear();
}

void VulkanRender::Impl::processDeferredGraphPreparation(Scene& scene) {
    if (m_deferred_prepare_indices.empty()) return;
    if (m_device_faulted || !m_device) return;

    const auto batch_started_at = std::chrono::steady_clock::now();
    std::size_t attempted = 0;
    std::size_t prepared = 0;

    while (attempted < kDeferredPrepareMaxPassesPerFrame && !m_deferred_prepare_indices.empty()) {
        if (attempted != 0) {
            const auto batch_elapsed_ms =
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - batch_started_at)
                                        .count()) /
                1000.0;
            if (batch_elapsed_ms >= kDeferredPrepareFrameBudgetMs) {
                break;
            }
        }

        const auto pass_index = m_deferred_prepare_indices.front();
        if (pass_index >= m_passes.size()) {
            m_deferred_prepare_indices.pop_front();
            m_deferred_waiting_indices_logged.erase(pass_index);
            continue;
        }

        auto* pass = m_passes[pass_index];
        if (pass == nullptr || pass->prepared()) {
            m_deferred_prepare_indices.pop_front();
            m_deferred_waiting_indices_logged.erase(pass_index);
            continue;
        }

        const auto key = pass->residencyKey();
        const auto resources_state = pass->requestDeferredPrepareResources(scene, *m_device);
        if (resources_state == DeferredPrepareResourcesState::Waiting) {
            if (m_deferred_waiting_indices_logged.insert(pass_index).second) {
                LOG_INFO("RenderGraphDeferredPrepareWait: index=%zu remaining=%zu key='%s'",
                         pass_index,
                         m_deferred_prepare_indices.size(),
                         key.c_str());
            }
            break;
        }

        m_deferred_waiting_indices_logged.erase(pass_index);

        const auto pass_started_at = std::chrono::steady_clock::now();
        pass->prepareDeferred(scene, *m_device, m_rendering_resources);
        const auto pass_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - pass_started_at)
                                         .count();
        attempted++;
        if (pass->prepared()) {
            prepared++;
            m_deferred_prepare_indices.pop_front();
        }

        LOG_INFO("RenderGraphDeferredPreparePass: index=%zu prepared=%s remaining=%zu "
                 "duration=%.2fms key='%s'",
                 pass_index,
                 pass->prepared() ? "true" : "false",
                 m_deferred_prepare_indices.size(),
                 static_cast<double>(pass_elapsed_us) / 1000.0,
                 key.c_str());
        if (!pass->prepared()) {
            break;
        }
        const auto batch_elapsed_ms =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - batch_started_at)
                                    .count()) /
            1000.0;
        if (batch_elapsed_ms >= kDeferredPrepareFrameBudgetMs) {
            break;
        }
    }

    const auto batch_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - batch_started_at)
                                      .count();
    if (attempted != 0) {
        LOG_INFO("RenderGraphDeferredPrepareBatch: attempted=%zu prepared=%zu remaining=%zu "
                 "duration=%.2fms",
                 attempted,
                 prepared,
                 m_deferred_prepare_indices.size(),
                 static_cast<double>(batch_elapsed_us) / 1000.0);
    }

    if (m_deferred_prepare_indices.empty()) {
        m_deferred_waiting_indices_logged.clear();
        m_device->tex_cache().EndDeferredGraphActivation();
        LOG_INFO("RenderGraphDeferredPrepareComplete");
    }
}

void VulkanRender::Impl::drawFrameSwapchain() {
    static size_t resource_index = 0;

    RenderingResources& rr = m_rendering_resources;
    resource_index         = (resource_index + 1) % 3;
    uint32_t image_index   = 0;
    {
        if (!checkVkResult(m_device->handle().AcquireNextImageKHR(*m_device->swapchain().handle(),
                                                                  vk_wait_time,
                                                                  *rr.sem_swap_wait_image,
                                                                  {},
                                                                  &image_index),
                           "acquire swapchain image"))
            return;
    }
    const auto& image = m_device->swapchain().images()[image_index];

    m_finpass->setPresent(image);

    for (auto* p : m_passes) {
        if (p->prepared()) {
            // Dynamic passes copy current CPU-side vertex/index bytes into staging before the
            // upload command is recorded. This keeps reused-source particle systems from binding a
            // freshly grown suballocation whose GPU contents have not been uploaded yet.
            p->updateBeforeUpload();
        }
    }

    if (!checkVkResult(rr.command.Begin(VkCommandBufferBeginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    }), "begin swapchain frame command buffer"))
        return;
    // Deferred pass preparation can allocate and write static vertex/index subranges between
    // frames. Recording the static upload here keeps those newly resident passes drawable without
    // a compile-time WaitIdle, matching the frame-budgeted residency model used by streaming
    // renderers.
    m_vertex_buf->recordUpload(rr.command);
    m_dyn_buf->recordUpload(rr.command);
    m_device->tex_cache().RecordUploads(rr.command);
    m_device->video_tex_cache().RecordUploads(rr.command);
    for (auto* p : m_passes) {
        if (p->prepared()) {
            p->execute(*m_device, rr);
        }
    }
    if (!checkVkResult(rr.command.End(), "end swapchain frame command buffer"))
        return;

    VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo         sub_info {
                .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext                = nullptr,
                .waitSemaphoreCount   = 1,
                .pWaitSemaphores      = rr.sem_swap_wait_image.address(),
                .pWaitDstStageMask    = &wait_dst_stage,
                .commandBufferCount   = 1,
                .pCommandBuffers      = rr.command.address(),
                .signalSemaphoreCount = 1,
                .pSignalSemaphores    = rr.sem_swap_finish.address(),
    };

    if (!checkVkResult(m_device->present_queue().handle.Submit(sub_info, *rr.fence_frame),
                       "submit swapchain frame"))
        return;
    VkPresentInfoKHR present_info {
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = rr.sem_swap_finish.address(),
        .swapchainCount     = 1,
        .pSwapchains        = m_device->swapchain().handle().address(),
        .pImageIndices      = &image_index,
    };
    if (!checkVkResult(m_device->present_queue().handle.Present(present_info),
                       "present swapchain frame"))
        return;

    if (!checkVkResult(rr.fence_frame.Wait(vk_wait_time), "wait swapchain frame fence"))
        return;
    m_device->tex_cache().RetireCompletedUploads();
    if (!checkVkResult(rr.fence_frame.Reset(), "reset swapchain frame fence"))
        return;
}

void VulkanRender::Impl::drawFrameOffscreen() {
    RenderingResources& rr = m_rendering_resources;
    if (!m_ex_swapchain) {
        return;
    }

    auto render_lock = m_ex_swapchain->acquireRenderLock();
    auto* inprogress_handle = m_ex_swapchain->getInprogress();
    if (!inprogress_handle) {
        return;
    }

    const uint32_t slot_id = static_cast<uint32_t>(inprogress_handle->id());
    if (m_offscreen_frame_release_cb) {
        /*
         * Vivid reuses the exported offscreen image ring directly as the display
         * transport. Mirror waywallen's BridgeProducerCore::acquireSlot(): wait
         * for the consumer release timeline immediately before recording GPU
         * writes into the selected slot. A timeout skips this render tick without
         * calling renderFrame(), so the ready slot remains the last fully
         * published image and no still-owned DMA-BUF is overwritten.
         */
        if (!m_offscreen_frame_release_cb(slot_id)) {
            return;
        }
    }

    ensureGpuFrameDiagnostics();
    const auto frame_started = std::chrono::steady_clock::now();
    if (m_gpu_diag_enabled) {
        m_frame_draw_stats.reset();
        m_vertex_buf->resetFrameStats();
        m_dyn_buf->resetFrameStats();
        m_dyn_stats_before_upload = {};
    }

    ImageParameters image = m_ex_swapchain->GetInprogressImage();

    m_finpass->setPresent(image);

    const double cpu_update_ms = TimeMs([&] {
        for (auto* p : m_passes) {
            if (p->prepared()) {
                // Offscreen rendering exports the result to GTK, making stale particle bytes visible as
                // source-switch flicker. Pre-updating dynamic mesh data aligns the following m_dyn_buf
                // upload with the frame that will be exported.
                p->updateBeforeUpload();
            }
        }
    });
    if (m_gpu_diag_enabled) m_dyn_stats_before_upload = m_dyn_buf->frameStats();

    if (!checkVkResult(rr.command.Begin(VkCommandBufferBeginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    }), "begin offscreen frame command buffer"))
        return;
    if (m_timestamps_ok) {
        rr.command.ResetQueryPool(*m_timestamp_pool, 0, kGpuTimestampQueryCount);
        rr.command.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *m_timestamp_pool,
                                  kGpuTimestampBegin);
    }
    if (m_pass_diag_enabled && m_pipeline_stats_ok) {
        rr.command.ResetQueryPool(*m_pipeline_stats_pool, 0, kGpuPassCapacity);
    }
    const double cpu_record_upload_ms = TimeMs([&] {
        m_vertex_buf->recordUpload(rr.command);
        m_dyn_buf->recordUpload(rr.command);
        if (m_timestamps_ok) {
            rr.command.WriteTimestamp(VK_PIPELINE_STAGE_TRANSFER_BIT, *m_timestamp_pool,
                                      kGpuTimestampAfterBuffers);
        }
        m_device->tex_cache().RecordUploads(rr.command);
        m_device->video_tex_cache().RecordUploads(rr.command);
        if (m_timestamps_ok) {
            rr.command.WriteTimestamp(VK_PIPELINE_STAGE_TRANSFER_BIT, *m_timestamp_pool,
                                      kGpuTimestampAfterTextures);
        }
    });

    if (m_gpu_diag_enabled) m_gpu_diag_passes.clear();
    const double cpu_execute_ms = TimeMs([&] {
        uint32_t pass_index = 0;
        for (auto* p : m_passes) {
            if (! p->prepared()) continue;
            if (m_gpu_diag_enabled) m_frame_draw_stats.prepared_passes++;
            const bool record_pass =
                m_gpu_diag_enabled && m_pass_diag_enabled && pass_index < kGpuPassCapacity;
            const uint32_t draws_before    = m_frame_draw_stats.shader_draws;
            const uint32_t vertices_before = m_frame_draw_stats.shader_vertices;
            const uint32_t indices_before  = m_frame_draw_stats.shader_indices;
            if (record_pass && m_pipeline_stats_ok) {
                rr.command.BeginQuery(*m_pipeline_stats_pool, pass_index, 0);
            }
            p->execute(*m_device, rr);
            if (record_pass && m_pipeline_stats_ok) {
                rr.command.EndQuery(*m_pipeline_stats_pool, pass_index);
            }
            if (record_pass && m_timestamps_ok) {
                rr.command.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *m_timestamp_pool,
                                          kGpuTimestampPassBase + pass_index);
            }
            if (record_pass) {
                const auto info = p->gpuDiagInfo();
                m_gpu_diag_passes.push_back(GpuPassDiagRecord {
                    .category  = info.category,
                    .layer_id  = info.layer_id,
                    .draws     = m_frame_draw_stats.shader_draws - draws_before,
                    .vertices  = m_frame_draw_stats.shader_vertices - vertices_before,
                    .indices   = m_frame_draw_stats.shader_indices - indices_before,
                    .primitive = info.primitive,
                    .blend     = info.blend,
                    .key       = p->residencyKey(),
                });
            }
            pass_index++;
        }
    });
    if (m_timestamps_ok) {
        rr.command.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *m_timestamp_pool,
                                  kGpuTimestampEnd);
    }

    if (!checkVkResult(rr.command.End(), "end offscreen frame command buffer"))
        return;

    VkSubmitInfo sub_info {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext              = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers    = rr.command.address(),
    };
    if (!checkVkResult(m_device->graphics_queue().handle.Submit(sub_info, *rr.fence_frame),
                       "submit offscreen frame"))
        return;

    const auto fence_started = std::chrono::steady_clock::now();
    if (!checkVkResult(rr.fence_frame.Wait(vk_wait_time), "wait offscreen frame fence"))
        return;
    const double cpu_fence_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fence_started)
            .count();
    m_device->tex_cache().RetireCompletedUploads();
    if (!checkVkResult(rr.fence_frame.Reset(), "reset offscreen frame fence"))
        return;
    m_ex_swapchain->renderFrame();
    if (m_gpu_diag_enabled) {
        const double cpu_frame_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      frame_started)
                .count();
        finishGpuFrameDiagnostics(cpu_update_ms, cpu_record_upload_ms, cpu_execute_ms, cpu_fence_ms,
                                  cpu_frame_ms);
    }
}

void VulkanRender::Impl::setRenderTargetSize(Scene& scene, rg::RenderGraph& rg) {
    auto& ext = m_device->out_extent();
    for (auto& item : scene.renderTargets) {
        auto& rt = item.second;
        if (rt.bind.enable && rt.bind.screen) {
            const i32 new_w = (i32)(rt.bind.scale * ext.width);
            const i32 new_h = (i32)(rt.bind.scale * ext.height);
            if (rt.width != new_w || rt.height != new_h) {
                LOG_INFO("SceneScreenBoundTarget: name='%s' previous=[%d %d] screen=[%d %d] "
                         "scale=%.3f extent=%ux%u",
                         item.first.c_str(),
                         rt.width,
                         rt.height,
                         new_w,
                         new_h,
                         rt.bind.scale,
                         ext.width,
                         ext.height);
            }
            rt.width  = new_w;
            rt.height = new_h;
            // Screen-sized render targets expose the full framebuffer as both their physical and
            // logical extent. Only text-owned runtime targets intentionally diverge these values.
            rt.mapWidth = rt.width;
            rt.mapHeight = rt.height;
        }
    }
    for (auto& item : scene.renderTargets) {
        auto& rt = item.second;
        if (rt.bind.screen || ! rt.bind.enable) continue;
        auto bind_rt = scene.renderTargets.find(rt.bind.name);
        if (rt.bind.name.empty() || bind_rt == scene.renderTargets.end()) {
            LOG_ERROR("unknonw render target bind: %s", rt.bind.name.c_str());
            continue;
        }
        rt.width  = (i32)(rt.bind.scale * bind_rt->second.width);
        rt.height = (i32)(rt.bind.scale * bind_rt->second.height);
        // Bound render targets inherit the sampled content rectangle from their source target so
        // shader uniforms continue to describe the authored image area rather than the raw backing
        // allocation. This keeps generic effect chains consistent whenever the source target uses
        // a logical content rectangle that differs from its physical allocation.
        rt.mapWidth = (i32)(rt.bind.scale * bind_rt->second.ContentWidth());
        rt.mapHeight = (i32)(rt.bind.scale * bind_rt->second.ContentHeight());
    }
    for (auto& item : scene.renderTargets) {
        auto& rt = item.second;
        if (! item.first.empty() && (rt.width * rt.height <= 4)) {
            LOG_ERROR("wrong size for render target: %s", item.first.c_str());
        } else if (rt.has_mipmap) {
            rt.mipmap_level =
                std::max(3u,
                         static_cast<uint>(std::floor(std::log2(std::min(rt.width, rt.height))))) -
                2u;
        }
    }
    scene.shaderValueUpdater->SetScreenSize((i32)ext.width, (i32)ext.height);
}

void VulkanRender::Impl::UpdateCameraFillMode(wallpaper::Scene&   scene,
                                              wallpaper::FillMode fillmode) {
    using namespace wallpaper;
    auto width  = m_device->out_extent().width;
    auto height = m_device->out_extent().height;

    if (width == 0 || height == 0) return;
    scene.physicalOutputExtent = { width, height };
    double sw = scene.ortho[0], sh = scene.ortho[1];
    double fboAspect = width / (double)height, sAspect = sw / sh;
    auto&  gCam    = *scene.cameras.at("global");
    auto&  gPerCam = *scene.cameras.at("global_perspective");

    // Camera-layer animation mutates the same shared "global" camera object that fill mode uses to
    // adapt a 16:9 project to the monitor aspect. Preserve the live zoom value here so the render
    // side remains the single source of truth for framebuffer-relative width/height while authored
    // zoom still narrows that already aspect-correct view.
    double active_global_zoom = scene.defaultGlobalCameraZoom;
    if (!std::isfinite(active_global_zoom) || active_global_zoom <= 0.0001) {
        active_global_zoom = 1.0;
    }

    // Perspective camera layers can animate FOV directly. Keep that authored value when the active
    // layer explicitly targets the shared perspective camera; otherwise derive FOV from the
    // orthographic framing so perspective particles continue to match the visible crop/fit window.
    bool  use_active_global_perspective_fov = false;
    float active_global_perspective_fov = 50.0f;
    if (scene.activeCameraLayerId != 0) {
        auto active_layer_it = scene.cameraLayers.find(scene.activeCameraLayerId);
        if (active_layer_it != scene.cameraLayers.end()) {
            const auto& active_layer = active_layer_it->second;
            if (active_layer.camera_name.empty() || active_layer.camera_name == "global") {
                if (std::isfinite(active_layer.zoom) && active_layer.zoom > 0.0001) {
                    active_global_zoom = active_layer.zoom;
                } else {
                    active_global_zoom = 1.0;
                }
            } else if (active_layer.camera_name == "global_perspective" &&
                       std::isfinite(active_layer.fov) && active_layer.fov > 0.0001f) {
                use_active_global_perspective_fov = true;
                active_global_perspective_fov = active_layer.fov;
            }
        }
    }

    double framed_width = sw;
    double framed_height = sh;
    double perspective_aspect = sAspect;

    switch (fillmode) {
    case FillMode::STRETCH:
        framed_width = sw;
        framed_height = sh;
        perspective_aspect = sAspect;
        break;
    case FillMode::ASPECTFIT:
        if (fboAspect < sAspect) {
            // Preserve the full scene width and add vertical space when the output is taller.
            framed_width = sw;
            framed_height = sw / fboAspect;
        } else {
            framed_width = sh * fboAspect;
            framed_height = sh;
        }
        perspective_aspect = fboAspect;
        break;
    case FillMode::ASPECTCROP:
    default:
        if (fboAspect > sAspect) {
            // Preserve the full scene width and crop vertically when the output is wider.
            framed_width = sw;
            framed_height = sw / fboAspect;
        } else {
            framed_width = sh * fboAspect;
            framed_height = sh;
        }
        perspective_aspect = fboAspect;
        break;
    }

    gCam.SetWidth(std::max(1.0, framed_width / active_global_zoom));
    gCam.SetHeight(std::max(1.0, framed_height / active_global_zoom));
    gPerCam.SetAspect(perspective_aspect);
    gPerCam.SetFov(use_active_global_perspective_fov
                       ? active_global_perspective_fov
                       : algorism::CalculatePersperctiveFov(1000.0f, gCam.Height()));
    gCam.Update();
    gPerCam.Update();
    scene.UpdateLinkedCamera("global");

    if (!scene.modelPerspectiveCameraName.empty()) {
        auto model_camera_it = scene.cameras.find(scene.modelPerspectiveCameraName);
        if (model_camera_it != scene.cameras.end() && model_camera_it->second) {
            // 3D model chunks render through a camera that is intentionally isolated from the
            // legacy `global_perspective` camera, but its projection still has to follow the same
            // fill-mode-adjusted framebuffer aspect. Without this, a 16:9-authored model scene keeps
            // its native projection while Vulkan draws into a 16:10 or other non-native viewport,
            // which changes the apparent object proportions even though the model transform itself
            // is uniform. Only the aspect is synchronized here: the authored 3D FOV and the
            // camera-path eye/center/up basis remain owned by the scene data and path playback.
            model_camera_it->second->SetAspect(perspective_aspect);
            model_camera_it->second->Update();
            scene.UpdateLinkedCamera(scene.modelPerspectiveCameraName);
        } else {
            // A named model camera should exist whenever model nodes were materialized. Log this
            // explicitly instead of silently falling back to another camera, because choosing a
            // substitute would hide the real render-graph/parser state mismatch and make 3D aspect
            // regressions harder to diagnose from run.log.
            LOG_ERROR("Scene3DModelCameraAspect: missing model camera '%s' while applying "
                      "fill-mode perspective aspect %.6f",
                      scene.modelPerspectiveCameraName.c_str(),
                      perspective_aspect);
        }
    }

    // Text layers with Wallpaper Engine's screen-anchor property are authored against the project
    // canvas edge, but the active orthographic camera edge moves when aspect crop/fit changes the
    // visible frame. Re-apply those anchor transforms after camera framing so HUD-style text
    // remains inside the actual output instead of the uncropped project bounds.
    ApplyTextLayerScreenAnchorTransforms(scene);
    // Text bridge images are final-screen-density resources, not glyph-atlas-density resources.
    // Recompute them only after fill mode and screen anchors have produced the frame's authoritative
    // camera and world transforms.
    UpdateAllTextLayerBridgeBackings(scene);
}

void VulkanRender::Impl::clearLastRenderGraph(bool clear_scene_caches) {
    if (m_device_faulted) {
        // After device loss, pass destruction can call vkDestroyPipeline and friends on a driver
        // context that already timed out.  Leave the bounded stale graph abandoned with the renderer
        // instead of turning a recoverable backend replacement into a process crash.
        return;
    }

    // A topology rebuild invalidates the compiled pass list and the backing mesh buffers that were
    // uploaded for the previous graph. Reallocating those buffers keeps the full rebuild path
    // conservative and mirrors the historical behavior used when nodes were added or removed.
    for (auto& p : m_passes) {
        p->destory(*m_device, m_rendering_resources);
    }
    m_passes.clear();
    m_compiled_pass_refs.clear();
    m_deferred_prepare_indices.clear();
    m_deferred_waiting_indices_logged.clear();
    m_device->tex_cache().CancelDeferredGraphActivation();
    if (clear_scene_caches) {
        // Scene switches and renderer shutdown still own a full cache teardown. Ordinary topology
        // rebuilds no longer do this: visibility-driven residency now releases only the keys that
        // became unreachable, so showing one deferred layer cannot evict every unrelated texture
        // and video decoder in the wallpaper.
        m_device->tex_cache().Clear();
        m_device->video_tex_cache().Clear();
        if (m_rendering_resources.pipeline_cache) {
            m_rendering_resources.pipeline_cache->clear();
        }
    }
    // Shared model depth images are tied to the compiled graph's output targets. Dropping them on
    // full graph rebuilds keeps 3D model depth opt-in and avoids stale depth attachments surviving
    // after scene topology or render-target ownership changes.
    m_rendering_resources.model_depth_images.clear();
    m_rendering_resources.masked_draw_attachments.clear();

    m_vertex_buf->destroy();
    m_dyn_buf->destroy();

    m_vertex_buf->allocate();
    m_dyn_buf->allocate();
}

void VulkanRender::Impl::releasePendingSceneResources(Scene& scene) {
    if (m_device_faulted || !m_device) return;
    if (scene.pendingStaticTextureReleaseKeys.empty() &&
        scene.pendingVideoTextureReleaseKeys.empty() &&
        scene.pendingRenderTargetReleaseKeys.empty()) {
        return;
    }

    const auto before_texture_bytes = m_device->tex_cache().GetTrackedBytes();
    const auto before_texture_count = m_device->tex_cache().GetTrackedImageCount();
    const auto before_video_bytes   = m_device->video_tex_cache().GetTrackedBytes();
    const auto before_video_count   = m_device->video_tex_cache().GetTrackedEntryCount();

    std::size_t released_static = 0;
    std::size_t released_render_targets = 0;
    std::size_t released_videos = 0;

    for (const auto& key : scene.pendingStaticTextureReleaseKeys) {
        if (m_device->tex_cache().ReleaseTexture(key)) released_static++;
        scene.DropParsedImageCache(key);
    }
    for (const auto& key : scene.pendingRenderTargetReleaseKeys) {
        if (m_device->tex_cache().ReleaseRenderTarget(key)) released_render_targets++;
    }
    for (const auto& key : scene.pendingVideoTextureReleaseKeys) {
        if (m_device->video_tex_cache().Release(key)) released_videos++;
    }

    LOG_INFO("SceneResidencyRelease: static=%zu/%zu render-target=%zu/%zu video=%zu/%zu "
             "texture-bytes-before=%zu texture-bytes-after=%zu texture-images-before=%zu "
             "texture-images-after=%zu video-bytes-before=%zu video-bytes-after=%zu "
             "video-entries-before=%zu video-entries-after=%zu",
             released_static,
             scene.pendingStaticTextureReleaseKeys.size(),
             released_render_targets,
             scene.pendingRenderTargetReleaseKeys.size(),
             released_videos,
             scene.pendingVideoTextureReleaseKeys.size(),
             before_texture_bytes,
             m_device->tex_cache().GetTrackedBytes(),
             before_texture_count,
             m_device->tex_cache().GetTrackedImageCount(),
             before_video_bytes,
             m_device->video_tex_cache().GetTrackedBytes(),
             before_video_count,
             m_device->video_tex_cache().GetTrackedEntryCount());

    scene.pendingStaticTextureReleaseKeys.clear();
    scene.pendingVideoTextureReleaseKeys.clear();
    scene.pendingRenderTargetReleaseKeys.clear();
}

void VulkanRender::Impl::clearRenderGraphResources() {
    // Resource-only rebuilds are hot resource refreshes, not a miniature full rebuild. Particle
    // effects already update every frame without clearing global caches; effect-backed text must
    // follow the same rule. TextureCache::Query now detects per-key TextureKey changes and
    // reallocates only the resized render target, so clearing the entire cache here would recreate
    // unrelated offscreen images and reintroduce the minute-rollover hitch.
}

void VulkanRender::Impl::compileRenderGraph(Scene& scene, rg::RenderGraph& rg,
                                            bool refresh_resources_only) {
    if (m_device_faulted) return;
    if (! m_inited) return;
    m_pass_loaded = false;
    const bool had_resident_graph = !m_compiled_pass_refs.empty();

    if (refresh_resources_only && !m_passes.empty()) {
        setRenderTargetSize(scene, rg);

        const auto dirty_render_targets = scene.dirtyRenderTargetKeys;
        const auto dirty_text_layers = scene.dirtyTextLayerIds;
        // A resource refresh can now be targeted either by render-target key or by text layer id.
        // Treating an empty render-target set as "refresh everything" was correct before direct
        // text had its own dirty set, but it would turn every Clock tick back into a full pass walk.
        const bool has_targeted_dirty_resources =
            !dirty_render_targets.empty() || !dirty_text_layers.empty();
        const bool refresh_all =
            scene.renderGraphAllResourcesDirty || !has_targeted_dirty_resources;
        std::size_t refreshed_passes = 0;
        std::size_t prepared_passes = 0;

        for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index) {
            auto* p = m_passes[pass_index];
            if (p == nullptr) continue;

            const bool affected =
                refresh_all || p->referencesAnyRenderTarget(dirty_render_targets) ||
                p->referencesAnyTextLayer(dirty_text_layers);
            if (!affected) continue;

            if (p->prepared()) {
                // Text bridge updates are now target-scoped like particle resources: refresh only
                // passes that touch the resized render targets so a one-pixel glyph-width change
                // cannot force every static shader pass in the wallpaper to rebind resources.
                p->refreshResources(scene, *m_device, m_rendering_resources);
                refreshed_passes++;
            }
            if (!p->prepared()) {
                p->prepare(scene, *m_device, m_rendering_resources);
                prepared_passes++;
            }
        }

        // Resource-only refreshes are intentionally silent in production; the counters stay local
        // so the branch preserves targeted text-bridge behavior without making minute rollovers
        // spend time formatting render-graph diagnostics.
        (void)refresh_all;
        (void)refreshed_passes;
        (void)prepared_passes;
        // Mature renderers do not submit a separate upload command and idle the whole device while
        // rebuilding resource bindings. The next draw command records all dirty vertex, dynamic,
        // and texture uploads before executing passes, preserving ordering without a render-thread
        // queue drain.
        m_pass_loaded = true;
        return;
    }

    auto nodes             = rg.topologicalOrder();
    auto node_release_texs = rg.getLastReadTexs(nodes);

    m_passes.clear();
    m_deferred_prepare_indices.clear();
    m_deferred_waiting_indices_logged.clear();
    m_device->tex_cache().CancelDeferredGraphActivation();
    m_passes.resize(nodes.size());

    std::unordered_map<std::string, std::shared_ptr<rg::Pass>> reusable_passes;
    std::unordered_map<std::string, std::size_t> old_key_counts;
    for (const auto& old_pass_ref : m_compiled_pass_refs) {
        auto old_pass = std::dynamic_pointer_cast<VulkanPass>(old_pass_ref);
        if (!old_pass) continue;
        const auto key = MakeResidencyInstanceKey(*old_pass, old_key_counts);
        if (!key.empty()) reusable_passes.emplace(key, old_pass_ref);
    }

    std::unordered_map<std::string, std::size_t> new_key_counts;
    std::unordered_set<VulkanPass*> reused_passes;
    std::vector<std::shared_ptr<rg::Pass>> next_compiled_pass_refs;
    next_compiled_pass_refs.reserve(nodes.size());
    std::size_t reused_count = 0;
    std::size_t new_count = 0;

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const auto node_id = nodes[index];
        auto       pass_ref = rg.getPassShared(node_id);
        assert(pass_ref != nullptr);
        auto*      vpass = dynamic_cast<VulkanPass*>(pass_ref.get());
        assert(vpass != nullptr);

        const auto key = MakeResidencyInstanceKey(*vpass, new_key_counts);
        if (!key.empty()) {
            if (auto reusable_it = reusable_passes.find(key);
                reusable_it != reusable_passes.end()) {
                auto reusable_vpass = std::dynamic_pointer_cast<VulkanPass>(reusable_it->second);
                if (reusable_vpass && reusable_vpass->canReuseForResidency(*vpass)) {
                    // Mature renderers do not destroy every pipeline just because one layer toggled
                    // visibility. The new render graph describes the fresh topology, while this
                    // handoff keeps matching prepared pass objects alive and updates only their
                    // graph-local gates/texture declarations. Removed passes are retired below,
                    // which preserves hidden-layer resource release without a whole-scene PSO
                    // rebuild.
                    reusable_vpass->absorbResidencyGraphState(*vpass);
                    pass_ref = reusable_it->second;
                    vpass = reusable_vpass.get();
                    rg.replacePass(node_id, pass_ref);
                    reused_passes.insert(vpass);
                    reused_count++;
                    reusable_passes.erase(reusable_it);
                }
            }
        }
        if (reused_passes.count(vpass) == 0) {
            new_count++;
        }

        // Release ownership is compiled from the current render graph topology, not from pass
        // construction. Clear stale metadata before assigning this graph's final-reader keys so
        // reused pass objects keep an exact lifecycle contract.
        vpass->clearReleaseTexs();
        for (auto& tex : node_release_texs[index]) {
            vpass->addReleaseTexs(spanone<const std::string_view> { tex->key() });
        }
        m_passes[index] = vpass;
        next_compiled_pass_refs.push_back(std::move(pass_ref));
    }

    std::unordered_set<VulkanPass*> destroyed_passes;
    std::size_t retired_count = 0;
    for (const auto& [_, stale_pass_ref] : reusable_passes) {
        auto stale_pass = std::dynamic_pointer_cast<VulkanPass>(stale_pass_ref);
        if (!stale_pass) continue;
        const auto before_destroy_count = destroyed_passes.size();
        DestroyPassOnce(stale_pass.get(), *m_device, m_rendering_resources, destroyed_passes);
        if (destroyed_passes.size() != before_destroy_count) retired_count++;
    }
    m_compiled_pass_refs = std::move(next_compiled_pass_refs);
    releasePendingSceneResources(scene);

    LOG_INFO("RenderGraphResidencyDiff: reused=%zu new=%zu retired=%zu graph-passes=%zu",
             reused_count,
             new_count,
             retired_count,
             nodes.size());

    m_passes.insert(m_passes.begin(), m_prepass.get());
    m_passes.push_back(m_finpass.get());

    setRenderTargetSize(scene, rg);

    std::size_t reused_refreshed_count = 0;
    std::size_t refreshed_count = 0;
    std::size_t prepared_count = 0;
    std::size_t dependency_prepared_count = 0;
    std::size_t deferred_count = 0;
    std::size_t deferred_waiting_count = 0;
    std::size_t already_prepared_count = 0;
    // CopyPass is a lightweight graph-residency pass, not a heavy shader pass: it registers
    // dynamic copy render targets such as `_rt_default_pingpong` in Scene::renderTargets and binds
    // their TextureCache images. Reused shader passes can legitimately sample those copy targets
    // during the same topology compile, so deferring CopyPass creation lets refreshed passes see a
    // missing input and black out the frame. Prepare copy dependencies up front, then keep the
    // expensive shader/image passes on the deferred residency queue.
    for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index) {
        auto* p = m_passes[pass_index];
        if (p == nullptr || p->prepared()) continue;
        if (dynamic_cast<CopyPass*>(p) == nullptr) continue;
        p->prepare(scene, *m_device, m_rendering_resources);
        dependency_prepared_count++;
    }
    for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index) {
        auto* p = m_passes[pass_index];
        if (p != nullptr && reused_passes.count(p) != 0 && p->prepared()) {
            p->refreshResources(scene, *m_device, m_rendering_resources);
            reused_refreshed_count++;
        }
        if (refresh_resources_only && p != nullptr && p->prepared()) {
            p->refreshResources(scene, *m_device, m_rendering_resources);
            refreshed_count++;
        }
        if (p != nullptr && !p->prepared()) {
            const bool is_copy_dependency = dynamic_cast<CopyPass*>(p) != nullptr;
            const bool can_defer_runtime_prepare =
                had_resident_graph && !refresh_resources_only && p != m_prepass.get() &&
                p != m_finpass.get() && !is_copy_dependency;
            if (can_defer_runtime_prepare) {
                // Runtime visibility changes should not monopolize the render thread by making
                // every newly-visible layer allocate textures, framebuffers, and staging uploads in
                // one compile call. Queue cold passes and let drawFrame() advance residency a pass
                // at a time while already-prepared scene content keeps rendering.
                if (p->requestDeferredPrepareResources(scene, *m_device) ==
                    DeferredPrepareResourcesState::Waiting) {
                    deferred_waiting_count++;
                }
                m_deferred_prepare_indices.push_back(pass_index);
                deferred_count++;
            } else {
                p->prepare(scene, *m_device, m_rendering_resources);
                prepared_count++;
            }
        } else if (p != nullptr) {
            already_prepared_count++;
        }
    }

    LOG_INFO("RenderGraphCompileSummary: total=%zu reused-refreshed=%zu refreshed=%zu "
             "prepared=%zu dependency-prepared=%zu deferred=%zu already-prepared=%zu mode=%s",
             m_passes.size(),
             reused_refreshed_count,
             refreshed_count,
             prepared_count,
             dependency_prepared_count,
             deferred_count,
             already_prepared_count,
             refresh_resources_only ? "resources" : "topology");
    if (deferred_count > 0) {
        m_device->tex_cache().BeginDeferredGraphActivation();
        LOG_INFO("RenderGraphDeferredPrepareQueued: count=%zu max-passes-per-frame=%zu "
                 "frame-budget=%.2fms resource-waiting=%zu",
                 deferred_count,
                 kDeferredPrepareMaxPassesPerFrame,
                 kDeferredPrepareFrameBudgetMs,
                 deferred_waiting_count);
    }

    // Upload work queued by prepare() is recorded at the start of the next frame command buffer.
    // Avoiding a compile-time queue submit + DeviceWaitIdle is what keeps visibility-driven graph
    // changes from behaving like a scene load.
    m_pass_loaded = true;
};

void VulkanRender::Impl::warmupRenderGraphPipelines(Scene& scene, rg::RenderGraph& rg) {
    if (m_device_faulted) return;
    if (!m_inited || !m_device || !m_rendering_resources.pipeline_cache) return;

    const auto started_at = std::chrono::steady_clock::now();
    auto       nodes      = rg.topologicalOrder();

    setRenderTargetSize(scene, rg);

    std::size_t pipeline_passes = 0;
    std::size_t warmed_passes   = 0;

    for (const auto node_id : nodes) {
        auto pass_ref = rg.getPassShared(node_id);
        auto vpass = std::dynamic_pointer_cast<VulkanPass>(pass_ref);
        if (!vpass) continue;
        pipeline_passes++;
        if (vpass->warmupPipeline(scene, *m_device, m_rendering_resources)) {
            warmed_passes++;
        }
    }

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started_at)
                                .count();
    LOG_INFO("RenderGraphPipelineWarmup: graph-passes=%zu pipeline-passes=%zu warmed=%zu "
             "cached-states=%zu duration=%.2fms",
             nodes.size(),
             pipeline_passes,
             warmed_passes,
             m_rendering_resources.pipeline_cache->size(),
             static_cast<double>(elapsed_us) / 1000.0);
}

#include "VideoTextureCache.hpp"

#include "Device.hpp"
#include "Image.hpp"
#include "Scene/SceneTexture.h"
#include "TextureCache.hpp"
#include "Util.hpp"
#include "Utils/Logging.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <limits.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <drm_fourcc.h>
#include <ffnvcodec/dynlink_cuda.h>
#include <gio/gio.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <gst/va/gstvaallocator.h>
#include <gst/va/gstvadisplay.h>
#include <gst/video/video-info-dma.h>
#include <gst/video/video.h>
#include <va/va_drmcommon.h>
#include <unistd.h>

using namespace wallpaper;
using namespace wallpaper::vulkan;

extern "C" {
typedef struct _GstCudaMemory GstCudaMemory;
typedef struct _GstCudaContext GstCudaContext;
typedef struct _GstCudaStream GstCudaStream;

gboolean gst_cuda_load_library(void);
gboolean gst_is_cuda_memory(GstMemory* mem);
GstCudaStream* gst_cuda_memory_get_stream(GstCudaMemory* mem);
CUstream gst_cuda_stream_get_handle(GstCudaStream* stream);
gboolean gst_cuda_context_push(GstCudaContext* ctx);
gboolean gst_cuda_context_pop(CUcontext* cuda_ctx);

CUresult CUDAAPI CuGetErrorName(CUresult error, const char** pStr);
CUresult CUDAAPI CuGetErrorString(CUresult error, const char** pStr);
CUresult CUDAAPI CuInit(unsigned int Flags);
CUresult CUDAAPI CuDeviceGetCount(int* count);
CUresult CUDAAPI CuDeviceGet(CUdevice* device, int ordinal);
CUresult CUDAAPI CuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI CuImportExternalMemory(CUexternalMemory* extMem_out,
                                        const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc);
CUresult CUDAAPI CuDestroyExternalMemory(CUexternalMemory extMem);
CUresult CUDAAPI CuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr,
                                                 CUexternalMemory extMem,
                                                 const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc);
CUresult CUDAAPI CuModuleLoadData(CUmodule* module, const void* image);
CUresult CUDAAPI CuModuleUnload(CUmodule module);
CUresult CUDAAPI CuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult CUDAAPI CuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void** kernelParams,
                                void** extra);
CUresult CUDAAPI CuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI CuCtxSynchronize(void);
}

namespace
{

constexpr GstMapFlags kGstMapReadCuda =
    static_cast<GstMapFlags>(GST_MAP_READ | (GST_MAP_FLAG_LAST << 1));

// Keep the NVIDIA path independent from libnvrtc: several distro nvcodec builds can
// decode CUDA memory without shipping the runtime compiler.  The driver can JIT PTX
// directly, so this embedded kernel avoids silently falling back to the CPU pipeline.
constexpr char kNv12ToRgbaCudaPtx[] = R"ptx(
.version 6.0
.target sm_52
.address_size 64

.visible .entry HanabiNv12ToRgba(
    .param .u64 y_plane_param,
    .param .u64 uv_plane_param,
    .param .u64 rgba_param,
    .param .u32 width_param,
    .param .u32 height_param,
    .param .u32 y_pitch_param,
    .param .u32 uv_pitch_param,
    .param .u32 rgba_pitch_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [y_plane_param];
    ld.param.u64 %rd2, [uv_plane_param];
    ld.param.u64 %rd3, [rgba_param];
    ld.param.u32 %r1, [width_param];
    ld.param.u32 %r2, [height_param];
    ld.param.u32 %r3, [y_pitch_param];
    ld.param.u32 %r4, [uv_pitch_param];
    ld.param.u32 %r5, [rgba_pitch_param];

    mov.u32 %r10, %ctaid.x;
    mov.u32 %r11, %ntid.x;
    mov.u32 %r12, %tid.x;
    mad.lo.u32 %r20, %r10, %r11, %r12;

    mov.u32 %r13, %ctaid.y;
    mov.u32 %r14, %ntid.y;
    mov.u32 %r15, %tid.y;
    mad.lo.u32 %r21, %r13, %r14, %r15;

    setp.ge.u32 %p1, %r20, %r1;
    @%p1 bra DONE;
    setp.ge.u32 %p2, %r21, %r2;
    @%p2 bra DONE;

    mad.lo.u32 %r30, %r21, %r3, %r20;
    cvt.u64.u32 %rd10, %r30;
    add.u64 %rd10, %rd1, %rd10;
    ld.global.u8 %r31, [%rd10];

    shr.u32 %r32, %r21, 1;
    and.b32 %r33, %r20, -2;
    mad.lo.u32 %r34, %r32, %r4, %r33;
    cvt.u64.u32 %rd11, %r34;
    add.u64 %rd11, %rd2, %rd11;
    ld.global.u8 %r35, [%rd11];
    add.u64 %rd12, %rd11, 1;
    ld.global.u8 %r36, [%rd12];

    sub.s32 %r37, %r31, 16;
    max.s32 %r37, %r37, 0;
    sub.s32 %r38, %r35, 128;
    sub.s32 %r39, %r36, 128;

    mul.lo.s32 %r40, %r37, 298;

    mul.lo.s32 %r41, %r39, 459;
    add.s32 %r41, %r41, %r40;
    add.s32 %r41, %r41, 128;
    shr.s32 %r41, %r41, 8;
    max.s32 %r41, %r41, 0;
    min.s32 %r41, %r41, 255;

    mul.lo.s32 %r42, %r38, 55;
    sub.s32 %r42, %r40, %r42;
    mul.lo.s32 %r43, %r39, 136;
    sub.s32 %r42, %r42, %r43;
    add.s32 %r42, %r42, 128;
    shr.s32 %r42, %r42, 8;
    max.s32 %r42, %r42, 0;
    min.s32 %r42, %r42, 255;

    mul.lo.s32 %r44, %r38, 541;
    add.s32 %r44, %r44, %r40;
    add.s32 %r44, %r44, 128;
    shr.s32 %r44, %r44, 8;
    max.s32 %r44, %r44, 0;
    min.s32 %r44, %r44, 255;

    shl.b32 %r50, %r20, 2;
    mad.lo.u32 %r51, %r21, %r5, %r50;
    cvt.u64.u32 %rd13, %r51;
    add.u64 %rd13, %rd3, %rd13;
    st.global.u8 [%rd13], %r41;
    add.u64 %rd14, %rd13, 1;
    st.global.u8 [%rd14], %r42;
    add.u64 %rd14, %rd13, 2;
    st.global.u8 [%rd14], %r44;
    add.u64 %rd14, %rd13, 3;
    mov.u32 %r52, 255;
    st.global.u8 [%rd14], %r52;

DONE:
    ret;
}
)ptx";

const char* CudaErrorName(CUresult result) {
    const char* name = nullptr;
    if (CuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) return name;
    return "unknown";
}

bool EnsureCudaDriverInitialized(std::string& diagnostic) {
    /*
     * Video textures use CUDA only to prove that NVDEC and the Vulkan renderer
     * are on the same physical NVIDIA GPU. The CUDA driver API requires an
     * explicit initialization before device enumeration on some systems, so do
     * that here and keep failures diagnostic instead of falling back elsewhere.
     */
    const CUresult result = CuInit(0);
    if (result == CUDA_SUCCESS) return true;

    diagnostic = std::string("cuInit failed: ") + CudaErrorName(result);
    return false;
}

std::string PciAddressForRenderNode(const std::string& render_node) {
    if (render_node.empty()) return {};

    const auto slash = render_node.rfind('/');
    const std::string basename =
        slash == std::string::npos ? render_node : render_node.substr(slash + 1);
    const std::string sysfs_path = "/sys/class/drm/" + basename + "/device";

    char resolved[PATH_MAX] = {0};
    if (realpath(sysfs_path.c_str(), resolved) == nullptr) return {};

    const char* address = std::strrchr(resolved, '/');
    address = address != nullptr ? address + 1 : resolved;
    return address != nullptr ? std::string(address) : std::string {};
}

std::optional<unsigned int>
CudaDeviceIdForPciAddress(const std::string& pci_address, std::string& diagnostic) {
    unsigned int domain = 0;
    unsigned int bus = 0;
    unsigned int device = 0;
    unsigned int function = 0;
    if (std::sscanf(pci_address.c_str(),
                    "%x:%x:%x.%x",
                    &domain,
                    &bus,
                    &device,
                    &function) != 4) {
        diagnostic = "invalid PCI address '" +
            (pci_address.empty() ? std::string("(empty)") : pci_address) + "'";
        return std::nullopt;
    }

    if (! EnsureCudaDriverInitialized(diagnostic)) return std::nullopt;

    int count = 0;
    CUresult result = CuDeviceGetCount(&count);
    if (result != CUDA_SUCCESS) {
        diagnostic = std::string("cuDeviceGetCount failed: ") + CudaErrorName(result);
        return std::nullopt;
    }

    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice cuda_device = 0;
        result = CuDeviceGet(&cuda_device, ordinal);
        if (result != CUDA_SUCCESS) continue;

        int cuda_domain = 0;
        int cuda_bus = 0;
        int cuda_device_number = 0;
        if (CuDeviceGetAttribute(&cuda_domain,
                                 CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                 cuda_device) != CUDA_SUCCESS ||
            CuDeviceGetAttribute(&cuda_bus,
                                 CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                 cuda_device) != CUDA_SUCCESS ||
            CuDeviceGetAttribute(&cuda_device_number,
                                 CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                 cuda_device) != CUDA_SUCCESS) {
            continue;
        }

        if (static_cast<unsigned int>(cuda_domain) == domain &&
            static_cast<unsigned int>(cuda_bus) == bus &&
            static_cast<unsigned int>(cuda_device_number) == device) {
            return static_cast<unsigned int>(ordinal);
        }
    }

    char message[128] = {0};
    std::snprintf(message,
                  sizeof(message),
                  "no CUDA device matches PCI %04x:%02x:%02x.%x",
                  domain,
                  bus,
                  device,
                  function);
    diagnostic = message;
    return std::nullopt;
}

std::optional<unsigned int>
CudaDeviceIdForRenderNode(const std::string& render_node, std::string& diagnostic) {
    const std::string pci_address = PciAddressForRenderNode(render_node);
    if (pci_address.empty()) {
        diagnostic = "render node '" +
            (render_node.empty() ? std::string("(empty)") : render_node) +
            "' has no PCI address";
        return std::nullopt;
    }
    return CudaDeviceIdForPciAddress(pci_address, diagnostic);
}

std::optional<unsigned int> CudaDeviceIdFromContext(GstCudaContext* cuda_context) {
    if (cuda_context == nullptr ||
        !g_object_class_find_property(G_OBJECT_GET_CLASS(cuda_context), "cuda-device-id")) {
        return std::nullopt;
    }

    guint cuda_device_id = G_MAXUINT;
    g_object_get(cuda_context, "cuda-device-id", &cuda_device_id, nullptr);
    if (cuda_device_id == G_MAXUINT) return std::nullopt;
    return cuda_device_id;
}

bool CudaContextMatchesExpected(GstCudaContext* cuda_context,
                                int expected_cuda_device_id,
                                std::string_view owner_label) {
    if (expected_cuda_device_id < 0) return true;

    const auto actual = CudaDeviceIdFromContext(cuda_context);
    if (! actual.has_value()) {
        LOG_ERROR("%.*s: CUDA context has no readable cuda-device-id; refusing "
                  "NVIDIA decode because the selected GPU cannot be proven",
                  static_cast<int>(owner_label.size()),
                  owner_label.data());
        return false;
    }

    if (actual.value() != static_cast<unsigned int>(expected_cuda_device_id)) {
        LOG_ERROR("%.*s: CUDA context device mismatch expected=%d actual=%u; "
                  "refusing cross-GPU NVIDIA decode",
                  static_cast<int>(owner_label.size()),
                  owner_label.data(),
                  expected_cuda_device_id,
                  actual.value());
        return false;
    }

    return true;
}

bool DecoderElementMatchesExpectedCudaDevice(GstElement* decoder,
                                             int expected_cuda_device_id) {
    if (expected_cuda_device_id < 0) return true;

    if (decoder == nullptr ||
        !g_object_class_find_property(G_OBJECT_GET_CLASS(decoder), "cuda-device-id")) {
        LOG_ERROR("video texture NVIDIA decoder exposes no cuda-device-id; "
                  "refusing decode because selected GPU cannot be proven");
        return false;
    }

    guint actual_cuda_device_id = G_MAXUINT;
    g_object_get(decoder, "cuda-device-id", &actual_cuda_device_id, nullptr);
    if (actual_cuda_device_id != static_cast<unsigned int>(expected_cuda_device_id)) {
        LOG_ERROR("video texture NVIDIA decoder CUDA device mismatch expected=%d "
                  "actual=%u; refusing cross-GPU decode",
                  expected_cuda_device_id,
                  actual_cuda_device_id);
        return false;
    }
    return true;
}

bool EndsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size(), suffix.size()) == suffix;
}

struct DecoderSettings {
    VideoTextureDecoderRoute decoder_route { VideoTextureDecoderRoute::Nvidia };
    std::string              render_node;
};

const char* DecoderRouteName(VideoTextureDecoderRoute route) {
    switch (route) {
    case VideoTextureDecoderRoute::Va: return "va";
    case VideoTextureDecoderRoute::Nvidia: return "nvidia";
    }
    return "nvidia";
}

std::string VaElementFactoryName(const std::string& render_node, const char* element_suffix) {
    constexpr std::string_view default_node = "/dev/dri/renderD128";
    if (render_node.empty() || render_node == default_node)
        return std::string("va") + element_suffix;

    const auto slash = render_node.rfind('/');
    const std::string basename =
        slash == std::string::npos ? render_node : render_node.substr(slash + 1);
    return std::string("va") + basename + element_suffix;
}

std::vector<std::string> VaElementFactoryCandidates(const std::string& render_node,
                                                    const char* element_suffix) {
    std::vector<std::string> candidates;
    const std::string node_factory = VaElementFactoryName(render_node, element_suffix);
    candidates.push_back(node_factory);
    const std::string plain_factory = std::string("va") + element_suffix;
    if (plain_factory != node_factory) candidates.push_back(plain_factory);
    return candidates;
}

DecoderSettings MakeDecoderSettings(VideoTextureDecoderSettings runtime_settings) {
    DecoderSettings settings {
        .decoder_route = runtime_settings.decoder_route,
        .render_node   = runtime_settings.render_node,
    };
    LOG_INFO("VideoTextureSettings: decoder-route=%s render-node=%s",
             DecoderRouteName(settings.decoder_route),
             settings.render_node.empty() ? "(unresolved)" : settings.render_node.c_str());
    return settings;
}

void SetPluginDecoderRanks(const char* plugin_name, guint rank, bool use_stateless = false) {
    GstRegistry* registry = gst_registry_get();
    if (registry == nullptr) return;

    GList* features = gst_registry_get_feature_list_by_plugin(registry, plugin_name);
    for (GList* it = features; it != nullptr; it = it->next) {
        auto* feature = GST_PLUGIN_FEATURE(it->data);
        if (feature == nullptr) continue;

        const char* name = gst_plugin_feature_get_name(feature);
        if (name == nullptr) continue;

        std::string_view feature_name(name);
        if (!EndsWith(feature_name, "dec") && !EndsWith(feature_name, "postproc")) continue;

        const bool is_stateless = feature_name.find("sl") != std::string_view::npos;
        if (is_stateless != use_stateless) continue;

        if (gst_plugin_feature_get_rank(feature) == rank) continue;
        gst_plugin_feature_set_rank(feature, rank);
    }
    gst_plugin_feature_list_free(features);
}

void ConfigureDecoderRanks(const DecoderSettings& settings) {
    // The producer resolves the physical GPU before scene Vulkan starts. These
    // rank tweaks only make the selected decoder route deterministic inside
    // GStreamer; they are not a separate GPU selection policy.
    constexpr guint preferred_rank = GST_RANK_PRIMARY + 4;
    if (settings.decoder_route == VideoTextureDecoderRoute::Va) {
        SetPluginDecoderRanks("va", preferred_rank, false);
        SetPluginDecoderRanks("nvcodec", GST_RANK_NONE, false);
        SetPluginDecoderRanks("nvcodec", GST_RANK_NONE, true);
    } else {
        SetPluginDecoderRanks("va", GST_RANK_NONE, false);
        SetPluginDecoderRanks("nvcodec", preferred_rank + 1, false);
        SetPluginDecoderRanks("nvcodec", preferred_rank, true);
    }
}

std::optional<guint> ElementFactoryRank(const char* name) {
    GstElementFactory* factory = gst_element_factory_find(name);
    if (factory == nullptr) return std::nullopt;
    guint rank = gst_plugin_feature_get_rank(GST_PLUGIN_FEATURE(factory));
    gst_object_unref(factory);
    return rank;
}

std::string ElementDevicePath(const char* factory_name) {
    GstElement* element = gst_element_factory_make(factory_name, nullptr);
    if (element == nullptr) return {};

    std::string path;
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), "device-path")) {
        gchar* raw_path = nullptr;
        g_object_get(element, "device-path", &raw_path, nullptr);
        if (raw_path != nullptr) {
            path = raw_path;
            g_free(raw_path);
        }
    }
    gst_object_unref(element);
    return path;
}

struct VaElementSelection {
    std::string factory_name;
    std::string device_path;
    guint rank { GST_RANK_NONE };
};

std::optional<VaElementSelection> SelectVaElementFactoryForRenderNode(const std::string& render_node,
                                                                      const char* element_suffix,
                                                                      bool require_positive_rank,
                                                                      std::string& diagnostic) {
    std::vector<std::string> details;

    if (render_node.empty()) {
        diagnostic = "selected render-node is empty";
        return std::nullopt;
    }

    /*
     * Factory names alone are not a reliable device identity. Depending on the
     * VA plugin/device layout, "vah264dec" can point at /dev/dri/renderD129
     * while "varenderD129h264dec" is not registered at all. The renderer is
     * already pinned to a Vulkan device by UUID, so the decode side must prove
     * that the accepted VA element advertises the same DRM render node.
     */
    for (const std::string& candidate : VaElementFactoryCandidates(render_node, element_suffix)) {
        const auto rank = ElementFactoryRank(candidate.c_str());
        const std::string device_path =
            rank.has_value() ? ElementDevicePath(candidate.c_str()) : "";

        gchar* detail = g_strdup_printf("%s(rank=%u device-path=%s)",
                                        candidate.c_str(),
                                        rank.value_or(GST_RANK_NONE),
                                        device_path.empty() ? "(none)" : device_path.c_str());
        details.emplace_back(detail ? detail : "");
        g_free(detail);

        if (! rank.has_value()) continue;
        if (require_positive_rank && rank.value() == GST_RANK_NONE) continue;
        if (device_path != render_node) continue;
        return VaElementSelection {
            .factory_name = candidate,
            .device_path = device_path,
            .rank = rank.value(),
        };
    }

    diagnostic.clear();
    for (std::size_t i = 0; i < details.size(); ++i) {
        if (i > 0) diagnostic += ", ";
        diagnostic += details[i];
    }
    return std::nullopt;
}

enum class VideoTextureTransferPath
{
    None,
    VaMemoryBgra,
    NvidiaCudaNv12,
    NvidiaStatelessCudaNv12,
};

bool IsNvidiaCudaPath(VideoTextureTransferPath path) {
    return path == VideoTextureTransferPath::NvidiaCudaNv12 ||
           path == VideoTextureTransferPath::NvidiaStatelessCudaNv12;
}

const char* VideoTextureTransferPathName(VideoTextureTransferPath path) {
    switch (path) {
    case VideoTextureTransferPath::VaMemoryBgra: return "va-vamemory-bgra";
    case VideoTextureTransferPath::NvidiaCudaNv12: return "nvidia-cuda-nv12";
    case VideoTextureTransferPath::NvidiaStatelessCudaNv12: return "nvidia-stateless-cuda-nv12";
    case VideoTextureTransferPath::None: return "none";
    }
    return "none";
}

struct VideoTextureDecodeGraph {
    VideoTextureTransferPath path { VideoTextureTransferPath::None };
    const char* name { "none" };
    const char* sink_caps { "" };
    std::string description;
};

std::optional<VideoTextureDecodeGraph>
BuildVideoTextureDecodeGraph(VideoTextureTransferPath path,
                             const std::string& decoder_factory,
                             const std::string& postproc_factory) {
    if (path == VideoTextureTransferPath::VaMemoryBgra) {
        if (decoder_factory.empty() || postproc_factory.empty()) return std::nullopt;
        return VideoTextureDecodeGraph {
            .path = path,
            .name = "va-vamemory-bgra-vulkan-image",
            .sink_caps = "video/x-raw(memory:VAMemory),format=(string)BGRA",
            .description =
                "giostreamsrc name=src "
                "! qtdemux name=demux "
                "demux.video_0 ! queue "
                "! h264parse "
                "! " + decoder_factory + " name=video_texture_decoder "
                "! " + postproc_factory + " name=video_texture_postproc "
                "! video/x-raw(memory:VAMemory),format=(string)BGRA "
                "! appsink name=sink sync=true max-buffers=1 drop=true",
        };
    }

    if (path == VideoTextureTransferPath::NvidiaCudaNv12) {
        if (decoder_factory.empty()) return std::nullopt;
        return VideoTextureDecodeGraph {
            .path = path,
            .name = "nvidia-cuda-nv12-vulkan-buffer",
            .sink_caps = "video/x-raw(memory:CUDAMemory),format=(string)NV12",
            .description =
                "giostreamsrc name=src "
                "! qtdemux name=demux "
                "demux.video_0 ! queue "
                "! h264parse "
                "! " + decoder_factory + " name=video_texture_decoder "
                "! video/x-raw(memory:CUDAMemory),format=(string)NV12 "
                "! appsink name=sink sync=true max-buffers=1 drop=true",
        };
    }

    if (path == VideoTextureTransferPath::NvidiaStatelessCudaNv12) {
        if (decoder_factory.empty()) return std::nullopt;
        return VideoTextureDecodeGraph {
            .path = path,
            .name = "nvidia-stateless-cuda-nv12-vulkan-buffer",
            .sink_caps = "video/x-raw(memory:CUDAMemory),format=(string)NV12",
            .description =
                "giostreamsrc name=src "
                "! qtdemux name=demux "
                "demux.video_0 ! queue "
                "! h264parse "
                "! " + decoder_factory + " name=video_texture_decoder "
                "! video/x-raw(memory:CUDAMemory),format=(string)NV12 "
                "! appsink name=sink sync=true max-buffers=1 drop=true",
        };
    }

    return std::nullopt;
}

bool SupportsVaDmabufImport(const Device& device) {
    return device.supportExt(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
           device.supportExt(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) &&
           device.supportExt(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);
}

struct VideoTextureDecoderSelection {
    VideoTextureTransferPath path { VideoTextureTransferPath::None };
    const char* name { "none" };
    std::string decoder_factory;
    std::string postproc_factory;
    std::string decoder_device_path;
    std::string postproc_device_path;
    guint rank { GST_RANK_NONE };
    int expected_cuda_device_id { -1 };
};

std::optional<VideoTextureDecoderSelection>
SelectVideoTextureDecoder(const Device& device, const DecoderSettings& settings) {
    const bool has_va_dmabuf_import = SupportsVaDmabufImport(device);
    const bool has_cuda_loader = gst_cuda_load_library();
    std::string cuda_diagnostic;
    std::optional<unsigned int> expected_cuda_device;
    if (settings.decoder_route == VideoTextureDecoderRoute::Nvidia && has_cuda_loader) {
        expected_cuda_device =
            CudaDeviceIdForRenderNode(settings.render_node, cuda_diagnostic);
        /*
         * NVDEC has no VA-style render-node property. Map the renderer's DRM
         * node to PCI and then to CUDA ordinal; without that proof, accepting
         * nvh264dec would usually mean "CUDA device 0", not necessarily the
         * scene GPU selected by RenderInitInfo::uuid.
         */
        if (! expected_cuda_device.has_value()) {
            LOG_ERROR("VideoTextureDecoderSelect: selected NVIDIA render-node=%s "
                      "has no matching CUDA device id: %s; CPU fallback is disabled",
                      settings.render_node.empty() ? "(unresolved)" : settings.render_node.c_str(),
                      cuda_diagnostic.empty() ? "(unknown)" : cuda_diagnostic.c_str());
            return std::nullopt;
        }
    }
    std::string va_decoder_diagnostic;
    std::string va_postproc_diagnostic;
    std::optional<VaElementSelection> va_decoder;
    std::optional<VaElementSelection> va_postproc;

    if (settings.decoder_route == VideoTextureDecoderRoute::Va) {
        va_decoder = SelectVaElementFactoryForRenderNode(settings.render_node,
                                                         "h264dec",
                                                         true,
                                                         va_decoder_diagnostic);
        va_postproc = SelectVaElementFactoryForRenderNode(settings.render_node,
                                                          "postproc",
                                                          false,
                                                          va_postproc_diagnostic);
    }

    std::vector<VideoTextureDecoderSelection> candidates;

    /*
     * Scene video textures are explicit GPU decoder graphs, not decodebin
     * graphs. The producer has already selected the Vulkan physical device by
     * UUID and passes the matching decoder route/render node through
     * RenderInitInfo; if that route is unavailable we fail instead of falling
     * back to CPU decode or another GPU.
     */
    if (settings.decoder_route == VideoTextureDecoderRoute::Va &&
        va_decoder.has_value() && va_postproc.has_value() && has_va_dmabuf_import) {
        candidates.push_back(VideoTextureDecoderSelection {
            .path = VideoTextureTransferPath::VaMemoryBgra,
            .name = "VA VAMemory BGRA",
            .decoder_factory = va_decoder->factory_name,
            .postproc_factory = va_postproc->factory_name,
            .decoder_device_path = va_decoder->device_path,
            .postproc_device_path = va_postproc->device_path,
            .rank = va_decoder->rank,
        });
    }
    if (settings.decoder_route == VideoTextureDecoderRoute::Nvidia && has_cuda_loader) {
        if (auto rank = ElementFactoryRank("nvh264dec"); rank.has_value()) {
            candidates.push_back(VideoTextureDecoderSelection {
                .path = VideoTextureTransferPath::NvidiaCudaNv12,
                .name = "NVIDIA CUDA NV12",
                .decoder_factory = "nvh264dec",
                .postproc_factory = {},
                .rank = rank.value(),
                .expected_cuda_device_id = static_cast<int>(expected_cuda_device.value()),
            });
        }
        if (auto rank = ElementFactoryRank("nvh264sldec"); rank.has_value()) {
            candidates.push_back(VideoTextureDecoderSelection {
                .path = VideoTextureTransferPath::NvidiaStatelessCudaNv12,
                .name = "NVIDIA stateless CUDA NV12",
                .decoder_factory = "nvh264sldec",
                .postproc_factory = {},
                .rank = rank.value(),
                .expected_cuda_device_id = static_cast<int>(expected_cuda_device.value()),
            });
        }
    }

    auto best = std::max_element(candidates.begin(),
                                 candidates.end(),
                                 [](const VideoTextureDecoderSelection& lhs,
                                    const VideoTextureDecoderSelection& rhs) {
                                     return lhs.rank < rhs.rank;
                                 });
    if (best != candidates.end() && best->rank > GST_RANK_NONE) {
        LOG_INFO("VideoTextureDecoderSelect: selected %s path decoder=%s rank=%u "
                 "decoder-route=%s render-node=%s postproc=%s "
                 "decoder-device-path=%s postproc-device-path=%s cuda-device=%d",
                 best->name,
                 best->decoder_factory.c_str(),
                 best->rank,
                 DecoderRouteName(settings.decoder_route),
                 settings.render_node.empty() ? "(unresolved)" : settings.render_node.c_str(),
                 best->postproc_factory.empty() ? "(none)" : best->postproc_factory.c_str(),
                 best->decoder_device_path.empty() ? "(none)" : best->decoder_device_path.c_str(),
                 best->postproc_device_path.empty() ? "(none)" : best->postproc_device_path.c_str(),
                 best->expected_cuda_device_id);
        return *best;
    }

    LOG_ERROR("VideoTextureDecoderSelect: no usable GPU decoder for decoder-route=%s "
             "render-node=%s va-decoder-candidates=[%s] va-postproc-candidates=[%s] "
             "va-dmabuf-import=%s "
             "nvh264dec-rank=%u nvh264sldec-rank=%u cuda-loader=%s; "
             "CPU fallback is disabled to preserve the selected GPU invariant",
             DecoderRouteName(settings.decoder_route),
             settings.render_node.empty() ? "(unresolved)" : settings.render_node.c_str(),
             va_decoder_diagnostic.empty() ? "(not checked)" : va_decoder_diagnostic.c_str(),
             va_postproc_diagnostic.empty() ? "(not checked)" : va_postproc_diagnostic.c_str(),
             has_va_dmabuf_import ? "true" : "false",
             ElementFactoryRank("nvh264dec").value_or(GST_RANK_NONE),
             ElementFactoryRank("nvh264sldec").value_or(GST_RANK_NONE),
             has_cuda_loader ? "true" : "false");
    return std::nullopt;
}

VkFormat TargetImageFormatForTransferPath(VideoTextureTransferPath path) {
    return path == VideoTextureTransferPath::VaMemoryBgra
        ? VK_FORMAT_B8G8R8A8_UNORM
        : VK_FORMAT_R8G8B8A8_UNORM;
}

long long ElapsedMillis(std::chrono::steady_clock::time_point start,
                        std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::optional<VmaImageParameters>
CreateVideoImage(const Device& device, uint32_t width, uint32_t height, const TextureSample& sample,
                 VkFormat format, VkImageUsageFlags usage) {
    VmaImageParameters image;
    do {
        VkSamplerCreateInfo sampler_info {
            .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext                   = nullptr,
            .magFilter               = ToVkType(sample.magFilter),
            .minFilter               = ToVkType(sample.minFilter),
            .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU            = ToVkType(sample.wrapS),
            .addressModeV            = ToVkType(sample.wrapT),
            .addressModeW            = ToVkType(sample.wrapT),
            .anisotropyEnable        = false,
            .maxAnisotropy           = 1.0f,
            .compareEnable           = false,
            .compareOp               = VK_COMPARE_OP_NEVER,
            .minLod                  = 0.0f,
            .maxLod                  = 1.0f,
            .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = false,
        };
        VkImageCreateInfo info {
            .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext                 = nullptr,
            .imageType             = VK_IMAGE_TYPE_2D,
            .format                = format,
            .extent                = VkExtent3D { width, height, 1 },
            .mipLevels             = 1,
            .arrayLayers           = 1,
            .samples               = VK_SAMPLE_COUNT_1_BIT,
            .tiling                = VK_IMAGE_TILING_OPTIMAL,
            .usage                 = usage,
            .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        image.extent = info.extent;

        VmaAllocationCreateInfo vma_info {};
        vma_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VVK_CHECK_ACT(break, vvk::CreateImage(device.vma_allocator(), info, vma_info, image.handle));

        VkImageViewCreateInfo view_info {
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext    = nullptr,
            .image    = *image.handle,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = format,
            .subresourceRange =
                VkImageSubresourceRange {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
        };
        VVK_CHECK_ACT(break, device.handle().CreateImageView(view_info, image.view));
        VVK_CHECK_ACT(break, device.handle().CreateSampler(sampler_info, image.sampler));
        return image;
    } while (false);
    return std::nullopt;
}

VkResult TransitionImageLayout(const vvk::Queue& queue, vvk::CommandBuffer& cmd,
                               const ImageParameters& image, VkImageLayout old_layout,
                               VkImageLayout new_layout) {
    VkResult result;
    do {
        result = cmd.Begin(VkCommandBufferBeginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        });
        if (result != VK_SUCCESS) break;

        VkImageMemoryBarrier barrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .oldLayout        = old_layout,
            .newLayout        = new_layout,
            .image            = image.handle,
            .subresourceRange =
                VkImageSubresourceRange {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
        };
        barrier.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VkAccessFlags {}
            : VkAccessFlags(VK_ACCESS_MEMORY_READ_BIT);
        barrier.dstAccessMask = new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            ? VkAccessFlags(VK_ACCESS_SHADER_READ_BIT)
            : VkAccessFlags(VK_ACCESS_TRANSFER_WRITE_BIT);
        cmd.PipelineBarrier(old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                                    : VK_PIPELINE_STAGE_TRANSFER_BIT,
                            new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                : VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            barrier);

        result = cmd.End();
        if (result != VK_SUCCESS) break;

        VkSubmitInfo submit {
            .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext              = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers    = cmd.address(),
        };
        result = queue.Submit(submit);
    } while (false);
    return result;
}

std::optional<uint32_t> FindMemoryType(const vvk::PhysicalDevice& gpu,
                                       uint32_t memory_type_bits,
                                       VkMemoryPropertyFlags preferred_flags) {
    const auto memory_properties = gpu.GetMemoryProperties().memoryProperties;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((memory_type_bits & (1u << i)) == 0) continue;
        if ((memory_properties.memoryTypes[i].propertyFlags & preferred_flags) == preferred_flags) {
            return i;
        }
    }
    return std::nullopt;
}

struct GstCudaMemoryPrefix {
    GstMemory mem;
    GstCudaContext* context;
    GstVideoInfo info;
};

GstCudaContext* GetCudaMemoryContext(GstMemory* memory) {
    if (memory == nullptr || ! gst_is_cuda_memory(memory)) return nullptr;
    return reinterpret_cast<GstCudaMemoryPrefix*>(memory)->context;
}

struct CudaExternalBuffer {
    VkDevice device { VK_NULL_HANDLE };
    const vvk::DeviceDispatch* dispatch { nullptr };
    VkBuffer buffer { VK_NULL_HANDLE };
    VkDeviceMemory memory { VK_NULL_HANDLE };
    VkDeviceSize size { 0 };
    int fd { -1 };
    CUexternalMemory cuda_memory {};
    CUdeviceptr cuda_ptr {};
    GstCudaContext* cuda_context { nullptr };

    CudaExternalBuffer() = default;
    CudaExternalBuffer(const CudaExternalBuffer&) = delete;
    CudaExternalBuffer& operator=(const CudaExternalBuffer&) = delete;

    CudaExternalBuffer(CudaExternalBuffer&& other) noexcept {
        *this = std::move(other);
    }

    CudaExternalBuffer& operator=(CudaExternalBuffer&& other) noexcept {
        if (this == &other) return *this;
        reset();
        device = std::exchange(other.device, VK_NULL_HANDLE);
        dispatch = std::exchange(other.dispatch, nullptr);
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        memory = std::exchange(other.memory, VK_NULL_HANDLE);
        size = std::exchange(other.size, 0);
        fd = std::exchange(other.fd, -1);
        cuda_memory = std::exchange(other.cuda_memory, nullptr);
        cuda_ptr = std::exchange(other.cuda_ptr, 0);
        cuda_context = std::exchange(other.cuda_context, nullptr);
        return *this;
    }

    ~CudaExternalBuffer() { reset(); }

    explicit operator bool() const { return buffer != VK_NULL_HANDLE && cuda_ptr != 0; }

    void reset() {
        if (cuda_memory != nullptr) {
            bool pushed = false;
            if (cuda_context != nullptr) pushed = gst_cuda_context_push(cuda_context);
            (void)CuDestroyExternalMemory(cuda_memory);
            if (pushed) {
                CUcontext popped {};
                (void)gst_cuda_context_pop(&popped);
            }
            cuda_memory = nullptr;
            cuda_ptr = 0;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (dispatch != nullptr && device != VK_NULL_HANDLE) {
            if (buffer != VK_NULL_HANDLE) {
                dispatch->vkDestroyBuffer(device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
            }
            if (memory != VK_NULL_HANDLE) {
                dispatch->vkFreeMemory(device, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
        }
        if (cuda_context != nullptr) {
            gst_object_unref(cuda_context);
            cuda_context = nullptr;
        }
        device = VK_NULL_HANDLE;
        dispatch = nullptr;
        size = 0;
    }
};

std::optional<CudaExternalBuffer> CreateCudaExternalTransferBuffer(const Device& device,
                                                                   VkDeviceSize size,
                                                                   GstCudaContext* cuda_context) {
    if (size == 0 || cuda_context == nullptr) return std::nullopt;
    if (device.handle().Dispatch().vkGetMemoryFdKHR == nullptr) {
        LOG_ERROR("video texture: Vulkan device has no vkGetMemoryFdKHR for CUDA interop");
        return std::nullopt;
    }

    CudaExternalBuffer result;
    result.device = *device.handle();
    result.dispatch = &device.handle().Dispatch();
    result.size = size;
    result.cuda_context = reinterpret_cast<GstCudaContext*>(gst_object_ref(cuda_context));

    VkExternalMemoryBufferCreateInfo external_buffer {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkBufferCreateInfo buffer_info {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_buffer,
        .flags = 0,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (result.dispatch->vkCreateBuffer(result.device, &buffer_info, nullptr, &result.buffer) != VK_SUCCESS) {
        LOG_ERROR("video texture: create Vulkan external CUDA buffer failed");
        return std::nullopt;
    }

    VkBufferMemoryRequirementsInfo2 req_info {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = nullptr,
        .buffer = result.buffer,
    };
    VkMemoryRequirements2 reqs {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = nullptr,
    };
    result.dispatch->vkGetBufferMemoryRequirements2(result.device, &req_info, &reqs);

    const auto memory_type =
        FindMemoryType(device.gpu(), reqs.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (! memory_type.has_value()) {
        LOG_ERROR("video texture: no device-local memory type for CUDA interop buffer");
        return std::nullopt;
    }

    VkExportMemoryAllocateInfo export_info {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkMemoryAllocateInfo alloc_info {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &export_info,
        .allocationSize = reqs.memoryRequirements.size,
        .memoryTypeIndex = memory_type.value(),
    };
    if (result.dispatch->vkAllocateMemory(result.device, &alloc_info, nullptr, &result.memory) != VK_SUCCESS ||
        result.dispatch->vkBindBufferMemory(result.device, result.buffer, result.memory, 0) != VK_SUCCESS) {
        LOG_ERROR("video texture: allocate/bind Vulkan external CUDA buffer failed");
        return std::nullopt;
    }

    VkMemoryGetFdInfoKHR fd_info {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext = nullptr,
        .memory = result.memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    if (result.dispatch->vkGetMemoryFdKHR(result.device, &fd_info, &result.fd) != VK_SUCCESS ||
        result.fd < 0) {
        LOG_ERROR("video texture: export Vulkan buffer fd for CUDA failed");
        return std::nullopt;
    }

    bool pushed = gst_cuda_context_push(cuda_context);
    if (! pushed) {
        LOG_ERROR("video texture: push CUDA context for Vulkan external buffer failed");
        return std::nullopt;
    }

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC import_desc {};
    import_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    import_desc.handle.fd = result.fd;
    import_desc.size = static_cast<unsigned long long>(reqs.memoryRequirements.size);
    CUresult cuda_result = CuImportExternalMemory(&result.cuda_memory, &import_desc);
    if (cuda_result == CUDA_SUCCESS) result.fd = -1;

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc {};
    buffer_desc.offset = 0;
    buffer_desc.size = static_cast<unsigned long long>(size);
    if (cuda_result == CUDA_SUCCESS) {
        cuda_result = CuExternalMemoryGetMappedBuffer(&result.cuda_ptr,
                                                      result.cuda_memory,
                                                      &buffer_desc);
    }
    CUcontext popped {};
    (void)gst_cuda_context_pop(&popped);

    if (cuda_result != CUDA_SUCCESS || result.cuda_ptr == 0) {
        LOG_ERROR("video texture: import Vulkan buffer into CUDA failed result=%s",
                  CudaErrorName(cuda_result));
        return std::nullopt;
    }
    return result;
}

struct DmabufImportedImage {
    VkDevice device { VK_NULL_HANDLE };
    const vvk::DeviceDispatch* dispatch { nullptr };
    VkImage image { VK_NULL_HANDLE };
    VkDeviceMemory memory { VK_NULL_HANDLE };
    VkFormat format { VK_FORMAT_UNDEFINED };
    uint32_t width { 0 };
    uint32_t height { 0 };

    DmabufImportedImage() = default;
    DmabufImportedImage(const DmabufImportedImage&) = delete;
    DmabufImportedImage& operator=(const DmabufImportedImage&) = delete;

    DmabufImportedImage(DmabufImportedImage&& other) noexcept {
        *this = std::move(other);
    }

    DmabufImportedImage& operator=(DmabufImportedImage&& other) noexcept {
        if (this == &other) return *this;
        reset();
        device = std::exchange(other.device, VK_NULL_HANDLE);
        dispatch = std::exchange(other.dispatch, nullptr);
        image = std::exchange(other.image, VK_NULL_HANDLE);
        memory = std::exchange(other.memory, VK_NULL_HANDLE);
        format = std::exchange(other.format, VK_FORMAT_UNDEFINED);
        width = std::exchange(other.width, 0);
        height = std::exchange(other.height, 0);
        return *this;
    }

    ~DmabufImportedImage() { reset(); }

    explicit operator bool() const { return image != VK_NULL_HANDLE; }

    void reset() {
        if (dispatch != nullptr && device != VK_NULL_HANDLE) {
            if (image != VK_NULL_HANDLE) {
                dispatch->vkDestroyImage(device, image, nullptr);
                image = VK_NULL_HANDLE;
            }
            if (memory != VK_NULL_HANDLE) {
                dispatch->vkFreeMemory(device, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
        }
        device = VK_NULL_HANDLE;
        dispatch = nullptr;
        format = VK_FORMAT_UNDEFINED;
        width = 0;
        height = 0;
    }
};

std::optional<off_t> QueryFdSize(int fd) {
    if (fd < 0) return std::nullopt;
    const off_t current = ::lseek(fd, 0, SEEK_CUR);
    const off_t end = ::lseek(fd, 0, SEEK_END);
    if (current >= 0) (void)::lseek(fd, current, SEEK_SET);
    if (end <= 0) return std::nullopt;
    return end;
}

bool ParseVaRgbaDmabufCaps(GstCaps* caps,
                           GstVideoInfoDmaDrm& drm_info,
                           VkFormat& vk_format,
                           uint64_t& drm_modifier) {
    if (caps == nullptr || ! gst_video_is_dma_drm_caps(caps)) return false;
    gst_video_info_dma_drm_init(&drm_info);
    if (! gst_video_info_dma_drm_from_caps(&drm_info, caps)) return false;

    const GstVideoFormat gst_format = gst_video_dma_drm_fourcc_to_format(drm_info.drm_fourcc);
    if (gst_format != GST_VIDEO_FORMAT_RGBA) return false;

    // GStreamer's DMA_DRM "ABGR8888" fourcc describes little-endian RGBA byte order.
    // Import only that layout so the Vulkan image copy preserves channel order without
    // a CPU swizzle or an extra shader pass.
    if (drm_info.drm_fourcc != DRM_FORMAT_ABGR8888) return false;

    vk_format = VK_FORMAT_R8G8B8A8_UNORM;
    drm_modifier = drm_info.drm_modifier == DRM_FORMAT_MOD_INVALID
        ? static_cast<uint64_t>(DRM_FORMAT_MOD_LINEAR)
        : drm_info.drm_modifier;
    return true;
}

std::string DrmFourccToString(uint32_t fourcc) {
    char text[5] {
        static_cast<char>(fourcc & 0xff),
        static_cast<char>((fourcc >> 8) & 0xff),
        static_cast<char>((fourcc >> 16) & 0xff),
        static_cast<char>((fourcc >> 24) & 0xff),
        0,
    };
    return text;
}

std::optional<VkFormat> VkFormatForRgbaDrmFourcc(uint32_t drm_format) {
    if (drm_format == DRM_FORMAT_ABGR8888) return VK_FORMAT_R8G8B8A8_UNORM;
    if (drm_format == DRM_FORMAT_ARGB8888) return VK_FORMAT_B8G8R8A8_UNORM;
    return std::nullopt;
}

std::optional<DmabufImportedImage> ImportSinglePlaneDrmRgbaImage(const Device& device,
                                                                 uint32_t width,
                                                                 uint32_t height,
                                                                 VkFormat vk_format,
                                                                 uint64_t drm_modifier,
                                                                 int fd,
                                                                 uint32_t object_size,
                                                                 uint32_t plane_offset,
                                                                 uint32_t plane_stride) {
    if (width == 0 || height == 0 || fd < 0 || plane_stride == 0) return std::nullopt;
    if (drm_modifier == DRM_FORMAT_MOD_INVALID) drm_modifier = DRM_FORMAT_MOD_LINEAR;

    DmabufImportedImage result;
    result.device = *device.handle();
    result.dispatch = &device.handle().Dispatch();
    result.format = vk_format;
    result.width = width;
    result.height = height;

    VkExternalMemoryImageCreateInfo external_image {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkSubresourceLayout plane_layout {
        .offset = static_cast<VkDeviceSize>(plane_offset),
        .size = static_cast<VkDeviceSize>(plane_stride) * height,
        .rowPitch = static_cast<VkDeviceSize>(plane_stride),
        .arrayPitch = 0,
        .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT modifier_info {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .pNext = &external_image,
        .drmFormatModifier = drm_modifier,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &plane_layout,
    };
    VkImageCreateInfo image_info {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &modifier_info,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = vk_format,
        .extent = VkExtent3D { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (result.dispatch->vkCreateImage(result.device, &image_info, nullptr, &result.image) != VK_SUCCESS) {
        return std::nullopt;
    }

    VkMemoryRequirements reqs {};
    result.dispatch->vkGetImageMemoryRequirements(result.device, result.image, &reqs);
    const auto memory_type = FindMemoryType(device.gpu(), reqs.memoryTypeBits, {});
    if (! memory_type.has_value()) {
        LOG_ERROR("video texture: no Vulkan memory type accepts VA DMABuf import");
        return std::nullopt;
    }

    const int import_fd = ::dup(fd);
    if (import_fd < 0) {
        LOG_ERROR("video texture: dup VA DMABuf fd failed errno=%d message=%s",
                  errno,
                  std::strerror(errno));
        return std::nullopt;
    }

    VkMemoryDedicatedAllocateInfo dedicated_info {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = nullptr,
        .image = result.image,
        .buffer = VK_NULL_HANDLE,
    };
    VkImportMemoryFdInfoKHR import_info {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = &dedicated_info,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = import_fd,
    };
    VkMemoryAllocateInfo alloc_info {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = std::max(reqs.size, static_cast<VkDeviceSize>(object_size)),
        .memoryTypeIndex = memory_type.value(),
    };
    if (result.dispatch->vkAllocateMemory(result.device, &alloc_info, nullptr, &result.memory) !=
        VK_SUCCESS) {
        ::close(import_fd);
        LOG_ERROR("video texture: import VA DMABuf into Vulkan memory failed");
        return std::nullopt;
    }
    if (result.dispatch->vkBindImageMemory(result.device, result.image, result.memory, 0) !=
        VK_SUCCESS) {
        LOG_ERROR("video texture: bind imported VA DMABuf image memory failed");
        return std::nullopt;
    }

    return result;
}

std::optional<DmabufImportedImage> ImportVaDmabufRgbaImage(const Device& device,
                                                           GstCaps* caps,
                                                           GstBuffer* buffer) {
    if (caps == nullptr || buffer == nullptr || gst_buffer_n_memory(buffer) != 1) {
        return std::nullopt;
    }
    GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
    if (memory == nullptr || ! gst_is_dmabuf_memory(memory)) {
        return std::nullopt;
    }

    GstVideoInfoDmaDrm drm_info;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    uint64_t drm_modifier = DRM_FORMAT_MOD_INVALID;
    if (! ParseVaRgbaDmabufCaps(caps, drm_info, vk_format, drm_modifier)) {
        return std::nullopt;
    }

    gsize memory_offset = 0;
    gsize memory_size = 0;
    gsize max_memory_size = gst_memory_get_sizes(memory, &memory_offset, &memory_size);
    const int fd = gst_dmabuf_memory_get_fd(memory);
    if (fd < 0 || max_memory_size == 0) return std::nullopt;

    GstVideoMeta* meta = gst_buffer_get_video_meta(buffer);
    const gsize plane_offset = meta != nullptr && meta->n_planes > 0
        ? static_cast<gsize>(meta->offset[0])
        : static_cast<gsize>(GST_VIDEO_INFO_PLANE_OFFSET(&drm_info.vinfo, 0));
    const gint plane_stride = meta != nullptr && meta->n_planes > 0
        ? meta->stride[0]
        : GST_VIDEO_INFO_PLANE_STRIDE(&drm_info.vinfo, 0);
    if (plane_stride <= 0) return std::nullopt;
    const gint width = GST_VIDEO_INFO_WIDTH(&drm_info.vinfo);
    const gint height = GST_VIDEO_INFO_HEIGHT(&drm_info.vinfo);
    if (width <= 0 || height <= 0) return std::nullopt;

    return ImportSinglePlaneDrmRgbaImage(device,
                                         static_cast<uint32_t>(width),
                                         static_cast<uint32_t>(height),
                                         vk_format,
                                         drm_modifier,
                                         fd,
                                         static_cast<uint32_t>(
                                             QueryFdSize(fd).value_or(static_cast<off_t>(max_memory_size))),
                                         static_cast<uint32_t>(memory_offset + plane_offset),
                                         static_cast<uint32_t>(plane_stride));
}

void CloseVaDescriptorFds(VADRMPRIMESurfaceDescriptor& descriptor) {
    for (uint32_t i = 0; i < descriptor.num_objects && i < 4; ++i) {
        if (descriptor.objects[i].fd >= 0) {
            ::close(descriptor.objects[i].fd);
            descriptor.objects[i].fd = -1;
        }
    }
}

std::optional<DmabufImportedImage> ExportVaMemoryRgbaImage(const Device& device,
                                                           GstCaps* caps,
                                                           GstBuffer* buffer) {
    if (caps == nullptr || buffer == nullptr || gst_buffer_n_memory(buffer) == 0) {
        return std::nullopt;
    }

    GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
    if (memory == nullptr || ! gst_memory_is_type(memory, GST_ALLOCATOR_VASURFACE)) {
        return std::nullopt;
    }

    GstVideoInfo info;
    if (! gst_video_info_from_caps(&info, caps) ||
        GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_BGRA) {
        return std::nullopt;
    }

    GstVaDisplay* display = gst_va_memory_peek_display(memory);
    VASurfaceID surface = gst_va_memory_get_surface(memory);
    if (display == nullptr) display = gst_va_buffer_peek_display(buffer);
    if (surface == VA_INVALID_SURFACE) surface = gst_va_buffer_get_surface(buffer);
    if (display == nullptr || surface == VA_INVALID_SURFACE) return std::nullopt;

    auto* va_display = reinterpret_cast<VADisplay>(gst_va_display_get_va_dpy(display));
    if (va_display == nullptr) return std::nullopt;

    VAStatus status = vaSyncSurface(va_display, surface);
    if (status != VA_STATUS_SUCCESS) {
        LOG_ERROR("video texture: vaSyncSurface failed status=%s", vaErrorStr(status));
        return std::nullopt;
    }

    VADRMPRIMESurfaceDescriptor descriptor {};
    for (auto& object : descriptor.objects) object.fd = -1;
    // GStreamer can decode and post-process into VAMemory on drivers where
    // directly negotiating memory:DMABuf fails. Exporting the ready VA surface
    // here keeps the frame on the GPU while giving Vulkan the dma-buf metadata
    // it needs for the existing image-copy path.
    status = vaExportSurfaceHandle(va_display,
                                   surface,
                                   VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                                   VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                                   &descriptor);
    if (status != VA_STATUS_SUCCESS) {
        LOG_ERROR("video texture: vaExportSurfaceHandle failed status=%s", vaErrorStr(status));
        return std::nullopt;
    }

    std::optional<DmabufImportedImage> imported;
    if (descriptor.num_layers == 1 && descriptor.layers[0].num_planes == 1 &&
        descriptor.num_objects > 0 && descriptor.layers[0].object_index[0] < descriptor.num_objects) {
        const auto& layer = descriptor.layers[0];
        const auto& object = descriptor.objects[layer.object_index[0]];
        if (auto vk_format = VkFormatForRgbaDrmFourcc(layer.drm_format); vk_format.has_value()) {
            imported = ImportSinglePlaneDrmRgbaImage(device,
                                                     descriptor.width,
                                                     descriptor.height,
                                                     vk_format.value(),
                                                     object.drm_format_modifier,
                                                     object.fd,
                                                     object.size,
                                                     layer.offset[0],
                                                     layer.pitch[0]);
        }
    }

    if (! imported.has_value()) {
        LOG_ERROR("video texture: unsupported VA exported surface fourcc=%s layers=%u objects=%u "
                  "layer0-format=%s layer0-planes=%u",
                  DrmFourccToString(descriptor.fourcc).c_str(),
                  descriptor.num_layers,
                  descriptor.num_objects,
                  descriptor.num_layers > 0
                      ? DrmFourccToString(descriptor.layers[0].drm_format).c_str()
                      : "none",
                  descriptor.num_layers > 0 ? descriptor.layers[0].num_planes : 0);
    }

    CloseVaDescriptorFds(descriptor);
    return imported;
}

} // namespace

struct VideoTextureCache::Entry {
    std::string key;
    std::vector<uint8_t> encoded;
    VmaImageParameters image;
    VmaBufferParameters staging;
    CudaExternalBuffer cuda_rgba_buffer;
    DmabufImportedImage va_imported_image;
    uint8_t* staging_mapped { nullptr };
    uint32_t width { 0 };
    uint32_t height { 0 };
    bool     dirty { false };
    bool     cuda_rgba_dirty { false };
    bool     va_dmabuf_dirty { false };
    bool     warned_size_mismatch { false };
    bool     warned_unexpected_format { false };
    bool     warned_cuda_interop_failed { false };
    bool     warned_va_dmabuf_import_failed { false };
    bool     pipeline_failed { false };
    VideoTextureTransferPath transfer_path { VideoTextureTransferPath::None };
    bool     paused { false };
    bool     stopped { false };
    bool     eos_loop_waiting_for_sample { false };
    bool     eos_loop_rebuild_attempted { false };
    uint64_t eos_loop_count { 0 };
    uint64_t uploaded_sample_count { 0 };
    CUmodule cuda_module {};
    CUfunction cuda_kernel {};
    std::string decoder_factory;
    std::string postproc_factory;
    int expected_cuda_device_id { -1 };

    GstElement* pipeline { nullptr };
    GstElement* source_elem { nullptr };
    GstElement* appsink_elem { nullptr };
    GstBus*     bus { nullptr };
    GInputStream* memory_stream { nullptr };

    ~Entry() {
        if (cuda_module != nullptr) {
            bool pushed = false;
            if (cuda_rgba_buffer.cuda_context != nullptr) {
                pushed = gst_cuda_context_push(cuda_rgba_buffer.cuda_context);
            }
            (void)CuModuleUnload(cuda_module);
            if (pushed) {
                CUcontext popped {};
                (void)gst_cuda_context_pop(&popped);
            }
            cuda_module = nullptr;
            cuda_kernel = nullptr;
        }
        if (staging_mapped != nullptr) staging.handle.UnMapMemory();
        if (pipeline != nullptr) (void)gst_element_set_state(pipeline, GST_STATE_NULL);
        if (bus != nullptr) gst_object_unref(bus);
        if (appsink_elem != nullptr) gst_object_unref(appsink_elem);
        if (source_elem != nullptr) gst_object_unref(source_elem);
        if (pipeline != nullptr) gst_object_unref(pipeline);
        if (memory_stream != nullptr) g_object_unref(memory_stream);
    }
};

VideoTextureCache::VideoTextureCache(const Device& device, VideoTextureDecoderSettings settings)
    : m_device(device), m_settings(settings) {
    if (! gst_is_initialized()) gst_init(nullptr, nullptr);
    ConfigureDecoderRanks(MakeDecoderSettings(m_settings));
}

VideoTextureCache::~VideoTextureCache() = default;

VideoTextureCache::Entry* VideoTextureCache::find(std::string_view key) {
    const auto it =
        std::find_if(m_entries.begin(), m_entries.end(), [key](const auto& entry) {
            return entry != nullptr && entry->key == key;
        });
    return it == m_entries.end() ? nullptr : it->get();
}

const VideoTextureCache::Entry* VideoTextureCache::find(std::string_view key) const {
    const auto it =
        std::find_if(m_entries.begin(), m_entries.end(), [key](const auto& entry) {
            return entry != nullptr && entry->key == key;
        });
    return it == m_entries.end() ? nullptr : it->get();
}

void VideoTextureCache::allocateCmd() {
    const auto& pool = m_device.cmd_pool();
    VVK_CHECK(pool.Allocate(1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, m_cmds));
    m_cmd = vvk::CommandBuffer(m_cmds[0], m_device.handle().Dispatch());
}

void VideoTextureCache::stopPipeline(Entry& entry) {
    if (entry.pipeline != nullptr) (void)gst_element_set_state(entry.pipeline, GST_STATE_NULL);
    if (entry.bus != nullptr) gst_object_unref(entry.bus);
    if (entry.appsink_elem != nullptr) gst_object_unref(entry.appsink_elem);
    if (entry.source_elem != nullptr) gst_object_unref(entry.source_elem);
    if (entry.pipeline != nullptr) gst_object_unref(entry.pipeline);
    if (entry.memory_stream != nullptr) g_object_unref(entry.memory_stream);
    entry.bus = nullptr;
    entry.appsink_elem = nullptr;
    entry.source_elem = nullptr;
    entry.pipeline = nullptr;
    entry.memory_stream = nullptr;
}

bool VideoTextureCache::startPipeline(Entry& entry) {
    stopPipeline(entry);
    entry.eos_loop_waiting_for_sample = false;
    entry.eos_loop_rebuild_attempted = false;
    entry.pipeline_failed = false;

    GError* error = nullptr;
    const auto pipeline_config =
        BuildVideoTextureDecodeGraph(entry.transfer_path,
                                     entry.decoder_factory,
                                     entry.postproc_factory);
    if (!pipeline_config.has_value()) {
        LOG_ERROR("video texture '%s': no GPU decode graph for transfer-path=%s render-node=%s",
                  entry.key.c_str(),
                  VideoTextureTransferPathName(entry.transfer_path),
                  m_settings.render_node.empty() ? "(unresolved)" : m_settings.render_node.c_str());
        return false;
    }
    entry.pipeline = gst_parse_launch(pipeline_config->description.c_str(), &error);
    if (entry.pipeline == nullptr) {
        LOG_ERROR("create video pipeline for '%s' failed: %s",
                  entry.key.c_str(),
                  error != nullptr ? error->message : "unknown error");
        if (error != nullptr) g_error_free(error);
        return false;
    }
    if (error != nullptr) {
        LOG_ERROR("create video pipeline for '%s' failed: %s", entry.key.c_str(), error->message);
        g_error_free(error);
        stopPipeline(entry);
        return false;
    }

    entry.memory_stream =
        G_INPUT_STREAM(g_memory_input_stream_new_from_data(entry.encoded.data(),
                                                           static_cast<gssize>(entry.encoded.size()),
                                                           nullptr));
    if (entry.memory_stream == nullptr) {
        LOG_ERROR("create video memory stream for '%s' failed", entry.key.c_str());
        stopPipeline(entry);
        return false;
    }

    entry.source_elem = gst_bin_get_by_name(GST_BIN(entry.pipeline), "src");
    entry.appsink_elem = gst_bin_get_by_name(GST_BIN(entry.pipeline), "sink");
    entry.bus = gst_element_get_bus(entry.pipeline);
    if (entry.source_elem == nullptr || entry.appsink_elem == nullptr || entry.bus == nullptr) {
        LOG_ERROR("video texture '%s' pipeline is missing giostreamsrc/appsink", entry.key.c_str());
        stopPipeline(entry);
        return false;
    }
    if (entry.expected_cuda_device_id >= 0) {
        GstElement* decoder = gst_bin_get_by_name(GST_BIN(entry.pipeline),
                                                  "video_texture_decoder");
        const bool matches =
            DecoderElementMatchesExpectedCudaDevice(decoder,
                                                    entry.expected_cuda_device_id);
        if (decoder != nullptr) gst_object_unref(decoder);
        if (! matches) {
            LOG_ERROR("video texture '%s': NVIDIA decoder is not on selected "
                      "render-node=%s; refusing cross-GPU import",
                      entry.key.c_str(),
                      m_settings.render_node.empty() ? "(unresolved)" : m_settings.render_node.c_str());
            stopPipeline(entry);
            return false;
        }
    }

    g_object_set(entry.source_elem, "stream", entry.memory_stream, nullptr);
    LOG_INFO("VideoTextureDecoderGraph key='%s' backend='%s' source=giostreamsrc bytes=%zu desc='%s'",
             entry.key.c_str(),
             pipeline_config->name,
             entry.encoded.size(),
             pipeline_config->description.c_str());

    GstCaps* sink_caps = gst_caps_from_string(pipeline_config->sink_caps);
    g_object_set(entry.appsink_elem, "caps", sink_caps, nullptr);
    gst_caps_unref(sink_caps);

    if (gst_element_set_state(entry.pipeline, GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("start video pipeline preroll for '%s' failed", entry.key.c_str());
        stopPipeline(entry);
        return false;
    }
    (void)gst_element_get_state(entry.pipeline, nullptr, nullptr, GST_SECOND);
    if (entry.appsink_elem != nullptr) {
        GstSample* preroll = gst_app_sink_try_pull_preroll(GST_APP_SINK(entry.appsink_elem), GST_SECOND);
        if (preroll != nullptr) {
            (void)uploadSample(entry, preroll);
            gst_sample_unref(preroll);
        }
    }

    const auto target_state = m_globally_paused ? GST_STATE_PAUSED : GST_STATE_PLAYING;
    if (gst_element_set_state(entry.pipeline, target_state) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("start video pipeline playback for '%s' failed", entry.key.c_str());
        stopPipeline(entry);
        return false;
    }
    entry.paused = false;
    entry.stopped = false;
    return true;
}

bool VideoTextureCache::restartPipeline(Entry& entry) {
    const bool was_paused = entry.paused;
    if (!startPipeline(entry)) {
        LOG_ERROR("video texture '%s': failed to rebuild pipeline for loop", entry.key.c_str());
        entry.pipeline_failed = true;
        return false;
    }
    if (was_paused) setPaused(entry, true);
    return true;
}

bool VideoTextureCache::loopPipeline(Entry& entry) {
    if (entry.pipeline == nullptr || entry.stopped) return false;

    if (entry.eos_loop_waiting_for_sample) {
        if (entry.eos_loop_rebuild_attempted) {
            LOG_ERROR("video texture '%s': EOS loop seek still produced no sample after rebuild",
                      entry.key.c_str());
            entry.pipeline_failed = true;
            return false;
        }

        // If EOS arrives again before a decoded frame proves that the flush seek worked, keep the
        // render resource stable but rebuild the decoder graph once. Repeating this indefinitely
        // would hide the real failure behind an EOS storm and could stall future texture updates.
        LOG_ERROR("video texture '%s': EOS loop seek produced no sample; restarting pipeline once",
                  entry.key.c_str());
        const uint64_t uploaded_before_restart = entry.uploaded_sample_count;
        if (!restartPipeline(entry)) return false;
        if (entry.uploaded_sample_count == uploaded_before_restart) {
            entry.eos_loop_waiting_for_sample = true;
            entry.eos_loop_rebuild_attempted = true;
        }
        return true;
    }

    entry.eos_loop_count++;
    const auto start = std::chrono::steady_clock::now();
    if (gst_element_set_state(entry.pipeline, GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("video texture '%s': failed to pause pipeline for EOS loop seek",
                  entry.key.c_str());
        entry.pipeline_failed = true;
        return false;
    }
    const auto paused = std::chrono::steady_clock::now();

    // Looping by seek keeps decoder state, negotiated caps, appsink ownership, and Vulkan texture
    // bindings stable. Rebuilding here can force shader/resource refresh at the exact frame where
    // a wallpaper loop is most noticeable.
    const gboolean seek_ok = gst_element_seek(
        entry.pipeline,
        1.0,
        GST_FORMAT_TIME,
        static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
        GST_SEEK_TYPE_SET,
        0,
        GST_SEEK_TYPE_NONE,
        -1);
    const auto seeked = std::chrono::steady_clock::now();
    if (! seek_ok) {
        LOG_ERROR("VideoTexturePerf eos-loop-seek-failed key='%s' loop=%llu paused_ms=%lld seek_ms=%lld",
                  entry.key.c_str(),
                  static_cast<unsigned long long>(entry.eos_loop_count),
                  ElapsedMillis(start, paused),
                  ElapsedMillis(paused, seeked));
        return restartPipeline(entry);
    }

    entry.eos_loop_waiting_for_sample = ! entry.paused && ! m_globally_paused;
    const auto target_state =
        (entry.paused || m_globally_paused) ? GST_STATE_PAUSED : GST_STATE_PLAYING;
    if (gst_element_set_state(entry.pipeline, target_state) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("video texture '%s': failed to resume pipeline after EOS loop seek",
                  entry.key.c_str());
        entry.pipeline_failed = true;
        return false;
    }
    const auto playing = std::chrono::steady_clock::now();
    LOG_INFO("VideoTexturePerf eos-loop-seek key='%s' loop=%llu paused_ms=%lld seek_ms=%lld play_ms=%lld",
             entry.key.c_str(),
             static_cast<unsigned long long>(entry.eos_loop_count),
             ElapsedMillis(start, paused),
             ElapsedMillis(paused, seeked),
             ElapsedMillis(seeked, playing));
    return true;
}

bool VideoTextureCache::applyPipelinePlaybackState(Entry& entry) {
    if (entry.pipeline == nullptr || entry.stopped) return true;

    const auto target_state =
        (entry.paused || m_globally_paused) ? GST_STATE_PAUSED : GST_STATE_PLAYING;
    if (gst_element_set_state(entry.pipeline, target_state) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("video texture '%s': failed to switch playback state paused=%s global_paused=%s",
                  entry.key.c_str(),
                  entry.paused ? "true" : "false",
                  m_globally_paused ? "true" : "false");
        entry.pipeline_failed = true;
        return false;
    }
    return true;
}

bool VideoTextureCache::setPaused(Entry& entry, bool paused) {
    if (! paused && entry.pipeline == nullptr) {
        return startPipeline(entry);
    }
    if (entry.paused == paused) return true;
    if (entry.pipeline == nullptr) {
        entry.paused = paused;
        entry.stopped = false;
        return true;
    }

    entry.paused = paused;
    entry.stopped = false;
    return applyPipelinePlaybackState(entry);
}

bool VideoTextureCache::stopPlayback(Entry& entry) {
    if (entry.stopped && entry.pipeline == nullptr) return true;

    // stop() is intentionally stronger than pause(): authored intro videos call it after their
    // fade-out and expect decoder, colorspace conversion, appsink polling, and Vulkan upload work
    // to end. Keep the already-created Vulkan image alive for descriptor stability, but release the
    // GStreamer graph so future frames are not decoded until play() rebuilds the pipeline.
    stopPipeline(entry);
    entry.paused = true;
    entry.stopped = true;
    entry.pipeline_failed = false;
    entry.eos_loop_waiting_for_sample = false;
    entry.eos_loop_rebuild_attempted = false;
    return true;
}

bool VideoTextureCache::seekTo(Entry& entry, double seconds) {
    if (entry.pipeline == nullptr) return false;

    const double clamped_seconds = std::max(0.0, seconds);
    const auto   target_time =
        static_cast<gint64>(clamped_seconds * static_cast<double>(GST_SECOND));

    // SceneScript setCurrentTime() is a visible decoder command: use a flushing accurate seek so
    // stale frames already queued in appsink are discarded before the next render poll observes the
    // texture. This keeps one-shot intro videos deterministic when scripts stop, rewind, then wait
    // for a later play() call.
    if (! gst_element_seek_simple(
            entry.pipeline,
            GST_FORMAT_TIME,
            static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE),
            target_time)) {
        LOG_ERROR("video texture '%s': failed to seek to %.3fs",
                  entry.key.c_str(),
                  clamped_seconds);
        return false;
    }

    if (entry.appsink_elem != nullptr) {
        while (GstSample* stale =
                   gst_app_sink_try_pull_sample(GST_APP_SINK(entry.appsink_elem), 0)) {
            gst_sample_unref(stale);
        }
    }

    entry.eos_loop_waiting_for_sample = false;
    entry.eos_loop_rebuild_attempted = false;
    LOG_INFO("video texture '%s': seek accepted seconds=%.3f", entry.key.c_str(), clamped_seconds);
    return true;
}

bool VideoTextureCache::uploadSample(VideoTextureCache::Entry& entry, ::GstSample* sample) {
    if (sample == nullptr) return false;

    GstCaps* caps = gst_sample_get_caps(sample);
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstVideoInfo info;
    if (caps == nullptr || buffer == nullptr || ! gst_video_info_from_caps(&info, caps)) {
        return false;
    }

    if (entry.transfer_path == VideoTextureTransferPath::VaMemoryBgra) {
        auto imported = ImportVaDmabufRgbaImage(m_device, caps, buffer);
        if (! imported.has_value()) imported = ExportVaMemoryRgbaImage(m_device, caps, buffer);
        if (imported.has_value()) {
            const auto copy_width = std::min<uint32_t>(entry.width, imported->width);
            const auto copy_height = std::min<uint32_t>(entry.height, imported->height);
            if ((copy_width != entry.width || copy_height != entry.height) &&
                ! entry.warned_size_mismatch) {
                LOG_INFO("video texture '%s' VA decoded size %ux%u differs from texture size %ux%u",
                         entry.key.c_str(),
                         imported->width,
                         imported->height,
                         entry.width,
                         entry.height);
                entry.warned_size_mismatch = true;
            }
            entry.va_imported_image = std::move(imported.value());
            entry.va_dmabuf_dirty = true;
            entry.uploaded_sample_count++;
            entry.eos_loop_waiting_for_sample = false;
            entry.eos_loop_rebuild_attempted = false;
            return true;
        }

        if (! entry.warned_va_dmabuf_import_failed) {
            gchar* caps_text = gst_caps_to_string(caps);
            LOG_ERROR("video texture '%s': VA surface export/import failed caps='%s' memories=%u",
                      entry.key.c_str(),
                      caps_text != nullptr ? caps_text : "unknown",
                      gst_buffer_n_memory(buffer));
            if (caps_text != nullptr) g_free(caps_text);
            entry.warned_va_dmabuf_import_failed = true;
        }
        return false;
    }

    if (IsNvidiaCudaPath(entry.transfer_path) &&
        GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_NV12 &&
        gst_buffer_n_memory(buffer) > 0 &&
        gst_is_cuda_memory(gst_buffer_peek_memory(buffer, 0))) {
        GstVideoFrame frame;
        if (! gst_video_frame_map(&frame, &info, buffer, kGstMapReadCuda)) {
            if (! entry.warned_cuda_interop_failed) {
                LOG_ERROR("video texture '%s': CUDA frame map failed", entry.key.c_str());
                entry.warned_cuda_interop_failed = true;
            }
            return false;
        }

        GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
        auto* cuda_context = GetCudaMemoryContext(memory);
        if (cuda_context == nullptr) {
            gst_video_frame_unmap(&frame);
            LOG_ERROR("video texture '%s': CUDA memory has no context", entry.key.c_str());
            return false;
        }
        if (! CudaContextMatchesExpected(cuda_context,
                                         entry.expected_cuda_device_id,
                                         entry.key)) {
            gst_video_frame_unmap(&frame);
            entry.pipeline_failed = true;
            return false;
        }

        const auto copy_width =
            std::min<uint32_t>(entry.width, static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(&info)));
        const auto copy_height =
            std::min<uint32_t>(entry.height, static_cast<uint32_t>(GST_VIDEO_INFO_HEIGHT(&info)));
        const VkDeviceSize rgba_size = static_cast<VkDeviceSize>(entry.width) * entry.height * 4u;
        if (! entry.cuda_rgba_buffer) {
            if (auto cuda_buffer =
                    CreateCudaExternalTransferBuffer(m_device, rgba_size, cuda_context);
                cuda_buffer.has_value()) {
                entry.cuda_rgba_buffer = std::move(cuda_buffer.value());
                LOG_INFO("video texture '%s': created NVIDIA CUDA/Vulkan shared RGBA buffer bytes=%llu",
                         entry.key.c_str(),
                         static_cast<unsigned long long>(rgba_size));
            } else {
                gst_video_frame_unmap(&frame);
                entry.warned_cuda_interop_failed = true;
                return false;
            }
        }

        if (entry.cuda_kernel == nullptr) {
            bool pushed = gst_cuda_context_push(cuda_context);
            CUresult cuda_result = pushed
                ? CuModuleLoadData(&entry.cuda_module, kNv12ToRgbaCudaPtx)
                : CUDA_ERROR_UNKNOWN;
            if (cuda_result == CUDA_SUCCESS) {
                cuda_result = CuModuleGetFunction(&entry.cuda_kernel,
                                                  entry.cuda_module,
                                                  "HanabiNv12ToRgba");
            }
            if (pushed) {
                CUcontext popped {};
                (void)gst_cuda_context_pop(&popped);
            }

            if (cuda_result != CUDA_SUCCESS || entry.cuda_kernel == nullptr) {
                gst_video_frame_unmap(&frame);
                LOG_ERROR("video texture '%s': load embedded CUDA NV12 converter failed result=%s",
                          entry.key.c_str(),
                          CudaErrorName(cuda_result));
                return false;
            }
        }

        auto* cuda_memory = reinterpret_cast<GstCudaMemory*>(memory);
        GstCudaStream* gst_stream = gst_cuda_memory_get_stream(cuda_memory);
        CUstream stream = gst_stream != nullptr ? gst_cuda_stream_get_handle(gst_stream) : nullptr;
        CUdeviceptr y_ptr =
            reinterpret_cast<CUdeviceptr>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
        CUdeviceptr uv_ptr =
            reinterpret_cast<CUdeviceptr>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
        CUdeviceptr rgba_ptr = entry.cuda_rgba_buffer.cuda_ptr;
        int width = static_cast<int>(copy_width);
        int height = static_cast<int>(copy_height);
        int y_pitch = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
        int uv_pitch = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
        int rgba_pitch = static_cast<int>(entry.width) * 4;
        void* params[] {
            &y_ptr,
            &uv_ptr,
            &rgba_ptr,
            &width,
            &height,
            &y_pitch,
            &uv_pitch,
            &rgba_pitch,
        };

        bool pushed = gst_cuda_context_push(cuda_context);
        CUresult cuda_result = pushed
            ? CuLaunchKernel(entry.cuda_kernel,
                             (copy_width + 15u) / 16u,
                             (copy_height + 15u) / 16u,
                             1,
                             16,
                             16,
                             1,
                             0,
                             stream,
                             params,
                             nullptr)
            : CUDA_ERROR_UNKNOWN;
        if (cuda_result == CUDA_SUCCESS) {
            cuda_result = stream != nullptr ? CuStreamSynchronize(stream) : CuCtxSynchronize();
        }
        if (pushed) {
            CUcontext popped {};
            (void)gst_cuda_context_pop(&popped);
        }
        gst_video_frame_unmap(&frame);

        if (cuda_result != CUDA_SUCCESS) {
            LOG_ERROR("video texture '%s': CUDA NV12->RGBA launch failed result=%s",
                      entry.key.c_str(),
                      CudaErrorName(cuda_result));
            return false;
        }

        entry.cuda_rgba_dirty = true;
        entry.uploaded_sample_count++;
        entry.eos_loop_waiting_for_sample = false;
        entry.eos_loop_rebuild_attempted = false;
        return true;
    }

    if (! entry.warned_unexpected_format) {
        gchar* caps_text = gst_caps_to_string(caps);
        LOG_ERROR("video texture '%s': GPU decoder path %s produced an unsupported "
                  "sample caps='%s'; CPU upload fallback is disabled",
                  entry.key.c_str(),
                  VideoTextureTransferPathName(entry.transfer_path),
                  caps_text != nullptr ? caps_text : "unknown");
        if (caps_text != nullptr) g_free(caps_text);
        entry.warned_unexpected_format = true;
    }
    return false;
}

ImageSlotsRef VideoTextureCache::Acquire(std::string_view key,
                                         const SceneTexture& texture,
                                         const Image& image,
                                         VideoTexturePlaybackState initial_state) {
    if (auto* entry = find(key)) {
        if (initial_state == VideoTexturePlaybackState::Stopped)
            stopPlayback(*entry);
        else
            setPaused(*entry, initial_state == VideoTexturePlaybackState::Paused);
        ImageSlotsRef ref;
        ref.slots.push_back(ImageParameters(entry->image));
        return ref;
    }

    if (image.slots.empty() || image.slots[0].mipmaps.empty() || image.slots[0].mipmaps[0].data == nullptr) {
        LOG_ERROR("video texture '%.*s' is missing embedded payload",
                  static_cast<int>(key.size()),
                  key.data());
        return {};
    }

    auto entry = std::make_unique<Entry>();
    entry->key.assign(key);
    entry->encoded.assign(image.slots[0].mipmaps[0].data.get(),
                          image.slots[0].mipmaps[0].data.get() + image.slots[0].mipmaps[0].size);

    entry->width = static_cast<uint32_t>(std::max(
        1, texture.mapWidth > 0 ? texture.mapWidth
                                : (texture.width > 0 ? texture.width : image.slots[0].width)));
    entry->height = static_cast<uint32_t>(std::max(
        1, texture.mapHeight > 0 ? texture.mapHeight
                                 : (texture.height > 0 ? texture.height : image.slots[0].height)));
    const auto decoder_selection = SelectVideoTextureDecoder(m_device, MakeDecoderSettings(m_settings));
    if (! decoder_selection.has_value()) {
        LOG_ERROR("video texture '%s': no GPU decoder available for render-node=%s",
                  entry->key.c_str(),
                  m_settings.render_node.empty() ? "(unresolved)" : m_settings.render_node.c_str());
        return {};
    }
    entry->transfer_path = decoder_selection->path;
    entry->decoder_factory = decoder_selection->decoder_factory;
    entry->postproc_factory = decoder_selection->postproc_factory;
    entry->expected_cuda_device_id = decoder_selection->expected_cuda_device_id;

    if (auto opt = CreateVideoImage(m_device,
                                    entry->width,
                                    entry->height,
                                    texture.sample,
                                    TargetImageFormatForTransferPath(entry->transfer_path),
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        opt.has_value()) {
        entry->image = std::move(opt.value());
    } else {
        LOG_ERROR("create video texture image for '%s' failed", entry->key.c_str());
        return {};
    }

    if (! CreateStagingBuffer(m_device.vma_allocator(),
                              static_cast<size_t>(entry->width) * entry->height * 4u,
                              entry->staging)) {
        LOG_ERROR("create video staging buffer for '%s' failed", entry->key.c_str());
        return {};
    }
    {
        void* mapped = nullptr;
        if (entry->staging.handle.MapMemory(&mapped) == VK_SUCCESS) {
            entry->staging_mapped = static_cast<uint8_t*>(mapped);
            std::memset(entry->staging_mapped, 0, entry->staging.req_size);
            (void)vmaFlushAllocation(m_device.vma_allocator(),
                                     entry->staging.handle.Allocation(),
                                     0,
                                     entry->staging.req_size);
        }
    }

    if (! m_cmd) allocateCmd();
    if (TransitionImageLayout(m_device.graphics_queue().handle,
                              m_cmd,
                              ImageParameters(entry->image),
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) != VK_SUCCESS) {
        LOG_ERROR("transition video texture '%s' image layout failed", entry->key.c_str());
        return {};
    }
    VVK_CHECK(m_device.handle().WaitIdle());

    if (initial_state == VideoTexturePlaybackState::Stopped) {
        entry->paused = true;
        entry->stopped = true;
    } else {
        if (!startPipeline(*entry)) {
            return {};
        }
        if (initial_state == VideoTexturePlaybackState::Paused) setPaused(*entry, true);
    }

    m_entries.emplace_back(std::move(entry));
    ImageSlotsRef ref;
    ref.slots.push_back(ImageParameters(m_entries.back()->image));
    return ref;
}

void VideoTextureCache::ApplyPlaybackStates(
    const std::unordered_map<std::string, bool>& paused_by_key,
    const std::unordered_set<std::string>& stopped_keys) {
    for (auto& entry_ptr : m_entries) {
        if (entry_ptr == nullptr) continue;
        if (stopped_keys.count(entry_ptr->key) != 0) {
            stopPlayback(*entry_ptr);
            continue;
        }
        auto state_it = paused_by_key.find(entry_ptr->key);
        if (state_it == paused_by_key.end()) continue;
        setPaused(*entry_ptr, state_it->second);
    }
}

void VideoTextureCache::SetGlobalPaused(bool paused) {
    if (m_globally_paused == paused) return;
    m_globally_paused = paused;

    // Shell-level pause stops the render timer, but GStreamer keeps decoding on its own threads
    // unless the pipeline is told to pause. Apply this immediately so video-texture wallpapers do
    // not burn CPU while the visible scene is frozen by GJS playback controls.
    for (auto& entry_ptr : m_entries) {
        if (entry_ptr == nullptr) continue;
        (void)applyPipelinePlaybackState(*entry_ptr);
    }
}

void VideoTextureCache::ApplySeekRequests(std::unordered_map<std::string, double>& seek_seconds_by_key) {
    for (auto iter = seek_seconds_by_key.begin(); iter != seek_seconds_by_key.end();) {
        auto* entry = find(iter->first);
        if (entry == nullptr) {
            ++iter;
            continue;
        }

        // A request is erased only after the concrete decoder accepts it. Init scripts can therefore
        // queue setCurrentTime() before CustomShaderPass has materialized the texture, without
        // losing the rewind on the frame where the video cache entry finally appears.
        if (seekTo(*entry, iter->second))
            iter = seek_seconds_by_key.erase(iter);
        else
            ++iter;
    }
}

void VideoTextureCache::Poll() {
    for (auto& entry_ptr : m_entries) {
        auto& entry = *entry_ptr;
        if (entry.pipeline_failed) continue;
        if (entry.stopped || entry.pipeline == nullptr) continue;
        if (m_globally_paused) continue;

        bool should_loop = false;
        while (entry.bus != nullptr) {
            GstMessage* message = gst_bus_pop(entry.bus);
            if (message == nullptr) break;

            switch (GST_MESSAGE_TYPE(message)) {
            case GST_MESSAGE_ERROR: {
                GError* error = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                LOG_ERROR("video texture '%s' pipeline error source=%s domain=%s code=%d "
                          "message=%s debug=%s",
                          entry.key.c_str(),
                          GST_MESSAGE_SRC_NAME(message),
                          error != nullptr ? g_quark_to_string(error->domain) : "unknown",
                          error != nullptr ? error->code : 0,
                          error != nullptr ? error->message : "unknown error",
                          debug != nullptr ? debug : "none");
                if (error != nullptr) g_error_free(error);
                if (debug != nullptr) g_free(debug);
                entry.pipeline_failed = true;
                break;
            }
            case GST_MESSAGE_EOS: should_loop = true; break;
            default: break;
            }

            gst_message_unref(message);
        }

        if (should_loop && !entry.pipeline_failed) {
            if (!loopPipeline(entry)) continue;
        }

        if (entry.paused) continue;

        GstSample* latest_sample = nullptr;
        while (entry.appsink_elem != nullptr) {
            GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(entry.appsink_elem), 0);
            if (sample == nullptr) break;
            if (latest_sample != nullptr) gst_sample_unref(latest_sample);
            latest_sample = sample;
        }
        if (latest_sample == nullptr) continue;

        (void)uploadSample(entry, latest_sample);
        gst_sample_unref(latest_sample);
    }
}

void VideoTextureCache::RecordUploads(vvk::CommandBuffer& cmd) {
    for (auto& entry_ptr : m_entries) {
        auto& entry = *entry_ptr;
        if (entry.pipeline_failed) continue;

        if (! entry.dirty && ! entry.cuda_rgba_dirty && ! entry.va_dmabuf_dirty) continue;
        const bool use_cuda_rgba = entry.cuda_rgba_dirty && entry.cuda_rgba_buffer.buffer != VK_NULL_HANDLE;
        const bool use_va_dmabuf = entry.va_dmabuf_dirty && entry.va_imported_image.image != VK_NULL_HANDLE;

        if (use_cuda_rgba) {
            VkBufferMemoryBarrier cuda_to_transfer {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = entry.cuda_rgba_buffer.buffer,
                .offset = 0,
                .size = entry.cuda_rgba_buffer.size,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
	                                VK_DEPENDENCY_BY_REGION_BIT,
	                                cuda_to_transfer);
        }
        if (use_va_dmabuf) {
            VkImageMemoryBarrier va_to_transfer {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = entry.va_imported_image.image,
                .subresourceRange =
                    VkImageSubresourceRange {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_DEPENDENCY_BY_REGION_BIT,
                                va_to_transfer);
        }

        VkImageMemoryBarrier to_transfer {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .srcAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image            = *entry.image.handle,
            .subresourceRange =
                VkImageSubresourceRange {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            to_transfer);

        if (use_va_dmabuf) {
            VkImageCopy copy {
                .srcSubresource =
                    VkImageSubresourceLayers {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .srcOffset = VkOffset3D { 0, 0, 0 },
                .dstSubresource =
                    VkImageSubresourceLayers {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .dstOffset = VkOffset3D { 0, 0, 0 },
                .extent =
                    VkExtent3D {
                        std::min(entry.width, entry.va_imported_image.width),
                        std::min(entry.height, entry.va_imported_image.height),
                        1,
                    },
            };
            VkImageCopy copy_regions[] { copy };
            cmd.CopyImage(entry.va_imported_image.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          *entry.image.handle,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          copy_regions);
        } else {
            VkBufferImageCopy copy {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    VkImageSubresourceLayers {
                        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel       = 0,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
                .imageOffset = VkOffset3D { 0, 0, 0 },
                .imageExtent = VkExtent3D { entry.width, entry.height, 1 },
            };
            VkBufferImageCopy copy_regions[] { copy };
            cmd.CopyBufferToImage(use_cuda_rgba ? entry.cuda_rgba_buffer.buffer : *entry.staging.handle,
                                  *entry.image.handle,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  copy_regions);
        }

        VkImageMemoryBarrier to_shader {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = *entry.image.handle,
            .subresourceRange = to_transfer.subresourceRange,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            to_shader);
        entry.dirty = false;
        entry.cuda_rgba_dirty = false;
        entry.va_dmabuf_dirty = false;
    }
}

void VideoTextureCache::Clear() {
    m_entries.clear();
}

bool VideoTextureCache::Release(std::string_view key) {
    const auto iter =
        std::find_if(m_entries.begin(), m_entries.end(), [key](const auto& entry) {
            return entry != nullptr && entry->key == key;
        });
    if (iter == m_entries.end()) return false;

    const std::string key_string(key);
    const auto        before_bytes = GetTrackedBytes();
    const auto        before_count = GetTrackedEntryCount();
    m_entries.erase(iter);
    LOG_INFO("VideoTextureCacheRelease: key='%s' bytes-before=%zu bytes-after=%zu "
             "entries-before=%zu entries-after=%zu",
             key_string.c_str(),
             before_bytes,
             GetTrackedBytes(),
             before_count,
             GetTrackedEntryCount());
    return true;
}

std::size_t VideoTextureCache::GetTrackedBytes() const {
    std::size_t total = 0;
    for (const auto& entry : m_entries) {
        if (! entry) continue;
        if (entry->image.handle) total += static_cast<std::size_t>(entry->image.handle.AllocationSize());
        if (entry->staging.handle) total += static_cast<std::size_t>(entry->staging.handle.AllocationSize());
        if (entry->cuda_rgba_buffer.buffer != VK_NULL_HANDLE) {
            total += static_cast<std::size_t>(entry->cuda_rgba_buffer.size);
        }
    }
    return total;
}

std::size_t VideoTextureCache::GetTrackedEntryCount() const {
    return m_entries.size();
}

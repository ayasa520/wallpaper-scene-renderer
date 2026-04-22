#include "VideoTextureCache.hpp"

#include "Core/ArrayHelper.hpp"
#include "Device.hpp"
#include "Image.hpp"
#include "Scene/SceneTexture.h"
#include "Shader.hpp"
#include "TextureCache.hpp"
#include "Util.hpp"
#include "Utils/Logging.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

using namespace wallpaper;
using namespace wallpaper::vulkan;

namespace
{

constexpr guint kAppSrcChunkSize = 256 * 1024;

constexpr std::string_view kNv12ConvertVert = R"(#version 320 es
layout(location = 0) out vec2 v_Texcoord;

void main() {
    const vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0, 3.0),
        vec2(3.0, -1.0)
    );
    const vec2 texcoords[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(0.0, 2.0),
        vec2(2.0, 0.0)
    );
    v_Texcoord = texcoords[gl_VertexIndex];
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
)";

constexpr std::string_view kNv12ConvertFrag = R"(#version 320 es
layout(location = 0) in vec2 v_Texcoord;
layout(location = 0) out vec4 out_FragColor;

layout(binding = 0) uniform sampler2D u_YTexture;
layout(binding = 1) uniform sampler2D u_UVTexture;

void main() {
    vec2 sample_uv = vec2(v_Texcoord.x, 1.0 - v_Texcoord.y);
    float y = texture(u_YTexture, sample_uv).r;
    vec2 uv = texture(u_UVTexture, sample_uv).rg - vec2(0.5, 0.5);

    float luma = 1.16438356 * max(y - 0.0625, 0.0);
    float r = luma + 1.79274107 * uv.y;
    float g = luma - 0.21324861 * uv.x - 0.53290933 * uv.y;
    float b = luma + 2.11240179 * uv.x;

    out_FragColor = vec4(clamp(vec3(r, g, b), 0.0, 1.0), 1.0);
}
)";

bool EndsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size(), suffix.size()) == suffix;
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

void ConfigureDecoderRanks() {
    // Mirror Hanabi's video backend strategy: nudge NV/VA decoders above libav
    // so decodebin will pick hardware acceleration when the runtime can actually load it.
    SetPluginDecoderRanks("nvcodec", GST_RANK_PRIMARY + 1, false);
    SetPluginDecoderRanks("va", GST_RANK_PRIMARY + 3, false);
}

std::string BuildVideoOnlyPipelineDescription() {
    return "appsrc name=src block=true format=bytes stream-type=random-access "
           "! qtdemux name=demux "
           "demux.video_0 ! queue "
           "! h264parse "
           "! decodebin "
           "! videoconvert "
           "! appsink name=sink sync=true max-buffers=1 drop=true";
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

TextureSample MakePlaneSample() {
    return TextureSample {
        .wrapS = TextureWrap::CLAMP_TO_EDGE,
        .wrapT = TextureWrap::CLAMP_TO_EDGE,
        .magFilter = TextureFilter::LINEAR,
        .minFilter = TextureFilter::LINEAR,
    };
}

std::optional<vvk::RenderPass> CreateNv12ConvertRenderPass(const vvk::Device& device, VkFormat format) {
    VkAttachmentDescription attachment {
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentReference attachment_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription subpass {
        .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &attachment_ref,
    };
    VkSubpassDependency dependency {
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };
    VkRenderPassCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &attachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &dependency,
    };
    vvk::RenderPass pass;
    if (device.CreateRenderPass(info, pass) != VK_SUCCESS) return std::nullopt;
    return pass;
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

} // namespace

struct VideoTextureCache::Entry {
    enum class UploadMode {
        RGBA,
        NV12,
    };

    std::string key;
    std::vector<uint8_t> encoded;
    VmaImageParameters image;
    VmaImageParameters plane_y_image;
    VmaImageParameters plane_uv_image;
    VmaBufferParameters staging;
    VmaBufferParameters plane_y_staging;
    VmaBufferParameters plane_uv_staging;
    uint8_t* staging_mapped { nullptr };
    uint8_t* plane_y_staging_mapped { nullptr };
    uint8_t* plane_uv_staging_mapped { nullptr };
    uint32_t width { 0 };
    uint32_t height { 0 };
    bool     dirty { false };
    bool     plane_dirty { false };
    bool     warned_size_mismatch { false };
    bool     pipeline_failed { false };
    bool     paused { false };
    size_t   read_offset { 0 };
    UploadMode upload_mode { UploadMode::RGBA };
    vvk::Framebuffer convert_framebuffer;

    GstElement* pipeline { nullptr };
    GstElement* appsrc_elem { nullptr };
    GstElement* appsink_elem { nullptr };
    GstBus*     bus { nullptr };

    static void NeedData(GstAppSrc* src, guint length, gpointer user_data) {
        auto* self = static_cast<Entry*>(user_data);
        if (self == nullptr || self->pipeline_failed) return;
        if (self->read_offset >= self->encoded.size()) {
            (void)gst_app_src_end_of_stream(src);
            return;
        }

        const auto chunk_size =
            std::min<size_t>(length > 0 ? static_cast<size_t>(length) : kAppSrcChunkSize,
                             self->encoded.size() - self->read_offset);
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, chunk_size, nullptr);
        if (buffer == nullptr) return;

        (void)gst_buffer_fill(buffer, 0, self->encoded.data() + self->read_offset, chunk_size);
        self->read_offset += chunk_size;

        const auto flow = gst_app_src_push_buffer(src, buffer);
        if (flow != GST_FLOW_OK && flow != GST_FLOW_EOS) {
            LOG_ERROR("video texture '%s' appsrc push failed: %d", self->key.c_str(), flow);
            self->pipeline_failed = true;
        }
    }

    static gboolean SeekData(GstAppSrc*, guint64 offset, gpointer user_data) {
        auto* self = static_cast<Entry*>(user_data);
        if (self == nullptr) return FALSE;
        self->read_offset = std::min<size_t>(static_cast<size_t>(offset), self->encoded.size());
        return TRUE;
    }

    ~Entry() {
        if (staging_mapped != nullptr) staging.handle.UnMapMemory();
        if (plane_y_staging_mapped != nullptr) plane_y_staging.handle.UnMapMemory();
        if (plane_uv_staging_mapped != nullptr) plane_uv_staging.handle.UnMapMemory();
        if (pipeline != nullptr) (void)gst_element_set_state(pipeline, GST_STATE_NULL);
        if (bus != nullptr) gst_object_unref(bus);
        if (appsink_elem != nullptr) gst_object_unref(appsink_elem);
        if (appsrc_elem != nullptr) gst_object_unref(appsrc_elem);
        if (pipeline != nullptr) gst_object_unref(pipeline);
    }
};

VideoTextureCache::VideoTextureCache(const Device& device): m_device(device) {
    if (! gst_is_initialized()) gst_init(nullptr, nullptr);
    ConfigureDecoderRanks();
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

bool VideoTextureCache::ensureNv12Pipeline() {
    if (m_nv12_pipeline_ready) return true;

    auto pass = CreateNv12ConvertRenderPass(m_device.handle(), VK_FORMAT_R8G8B8A8_UNORM);
    if (!pass.has_value()) return false;

    std::vector<Uni_ShaderSpv> spvs;
    ShaderCompOpt opt;
    opt.client_ver             = glslang::EShTargetVulkan_1_1;
    opt.relaxed_errors_glsl    = true;
    opt.relaxed_rules_vulkan   = true;
    opt.suppress_warnings_glsl = true;

    std::array<ShaderCompUnit, 2> units;
    units[0] = ShaderCompUnit { .stage = EShLangVertex, .src = std::string(kNv12ConvertVert) };
    units[1] = ShaderCompUnit { .stage = EShLangFragment, .src = std::string(kNv12ConvertFrag) };
    if (! CompileAndLinkShaderUnits(units, opt, spvs)) return false;

    DescriptorSetInfo descriptor_info;
    descriptor_info.push_descriptor = true;
    descriptor_info.bindings = {
        VkDescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        VkDescriptorSetLayoutBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    GraphicsPipeline pipeline;
    pipeline.toDefault();
    m_nv12_pipeline.debug_name = "VideoTextureCache::NV12";
    pipeline.setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addDescriptorSetInfo(spanone { descriptor_info });
    for (auto& spv : spvs) pipeline.addStage(std::move(spv));

    auto render_pass = std::move(pass.value());
    if (! pipeline.create(m_device, render_pass, m_nv12_pipeline)) return false;
    m_nv12_pipeline_ready = true;
    return true;
}

void VideoTextureCache::stopPipeline(Entry& entry) {
    if (entry.pipeline != nullptr) (void)gst_element_set_state(entry.pipeline, GST_STATE_NULL);
    if (entry.bus != nullptr) gst_object_unref(entry.bus);
    if (entry.appsink_elem != nullptr) gst_object_unref(entry.appsink_elem);
    if (entry.appsrc_elem != nullptr) gst_object_unref(entry.appsrc_elem);
    if (entry.pipeline != nullptr) gst_object_unref(entry.pipeline);
    entry.bus = nullptr;
    entry.appsink_elem = nullptr;
    entry.appsrc_elem = nullptr;
    entry.pipeline = nullptr;
}

bool VideoTextureCache::startPipeline(Entry& entry) {
    stopPipeline(entry);
    entry.read_offset = 0;

    GError* error = nullptr;
    const auto pipeline_desc = BuildVideoOnlyPipelineDescription();
    entry.pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
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

    entry.appsrc_elem = gst_bin_get_by_name(GST_BIN(entry.pipeline), "src");
    entry.appsink_elem = gst_bin_get_by_name(GST_BIN(entry.pipeline), "sink");
    entry.bus = gst_element_get_bus(entry.pipeline);
    if (entry.appsrc_elem == nullptr || entry.appsink_elem == nullptr || entry.bus == nullptr) {
        LOG_ERROR("video texture '%s' pipeline is missing appsrc/appsink", entry.key.c_str());
        stopPipeline(entry);
        return false;
    }

    GstCaps* caps = gst_caps_from_string("video/quicktime, variant=(string)iso");
    g_object_set(entry.appsrc_elem,
                 "caps",
                 caps,
                 "size",
                 static_cast<gint64>(entry.encoded.size()),
                 nullptr);
    gst_caps_unref(caps);

    GstCaps* sink_caps =
        gst_caps_from_string("video/x-raw,format=(string)NV12;video/x-raw,format=(string)RGBA");
    g_object_set(entry.appsink_elem, "caps", sink_caps, nullptr);
    gst_caps_unref(sink_caps);

    static GstAppSrcCallbacks callbacks {
        .need_data = &Entry::NeedData,
        .enough_data = nullptr,
        .seek_data = &Entry::SeekData,
    };
    gst_app_src_set_callbacks(GST_APP_SRC(entry.appsrc_elem), &callbacks, &entry, nullptr);

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

    if (gst_element_set_state(entry.pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("start video pipeline playback for '%s' failed", entry.key.c_str());
        stopPipeline(entry);
        return false;
    }
    entry.paused = false;
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

bool VideoTextureCache::setPaused(Entry& entry, bool paused) {
    if (entry.paused == paused) return true;
    if (entry.pipeline == nullptr) {
        entry.paused = paused;
        return true;
    }

    // getVideoTexture().pause() is often called every scene tick by authored Wallpaper Engine
    // scripts. Make the cache transition the GStreamer pipeline only when the desired state
    // changes so hidden videos stop decoding without turning repeated script calls into stalls.
    const auto target_state = paused ? GST_STATE_PAUSED : GST_STATE_PLAYING;
    if (gst_element_set_state(entry.pipeline, target_state) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("video texture '%s': failed to %s playback",
                  entry.key.c_str(),
                  paused ? "pause" : "resume");
        entry.pipeline_failed = true;
        return false;
    }
    entry.paused = paused;
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

    GstVideoFrame frame;
    if (! gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
        return false;
    }

    const auto copy_width = std::min<uint32_t>(entry.width, static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(&info)));
    const auto copy_height =
        std::min<uint32_t>(entry.height, static_cast<uint32_t>(GST_VIDEO_INFO_HEIGHT(&info)));
    if ((copy_width != entry.width || copy_height != entry.height) && ! entry.warned_size_mismatch) {
        LOG_INFO("video texture '%s' decoded size %ux%u differs from texture size %ux%u",
                 entry.key.c_str(),
                 GST_VIDEO_INFO_WIDTH(&info),
                 GST_VIDEO_INFO_HEIGHT(&info),
                 entry.width,
                 entry.height);
        entry.warned_size_mismatch = true;
    }

    if (GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_NV12) {
        const auto chroma_width = (copy_width + 1u) / 2u;
        const auto chroma_height = (copy_height + 1u) / 2u;
        auto* y_dst = entry.plane_y_staging_mapped;
        auto* uv_dst = entry.plane_uv_staging_mapped;
        if (y_dst == nullptr || uv_dst == nullptr) {
            gst_video_frame_unmap(&frame);
            return false;
        }

        const auto* y_src = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
        const auto* uv_src = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
        const auto y_src_stride = static_cast<size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0));
        const auto uv_src_stride = static_cast<size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1));
        const auto y_dst_stride = static_cast<size_t>(entry.width);
        const auto uv_dst_stride = static_cast<size_t>((entry.width + 1u) / 2u) * 2u;

        for (uint32_t y = 0; y < copy_height; y++) {
            std::memcpy(y_dst + static_cast<size_t>(y) * y_dst_stride,
                        y_src + static_cast<size_t>(y) * y_src_stride,
                        copy_width);
            if (copy_width < entry.width) {
                std::memset(y_dst + static_cast<size_t>(y) * y_dst_stride + copy_width,
                            0,
                            y_dst_stride - copy_width);
            }
        }
        if (copy_height < entry.height) {
            std::memset(y_dst + static_cast<size_t>(copy_height) * y_dst_stride,
                        0,
                        (static_cast<size_t>(entry.height) - copy_height) * y_dst_stride);
        }

        for (uint32_t y = 0; y < chroma_height; y++) {
            std::memcpy(uv_dst + static_cast<size_t>(y) * uv_dst_stride,
                        uv_src + static_cast<size_t>(y) * uv_src_stride,
                        chroma_width * 2u);
            if (chroma_width * 2u < uv_dst_stride) {
                std::memset(uv_dst + static_cast<size_t>(y) * uv_dst_stride + chroma_width * 2u,
                            0,
                            uv_dst_stride - chroma_width * 2u);
            }
        }
        const auto plane_uv_height = (entry.height + 1u) / 2u;
        if (chroma_height < plane_uv_height) {
            std::memset(uv_dst + static_cast<size_t>(chroma_height) * uv_dst_stride,
                        0,
                        (static_cast<size_t>(plane_uv_height) - chroma_height) * uv_dst_stride);
        }

        (void)vmaFlushAllocation(m_device.vma_allocator(),
                                 entry.plane_y_staging.handle.Allocation(),
                                 0,
                                 entry.plane_y_staging.req_size);
        (void)vmaFlushAllocation(m_device.vma_allocator(),
                                 entry.plane_uv_staging.handle.Allocation(),
                                 0,
                                 entry.plane_uv_staging.req_size);
        entry.upload_mode = Entry::UploadMode::NV12;
        entry.plane_dirty = true;
        gst_video_frame_unmap(&frame);
        return true;
    }

    uint8_t* mapped = entry.staging_mapped;
    if (mapped == nullptr) {
        gst_video_frame_unmap(&frame);
        return false;
    }

    {
        auto* dst = mapped;
        const auto* src_base = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
        const auto src_stride = static_cast<size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0));
        const auto dst_stride = static_cast<size_t>(entry.width) * 4u;
        const auto row_bytes = static_cast<size_t>(copy_width) * 4u;
        for (uint32_t y = 0; y < copy_height; y++) {
            std::memcpy(dst + static_cast<size_t>(y) * dst_stride,
                        src_base + static_cast<size_t>(y) * src_stride,
                        row_bytes);
            if (copy_width < entry.width) {
                std::memset(dst + static_cast<size_t>(y) * dst_stride + row_bytes,
                            0,
                            dst_stride - row_bytes);
            }
        }
        if (copy_height < entry.height) {
            std::memset(dst + static_cast<size_t>(copy_height) * dst_stride,
                        0,
                        (static_cast<size_t>(entry.height) - copy_height) * dst_stride);
        }
        (void)vmaFlushAllocation(m_device.vma_allocator(),
                                 entry.staging.handle.Allocation(),
                                 0,
                                 entry.staging.req_size);
        entry.upload_mode = Entry::UploadMode::RGBA;
        entry.dirty = true;
    }

    gst_video_frame_unmap(&frame);
    return true;
}

ImageSlotsRef VideoTextureCache::Acquire(std::string_view key, const SceneTexture& texture,
                                         const Image& image, bool paused) {
    if (auto* entry = find(key)) {
        setPaused(*entry, paused);
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

    if (auto opt = CreateVideoImage(m_device,
                                    entry->width,
                                    entry->height,
                                    texture.sample,
                                    VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT |
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        opt.has_value()) {
        entry->image = std::move(opt.value());
    } else {
        LOG_ERROR("create video texture image for '%s' failed", entry->key.c_str());
        return {};
    }

    const auto plane_sample = MakePlaneSample();
    if (auto opt = CreateVideoImage(m_device,
                                    entry->width,
                                    entry->height,
                                    plane_sample,
                                    VK_FORMAT_R8_UNORM,
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        opt.has_value()) {
        entry->plane_y_image = std::move(opt.value());
    } else {
        LOG_ERROR("create video texture Y plane for '%s' failed", entry->key.c_str());
        return {};
    }

    if (auto opt = CreateVideoImage(m_device,
                                    (entry->width + 1u) / 2u,
                                    (entry->height + 1u) / 2u,
                                    plane_sample,
                                    VK_FORMAT_R8G8_UNORM,
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        opt.has_value()) {
        entry->plane_uv_image = std::move(opt.value());
    } else {
        LOG_ERROR("create video texture UV plane for '%s' failed", entry->key.c_str());
        return {};
    }

    if (! CreateStagingBuffer(m_device.vma_allocator(),
                              static_cast<size_t>(entry->width) * entry->height * 4u,
                              entry->staging)) {
        LOG_ERROR("create video staging buffer for '%s' failed", entry->key.c_str());
        return {};
    }
    if (! CreateStagingBuffer(m_device.vma_allocator(),
                              static_cast<size_t>(entry->width) * entry->height,
                              entry->plane_y_staging)) {
        LOG_ERROR("create video Y staging buffer for '%s' failed", entry->key.c_str());
        return {};
    }
    if (! CreateStagingBuffer(m_device.vma_allocator(),
                              static_cast<size_t>((entry->width + 1u) / 2u) *
                                  ((entry->height + 1u) / 2u) * 2u,
                              entry->plane_uv_staging)) {
        LOG_ERROR("create video UV staging buffer for '%s' failed", entry->key.c_str());
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
        if (entry->plane_y_staging.handle.MapMemory(&mapped) == VK_SUCCESS) {
            entry->plane_y_staging_mapped = static_cast<uint8_t*>(mapped);
            std::memset(entry->plane_y_staging_mapped, 0, entry->plane_y_staging.req_size);
            (void)vmaFlushAllocation(m_device.vma_allocator(),
                                     entry->plane_y_staging.handle.Allocation(),
                                     0,
                                     entry->plane_y_staging.req_size);
        }
        if (entry->plane_uv_staging.handle.MapMemory(&mapped) == VK_SUCCESS) {
            entry->plane_uv_staging_mapped = static_cast<uint8_t*>(mapped);
            std::memset(entry->plane_uv_staging_mapped, 0, entry->plane_uv_staging.req_size);
            (void)vmaFlushAllocation(m_device.vma_allocator(),
                                     entry->plane_uv_staging.handle.Allocation(),
                                     0,
                                     entry->plane_uv_staging.req_size);
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
    if (TransitionImageLayout(m_device.graphics_queue().handle,
                              m_cmd,
                              ImageParameters(entry->plane_y_image),
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) != VK_SUCCESS) {
        LOG_ERROR("transition video texture '%s' Y plane layout failed", entry->key.c_str());
        return {};
    }
    if (TransitionImageLayout(m_device.graphics_queue().handle,
                              m_cmd,
                              ImageParameters(entry->plane_uv_image),
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) != VK_SUCCESS) {
        LOG_ERROR("transition video texture '%s' UV plane layout failed", entry->key.c_str());
        return {};
    }
    VVK_CHECK(m_device.handle().WaitIdle());

    if (! ensureNv12Pipeline()) {
        LOG_ERROR("create video texture conversion pipeline for '%s' failed", entry->key.c_str());
        return {};
    }
    VkFramebufferCreateInfo framebuffer_info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .renderPass      = *m_nv12_pipeline.pass,
        .attachmentCount = 1,
        .pAttachments    = entry->image.view.address(),
        .width           = entry->width,
        .height          = entry->height,
        .layers          = 1,
    };
    if (m_device.handle().CreateFramebuffer(framebuffer_info, entry->convert_framebuffer) != VK_SUCCESS) {
        LOG_ERROR("create video texture conversion framebuffer for '%s' failed", entry->key.c_str());
        return {};
    }

    if (!startPipeline(*entry)) {
        return {};
    }
    if (paused) setPaused(*entry, true);

    m_entries.emplace_back(std::move(entry));
    ImageSlotsRef ref;
    ref.slots.push_back(ImageParameters(m_entries.back()->image));
    return ref;
}

void VideoTextureCache::ApplyPlaybackStates(const std::unordered_map<std::string, bool>& paused_by_key) {
    for (auto& entry_ptr : m_entries) {
        if (entry_ptr == nullptr) continue;
        auto state_it = paused_by_key.find(entry_ptr->key);
        if (state_it == paused_by_key.end()) continue;
        setPaused(*entry_ptr, state_it->second);
    }
}

void VideoTextureCache::Poll() {
    for (auto& entry_ptr : m_entries) {
        auto& entry = *entry_ptr;
        if (entry.pipeline_failed) continue;

        bool should_restart = false;
        while (entry.bus != nullptr) {
            GstMessage* message = gst_bus_pop(entry.bus);
            if (message == nullptr) break;

            switch (GST_MESSAGE_TYPE(message)) {
            case GST_MESSAGE_ERROR: {
                GError* error = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                LOG_ERROR("video texture '%s' pipeline error: %s",
                          entry.key.c_str(),
                          error != nullptr ? error->message : "unknown error");
                if (error != nullptr) g_error_free(error);
                if (debug != nullptr) g_free(debug);
                entry.pipeline_failed = true;
                break;
            }
            case GST_MESSAGE_EOS: should_restart = true; break;
            default: break;
            }

            gst_message_unref(message);
        }

        if (should_restart && !entry.pipeline_failed) {
            if (!restartPipeline(entry)) continue;
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

        auto image_range = VkImageSubresourceRange {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        };

        if (entry.plane_dirty && entry.upload_mode == Entry::UploadMode::NV12) {
            const auto upload_plane = [&](const VmaBufferParameters& staging, const VmaImageParameters& image,
                                          uint32_t width, uint32_t height, VkImageMemoryBarrier& to_transfer,
                                          VkImageMemoryBarrier& to_shader) {
                to_transfer = VkImageMemoryBarrier {
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext            = nullptr,
                    .srcAccessMask    = VK_ACCESS_SHADER_READ_BIT,
                    .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .image            = *image.handle,
                    .subresourceRange = image_range,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_DEPENDENCY_BY_REGION_BIT,
                                    to_transfer);

                const VkBufferImageCopy copy {
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
                    .imageExtent = VkExtent3D { width, height, 1 },
                };
                VkBufferImageCopy copy_regions[] { copy };
                cmd.CopyBufferToImage(*staging.handle,
                                      *image.handle,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      copy_regions);

                to_shader = VkImageMemoryBarrier {
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext            = nullptr,
                    .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .image            = *image.handle,
                    .subresourceRange = image_range,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                    VK_DEPENDENCY_BY_REGION_BIT,
                                    to_shader);
            };

            VkImageMemoryBarrier y_to_transfer {}, y_to_shader {};
            VkImageMemoryBarrier uv_to_transfer {}, uv_to_shader {};
            upload_plane(entry.plane_y_staging,
                         entry.plane_y_image,
                         entry.width,
                         entry.height,
                         y_to_transfer,
                         y_to_shader);
            upload_plane(entry.plane_uv_staging,
                         entry.plane_uv_image,
                         (entry.width + 1u) / 2u,
                         (entry.height + 1u) / 2u,
                         uv_to_transfer,
                         uv_to_shader);

            const VkImageMemoryBarrier to_color {
                .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext            = nullptr,
                .srcAccessMask    = VK_ACCESS_SHADER_READ_BIT,
                .dstAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .image            = *entry.image.handle,
                .subresourceRange = image_range,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_DEPENDENCY_BY_REGION_BIT,
                                to_color);

            const std::array descriptor_writes {
                VkWriteDescriptorSet {
                    .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext           = nullptr,
                    .dstSet          = {},
                    .dstBinding      = 0,
                    .descriptorCount = 1,
                    .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                },
                VkWriteDescriptorSet {
                    .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext           = nullptr,
                    .dstSet          = {},
                    .dstBinding      = 1,
                    .descriptorCount = 1,
                    .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                },
            };
            const std::array descriptor_images {
                VkDescriptorImageInfo {
                    .sampler     = *entry.plane_y_image.sampler,
                    .imageView   = *entry.plane_y_image.view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                },
                VkDescriptorImageInfo {
                    .sampler     = *entry.plane_uv_image.sampler,
                    .imageView   = *entry.plane_uv_image.view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                },
            };
            auto writes = descriptor_writes;
            writes[0].pImageInfo = &descriptor_images[0];
            writes[1].pImageInfo = &descriptor_images[1];

            const VkClearValue clear_value { .color = { 0.0f, 0.0f, 0.0f, 1.0f } };
            const VkRenderPassBeginInfo render_pass_info {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = *m_nv12_pipeline.pass,
                .framebuffer = *entry.convert_framebuffer,
                .renderArea = VkRect2D { { 0, 0 }, { entry.width, entry.height } },
                .clearValueCount = 1,
                .pClearValues = &clear_value,
            };
            cmd.BeginRenderPass(render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_nv12_pipeline.handle);
            cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_nv12_pipeline.layout, 0, writes);
            const VkViewport viewport {
                .x        = 0.0f,
                .y        = static_cast<float>(entry.height),
                .width    = static_cast<float>(entry.width),
                .height   = -static_cast<float>(entry.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            const VkRect2D scissor { { 0, 0 }, { entry.width, entry.height } };
            VkViewport viewports[] { viewport };
            VkRect2D   scissors[] { scissor };
            cmd.SetViewport(0, viewports);
            cmd.SetScissor(0, scissors);
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRenderPass();

            const VkImageMemoryBarrier to_shader {
                .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext            = nullptr,
                .srcAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
                .oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .image            = *entry.image.handle,
                .subresourceRange = image_range,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                VK_DEPENDENCY_BY_REGION_BIT,
                                to_shader);
            entry.plane_dirty = false;
        }

        if (! entry.dirty || entry.upload_mode != Entry::UploadMode::RGBA) continue;

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
        cmd.CopyBufferToImage(*entry.staging.handle,
                              *entry.image.handle,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              copy_regions);

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
    }
}

void VideoTextureCache::Clear() {
    m_entries.clear();
}

std::size_t VideoTextureCache::GetTrackedBytes() const {
    std::size_t total = 0;
    for (const auto& entry : m_entries) {
        if (! entry) continue;
        if (entry->image.handle) total += static_cast<std::size_t>(entry->image.handle.AllocationSize());
        if (entry->plane_y_image.handle) {
            total += static_cast<std::size_t>(entry->plane_y_image.handle.AllocationSize());
        }
        if (entry->plane_uv_image.handle) {
            total += static_cast<std::size_t>(entry->plane_uv_image.handle.AllocationSize());
        }
        if (entry->staging.handle) total += static_cast<std::size_t>(entry->staging.handle.AllocationSize());
        if (entry->plane_y_staging.handle) {
            total += static_cast<std::size_t>(entry->plane_y_staging.handle.AllocationSize());
        }
        if (entry->plane_uv_staging.handle) {
            total += static_cast<std::size_t>(entry->plane_uv_staging.handle.AllocationSize());
        }
    }
    return total;
}

std::size_t VideoTextureCache::GetTrackedEntryCount() const {
    return m_entries.size();
}

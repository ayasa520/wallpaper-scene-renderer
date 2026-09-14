#pragma once

#include "Parameters.hpp"
#include "Swapchain/ExSwapchain.hpp"
#include "Type.hpp"
#include "Core/NoCopyMove.hpp"
#include "Core/MapSet.hpp"
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wallpaper
{

class Image;

namespace vulkan
{

VkFormat             ToVkType(TextureFormat);
VkSamplerAddressMode ToVkType(TextureWrap);
VkFilter             ToVkType(TextureFilter);

enum class TexUsage
{
    COLOR,
    DEPTH
};

using TexHash = std::size_t;

struct TextureCachePendingImageUpload {
    std::string                      key;
    ImageParameters                  image;
    VkImageLayout                    old_layout { VK_IMAGE_LAYOUT_UNDEFINED };
    std::vector<VmaBufferParameters> stage_bufs;
    std::vector<VkExtent3D>          extents;
};

struct TextureCachePendingRenderTargetClear {
    std::string     key;
    ImageParameters image;
};

struct TextureKey {
    i32           width { 0 };
    i32           height { 0 };
    TexUsage      usage { TexUsage::COLOR };
    TextureFormat format { TextureFormat::RGBA8 };
    TextureSample sample {};
    uint          mipmap_level { 1 };
    uint          sample_count { 1 };

    static TexHash HashValue(const TextureKey&);

    bool operator==(const TextureKey&) const = default;
};

class TextureCache : NoCopy, NoMove {
public:
    TextureCache(const Device&);
    ~TextureCache();

    void Clear();
    bool ReleaseTexture(std::string_view key);
    bool ReleaseRenderTarget(std::string_view key);

    std::optional<ExImageParameters> CreateExTex(uint32_t witdh, uint32_t height, VkFormat,
                                                 VkImageTiling, ExternalFrameExportMode,
                                                 uint32_t export_drm_fourcc = 0,
                                                 std::span<const uint64_t> export_drm_modifiers = {},
                                                 ExternalFrameMemoryPreference memory_preference =
                                                     ExternalFrameMemoryPreference::Default);
    ImageSlotsRef                    CreateTex(Image&);
    std::optional<ImageSlotsRef>     FindTex(std::string_view key) const;

    std::optional<ImageParameters> Query(std::string_view key, TextureKey content_hash,
                                         bool persist = false);
    // A generation identifies a physical allocation for the lifetime of this cache.
    // Pass recreation and pooled reuse retain it; replacement assigns a new value even
    // if the driver reuses a handle. Query must establish the key before this lookup.
    uint64_t RenderTargetGeneration(std::string_view key) const;

    void RecordUploads(vvk::CommandBuffer&);
    void RetireCompletedUploads();
    void MarkShareReady(std::string_view key);
    std::size_t GetTrackedBytes() const;
    std::size_t GetTrackedImageCount() const;

    void RecGenerateMipmaps(vvk::CommandBuffer& cmd, const ImageParameters& image) const;

private:
    std::optional<VmaImageParameters> CreateTex(TextureKey);
    void                              allocateCmd();
    void                              purgeQueuedWorkForKey(std::string_view key);
    vvk::CommandBuffers               m_tex_cmds;
    vvk::CommandBuffer                m_tex_cmd;

    struct ImageCacheRevision {
        uint64_t content { 0 };
        uint64_t texture_resolution_epoch { 0 };

        bool operator==(const ImageCacheRevision&) const = default;
    };

    static ImageCacheRevision CacheRevisionFor(const Image& image);

    const Device&                         m_device;
    Map<std::string, ImageSlots>          m_tex_map;
    Map<std::string, ImageCacheRevision>  m_tex_revision_map;

    struct QueryTex {
        idx                index { 0 };
        bool               share_ready { false };
        bool               persist { false };
        TextureKey         content_key;
        TexHash            content_hash { 0 };
        uint64_t           generation { 0 };
        VmaImageParameters image;
        Set<std::string>   query_keys;
    };
    std::vector<std::unique_ptr<QueryTex>> m_query_texs;
    Map<std::string, QueryTex*>            m_query_map;
    uint64_t                              m_render_target_generation { 0 };
    std::vector<TextureCachePendingImageUpload>       m_pending_image_uploads;
    std::vector<TextureCachePendingImageUpload>       m_inflight_image_uploads;
    std::vector<TextureCachePendingRenderTargetClear> m_pending_render_target_clears;
};

} // namespace vulkan
} // namespace wallpaper

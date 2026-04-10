#pragma once

#include "Core/NoCopyMove.hpp"
#include "GraphicsPipeline.hpp"
#include "Parameters.hpp"

#include <memory>
#include <string_view>
#include <vector>

typedef struct _GstSample GstSample;

namespace wallpaper
{

class Image;
struct SceneTexture;

namespace vulkan
{

class Device;

class VideoTextureCache : NoCopy, NoMove {
public:
    VideoTextureCache(const Device&);
    ~VideoTextureCache();

    ImageSlotsRef Acquire(std::string_view key, const SceneTexture&, const Image&);
    void          Poll();
    void          RecordUploads(vvk::CommandBuffer&);
    void          Clear();

private:
    struct Entry;

    Entry*       find(std::string_view key);
    const Entry* find(std::string_view key) const;
    void         allocateCmd();
    bool         ensureNv12Pipeline();
    bool         startPipeline(Entry&);
    void         stopPipeline(Entry&);
    bool         restartPipeline(Entry&);
    bool         uploadSample(Entry&, ::GstSample*);

    const Device& m_device;
    vvk::CommandBuffers m_cmds;
    vvk::CommandBuffer  m_cmd;
    PipelineParameters  m_nv12_pipeline;
    bool                m_nv12_pipeline_ready { false };
    std::vector<std::unique_ptr<Entry>> m_entries;
};

} // namespace vulkan
} // namespace wallpaper

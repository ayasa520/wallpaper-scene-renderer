#pragma once
#include "Core/NoCopyMove.hpp"
#include "MaskedDrawAttachmentCache.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/StagingBuffer.hpp"
#include "vvk/vma_wrapper.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace wallpaper
{
namespace vulkan
{

struct FrameDrawStats {
    uint32_t shader_draws { 0 };
    uint32_t shader_vertices { 0 };
    uint32_t shader_indices { 0 };
    uint32_t prepared_passes { 0 };

    void reset() {
        shader_draws     = 0;
        shader_vertices  = 0;
        shader_indices   = 0;
        prepared_passes  = 0;
    }
};

struct RenderingResources {
    vvk::CommandBuffer command;

    vvk::Semaphore sem_swap_wait_image;
    vvk::Semaphore sem_swap_finish;
    vvk::Fence     fence_frame;

    StagingBuffer* vertex_buf;
    StagingBuffer* dyn_buf;

    FrameDrawStats* frame_draw_stats { nullptr };

    std::shared_ptr<GraphicsPipelineStateCache> pipeline_cache;

    // 3D model chunks are emitted as separate CustomShaderPass instances, but authored WE models
    // rely on them sharing one depth buffer per output target. Keeping that depth storage here makes
    // the behavior opt-in for model passes and leaves all legacy 2D render targets color-only.
    std::unordered_map<std::string, VmaImageParameters> model_depth_images;

    MaskedDrawAttachmentCache masked_draw_attachments;
};
} // namespace vulkan
} // namespace wallpaper

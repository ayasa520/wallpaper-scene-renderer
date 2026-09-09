#pragma once
#include "Core/NoCopyMove.hpp"
#include "MaskedDrawAttachmentCache.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/ImmutableMeshStore.hpp"
#include "Vulkan/StagingBuffer.hpp"
#include "vvk/vma_wrapper.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace wallpaper
{
class Scene;

namespace vulkan
{

struct ModelDepthAttachment {
    VmaImageParameters image;
    // Clearing is optional scene behavior; layout initialization is a Vulkan resource
    // lifetime requirement. Keep it with the allocation so a resize or sample-count
    // change resets the state, without inventing a first-frame content clear.
    VkImageLayout layout { VK_IMAGE_LAYOUT_UNDEFINED };
};

struct RenderingResources {
    Scene* scene { nullptr };
    bool   msaa_compose_dirty { false };

    vvk::CommandBuffer command;

    vvk::Semaphore sem_swap_wait_image;
    vvk::Semaphore sem_swap_finish;
    vvk::Fence     fence_frame;
    /*
     * Offscreen export: one SYNC_FD-exportable binary semaphore per ring slot.
     * The frame submit signals the in-progress slot's semaphore and its
     * exported sync-file becomes the frame's explicit acquire fence, letting
     * the producer publish without a CPU fence wait.
     */
    std::array<vvk::Semaphore, 3> sem_offscreen_acquire;

    StagingBuffer* vertex_buf;
    StagingBuffer* dyn_buf;
    ImmutableMeshStore immutable_meshes;

    std::shared_ptr<GraphicsPipelineStateCache> pipeline_cache;

    // Main/reflection shader draws and model chunks share one depth buffer per output target.
    // Depth storage belongs to that destination, not to a particular material or visible owner.
    // Ordinary effect FBOs remain color-only; masked meshes own separate stencil attachments.
    std::unordered_map<std::string, ModelDepthAttachment> model_depth_images;
    std::unordered_map<std::string, VmaImageParameters> model_depth_resolved;
    // Outputs whose multisampled model depth has been written since the last single-sample
    // resolve. Model draws only mark this; the depth-sampling consumer resolves once on demand,
    // so a frame with hundreds of model chunks does not pay one full-extent resolve per chunk.
    std::unordered_set<std::string> model_depth_dirty;

    MaskedDrawAttachmentCache masked_draw_attachments;
};
} // namespace vulkan
} // namespace wallpaper

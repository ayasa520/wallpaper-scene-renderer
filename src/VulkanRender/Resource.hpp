#pragma once
#include "Core/NoCopyMove.hpp"
#include "FrameTraceDump.hpp"
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

struct MipHistoryState {
    // This state follows the physical texture, independently of graph/pass residency.
    // Preparation may register a new allocation and its pending creation clear, but
    // only a submitted clear advances the applied revision or consumes that request.
    uint64_t generation { 0 };
    uint64_t submitted_disable_revision { 0 };
    bool     creation_clear_pending { false };
};

struct RenderingResources {
    Scene* scene { nullptr };
    bool   msaa_compose_dirty { false };
    MipHistoryState mip_history;

    // Command diagnostics follow the renderer lifetime rather than graph/pass objects,
    // which can be replaced while the same physical targets remain live across frames.
    bool     trace_render_commands { false };
    uint64_t trace_render_frame { 0 };
    uint64_t trace_render_command { 0 };
    // Draws recorded since renderer init, counted whether or not tracing is on. Selects the
    // draws the structural trace dump writes; matches the embedding's lockstep draw count.
    uint64_t       draw_index { 0 };
    FrameTraceDump frame_trace_dump;

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

    // Main/reflection shader draws, model chunks and glyphs share one depth buffer per output target.
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

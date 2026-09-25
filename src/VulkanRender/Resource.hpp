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
#include <optional>
#include <string>
#include <unordered_map>

namespace wallpaper
{
class Scene;

namespace vulkan
{

struct ModelDepthResolve {
    // Member order releases the framebuffer before its render pass and resolved image view.
    VmaImageParameters image;
    vvk::RenderPass pass;
    vvk::Framebuffer framebuffer;
};

struct ModelDepthAttachment {
    static constexpr VkFormat format = VK_FORMAT_D32_SFLOAT;

    VmaImageParameters image;
    // Clearing is optional scene behavior; layout initialization is a Vulkan resource
    // lifetime requirement. Keep it with the allocation so a resize or sample-count
    // change resets the state, without inventing a first-frame content clear.
    VkImageLayout layout { VK_IMAGE_LAYOUT_UNDEFINED };
    // The resolve framebuffer references both this source and its single-sample image. Keep
    // them in one allocation generation so graph teardown destroys the framebuffer before
    // either view. Replacement also resets resolve explicitly before replacing the source.
    std::optional<ModelDepthResolve> resolve;
    // Draws and clears mark depth dirty; the sampling consumer resolves once on demand,
    // independently of the number of model chunks that wrote this destination.
    bool resolve_dirty { false };
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
    [[no_unique_address]] FrameTraceDump frame_trace_dump;

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

    MaskedDrawAttachmentCache masked_draw_attachments;
};
} // namespace vulkan
} // namespace wallpaper

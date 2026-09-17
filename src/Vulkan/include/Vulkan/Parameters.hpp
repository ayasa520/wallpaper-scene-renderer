#pragma once

#include "Instance.hpp"
#include "Swapchain.hpp"
#include "Swapchain/ExSwapchain.hpp"
#include "Core/NoCopyMove.hpp"
#include "vk_mem_alloc.h"
#include "vvk/vma_wrapper.hpp"

#include <memory>
#include <utility>

namespace wallpaper
{
namespace vulkan
{

struct QueueParameters {
    vvk::Queue handle;
    uint32_t   family_index;
};

struct VmaBufferParameters {
    vvk::VmaBuffer handle;
    std::size_t    req_size;

    VmaBufferParameters();
    ~VmaBufferParameters();
    VmaBufferParameters(VmaBufferParameters&& o) noexcept;
    VmaBufferParameters& operator=(VmaBufferParameters&& o) noexcept;
};

struct BufferParameters {
    VkBuffer    handle;
    std::size_t req_size;
    BufferParameters()  = default;
    ~BufferParameters() = default;
    BufferParameters(const VmaBufferParameters& o) noexcept
        : handle(*o.handle), req_size(o.req_size) {}
};

struct VmaImageParameters : NoCopy {
    vvk::VmaImage  handle;
    vvk::ImageView view;
    vvk::Sampler   sampler;
    VkExtent3D     extent;
    uint           mipmap_level { 1 };
    uint           samples { 1 };

    VmaImageParameters();
    ~VmaImageParameters();
    VmaImageParameters(VmaImageParameters&& o) noexcept;
    VmaImageParameters& operator=(VmaImageParameters&& o) noexcept;
};

struct ExImageParameters : NoCopy {
    vvk::DeviceMemory    mem {};
    VkMemoryRequirements mem_reqs {};

    vvk::Image     handle;
    vvk::ImageView view;
    vvk::Sampler   sampler;
    VkExtent3D     extent;
    uint           mipmap_level { 1 };
    int            fd { -1 };
    ExternalFrameHandleType          handle_type { ExternalFrameHandleType::NONE };
    uint32_t                         drm_fourcc { 0 };
    uint64_t                         drm_modifier { ExHandle::INVALID_DRM_MODIFIER };
    uint32_t                         n_planes { 0 };
    bool                             premultiplied { false };
    std::array<ExPlane, ExHandle::MAX_PLANES> planes {};

    ExImageParameters();
    ~ExImageParameters();
    ExImageParameters(ExImageParameters&& o) noexcept;
    ExImageParameters& operator=(ExImageParameters&& o) noexcept;
};

struct ImageParameters {
    VkImage     handle;
    VkImageView view;
    VkSampler   sampler;
    VkExtent3D  extent;
    uint        mipmap_level { 1 };
    uint        samples { 1 };

    // Prepared passes retain a target allocation independently of its transient lookup name.
    // The final-reader handoff permits reuse within the graph; it does not end the lifetime of
    // copied descriptors/framebuffers that will execute that graph again. Imported and external
    // images continue to have their own owners and leave this target-specific reference empty.
    std::shared_ptr<const VmaImageParameters> allocation_owner;

    ImageParameters()  = default;
    ~ImageParameters() = default;
    ImageParameters(const VmaImageParameters& o) noexcept
        : handle(*o.handle),
          view(*o.view),
          sampler(*o.sampler),
          extent(o.extent),
          mipmap_level(o.mipmap_level),
          samples(o.samples) {}
    ImageParameters(const ExImageParameters& o) noexcept
        : handle(*o.handle),
          view(*o.view),
          sampler(*o.sampler),
          extent(o.extent),
          mipmap_level(o.mipmap_level) {}
    explicit ImageParameters(std::shared_ptr<const VmaImageParameters> image) noexcept
        : ImageParameters(*image) {
        allocation_owner = std::move(image);
    }
};

struct ImageSlots : NoCopy {
    std::vector<VmaImageParameters> slots;

    ImageSlots();
    ~ImageSlots();
    ImageSlots(ImageSlots&& o) noexcept;
    ImageSlots& operator=(ImageSlots&& o) noexcept;
};

struct ImageSlotsRef {
    std::vector<ImageParameters> slots;

    idx active { 0 };

    auto& getActive() const {
        if (active > 0 && active >= std::ssize(slots)) return slots[0];
        return slots[(usize)active];
    }
    ImageSlotsRef();
    ~ImageSlotsRef();
    ImageSlotsRef(const ImageSlots&);
};

} // namespace vulkan
} // namespace wallpaper

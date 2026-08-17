#pragma once

#include "Core/NoCopyMove.hpp"
#include "Parameters.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace wallpaper
{
class SceneMesh;
}

namespace wallpaper
{
namespace vulkan
{

class Device;

// One GPU buffer created at exact host byte width, uploaded once, then left immutable.
// The allocation is never grown or merged with other meshes.
class ImmutableGpuBuffer : NoCopy {
public:
    ImmutableGpuBuffer() = default;
    ImmutableGpuBuffer(ImmutableGpuBuffer&&) noexcept;
    ImmutableGpuBuffer& operator=(ImmutableGpuBuffer&&) noexcept;
    ~ImmutableGpuBuffer();

    bool create(const Device&, VkBufferUsageFlags, std::span<const uint8_t> data);
    void recordUpload(vvk::CommandBuffer&);
    void retireStaging();
    void abandon() noexcept;

    VkBuffer     handle() const;
    VkDeviceSize size() const { return m_size; }
    explicit     operator bool() const { return handle() != VK_NULL_HANDLE; }

private:
    VmaBufferParameters m_gpu;
    VmaBufferParameters m_stage;
    VmaAllocator        m_allocator { VK_NULL_HANDLE };
    VkDeviceSize        m_size { 0 };
    bool                m_copy_recorded { false };
};

struct ImmutableMeshGpu {
    std::vector<ImmutableGpuBuffer> vertices;
    ImmutableGpuBuffer              index;
    uint32_t                        index_element_bytes { 2 };
    bool                            has_index { false };
};

// Static file meshes each own exact-size VB/IB pairs. The store de-duplicates by the mesh
// storage object so a color pass and the shadow atlas bind the same GPU buffers after the
// host vertex bytes have been dropped.
class ImmutableMeshStore : NoCopy, NoMove {
public:
    std::shared_ptr<ImmutableMeshGpu> getOrCreate(const Device&, const SceneMesh&);
    void                              recordUploads(vvk::CommandBuffer&);
    void                              retireStaging();
    void                              clear();
    void                              abandon() noexcept;

private:
    std::unordered_map<const void*, std::shared_ptr<ImmutableMeshGpu>> m_meshes;
};

} // namespace vulkan
} // namespace wallpaper

#include "Vulkan/ImmutableMeshStore.hpp"

#include "Device.hpp"
#include "Scene/SceneMesh.h"
#include "Util.hpp"

#include <cstring>
#include <utility>

using namespace wallpaper::vulkan;

namespace
{

std::optional<VmaBufferParameters> CreateGpuBuffer(VmaAllocator allocator, VkBufferUsageFlags usage,
                                                   std::size_t size) {
    VmaBufferParameters buffer;
    VkBufferCreateInfo  ci {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .pNext = nullptr,
         .size  = size,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
    };
    buffer.req_size                  = ci.size;
    VmaAllocationCreateInfo vma_info = {};
    vma_info.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
    VVK_CHECK_ACT(return std::nullopt, vvk::CreateBuffer(allocator, ci, vma_info, buffer.handle));
    return buffer;
}

void RecordCopyBuffer(const VmaBufferParameters& dst, const VmaBufferParameters& src,
                      VkDeviceSize size, vvk::CommandBuffer& cmd) {
    VkBufferCopy copy {
        .srcOffset = 0,
        .dstOffset = 0,
        .size      = size,
    };
    cmd.CopyBuffer(*src.handle, *dst.handle, copy);

    VkBufferMemoryBarrier bar {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext         = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
        .buffer        = *dst.handle,
        .offset        = 0,
        .size          = size,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        bar);
}

} // namespace

ImmutableGpuBuffer::ImmutableGpuBuffer(ImmutableGpuBuffer&& o) noexcept
    : m_gpu(std::move(o.m_gpu)),
      m_stage(std::move(o.m_stage)),
      m_allocator(o.m_allocator),
      m_size(o.m_size),
      m_copy_recorded(o.m_copy_recorded) {
    o.m_allocator     = VK_NULL_HANDLE;
    o.m_size          = 0;
    o.m_copy_recorded = false;
}

ImmutableGpuBuffer& ImmutableGpuBuffer::operator=(ImmutableGpuBuffer&& o) noexcept {
    if (this == &o) return *this;
    m_gpu             = std::move(o.m_gpu);
    m_stage           = std::move(o.m_stage);
    m_allocator       = o.m_allocator;
    m_size            = o.m_size;
    m_copy_recorded   = o.m_copy_recorded;
    o.m_allocator     = VK_NULL_HANDLE;
    o.m_size          = 0;
    o.m_copy_recorded = false;
    return *this;
}

ImmutableGpuBuffer::~ImmutableGpuBuffer() = default;

bool ImmutableGpuBuffer::create(const Device& device, VkBufferUsageFlags usage,
                                std::span<const uint8_t> data) {
    if (data.empty()) return false;
    auto gpu = CreateGpuBuffer(device.vma_allocator(), usage, data.size());
    if (! gpu.has_value()) return false;
    m_gpu       = std::move(*gpu);
    m_allocator = device.vma_allocator();
    m_size      = data.size();

    if (! CreateStagingBuffer(device.vma_allocator(), data.size(), m_stage)) return false;
    void* mapped { nullptr };
    VVK_CHECK_BOOL_RE(m_stage.handle.MapMemory(&mapped));
    std::memcpy(mapped, data.data(), data.size());
    m_stage.handle.UnMapMemory();
    m_copy_recorded = false;
    return true;
}

void ImmutableGpuBuffer::recordUpload(vvk::CommandBuffer& cmd) {
    if (! m_gpu.handle || ! m_stage.handle || m_copy_recorded || m_size == 0) return;
    if (m_allocator != VK_NULL_HANDLE) {
        VVK_CHECK_VOID_RE(vmaFlushAllocation(m_allocator, m_stage.handle.Allocation(), 0, m_size));
    }
    RecordCopyBuffer(m_gpu, m_stage, m_size, cmd);
    m_copy_recorded = true;
}

VkBuffer ImmutableGpuBuffer::handle() const {
    return m_gpu.handle ? *m_gpu.handle : VK_NULL_HANDLE;
}

void ImmutableGpuBuffer::retireStaging() {
    m_stage     = {};
    m_allocator = m_gpu.handle ? m_allocator : VK_NULL_HANDLE;
}

void ImmutableGpuBuffer::abandon() noexcept {
    m_gpu.handle.abandon();
    m_stage.handle.abandon();
    m_allocator     = VK_NULL_HANDLE;
    m_size          = 0;
    m_copy_recorded = true;
}

std::shared_ptr<ImmutableMeshGpu> ImmutableMeshStore::getOrCreate(const Device& device,
                                                                  const SceneMesh& mesh) {
    if (mesh.VertexCount() == 0) return nullptr;
    const void* key = mesh.GpuStorageKey();
    if (key == nullptr) return nullptr;

    if (auto it = m_meshes.find(key); it != m_meshes.end()) return it->second;
    if (! mesh.HasCpuPayload()) return nullptr;

    auto gpu = std::make_shared<ImmutableMeshGpu>();
    gpu->vertices.resize(mesh.VertexCount());
    for (size_t i = 0; i < mesh.VertexCount(); ++i) {
        const auto& vertex = mesh.GetVertexArray(i);
        const auto  bytes  = vertex.DataSizeOf();
        if (bytes == 0 || vertex.Data() == nullptr) return nullptr;
        if (! gpu->vertices[i].create(device,
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                      { reinterpret_cast<const uint8_t*>(vertex.Data()), bytes })) {
            return nullptr;
        }
    }

    if (mesh.IndexCount() > 0) {
        const auto& indice = mesh.GetIndexArray(0);
        const auto  bytes  = static_cast<size_t>(mesh.LogicalIndexCount()) *
                            static_cast<size_t>(mesh.IndexElementBytes());
        if (bytes > 0 && indice.Data() != nullptr && bytes <= indice.DataSizeOf()) {
            if (! gpu->index.create(device,
                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                    { reinterpret_cast<const uint8_t*>(indice.Data()), bytes })) {
                return nullptr;
            }
            gpu->has_index           = true;
            gpu->index_element_bytes = mesh.IndexElementBytes();
        }
    }

    m_meshes.emplace(key, gpu);
    return gpu;
}

void ImmutableMeshStore::recordUploads(vvk::CommandBuffer& cmd) {
    for (auto& [_, mesh] : m_meshes) {
        if (! mesh) continue;
        for (auto& vertex : mesh->vertices) vertex.recordUpload(cmd);
        if (mesh->has_index) mesh->index.recordUpload(cmd);
    }
}

void ImmutableMeshStore::retireStaging() {
    for (auto& [_, mesh] : m_meshes) {
        if (! mesh) continue;
        for (auto& vertex : mesh->vertices) vertex.retireStaging();
        if (mesh->has_index) mesh->index.retireStaging();
    }
}

void ImmutableMeshStore::clear() { m_meshes.clear(); }

void ImmutableMeshStore::abandon() noexcept {
    for (auto& [_, mesh] : m_meshes) {
        if (! mesh) continue;
        for (auto& vertex : mesh->vertices) vertex.abandon();
        mesh->index.abandon();
    }
    m_meshes.clear();
}

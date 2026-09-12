#pragma once

#include "Vulkan/Device.hpp"
#include "Vulkan/Parameters.hpp"

#include <string>
#include <string_view>
#include <memory>
#include <unordered_map>

namespace wallpaper::vulkan
{

class MaskedDrawAttachmentCache {
public:
    enum class Role { Accumulated, Intermediate };

    std::shared_ptr<VmaImageParameters> acquire(const Device&, std::string_view output, VkExtent3D,
                                               Role);
    void clear();
    void abandon();

private:
    // A resized destination can replace the cache entry while another prepared draw still
    // owns a framebuffer for the old extent. Keep that image alive until its last framebuffer
    // is dropped; cache replacement must never destroy an attached image view.
    struct Entry {
        std::shared_ptr<VmaImageParameters> accumulated;
        std::shared_ptr<VmaImageParameters> intermediate;
    };
    std::unordered_map<std::string, Entry> m_entries;
};

} // namespace wallpaper::vulkan

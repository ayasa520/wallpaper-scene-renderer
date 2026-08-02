#pragma once

#include "Vulkan/Device.hpp"
#include "Vulkan/Parameters.hpp"

#include <string>
#include <string_view>
#include <unordered_map>

namespace wallpaper::vulkan
{

class MaskedDrawAttachmentCache {
public:
    VmaImageParameters* acquire(const Device&, std::string_view output, VkExtent3D, VkFormat);
    void clear();
    void abandon();

private:
    struct Entry {
        VkFormat           format { VK_FORMAT_UNDEFINED };
        VmaImageParameters image;
    };

    std::unordered_map<std::string, Entry> m_entries;
};

} // namespace wallpaper::vulkan

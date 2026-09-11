#pragma once

#include "Scene/SceneMesh.h"
#include "SpecTexs.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/Shader.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace wallpaper::vulkan
{

struct MaskedDrawShaderContract {
    ShaderReflected reflection;
    DescriptorSetInfo descriptors;
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::array<int32_t, WE_GLTEX_NAMES.size()> texture_bindings;
    uint32_t uniform_binding { 0 };

    bool hasUniform(std::string_view name) const;
};

// Mask and clipped executables have independent reflected layouts. Resolve names against
// the imported mesh instead of assigning locations or reusing the visible shader's UBO.
bool ReflectMaskedDrawShaderContract(const SceneMesh&, const SceneMaterial&,
                                     std::vector<Uni_ShaderSpv>&, MaskedDrawShaderContract&);

} // namespace wallpaper::vulkan

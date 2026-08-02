#pragma once

#include "Scene/SceneShader.h"
#include "SpecTexs.hpp"
#include "Vulkan/Shader.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace wallpaper::vulkan
{

inline constexpr uint32_t kMaskedDrawUniformBinding = 0;
inline constexpr uint32_t kMaskedDrawTextureBinding = 1;
inline constexpr uint32_t kMaskedDrawStencilReference = 1;

inline constexpr std::array<std::string_view, 4> kMaskedDrawVertexAttributes {
    WE_IN_POSITION,
    WE_IN_BLENDINDICES,
    WE_IN_BLENDWEIGHTS,
    WE_IN_TEXCOORD,
};

std::optional<std::vector<ShaderCode>> CompileMaskedDrawMaskShaderCodes(uint32_t bone_count);

} // namespace wallpaper::vulkan

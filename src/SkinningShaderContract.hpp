#pragma once

#include <Eigen/Geometry>

#include <cstddef>
#include <span>
#include <vector>

namespace wallpaper
{

inline constexpr std::size_t kDxcSkinningMatrixFloatCount = 16;

inline std::vector<float>
PackDxcRowVectorSkinningUniform(std::span<const Eigen::Affine3f> matrices) {
    // Wallpaper Engine authors skinning uniforms as GLSL `mat4x3`: four columns (xyz plus
    // translation) by three rows, then multiplies `mul(float4(position, 1), g_Bones[i])` to get
    // xyz. Vivid's DXC bridge spells that type as HLSL `float3x4` and swaps the multiply to native
    // `mul(M, v)`. SPIR-V's uniform-buffer layout gives every array element four 16-byte columns,
    // including the otherwise-unused fourth row. Keep this packing in one shared contract so the
    // visible-material and shadow-caster paths cannot disagree about the palette stride.
    std::vector<float> packed;
    packed.reserve(matrices.size() * kDxcSkinningMatrixFloatCount);
    for (const auto& affine : matrices) {
        const Eigen::Matrix4f matrix = affine.matrix();
        for (int column = 0; column < 4; ++column) {
            packed.push_back(matrix(0, column));
            packed.push_back(matrix(1, column));
            packed.push_back(matrix(2, column));
            packed.push_back(0.0f);
        }
    }
    return packed;
}

} // namespace wallpaper

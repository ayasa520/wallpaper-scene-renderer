#pragma once
#include <string>
#include <cstdint>
#include <array>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "WPPuppet.hpp"

namespace wallpaper
{

class WPShaderInfo;

namespace wpscene
{
class WPMaterial;
};
namespace fs
{
class VFS;
};

struct WPMdl {
    enum class MeshKind
    {
        Unknown,
        Static,
        StaticImage,
        Puppet,
    };

    i32 mdlv { 13 };
    i32 mdls { 1 };
    i32 mdla { 1 };
    MeshKind kind { MeshKind::Unknown };

    std::string source_path;
    std::string mat_json_file;
    struct StaticChunk {
        std::string                         material_json_file;
        // Older static model formats can store a prefixed material table before the geometry bytes.
        // Scene model objects select one of those entries through their `skin` index, so the parser
        // keeps the full table here while material_json_file remains the default/fallback material.
        std::vector<std::string>            material_json_variants;
        uint32_t                            vertex_flag { 0 };
        uint32_t                            vertex_stride { 0 };
        std::vector<uint8_t>                vertex_blob;
        std::vector<std::array<uint32_t, 3>> indices;
        uint32_t                            index_element_bytes { 2 };
        std::array<float, 3>                bounds_min { 0.0f, 0.0f, 0.0f };
        std::array<float, 3>                bounds_max { 0.0f, 0.0f, 0.0f };
    };
    std::vector<StaticChunk> static_chunks;

    struct Vertex {
        std::array<float, 3>    position;
        std::array<uint32_t, 4> blend_indices;
        std::array<float, 4>    weight;
        std::array<float, 2>    texcoord;
    };
    std::vector<Vertex>                  vertexs;
    std::vector<std::array<uint16_t, 3>> indices;
    // Vertex positions are stored in the MDL asset's puppet-local coordinate space.
    PuppetBounds3D asset_bounds;
    // Immutable union of the authored animation envelopes, also in puppet-local coordinates.
    // Runtime pose observation is intentionally not written back into this parsed asset contract.
    PuppetBounds3D authored_pose_bounds;

    struct Part {
        uint32_t id { 0 };
        uint32_t first_index { 0 };
        uint32_t index_count { 0 };
    };
    std::vector<Part> parts;

    struct MaskBlock {
        std::string           material;
        std::vector<uint32_t> clipped_part_indices;
        std::vector<uint32_t> source_part_indices;
    };
    std::vector<MaskBlock> masks;

    // std::vector<Eigen::Matrix<float, 3, 4>> bones;
    std::shared_ptr<WPPuppet> puppet;
    // combo
    // SKINNING = 1
    // BONECOUNT

    // input
    // uvec4 a_BlendIndices
    // vec4 a_BlendWeights
    // uniform mat4x3 g_Bones[BONECOUNT]
};

class SceneMesh;

class WPMdlParser {
public:
    static bool Parse(std::string_view path, fs::VFS&, WPMdl&);
    static bool ParseStaticModel(std::string_view path, fs::VFS&, WPMdl&);

    static void AddPuppetShaderInfo(WPShaderInfo& info, const WPMdl& mdl);
    static void AddPuppetMatInfo(wpscene::WPMaterial& mat, const WPMdl& mdl);

    static void GenPuppetMesh(SceneMesh& mesh, const WPMdl& mdl);
    static void GenStaticMesh(SceneMesh& mesh, const WPMdl::StaticChunk& chunk);
};

} // namespace wallpaper

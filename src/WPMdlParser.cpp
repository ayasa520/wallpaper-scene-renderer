#include "WPMdlParser.hpp"
#include "Fs/VFS.h"
#include "Fs/IBinaryStream.h"
#include "Fs/MemBinaryStream.h"
#include "WPCommon.hpp"
#include "Utils/Logging.h"
#include "Scene/SceneMesh.h"
#include "SpecTexs.hpp"
#include "wpscene/WPMaterial.h"
#include "WPShaderParser.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace wallpaper;

namespace
{
constexpr uint32_t kStaticPositionTexcoordFlag = 9;
constexpr uint32_t kStaticNormalFlag           = 11;
constexpr uint32_t kStaticTangentSpaceFlag     = 15;
constexpr uint32_t kStaticTangentSpaceSecondUvFlag = 39;
constexpr uint32_t kStaticPositionTexcoordFloats = 5;
constexpr uint32_t kStaticNormalFloats           = 8;
constexpr uint32_t kStaticTangentSpaceFloats     = 12;
constexpr uint32_t kStaticTangentSpaceSecondUvFloats = 14;
constexpr uint32_t kStaticTriangleIndexBytes     = 2 * 3;
constexpr uint32_t kStaticSkinnedAttributeMask   = 0x01800000;
constexpr uint32_t kStaticSkinFloats             = 8;
constexpr uint32_t kStaticWideIndexFlag          = 1;
constexpr uint32_t kStaticExtraFieldFlag         = 2;

enum class StaticHeaderFieldRole
{
    Reserved,
    MaterialPathCount,
    GeometryChunkCount,
    GeometryAndMaterialPathCount,
};

enum class StaticMaterialPathLayout
{
    InterleavedPerChunk,
    PrefixedSkinVariantTable,
};

enum class StaticChunkLayout
{
    LegacyReservedVertexBlock,
    Version23Chunk,
};

struct StaticMdlFormat {
    int32_t                  version { 0 };
    StaticHeaderFieldRole    second_field { StaticHeaderFieldRole::Reserved };
    StaticHeaderFieldRole    third_field { StaticHeaderFieldRole::GeometryChunkCount };
    StaticMaterialPathLayout material_layout { StaticMaterialPathLayout::InterleavedPerChunk };
    StaticChunkLayout        chunk_layout { StaticChunkLayout::LegacyReservedVertexBlock };
};

struct StaticMdlHeader {
    uint32_t mdl_flag { 0 };
    uint32_t reserved { 0 };
    uint32_t material_path_count { 0 };
    uint32_t geometry_chunk_count { 0 };
    StaticMaterialPathLayout material_layout { StaticMaterialPathLayout::InterleavedPerChunk };
    StaticChunkLayout        chunk_layout { StaticChunkLayout::LegacyReservedVertexBlock };

    bool HasPrefixedMaterialTable() const {
        return material_layout == StaticMaterialPathLayout::PrefixedSkinVariantTable;
    }

    bool UsesSkinVariantMaterials() const {
        return material_layout == StaticMaterialPathLayout::PrefixedSkinVariantTable;
    }

    bool UsesVersion23Chunks() const {
        return chunk_layout == StaticChunkLayout::Version23Chunk;
    }
};

constexpr std::array<StaticMdlFormat, 3> kStaticMdlFormats {{
    { 4,
      StaticHeaderFieldRole::MaterialPathCount,
      StaticHeaderFieldRole::GeometryChunkCount,
      StaticMaterialPathLayout::PrefixedSkinVariantTable,
      StaticChunkLayout::LegacyReservedVertexBlock },
    { 14,
      StaticHeaderFieldRole::Reserved,
      StaticHeaderFieldRole::GeometryAndMaterialPathCount,
      StaticMaterialPathLayout::InterleavedPerChunk,
      StaticChunkLayout::LegacyReservedVertexBlock },
    { 23,
      StaticHeaderFieldRole::MaterialPathCount,
      StaticHeaderFieldRole::GeometryChunkCount,
      StaticMaterialPathLayout::InterleavedPerChunk,
      StaticChunkLayout::Version23Chunk },
}};

bool CanReadBytes(const fs::IBinaryStream& f, uint64_t byte_count);
bool ReadBoundedString(fs::IBinaryStream& f, std::string& value);

WPPuppet::PlayMode ToPlayMode(std::string_view m) {
    if (m == "loop" || m.empty()) return WPPuppet::PlayMode::Loop;
    if (m == "mirror") return WPPuppet::PlayMode::Mirror;
    if (m == "single") return WPPuppet::PlayMode::Single;

    LOG_ERROR("unknown puppet animation play mode \"%s\"", m.data());
    assert(m == "loop");
    return WPPuppet::PlayMode::Loop;
}

int32_t SeekNextMDLVersion(fs::IBinaryStream& f, std::string_view prefix) {
    const auto start = f.Tell();
    const auto end = f.Size();
    for (auto pos = start; pos + 9 <= end; ++pos) {
        f.SeekSet(pos);
        auto ver = ReadVersion(prefix, f);
        if (ver > 0)
            return ver;
    }
    f.SeekSet(start);
    return 0;
}

bool SeekNextMDLSection(fs::IBinaryStream& f, std::span<const std::string_view> prefixes) {
    const auto start = f.Tell();
    const auto end = f.Size();
    idx best_pos = -1;
    for (const auto prefix : prefixes) {
        for (auto pos = start; pos + 9 <= end; ++pos) {
            f.SeekSet(pos);
            auto ver = ReadVersion(prefix, f);
            if (ver > 0) {
                if (best_pos < 0 || pos < best_pos)
                    best_pos = pos;
                break;
            }
        }
    }
    if (best_pos >= 0) {
        f.SeekSet(best_pos);
        return true;
    }
    f.SeekSet(start);
    return false;
}

uint32_t StaticAttributeFlag(uint32_t mdl_flag) { return mdl_flag & 0xffu; }

bool StaticVertexHasSkinAttributes(uint32_t mdl_flag) {
    return (mdl_flag & kStaticSkinnedAttributeMask) != 0;
}

uint32_t StaticAttributeFloatCount(uint32_t mdl_flag) {
    switch (StaticAttributeFlag(mdl_flag)) {
    case kStaticPositionTexcoordFlag: return kStaticPositionTexcoordFloats;
    case kStaticNormalFlag: return kStaticNormalFloats;
    case kStaticTangentSpaceFlag: return kStaticTangentSpaceFloats;
    case kStaticTangentSpaceSecondUvFlag: return kStaticTangentSpaceSecondUvFloats;
    default: return 0;
    }
}

constexpr std::array<uint32_t, 26> kOfficialVertexMasks {{
    0x00000001, 0x00010000, 0x02000000, 0x00000002, 0x00000004, 0x00800000,
    0x01000000, 0x00000008, 0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800, 0x00001000, 0x00002000,
    0x00004000, 0x00020000, 0x00040000, 0x00080000, 0x00100000, 0x00200000,
    0x00400000, 0x00008000,
}};
constexpr std::array<uint32_t, 26> kOfficialVertexSizes {{
    12, 16, 12, 12, 16, 16, 16, 8, 12, 16, 8, 12, 16, 8, 12, 16, 8, 12, 16, 8, 12, 16, 8, 12, 16, 16,
}};

uint32_t OfficialVertexStride(uint32_t mdl_flag) {
    // Attribute bits are independent; stride is the sum of each set bit's payload size
    // in table order. Known scene flags (9, 11, 15, 39, and the skinned 0x0180000F
    // variant) match the older exact-flag float counts used to unpack vertices.
    uint32_t stride = 0;
    for (size_t i = 0; i < kOfficialVertexMasks.size(); i++) {
        if ((mdl_flag & kOfficialVertexMasks[i]) != 0) stride += kOfficialVertexSizes[i];
    }
    return stride;
}

const char* OfficialVertexAttrName(uint32_t mask) {
    switch (mask) {
    case 0x00000001: return WE_IN_POSITION.data();
    case 0x00000002: return WE_IN_NORMAL.data();
    case 0x00000004: return WE_IN_TANGENT4.data();
    case 0x00800000: return WE_IN_BLENDINDICES.data();
    case 0x01000000: return WE_IN_BLENDWEIGHTS.data();
    case 0x00000008: return WE_IN_TEXCOORD.data();
    case 0x00000020: return WE_IN_TEXCOORDVEC4.data();
    default: return nullptr;
    }
}

VertexType OfficialVertexAttrType(uint32_t mask, uint32_t size) {
    if (mask == 0x00800000) return VertexType::UINT4;
    switch (size) {
    case 8: return VertexType::FLOAT2;
    case 12: return VertexType::FLOAT3;
    default: return VertexType::FLOAT4;
    }
}

std::vector<SceneVertexArray::SceneVertexAttribute>
BuildOfficialStaticVertexAttributes(uint32_t mdl_flag) {
    std::vector<SceneVertexArray::SceneVertexAttribute> attrs;
    usize offset            = 0;
    usize texcoord_vec4_off = 0;
    bool  has_texcoord      = false;
    bool  has_texcoord_c2   = false;
    bool  has_texcoord_vec4 = false;

    for (size_t i = 0; i < kOfficialVertexMasks.size(); i++) {
        const uint32_t mask = kOfficialVertexMasks[i];
        if ((mdl_flag & mask) == 0) continue;
        const uint32_t size = kOfficialVertexSizes[i];
        SceneVertexArray::SceneVertexAttribute attr;
        attr.padding = false;
        attr.type    = OfficialVertexAttrType(mask, size);
        if (const char* name = OfficialVertexAttrName(mask); name != nullptr) {
            attr.name = name;
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "_mdl_0x%08x", mask);
            attr.name = buf;
        }
        if (attr.name == WE_IN_TEXCOORD) {
            has_texcoord = true;
        } else if (attr.name == WE_IN_TEXCOORDC2) {
            has_texcoord_c2 = true;
        } else if (attr.name == WE_IN_TEXCOORDVEC4) {
            has_texcoord_vec4 = true;
            texcoord_vec4_off = offset;
        }
        attrs.push_back(std::move(attr));
        offset += size;
    }

    auto add_alias = [&](std::string_view name, VertexType type, usize byte_offset) {
        SceneVertexArray::SceneVertexAttribute alias;
        alias.name              = std::string(name);
        alias.type              = type;
        alias.padding           = false;
        alias.alias             = true;
        alias.alias_byte_offset = byte_offset;
        attrs.push_back(std::move(alias));
    };
    // Flag 39 stores a_TexCoordVec4 instead of a_TexCoord. generic4 still declares
    // a_TexCoord; LIGHTMAP generic declares a_TexCoordVec4. Aliases share the file slot
    // and do not add stride. Flag 15 has only the 8-byte a_TexCoord at the end of 48 B.
    if (has_texcoord_vec4) {
        if (! has_texcoord)
            add_alias(WE_IN_TEXCOORD, VertexType::FLOAT2, texcoord_vec4_off);
        if (! has_texcoord_c2)
            add_alias(WE_IN_TEXCOORDC2, VertexType::FLOAT2, texcoord_vec4_off + 8);
    }
    return attrs;
}

uint32_t StaticVertexFloatCount(uint32_t mdl_flag) {
    const uint32_t attribute_floats = StaticAttributeFloatCount(mdl_flag);
    if (attribute_floats == 0) return 0;
    return attribute_floats + (StaticVertexHasSkinAttributes(mdl_flag) ? kStaticSkinFloats : 0);
}

const StaticMdlFormat* FindStaticMdlFormat(int32_t mdl_version) {
    const auto it = std::find_if(kStaticMdlFormats.begin(),
                                 kStaticMdlFormats.end(),
                                 [mdl_version](const StaticMdlFormat& format) {
                                     return format.version == mdl_version;
                                 });
    return it != kStaticMdlFormats.end() ? &*it : nullptr;
}

void ApplyStaticHeaderField(StaticMdlHeader& header,
                            StaticHeaderFieldRole role,
                            uint32_t value) {
    switch (role) {
    case StaticHeaderFieldRole::Reserved:
        header.reserved = value;
        break;
    case StaticHeaderFieldRole::MaterialPathCount:
        header.material_path_count = value;
        break;
    case StaticHeaderFieldRole::GeometryChunkCount:
        header.geometry_chunk_count = value;
        break;
    case StaticHeaderFieldRole::GeometryAndMaterialPathCount:
        header.geometry_chunk_count = value;
        header.material_path_count  = value;
        break;
    }
}

bool ReadStaticMdlHeader(fs::IBinaryStream& f,
                         int32_t            mdl_version,
                         std::string_view   path,
                         StaticMdlHeader&   header) {
    header.mdl_flag = f.ReadUint32();
    const uint32_t second_header_field = f.ReadUint32();
    const uint32_t third_header_field  = f.ReadUint32();

    const auto* format = FindStaticMdlFormat(mdl_version);
    if (format == nullptr) {
        LOG_ERROR("static mdl unsupported header version path='%.*s' version=%d flag=%u raw-field-1=%u "
                  "raw-field-2=%u",
                  static_cast<int>(path.size()),
                  path.data(),
                  mdl_version,
                  header.mdl_flag,
                  second_header_field,
                  third_header_field);
        return false;
    }

    // The static MDL header has only two numeric slots after the vertex flag, but those slots have
    // different meanings across format versions. A small format descriptor keeps the parse policy
    // data-driven and prevents version-specific branches from leaking into the chunk reader.
    header.material_layout = format->material_layout;
    header.chunk_layout    = format->chunk_layout;
    ApplyStaticHeaderField(header, format->second_field, second_header_field);
    ApplyStaticHeaderField(header, format->third_field, third_header_field);
    return true;
}

std::vector<std::string> ReadPrefixedStaticMaterialPaths(fs::IBinaryStream& f,
                                                         const StaticMdlHeader& header) {
    std::vector<std::string> material_paths;
    if (! header.HasPrefixedMaterialTable()) return material_paths;

    material_paths.reserve(header.material_path_count);
    for (uint32_t material_index = 0; material_index < header.material_path_count;
         material_index++) {
        // Prefixed material tables belong to the model header, not to individual chunk byte blocks.
        // Reading the whole table up front lets the chunk reader stay focused on geometry bytes and
        // lets the scene material resolver apply the skin index later.
        material_paths.push_back(f.ReadStr());
    }
    return material_paths;
}

std::string ReadStaticChunkMaterialPath(fs::IBinaryStream& f,
                                        const StaticMdlHeader& header,
                                        const std::vector<std::string>& prefixed_material_paths,
                                        uint32_t chunk_index) {
    if (! header.HasPrefixedMaterialTable()) return f.ReadStr();

    // For prefixed material tables, the first entries are valid fallback materials for geometry
    // chunks, while the full table is retained as skin variants on the chunk. The invariant below
    // is validated before parsing begins so this index is stable and version-agnostic.
    return prefixed_material_paths[chunk_index];
}

void UpdateStaticBounds(WPMdl::StaticChunk& chunk) {
    const uint32_t stride = chunk.vertex_stride;
    if (stride < 12 || chunk.vertex_blob.size() < stride) return;

    const size_t count = chunk.vertex_blob.size() / stride;
    const float* first = reinterpret_cast<const float*>(chunk.vertex_blob.data());
    chunk.bounds_min   = { first[0], first[1], first[2] };
    chunk.bounds_max   = chunk.bounds_min;
    for (size_t i = 0; i < count; ++i) {
        const float* vertex =
            reinterpret_cast<const float*>(chunk.vertex_blob.data() + i * stride);
        for (uint j = 0; j < 3; j++) {
            chunk.bounds_min[j] = std::min(chunk.bounds_min[j], vertex[j]);
            chunk.bounds_max[j] = std::max(chunk.bounds_max[j], vertex[j]);
        }
    }
}

bool ReadStaticChunk(fs::IBinaryStream& f,
                     uint32_t          mdl_flag,
                     std::string       material_json_file,
                     WPMdl::StaticChunk& chunk) {
    const uint32_t vertex_float_count = StaticVertexFloatCount(mdl_flag);
    if (vertex_float_count == 0) {
        LOG_ERROR("static mdl has unknown vertex flag %u before material '%s'",
                  mdl_flag,
                  material_json_file.c_str());
        return false;
    }

    f.ReadInt32(); // Static MDLV0014 chunks carry a reserved zero before the vertex byte block.

    const uint32_t vertex_size = f.ReadUint32();
    const uint32_t vertex_stride = vertex_float_count * sizeof(float);
    if (vertex_size == 0 || vertex_size % vertex_stride != 0) {
        LOG_ERROR("static mdl material '%s' has unsupported vertex byte size %u for stride %u",
                  material_json_file.c_str(),
                  vertex_size,
                  vertex_stride);
        return false;
    }

    chunk.material_json_file = std::move(material_json_file);
    chunk.vertex_flag        = mdl_flag;
    chunk.vertex_stride      = vertex_stride;
    chunk.vertex_blob.resize(vertex_size);
    if (f.Read(chunk.vertex_blob.data(), vertex_size) != vertex_size) {
        LOG_ERROR("static mdl material '%s' vertex blob is truncated",
                  chunk.material_json_file.c_str());
        return false;
    }

    const uint32_t indices_size = f.ReadUint32();
    if (indices_size == 0 || indices_size % kStaticTriangleIndexBytes != 0) {
        LOG_ERROR("static mdl material '%s' has unsupported index byte size %u",
                  chunk.material_json_file.c_str(),
                  indices_size);
        return false;
    }

    chunk.indices.resize(indices_size / kStaticTriangleIndexBytes);
    for (auto& index : chunk.indices) {
        for (auto& v : index) v = f.ReadUint16();
    }
    UpdateStaticBounds(chunk);
    return true;
}

bool SkipSizedBlob(fs::IBinaryStream& f, uint32_t chunk_index, std::string_view path,
                   std::string_view material, const char* field) {
    if (! CanReadBytes(f, sizeof(uint32_t))) {
        LOG_ERROR("static mdl v23 chunk %u %s size is truncated: path='%.*s' material='%s'",
                  chunk_index,
                  field,
                  static_cast<int>(path.size()),
                  path.data(),
                  material.data());
        return false;
    }
    const uint32_t byte_size = f.ReadUint32();
    if (! CanReadBytes(f, byte_size) || ! f.SeekCur(static_cast<idx>(byte_size))) {
        LOG_ERROR("static mdl v23 chunk %u %s payload is truncated: path='%.*s' material='%s' "
                  "bytes=%u",
                  chunk_index,
                  field,
                  static_cast<int>(path.size()),
                  path.data(),
                  material.data(),
                  byte_size);
        return false;
    }
    return true;
}

bool SkipStaticV23ChunkTail(fs::IBinaryStream& f, uint32_t chunk_index, std::string_view path,
                            const std::string& material, size_t vertex_count) {
    if (! CanReadBytes(f, 2)) {
        LOG_ERROR("static mdl v23 chunk %u aux/part flags are truncated: path='%.*s' "
                  "material='%s'",
                  chunk_index,
                  static_cast<int>(path.size()),
                  path.data(),
                  material.c_str());
        return false;
    }

    const uint8_t has_aux = f.ReadUint8();
    if (has_aux > 1) {
        LOG_ERROR("static mdl v23 chunk %u has invalid aux flag %u: path='%.*s' material='%s'",
                  chunk_index,
                  has_aux,
                  static_cast<int>(path.size()),
                  path.data(),
                  material.c_str());
        return false;
    }
    if (has_aux != 0) {
        if (! CanReadBytes(f, sizeof(uint32_t))) {
            LOG_ERROR("static mdl v23 chunk %u aux header is truncated: path='%.*s' "
                      "material='%s'",
                      chunk_index,
                      static_cast<int>(path.size()),
                      path.data(),
                      material.c_str());
            return false;
        }
        f.ReadUint32();
        if (! SkipSizedBlob(f, chunk_index, path, material, "aux-positions")) return false;
    }

    const uint8_t has_parts = f.ReadUint8();
    if (has_parts > 1) {
        LOG_ERROR("static mdl v23 chunk %u has invalid part flag %u: path='%.*s' material='%s'",
                  chunk_index,
                  has_parts,
                  static_cast<int>(path.size()),
                  path.data(),
                  material.c_str());
        return false;
    }
    if (has_parts != 0 && ! SkipSizedBlob(f, chunk_index, path, material, "parts")) {
        return false;
    }

    if (! CanReadBytes(f, sizeof(uint32_t))) {
        LOG_ERROR("static mdl v23 chunk %u record count is truncated: path='%.*s' material='%s'",
                  chunk_index,
                  static_cast<int>(path.size()),
                  path.data(),
                  material.c_str());
        return false;
    }
    const uint32_t rec_count = f.ReadUint32();
    for (uint32_t rec_index = 0; rec_index < rec_count; rec_index++) {
        if (! CanReadBytes(f, sizeof(uint64_t) + sizeof(uint32_t))) {
            LOG_ERROR("static mdl v23 chunk %u record %u is truncated: path='%.*s' material='%s'",
                      chunk_index,
                      rec_index,
                      static_cast<int>(path.size()),
                      path.data(),
                      material.c_str());
            return false;
        }
        f.ReadUint64();
        std::string rec_name;
        if (! ReadBoundedString(f, rec_name)) {
            LOG_ERROR("static mdl v23 chunk %u record %u name is truncated: path='%.*s' "
                      "material='%s'",
                      chunk_index,
                      rec_index,
                      static_cast<int>(path.size()),
                      path.data(),
                      material.c_str());
            return false;
        }
        f.ReadUint32();
        for (const char* list_name : { "record-a", "record-b" }) {
            if (! CanReadBytes(f, sizeof(uint32_t))) {
                LOG_ERROR("static mdl v23 chunk %u record %u %s count is truncated: "
                          "path='%.*s' material='%s' name='%s'",
                          chunk_index,
                          rec_index,
                          list_name,
                          static_cast<int>(path.size()),
                          path.data(),
                          material.c_str(),
                          rec_name.c_str());
                return false;
            }
            const uint32_t count = f.ReadUint32();
            const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(uint32_t);
            if (! CanReadBytes(f, bytes) || ! f.SeekCur(static_cast<idx>(bytes))) {
                LOG_ERROR("static mdl v23 chunk %u record %u %s payload is truncated: "
                          "path='%.*s' material='%s' name='%s' count=%u",
                          chunk_index,
                          rec_index,
                          list_name,
                          static_cast<int>(path.size()),
                          path.data(),
                          material.c_str(),
                          rec_name.c_str(),
                          count);
                return false;
            }
        }
    }
    (void)vertex_count;
    return true;
}

bool ReadStaticV23Chunk(fs::IBinaryStream& f, std::string material_json_file,
                        uint32_t chunk_index, std::string_view path,
                        WPMdl::StaticChunk& chunk) {
    if (! CanReadBytes(f, sizeof(uint32_t) + sizeof(float) * 6 + sizeof(uint32_t) * 2)) {
        LOG_ERROR("static mdl v23 chunk %u header is truncated: path='%.*s' material='%s'",
                  chunk_index,
                  static_cast<int>(path.size()),
                  path.data(),
                  material_json_file.c_str());
        return false;
    }

    const uint32_t chunk_info = f.ReadUint32();
    if ((chunk_info & kStaticExtraFieldFlag) != 0) {
        if (! CanReadBytes(f, sizeof(uint32_t))) {
            LOG_ERROR("static mdl v23 chunk %u extra field is truncated: path='%.*s' "
                      "material='%s'",
                      chunk_index,
                      static_cast<int>(path.size()),
                      path.data(),
                      material_json_file.c_str());
            return false;
        }
        f.ReadUint32();
    }
    for (auto& v : chunk.bounds_min) v = f.ReadFloat();
    for (auto& v : chunk.bounds_max) v = f.ReadFloat();

    const uint32_t vertex_flag        = f.ReadUint32();
    const uint32_t vertex_size        = f.ReadUint32();
    const uint32_t vertex_float_count = StaticVertexFloatCount(vertex_flag);
    const uint32_t official_stride    = OfficialVertexStride(vertex_flag);
    if (vertex_float_count == 0 || official_stride == 0 ||
        official_stride != vertex_float_count * sizeof(float)) {
        LOG_ERROR("static mdl v23 chunk %u has unknown vertex flag 0x%x: path='%.*s' "
                  "material='%s' official-stride=%u unpack-floats=%u",
                  chunk_index,
                  vertex_flag,
                  static_cast<int>(path.size()),
                  path.data(),
                  material_json_file.c_str(),
                  official_stride,
                  vertex_float_count);
        return false;
    }

    const uint32_t vertex_stride = official_stride;
    if (vertex_size == 0 || vertex_size % vertex_stride != 0 ||
        ! CanReadBytes(f, vertex_size)) {
        LOG_ERROR("static mdl v23 chunk %u has unsupported vertex byte size %u for stride %u: "
                  "path='%.*s' material='%s' flag=0x%x",
                  chunk_index,
                  vertex_size,
                  vertex_stride,
                  static_cast<int>(path.size()),
                  path.data(),
                  material_json_file.c_str(),
                  vertex_flag);
        return false;
    }

    chunk.material_json_file = std::move(material_json_file);
    chunk.vertex_flag        = vertex_flag;
    chunk.vertex_stride      = vertex_stride;
    chunk.vertex_blob.resize(vertex_size);
    if (f.Read(chunk.vertex_blob.data(), vertex_size) != vertex_size) {
        LOG_ERROR("static mdl v23 chunk %u vertex blob is truncated: path='%.*s' material='%s'",
                  chunk_index,
                  static_cast<int>(path.size()),
                  path.data(),
                  chunk.material_json_file.c_str());
        return false;
    }

    if (! CanReadBytes(f, sizeof(uint32_t))) {
        LOG_ERROR("static mdl v23 chunk %u index size is truncated: path='%.*s' material='%s'",
                  chunk_index,
                  static_cast<int>(path.size()),
                  path.data(),
                  chunk.material_json_file.c_str());
        return false;
    }

    const uint32_t vertex_count =
        chunk.vertex_stride > 0
            ? static_cast<uint32_t>(chunk.vertex_blob.size() / chunk.vertex_stride)
            : 0;
    const uint32_t indices_size     = f.ReadUint32();
    const bool     wide_indices     = (chunk_info & kStaticWideIndexFlag) != 0;
    const uint32_t index_value_bytes = wide_indices ? sizeof(uint32_t) : sizeof(uint16_t);
    const uint32_t triangle_bytes    = index_value_bytes * 3;
    chunk.index_element_bytes        = index_value_bytes;
    if (indices_size == 0 || indices_size % triangle_bytes != 0 ||
        ! CanReadBytes(f, indices_size)) {
        LOG_ERROR("static mdl v23 chunk %u has unsupported index byte size %u (width=%u): "
                  "path='%.*s' material='%s' verts=%u info=%u",
                  chunk_index,
                  indices_size,
                  index_value_bytes,
                  static_cast<int>(path.size()),
                  path.data(),
                  chunk.material_json_file.c_str(),
                  vertex_count,
                  chunk_info);
        return false;
    }

    chunk.indices.resize(indices_size / triangle_bytes);
    uint32_t max_index = 0;
    for (auto& index : chunk.indices) {
        for (auto& v : index) {
            v = wide_indices ? f.ReadUint32() : f.ReadUint16();
            max_index = std::max(max_index, v);
        }
    }
    if (max_index >= vertex_count) {
        LOG_ERROR("static mdl v23 chunk %u index %u is outside vertex count %u: path='%.*s' "
                  "material='%s' stride=%u width=%u",
                  chunk_index,
                  max_index,
                  vertex_count,
                  static_cast<int>(path.size()),
                  path.data(),
                  chunk.material_json_file.c_str(),
                  vertex_stride,
                  index_value_bytes);
        return false;
    }

    if (! SkipStaticV23ChunkTail(f, chunk_index, path, chunk.material_json_file, vertex_count)) {
        return false;
    }
    return true;
}

bool CanReadBytes(const fs::IBinaryStream& f, uint64_t byte_count) {
    const auto position = f.Tell();
    const auto size     = f.Size();
    return position >= 0 && size >= position &&
           byte_count <= static_cast<uint64_t>(size - position);
}

bool ReadBoundedString(fs::IBinaryStream& f, std::string& value) {
    value.clear();
    while (CanReadBytes(f, 1)) {
        const char c = static_cast<char>(f.ReadUint8());
        if (c == '\0') return true;
        value.push_back(c);
    }
    return false;
}

bool ReadPuppetPartsAndMasks(fs::IBinaryStream& f,
                             std::string_view   path,
                             uint32_t           vertex_count,
                             WPMdl&             mdl) {
    mdl.parts.clear();
    mdl.masks.clear();

    if (mdl.mdlv >= 21) {
        if (! CanReadBytes(f, sizeof(uint8_t))) {
            LOG_ERROR("puppet mdl auxiliary-position flag is truncated: %.*s",
                      static_cast<int>(path.size()),
                      path.data());
            return false;
        }

        const uint8_t has_auxiliary_positions = f.ReadUint8();
        if (has_auxiliary_positions > 1) {
            LOG_ERROR("puppet mdl has invalid auxiliary-position flag %u: %.*s",
                      has_auxiliary_positions,
                      static_cast<int>(path.size()),
                      path.data());
            return false;
        }
        if (has_auxiliary_positions != 0) {
            if (! CanReadBytes(f, sizeof(uint32_t) * 2)) {
                LOG_ERROR("puppet mdl auxiliary-position header is truncated: %.*s",
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }

            const uint32_t stream_count = f.ReadUint32();
            const uint32_t payload_size = f.ReadUint32();
            const uint64_t expected_size =
                static_cast<uint64_t>(vertex_count) * 3 * sizeof(float);
            if (stream_count != 1 || payload_size != expected_size ||
                ! CanReadBytes(f, payload_size)) {
                LOG_ERROR("puppet mdl has invalid auxiliary-position payload: %.*s "
                          "streams=%u bytes=%u expected=%llu remaining=%lld",
                          static_cast<int>(path.size()),
                          path.data(),
                          stream_count,
                          payload_size,
                          static_cast<unsigned long long>(expected_size),
                          static_cast<long long>(f.Size() - f.Tell()));
                return false;
            }

            // MDLV0021 and later partitioned puppet meshes can store a second xyz position for every
            // vertex after the index buffer. The primary stream already contains the runtime-skinned
            // position and texture coordinates; this authored reference-position stream belongs to
            // the part/clipping metadata and is not a Vulkan vertex attribute in the current puppet
            // shader contract. Consume it as one validated block so the following part table remains
            // exactly aligned across MDLV0021, MDLV0022, and MDLV0023 files.
            if (! f.SeekCur(payload_size)) {
                LOG_ERROR("puppet mdl failed to skip auxiliary-position payload: %.*s",
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }
        }
    }

    if (mdl.mdlv >= 21) {
        if (! CanReadBytes(f, sizeof(uint8_t))) {
            LOG_ERROR("puppet mdl part-table flag is truncated: %.*s",
                      static_cast<int>(path.size()),
                      path.data());
            return false;
        }

        const uint8_t has_parts = f.ReadUint8();
        if (has_parts > 1) {
            LOG_ERROR("puppet mdl has invalid part-table flag %u: %.*s",
                      has_parts,
                      static_cast<int>(path.size()),
                      path.data());
            return false;
        }
        if (has_parts != 0) {
            if (! CanReadBytes(f, sizeof(uint32_t))) {
                LOG_ERROR("puppet mdl part-table size is truncated: %.*s",
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }

            constexpr uint32_t kPartRecordBytes = sizeof(uint32_t) * 4;
            const uint32_t     parts_size       = f.ReadUint32();
            if (parts_size % kPartRecordBytes != 0 || ! CanReadBytes(f, parts_size)) {
                LOG_ERROR("puppet mdl has invalid part-table byte size %u: %.*s",
                          parts_size,
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }

            const uint32_t scalar_index_count = static_cast<uint32_t>(mdl.indices.size() * 3);
            const uint32_t part_count         = parts_size / kPartRecordBytes;
            uint32_t       next_first_index   = 0;
            mdl.parts.reserve(part_count);
            for (uint32_t part_index = 0; part_index < part_count; part_index++) {
                WPMdl::Part part;
                part.id                    = f.ReadUint32();
                const uint32_t reserved    = f.ReadUint32();
                part.first_index           = f.ReadUint32();
                part.index_count           = f.ReadUint32();
                const bool range_in_bounds =
                    part.first_index <= scalar_index_count &&
                    part.index_count <= scalar_index_count - part.first_index;
                if (reserved != 0 || ! range_in_bounds || part.index_count % 3 != 0 ||
                    part.first_index != next_first_index) {
                    LOG_ERROR("puppet mdl has invalid part %u: %.*s id=%u reserved=%u "
                              "first-index=%u index-count=%u expected-first=%u total-indices=%u",
                              part_index,
                              static_cast<int>(path.size()),
                              path.data(),
                              part.id,
                              reserved,
                              part.first_index,
                              part.index_count,
                              next_first_index,
                              scalar_index_count);
                    return false;
                }
                next_first_index += part.index_count;
                mdl.parts.push_back(part);
            }
            if (next_first_index != scalar_index_count) {
                LOG_ERROR("puppet mdl part table does not cover its index buffer: %.*s "
                          "covered=%u total=%u",
                          static_cast<int>(path.size()),
                          path.data(),
                          next_first_index,
                          scalar_index_count);
                return false;
            }
        }
    }

    if (mdl.mdlv >= 22) {
        if (! CanReadBytes(f, sizeof(uint32_t))) {
            LOG_ERROR("puppet mdl mask count is truncated: %.*s",
                      static_cast<int>(path.size()),
                      path.data());
            return false;
        }

        const uint32_t mask_count = f.ReadUint32();
        mdl.masks.reserve(mask_count);
        for (uint32_t mask_index = 0; mask_index < mask_count; mask_index++) {
            if (! CanReadBytes(f, sizeof(uint32_t) * 2)) {
                LOG_ERROR("puppet mdl mask %u header is truncated: %.*s",
                          mask_index,
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }

            WPMdl::MaskBlock mask;
            f.ReadUint32(); // unknown mask record value
            const uint32_t reserved_before = f.ReadUint32();
            if (! ReadBoundedString(f, mask.material) ||
                ! CanReadBytes(f, sizeof(uint32_t) * 2)) {
                LOG_ERROR("puppet mdl mask %u material block is truncated: %.*s",
                          mask_index,
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }

            const uint32_t reserved_after = f.ReadUint32();
            const uint32_t clipped_count  = f.ReadUint32();
            if (clipped_count > mdl.parts.size() ||
                ! CanReadBytes(f, static_cast<uint64_t>(clipped_count) * sizeof(uint32_t))) {
                LOG_ERROR("puppet mdl mask %u has invalid clipped-part count %u: %.*s",
                          mask_index,
                          clipped_count,
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }
            mask.clipped_part_indices.resize(clipped_count);
            for (auto& part_index : mask.clipped_part_indices) {
                part_index = f.ReadUint32();
                if (part_index >= mdl.parts.size()) {
                    LOG_ERROR("puppet mdl mask %u references clipped part %u of %zu: %.*s",
                              mask_index,
                              part_index,
                              mdl.parts.size(),
                              static_cast<int>(path.size()),
                              path.data());
                    return false;
                }
            }

            if (! CanReadBytes(f, sizeof(uint32_t))) {
                LOG_ERROR("puppet mdl mask %u source-part count is truncated: %.*s",
                          mask_index,
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }
            const uint32_t source_count = f.ReadUint32();
            if (source_count > mdl.parts.size() ||
                ! CanReadBytes(f, static_cast<uint64_t>(source_count) * sizeof(uint32_t))) {
                LOG_ERROR("puppet mdl mask %u has invalid source-part count %u: %.*s",
                          mask_index,
                          source_count,
                          static_cast<int>(path.size()),
                          path.data());
                return false;
            }
            mask.source_part_indices.resize(source_count);
            for (auto& part_index : mask.source_part_indices) {
                part_index = f.ReadUint32();
                if (part_index >= mdl.parts.size()) {
                    LOG_ERROR("puppet mdl mask %u references source part %u of %zu: %.*s",
                              mask_index,
                              part_index,
                              mdl.parts.size(),
                              static_cast<int>(path.size()),
                              path.data());
                    return false;
                }
            }

            if (reserved_before != 0 || reserved_after != 0) {
                LOG_INFO("puppet mdl mask %u has non-zero reserved values: %.*s before=%u "
                         "after=%u",
                         mask_index,
                         static_cast<int>(path.size()),
                         path.data(),
                         reserved_before,
                         reserved_after);
            }
            mdl.masks.push_back(std::move(mask));
        }
    }

    LOG_INFO("read puppet mesh metadata: path='%.*s' mdlv=%d vertices=%u parts=%zu masks=%zu",
             static_cast<int>(path.size()),
             path.data(),
             mdl.mdlv,
             vertex_count,
             mdl.parts.size(),
             mdl.masks.size());
    return true;
}

Eigen::Vector3f SkinPuppetVertex(const WPMdl::Vertex& vertex,
                                std::span<const Eigen::Affine3f> skinning) {
    const Eigen::Vector3f bind_position(vertex.position.data());
    Eigen::Vector3f result = Eigen::Vector3f::Zero();
    float total_weight = 0.0f;
    for (size_t influence = 0; influence < vertex.weight.size(); influence++) {
        const float weight = vertex.weight[influence];
        const uint32_t bone_index = vertex.blend_indices[influence];
        if (weight <= 0.0f || bone_index >= skinning.size()) continue;
        result += weight * (skinning[bone_index] * bind_position);
        total_weight += weight;
    }
    return total_weight > 0.0f ? result : bind_position;
}

bool AnimationHasCompleteFrames(const WPPuppet& puppet, const WPPuppet::Animation& animation) {
    if (animation.length <= 0 || animation.bframes_array.size() < puppet.bones.size()) return false;
    return std::all_of(animation.bframes_array.begin(),
                       animation.bframes_array.begin() + puppet.bones.size(),
                       [&animation](const auto& frames) {
                           return frames.frames.size() >= static_cast<usize>(animation.length);
                       });
}

PuppetBounds3D SampleAnimationBounds(const WPMdl& mdl,
                                     const WPPuppet::Animation& animation) {
    PuppetBounds3D sampled_bounds;
    if (mdl.puppet == nullptr || !AnimationHasCompleteFrames(*mdl.puppet, animation)) {
        return sampled_bounds;
    }

    std::vector<WPPuppetLayer::AnimationLayer> layers {
        WPPuppetLayer::AnimationLayer {
            .id = animation.id,
            .rate = 1.0,
            .blend = 1.0,
            .visible = true,
            .playing = false,
        },
    };
    WPPuppetLayer layer(mdl.puppet);
    layer.prepared(layers);
    auto* runtime_layer = layer.AnimationLayerState(0);
    if (runtime_layer == nullptr) return sampled_bounds;

    // Every authored keyframe is sampled directly. Setting absolute clip time before genFrame()
    // makes the envelope deterministic and independent from loop/mirror wrapping or call order.
    for (int32_t frame = 0; frame < animation.length; frame++) {
        runtime_layer->cur_time = animation.frame_time * static_cast<double>(frame);
        const auto skinning = layer.genFrame(0.0);
        for (const auto& vertex : mdl.vertexs) {
            sampled_bounds.Include(SkinPuppetVertex(vertex, skinning));
        }
    }
    return sampled_bounds;
}

void ComputePuppetAnimationBounds(WPMdl& mdl) {
    mdl.asset_bounds = {};
    for (const auto& vertex : mdl.vertexs) {
        mdl.asset_bounds.Include(Eigen::Vector3f(vertex.position.data()));
    }
    if (!mdl.asset_bounds.IsFiniteAndOrdered()) return;

    mdl.authored_pose_bounds = mdl.asset_bounds;
    if (mdl.puppet == nullptr) return;

    for (auto& animation : mdl.puppet->anims) {
        const PuppetBounds3D sampled_bounds = SampleAnimationBounds(mdl, animation);
        if (!sampled_bounds.IsFiniteAndOrdered()) {
            LOG_ERROR("ScenePuppetAuthoredEnvelope: mdl='%s' mdlv=%d mdla=%d animation=%d "
                      "name='%s' incomplete bone frames; no deterministic sample envelope",
                      mdl.source_path.c_str(),
                      mdl.mdlv,
                      mdl.mdla,
                      animation.id,
                      animation.name.c_str());
        }

        if (!animation.authored_pose_bounds.IsFiniteAndOrdered() &&
            sampled_bounds.IsFiniteAndOrdered()) {
            animation.authored_pose_bounds = sampled_bounds;
            animation.authored_bounds_source = WPPuppet::AuthoredBoundsSource::SampledFrames;
        }
        mdl.authored_pose_bounds.Include(animation.authored_pose_bounds);
    }
}
} // namespace

// bytes * size
constexpr uint32_t singile_vertex  = 4 * (3 + 4 + 4 + 2);
constexpr uint32_t singile_indices = 2 * 3;
constexpr uint32_t std_format_vertex_size_herald_value = 0x01800009;

// number of bytes in an MDAT attachment after the attachment name
constexpr uint32_t mdat_attachment_data_byte_length = 64;

// alternative consts for alternative mdl format
constexpr uint32_t alt_singile_vertex = 4 * (3 + 4 + 4 + 2 + 7);
constexpr uint32_t alt_format_vertex_size_herald_value = 0x0180000F;
constexpr uint32_t static_image_vertex_size_marker      = 0x0000000F;
constexpr uint32_t static_image_singile_vertex          = 4 * (3 + 3 + 4 + 2);

constexpr uint32_t singile_bone_frame = 4 * 9;

namespace
{
idx MdlaReadEnd(const fs::MemBinaryStream& f, uint32_t declared_end) {
    if (declared_end > 0 && declared_end <= static_cast<uint32_t>(f.Size())) {
        return static_cast<idx>(declared_end);
    }
    return f.Size();
}

bool CanReadMdla(const fs::MemBinaryStream& f, uint64_t byte_count, uint32_t declared_end) {
    const idx end = MdlaReadEnd(f, declared_end);
    return f.Tell() >= 0 && byte_count <= static_cast<uint64_t>(end - f.Tell());
}

bool PeekUint8At(fs::MemBinaryStream& f, idx offset, uint8_t& value, uint32_t declared_end) {
    const idx end = MdlaReadEnd(f, declared_end);
    if (offset < 0 || offset + 1 > end) return false;
    const idx saved = f.Tell();
    f.SeekSet(offset);
    value = f.ReadUint8();
    f.SeekSet(saved);
    return true;
}

bool PeekUint32At(fs::MemBinaryStream& f, idx offset, uint32_t& value,
                  uint32_t declared_end) {
    const idx end = MdlaReadEnd(f, declared_end);
    if (offset < 0 || offset + 4 > end) return false;
    const idx saved = f.Tell();
    f.SeekSet(offset);
    value = f.ReadUint32();
    f.SeekSet(saved);
    return true;
}

bool IsAnimTransMainSize(uint32_t byte_size, int32_t animation_length) {
    if (animation_length < 0 || byte_size == 0 || byte_size % sizeof(float) != 0) return false;
    const uint64_t samples = static_cast<uint64_t>(animation_length) + 1;
    return byte_size == samples * singile_bone_frame || byte_size == samples * sizeof(float);
}

bool NextIsAnimTransMain(fs::MemBinaryStream& f, int32_t animation_length,
                         uint32_t declared_end) {
    uint32_t byte_size = 0;
    return PeekUint32At(f, f.Tell(), byte_size, declared_end) &&
           IsAnimTransMainSize(byte_size, animation_length);
}

bool NextAfterZeroIsAnimTransMain(fs::MemBinaryStream& f, int32_t animation_length,
                                  uint32_t declared_end) {
    const idx offset = f.Tell();
    uint32_t zero = 0;
    uint32_t byte_size = 0;
    return PeekUint32At(f, offset, zero, declared_end) && zero == 0 &&
           PeekUint32At(f, offset + 4, byte_size, declared_end) &&
           IsAnimTransMainSize(byte_size, animation_length);
}

bool NextIsAnimBoneCurves(fs::MemBinaryStream& f, uint32_t declared_end) {
    const idx offset = f.Tell();
    uint8_t has_curves = 0;
    if (!PeekUint8At(f, offset, has_curves, declared_end)) return false;
    if (has_curves == 0) return true;
    if (has_curves != 1) return false;

    uint32_t zero = 0;
    uint32_t byte_size = 0;
    return PeekUint32At(f, offset + 1, zero, declared_end) && zero == 0 &&
           PeekUint32At(f, offset + 5, byte_size, declared_end) &&
           byte_size % sizeof(float) == 0;
}

bool NextIsAnimationRecordPadding(fs::MemBinaryStream& f, uint32_t declared_end) {
    const idx offset = f.Tell();
    if (declared_end == 0 || offset + 12 > static_cast<idx>(declared_end)) return false;
    uint32_t zero = 0;
    uint32_t next_id = 0;
    uint32_t next_unk = 0;
    return PeekUint32At(f, offset, zero, declared_end) && zero == 0 &&
           PeekUint32At(f, offset + 4, next_id, declared_end) && next_id != 0 &&
           PeekUint32At(f, offset + 8, next_unk, declared_end) && next_unk == 0;
}

bool ReadFloatPayload(fs::MemBinaryStream& f, uint32_t byte_size, uint32_t declared_end,
                      std::vector<float>& values, std::string_view label,
                      std::string_view path) {
    if (byte_size % sizeof(float) != 0 || !CanReadMdla(f, byte_size, declared_end)) {
        LOG_ERROR("MDLA %.*s payload has invalid byte size %u: %.*s",
                  static_cast<int>(label.size()), label.data(), byte_size,
                  static_cast<int>(path.size()), path.data());
        return false;
    }
    values.resize(byte_size / sizeof(float));
    for (auto& value : values) value = f.ReadFloat();
    return true;
}

bool ParseAnimationBoneCurves(fs::MemBinaryStream& f,
                              std::vector<WPPuppet::BoneFrameCurve>& curves,
                              uint32_t bone_count, uint32_t declared_end,
                              std::string_view label, std::string_view path) {
    if (!CanReadMdla(f, 1, declared_end)) return false;
    const uint8_t has_curves = f.ReadUint8();
    if (has_curves == 0) return true;
    if (has_curves != 1) {
        LOG_ERROR("MDLA %.*s presence flag is %u, expected 0 or 1: %.*s",
                  static_cast<int>(label.size()), label.data(), has_curves,
                  static_cast<int>(path.size()), path.data());
        return false;
    }

    curves.resize(bone_count);
    for (uint32_t bone_index = 0; bone_index < bone_count; bone_index++) {
        if (!CanReadMdla(f, 8, declared_end)) return false;
        const uint32_t reserved = f.ReadUint32();
        const uint32_t byte_size = f.ReadUint32();
        if (reserved != 0) {
            LOG_ERROR("MDLA %.*s bone %u reserved field is %u: %.*s",
                      static_cast<int>(label.size()), label.data(), bone_index, reserved,
                      static_cast<int>(path.size()), path.data());
            return false;
        }
        if (!ReadFloatPayload(f, byte_size, declared_end, curves[bone_index].values,
                              label, path)) {
            return false;
        }
    }
    return true;
}

bool ParseAnimationTransMainTrack(fs::MemBinaryStream& f, std::vector<float>& values,
                                  int32_t animation_length, uint32_t declared_end,
                                  std::string_view path) {
    if (!CanReadMdla(f, 4, declared_end)) return false;
    const uint32_t byte_size = f.ReadUint32();
    if (!IsAnimTransMainSize(byte_size, animation_length)) {
        LOG_ERROR("MDLA animation translation track has invalid byte size %u for length %d: %.*s",
                  byte_size, animation_length, static_cast<int>(path.size()), path.data());
        return false;
    }
    return ReadFloatPayload(f, byte_size, declared_end, values, "translation", path);
}

bool ParseAnimationRecord(fs::MemBinaryStream& f, WPPuppet::Animation& animation,
                          int32_t mdla_version, uint32_t declared_end,
                          std::string_view path) {
    if (!CanReadMdla(f, 8, declared_end)) return false;
    animation.id = f.ReadInt32();
    animation.unk_after_id = f.ReadUint32();
    if (animation.id <= 0 || animation.unk_after_id != 0) {
        LOG_ERROR("MDLA animation header is invalid id=%d reserved=%u: %.*s",
                  animation.id, animation.unk_after_id,
                  static_cast<int>(path.size()), path.data());
        return false;
    }

    animation.name = f.ReadStr();
    if (animation.name.empty()) animation.name = f.ReadStr();
    animation.mode = ToPlayMode(f.ReadStr());
    animation.fps = f.ReadFloat();
    animation.length = f.ReadInt32();
    const int32_t animation_reserved = f.ReadInt32();
    if (!std::isfinite(animation.fps) || animation.fps <= 0.0 || animation.length <= 0 ||
        animation_reserved != 0) {
        LOG_ERROR("MDLA animation metadata is invalid id=%d name='%s' fps=%.3f length=%d "
                  "reserved=%d: %.*s",
                  animation.id, animation.name.c_str(), animation.fps, animation.length,
                  animation_reserved, static_cast<int>(path.size()), path.data());
        return false;
    }

    if (!CanReadMdla(f, 4, declared_end)) return false;
    const uint32_t bone_count = f.ReadUint32();
    animation.bframes_array.resize(bone_count);
    for (uint32_t bone_index = 0; bone_index < bone_count; bone_index++) {
        if (!CanReadMdla(f, 8, declared_end)) return false;
        const int32_t track_reserved = f.ReadInt32();
        const uint32_t byte_size = f.ReadUint32();
        if (track_reserved != 0 || byte_size % singile_bone_frame != 0 ||
            !CanReadMdla(f, byte_size, declared_end)) {
            LOG_ERROR("MDLA animation %d bone %u track is invalid reserved=%d bytes=%u: %.*s",
                      animation.id, bone_index, track_reserved, byte_size,
                      static_cast<int>(path.size()), path.data());
            return false;
        }
        auto& frames = animation.bframes_array[bone_index].frames;
        frames.resize(byte_size / singile_bone_frame);
        for (auto& frame : frames) {
            for (auto& value : frame.position) value = f.ReadFloat();
            for (auto& value : frame.angle) value = f.ReadFloat();
            for (auto& value : frame.scale) value = f.ReadFloat();
        }
    }

    if (mdla_version >= 3) {
        if (!CanReadMdla(f, 4, declared_end)) return false;
        const uint32_t trans_flag = f.ReadUint32();
        if (trans_flag == 1) {
            auto& trans = animation.trans.emplace();
            if (!CanReadMdla(f, 4, declared_end)) return false;
            const uint32_t extra_size = f.ReadUint32();
            if (extra_size > 0) {
                if (!ReadFloatPayload(f, extra_size, declared_end, trans.extra_track,
                                      "translation-extra", path) ||
                    !CanReadMdla(f, 4, declared_end)) {
                    return false;
                }
                const uint32_t extra_reserved = f.ReadUint32();
                if (extra_reserved != 0) return false;
            }
            if (!CanReadMdla(f, 4, declared_end)) return false;
            const uint32_t main_size = f.ReadUint32();
            if (!ReadFloatPayload(f, main_size, declared_end, trans.main_track,
                                  "translation-main", path)) {
                return false;
            }
            if (extra_size > 0) {
                if (!CanReadMdla(f, 4, declared_end) || f.ReadUint32() != 0) return false;
            }
        } else if (trans_flag == 0) {
            if (NextIsAnimTransMain(f, animation.length, declared_end)) {
                auto& trans = animation.trans.emplace();
                if (!ParseAnimationTransMainTrack(f, trans.main_track, animation.length,
                                                  declared_end, path)) {
                    return false;
                }
                while (NextAfterZeroIsAnimTransMain(f, animation.length, declared_end)) {
                    if (f.ReadUint32() != 0) return false;
                    auto& tail = trans.tail_tracks.emplace_back();
                    if (!ParseAnimationTransMainTrack(f, tail, animation.length,
                                                      declared_end, path)) {
                        return false;
                    }
                }
            }
        } else {
            LOG_ERROR("MDLA animation %d translation flag is %u: %.*s",
                      animation.id, trans_flag, static_cast<int>(path.size()), path.data());
            return false;
        }
        if (!ParseAnimationBoneCurves(f, animation.blend_curves, bone_count, declared_end,
                                      "blend-curves", path)) {
            return false;
        }
    }

    if (mdla_version >= 4) {
        if (!CanReadMdla(f, 1, declared_end)) return false;
        const uint8_t has_events = f.ReadUint8();
        if (has_events > 1) return false;
        if (has_events == 1) {
            if (!CanReadMdla(f, 4, declared_end)) return false;
            const uint32_t event_count = f.ReadUint32();
            animation.v4_events.resize(event_count);
            for (auto& event : animation.v4_events) {
                if (!CanReadMdla(f, 12, declared_end)) return false;
                event.time = f.ReadFloat();
                event.flags = f.ReadUint32();
                const uint32_t byte_size = f.ReadUint32();
                if (!ReadFloatPayload(f, byte_size, declared_end, event.values,
                                      "v4-event", path)) {
                    return false;
                }
            }
        }
    }

    if (mdla_version >= 5) {
        if (!CanReadMdla(f, 6 * sizeof(float), declared_end)) return false;
        PuppetBounds3D parsed_bounds;
        parsed_bounds.valid = true;
        for (auto& value : parsed_bounds.min) value = f.ReadFloat();
        for (auto& value : parsed_bounds.max) value = f.ReadFloat();
        if (!parsed_bounds.IsFiniteAndOrdered()) {
            LOG_ERROR("MDLA animation %d name='%s' has invalid authored AABB "
                      "[%.3f %.3f %.3f]-[%.3f %.3f %.3f]: %.*s",
                      animation.id, animation.name.c_str(),
                      parsed_bounds.min.x(), parsed_bounds.min.y(), parsed_bounds.min.z(),
                      parsed_bounds.max.x(), parsed_bounds.max.y(), parsed_bounds.max.z(),
                      static_cast<int>(path.size()), path.data());
        } else {
            animation.authored_pose_bounds = parsed_bounds;
            animation.authored_bounds_source = WPPuppet::AuthoredBoundsSource::MdlaAabb;
        }
    }

    if (mdla_version == 6 && NextIsAnimBoneCurves(f, declared_end)) {
        if (!ParseAnimationBoneCurves(f, animation.scalar_curves, bone_count, declared_end,
                                      "scalar-curves", path)) {
            return false;
        }
    }

    if (!CanReadMdla(f, 4, declared_end)) return false;
    const uint32_t event_count = f.ReadUint32();
    animation.events.resize(event_count);
    for (auto& event : animation.events) {
        if (!CanReadMdla(f, 4, declared_end)) return false;
        event.time_value = f.ReadUint32();
        event.event_json = f.ReadStr();
        if (f.Tell() > MdlaReadEnd(f, declared_end)) return false;
    }

    if (NextIsAnimationRecordPadding(f, declared_end)) {
        if (f.ReadUint32() != 0) return false;
    }
    return true;
}

bool ConsumeMdlaZeroPadding(fs::MemBinaryStream& f, uint32_t declared_end,
                            std::string_view path) {
    const idx end = MdlaReadEnd(f, declared_end);
    while (f.Tell() < end) {
        if (f.ReadUint8() != 0) {
            LOG_ERROR("MDLA body contains non-zero trailing data at 0x%llx before end 0x%llx: %.*s",
                      static_cast<unsigned long long>(f.Tell() - 1),
                      static_cast<unsigned long long>(end),
                      static_cast<int>(path.size()), path.data());
            return false;
        }
    }
    return f.Tell() == end;
}

bool ReadPuppetSkeletonAndAnimations(fs::MemBinaryStream& f, std::string_view path,
                                     WPMdl& mdl) {
    const std::string str_path(path);
    mdl.mdls = ReadMDLVesion(f);
    if (mdl.mdls == 0) {
        mdl.mdls = SeekNextMDLVersion(f, "MDLS");
    }
    if (mdl.mdls == 0) {
        LOG_ERROR("failed to locate MDLS section: %s", str_path.c_str());
        return false;
    }

    const size_t bones_file_end = f.ReadUint32();
    (void)bones_file_end;

    const uint16_t bones_num = f.ReadUint16();
    f.ReadUint16(); // MDLS bone-table header value.

    mdl.puppet  = std::make_shared<WPPuppet>();
    auto& bones = mdl.puppet->bones;
    auto& anims = mdl.puppet->anims;

    bones.resize(bones_num);
    for (uint i = 0; i < bones_num; i++) {
        auto& bone = bones[i];
        bone.name = f.ReadStr();
        f.ReadInt32(); // Authored MDLS bone metadata.

        bone.parent = f.ReadUint32();
        if (bone.parent >= i && ! bone.noParent()) {
            LOG_INFO("mdl bone %u has out-of-order parent index %u, fallback to root",
                     i,
                     bone.parent);
            bone.parent = 0xFFFFFFFFu;
        }

        const uint32_t size = f.ReadUint32();
        if (size != 64) {
            LOG_ERROR("mdl unsupport bones size: %d", size);
            return false;
        }
        for (auto column : bone.transform.matrix().colwise()) {
            for (auto& value : column) value = f.ReadFloat();
        }

        f.ReadStr(); // Bone simulation JSON.
    }

    if (mdl.mdls > 1) {
        const int16_t reserved = f.ReadInt16();
        if (reserved != 0) {
            LOG_INFO("puppet: one unk is not 0, may be wrong");
        }

        const uint8_t has_transforms = f.ReadUint8();
        if (has_transforms) {
            for (uint i = 0; i < bones_num; i++) {
                for (uint j = 0; j < 16; j++) f.ReadFloat();
            }
        }
        const uint32_t metadata_count = f.ReadUint32();
        for (uint i = 0; i < metadata_count; i++) {
            for (int j = 0; j < 3; j++) f.ReadUint32();
        }

        f.ReadUint32(); // MDLS metadata field.

        const uint8_t has_offset_transforms = f.ReadUint8();
        if (has_offset_transforms) {
            for (uint i = 0; i < bones_num; i++) {
                for (uint j = 0; j < 3; j++) f.ReadFloat();
                for (uint j = 0; j < 16; j++) f.ReadFloat();
            }
        }

        const uint8_t has_indices = f.ReadUint8();
        if (has_indices) {
            for (uint i = 0; i < bones_num; i++) f.ReadUint32();
        }
    }

    {
        const auto probe_pos = f.Tell();
        bool       aligned   = false;
        for (const auto prefix : { std::string_view("MDAT"), std::string_view("MDLA") }) {
            const auto version = ReadVersion(prefix, f);
            f.SeekSet(probe_pos);
            if (version > 0) {
                aligned = true;
                break;
            }
        }
        constexpr std::array<std::string_view, 2> kAnimSections { "MDAT", "MDLA" };
        if (! aligned) SeekNextMDLSection(f, kAnimSections);
    }

    // MDAT attachment blocks may occur between MDLS and MDLA. Preserve their bone-local matrices
    // while walking to the animation table so both legacy single-mesh puppets and MDLV0023 model
    // chunks share exactly one skeleton/animation parser.
    std::string section_type;
    std::string section_version;
    do {
        if (f.Tell() >= f.Size()) {
            LOG_ERROR("failed to locate MDLA section before EOF: %s", str_path.c_str());
            return false;
        }
        const std::string section = f.ReadStr();
        if (section.length() != 8) continue;

        section_type    = section.substr(0, 4);
        section_version = section.substr(4, 4);
        if (section_type != "MDAT") continue;

        f.ReadUint32();
        const uint16_t attachment_count = f.ReadUint16();
        for (uint16_t i = 0; i < attachment_count; i++) {
            WPPuppet::Attachment attachment;
            attachment.bone_index = f.ReadUint16();
            attachment.name       = f.ReadStr();
            for (auto column : attachment.transform.matrix().colwise()) {
                for (auto& value : column) value = f.ReadFloat();
            }
            mdl.puppet->attachments.push_back(std::move(attachment));
        }
    } while (section_type != "MDLA");

    if (! section_version.empty()) {
        mdl.mdla = std::stoi(section_version);
        if (mdl.mdla != 0) {
            if (mdl.mdla < 1 || mdl.mdla > 6) {
                LOG_ERROR("unsupported MDLA version %d: %s", mdl.mdla, str_path.c_str());
                return false;
            }
            const uint32_t declared_end = f.ReadUint32();
            if (declared_end > static_cast<uint32_t>(f.Size()) ||
                (declared_end > 0 && declared_end < static_cast<uint32_t>(f.Tell()))) {
                LOG_ERROR("MDLA declared end 0x%x is outside file size 0x%llx: %s",
                          declared_end,
                          static_cast<unsigned long long>(f.Size()),
                          str_path.c_str());
                return false;
            }

            const uint32_t animation_count = f.ReadUint32();
            anims.resize(animation_count);
            for (auto& animation : anims) {
                if (! ParseAnimationRecord(f, animation, mdl.mdla, declared_end, str_path)) {
                    return false;
                }
            }
            if (! ConsumeMdlaZeroPadding(f, declared_end, str_path)) return false;
        }
    }

    mdl.puppet->prepared();
    ComputePuppetAnimationBounds(mdl);
    LOG_INFO("read puppet: mdlv: %d, nmdls: %d, mdla: %d, bones: %d, anims: %d",
             mdl.mdlv,
             mdl.mdls,
             mdl.mdla,
             mdl.puppet->bones.size(),
             mdl.puppet->anims.size());
    return true;
}
} // namespace

bool WPMdlParser::ParseStaticModel(std::string_view path, fs::VFS& vfs, WPMdl& mdl,
                                   bool load_animation_data) {
    auto str_path = std::string(path);
    auto pfile    = vfs.Open("/assets/" + str_path);
    if (! pfile) {
        LOG_ERROR("static mdl open failed: %s", str_path.c_str());
        return false;
    }

    auto memfile = fs::MemBinaryStream(*pfile);
    auto& f      = memfile;

    mdl = WPMdl {};
    mdl.source_path = str_path;
    mdl.mdlv = ReadMDLVesion(f);
    if (mdl.mdlv <= 0) {
        LOG_ERROR("static mdl version read failed: %s", str_path.c_str());
        return false;
    }

    StaticMdlHeader header;
    if (! ReadStaticMdlHeader(f, mdl.mdlv, str_path, header)) return false;

    const uint32_t mdl_flag = header.mdl_flag;
    const uint32_t chunk_count = header.geometry_chunk_count;
    if (chunk_count == 0) {
        LOG_ERROR("static mdl has no chunks: %s", str_path.c_str());
        return false;
    }
    if (header.material_path_count == 0) {
        LOG_ERROR("static mdl has no material paths: %s", str_path.c_str());
        return false;
    }
    if (header.HasPrefixedMaterialTable() && header.material_path_count < chunk_count) {
        LOG_ERROR("static mdl has fewer prefixed material paths than chunks: %s materials=%u chunks=%u",
                  str_path.c_str(),
                  header.material_path_count,
                  chunk_count);
        return false;
    }

    mdl.kind = WPMdl::MeshKind::Static;
    mdl.static_chunks.clear();
    mdl.static_chunks.reserve(chunk_count);

    const auto prefixed_material_paths = ReadPrefixedStaticMaterialPaths(f, header);

    for (uint32_t chunk_index = 0; chunk_index < chunk_count; chunk_index++) {
        std::string material_json_file;
        if (header.UsesVersion23Chunks()) {
            for (uint32_t material_index = 0; material_index < header.material_path_count;
                 material_index++) {
                std::string material_path;
                if (! ReadBoundedString(f, material_path) || material_path.empty()) {
                    LOG_ERROR("static mdl v23 chunk %u material %u is empty: %s",
                              chunk_index,
                              material_index,
                              str_path.c_str());
                    return false;
                }
                if (material_index == 0) material_json_file = std::move(material_path);
            }
        } else {
            material_json_file =
                ReadStaticChunkMaterialPath(f, header, prefixed_material_paths, chunk_index);
        }
        if (material_json_file.empty()) {
            LOG_ERROR("static mdl chunk %u has empty material path: %s",
                      chunk_index,
                      str_path.c_str());
            return false;
        }

        WPMdl::StaticChunk chunk;
        if (header.UsesVersion23Chunks()) {
            if (! ReadStaticV23Chunk(f, std::move(material_json_file), chunk_index, path, chunk)) {
                return false;
            }
        } else if (! ReadStaticChunk(f, mdl_flag, std::move(material_json_file), chunk)) {
            return false;
        }
        if (header.UsesSkinVariantMaterials()) {
            // Keep the variant table on the parsed chunk so WPSceneParser can apply the model
            // object's skin index after sidecar material remapping. This keeps binary parsing
            // independent from scene-object policy while still preserving the authored skin choice.
            chunk.material_json_variants = prefixed_material_paths;
        }
        mdl.static_chunks.push_back(std::move(chunk));
    }

    const bool has_skinning = std::any_of(
        mdl.static_chunks.begin(), mdl.static_chunks.end(), [](const WPMdl::StaticChunk& chunk) {
            return StaticVertexHasSkinAttributes(chunk.vertex_flag);
        });
    if (load_animation_data && has_skinning) {
        mdl.kind = WPMdl::MeshKind::Puppet;
        return ReadPuppetSkeletonAndAnimations(f, path, mdl);
    }
    return true;
}

bool WPMdlParser::Parse(std::string_view path, fs::VFS& vfs, WPMdl& mdl) {
    auto str_path = std::string(path);
    auto pfile    = vfs.Open("/assets/" + str_path);
    if (! pfile) return false;
    auto memfile  = fs::MemBinaryStream(*pfile);
    auto& f = memfile;

    mdl.source_path = str_path;

    mdl.mdlv = ReadMDLVesion(f);

    int32_t mdl_flag = f.ReadInt32();
    const bool static_image_mesh = mdl_flag == 9;
    if (static_image_mesh) {
        // Flag 9 image puppet files are authored static image meshes, not broken animated
        // puppets. They reuse the puppet slot to carry crop/shape geometry for image layers and
        // intentionally stop before MDLS/MDLA skeleton data, so routing them through the animated
        // puppet reader would either reject them or interpret their compact vertex stream with the
        // wrong stride.
        mdl.kind = WPMdl::MeshKind::StaticImage;
    } else {
        mdl.kind = WPMdl::MeshKind::Puppet;
    }
    f.ReadInt32(); // unk, 1
    f.ReadInt32(); // unk, 1

    mdl.mat_json_file = f.ReadStr();
    // 0    
    f.ReadInt32();

    bool alt_mdl_format = false;
    uint32_t curr = f.ReadUint32();

    auto is_alt_vertex_marker = [&](uint32_t value) {
        return value == alt_format_vertex_size_herald_value ||
               (static_image_mesh && value == static_image_vertex_size_marker);
    };

    // If the uint at the normal vertex size position is 0, this file uses a marker-delimited
    // vertex-size block. Static image puppets have their own marker value because their file
    // family is image-layer geometry rather than animated skeleton data, but the cursor contract is
    // the same: seek the marker, then read the byte size immediately after it.
    if(curr == 0){
        alt_mdl_format = true;
        while (! is_alt_vertex_marker(curr) && f.Tell() < f.Size()){
            curr = f.ReadUint32();
        }
        if (! is_alt_vertex_marker(curr)) {
            LOG_ERROR("failed to locate alternative vertex herald 0x%08x", alt_format_vertex_size_herald_value);
            return false;
        }
        curr = f.ReadUint32();
    }
    else if(curr == std_format_vertex_size_herald_value ||
            (static_image_mesh && curr == static_image_vertex_size_marker)){
        curr = f.ReadUint32();
    }

    uint32_t vertex_size = curr;
    const uint32_t vertex_stride =
        static_image_mesh ? static_image_singile_vertex
                          : (alt_mdl_format ? alt_singile_vertex : singile_vertex);
    if (vertex_size % vertex_stride != 0) {
        LOG_ERROR("unsupport mdl vertex size %d", vertex_size);
        return false;
    }

    uint32_t vertex_num = vertex_size / vertex_stride;
    mdl.vertexs.resize(vertex_num);
    for (auto& vert : mdl.vertexs) {
        if (static_image_mesh) {
            // Static image puppet meshes store position.xyz, normal.xyz, a vec4 payload, and
            // texcoord.xy. They have no skinning attributes, so synthesize a neutral bone binding
            // while preserving the authored shape and UV crop exactly for the final image layer.
            for (auto& v : vert.position) v = f.ReadFloat();
            for (int i = 0; i < 7; i++) f.ReadFloat();
            vert.blend_indices = { 0, 0, 0, 0 };
            vert.weight        = { 0.0f, 0.0f, 0.0f, 1.0f };
            for (auto& v : vert.texcoord) v = f.ReadFloat();
        } else {
            for (auto& v : vert.position) v = f.ReadFloat();
            // If using the alternative MDL format, vertexes contain 7 extra 32-bit chunks between
            // position and blend indices. They are opaque payload for animated puppet meshes and
            // must stay separate from the compact static-image path above.
            if(alt_mdl_format) {for (int i = 0; i < 7; i++) f.ReadUint32();}
            for (auto& v : vert.blend_indices) v = f.ReadUint32();
            for (auto& v : vert.weight) v = f.ReadFloat();
            for (auto& v : vert.texcoord) v = f.ReadFloat();
        }
    }

    uint32_t indices_size = f.ReadUint32();
    if (indices_size % singile_indices != 0) {
        LOG_ERROR("unsupport mdl indices size %d", indices_size);
        return false;
    }

    uint32_t indices_num = indices_size / singile_indices;
    mdl.indices.resize(indices_num);
    for (auto& id : mdl.indices) {
        for (auto& v : id) v = f.ReadUint16();
    }

    if (static_image_mesh) {
        // Static image puppet meshes end after the index buffer. Returning here makes the contract
        // explicit for callers: the mesh is valid render geometry, but there is no WPPuppet object,
        // no bone uniforms, and no animation state to attach to the scene node.
        LOG_INFO("read static image puppet: mdlv: %d, vertices: %u, indices: %u",
                 mdl.mdlv,
                 vertex_num,
                 indices_num);
        return true;
    }

    if (! ReadPuppetPartsAndMasks(f, path, vertex_num, mdl)) return false;

    return ReadPuppetSkeletonAndAnimations(f, path, mdl);
}

void WPMdlParser::GenPuppetMesh(SceneMesh& mesh, const WPMdl& mdl) {
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 },
                              { WE_IN_BLENDINDICES.data(), VertexType::UINT4 },
                              { WE_IN_BLENDWEIGHTS.data(), VertexType::FLOAT4 },
                              { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 } },
                            mdl.vertexs.size());

    std::array<float, 16> one_vert;
    auto                  to_one = [](const WPMdl::Vertex& in, decltype(one_vert)& out) {
        uint offset = 0;
        memcpy(out.data() + 4 * (offset++), in.position.data(), sizeof(in.position));
        memcpy(out.data() + 4 * (offset++), in.blend_indices.data(), sizeof(in.blend_indices));
        memcpy(out.data() + 4 * (offset++), in.weight.data(), sizeof(in.weight));
        memcpy(out.data() + 4 * (offset++), in.texcoord.data(), sizeof(in.texcoord));
    };
    for (uint i = 0; i < mdl.vertexs.size(); i++) {
        auto& v = mdl.vertexs[i];
        to_one(v, one_vert);
        vertex.SetVertexs(i, one_vert);
    }
    std::vector<uint32_t> indices;
    size_t                u16_count = mdl.indices.size() * 3;
    indices.resize(u16_count / 2 + 1);
    memcpy(indices.data(), mdl.indices.data(), u16_count * sizeof(uint16_t));

    mesh.AddVertexArray(std::move(vertex));
    mesh.AddIndexArray(SceneIndexArray(indices));

    SceneMesh::MaskedDrawPlan masked_draw;
    if (! mdl.masks.empty()) {
        // Resolve MDLV part indices at the parser boundary. Renderer code receives only concrete
        // index ranges and a stable authored draw schedule, so it never needs to understand MDLV
        // part tables or rebuild a range-to-mask lookup during Vulkan pass preparation.
        std::vector<int32_t> group_by_part(mdl.parts.size(), -1);
        masked_draw.groups.reserve(mdl.masks.size());
        for (size_t group_index = 0; group_index < mdl.masks.size(); group_index++) {
            const auto& mask = mdl.masks[group_index];
            SceneMesh::MaskedDrawGroup group;
            group.maskTexture = mask.material;
            group.maskRanges.reserve(mask.source_part_indices.size());
            for (const auto part_index : mask.source_part_indices) {
                const auto& part = mdl.parts[part_index];
                group.maskRanges.push_back(
                    { .firstIndex = part.first_index, .indexCount = part.index_count });
            }
            group.contentRanges.reserve(mask.clipped_part_indices.size());
            for (const auto part_index : mask.clipped_part_indices) {
                const auto& part = mdl.parts[part_index];
                group.contentRanges.push_back(
                    { .firstIndex = part.first_index, .indexCount = part.index_count });
                // Preserve the established last-mask-wins rule for malformed overlapping metadata.
                // Valid Wallpaper Engine assets assign a clipped part to exactly one mask group.
                group_by_part[part_index] = static_cast<int32_t>(group_index);
            }
            masked_draw.groups.push_back(std::move(group));
        }

        masked_draw.unmaskedRanges.reserve(mdl.parts.size());
        masked_draw.orderedRanges.reserve(mdl.parts.size());
        for (size_t part_index = 0; part_index < mdl.parts.size(); part_index++) {
            const auto& part = mdl.parts[part_index];
            const SceneMesh::DrawRange range {
                .firstIndex = part.first_index,
                .indexCount = part.index_count,
            };
            const auto group_index = group_by_part[part_index];
            masked_draw.orderedRanges.push_back(
                { .range = range, .groupIndex = group_index });
            if (group_index < 0) masked_draw.unmaskedRanges.push_back(range);
        }
    }
    const uint32_t bone_count = mdl.puppet != nullptr
                                    ? static_cast<uint32_t>(mdl.puppet->bones.size())
                                    : 0;
    if (! masked_draw.empty()) {
        LOG_INFO("build masked draw plan: groups=%zu unmasked-ranges=%zu ordered-ranges=%zu "
                 "bones=%u",
                 masked_draw.groups.size(),
                 masked_draw.unmaskedRanges.size(),
                 masked_draw.orderedRanges.size(),
                 bone_count);
        for (size_t group_index = 0; group_index < masked_draw.groups.size(); group_index++) {
            const auto& group = masked_draw.groups[group_index];
            LOG_INFO("masked draw group: group=%zu texture='%s' mask-ranges=%zu "
                     "content-ranges=%zu",
                     group_index,
                     group.maskTexture.c_str(),
                     group.maskRanges.size(),
                     group.contentRanges.size());
            for (const auto& range : group.maskRanges) {
                LOG_INFO("masked draw range: group=%zu role=mask first-index=%u "
                         "index-count=%u",
                         group_index,
                         range.firstIndex,
                         range.indexCount);
            }
            for (const auto& range : group.contentRanges) {
                LOG_INFO("masked draw range: group=%zu role=content first-index=%u "
                         "index-count=%u",
                         group_index,
                         range.firstIndex,
                         range.indexCount);
            }
        }
    }
    mesh.SetMaskedDraw(std::move(masked_draw));
    mesh.SetSkinning({ .boneCount = bone_count });
    mesh.SetFileImmutable(true);
    if (mdl.asset_bounds.IsFiniteAndOrdered()) {
        mesh.SetBounds(mdl.asset_bounds.min, mdl.asset_bounds.max);
    }
}

void WPMdlParser::GenStaticMesh(SceneMesh& mesh, const WPMdl::StaticChunk& chunk) {
    const uint32_t stride = chunk.vertex_stride != 0
                                ? chunk.vertex_stride
                                : OfficialVertexStride(chunk.vertex_flag);
    const size_t vertex_count = stride > 0 ? chunk.vertex_blob.size() / stride : 0;
    auto         attrs        = BuildOfficialStaticVertexAttributes(chunk.vertex_flag);
    SceneVertexArray vertex(attrs, vertex_count);
    if (vertex.OneSizeOf() != stride ||
        ! vertex.SetPackedBytes(chunk.vertex_blob)) {
        LOG_ERROR("static mesh packed layout mismatch: flag=0x%x blob=%zu stride=%u packed=%zu",
                  chunk.vertex_flag,
                  chunk.vertex_blob.size(),
                  stride,
                  vertex.OneSizeOf());
        return;
    }

    const bool need_uint32 = chunk.index_element_bytes == 4;
    if (need_uint32) {
        std::vector<uint32_t> indices;
        indices.reserve(chunk.indices.size() * 3);
        for (const auto& triangle : chunk.indices) {
            indices.insert(indices.end(), triangle.begin(), triangle.end());
        }
        mesh.AddVertexArray(std::move(vertex));
        mesh.AddIndexArray(SceneIndexArray(indices));
        mesh.SetIndexElementBytes(4);
    } else {
        std::vector<uint16_t> packed;
        packed.reserve(chunk.indices.size() * 3);
        for (const auto& triangle : chunk.indices) {
            packed.push_back(static_cast<uint16_t>(triangle[0]));
            packed.push_back(static_cast<uint16_t>(triangle[1]));
            packed.push_back(static_cast<uint16_t>(triangle[2]));
        }
        std::vector<uint32_t> indices;
        indices.resize(packed.size() / 2 + 1);
        memcpy(indices.data(), packed.data(), packed.size() * sizeof(uint16_t));
        mesh.AddVertexArray(std::move(vertex));
        mesh.AddIndexArray(SceneIndexArray(indices));
        mesh.SetIndexElementBytes(2);
    }
    mesh.SetFileImmutable(true);
    mesh.SetBounds(Eigen::Vector3f(chunk.bounds_min[0], chunk.bounds_min[1], chunk.bounds_min[2]),
                   Eigen::Vector3f(chunk.bounds_max[0], chunk.bounds_max[1], chunk.bounds_max[2]));
}

void WPMdlParser::AddPuppetShaderInfo(WPShaderInfo& info, const WPMdl& mdl) {
    info.combos["SKINNING"]  = "1";
    info.combos["BONECOUNT"] = std::to_string(mdl.puppet->bones.size());
}

void WPMdlParser::AddPuppetMatInfo(wpscene::WPMaterial& mat, const WPMdl& mdl) {
    mat.combos["SKINNING"]  = 1;
    mat.combos["BONECOUNT"] = (i32)mdl.puppet->bones.size();
    mat.use_puppet          = true;
}

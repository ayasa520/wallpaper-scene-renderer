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

using namespace wallpaper;

namespace
{

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

constexpr uint32_t singile_bone_frame = 4 * 9;

bool WPMdlParser::Parse(std::string_view path, fs::VFS& vfs, WPMdl& mdl) {
    auto str_path = std::string(path);
    auto pfile    = vfs.Open("/assets/" + str_path);
    auto memfile  = fs::MemBinaryStream(*pfile);
    if (! pfile) return false;
    auto& f = memfile;

    mdl.mdlv = ReadMDLVesion(f);

    int32_t mdl_flag = f.ReadInt32();
    if (mdl_flag == 9) {
        LOG_INFO("puppet '%s' is not complete, ignore", str_path.c_str());
        return false;
    };
    f.ReadInt32(); // unk, 1
    f.ReadInt32(); // unk, 1

    mdl.mat_json_file = f.ReadStr();
    // 0    
    f.ReadInt32();

    bool alt_mdl_format = false;
    uint32_t curr = f.ReadUint32();

    // if the uint at the normal vertex size position is 0, then this file
    // uses the alternative MDL format, therefore the actual vertex size is
    // located after the herald value, and we'll need to account for other differences later on.
    if(curr == 0){
        alt_mdl_format = true;
        while (curr != alt_format_vertex_size_herald_value && f.Tell() < f.Size()){
            curr = f.ReadUint32();
        }
        if (curr != alt_format_vertex_size_herald_value) {
            LOG_ERROR("failed to locate alternative vertex herald 0x%08x", alt_format_vertex_size_herald_value);
            return false;
        }
        curr = f.ReadUint32();
    }
    else if(curr == std_format_vertex_size_herald_value){
        curr = f.ReadUint32();
    }

    uint32_t vertex_size = curr;
    if (vertex_size % (alt_mdl_format? alt_singile_vertex : singile_vertex) != 0) {
        LOG_ERROR("unsupport mdl vertex size %d", vertex_size);
        return false;
    }

    // if using the alternative MDL format, vertexes contain 7 extra 32-bit chunks between
    // position and blend indices
    uint32_t vertex_num = vertex_size / (alt_mdl_format ? alt_singile_vertex : singile_vertex);
    mdl.vertexs.resize(vertex_num);
    for (auto& vert : mdl.vertexs) {
        for (auto& v : vert.position) v = f.ReadFloat();
        if(alt_mdl_format) {for (int i = 0; i < 7; i++) f.ReadUint32();}
        for (auto& v : vert.blend_indices) v = f.ReadUint32();
        for (auto& v : vert.weight) v = f.ReadFloat();
        for (auto& v : vert.texcoord) v = f.ReadFloat();
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

    mdl.mdls = ReadMDLVesion(f);
    if (mdl.mdls == 0) {
        mdl.mdls = SeekNextMDLVersion(f, "MDLS");
    }
    if (mdl.mdls == 0) {
        LOG_ERROR("failed to locate MDLS section");
        return false;
    }

    size_t bones_file_end = f.ReadUint32();
    (void)bones_file_end;

    uint16_t bones_num = f.ReadUint16();
    // 1 byte
    f.ReadUint16(); // unk

    mdl.puppet  = std::make_shared<WPPuppet>();
    auto& bones = mdl.puppet->bones;
    auto& anims = mdl.puppet->anims;

    bones.resize(bones_num);
    for (uint i = 0; i < bones_num; i++) {
        auto&       bone = bones[i];
        std::string name = f.ReadStr();
        f.ReadInt32(); // unk

        bone.parent = f.ReadUint32();
        if (bone.parent >= i && !bone.noParent()) {
            LOG_INFO("mdl bone %u has out-of-order parent index %u, fallback to root", i, bone.parent);
            bone.parent = 0xFFFFFFFFu;
        }

        uint32_t size = f.ReadUint32();
        if (size != 64) {
            LOG_ERROR("mdl unsupport bones size: %d", size);
            return false;
        }
        for (auto row : bone.transform.matrix().colwise()) {
            for (auto& x : row) x = f.ReadFloat();
        }

        std::string bone_simulation_json = f.ReadStr();
        /*
        auto trans = bone.transform.translation();
        LOG_INFO("trans: %f %f %f", trans[0], trans[1], trans[2]);
        */
    }

    if (mdl.mdls > 1) {
        int16_t unk = f.ReadInt16();
        if (unk != 0) {
            LOG_INFO("puppet: one unk is not 0, may be wrong");
        }

        uint8_t has_trans = f.ReadUint8();
        if (has_trans) {
            for (uint i = 0; i < bones_num; i++)
                for (uint j = 0; j < 16; j++) f.ReadFloat(); // mat
        }
        uint32_t size_unk = f.ReadUint32();
        for (uint i = 0; i < size_unk; i++)
            for (int j = 0; j < 3; j++) f.ReadUint32();

        f.ReadUint32(); // unk

        uint8_t has_offset_trans = f.ReadUint8();
        if (has_offset_trans) {
            for (uint i = 0; i < bones_num; i++) {
                for (uint j = 0; j < 3; j++) f.ReadFloat();  // like pos
                for (uint j = 0; j < 16; j++) f.ReadFloat(); // mat
            }
        }

        uint8_t has_index = f.ReadUint8();
        if (has_index) {
            for (uint i = 0; i < bones_num; i++) {
                f.ReadUint32();
            }
        }
    }

    {
        const auto probe_pos = f.Tell();
        bool aligned = false;
        for (const auto prefix : { std::string_view("MDAT"), std::string_view("MDLA") }) {
            auto ver = ReadVersion(prefix, f);
            f.SeekSet(probe_pos);
            if (ver > 0) {
                aligned = true;
                break;
            }
        }
        constexpr std::array<std::string_view, 2> kAnimSections { "MDAT", "MDLA" };
        if (!aligned)
            SeekNextMDLSection(f, kAnimSections);
    }

    // sometimes there can be one or more zero bytes and/or MDAT sections containing
    // attachments before the MDLA section, so we need to skip them
    std::string mdType = "";
    std::string mdVersion;
    
    do {
        if (f.Tell() >= f.Size()) {
            LOG_ERROR("failed to locate MDLA section before EOF");
            return false;
        }
        std::string mdPrefix = f.ReadStr();

        // sometimes there can be other garbage in this gap, so we need to 
        // skip over that as well
        if(mdPrefix.length() == 8){
            mdType = mdPrefix.substr(0, 4);
            mdVersion = mdPrefix.substr(4, 4);

            if(mdType == "MDAT"){
                f.ReadUint32(); // skip 4 bytes
                uint32_t num_attachments = f.ReadUint16(); // number of attachments in the MDAT section

                for(int i = 0; i < num_attachments; i++){
                    f.ReadUint16(); // skip 2 bytes
                    std::string attachment_name = f.ReadStr(); // attachment name
                    int bytesToRead = mdat_attachment_data_byte_length;
                    for(int j = 0; j < bytesToRead; j++){
                        f.ReadUint8();
                    }

                }
            }
        }
    } while (mdType != "MDLA");
    

    if(mdType == "MDLA" && mdVersion.length() > 0){
        mdl.mdla = std::stoi(mdVersion);
        if (mdl.mdla != 0) {
            uint end_size = f.ReadUint32();
            (void)end_size;

            uint anim_num = f.ReadUint32();
            anims.resize(anim_num);
            for (auto& anim : anims) {
                // there can be a variable number of 32-bit 0s between animations
                anim.id = 0;
                while(anim.id == 0){
                    if (f.Tell() >= f.Size()) {
                        LOG_ERROR("unexpected EOF while reading animation id");
                        return false;
                    }
                    const auto before = f.Tell();
                    anim.id = f.ReadInt32();
                    const auto after = f.Tell();
                    if (after <= before) {
                        LOG_ERROR("stream did not advance while reading animation id");
                        return false;
                    }
                }
    
                if (anim.id <= 0) {
                    LOG_ERROR("wrong anime id %d", anim.id);
                    return false;
                }
                f.ReadInt32();
                anim.name   = f.ReadStr();
                if(anim.name.empty()){
                    anim.name = f.ReadStr();
                }
                anim.mode   = ToPlayMode(f.ReadStr());
                anim.fps    = f.ReadFloat();
                anim.length = f.ReadInt32();
                f.ReadInt32();

                uint32_t b_num = f.ReadUint32();
                anim.bframes_array.resize(b_num);
                for (auto& bframes : anim.bframes_array) {
                    f.ReadInt32();
                    uint32_t byte_size = f.ReadUint32();
                    uint32_t num       = byte_size / singile_bone_frame;
                    if (byte_size % singile_bone_frame != 0) {
                        LOG_ERROR("wrong bone frame size %d", byte_size);
                        return false;
                    }
                    bframes.frames.resize(num);
                    for (auto& frame : bframes.frames) {
                        for (auto& v : frame.position) v = f.ReadFloat();
                        for (auto& v : frame.angle) v = f.ReadFloat();
                        for (auto& v : frame.scale) v = f.ReadFloat();
                    }
                }
                
                // in the alternative MDL format there are 2 empty bytes followed
                // by a variable number of 32-bit 0s between animations. We'll read
                // the two bytes now so that the cursor is aligned to read through the
                // 32-bit 0s in the next iteration
                if(alt_mdl_format)
                {
                    f.ReadUint8();
                    f.ReadUint8();
                    if (mdl.mdla >= 3)
                        f.ReadUint8();
                }
                else if(mdl.mdla >= 3){
                    // Newer MDLA variants insert an extra 8-bit zero between animations.
                    // If we don't consume it here, subsequent animation ids are shifted by 8 bits.
                    f.ReadUint8();
                }
                else{
                    uint32_t unk_extra_uint = f.ReadUint32();
                    for (uint i = 0; i < unk_extra_uint; i++) {
                        f.ReadFloat();
                        // data is like: {"$$hashKey":"object:2110","frame":1,"name":"random_anim"}
                        std::string unk_extra = f.ReadStr();
                    }
                }
            }
        }
    }
    
    mdl.puppet->prepared();

    LOG_INFO("read puppet: mdlv: %d, nmdls: %d, mdla: %d, bones: %d, anims: %d",
             mdl.mdlv,
             mdl.mdls,
             mdl.mdla,
             mdl.puppet->bones.size(),
             mdl.puppet->anims.size());
    return true;
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

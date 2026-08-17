#pragma once

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace wallpaper
{

// Official #require LightingV1 expands to PerformLighting_V1 plus the per-light
// uniform arrays. Snippets match the host injector: point/spot/tube use
// ComputePBRLightShadow; directional uses ComputePBRLightShadowInfinite;
// shadowed point lights go through CalculateProjectedCoordsPoint +
// PerformPointShadowMapping; shadowed spots use CalculateProjectedCoords +
// PerformShadowMapping; shadowed directionals mix three cascade projections.
struct LightingV1Desc {
    int  point { 0 };
    int  spot { 0 };
    int  directional { 0 };
    int  tube { 0 };
    bool shadows { false };
    std::vector<char> point_shadow;
    std::vector<char> spot_shadow;
    std::vector<char> spot_cookie;
    std::vector<char> directional_shadow;
};

inline int ShadowPcfTaps(int quality) {
    if (quality <= 0) return 0;
    return quality == 1 ? 1 : 9;
}

inline std::string ExpandLightingV1(const LightingV1Desc& desc) {
    std::ostringstream out;
    auto emit_vec4 = [&](std::string_view name, int count) {
        if (count <= 0) return;
        out << "uniform vec4 " << name << "[" << count << "];\n";
    };
    auto emit_mat4 = [&](std::string_view name, int count) {
        if (count <= 0) return;
        out << "uniform mat4 " << name << "[" << count << "];\n";
    };

    emit_vec4("g_LPoint_Origin", desc.point);
    emit_vec4("g_LPoint_Color", desc.point);
    emit_vec4("g_LSpot_Origin", desc.spot);
    emit_vec4("g_LSpot_Color", desc.spot);
    emit_vec4("g_LSpot_Direction", desc.spot);
    emit_vec4("g_LSpot_Exponent", desc.spot);
    emit_vec4("g_LDirectional_Color", desc.directional);
    emit_vec4("g_LDirectional_Direction", desc.directional);
    emit_vec4("g_LTube_OriginA", desc.tube);
    emit_vec4("g_LTube_OriginB", desc.tube);
    emit_vec4("g_LTube_Color", desc.tube);

    int point_shadow_n = 0;
    for (char flag : desc.point_shadow) {
        if (flag) point_shadow_n++;
    }
    int spot_shadow_n = 0;
    for (char flag : desc.spot_shadow) {
        if (flag) spot_shadow_n++;
    }
    int cookie_only_n = 0;
    for (int i = 0; i < desc.spot; ++i) {
        const bool shadow = desc.shadows && i < static_cast<int>(desc.spot_shadow.size()) &&
                            desc.spot_shadow[static_cast<size_t>(i)];
        const bool cookie = i < static_cast<int>(desc.spot_cookie.size()) &&
                            desc.spot_cookie[static_cast<size_t>(i)];
        if (cookie && ! shadow) cookie_only_n++;
    }
    int dir_shadow_n = 0;
    for (char flag : desc.directional_shadow) {
        if (flag) dir_shadow_n++;
    }
    const int cascade_n = dir_shadow_n * 3;
    const int proj_n    = spot_shadow_n + cookie_only_n + cascade_n;

    if (desc.shadows && point_shadow_n > 0) {
        emit_vec4("g_LFeature_ShadowPointProjection", desc.point);
        emit_vec4("g_LFeature_ShadowPointProjectionTransform", desc.point);
    }
    if (desc.shadows && proj_n > 0) {
        emit_mat4("g_LFeature_ShadowProjection", proj_n);
        emit_vec4("g_LFeature_ShadowProjectionTransform", proj_n);
    }

    out << "vec3 PerformLighting_V1(vec3 worldPos, vec3 color, vec3 normal, vec3 viewVector, "
           "vec3 specularTint, vec3 ambient, float roughness, float metallic)\n{\n";
    out << "\tvec3 light = CAST3(0.0);\n";

    for (int i = 0; i < desc.point; ++i) {
        const bool shadow = desc.shadows && i < static_cast<int>(desc.point_shadow.size()) &&
                            desc.point_shadow[static_cast<size_t>(i)];
        out << "\t{\n";
        out << "\tconst uint i = " << i << ";\n";
        out << "\tvec3 lightDelta = g_LPoint_Origin[i].xyz - worldPos;\n";
        if (shadow) {
            out << "\tvec4 projectedCoords = CalculateProjectedCoordsPoint(worldPos, "
                   "g_LPoint_Origin[i].xyz, g_LFeature_ShadowPointProjection[i], "
                   "g_LFeature_ShadowPointProjectionTransform[i]);\n";
            out << "\tfloat shadowFactor = PerformPointShadowMapping(projectedCoords);\n";
            out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
                   "g_LPoint_Color[i].rgb, g_LPoint_Color[i].w, g_LPoint_Origin[i].w, "
                   "specularTint, ambient, roughness, metallic, shadowFactor);\n";
        } else {
            out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
                   "g_LPoint_Color[i].rgb, g_LPoint_Color[i].w, g_LPoint_Origin[i].w, "
                   "specularTint, ambient, roughness, metallic, 1.0);\n";
        }
        out << "\t}\n";
    }

    int spot_proj = 0;
    for (int i = 0; i < desc.spot; ++i) {
        const bool shadow = desc.shadows && i < static_cast<int>(desc.spot_shadow.size()) &&
                            desc.spot_shadow[static_cast<size_t>(i)];
        const bool cookie = i < static_cast<int>(desc.spot_cookie.size()) &&
                            desc.spot_cookie[static_cast<size_t>(i)];
        out << "\t{\n";
        out << "\tconst uint i = " << i << ";\n";
        out << "\tvec3 lightDelta = g_LSpot_Origin[i].xyz - worldPos;\n";
        if (shadow) {
            out << "\tvec3 projectedCoords = CalculateProjectedCoords(worldPos, "
                   "g_LFeature_ShadowProjection[" << spot_proj << "]);\n";
            out << "\tfloat shadowFactor = PerformShadowMapping(projectedCoords, "
                   "g_LFeature_ShadowProjectionTransform[" << spot_proj << "]);\n";
            spot_proj++;
            if (cookie) {
                out << "\tvec3 colorCookie = texSample2D(COOKIE_SAMPLER, projectedCoords.xy).rgb;\n";
                out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
                       "g_LSpot_Color[i].rgb * colorCookie, g_LSpot_Color[i].w, "
                       "g_LSpot_Exponent[i].x, specularTint, ambient, roughness, metallic, "
                       "shadowFactor);\n";
            } else {
                out << "\tfloat spotCookie = -dot(normalize(lightDelta), g_LSpot_Direction[i].xyz);\n";
                out << "\tspotCookie = smoothstep(g_LSpot_Direction[i].w, g_LSpot_Origin[i].w, "
                       "spotCookie);\n";
                out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
                       "g_LSpot_Color[i].rgb * spotCookie, g_LSpot_Color[i].w, "
                       "g_LSpot_Exponent[i].x, specularTint, ambient, roughness, metallic, "
                       "shadowFactor);\n";
            }
        } else if (cookie) {
            out << "\tvec3 projectedCoords = CalculateProjectedCoords(worldPos, "
                   "g_LFeature_ShadowProjection[" << spot_proj << "]);\n";
            out << "\tvec3 colorCookie = texSample2D(COOKIE_SAMPLER, projectedCoords.xy).rgb;\n";
            out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
                   "g_LSpot_Color[i].rgb * colorCookie, g_LSpot_Color[i].w, "
                   "g_LSpot_Exponent[i].x, specularTint, ambient, roughness, metallic, 1.0);\n";
            spot_proj++;
        } else {
            out << "\tfloat spotCookie = -dot(normalize(lightDelta), g_LSpot_Direction[i].xyz);\n";
            out << "\tspotCookie = smoothstep(g_LSpot_Direction[i].w, g_LSpot_Origin[i].w, "
                   "spotCookie);\n";
            out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
                   "g_LSpot_Color[i].rgb * spotCookie, g_LSpot_Color[i].w, "
                   "g_LSpot_Exponent[i].x, specularTint, ambient, roughness, metallic, 1.0);\n";
        }
        out << "\t}\n";
    }

    int cascade_base = spot_proj;
    for (int i = 0; i < desc.directional; ++i) {
        const bool shadow = desc.shadows && i < static_cast<int>(desc.directional_shadow.size()) &&
                            desc.directional_shadow[static_cast<size_t>(i)];
        out << "\t{\n";
        out << "\tconst uint i = " << i << ";\n";
        if (shadow) {
            const int p1 = cascade_base;
            const int p2 = cascade_base + 1;
            const int p3 = cascade_base + 2;
            cascade_base += 3;
            out << "\tconst uint p1 = " << p1 << ";\n";
            out << "\tconst uint p2 = " << p2 << ";\n";
            out << "\tconst uint p3 = " << p3 << ";\n";
            out << "\tvec4 projectedCoords1 = CalculateProjectedCoordsCascades(worldPos, "
                   "g_LFeature_ShadowProjection[p1]);\n";
            out << "\tvec4 projectedCoords2 = CalculateProjectedCoordsCascades(worldPos, "
                   "g_LFeature_ShadowProjection[p2]);\n";
            out << "\tvec4 projectedCoords3 = CalculateProjectedCoordsCascades(worldPos, "
                   "g_LFeature_ShadowProjection[p3]);\n";
            out << "\tvec4 uvTransforms = mix(g_LFeature_ShadowProjectionTransform[p1], "
                   "g_LFeature_ShadowProjectionTransform[p2], projectedCoords1.w);\n";
            out << "\tuvTransforms = mix(uvTransforms, g_LFeature_ShadowProjectionTransform[p3], "
                   "projectedCoords2.w);\n";
            out << "\tprojectedCoords1.xyz = mix(projectedCoords1.xyz, projectedCoords2.xyz, "
                   "projectedCoords1.w);\n";
            out << "\tprojectedCoords1.xyz = mix(projectedCoords1.xyz, projectedCoords3.xyz, "
                   "projectedCoords2.w);\n";
            out << "\tfloat shadowFactor = max(projectedCoords3.w, "
                   "PerformShadowMapping(projectedCoords1.xyz, uvTransforms));\n";
            out << "\tlight += ComputePBRLightShadowInfinite(normal, "
                   "g_LDirectional_Direction[i].xyz, viewVector, color, "
                   "g_LDirectional_Color[i].rgb, specularTint, ambient, roughness, metallic, "
                   "shadowFactor);\n";
        } else {
            out << "\tlight += ComputePBRLightShadowInfinite(normal, "
                   "g_LDirectional_Direction[i].xyz, viewVector, color, "
                   "g_LDirectional_Color[i].rgb, specularTint, ambient, roughness, metallic, "
                   "1.0);\n";
        }
        out << "\t}\n";
    }

    for (int i = 0; i < desc.tube; ++i) {
        out << "\t{\n";
        out << "\tconst uint i = " << i << ";\n";
        out << "\tvec3 lightDelta = PointSegmentDelta(worldPos, g_LTube_OriginA[i].xyz, "
               "g_LTube_OriginB[i].xyz);\n";
        out << "\tlight += ComputePBRLightShadow(normal, lightDelta, viewVector, color, "
               "g_LTube_Color[i].rgb, g_LTube_Color[i].w, g_LTube_OriginA[i].w, "
               "specularTint, ambient, roughness, metallic, 1.0);\n";
        out << "\t}\n";
    }

    out << "\treturn light;\n}\n";
    return out.str();
}

inline bool ExpandRequireLightingV1(std::string& src, const LightingV1Desc& desc) {
    const auto pos = src.find("#require LightingV1");
    if (pos == std::string::npos) return false;
    auto line_end = src.find('\n', pos);
    if (line_end == std::string::npos) line_end = src.size();
    src.replace(pos, line_end - pos, ExpandLightingV1(desc));
    return true;
}

} // namespace wallpaper

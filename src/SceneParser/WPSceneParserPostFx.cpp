#include "WPSceneParser.hpp"
#include "WPSceneParserShared.hpp"

// Post-process configuration for parsed scenes: scene bloom (LDR and HDR chains), shadow
// atlases, and volumetric lights. Split from WPSceneParser.cpp as a cohesive unit; the only
// parser internals it consumes are ParseContext and LoadMaterial, both declared in the shared
// header, and the core parser calls back into the Configure*/SceneHasShadowLights entry
// points declared there.

#include "Utils/Logging.h"
#include "Utils/Algorism.h"
#include "Core/StringHelper.hpp"
#include "SpecTexs.hpp"
#include "VulkanRender/Msaa.hpp"
#include "Scene/ShadowAtlas.hpp"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneTexture.h"
#include "WPJson.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <Eigen/Dense>

using namespace wallpaper;
using namespace Eigen;

void ClearSceneBloomGraph(Scene& scene) {
    for (const auto& output : scene.bloom.outputs) {
        if (output.rfind("__hanabi_scene_bloom", 0) == 0 || output.rfind("_rt_hdr_bloom", 0) == 0 ||
            output == "_rt_Bloom") {
            scene.renderTargets.erase(output);
            scene.pendingRenderTargetReleaseKeys.insert(output);
        }
    }
    scene.bloom.nodes.clear();
    scene.bloom.outputs.clear();
    scene.bloom.node.reset();
    scene.bloom.built_quality = 0;
}

bool ConfigureSceneHdrBloomPass(ParseContext& context) {
    auto& scene = *context.scene;
    const i32 scene_width  = std::max(1, context.ortho_w);
    const i32 scene_height = std::max(1, context.ortho_h);
    const int levels =
        std::max(2, scene.bloom.hdrIterations > 0 ? scene.bloom.hdrIterations : 2);
    const bool display_hdr = scene.bloom.quality >= 3;
    ClearSceneBloomGraph(scene);

    constexpr std::string_view fullscreen_vertex_source = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;
varying vec2 v_TexCoord;
void main() {
    gl_Position = vec4(a_Position, 1.0);
    v_TexCoord = a_TexCoord;
}
)";
    constexpr std::string_view hdr_downsample_fragment_source = R"(
varying vec2 v_TexCoord;
uniform sampler2D g_Texture0;
uniform vec4 g_RenderVar0;
#if BICUBIC
vec4 cubic(float v)
{
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}
vec4 textureBicubic(vec2 texCoords)
{
    float sc = 0.5;
    vec2 texSize = sc / g_RenderVar0.xy;
    vec2 invTexSize = g_RenderVar0.xy / sc;
    texCoords = texCoords * texSize - 0.5;
    vec2 fxy = frac(texCoords);
    texCoords -= fxy;
    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);
    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;
    offset *= invTexSize.xxyy;
    vec4 sample0 = texSample2D(g_Texture0, offset.xz);
    vec4 sample1 = texSample2D(g_Texture0, offset.yz);
    vec4 sample2 = texSample2D(g_Texture0, offset.xw);
    vec4 sample3 = texSample2D(g_Texture0, offset.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}
#endif
#if BLOOM
uniform float g_BloomStrength;
uniform vec4 g_BloomBlendParams;
uniform vec3 g_BloomTint;
#endif
#if UPSAMPLE
uniform float g_BloomScatter;
#endif
void main() {
#if BICUBIC
    vec3 albedo = textureBicubic(v_TexCoord + g_RenderVar0.xy).rgb +
                    textureBicubic(v_TexCoord + g_RenderVar0.zy).rgb +
                    textureBicubic(v_TexCoord + g_RenderVar0.xw).rgb +
                    textureBicubic(v_TexCoord + g_RenderVar0.zw).rgb;
#else
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.xy).rgb +
                    texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.zy).rgb +
                    texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.xw).rgb +
                    texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.zw).rgb;
#endif
#if UPSAMPLE
    albedo *= 0.25 * g_BloomScatter;
#else
    albedo *= 0.25;
#endif
#if BLOOM
    albedo = max(CAST3(0), albedo);
    float brightness = max(albedo.r, max(albedo.g, albedo.b));
    float soft = brightness - g_BloomBlendParams.y;
    soft = clamp(soft, 0.0, g_BloomBlendParams.z);
    soft = soft * soft * g_BloomBlendParams.w;
    float contribution = max(soft, brightness - g_BloomBlendParams.x);
    contribution /= max(brightness, 0.00001);
    albedo *= contribution * g_BloomStrength * g_BloomTint;
#endif
    gl_FragColor = vec4(albedo, 1.0);
}
)";
    constexpr std::string_view combine_hdr_fragment_source = R"(
varying vec2 v_TexCoord;
uniform sampler2D g_Texture0;
uniform sampler2D g_Texture1;
uniform vec2 g_TexelSize;
uniform vec4 g_RenderVar0;
vec3 lin(vec3 v)
{
    vec3 c = step(0.04045, v);
    return c * (pow((v + 0.055) / 1.055, CAST3(2.4))) + (1.0 - c) * (v / 12.92);
}
void main()
{
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord).rgb;
    vec3 bloom1 = texSample2D(g_Texture1, v_TexCoord + g_TexelSize).rgb +
                texSample2D(g_Texture1, v_TexCoord - g_TexelSize).rgb +
                texSample2D(g_Texture1, v_TexCoord + vec2(g_TexelSize.x, -g_TexelSize.y)).rgb +
                texSample2D(g_Texture1, v_TexCoord + vec2(-g_TexelSize.x, g_TexelSize.y)).rgb;
    bloom1 *= 0.25;
#if DISPLAYHDR == 1
    albedo = saturate(albedo);
    albedo += bloom1;
    float hdrFactors = g_RenderVar0.y * smoothstep(1.0, 5.0, dot(vec3(0.299, 0.587, 0.114), albedo)) + g_RenderVar0.x;
    gl_FragColor = vec4(lin((max(CAST3(0.0), albedo))) * hdrFactors, 1.0);
#else
    albedo += bloom1;
    gl_FragColor = vec4(saturate(lin(albedo)) * g_RenderVar0.x, 1.0);
#endif
}
)";

    const auto compile_shader = [&](std::string      name,
                                    std::string_view vertex_source,
                                    std::string_view fragment_source,
                                    usize            texture_count,
                                    Combos           combos) -> std::shared_ptr<SceneShader> {
        auto shader  = std::make_shared<SceneShader>();
        shader->name = std::move(name);
        WPShaderInfo                 shader_info;
        shader_info.combos           = std::move(combos);
        std::array                   units { WPShaderUnit {
                                                 .stage           = ShaderType::VERTEX,
                                                 .src             = std::string(vertex_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".vert",
                                             },
                                             WPShaderUnit {
                                                 .stage           = ShaderType::FRAGMENT,
                                                 .src             = std::string(fragment_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".frag",
                                             } };
        std::vector<WPShaderTexInfo> texinfos(texture_count, WPShaderTexInfo { .enabled = true });
        for (auto& unit : units) {
            unit.src = WPShaderParser::PreShaderSrc(*context.vfs, unit.src, &shader_info, texinfos);
        }
        shader->default_uniforms = shader_info.svs;
        if (! WPShaderParser::CompileToSpv(
                scene.scene_id, units, shader->codes, *context.vfs, &shader_info, texinfos)) {
            LOG_ERROR("SceneBloomConfig: HDR compile failed pass='%s'", shader->name.c_str());
            return nullptr;
        }
        return shader;
    };

    const auto make_node = [&](std::string name,
                               std::shared_ptr<SceneShader>
                                   shader,
                               std::vector<std::string>
                                            textures,
                               ShaderValues const_values,
                               BlendMode    blend) -> std::shared_ptr<SceneNode> {
        SceneMaterial material;
        material.name     = name;
        material.textures = std::move(textures);
        material.defines.reserve(material.textures.size());
        for (usize i = 0; i < material.textures.size(); ++i) {
            material.defines.push_back("g_Texture" + std::to_string(i));
        }
        material.customShader.shader      = std::move(shader);
        material.customShader.constValues = std::move(const_values);
        material.blenmode                 = blend;
        auto mesh = std::make_shared<SceneMesh>();
        mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        mesh->AddMaterial(std::move(material));
        auto node = std::make_shared<SceneNode>();
        node->SetName(name);
        node->AddMesh(mesh);
        context.shader_updater->SetNodeData(node.get(), WPShaderValueData {});
        return node;
    };

    auto extract_shader = compile_shader("__hanabi_scene_hdr_extract",
                                         fullscreen_vertex_source,
                                         hdr_downsample_fragment_source,
                                         1,
                                         {{"BLOOM", "1"}});
    auto down_shader    = compile_shader("__hanabi_scene_hdr_down",
                                         fullscreen_vertex_source,
                                         hdr_downsample_fragment_source,
                                         1,
                                         {});
    auto up_shader      = compile_shader("__hanabi_scene_hdr_up",
                                         fullscreen_vertex_source,
                                         hdr_downsample_fragment_source,
                                         1,
                                         {{"UPSAMPLE", "1"}});
    auto up_cubic_shader = compile_shader("__hanabi_scene_hdr_up_cubic",
                                          fullscreen_vertex_source,
                                          hdr_downsample_fragment_source,
                                          1,
                                          {{"UPSAMPLE", "1"}, {"BICUBIC", "1"}});
    auto combine_shader = compile_shader("__hanabi_scene_hdr_combine",
                                         fullscreen_vertex_source,
                                         combine_hdr_fragment_source,
                                         2,
                                         {{"DISPLAYHDR", display_hdr ? "1" : "0"}});
    if (extract_shader == nullptr || down_shader == nullptr || up_shader == nullptr ||
        up_cubic_shader == nullptr || combine_shader == nullptr) {
        ClearSceneBloomGraph(scene);
        return false;
    }

    const float threshold = scene.bloom.hdrThreshold;
    const float feather   = scene.bloom.hdrFeather;
    const float scatter   = scene.bloom.hdrScatter;
    const float knee      = threshold * feather;
    const std::array<float, 4> blend_params {
        threshold,
        threshold - knee,
        knee + knee,
        0.25f / (knee + 1.0e-5f),
    };
    // Dual-filter upsample is additive. Extract gain is
    // strength / (scatter^(levels-2) + 1) so scatter > 1 does not stack raw energy.
    const float extract_strength =
        scene.bloom.hdrStrength / (std::pow(scatter, static_cast<float>(levels - 2)) + 1.0f);

    std::vector<std::string> mip_targets;
    mip_targets.reserve(static_cast<usize>(levels));
    for (int i = 0; i < levels; ++i) {
        const int   denom  = 2 << i;
        const i32   width  = std::max(1, scene_width / denom);
        const i32   height = std::max(1, scene_height / denom);
        const float scale  = 1.0f / static_cast<float>(denom);
        std::string target = (i == 0) ? std::string("_rt_Bloom")
                                      : ("_rt_hdr_bloom_" + std::to_string(i));
        scene.renderTargets[target] = {
            .width     = width,
            .height    = height,
            .mapWidth  = width,
            .mapHeight = height,
            .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = scale },
        };
        mip_targets.push_back(std::move(target));
    }

    const auto texel4 = [](i32 w, i32 h) -> std::array<float, 4> {
        const float tx = 1.0f / static_cast<float>(std::max(1, w));
        const float ty = 1.0f / static_cast<float>(std::max(1, h));
        return { tx, ty, -tx, -ty };
    };

    scene.bloom.nodes.clear();
    scene.bloom.outputs.clear();

    ShaderValues extract_values;
    extract_values["g_RenderVar0"]       = ShaderValue(texel4(scene_width, scene_height));
    extract_values["g_BloomStrength"]    = ShaderValue(extract_strength);
    extract_values["g_BloomBlendParams"] = ShaderValue(blend_params);
    extract_values["g_BloomTint"]        = ShaderValue(scene.bloom.tint);
    auto extract_node                    = make_node("__hanabi_scene_hdr_extract",
                                  extract_shader,
                                  { SpecTex_Default.data() },
                                  std::move(extract_values),
                                  BlendMode::Disable);
    scene.bloom.nodes.push_back(extract_node);
    scene.bloom.outputs.push_back(mip_targets[0]);
    scene.bloom.node = extract_node;

    for (int i = 1; i < levels; ++i) {
        const i32 src_w = std::max(1, scene_width / (2 << (i - 1)));
        const i32 src_h = std::max(1, scene_height / (2 << (i - 1)));
        ShaderValues down_values;
        down_values["g_RenderVar0"] = ShaderValue(texel4(src_w, src_h));
        auto down_node              = make_node("__hanabi_scene_hdr_down_" + std::to_string(i),
                                   down_shader,
                                   { mip_targets[static_cast<usize>(i - 1)] },
                                   std::move(down_values),
                                   BlendMode::Disable);
        scene.bloom.nodes.push_back(down_node);
        scene.bloom.outputs.push_back(mip_targets[static_cast<usize>(i)]);
    }

    for (int i = levels - 2; i >= 0; --i) {
        const i32 src_w = std::max(1, scene_width / (2 << (i + 1)));
        const i32 src_h = std::max(1, scene_height / (2 << (i + 1)));
        ShaderValues up_values;
        up_values["g_RenderVar0"]    = ShaderValue(texel4(src_w, src_h));
        up_values["g_BloomScatter"]  = ShaderValue(scatter);
        const bool cubic             = (i == 0);
        auto up_node                 = make_node(cubic ? "__hanabi_scene_hdr_up_cubic"
                                                      : ("__hanabi_scene_hdr_up_" + std::to_string(i)),
                                cubic ? up_cubic_shader : up_shader,
                                { mip_targets[static_cast<usize>(i + 1)] },
                                std::move(up_values),
                                BlendMode::Additive);
        scene.bloom.nodes.push_back(up_node);
        scene.bloom.outputs.push_back(mip_targets[static_cast<usize>(i)]);
    }

    const i32 bloom_w = std::max(1, scene_width / 2);
    const i32 bloom_h = std::max(1, scene_height / 2);
    ShaderValues combine_values;
    combine_values["g_TexelSize"]   = ShaderValue(std::array<float, 2> {
        1.0f / static_cast<float>(bloom_w),
        1.0f / static_cast<float>(bloom_h),
    });
    combine_values["g_RenderVar0"]  = ShaderValue(std::array<float, 4> { 1.0f, 0.0f, 0.0f, 0.0f });
    auto combine_node               = make_node("__hanabi_scene_hdr_combine",
                                  combine_shader,
                                  { SpecTex_Default.data(), mip_targets[0] },
                                  std::move(combine_values),
                                  BlendMode::Disable);
    scene.bloom.nodes.push_back(combine_node);
    scene.bloom.outputs.push_back(SpecTex_Default.data());
    scene.bloom.built_quality = scene.bloom.quality;

    LOG_INFO("ScenePostProcess: quality=%d path=%s passes=%zu levels=%d "
             "mip0=%dx%d strength=%.3f extract=%.5f threshold=%.3f "
             "scatter=%.3f feather=%.3f",
             scene.bloom.quality,
             display_hdr ? "displayhdr" : "hdr",
             scene.bloom.nodes.size(),
             levels,
             bloom_w,
             bloom_h,
             scene.bloom.hdrStrength,
             extract_strength,
             scene.bloom.hdrThreshold,
             scatter,
             scene.bloom.hdrFeather);
    return true;
}

bool ConfigureSceneBloomPass(ParseContext& context) {
    if (context.scene == nullptr || context.vfs == nullptr) return false;

    auto& scene = *context.scene;
    scene.bloom.quality = std::clamp(scene.bloom.quality, 0, 3);
    if (scene.bloom.quality <= 0) {
        ClearSceneBloomGraph(scene);
        LOG_INFO("ScenePostProcess: quality=0 path=none passes=0");
        return false;
    }
    // Only the display-HDR host mode runs the float-format mip chain whose combine writes
    // linear values for an HDR surface. Ultra on a standard-range output keeps the LDR bloom
    // chain; feeding the linearizing combine into a normalized output darkens the whole frame.
    const bool use_hdr = scene.bloom.quality >= 3 && scene.bloom.hdr && scene.bloom.enabled;
    if (use_hdr) {
        return ConfigureSceneHdrBloomPass(context);
    }
    // Build the LDR Bloom node even when the authored user toggle currently disables it, as long
    // as the scene carries non-zero Bloom settings. Runtime toggles can then update `u_enabled`
    // in place. Ultra/displayhdr with authored HDR uses the separate HDR chain above.
    const bool has_ldr_bloom_work = scene.bloom.enabled || scene.bloom.strength > 0.0f;
    if (! has_ldr_bloom_work) {
        LOG_INFO("SceneBloomConfig: enabled=%s strength=%.3f threshold=%.3f "
                 "hdr-requested=%s render-hdr=false active=false",
                 scene.bloom.enabled ? "true" : "false",
                 scene.bloom.strength,
                 scene.bloom.threshold,
                 scene.bloom.hdr ? "true" : "false");
        return false;
    }

    ClearSceneBloomGraph(scene);
    const i32 scene_width    = std::max(1, context.ortho_w);
    const i32 scene_height   = std::max(1, context.ortho_h);
    const i32 quarter_width  = std::max(1, scene_width / 4);
    const i32 quarter_height = std::max(1, scene_height / 4);
    const i32 eighth_width   = std::max(1, scene_width / 8);
    const i32 eighth_height  = std::max(1, scene_height / 8);

    constexpr std::string_view quarter_target = "__hanabi_scene_bloom_quarter";
    constexpr std::string_view eighth_target  = "__hanabi_scene_bloom_eighth";
    constexpr std::string_view blur_target    = "__hanabi_scene_bloom_blur";

    // Wallpaper Engine's scene Bloom is implemented by the stock utility assets as a four-pass
    // render-target chain: quarter-size extraction, eighth-size horizontal blur, eighth-size
    // vertical blur, then additive combine into `_rt_default`. Rebuilding that topology here keeps
    // the high-channel clipping and pink highlight rolloff aligned with the Windows renderer,
    // instead of relying on a hand-tuned single-pass approximation.
    scene.renderTargets[std::string(quarter_target)] = {
        .width     = quarter_width,
        .height    = quarter_height,
        .mapWidth  = quarter_width,
        .mapHeight = quarter_height,
        .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = 0.25 },
    };
    scene.renderTargets[std::string(eighth_target)] = {
        .width     = eighth_width,
        .height    = eighth_height,
        .mapWidth  = eighth_width,
        .mapHeight = eighth_height,
        .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = 0.125 },
    };
    scene.renderTargets[std::string(blur_target)] = {
        .width     = eighth_width,
        .height    = eighth_height,
        .mapWidth  = eighth_width,
        .mapHeight = eighth_height,
        .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = 0.125 },
    };

    constexpr std::string_view fullscreen_vertex_source           = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

varying vec2 v_TexCoord;

void main() {
    gl_Position = vec4(a_Position, 1.0);
    v_TexCoord = a_TexCoord;
}
)";
    constexpr std::string_view downsample_quarter_vertex_source   = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

uniform vec2 g_TexelSize;

varying vec2 v_TexCoord[4];

void main() {
    gl_Position = vec4(a_Position, 1.0);
    v_TexCoord[0] = a_TexCoord - g_TexelSize;
    v_TexCoord[1] = a_TexCoord + g_TexelSize;
    v_TexCoord[2] = a_TexCoord + vec2(-g_TexelSize.x, g_TexelSize.y);
    v_TexCoord[3] = a_TexCoord + vec2(g_TexelSize.x, -g_TexelSize.y);
}
)";
    constexpr std::string_view downsample_quarter_fragment_source = R"(
varying vec2 v_TexCoord[4];

uniform sampler2D g_Texture0;

uniform float u_enabled; // {"material":"Bloom enabled","default":0,"range":[0,1]}
uniform float g_BloomStrength; // {"material":"bloomstrength","default":2,"range":[0,4]}
uniform float g_BloomThreshold; // {"material":"bloomthreshold","default":0.65,"range":[0,0.999]}
uniform vec3 g_BloomTint; // {"material":"bloomtint","default":"1 1 1"}

void main() {
    // Keep the generated render graph stable for runtime user-property toggles. A disabled scene
    // Bloom still writes black into the private Bloom targets so the later additive combine becomes
    // visually neutral without needing to destroy and rebuild graph nodes.
    if (u_enabled <= 0.0 || g_BloomStrength <= 0.0) {
        gl_FragColor = vec4(CAST3(0), 1.0);
        return;
    }

    vec3 albedo = texSample2D(g_Texture0, v_TexCoord[0]).rgb +
                    texSample2D(g_Texture0, v_TexCoord[1]).rgb +
                    texSample2D(g_Texture0, v_TexCoord[2]).rgb +
                    texSample2D(g_Texture0, v_TexCoord[3]).rgb;
    albedo *= 0.25;

    float scale = max(max(albedo.x, albedo.y), albedo.z);
    albedo *= saturate(scale - g_BloomThreshold);

    float grayscale = dot(vec3(0.2989, 0.5870, 0.1140), albedo);
    float sat = 1.0;
    albedo = -grayscale * sat + albedo * (1.0 + sat);

    gl_FragColor = vec4(max(CAST3(0), albedo * g_BloomStrength * g_BloomTint), 1.0);
}
)";
    constexpr std::string_view blur_x_vertex_source               = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

uniform vec2 g_TexelSize;

varying vec2 v_TexCoord[13];

void main() {
    gl_Position = vec4(a_Position, 1);

    float localTexel = g_TexelSize.x * 8.0;
    v_TexCoord[0] = vec2(a_TexCoord.x - localTexel * 6.0, a_TexCoord.y);
    v_TexCoord[1] = vec2(a_TexCoord.x - localTexel * 5.0, a_TexCoord.y);
    v_TexCoord[2] = vec2(a_TexCoord.x - localTexel * 4.0, a_TexCoord.y);
    v_TexCoord[3] = vec2(a_TexCoord.x - localTexel * 3.0, a_TexCoord.y);
    v_TexCoord[4] = vec2(a_TexCoord.x - localTexel * 2.0, a_TexCoord.y);
    v_TexCoord[5] = vec2(a_TexCoord.x - localTexel * 1.0, a_TexCoord.y);
    v_TexCoord[6] = a_TexCoord;
    v_TexCoord[7] = vec2(a_TexCoord.x + localTexel * 1.0, a_TexCoord.y);
    v_TexCoord[8] = vec2(a_TexCoord.x + localTexel * 2.0, a_TexCoord.y);
    v_TexCoord[9] = vec2(a_TexCoord.x + localTexel * 3.0, a_TexCoord.y);
    v_TexCoord[10] = vec2(a_TexCoord.x + localTexel * 4.0, a_TexCoord.y);
    v_TexCoord[11] = vec2(a_TexCoord.x + localTexel * 5.0, a_TexCoord.y);
    v_TexCoord[12] = vec2(a_TexCoord.x + localTexel * 6.0, a_TexCoord.y);
}
)";
    constexpr std::string_view blur_y_vertex_source               = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

uniform vec2 g_TexelSize;

varying vec2 v_TexCoord[13];

void main() {
    gl_Position = vec4(a_Position, 1);

    float localTexel = g_TexelSize.y * 8.0;
    v_TexCoord[0] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 6.0);
    v_TexCoord[1] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 5.0);
    v_TexCoord[2] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 4.0);
    v_TexCoord[3] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 3.0);
    v_TexCoord[4] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 2.0);
    v_TexCoord[5] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 1.0);
    v_TexCoord[6] = a_TexCoord;
    v_TexCoord[7] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 1.0);
    v_TexCoord[8] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 2.0);
    v_TexCoord[9] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 3.0);
    v_TexCoord[10] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 4.0);
    v_TexCoord[11] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 5.0);
    v_TexCoord[12] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 6.0);
}
)";
    constexpr std::string_view blur_fragment_source               = R"(
varying vec2 v_TexCoord[13];

uniform sampler2D g_Texture0;

void main() {
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord[0]).rgb * 0.006299 +
                    texSample2D(g_Texture0, v_TexCoord[1]).rgb * 0.017298 +
                    texSample2D(g_Texture0, v_TexCoord[2]).rgb * 0.039533 +
                    texSample2D(g_Texture0, v_TexCoord[3]).rgb * 0.075189 +
                    texSample2D(g_Texture0, v_TexCoord[4]).rgb * 0.119007 +
                    texSample2D(g_Texture0, v_TexCoord[5]).rgb * 0.156756 +
                    texSample2D(g_Texture0, v_TexCoord[6]).rgb * 0.171834 +
                    texSample2D(g_Texture0, v_TexCoord[7]).rgb * 0.156756 +
                    texSample2D(g_Texture0, v_TexCoord[8]).rgb * 0.119007 +
                    texSample2D(g_Texture0, v_TexCoord[9]).rgb * 0.075189 +
                    texSample2D(g_Texture0, v_TexCoord[10]).rgb * 0.039533 +
                    texSample2D(g_Texture0, v_TexCoord[11]).rgb * 0.017298 +
                    texSample2D(g_Texture0, v_TexCoord[12]).rgb * 0.006299;

    gl_FragColor = vec4(albedo, 1.0);
}
)";
    constexpr std::string_view combine_fragment_source            = R"(
varying vec2 v_TexCoord;

uniform sampler2D g_Texture0;
uniform sampler2D g_Texture1;

void main() {
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord).rgb;
    vec3 bloom = texSample2D(g_Texture1, v_TexCoord).rgb;
    albedo += bloom;

    gl_FragColor = vec4(albedo, 1.0);
}
)";

    const auto compile_shader = [&](std::string      name,
                                    std::string_view vertex_source,
                                    std::string_view fragment_source,
                                    usize texture_count) -> std::shared_ptr<SceneShader> {
        auto shader  = std::make_shared<SceneShader>();
        shader->name = std::move(name);

        WPShaderInfo                 shader_info;
        std::array                   units { WPShaderUnit {
                                                 .stage           = ShaderType::VERTEX,
                                                 .src             = std::string(vertex_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".vert",
                                             },
                                             WPShaderUnit {
                                                 .stage           = ShaderType::FRAGMENT,
                                                 .src             = std::string(fragment_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".frag",
                                             } };
        std::vector<WPShaderTexInfo> texinfos(texture_count, WPShaderTexInfo { .enabled = true });
        for (auto& unit : units) {
            unit.src = WPShaderParser::PreShaderSrc(*context.vfs, unit.src, &shader_info, texinfos);
        }
        shader->default_uniforms = shader_info.svs;
        if (! WPShaderParser::CompileToSpv(
                scene.scene_id, units, shader->codes, *context.vfs, &shader_info, texinfos)) {
            LOG_ERROR("SceneBloomConfig: compile failed pass='%s'", shader->name.c_str());
            return nullptr;
        }
        return shader;
    };

    const std::array<float, 2> scene_texel_size {
        1.0f / static_cast<float>(scene_width),
        1.0f / static_cast<float>(scene_height),
    };

    auto downsample_shader = compile_shader("__hanabi_scene_bloom_downsample_quarter",
                                            downsample_quarter_vertex_source,
                                            downsample_quarter_fragment_source,
                                            1);
    auto blur_x_shader     = compile_shader("__hanabi_scene_bloom_downsample_eighth_blur",
                                            blur_x_vertex_source,
                                            blur_fragment_source,
                                            1);
    auto blur_y_shader =
        compile_shader("__hanabi_scene_bloom_blur", blur_y_vertex_source, blur_fragment_source, 1);
    auto combine_shader = compile_shader(
        "__hanabi_scene_bloom_combine", fullscreen_vertex_source, combine_fragment_source, 2);
    if (downsample_shader == nullptr || blur_x_shader == nullptr || blur_y_shader == nullptr ||
        combine_shader == nullptr) {
        return false;
    }

    const auto make_node = [&](std::string name,
                               std::shared_ptr<SceneShader>
                                   shader,
                               std::vector<std::string>
                                            textures,
                               ShaderValues const_values) -> std::shared_ptr<SceneNode> {
        SceneMaterial material;
        material.name     = name;
        material.textures = std::move(textures);
        material.defines.reserve(material.textures.size());
        for (usize i = 0; i < material.textures.size(); ++i) {
            material.defines.push_back("g_Texture" + std::to_string(i));
        }
        material.customShader.shader      = std::move(shader);
        material.customShader.constValues = std::move(const_values);
        material.blenmode                 = BlendMode::Disable;

        auto mesh = std::make_shared<SceneMesh>();
        mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        mesh->AddMaterial(std::move(material));

        auto node = std::make_shared<SceneNode>();
        node->SetName(name);
        node->AddMesh(mesh);
        context.shader_updater->SetNodeData(node.get(), WPShaderValueData {});
        return node;
    };

    float ldr_strength  = scene.bloom.strength;
    // The extraction threshold always follows the authored standard-range bloom key, even when
    // ultra derives the strength from the HDR keys. Feeding the HDR threshold into the LDR chain
    // makes every moderately bright surface bloom and washes out planet day sides.
    const float ldr_threshold = scene.bloom.threshold;
    if (scene.bloom.quality >= 2 && scene.bloom.hdr) {
        // Ultra quality derives the LDR chain strength from the authored HDR bloom keys. The
        // divisor is pow(scatter, iterations - 2) + 1 with the iteration count clamped to its
        // floor of 2 when no HDR mip chain exists, so the standard-range chain receives
        // hdrStrength / 2.
        ldr_strength = scene.bloom.hdrStrength /
                       (std::pow(std::max(1.0f, scene.bloom.hdrScatter), 0.0f) + 1.0f);
    }

    ShaderValues downsample_values;
    downsample_values["g_TexelSize"]      = ShaderValue(scene_texel_size);
    downsample_values["u_enabled"]        = ShaderValue(scene.bloom.enabled ? 1.0f : 0.0f);
    downsample_values["g_BloomStrength"]  = ShaderValue(ldr_strength);
    downsample_values["g_BloomThreshold"] = ShaderValue(ldr_threshold);
    downsample_values["g_BloomTint"]      = ShaderValue(scene.bloom.tint);

    ShaderValues blur_values;
    blur_values["g_TexelSize"] = ShaderValue(scene_texel_size);

    auto downsample_node = make_node("__hanabi_scene_bloom_downsample_quarter",
                                     downsample_shader,
                                     { SpecTex_Default.data() },
                                     std::move(downsample_values));
    auto blur_x_node     = make_node("__hanabi_scene_bloom_downsample_eighth_blur",
                                     blur_x_shader,
                                     { std::string(quarter_target) },
                                     blur_values);
    auto blur_y_node     = make_node(
        "__hanabi_scene_bloom_blur", blur_y_shader, { std::string(eighth_target) }, blur_values);
    auto combine_node = make_node("__hanabi_scene_bloom_combine",
                                  combine_shader,
                                  { SpecTex_Default.data(), std::string(blur_target) },
                                  {});

    scene.bloom.node    = downsample_node;
    scene.bloom.nodes   = { downsample_node, blur_x_node, blur_y_node, combine_node };
    scene.bloom.outputs = {
        std::string(quarter_target),
        std::string(eighth_target),
        std::string(blur_target),
        SpecTex_Default.data(),
    };

    scene.bloom.built_quality = scene.bloom.quality;
    LOG_INFO("ScenePostProcess: quality=%d path=ldr passes=%zu quarter=%dx%d eighth=%dx%d",
             scene.bloom.quality,
             scene.bloom.nodes.size(),
             quarter_width,
             quarter_height,
             eighth_width,
             eighth_height);
    LOG_INFO("SceneBloomConfig: enabled=%s strength=%.3f threshold=%.3f tint=[%.3f,%.3f,%.3f] "
             "hdr-requested=%s render-hdr=false hdr-strength=%.3f hdr-threshold=%.3f "
             "hdr-scatter=%.3f hdr-feather=%.3f hdr-iterations=%d active=true passes=%zu "
             "quarter=%dx%d eighth=%dx%d",
             scene.bloom.enabled ? "true" : "false",
             scene.bloom.strength,
             scene.bloom.threshold,
             scene.bloom.tint[0],
             scene.bloom.tint[1],
             scene.bloom.tint[2],
             scene.bloom.hdr ? "true" : "false",
             scene.bloom.hdrStrength,
             scene.bloom.hdrThreshold,
             scene.bloom.hdrScatter,
             scene.bloom.hdrFeather,
             scene.bloom.hdrIterations,
             scene.bloom.nodes.size(),
             quarter_width,
             quarter_height,
             eighth_width,
             eighth_height);
    return true;
}

bool SceneHasShadowLights(const Scene& scene) {
    for (char flag : scene.lighting.point_shadow) {
        if (flag) return true;
    }
    for (char flag : scene.lighting.spot_shadow) {
        if (flag) return true;
    }
    for (char flag : scene.lighting.directional_shadow) {
        if (flag) return true;
    }
    return false;
}

LightingV1Desc LightingDescFromScene(const Scene& scene) {
    LightingV1Desc desc;
    desc.point               = scene.lighting.point;
    desc.spot                = scene.lighting.spot;
    desc.directional         = scene.lighting.directional;
    desc.tube                = scene.lighting.tube;
    desc.shadows             = scene.shadows.quality != 0;
    desc.point_shadow        = scene.lighting.point_shadow;
    desc.spot_shadow         = scene.lighting.spot_shadow;
    desc.spot_cookie         = scene.lighting.spot_cookie;
    desc.directional_shadow  = scene.lighting.directional_shadow;
    return desc;
}

void ConfigureSceneShadows(Scene& scene) {
    for (auto& light : scene.lights) {
        if (light) light->clearShadowAtlasSlot();
    }
    const int quality = scene.shadows.quality;
    scene.shadows.built_quality = quality;
    scene.shadows.atlas_active  = false;
    if (quality <= 0) {
        scene.renderTargets[std::string(SpecTex_ShadowAtlas)] = {
            .width           = 2,
            .height          = 2,
            .mapWidth        = 2,
            .mapHeight       = 2,
            .allowReuse      = false,
            .comparisonDepth = true,
        };
        return;
    }

    const int face = ShadowFaceSize(quality);
    std::vector<SceneLight*> point_casters;
    std::vector<SceneLight*> spot_casters;
    std::vector<SceneLight*> dir_casters;
    for (auto& light : scene.lights) {
        if (! light || ! light->castsShadows()) continue;
        if (light->type() == SceneLightType::Point) point_casters.push_back(light.get());
        else if (light->type() == SceneLightType::Spot) spot_casters.push_back(light.get());
        else if (light->type() == SceneLightType::Directional) dir_casters.push_back(light.get());
    }
    const int point_n    = static_cast<int>(point_casters.size());
    const int spot_n     = static_cast<int>(spot_casters.size());
    const int dir_n      = static_cast<int>(dir_casters.size());
    const int cascade_n  = dir_n * 3;
    const int slots      = point_n + spot_n + cascade_n;
    const int n          = std::max(1, slots);
    const int w          = std::max(2, n * face);
    const int h          = std::max(2, face);
    int       cursor     = 0;
    auto pack = [&](SceneLight& light, bool point, bool directional, int cascade_index) {
        SceneLight::ShadowAtlasSlot slot;
        slot.packed         = true;
        slot.point          = point;
        slot.directional    = directional;
        slot.cascade_index  = cascade_index;
        slot.quality        = quality;
        slot.x              = cursor * face;
        slot.y              = 0;
        slot.size           = face;
        slot.atlas_w        = w;
        slot.atlas_h        = h;
        if (directional) light.setCascadeAtlasSlot(cascade_index, slot);
        else light.setShadowAtlasSlot(slot);
        cursor++;
    };
    for (SceneLight* light : point_casters) pack(*light, true, false, 0);
    for (SceneLight* light : spot_casters) pack(*light, false, false, 0);
    for (SceneLight* light : dir_casters) {
        for (int cascade = 0; cascade < 3; ++cascade) {
            pack(*light, false, true, cascade);
        }
    }
    scene.renderTargets[std::string(SpecTex_ShadowAtlas)] = {
        .width           = w,
        .height          = h,
        .mapWidth        = w,
        .mapHeight       = h,
        .allowReuse      = false,
        .comparisonDepth = true,
        .sample          = { TextureWrap::CLAMP_TO_EDGE,
                             TextureWrap::CLAMP_TO_EDGE,
                             TextureFilter::LINEAR,
                             TextureFilter::LINEAR },
    };
    scene.shadows.atlas_active = true;
}

void ClearSceneVolumetrics(Scene& scene) {
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsBack));
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsSingle));
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsLightBuffer));
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsLightBufferB));
    scene.volumetrics.lights.clear();
    scene.volumetrics.nodes.clear();
    scene.volumetrics.outputs.clear();
    scene.volumetrics.blur_h.reset();
    scene.volumetrics.blur_v.reset();
    scene.volumetrics.combine.reset();
    scene.volumetrics.active        = false;
    scene.volumetrics.built_quality = scene.volumetrics.quality == 0 ? 0 : -1;
}

void AddInwardUnitCube(SceneMesh& mesh) {
    // Cookie hull: eight float3 corners. Combined with AltVP = inverse(light clip)
    // this is the unit cube in [-1,1]. Inward winding + cullmode "normal" writes
    // the far faces that volumetricsfront treats as backDepth.
    constexpr float p[8 * 3] = {
        -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,
        -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,
    };
    const uint16_t idx[] = {
        0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    };
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 } }, 8);
    vertex.SetVertex(WE_IN_POSITION, std::span<const float>(p, 24));
    mesh.AddVertexArray(std::move(vertex));
    SceneIndexArray indices((sizeof(idx) / sizeof(idx[0]) + 1) / 2);
    indices.AssignHalf(0, std::span<const uint16_t>(idx, sizeof(idx) / sizeof(idx[0])));
    mesh.AddIndexArray(std::move(indices));
}

void AddInwardPointSphere(SceneMesh& mesh) {
    // Point hull: unit sphere. AltVP is T(origin)*S(radius).
    // Latitude step π/24, longitude step 2π/24.
    // Poles at (0, ±1, 0), then 23 rings × 25 verts (last duplicates first).
    // Inward winding: (north[j], north[j+1], south[j]) is CW from outside.
    constexpr int    rings       = 23;
    constexpr int    slices      = 24;
    constexpr int    ring_stride = slices + 1;
    constexpr float  lat_step    = 0.1308997f;
    constexpr float  lon_step    = 0.2617994f;
    std::vector<float>    pos;
    std::vector<uint16_t> idx;
    pos.reserve(static_cast<size_t>((2 + rings * ring_stride) * 3));
    pos.insert(pos.end(), { 0.0f, 1.0f, 0.0f });
    pos.insert(pos.end(), { 0.0f, -1.0f, 0.0f });
    for (int ring = 1; ring <= rings; ++ring) {
        const float lat = static_cast<float>(ring) * lat_step;
        const float sl  = std::sin(lat);
        const float cl  = std::cos(lat);
        for (int slice = 0; slice < ring_stride; ++slice) {
            const float lon = static_cast<float>(slice) * lon_step;
            pos.insert(pos.end(), { std::cos(lon) * sl, cl, std::sin(lon) * sl });
        }
    }
    const auto ring_start = [](int ring_index) -> uint16_t {
        return static_cast<uint16_t>(2 + ring_index * ring_stride);
    };
    for (int ring = 0; ring < rings - 1; ++ring) {
        const uint16_t curr = ring_start(ring);
        const uint16_t next = ring_start(ring + 1);
        for (int slice = 0; slice < slices; ++slice) {
            const uint16_t a = static_cast<uint16_t>(curr + slice);
            const uint16_t b = static_cast<uint16_t>(curr + slice + 1);
            const uint16_t c = static_cast<uint16_t>(next + slice);
            const uint16_t d = static_cast<uint16_t>(next + slice + 1);
            idx.insert(idx.end(), { a, b, c, b, d, c });
        }
    }
    const uint16_t first = ring_start(0);
    const uint16_t last  = ring_start(rings - 1);
    for (int slice = 0; slice < slices; ++slice) {
        idx.insert(idx.end(),
                   { 0, static_cast<uint16_t>(first + slice + 1),
                     static_cast<uint16_t>(first + slice) });
        idx.insert(idx.end(),
                   { 1, static_cast<uint16_t>(last + slice),
                     static_cast<uint16_t>(last + slice + 1) });
    }
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 } },
                            static_cast<uint32_t>(pos.size() / 3));
    vertex.SetVertex(WE_IN_POSITION, std::span<const float>(pos.data(), pos.size()));
    mesh.AddVertexArray(std::move(vertex));
    SceneIndexArray indices((idx.size() + 1) / 2);
    indices.AssignHalf(0, std::span<const uint16_t>(idx.data(), idx.size()));
    mesh.AddIndexArray(std::move(indices));
}

void AddInwardSpotCone(SceneMesh& mesh) {
    // Spot hull: 32 slices, rings at z=1 and z=0 when reverse-depth is off.
    // Apex at z=0, unit base at z=1.
    constexpr int slices = 32;
    std::vector<float>    pos;
    std::vector<uint16_t> idx;
    pos.reserve(static_cast<size_t>((slices + 2) * 3));
    pos.insert(pos.end(), { 0.0f, 0.0f, 0.0f });
    pos.insert(pos.end(), { 0.0f, 0.0f, 1.0f });
    for (int i = 0; i < slices; ++i) {
        const float a = (static_cast<float>(i) / static_cast<float>(slices)) * 6.283185307179586f;
        pos.insert(pos.end(), { std::cos(a), std::sin(a), 1.0f });
    }
    for (int i = 0; i < slices; ++i) {
        const uint16_t a = static_cast<uint16_t>(2 + i);
        const uint16_t b = static_cast<uint16_t>(2 + ((i + 1) % slices));
        idx.insert(idx.end(), { 0, b, a });
        idx.insert(idx.end(), { 1, a, b });
    }
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 } },
                            static_cast<uint32_t>(pos.size() / 3));
    vertex.SetVertex(WE_IN_POSITION, std::span<const float>(pos.data(), pos.size()));
    mesh.AddVertexArray(std::move(vertex));
    SceneIndexArray indices((idx.size() + 1) / 2);
    indices.AssignHalf(0, std::span<const uint16_t>(idx.data(), idx.size()));
    mesh.AddIndexArray(std::move(indices));
}

bool LoadVolumetricUtilMaterial(fs::VFS& vfs, Scene& scene, WPShaderValueUpdater* updater,
                                SceneNode& node, std::string_view json_path,
                                const std::unordered_map<std::string, int32_t>& extra_combos,
                                SceneLight* light, const std::string& cookie,
                                const std::optional<SceneModelRenderState>& render_state,
                                std::shared_ptr<SceneMesh> mesh) {
    nlohmann::json json;
    if (! PARSE_JSON(fs::GetFileContent(vfs, std::string(json_path)), json)) {
        LOG_ERROR("SceneVolumetrics: failed to parse '%.*s'",
                  static_cast<int>(json_path.size()),
                  json_path.data());
        return false;
    }
    wpscene::WPMaterial wpmat;
    if (! wpmat.FromJson(json)) {
        LOG_ERROR("SceneVolumetrics: invalid material '%.*s'",
                  static_cast<int>(json_path.size()),
                  json_path.data());
        return false;
    }
    for (const auto& [name, value] : extra_combos) {
        wpmat.combos[name] = value;
    }
    if (! cookie.empty()) {
        if (wpmat.textures.size() < 3) wpmat.textures.resize(3);
        wpmat.textures[2] = cookie;
    }

    SceneMaterial     material;
    WPShaderValueData sv_data;
    sv_data.volumetric_light = light;
    sv_data.volumetric_pass  = light != nullptr;
    WPShaderInfo shader_info;
    if (! LoadMaterial(vfs, wpmat, &scene, &node, &material, &sv_data, nullptr, &shader_info)) {
        LOG_ERROR("SceneVolumetrics: compile failed '%.*s'",
                  static_cast<int>(json_path.size()),
                  json_path.data());
        return false;
    }
    if (render_state.has_value()) material.modelRenderState = *render_state;
    mesh->AddMaterial(std::move(material));
    node.AddMesh(mesh);
    updater->SetNodeData(&node, sv_data);
    return true;
}

bool ConfigureSceneVolumetricsImpl(Scene& scene, fs::VFS& vfs) {
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());
    if (updater == nullptr) return false;

    ConfigureSceneShadows(scene);
    ClearSceneVolumetrics(scene);
    const int quality = scene.volumetrics.quality;
    if (quality <= 0) {
        scene.volumetrics.built_quality = 0;
        return true;
    }

    std::vector<SceneLight*> lights;
    for (auto& light : scene.lights) {
        if (light && light->castVolumetrics() && light->typeSupportsVolumetrics()) {
            lights.push_back(light.get());
        }
    }
    if (lights.empty()) {
        scene.volumetrics.built_quality = quality;
        return true;
    }

    // LightBuffer scale: 1/8 below high quality, 1/4 at high/ultra. Back stays full-res.
    const double scale = quality < 3 ? 0.125 : 0.25;
    const i32    full_w = std::max(1, scene.ortho[0]);
    const i32    full_h = std::max(1, scene.ortho[1]);
    const i32    buf_w  = std::max(1, static_cast<i32>(full_w * scale));
    const i32    buf_h  = std::max(1, static_cast<i32>(full_h * scale));

    scene.renderTargets[std::string(SpecTex_VolumetricsBack)] = {
        .width     = full_w,
        .height    = full_h,
        .mapWidth  = full_w,
        .mapHeight = full_h,
        .withDepth = true,
        .bind      = { .enable = true, .screen = true, .scale = 1.0 },
    };
    scene.renderTargets[std::string(SpecTex_VolumetricsSingle)] = {
        .width     = buf_w,
        .height    = buf_h,
        .mapWidth  = buf_w,
        .mapHeight = buf_h,
        .bind      = { .enable = true, .screen = true, .scale = scale },
    };
    scene.renderTargets[std::string(SpecTex_VolumetricsLightBuffer)] = {
        .width     = buf_w,
        .height    = buf_h,
        .mapWidth  = buf_w,
        .mapHeight = buf_h,
        .bind      = { .enable = true, .screen = true, .scale = scale },
    };
    if (quality < 3) {
        scene.renderTargets[std::string(SpecTex_VolumetricsLightBufferB)] = {
            .width     = buf_w,
            .height    = buf_h,
            .mapWidth  = buf_w,
            .mapHeight = buf_h,
            .bind      = { .enable = true, .screen = true, .scale = scale },
        };
    }

    SceneModelRenderState back_state;
    back_state.depthTest     = true;
    back_state.depthWrite    = true;
    back_state.cullMode      = SceneCullMode::Back;
    back_state.colorLoadMode = SceneModelColorLoadMode::Clear;
    // The hull is drawn before the camera-inside test, so both hemispheres rasterize
    // when the eye is inside the volume. Default LESS keeps the near wall and the
    // fullscreen ray (shader z=0 = D3D near) never enters the light. GREATER + clear 0
    // keeps max window Z, matching a D3D depth RT sampled as backDepth.
    back_state.depthGreater  = true;
    back_state.depthClear    = 0.0f;

    SceneModelRenderState front_hull_state;
    front_hull_state.depthTest     = false;
    front_hull_state.depthWrite    = false;
    front_hull_state.cullMode      = SceneCullMode::Back;
    front_hull_state.colorLoadMode = SceneModelColorLoadMode::Load;

    SceneModelRenderState fullscreen_state;
    fullscreen_state.depthTest     = false;
    fullscreen_state.depthWrite    = false;
    fullscreen_state.cullMode      = SceneCullMode::None;
    fullscreen_state.colorLoadMode = SceneModelColorLoadMode::Load;

    for (SceneLight* light : lights) {
        std::unordered_map<std::string, int32_t> combos;
        const bool shadow =
            light->castsShadows() && scene.shadows.quality != 0;
        combos["QUALITY"] = quality;
        combos["SHADOW"]  = shadow ? 1 : 0;
        if (light->hasCookie()) combos["COOKIE"] = 1;
        if (light->type() == SceneLightType::Point) {
            combos["POINTLIGHT"]                    = 1;
            combos["LIGHTS_SHADOW_MAPPING_QUALITY"] = scene.shadows.quality;
        }
        const std::string cookie = light->hasCookie() ? light->cookie() : std::string();

        auto back_node        = std::make_shared<SceneNode>();
        auto front_node       = std::make_shared<SceneNode>();
        auto fullscreen_node  = std::make_shared<SceneNode>();
        back_node->SetName("volumetrics_back");
        front_node->SetName("volumetrics_front");
        fullscreen_node->SetName("volumetrics_fullscreen");

        auto back_mesh  = std::make_shared<SceneMesh>();
        auto front_mesh = std::make_shared<SceneMesh>();
        auto fs_mesh    = std::make_shared<SceneMesh>();
        if (light->type() == SceneLightType::Point) {
            AddInwardPointSphere(*back_mesh);
            AddInwardPointSphere(*front_mesh);
        } else if (light->hasCookie()) {
            AddInwardUnitCube(*back_mesh);
            AddInwardUnitCube(*front_mesh);
        } else {
            AddInwardSpotCone(*back_mesh);
            AddInwardSpotCone(*front_mesh);
        }
        fs_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);

        if (! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *back_node,
                                         "/assets/materials/util/volumetrics_back.json",
                                         {},
                                         light,
                                         {},
                                         back_state,
                                         back_mesh) ||
            ! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *front_node,
                                         "/assets/materials/util/volumetrics_front.json",
                                         combos,
                                         light,
                                         cookie,
                                         front_hull_state,
                                         front_mesh) ||
            ! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *fullscreen_node,
                                         "/assets/materials/util/volumetrics_fullscreen.json",
                                         combos,
                                         light,
                                         cookie,
                                         fullscreen_state,
                                         fs_mesh)) {
            ClearSceneVolumetrics(scene);
            return false;
        }

        Scene::VolumetricLightPass pass;
        pass.light      = light;
        pass.back       = back_node;
        pass.front      = front_node;
        pass.fullscreen = fullscreen_node;
        scene.volumetrics.lights.push_back(std::move(pass));
        scene.volumetrics.nodes.push_back(back_node);
        scene.volumetrics.nodes.push_back(front_node);
        scene.volumetrics.nodes.push_back(fullscreen_node);
    }

    if (quality < 3) {
        auto blur_h = std::make_shared<SceneNode>();
        auto blur_v = std::make_shared<SceneNode>();
        blur_h->SetName("volumetrics_blur_h");
        blur_v->SetName("volumetrics_blur_v");
        auto blur_h_mesh = std::make_shared<SceneMesh>();
        auto blur_v_mesh = std::make_shared<SceneMesh>();
        blur_h_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        blur_v_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        if (! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *blur_h,
                                         "/assets/materials/util/volumetrics_blur_h.json",
                                         {},
                                         nullptr,
                                         {},
                                         std::nullopt,
                                         blur_h_mesh) ||
            ! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *blur_v,
                                         "/assets/materials/util/volumetrics_blur_v.json",
                                         {},
                                         nullptr,
                                         {},
                                         std::nullopt,
                                         blur_v_mesh)) {
            ClearSceneVolumetrics(scene);
            return false;
        }
        scene.volumetrics.blur_h = blur_h;
        scene.volumetrics.blur_v = blur_v;
        scene.volumetrics.nodes.push_back(blur_h);
        scene.volumetrics.nodes.push_back(blur_v);
    }

    auto combine = std::make_shared<SceneNode>();
    combine->SetName("volumetrics_combine");
    auto combine_mesh = std::make_shared<SceneMesh>();
    combine_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
    if (! LoadVolumetricUtilMaterial(vfs,
                                     scene,
                                     updater,
                                     *combine,
                                     "/assets/materials/util/volumetrics_combine.json",
                                     {},
                                     nullptr,
                                     {},
                                     std::nullopt,
                                     combine_mesh)) {
        ClearSceneVolumetrics(scene);
        return false;
    }
    scene.volumetrics.combine = combine;
    scene.volumetrics.nodes.push_back(combine);
    scene.volumetrics.active        = true;
    scene.volumetrics.built_quality = quality;
    return true;
}


bool wallpaper::ConfigureSceneVolumetrics(Scene& scene, fs::VFS& vfs) {
    return ConfigureSceneVolumetricsImpl(scene, vfs);
}

bool wallpaper::ConfigureSceneBloom(Scene& scene, fs::VFS& vfs) {
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());
    if (updater == nullptr) return false;
    ParseContext context;
    context.scene          = std::shared_ptr<Scene>(&scene, [](Scene*) {});
    context.shader_updater = updater;
    context.ortho_w        = std::max(1, scene.ortho[0]);
    context.ortho_h        = std::max(1, scene.ortho[1]);
    context.vfs            = &vfs;
    return ConfigureSceneBloomPass(context);
}

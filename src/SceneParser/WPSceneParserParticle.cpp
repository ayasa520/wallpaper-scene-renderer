#include "WPSceneParser.hpp"
#include "WPSceneParserShared.hpp"

// Particle layer materialization: emitter/initializer/operator/control-point loading, particle
// mesh capacity planning, extra trail/rope renderers, and ParseParticleObj. Split from
// WPSceneParser.cpp as a cohesive unit; the parser internals it consumes and the
// ParseParticleObj entry point the core parser dispatches into are declared in the shared
// header.

#include "Utils/Logging.h"
#include "Utils/Algorism.h"
#include "Core/ArrayHelper.hpp"
#include "SpecTexs.hpp"
#include "Scene/SceneImageEffectLayer.h"
#include "WPJson.hpp"
#include "WPParticleParser.hpp"
#include "Particle/ParticleRenderPlan.h"
#include "Particle/WPParticleRawGener.h"
#include "Particle/ParticleSystem.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <Eigen/Dense>

using namespace wallpaper;
using namespace Eigen;

void ConfigureParticleMesh(SceneMesh& mesh, uint32_t count, const ParticleRenderPlan& plan) {
    if (plan.IsRope()) {
        std::vector<SceneVertexArray::SceneVertexAttribute> attrs {
            { WE_IN_POSITIONVEC4.data(), VertexType::FLOAT4 },
            { WE_IN_TEXCOORDVEC4.data(), VertexType::FLOAT4 },
            { WE_IN_TEXCOORDVEC4C1.data(), VertexType::FLOAT4 },
        };
        if (plan.thick_format) {
            attrs.push_back({ WE_IN_TEXCOORDVEC4C2.data(), VertexType::FLOAT4 });
            attrs.push_back({ WE_IN_TEXCOORDVEC4C3.data(), VertexType::FLOAT4 });
        } else {
            attrs.push_back({ WE_IN_TEXCOORDVEC3C2.data(), VertexType::FLOAT3 });
        }
        attrs.push_back({ WE_IN_COLOR.data(), VertexType::FLOAT4 });

        // Rope renderers always submit one point per logical segment. Their stock geometry stage
        // owns spline expansion, so an index buffer or CPU-generated ribbon would describe a
        // different shader ABI than the one resolved in ParticleRenderPlan.
        mesh.AddVertexArray(SceneVertexArray(attrs, count));
        mesh.SetPrimitive(MeshPrimitive::POINT);
        return;
    }

    std::vector<SceneVertexArray::SceneVertexAttribute> attrs {
        { WE_IN_POSITION.data(), VertexType::FLOAT3 },
        { WE_IN_TEXCOORDVEC4.data(), VertexType::FLOAT4 },
        { WE_IN_COLOR.data(), VertexType::FLOAT4 },
    };
    if (plan.thick_format) {
        attrs.push_back({ WE_IN_TEXCOORDVEC4C1.data(), VertexType::FLOAT4 });
    }

    if (plan.expansion == ParticleExpansionMode::GeometryPoint) {
        mesh.AddVertexArray(SceneVertexArray(attrs, count));
        mesh.SetPrimitive(MeshPrimitive::POINT);
        return;
    }

    // Authored particle materials without a geometry stage consume four vertex invocations per
    // live particle. The extra attribute carries the authored rotation pair, while TexCoordVec4.xy
    // carries the individual quad corner expected by the original vertex shader.
    attrs.push_back({ WE_IN_TEXCOORDC2.data(), VertexType::FLOAT2 });
    mesh.AddVertexArray(
        SceneVertexArray(attrs, static_cast<size_t>(count) * static_cast<size_t>(4)));
    mesh.AddIndexArray(SceneIndexArray(count));
    mesh.SetPrimitive(MeshPrimitive::TRIANGLE);
}

constexpr u32 kMaxParticleCount = 20000;
constexpr i32 kMaxParticleTrailSegments = 256;

struct ParticleRendererSpec {
    std::string_view         name;
    ParticleRendererKind     kind;
    ParticleShaderSelection shader_selection;
    bool                     thick_format;

    constexpr bool IsRope() const noexcept {
        return kind == ParticleRendererKind::Rope || kind == ParticleRendererKind::RopeTrail;
    }
    constexpr bool IsTrail() const noexcept {
        return kind == ParticleRendererKind::SpriteTrail ||
            kind == ParticleRendererKind::RopeTrail;
    }
    constexpr bool UsesHistory() const noexcept {
        return kind == ParticleRendererKind::RopeTrail;
    }
};

struct ParticleOrientationBasis {
    std::array<float, 3> right { 1.0f, 0.0f, 0.0f };
    std::array<float, 3> up { 0.0f, 1.0f, 0.0f };
    std::array<float, 3> forward { 0.0f, 0.0f, 1.0f };
};

// The stock particle shaders expand a sprite in the authored orientation basis before applying
// the scene camera. A fixed orientation is a world-space plane: its authored axis is the sprite U
// direction and a perpendicular world direction is V. Keeping this basis in the parser preserves
// the shader contract and makes beam textures point along their 3D travel plane instead of using
// the old screen-space identity basis.
ParticleOrientationBasis ResolveParticleOrientationBasis(const wpscene::ParticleRender& renderer) {
    ParticleOrientationBasis basis;
    if (renderer.orientation != "fixed") return basis;

    Eigen::Vector3f axis { renderer.axis[0], renderer.axis[1], renderer.axis[2] };
    if (axis.squaredNorm() < 1e-12f) axis = Eigen::Vector3f::UnitY();
    axis.normalize();

    Eigen::Vector3f up;
    if (std::abs(axis.x()) < 1e-6f && std::abs(axis.z()) < 1e-6f) {
        up = Eigen::Vector3f(0.0f, 0.0f, axis.y() >= 0.0f ? 1.0f : -1.0f);
    } else {
        up = axis.cross(Eigen::Vector3f::UnitY()).normalized();
    }
    const Eigen::Vector3f forward = axis.cross(up).normalized();
    basis.right = { axis.x(), axis.y(), axis.z() };
    basis.up = { up.x(), up.y(), up.z() };
    basis.forward = { forward.x(), forward.y(), forward.z() };
    return basis;
}

int ResolveParticleOrientationCombo(std::string_view orientation) {
    if (orientation == "upright") return 1;
    if (orientation == "fixed") return 2;
    return 0;
}

constexpr std::array kParticleRendererSpecs {
    ParticleRendererSpec { "sprite",
                           ParticleRendererKind::Sprite,
                           ParticleShaderSelection::Authored,
                           false },
    ParticleRendererSpec { "spritetrail",
                           ParticleRendererKind::SpriteTrail,
                           ParticleShaderSelection::Authored,
                           true },
    ParticleRendererSpec { "rope",
                           ParticleRendererKind::Rope,
                           ParticleShaderSelection::StockRope,
                           true },
    ParticleRendererSpec { "ropetrail",
                           ParticleRendererKind::RopeTrail,
                           ParticleShaderSelection::StockRope,
                           false },
};

std::optional<ParticleRendererSpec> DescribeParticleRender(std::string_view renderer_name) {
    // Renderer classification is an ABI decision: it controls shader stages, mesh layout, CPU
    // extraction, and history ownership. Exact names keep rope and ropetrail from drifting into
    // the same runtime path merely because they share a prefix.
    for (const auto& spec : kParticleRendererSpecs) {
        if (spec.name == renderer_name) return spec;
    }
    return std::nullopt;
}

std::optional<u32> CheckedParticleMeshCapacity(std::string_view object_name,
                                               std::string_view renderer_name,
                                               std::initializer_list<uint64_t> factors) {
    uint64_t capacity { 1 };
    for (const uint64_t factor : factors) {
        if (factor != 0 && capacity > std::numeric_limits<uint64_t>::max() / factor) {
            LOG_ERROR("particle mesh capacity overflow object='%.*s' renderer='%.*s'",
                      static_cast<int>(object_name.size()),
                      object_name.data(),
                      static_cast<int>(renderer_name.size()),
                      renderer_name.data());
            return std::nullopt;
        }
        capacity *= factor;
    }
    if (capacity > std::numeric_limits<u32>::max()) {
        LOG_ERROR("particle mesh capacity exceeds uint32 object='%.*s' renderer='%.*s' "
                  "capacity=%llu",
                  static_cast<int>(object_name.size()),
                  object_name.data(),
                  static_cast<int>(renderer_name.size()),
                  renderer_name.data(),
                  static_cast<unsigned long long>(capacity));
        return std::nullopt;
    }
    return static_cast<u32>(capacity);
}

ParticleAnimationMode ToAnimMode(const std::string& str) {
    if (str == "randomframe")
        return ParticleAnimationMode::RANDOMONE;
    else if (str == "sequence")
        return ParticleAnimationMode::SEQUENCE;
    else {
        return ParticleAnimationMode::SEQUENCE;
    }
}

void ApplyLayerControlPointOverrides(ParticleSubSystem&                       pSys,
                                     const wpscene::ParticleInstanceoverride& over) {
    std::span<ParticleControlpoint> pcs = pSys.Controlpoints();
    const usize override_count          = std::min(pcs.size(), over.controlpointOffsets.size());
    for (usize i = 0; i < override_count; i++) {
        if (! over.controlpointOffsets[i].has_value()) continue;

        // Instance overrides are authored at the scene layer level, after the particle asset has
        // supplied default control point flags. Only the offset is replaced here so link_mouse and
        // worldspace semantics continue to come from the particle asset definition.
        pcs[i].base_offset =
            Eigen::Vector3d { array_cast<double>(*over.controlpointOffsets[i]).data() };
        pcs[i].offset = pcs[i].base_offset;
    }
}

void LoadControlPoint(ParticleSubSystem& pSys, const wpscene::Particle& wp,
                      const wpscene::ParticleInstanceoverride& over) {
    std::span<ParticleControlpoint> pcs = pSys.Controlpoints();
    usize                           s   = std::min(pcs.size(), wp.controlpoints.size());
    for (usize i = 0; i < s; i++) {
        pcs[i].base_offset =
            Eigen::Vector3d { array_cast<double>(wp.controlpoints[i].offset).data() };
        pcs[i].offset = pcs[i].base_offset;
        pcs[i].link_mouse =
            wp.controlpoints[i].flags[wpscene::ParticleControlpoint::FlagEnum::link_mouse];
        pcs[i].worldspace =
            wp.controlpoints[i].flags[wpscene::ParticleControlpoint::FlagEnum::worldspace];
    }
    ApplyLayerControlPointOverrides(pSys, over);
}

// A scene layer's instanceoverride belongs to the particle asset instantiated directly by that
// layer. Nested particle assets are separate authored systems: their own colorrandom ranges must
// remain intact so a root-layer colorn value does not flatten every child gradient (for example,
// the Matrix rain's blue glyph stream) before the scene compositor applies its parent effects.
wpscene::ParticleInstanceoverride ResolveParticleSubsystemOverride(
    const wpscene::ParticleInstanceoverride& layer_override, bool is_child_subsystem) {
    if (! is_child_subsystem) return layer_override;

    auto child_override = layer_override;
    child_override.overColor  = false;
    child_override.overColorn = false;
    return child_override;
}

ParticleAudioResponseParams ReadParticleAudioResponse(const nlohmann::json& json) {
    /*
     * Keep audio-response parsing in one helper so emitter, initializer, and operator paths share
     * the same defaults, endpoint clamping, and reversed frequency-interval handling.
     */
    ParticleAudioResponseParams response;
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingmode", response.mode);
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingexponent", response.exponent);
    if (json.contains("audioprocessingbounds") && json.at("audioprocessingbounds").is_string()) {
        AssignAudioBoundsFromAuthoredString(json.at("audioprocessingbounds").get<std::string>(),
                                            response.bounds);
    } else {
        GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingbounds", response.bounds);
    }
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingfrequencystart", response.frequency_start);
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingfrequencyend", response.frequency_end);
    response.frequency_start = std::min<uint32_t>(response.frequency_start, 15);
    response.frequency_end   = std::min<uint32_t>(response.frequency_end, 15);
    if (response.frequency_end < response.frequency_start)
        std::swap(response.frequency_start, response.frequency_end);
    return response;
}

ParticleAudioResponseFactor BindParticleAudioResponse(ParticleSystem& system,
                                                      ParticleAudioResponseParams response) {
    if (response.mode == 0) return {};

    system.MarkNeedsAudioSpectrum();
    ParticleSystem* system_ptr = &system;
    return [system_ptr, response]() { return system_ptr->AudioResponseFactor(response); };
}

void LoadInitializer(ParticleSubSystem& pSys, const wpscene::Particle& wp,
                     const wpscene::ParticleInstanceoverride& over, ParticleSystem& system) {
    for (const auto& ini : wp.initializers) {
        // A layer color override depends on colorrandom's authored endpoints and their RGB midpoint.
        // Wallpaper Engine shifts those endpoints by the override/reference delta in HSV space and
        // performs the existing per-particle interpolation afterward, so the initializer must remain
        // in the stream instead of being replaced by the requested layer color.
        ParticleAudioResponseFactor audio_factor =
            BindParticleAudioResponse(system, ReadParticleAudioResponse(ini));
        pSys.AddInitializer(WPParticleParser::genParticleInitOp(ini, std::move(audio_factor)));
    }
    if (over.enabled) pSys.AddInitializer(WPParticleParser::genOverrideInitOp(over));
}
void LoadOperator(ParticleSystem& system, ParticleSubSystem& pSys, const wpscene::Particle& wp,
                  const wpscene::ParticleInstanceoverride& over) {
    for (const auto& op : wp.operators) {
        ParticleAudioResponseFactor audio_factor =
            BindParticleAudioResponse(system, ReadParticleAudioResponse(op));
        pSys.AddOperator(WPParticleParser::genParticleOperatorOp(op, over, std::move(audio_factor)));
    }
}
void LoadEmitter(ParticleSystem& system, ParticleSubSystem& pSys, const wpscene::Particle& wp,
                 float count) {
    for (const auto& em : wp.emitters) {
        auto newEm = em;
        newEm.rate *= count;
        // newEm.origin[2] -= perspectiveZ;

        /*
         * Wallpaper Engine's audio-responsive emitters ("Custom BEAT Particle" style assets) are
         * only active while audio is playing: the authored rate is scaled by the live loudness of
         * the selected channel through the authored bounds window. Without this factor such
         * emitters run at full authored rate forever; assets author beat-burst rates in the
         * thousands per second, which floods the simulation with tens of thousands of permanent
         * particles and dominates both CPU and GPU frame time.
         */
        ParticleAudioResponseFactor audio_rate_factor =
            BindParticleAudioResponse(system, em.audio_response);
        ParticleEmitterTiming timing;
        auto op = WPParticleParser::genParticleEmittOp(newEm, std::move(audio_rate_factor), timing);
        pSys.AddEmitter(std::move(op), std::move(timing));
    }
}

ParticleSubSystem::SpawnType ParseSpawnType(std::string_view str) {
    using ST = ParticleSubSystem::SpawnType;
    ST type { ST::STATIC };
    if (str == "eventfollow") {
        type = ST::EVENT_FOLLOW;
    } else if (str == "eventspawn") {
        type = ST::EVENT_SPAWN;
    } else if (str == "eventdeath") {
        type = ST::EVENT_DEATH;
    }
    return type;
};

BlendMode ParseBlendMode(std::string_view str) {
    BlendMode bm;
    if (str == "translucent") {
        bm = BlendMode::Translucent;
    } else if (str == "additive") {
        bm = BlendMode::Additive;
    } else if (str == "normal") {
        bm = BlendMode::Normal;
    } else if (str == "alphatocoverage") {
        bm = BlendMode::AlphaToCoverage;
    } else if (str == "disabled") {
        // seems disabled is normal
        bm = BlendMode::Normal;
    } else {
        bm = BlendMode::Normal;
        LOG_ERROR("unknown blending: %s", str.data());
    }
    return bm;
}



void AttachExtraParticleRenderer(ParseContext& context, wpscene::WPParticleObject& wppartobj,
                                 const wpscene::Particle& particle_obj,
                                 const wpscene::ParticleRender& particle_renderer,
                                 SceneNode& parent_node, ParticleSubSystem& subsystem,
                                 const WPShaderValueData& parent_sv,
                                 const ParticleRawGenSpecOp& spec_op, u32 maxcount,
                                 u32 mesh_instancecount, std::string_view particle_debug_name,
                                 bool register_layer_nodes) {
    const auto renderer_spec = DescribeParticleRender(particle_renderer.name);
    if (! renderer_spec.has_value()) {
        LOG_ERROR("particle object '%s' has unsupported extra renderer '%s'",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str());
        return;
    }
    const auto& renderer          = *renderer_spec;
    const bool  render_rope_trail = renderer.UsesHistory();
    const bool  rope_shader       = renderer.IsRope();
    const bool  has_trail         = renderer.IsTrail();

    auto extra_node = std::make_shared<SceneNode>(
        Vector3f::Zero(), Vector3f::Ones(), Vector3f::Zero(),
        wppartobj.name + " " + particle_renderer.name);
    extra_node->ID() = wppartobj.id;
    extra_node->SetCamera(parent_node.Camera());

    auto effective_material = particle_obj.material;
    if (renderer.shader_selection == ParticleShaderSelection::StockRope) {
        effective_material.shader = "genericropeparticle";
    }

    WPShaderValueData svData;
    svData.CopyParallaxContractFrom(parent_sv);

    const ParticleOrientationBasis orientation_basis =
        ResolveParticleOrientationBasis(particle_renderer);

    WPShaderInfo shaderInfo;
    shaderInfo.baseConstSvs                         = context.global_base_uniforms;
    shaderInfo.baseConstSvs["g_OrientationUp"]      = orientation_basis.up;
    shaderInfo.baseConstSvs["g_OrientationRight"]   = orientation_basis.right;
    shaderInfo.baseConstSvs["g_OrientationForward"] = orientation_basis.forward;
    shaderInfo.baseConstSvs["g_ViewUp"]             = std::array { 0.0f, 1.0f, 0.0f };
    shaderInfo.baseConstSvs["g_ViewRight"]          = std::array { 1.0f, 0.0f, 0.0f };
    shaderInfo.baseConstSvs["g_EyePosition"]        = std::array {
        static_cast<float>(context.ortho_w) / 2.0f,
        static_cast<float>(context.ortho_h) / 2.0f,
        1000.0f,
    };
    shaderInfo.combos["ORIENTATION"] =
        std::to_string(ResolveParticleOrientationCombo(particle_renderer.orientation));

    const uint16_t trail_length = render_rope_trail
        ? static_cast<uint16_t>(
              std::clamp(particle_renderer.segments, 1, kMaxParticleTrailSegments))
        : 0;
    auto mesh_capacity = CheckedParticleMeshCapacity(
        particle_debug_name,
        particle_renderer.name,
        { static_cast<uint64_t>(maxcount), static_cast<uint64_t>(mesh_instancecount) });
    if (! mesh_capacity.has_value()) return;
    if (render_rope_trail) {
        mesh_capacity = CheckedParticleMeshCapacity(
            particle_debug_name,
            particle_renderer.name,
            { static_cast<uint64_t>(maxcount),
              static_cast<uint64_t>(mesh_instancecount),
              static_cast<uint64_t>(trail_length) });
        if (! mesh_capacity.has_value()) return;
    }

    if (has_trail) {
        shaderInfo.baseConstSvs["g_RenderVar0"] = std::array {
            particle_renderer.length,
            particle_renderer.maxlength,
            render_rope_trail ? 0.0f : particle_renderer.minlength,
            static_cast<float>(maxcount - 1),
        };
        shaderInfo.combos["TRAILRENDERER"] = "1";
    }
    if (renderer.thick_format) shaderInfo.combos["THICKFORMAT"] = "1";
    if (rope_shader) {
        const uint32_t subdivision = static_cast<uint32_t>(
            std::lround(std::max(0.0f, particle_renderer.subdivision)));
        shaderInfo.combos["TRAILSUBDIVISION"] = std::to_string(subdivision);
    }
    if (! particle_obj.flags[wpscene::Particle::FlagEnum::spritenoframeblending] &&
        particle_obj.animationmode != "randomframe") {
        shaderInfo.combos["SPRITESHEETBLEND"] = "1";
    }

    SceneMaterial material;
    std::optional<MaterialLoadResult> material_result;
    try {
        material_result = LoadMaterial(*context.vfs,
                                       effective_material,
                                       context.scene.get(),
                                       extra_node.get(),
                                       &material,
                                       &svData,
                                       context.user_properties,
                                       &shaderInfo,
                                       rope_shader ? GeometryStagePolicy::Required
                                                   : GeometryStagePolicy::MatchMaterial);
    } catch (const std::exception& e) {
        LOG_ERROR("load particleobj '%s' extra renderer '%s' material exception: %s",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str(),
                  e.what());
    }
    if (! material_result.has_value()) {
        LOG_ERROR("load particleobj '%s' extra renderer '%s' material faild",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str());
        return;
    }
    LoadConstvalue(material, effective_material, shaderInfo);
    LoadUserShaderValue(material, effective_material, shaderInfo, context.user_properties);

    auto  spMesh = std::make_shared<SceneMesh>(true);
    const ParticleRenderPlan render_plan {
        .renderer         = renderer.kind,
        .shader_selection = renderer.shader_selection,
        .expansion        = material_result->geometry_stage_loaded
                    ? ParticleExpansionMode::GeometryPoint
                    : ParticleExpansionMode::IndexedQuad,
        .thick_format     = renderer.thick_format || material.hasSprite,
    };
    ConfigureParticleMesh(*spMesh, *mesh_capacity, render_plan);
    spMesh->AddMaterial(std::move(material));
    extra_node->AddMesh(spMesh);
    RegisterUserShaderValueBindings(
        context, effective_material, shaderInfo, extra_node.get(), wppartobj.id, wppartobj.name);

    subsystem.AddRenderOutput(spMesh, render_plan, spec_op);
    parent_node.AppendChild(extra_node);
    if (register_layer_nodes) {
        context.scene->AddLayerRuntimeNode(wppartobj.id, extra_node.get());
    }
    context.shader_updater->SetNodeData(extra_node.get(), svData);
}

void ParseParticleObj(ParseContext& context, wpscene::WPParticleObject& wppartobj,
                      ParticleChildPtr child_ptr) {
    struct ChildData {
        ChildData() = default;
        ChildData(const wpscene::ParticleChild& o)
            : type(o.type),
              maxcount(o.maxcount),
              controlpointstartindex(o.controlpointstartindex),
              probability(o.probability) {}
        std::string type { "static" };
        i32         maxcount { 20 };
        i32         controlpointstartindex { 0 };
        float       probability { 1.0f };
    };

    wpscene::Particle*         p_particle_obj { nullptr };
    std::shared_ptr<SceneNode> spNode;
    ChildData                  child_data;

    bool        is_child            = child_ptr.child != nullptr;
    if (is_child) {
        p_particle_obj = &(child_ptr.child->obj);
        spNode         = std::make_shared<SceneNode>(Vector3f(child_ptr.child->origin.data()),
                                                     Vector3f(child_ptr.child->scale.data()),
                                                     Vector3f(child_ptr.child->angles.data()),
                                                     child_ptr.child->name);
        child_data     = ChildData(*child_ptr.child);

    } else {
        p_particle_obj = &wppartobj.particleObj;
        spNode         = std::make_shared<SceneNode>(Vector3f(wppartobj.origin.data()),
                                                     Vector3f(wppartobj.scale.data()),
                                                     Vector3f(wppartobj.angles.data()),
                                                     wppartobj.name);
        spNode->ID()   = wppartobj.id;
    }

    const auto override =
        ResolveParticleSubsystemOverride(wppartobj.instanceoverride, is_child);

    auto& particle_obj = *p_particle_obj;
    auto& vfs          = *context.vfs;

    const auto& particle_renderer = particle_obj.renderers.at(0);
    const auto  renderer_spec     = DescribeParticleRender(particle_renderer.name);
    if (! renderer_spec.has_value()) {
        LOG_ERROR("particle object '%s' has unsupported renderer '%s'",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str());
        return;
    }
    const auto& renderer          = *renderer_spec;
    const bool  render_rope_trail = renderer.UsesHistory();
    const bool  rope_shader       = renderer.IsRope();
    const bool  has_trail         = renderer.IsTrail();
    const std::string_view particle_debug_name =
        is_child ? std::string_view(child_ptr.child->name) : std::string_view(wppartobj.name);

    // Resolve the effective material without modifying the parsed particle asset. Child parsing,
    // runtime property registration, and diagnostics can therefore continue to inspect authored
    // data, while rope renderers receive the stock shader required by their segment ABI.
    auto effective_material = particle_obj.material;
    if (renderer.shader_selection == ParticleShaderSelection::StockRope) {
        effective_material.shader = "genericropeparticle";
    }

    // wppartobj.origin[1] = context.ortho_h - wppartobj.origin[1];

    if (! context.scene->cameraOrthographic &&
        ! context.scene->modelPerspectiveCameraName.empty()) {
        // In a 3D scene particles and model geometry share one perspective camera. This keeps the
        // view-dependent billboard basis and projection synchronized while the authored camera
        // scripts move the Sonic shot.
        spNode->SetCamera(context.scene->modelPerspectiveCameraName);
    } else if (particle_obj.flags[wpscene::Particle::FlagEnum::perspective]) {
        spNode->SetCamera("global_perspective");
    }

    SceneMaterial     material;
    WPShaderValueData svData;

    if (! is_child) {
        svData.parallaxDepth = { wppartobj.parallaxDepth[0], wppartobj.parallaxDepth[1] };
        svData.parallaxDepthAuthored = wppartobj.parallaxDepthAuthored;
    }

    const ParticleOrientationBasis orientation_basis =
        ResolveParticleOrientationBasis(particle_renderer);

    WPShaderInfo shaderInfo;
    shaderInfo.baseConstSvs                         = context.global_base_uniforms;
    shaderInfo.baseConstSvs["g_OrientationUp"]      = orientation_basis.up;
    shaderInfo.baseConstSvs["g_OrientationRight"]   = orientation_basis.right;
    shaderInfo.baseConstSvs["g_OrientationForward"] = orientation_basis.forward;
    shaderInfo.baseConstSvs["g_ViewUp"]             = std::array { 0.0f, 1.0f, 0.0f };
    shaderInfo.baseConstSvs["g_ViewRight"]          = std::array { 1.0f, 0.0f, 0.0f };
    shaderInfo.baseConstSvs["g_EyePosition"]        = std::array {
        static_cast<float>(context.ortho_w) / 2.0f,
        static_cast<float>(context.ortho_h) / 2.0f,
        1000.0f,
    };
    const int orientation_combo = ResolveParticleOrientationCombo(particle_renderer.orientation);
    shaderInfo.combos["ORIENTATION"] = std::to_string(orientation_combo);

    const auto spawn_type              = ParseSpawnType(child_data.type);
    const u32  authored_child_maxcount = static_cast<u32>(std::max<i32>(1, child_data.maxcount));
    u32        mesh_instancecount = static_cast<u32>(std::max<i32>(1, child_ptr.max_instancecount));
    if (is_child) {
        switch (spawn_type) {
        case ParticleSubSystem::SpawnType::STATIC:
            // A static child relationship creates one persistent child emitter at this authored
            // transform. Its `maxcount` is a runtime cap for reusable child instances, not a request
            // to pre-seed N identical emitters. Pre-seeding duplicates this transform exactly; for
            // additive sprite effects such as the Matrix rain that stacks the same glyph stream on
            // itself, over-brightens the result, and washes the parent blue tint toward white.
            mesh_instancecount = 1;
            break;
        case ParticleSubSystem::SpawnType::EVENT_FOLLOW:
            // Follow instances are bounded by live parent particles as well as by authored child
            // maxcount. This keeps one live parent particle from reserving the default 20 follow
            // instances that Wallpaper Engine cannot actually use at the same time.
            mesh_instancecount = std::max<u32>(
                1, std::min<u32>(authored_child_maxcount, child_ptr.parent_live_particle_slots));
            break;
        case ParticleSubSystem::SpawnType::EVENT_SPAWN:
        case ParticleSubSystem::SpawnType::EVENT_DEATH:
            // Spawn/death children can outlive the triggering parent particle, so keep their local
            // authored cap. The cap is local to this child subsystem, not multiplied by ancestor
            // static child counts.
            mesh_instancecount = authored_child_maxcount;
            break;
        }
    }

    const u32 maxcount = std::clamp(particle_obj.maxcount, 1u, kMaxParticleCount);
    const uint16_t trail_length = render_rope_trail
        ? static_cast<uint16_t>(
              std::clamp(particle_renderer.segments, 1, kMaxParticleTrailSegments))
        : 0;
    const auto live_particle_slots = CheckedParticleMeshCapacity(
        particle_debug_name,
        particle_renderer.name,
        { static_cast<uint64_t>(maxcount), static_cast<uint64_t>(mesh_instancecount) });
    if (! live_particle_slots.has_value()) return;
    auto mesh_capacity = live_particle_slots;
    if (render_rope_trail) {
        mesh_capacity = CheckedParticleMeshCapacity(
            particle_debug_name,
            particle_renderer.name,
            { static_cast<uint64_t>(maxcount),
              static_cast<uint64_t>(mesh_instancecount),
              static_cast<uint64_t>(trail_length) });
        if (! mesh_capacity.has_value()) return;
    }

    if (has_trail) {
        shaderInfo.baseConstSvs["g_RenderVar0"] = std::array {
            particle_renderer.length,
            particle_renderer.maxlength,
            // Sprite trails clamp length to this authored floor. Rope trails overwrite z each
            // frame with the normalized remainder, so they keep a parse-time zero here.
            render_rope_trail ? 0.0f : particle_renderer.minlength,
            static_cast<float>(maxcount - 1),
        };
        shaderInfo.combos["TRAILRENDERER"] = "1";
    }
    if (renderer.thick_format) {
        // Wallpaper Engine unconditionally selects THICKFORMAT for the stock rope renderer. The
        // endpoint size and color are part of that ABI and must reach the geometry shader so each
        // segment interpolates into the next particle instead of changing abruptly at its boundary.
        shaderInfo.combos["THICKFORMAT"] = "1";
    }
    if (rope_shader) {
        // Wallpaper Engine selects the GS-enabled vertex branch and expands each rope segment in
        // genericropeparticle.geom. Its subdivision value is the count of additional
        // pairs emitted between endpoints, so forward the authored value verbatim to the shader
        // combo rather than pre-tessellating it in the particle uploader.
        const uint32_t subdivision = static_cast<uint32_t>(
            std::lround(std::max(0.0f, particle_renderer.subdivision)));
        shaderInfo.combos["TRAILSUBDIVISION"] = std::to_string(subdivision);
    }

    if (! particle_obj.flags[wpscene::Particle::FlagEnum::spritenoframeblending] &&
        particle_obj.animationmode != "randomframe") {
        // randomframe encodes one atlas cell per particle. Frame blending would mix that cell with
        // the next one, so keep SPRITESHEETBLEND for sequenced sprite playback only.
        shaderInfo.combos["SPRITESHEETBLEND"] = "1";
    }

    std::optional<MaterialLoadResult> material_result;
    try {
        material_result = LoadMaterial(vfs,
                                       effective_material,
                                       context.scene.get(),
                                       spNode.get(),
                                       &material,
                                       &svData,
                                       context.user_properties,
                                       &shaderInfo,
                                       rope_shader ? GeometryStagePolicy::Required
                                                   : GeometryStagePolicy::MatchMaterial);
    } catch (const std::exception& e) {
        LOG_ERROR("load particleobj '%s' material exception: %s", wppartobj.name.c_str(), e.what());
    }
    if (! material_result.has_value()) {
        LOG_ERROR("load particleobj '%s' material faild", wppartobj.name.c_str());
        return;
    }
    LoadConstvalue(material, effective_material, shaderInfo);
    LoadUserShaderValue(material, effective_material, shaderInfo, context.user_properties);
    auto  spMesh             = std::make_shared<SceneMesh>(true);
    auto& mesh               = *spMesh;
    auto  animationmode      = ToAnimMode(particle_obj.animationmode);
    auto  sequencemultiplier = particle_obj.sequencemultiplier;
    float sprite_frame_count = 0.0f;
    if (const auto it = material.customShader.constValues.find("g_RenderVar1");
        it != material.customShader.constValues.end() && it->second.size() >= 3) {
        // g_RenderVar1.z is the SpriteAnimation frame count of the current particle texture.
        sprite_frame_count = it->second[2];
    }

    const ParticleRenderPlan render_plan {
        .renderer         = renderer.kind,
        .shader_selection = renderer.shader_selection,
        .expansion        = material_result->geometry_stage_loaded
                    ? ParticleExpansionMode::GeometryPoint
                    : ParticleExpansionMode::IndexedQuad,
        .thick_format     = renderer.thick_format || material.hasSprite,
    };
    ConfigureParticleMesh(mesh, *mesh_capacity, render_plan);

    std::function<void(float)> trail_uniform_update;
    if (render_rope_trail) {
        std::weak_ptr<SceneMesh> weak_mesh = spMesh;
        trail_uniform_update = [weak_mesh](float normalized_remainder) {
            const auto mesh = weak_mesh.lock();
            if (! mesh || mesh->Material() == nullptr) return;
            auto& values = mesh->Material()->customShader.constValues;
            const auto render_var = values.find("g_RenderVar0");
            if (render_var == values.end() || render_var->second.size() < 3) return;
            render_var->second[2] = normalized_remainder;
        };
    }

    const ParticleRawGenSpecOp spec_op = [=](const Particle& p, const ParticleRawGenSpec& spec) {
        auto& lifetime = *(spec.lifetime);
        if (lifetime <= 0.0f) {
            lifetime = 0.0f;
            return;
        }
        switch (animationmode) {
        case ParticleAnimationMode::RANDOMONE:
            lifetime = sprite_frame_count > 1.0f
                           ? RandomParticleFrameLifetime(p, sprite_frame_count)
                           : std::floor(p.init.lifetime);
            break;
        case ParticleAnimationMode::SEQUENCE:
            lifetime = (1.0f - (p.lifetime / p.init.lifetime)) * sequencemultiplier;
            break;
        }
    };
    auto particleSub = std::make_unique<ParticleSubSystem>(
        *context.scene->paritileSys,
        spMesh,
        maxcount,
        override.rate,
        child_data.maxcount,
        child_data.probability,
        spawn_type,
        render_plan,
        spec_op,
        trail_length,
        render_rope_trail ? static_cast<double>(particle_renderer.length) : 0.0,
        std::move(trail_uniform_update));
    auto* particle_subsystem = particleSub.get();
    particleSub->SetSceneNode(spNode.get());
    // PARTICLE_OVERRIDE_SCHEMA stores alpha in the live layer override record. It multiplies the
    // fully evaluated particle alpha at render time, rather than becoming part of the particle's
    // initializer state; this is what lets authored scripts fade all existing speed-line particles
    // to zero immediately when Sonic enters the slow state.
    particleSub->SetRuntimeAlphaOverride(override.alpha);
    // instanceoverride.size is baked into initializer output during cold parse. Keep that parsed
    // multiplier as the runtime reference so later user-property size edits can rescale live
    // particles by ratio instead of mistaking the multiplier for an absolute particle size.
    particleSub->SetRuntimeSizeReference(override.size);

    LoadEmitter(*context.scene->paritileSys, *particleSub, particle_obj, override.count);
    LoadInitializer(*particleSub, particle_obj, override, *context.scene->paritileSys);
    LoadOperator(*context.scene->paritileSys, *particleSub, particle_obj, override);
    LoadControlPoint(*particleSub, particle_obj, override);

    mesh.AddMaterial(std::move(material));
    spNode->AddMesh(spMesh);
    RegisterUserShaderValueBindings(
        context, effective_material, shaderInfo, spNode.get(), wppartobj.id, wppartobj.name);

    if (! is_child) {
        ConfigureBoneAttachment(context,
                                wppartobj.parent,
                                wppartobj.attachment,
                                Eigen::Affine3f(spNode->GetLocalTrans().cast<float>()),
                                "particle object",
                                wppartobj.name,
                                svData);
    }

    for (auto& child : particle_obj.children) {
        ParseParticleObj(context,
                         wppartobj,
                         {
                             .child                      = &child,
                             .node_parent                = spNode.get(),
                             .particle_parent            = particleSub.get(),
                             .max_instancecount          = static_cast<i32>(mesh_instancecount),
                             .parent_live_particle_slots = *live_particle_slots,
                         });
    }

    for (size_t renderer_index = 1; renderer_index < particle_obj.renderers.size();
         renderer_index++) {
        AttachExtraParticleRenderer(context,
                                    wppartobj,
                                    particle_obj,
                                    particle_obj.renderers[renderer_index],
                                    *spNode,
                                    *particleSub,
                                    svData,
                                    spec_op,
                                    maxcount,
                                    mesh_instancecount,
                                    particle_debug_name,
                                    ! is_child);
    }

    if (is_child)
        child_ptr.particle_parent->AddChild(std::move(particleSub));
    else {
        context.scene->paritileSys->subsystems.emplace_back(std::move(particleSub));
        context.scene->AddLayerRuntimeParticleSubsystem(wppartobj.id, particle_subsystem);
    }

    if (is_child) {
        child_ptr.node_parent->AppendChild(spNode);
        svData.InheritParentTransform(child_ptr.node_parent);
    } else {
        if (LayerUsesRoutedParent(wppartobj.parent, wppartobj.attachment)) {
            ConfigureInheritedParentBinding(context, wppartobj.parent, svData);
            context.scene->sceneGraph->AppendChild(spNode);
        } else {
            AttachNodeToScene(context, spNode, wppartobj.parent, wppartobj.name, &svData);
        }
        context.object_nodes[wppartobj.id] = spNode;
        context.scene->AddLayerRuntimeNode(wppartobj.id, spNode.get());
        RegisterLayerSceneState(
            context, wppartobj.id, wppartobj.parent, wppartobj.attachment, wppartobj.visible);
        context.scene->ApplyLayerVisibility(wppartobj.id);
    }
    context.shader_updater->SetNodeData(spNode.get(), svData);

    if (! is_child && particle_obj.starttime > 0) {
        particle_subsystem->Prewarm(static_cast<double>(particle_obj.starttime));
    }
}

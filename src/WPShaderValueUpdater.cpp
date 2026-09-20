#include "WPShaderValueUpdater.hpp"
#include "WPNodeTransformResolver.hpp"
#include "SkinningShaderContract.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Transform.h"
#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneTextPrimitive.h"
#include "SpriteAnimation.hpp"
#include "SpecTexs.hpp"
#include "Core/ArrayHelper.hpp"
#include "Utils/Algorism.h"
#include "Utils/Eigen.h"
#include "Utils/Logging.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

using namespace wallpaper;
using namespace Eigen;

namespace
{
constexpr float kDefaultMouseCoord = 0.5f;
constexpr double kParallaxDelayRange = 3.0;
constexpr double kParallaxResponseRate = 10.0;
constexpr std::array<uint32_t, 3> kAudioSpectrumResolutions { 16, 32, 64 };
constexpr std::array<const char*, 3> kAudioSpectrumLeftUniforms {
    "g_AudioSpectrum16Left",
    "g_AudioSpectrum32Left",
    "g_AudioSpectrum64Left",
};
constexpr std::array<const char*, 3> kAudioSpectrumRightUniforms {
    "g_AudioSpectrum16Right",
    "g_AudioSpectrum32Right",
    "g_AudioSpectrum64Right",
};

std::array<float, 4> TextureTexelUniform(const std::array<i32, 4>& resolution) {
    const auto width  = static_cast<float>(resolution[0]);
    const auto height = static_cast<float>(resolution[1]);
    // Read the dimensions of the resource actually bound to the texture slot and publish
    // reciprocal dimensions in `.xy` followed by the dimensions in `.zw`. Shadow PCF uses `.xy`
    // as its atlas sampling step, so this value must describe the physical allocation rather
    // than its logical content rectangle; otherwise every PCF tap collapses onto the same
    // comparison coordinate.
    return { 1.0f / width, 1.0f / height, width, height };
}

struct MeshBounds2D {
    bool     valid { false };
    Vector3d center { Vector3d::Zero() };
    Vector2d halfExtent { Vector2d::Ones() };
};

Matrix4d ApplyMeshGeometryTransform(const Matrix4d& model, const SceneMesh* mesh) {
    if (mesh == nullptr) return model;
    return model * mesh->GeometryTransform().matrix().cast<double>();
}

Matrix4d DestinationViewProjection(const SceneCamera& camera, bool reflected) {
    if (!reflected) return camera.GetViewProjectionMatrix();
    // The mirror belongs to the incoming destination, after the view and before object
    // placement. Invert projection Y for the top-down framebuffer convention while leaving
    // the authored model matrix unchanged. Private projections select the ordinary branch.
    Matrix4d projection = camera.GetProjectionMatrix();
    projection(1, 1) = -projection(1, 1);
    return projection * camera.GetViewMatrix() *
        Affine3d(Eigen::Scaling(1.0, -1.0, 1.0)).matrix();
}

Matrix4d ModelPerspectiveViewProjection(const Scene& scene, const Matrix4d& incoming_projection,
                                        Matrix4d destination, bool centered_destination, int32_t layer_id,
                                        uint64_t frame_serial, bool reflected, bool trace = true) {
    constexpr double kMinimumFrameFov = 0.1;
    constexpr double kMaximumFrameFov = 179.9;
    constexpr double kPerspectiveFrameDistance = 2000.0;
    constexpr double kNearPlane = 5.0;
    constexpr double kMinimumFarPlane = 15000.0;
    constexpr double kFarPlaneMargin = 1000.0;

    // The angle belongs to the scene frame, while distance is derived from this invocation's
    // incoming projection. An orthographic frame uses its effective auxiliary angle; a 3D
    // frame derives that angle from its selected projection at a reference distance. A fixed
    // eye combined with the raw auxiliary angle would shrink the Z=0 plane and lose live zoom.
    const double angle = scene.cameraOrthographic
        ? Radians(std::clamp(static_cast<double>(scene.generalProjection.perspectiveOverrideFov),
                             kMinimumFrameFov, kMaximumFrameFov))
        : 2.0 * std::atan(1.0 /
            (scene.activeCamera->GetProjectionMatrix()(1, 1) * kPerspectiveFrameDistance));
    const double distance = 1.0 / (std::tan(angle * 0.5) * incoming_projection(1, 1));
    const double far_plane = std::max(kMinimumFarPlane, distance + kFarPlaneMargin);
    const double aspect = static_cast<double>(scene.physicalOutputExtent[0]) /
                          static_cast<double>(scene.physicalOutputExtent[1]);

    // The ordinary orthographic view already includes the scene's canvas-framing anchor in
    // its translation. A composition child walk instead supplies identity destination and a
    // centered private projection, so that invocation still owes the same frame-owned anchor.
    // Consume it here, in destination space, rather than baking it into the child's model or
    // inheriting the composition owner's view. The anchor uses canvas units, not the private
    // target size or physical output pixels, and belongs only to an orthographic scene frame;
    // a centered child destination in a 3D scene has no such translation to consume.
    const Vector2d frame_anchor = centered_destination && scene.cameraOrthographic
        ? Vector2d(scene.ortho[0] * 0.5, scene.ortho[1] * 0.5) : Vector2d::Zero();
    const Vector3d incoming_translation = destination.block<3, 1>(0, 3);
    destination.block<2, 1>(0, 3) -= frame_anchor;
    // Preserve all other incoming X/Y displacement and basis, including root parallax and
    // reflection, and replace only destination Z. The signed projection scale is significant:
    // reflection is incoming draw state, not a second mirror applied after this projection.
    // Neither the shared camera nor the frame eye/basis uniforms are modified here.
    destination(2, 3) = -distance;
    const Matrix4d projection = Perspective(angle, aspect, kNearPlane, far_plane);
    if (trace && std::getenv("WESCENE_TRACE_MODEL_PROJECTION") != nullptr) {
        LOG_INFO("SceneModelProjection: frame=%llu layer=%d reflection=%s orthographic=%s "
                 "angle-radians=%.9f incoming-p11=%.9f distance=%.9f aspect=%.9f "
                 "near=%.6f far=%.6f incoming-destination=[%.9f %.9f %.9f] "
                 "destination=[%.9f %.9f %.9f] p00=%.9f p11=%.9f p22=%.9f p23=%.9f "
                 "centered-destination=%s frame-anchor=[%.9f %.9f]",
                 static_cast<unsigned long long>(frame_serial), layer_id,
                 reflected ? "true" : "false", scene.cameraOrthographic ? "true" : "false",
                 angle, incoming_projection(1, 1), distance, aspect, kNearPlane, far_plane,
                 incoming_translation.x(), incoming_translation.y(), incoming_translation.z(),
                 destination(0, 3), destination(1, 3), destination(2, 3),
                 projection(0, 0), projection(1, 1), projection(2, 2), projection(2, 3),
                 centered_destination ? "true" : "false", frame_anchor.x(), frame_anchor.y());
    }
    return projection * destination;
}

std::array<float, 12> NormalizedModelBasis(const Matrix4d& model) {
    // Both normal uploads remove the length of each basis column directly, retaining shear and
    // reflection. A mat3 occupies three padded float4 columns in the DXC constant-buffer layout.
    Eigen::Matrix3d basis = model.topLeftCorner<3, 3>();
    std::array<float, 12> packed {};
    for (int column = 0; column < 3; ++column) {
        basis.col(column).normalize();
        for (int row = 0; row < 3; ++row) {
            packed[static_cast<size_t>(column) * 4 + row] =
                static_cast<float>(basis(row, column));
        }
    }
    return packed;
}

float SanitizeMouseCoord(double value) {
    if (! std::isfinite(value)) return kDefaultMouseCoord;
    return std::clamp(static_cast<float>(value), 0.0f, 1.0f);
}

MeshBounds2D ComputeMeshBounds2D(const SceneMesh* mesh) {
    if (mesh == nullptr || mesh->VertexCount() == 0) return {};

    if (mesh->HasBounds()) {
        const auto min_pos = mesh->BoundsMin().cast<double>();
        const auto max_pos = mesh->BoundsMax().cast<double>();
        const auto center  = (min_pos + max_pos) * 0.5;
        const auto halfExtent = Vector2d(std::max((max_pos.x() - min_pos.x()) * 0.5, 1e-6),
                                         std::max((max_pos.y() - min_pos.y()) * 0.5, 1e-6));
        return MeshBounds2D { .valid = true, .center = center, .halfExtent = halfExtent };
    }

    const auto& vertexArray = mesh->GetVertexArray(0);
    if (vertexArray.VertexCount() == 0) return {};

    const auto attrOffsets = vertexArray.GetAttrOffsetMap();
    if (!exists(attrOffsets, std::string(WE_IN_POSITION))) return {};

    const auto& posAttr     = attrOffsets.at(std::string(WE_IN_POSITION));
    const auto  components  = SceneVertexArray::TypeCount(posAttr.attr.type);
    const auto  stride      = vertexArray.OneSize();
    const auto  offset      = posAttr.offset / sizeof(float);
    const auto* vertexData  = vertexArray.Data();
    const auto  vertexCount = vertexArray.VertexCount();
    if (vertexData == nullptr || components < 2) return {};

    Vector3d minPos(std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity());
    Vector3d maxPos(-std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity());

    for (usize i = 0; i < vertexCount; ++i) {
        const auto base = i * stride + offset;
        const auto x    = static_cast<double>(vertexData[base + 0]);
        const auto y    = static_cast<double>(vertexData[base + 1]);
        const auto z    = components >= 3 ? static_cast<double>(vertexData[base + 2]) : 0.0;
        minPos = minPos.cwiseMin(Vector3d(x, y, z));
        maxPos = maxPos.cwiseMax(Vector3d(x, y, z));
    }

    const auto center     = (minPos + maxPos) * 0.5;
    const auto halfExtent = Vector2d(std::max((maxPos.x() - minPos.x()) * 0.5, 1e-6),
                                     std::max((maxPos.y() - minPos.y()) * 0.5, 1e-6));
    return MeshBounds2D { .valid = true, .center = center, .halfExtent = halfExtent };
}

bool IsModelMaterial(const SceneDraw& draw) {
    auto* mesh = draw.Mesh();
    const auto* material = mesh != nullptr ? mesh->Material() : nullptr;
    return material != nullptr && material->modelRenderState.has_value();
}

ShaderValue ToDxcCBufferMatrixUniform(const Matrix4d& matrix) {
    // The DXC WE prologue maps authored `mul(v, M)` to native `mul(M, v)` so shader code observes
    // the same column-vector transform contract as the renderer. Keep Eigen's column-major matrix
    // bytes untouched; changing layout here would make uniform upload policy depend on the source
    // spelling of every shader expression instead of on the single language bridge in WPShaderParser.
    return ShaderValue::fromMatrix(matrix.cast<float>());
}

ShaderValue ToDxcRowVectorSkinningUniform(std::span<const Affine3f> matrices) {
    return ShaderValue(PackDxcRowVectorSkinningUniform(matrices));
}

Matrix4d ComputeEffectTextureProjection(const SceneMesh* projectionMesh,
                                        const Matrix4d&  projectionModelTrans,
                                        const Matrix4d&  viewProjectionTrans) {
    if (projectionMesh == nullptr) return Matrix4d::Identity();

    const auto bounds = ComputeMeshBounds2D(projectionMesh);
    if (!bounds.valid) return viewProjectionTrans * projectionModelTrans;

    const auto localFromNormalized =
        (Affine3d(Eigen::Translation3d(bounds.center)) *
         Eigen::Scaling(bounds.halfExtent.x(), bounds.halfExtent.y(), 1.0))
            .matrix();
    return viewProjectionTrans * projectionModelTrans * localFromNormalized;
}

std::string_view ResolveEffectiveDrawCameraName(const SceneDraw& draw) {
    // Publication has an explicit projection selection and no parent hierarchy. Node-based
    // source draws still inherit their camera from the traversal parent, including glyph quads
    // whose logical text owner supplies the private source camera.
    if (draw.Phase() != nullptr) return draw.Camera();
    for (auto* current = draw.Node(); current != nullptr; current = current->Parent()) {
        if (!current->Camera().empty()) return current->Camera();
    }
    return {};
}

} // namespace

void WPShaderValueUpdater::PrepareFrame() {
    m_puppet_frame_serial++;
    m_modelTransformCache.clear();
    m_parallaxOffsetCache.clear();
    // 3D model camera paths are sampled before uniforms so the model-only camera projection,
    // g_EyePosition, and view-basis uniforms all describe the same frame. Scenes without model
    // camera paths return immediately inside Scene and keep the legacy 2D path untouched.
    if (m_scene != nullptr) {
        m_scene->UpdateModelCameraPath();
    }

    // Routed lights publish their ancestor-composed world transform once per frame so lighting,
    // shadow, and volumetric consumers all read the same placement that the authored parent
    // chain (including script-driven group transforms) produces.
    if (m_scene != nullptr) {
        WPNodeTransformResolver light_resolver(*m_scene,
                                               m_parallax,
                                               m_nodeDataMap,
                                               m_modelTransformCache,
                                               m_parallaxOffsetCache,
                                               m_parallaxPointerPos,
                                               m_puppet_frame_serial);
        for (auto& light : m_scene->lights) {
            if (! light || light->node() == nullptr) continue;
            // Publish for every parented light, not only transform-binding inheritors: a light
            // whose handle is routed (physical root parent, authored parent chain in the layer
            // binding) has no useful physical-graph fallback, and skipping it freezes the light
            // at its parse-time placement while scripts keep moving the authored parent.
            light->SetResolvedWorldTransform(
                light_resolver.ResolveRawModelTransform(light->node()));
        }
    }
    /*
        using namespace std::chrono;
        auto nowTime = system_clock::to_time_t(system_clock::now());
        auto cTime   = std::localtime(&nowTime);
        m_dayTime =
            (((cTime->tm_hour * 60) + cTime->tm_min) * 60 + cTime->tm_sec) / (24.0f * 60.0f
       * 60.0f);
    */
    UpdatePointerState();
    AdvanceAllPuppets();
}

void WPShaderValueUpdater::FrameBegin() {}

void WPShaderValueUpdater::UpdatePointerState() {
    m_pointerPosLast = m_pointerPos;
    m_pointerPos     = m_pointerPosInput;

    // Disabling camera parallax bypasses the complete look-at update, so retain the last filtered
    // target while disabled. Raw pointer uniforms above must continue advancing every frame
    // independently; otherwise cursor feedback inherits camera-only delay semantics.
    if (! m_parallax.enable) return;

    if (!(m_parallax.delay > 0.0f) || ! std::isfinite(m_parallax.delay)) {
        m_parallaxPointerPos = m_pointerPosInput;
        return;
    }

    const double frameTime = std::max(m_scene->frameTime, 0.0);
    // The authored 0..3 delay setting maps to a response rate instead of a settling duration.
    // Keep that curve intact: scene authors tune the slider against this exact relationship, and
    // the per-frame clamp allows an immediate response at high rates.
    const double responseRate =
        kParallaxResponseRate * (1.0 - static_cast<double>(m_parallax.delay) / kParallaxDelayRange);
    const double t = std::min(1.0, responseRate * frameTime);
    m_parallaxPointerPos = std::array {
        (float)algorism::lerp(t, m_parallaxPointerPos[0], m_pointerPosInput[0]),
        (float)algorism::lerp(t, m_parallaxPointerPos[1], m_pointerPosInput[1]),
    };
}

void WPShaderValueUpdater::AdvanceAllPuppets() {
    if (!m_scene) return;
    const double frame_time = m_scene->frameTime;
    std::unordered_set<const void*> advanced_runtimes;
    std::vector<SceneNode*> notification_nodes;

    for (auto& [addr, nodeData] : m_nodeDataMap) {
        if (!nodeData.puppet_layer.hasPuppet()) continue;
        const void* runtime = nodeData.puppet_layer.RuntimeIdentity();
        if (!advanced_runtimes.insert(runtime).second) continue;
        auto* node = static_cast<SceneNode*>(addr);
        const Affine3f world_from_model(
            ResolveModelTransformForProjection(node, nullptr, false).cast<float>());
        nodeData.puppet_layer.AdvanceIfNeeded(frame_time, m_puppet_frame_serial, world_from_model);
        notification_nodes.push_back(node);
    }

    if (m_scene->scriptHost != nullptr) {
        for (auto* node : notification_nodes) {
            m_scene->scriptHost->NotifyAnimationLayersAdvanced(node);
        }
    }
}

void WPShaderValueUpdater::FrameEnd() {}

Matrix4d WPShaderValueUpdater::ResolveModelTransformForProjection(
    const SceneDraw& draw, const SceneCamera* camera, bool apply_parallax) {
    if (m_scene == nullptr || !draw.Valid()) return Matrix4d::Identity();

    // Projection can run before render-graph refresh while ordinary uniform updates happen during
    // draw. Isolated caches make this query observe the current node graph without consuming a
    // matrix cached before a script changed an origin, scale, parent, or attachment in this frame.
    Map<void*, Matrix4d> local_model_cache;
    Map<void*, Vector3f> local_parallax_cache;
    WPNodeTransformResolver transform_resolver(*m_scene,
                                               m_parallax,
                                               m_nodeDataMap,
                                               local_model_cache,
                                               local_parallax_cache,
                                               m_parallaxPointerPos,
                                               m_puppet_frame_serial);


    return transform_resolver.ResolveParallaxedModelTransform(draw, camera, apply_parallax);
}

Matrix4d WPShaderValueUpdater::ResolveModelViewProjectionForInput(const SceneDraw& draw) {
    const auto& camera = *m_scene->activeCamera;
    Map<void*, Matrix4d> model_cache;
    Map<void*, Vector3f> parallax_cache;
    WPNodeTransformResolver resolver(*m_scene, m_parallax, m_nodeDataMap, model_cache,
                                      parallax_cache, m_parallaxPointerPos, m_puppet_frame_serial);
    Matrix4d destination = camera.GetViewMatrix();
    if (const auto* data = GetNodeData(draw.DataKey());
        data != nullptr && data->AppliesModelParallax()) {
        const Vector3d offset = resolver.ResolveParallaxOffset(draw, &camera).cast<double>();
        destination = destination * Affine3d(Translation3d(offset)).matrix();
    }
    const auto& owner = *m_scene->FindSceneObject(draw.LayerId(*m_scene));
    // Hit tests address the main scene destination, not its reflected copy. Reuse the draw's
    // auxiliary projection calculation, including live FOV/zoom and destination displacement;
    // projecting a model with the particle camera or a world-plane shortcut would diverge as
    // soon as a vertex leaves Z=0. Local caches observe preceding script transform writes.
    if (owner.ModelPerspective()) {
        return ModelPerspectiveViewProjection(*m_scene, camera.GetProjectionMatrix(), destination, false,
                                                owner.Id(), m_puppet_frame_serial, false, false);
    }
    return camera.GetProjectionMatrix() * destination;
}

WPShaderValueUpdater::EffectProjectionSnapshot
WPShaderValueUpdater::ResolveEffectProjectionSnapshot(const SceneImageEffectLayer& layer,
    std::string_view camera_name, bool reflected) {
    if (layer.UsesShapeDraw()) {
        // Shape's direct callback never enters the image destination operation that captures
        // owner I and placement MVP. Its stored effect matrices therefore retain the
        // constructor's identity on this draw path. Live owner/camera matrices still drive
        // rasterization; substituting them here would invent an image snapshot boundary merely
        // because a shader requests it.
        return { Matrix4d::Identity(), Matrix4d::Identity(),
                 Matrix4d::Identity(), Matrix4d::Identity() };
    }
    const bool composition = !camera_name.empty();
    const auto* camera = composition ? m_scene->cameras.at(std::string(camera_name)).get()
                                     : m_scene->activeCamera;

    // Snapshot I is replaced by the raw owner matrix before private raster state is installed.
    // Resolve the authored owner even when the final draw uses an identity fullscreen quad. Use
    // separate query caches because a preceding source draw or diagnostic may have evaluated
    // parallax with a private camera.
    Map<void*, Matrix4d> model_cache;
    Map<void*, Vector3f> parallax_cache;
    WPNodeTransformResolver resolver(*m_scene, m_parallax, m_nodeDataMap, model_cache,
                                      parallax_cache, m_parallaxPointerPos, m_puppet_frame_serial);
    const SceneDraw owner_draw(layer.Owner().LayerNode());
    EffectProjectionSnapshot snapshot;
    snapshot.layer_model = resolver.ResolveRawModelTransform(owner_draw);
    snapshot.placed_model = snapshot.layer_model;

    // The composition child walk uses a centered camera and identity destination throughout. Its
    // inverse-owner I participates in child rasterization, but is overwritten for this snapshot.
    // Outside that walk, root displacement remains in the destination. Source suppression and the
    // final material's camera describe later draw phases and cannot change either of these saved
    // factors.
    if (composition) {
        snapshot.view_projection = camera->GetProjectionMatrix();
        snapshot.incoming_view_projection = snapshot.view_projection;
    } else {
        snapshot.view_projection = DestinationViewProjection(*camera, reflected);
        const Vector3d offset = resolver.ResolveParallaxOffset(owner_draw, camera).cast<double>();
        snapshot.incoming_view_projection = snapshot.view_projection *
            Affine3d(Translation3d(offset)).matrix();
        snapshot.placed_model =
            Affine3d(Translation3d(offset)).matrix() * snapshot.layer_model;
    }
    return snapshot;
}

void WPShaderValueUpdater::MouseInput(double x, double y) {
    m_pointerPosInput[0] = SanitizeMouseCoord(x);
    m_pointerPosInput[1] = SanitizeMouseCoord(y);
}

void WPShaderValueUpdater::InitUniforms(const SceneDraw& draw, const ExistsUniformOp& existsOp) {
    m_nodeUniformInfoMap[draw.DataKey()] = WPUniformInfo();
    auto& info                  = m_nodeUniformInfoMap[draw.DataKey()];
    info.has_ALPHA              = existsOp("g_Alpha");
    info.has_COLOR              = existsOp("g_Color");
    info.has_COLOR4             = existsOp("g_Color4");
    info.has_MI                 = existsOp(G_MI);
    info.has_M                  = existsOp(G_M);
    info.has_AM                 = existsOp(G_AM);
    info.has_ANM                = existsOp(G_ANM);
    info.has_AVP                = existsOp(G_AVP);
    info.has_EM                 = existsOp(G_EM);
    info.has_RV0                = existsOp(G_RV0);
    info.has_RV1                = existsOp(G_RV1);
    info.has_RV2                = existsOp(G_RV2);
    info.has_RV3                = existsOp(G_RV3);
    info.has_RV4                = existsOp(G_RV4);
    info.has_MVP                = existsOp(G_MVP);
    info.has_LMM                = existsOp(G_LMM);
    info.has_EMVP               = existsOp(G_EMVP);
    info.has_EMVPI              = existsOp(G_EMVPI);
    info.has_MVPI               = existsOp(G_MVPI);
    info.has_ETVP               = existsOp(G_ETVP);
    info.has_ETVPI              = existsOp(G_ETVPI);

    info.has_VP = existsOp(G_VP);

    info.has_BONES            = existsOp(G_BONES);
    info.has_TIME             = existsOp(G_TIME);
    info.has_DAYTIME          = existsOp(G_DAYTIME);
    info.has_FRAMETIME        = existsOp(G_FRAMETIME);
    info.has_POINTERPOSITION  = existsOp(G_POINTERPOSITION);
    info.has_POINTERPOSITIONLAST = existsOp(G_POINTERPOSITIONLAST);
    info.has_POINTERSTATE     = existsOp(G_POINTERSTATE);
    info.has_PARALLAXPOSITION = existsOp(G_PARALLAXPOSITION);
    info.has_TEXELSIZE        = existsOp(G_TEXELSIZE);
    info.has_TEXELSIZEHALF    = existsOp(G_TEXELSIZEHALF);
    info.has_SCREEN           = existsOp(G_SCREEN);
    info.has_LP               = existsOp(G_LP);
    info.has_model_LCP        = IsModelMaterial(draw) && existsOp(G_LCP);
    info.has_LCR              = IsModelMaterial(draw) && existsOp(G_LCR);
    info.has_LPOINT_ORIGIN    = existsOp(G_LPOINT_ORIGIN);
    info.has_LPOINT_COLOR     = existsOp(G_LPOINT_COLOR);
    info.has_LSPOT_ORIGIN     = existsOp(G_LSPOT_ORIGIN);
    info.has_LSPOT_COLOR      = existsOp(G_LSPOT_COLOR);
    info.has_LSPOT_DIRECTION  = existsOp(G_LSPOT_DIRECTION);
    info.has_LSPOT_EXPONENT   = existsOp(G_LSPOT_EXPONENT);
    info.has_LDIR_COLOR       = existsOp(G_LDIR_COLOR);
    info.has_LDIR_DIRECTION   = existsOp(G_LDIR_DIRECTION);
    info.has_LTUBE_ORIGINA    = existsOp(G_LTUBE_ORIGINA);
    info.has_LTUBE_ORIGINB    = existsOp(G_LTUBE_ORIGINB);
    info.has_LTUBE_COLOR      = existsOp(G_LTUBE_COLOR);
    info.has_LFEAT_SHADOW_POINT_PROJ  = existsOp(G_LFEAT_SHADOW_POINT_PROJ);
    info.has_LFEAT_SHADOW_POINT_XFORM = existsOp(G_LFEAT_SHADOW_POINT_XFORM);
    info.has_LFEAT_SHADOW_PROJ        = existsOp(G_LFEAT_SHADOW_PROJ);
    info.has_LFEAT_SHADOW_PROJ_XFORM  = existsOp(G_LFEAT_SHADOW_PROJ_XFORM);
    // Particle shaders in a 3D scene share the model camera's view basis. Eye position has
    // its own frame-level binding below and does not depend on this draw-camera selection.
    const bool follows_scene_camera =
        IsModelMaterial(draw) ||
        (m_scene != nullptr && ! m_scene->modelPerspectiveCameraName.empty() &&
         draw.Camera() == m_scene->modelPerspectiveCameraName);
    // Every material requesting g_EyePosition receives the scene frame eye, including
    // lit/reflected image sources and private effect passes.
    info.has_EYE_POSITION     = existsOp(G_EYE_POSITION);
    info.has_NORMAL_MODEL_MATRIX = existsOp(G_NORMAL_MODEL_MATRIX);
    info.has_VIEWUP           = follows_scene_camera && existsOp(G_VIEWUP);
    info.has_VIEWRIGHT        = follows_scene_camera && existsOp(G_VIEWRIGHT);
    info.has_VIEWFORWARD      = follows_scene_camera && existsOp(G_VIEWFORWARD);
    for (size_t index = 0; index < kAudioSpectrumResolutions.size(); index++) {
        info.has_audio_spectrum_left[index] = existsOp(kAudioSpectrumLeftUniforms[index]);
        info.has_audio_spectrum_right[index] = existsOp(kAudioSpectrumRightUniforms[index]);
    }

    std::accumulate(begin(info.texs), end(info.texs), 0, [&existsOp](uint index, auto& value) {
        value.has_resolution = existsOp(WE_GLTEX_RESOLUTION_NAMES[index]);
        value.has_texel      = existsOp(WE_GLTEX_TEXEL_NAMES[index]);
        value.has_mipmap     = existsOp(WE_GLTEX_MIPMAPINFO_NAMES[index]);
        return index + 1;
    });
}

void WPShaderValueUpdater::UpdateUniforms(const SceneDraw& draw, sprite_map_t& sprites,
                                          const UpdateUniformOp& updateOp,
                                          const ShaderUniformOverrides* overrides) {
    const std::string effect_snapshot_camera = overrides != nullptr
        ? std::string(overrides->effect_snapshot_camera) : std::string();
    const bool reflection_snapshot = overrides != nullptr && overrides->reflection_snapshot;
    auto* pNode = draw.Node();
    const auto node_cam_name = ResolveEffectiveDrawCameraName(draw);
    const bool use_active_camera_for_uniforms =
        overrides != nullptr && overrides->use_active_camera_for_uniforms;
    const bool has_named_camera_override =
        overrides != nullptr && overrides->use_camera_override && !overrides->camera_name.empty();
    const bool has_camera_override = use_active_camera_for_uniforms || has_named_camera_override;
    const std::string_view uniform_cam_name =
        use_active_camera_for_uniforms
            ? std::string_view {}
            : (has_named_camera_override ? overrides->camera_name : node_cam_name);

    const SceneCamera* camera;
    if (! uniform_cam_name.empty()) {
        auto camera_it = m_scene->cameras.find(std::string(uniform_cam_name));
        if (camera_it != m_scene->cameras.end()) {
            camera = camera_it->second.get();
        } else {
            LOG_ERROR("ShaderUniformCameraOverride: camera '%.*s' not found for node '%s'",
                      static_cast<int>(uniform_cam_name.size()),
                      uniform_cam_name.data(),
                      draw.Name().c_str());
            camera = m_scene->activeCamera;
        }
    } else {
        camera = m_scene->activeCamera;
    }

    if (! camera) return;

    // A node-selected perspective camera changes projection without entering a private
    // destination. Perspective particles must retain the scene root's parallax displacement,
    // evaluated against the same active frame camera as ordinary layer drawing. Named pass
    // overrides still select their own source/composition contract below; keep the displacement
    // out of the raw owner matrix so projection and authored transform queries stay independent.
    const bool node_perspective_camera = !has_camera_override && camera->IsPerspective();
    const bool use_active_parallax_camera =
        (node_perspective_camera ||
         (has_camera_override && overrides->use_active_camera_for_parallax)) &&
        m_scene->activeCamera != nullptr;
    const SceneCamera* model_parallax_camera =
        use_active_parallax_camera ? m_scene->activeCamera : camera;
    const bool use_camera_local_transform_caches =
        has_camera_override && model_parallax_camera != m_scene->activeCamera;

    Map<void*, Matrix4d> localModelTransformCache;
    Map<void*, Vector3f> localParallaxOffsetCache;
    auto& modelTransformCache =
        use_camera_local_transform_caches ? localModelTransformCache : m_modelTransformCache;
    auto& parallaxOffsetCache =
        use_camera_local_transform_caches ? localParallaxOffsetCache : m_parallaxOffsetCache;

    WPNodeTransformResolver transformResolver(*m_scene,
                                              m_parallax,
                                              m_nodeDataMap,
                                              modelTransformCache,
                                              parallaxOffsetCache,
                                              m_parallaxPointerPos,
                                              m_puppet_frame_serial);


    // Text is now allowed to be a first-class renderable without a backing SceneMesh material.
    // The old updater returned early here, which made transform uniforms unavailable to any
    // render path that was not disguised as a mesh/custom-shader node. Keeping material access
    // optional lets the dedicated text pass reuse the same attachment/parallax/camera transform
    // logic while still skipping mesh-only material uniform work when no mesh exists.
    auto* material = draw.Mesh() != nullptr ? draw.Mesh()->Material() : nullptr;
    // auto& shadervs = material->customShader.updateValueList;
    // const auto& valueSet = material->customShader.valueSet;

    assert(exists(m_nodeUniformInfoMap, draw.DataKey()));
    const auto& info = m_nodeUniformInfoMap[draw.DataKey()];

    const auto* color_data = GetNodeData(draw.DataKey());
    if (color_data != nullptr && color_data->text_color_owner != nullptr &&
        (info.has_ALPHA || info.has_COLOR || info.has_COLOR4)) {
        const auto& primitive = *color_data->text_color_owner->Text();
        const auto color = primitive.ForegroundColor(m_scene->UsesHdrMaterials());
        // These engine uniforms carry the current glyph source state through every authored
        // material command. Write them after material controls have been populated, so an absent
        // control cannot leave a zero color in the buffer. Resolve the owner on each submission
        // because text shaping may replace the primitive without rebuilding these effect passes.
        if (info.has_ALPHA) updateOp("g_Alpha", color[3]);
        if (info.has_COLOR) {
            updateOp("g_Color", std::array<float, 3> { color[0], color[1], color[2] });
        }
        if (info.has_COLOR4) updateOp("g_Color4", color);
        if (std::getenv("WESCENE_TRACE_TEXT_COLOR") != nullptr) {
            LOG_INFO("SceneTextEffectColor: frame=%llu layer=%d node='%s' reflection=%s "
                     "rgba=[%.6f %.6f %.6f %.6f] brightness=%.6f uniforms=[%d %d %d] host-hdr=%s",
                     static_cast<unsigned long long>(m_puppet_frame_serial),
                     draw.LayerId(*m_scene), draw.Name().c_str(),
                     overrides != nullptr && overrides->reflection_pass ? "true" : "false",
                     color[0], color[1], color[2], color[3], primitive.object.brightness,
                     info.has_ALPHA, info.has_COLOR, info.has_COLOR4,
                     m_scene->UsesHdrMaterials() ? "true" : "false");
        }
    }

    bool hasNodeData = exists(m_nodeDataMap, draw.DataKey());
    const auto* layer = hasNodeData
        ? m_nodeDataMap.at(draw.DataKey()).effect_layer_projection.layer : nullptr;
    const bool image_prelighting_source = layer != nullptr &&
        layer->Owner().LayerNode() == pNode && layer->UsesPrelightingSource();
    if (info.has_BONES) {
        // Publication is a drawing phase of its authored owner. Read skinning from that
        // owner's canonical node, while keeping the phase's projection/material data separate.
        // Registering a second pose consumer in the animation-advance map would allow a phase
        // address to enter SceneNode notification paths and duplicate animation ownership.
        const void* pose_key = draw.Phase() != nullptr
            ? draw.Phase()->Owner().LayerNode() : draw.DataKey();
        const auto* pose_data = GetNodeData(pose_key);
        if (pose_data != nullptr && pose_data->puppet_layer.hasPuppet()) {
            const auto pose = pose_data->puppet_layer.PoseSnapshot();
            // PrepareFrame() is the sole pose-advance boundary. Uniform consumers only publish the
            // immutable snapshot selected for this frame, so mask pre-passes, clipped main passes
            // and effect writers cannot independently advance animation or mutate render topology.
            assert(pose.frame_serial == m_puppet_frame_serial);
            updateOp(G_BONES, ToDxcRowVectorSkinningUniform(pose.skinning));
            const char* trace_layer = std::getenv("WESCENE_TRACE_TRANSFORM_LAYER");
            if (draw.Phase() != nullptr && trace_layer != nullptr &&
                std::to_string(draw.LayerId(*m_scene)) == trace_layer &&
                (m_puppet_frame_serial == 1 || m_puppet_frame_serial == 121 ||
                 m_puppet_frame_serial == 601)) {
                LOG_INFO("ScenePuppetPublicationPose: layer=%d node='%s' owner='%s' "
                         "frame=%llu pose-frame=%llu revision=%llu bones=%zu",
                         draw.LayerId(*m_scene), draw.Name().c_str(),
                         draw.Phase()->Owner().LayerNode()->Name().c_str(),
                         static_cast<unsigned long long>(m_puppet_frame_serial),
                         static_cast<unsigned long long>(pose.frame_serial),
                         static_cast<unsigned long long>(pose.revision), pose.skinning.size());
            }
        }
    }

    if (material != nullptr) {
        const auto* bound_textures = overrides != nullptr && overrides->textures.has_value()
            ? &*overrides->textures : nullptr;
        const auto tex_count = std::min(bound_textures != nullptr
            ? bound_textures->size() : material->textures.size(), info.texs.size());
        for (size_t i = 0; i < tex_count; i++) {
            const auto& texture_uniforms = info.texs[i];
            if (!texture_uniforms.has_resolution && !texture_uniforms.has_texel &&
                !texture_uniforms.has_mipmap) continue;
            const auto& name = bound_textures != nullptr
                ? (*bound_textures)[i] : material->Texture(i);
            if (name.empty()) continue;
            // Effect chains can select a new destination after text re-layout or a pass swap.
            // Resolve dimensions from this invocation's descriptor bindings. Another invocation
            // can use the same live material with a different FBO permutation; neither its
            // texture names nor a parse-time list describe the image sampled by this draw.
            const auto target_it = m_scene->renderTargets.find(name);
            if (target_it != m_scene->renderTargets.end()) {
                const auto& target = target_it->second;
                const auto resolution = target.ResolutionVector();
                if (texture_uniforms.has_resolution) {
                    updateOp(WE_GLTEX_RESOLUTION_NAMES[i],
                             ShaderValue(array_cast<float>(resolution)));
                }
                if (texture_uniforms.has_texel) {
                    updateOp(WE_GLTEX_TEXEL_NAMES[i], TextureTexelUniform(resolution));
                }
                if (texture_uniforms.has_mipmap) {
                    updateOp(WE_GLTEX_MIPMAPINFO_NAMES[i], (float)target.mipmap_level);
                }
                continue;
            }
            const auto texture_it = m_scene->textures.find(name);
            if (texture_it == m_scene->textures.end()) continue;
            const auto resolution =
                m_scene->EffectiveImportedTextureResolution(texture_it->second);
            if (texture_uniforms.has_resolution) {
                updateOp(WE_GLTEX_RESOLUTION_NAMES[i],
                         ShaderValue(array_cast<float>(resolution)));
            }
            if (texture_uniforms.has_texel) {
                updateOp(WE_GLTEX_TEXEL_NAMES[i], TextureTexelUniform(resolution));
            }
        }
    }

    bool reqMI    = info.has_MI;
    bool reqM     = info.has_M;
    bool reqAM    = info.has_AM;
    bool reqMVP   = info.has_MVP;
    bool reqLMM   = info.has_LMM;
    bool reqEM    = info.has_EM;
    bool reqEMVP  = info.has_EMVP;
    bool reqEMVPI = info.has_EMVPI;
    bool reqMVPI  = info.has_MVPI;
    bool reqETVP  = info.has_ETVP;
    bool reqETVPI = info.has_ETVPI;

    Matrix4d viewProTrans = DestinationViewProjection(*camera,
        overrides != nullptr && overrides->reflection_raster);
    const auto camera_node = camera->GetAttachedNode();
    const bool reflection_pass = overrides != nullptr && overrides->reflection_pass;

    if (reqM || reqAM || reqMVP || reqLMM || reqEM || reqEMVP || reqEMVPI || reqMI ||
        reqMVPI || reqETVP || reqETVPI || info.has_VP || info.has_NORMAL_MODEL_MATRIX) {
        // The model uniform is the object's I transform. Camera parallax belongs to the
        // destination state used for this draw, so VP includes that translation while M does not.
        // Keeping the factors separate also makes shaders that multiply VP and M agree with
        // shaders that consume the combined MVP, without storing a camera-dependent transform on
        // any authored object or detached publication node. Ordinary offscreen source
        // rasterization selects an identity I matrix. Its geometry and skinning still belong to
        // the authored object; scripts and child queries always resolve that object's live
        // transform independently of this submitted draw state.
        const bool geometry_space =
            overrides != nullptr && overrides->model_space == ShaderModelSpace::Geometry;
        std::optional<EffectProjectionSnapshot> layer_snapshot;
        if (overrides != nullptr && overrides->model_space == ShaderModelSpace::LayerSnapshot) {
            // Framebuffer initialization replaces the model with the aligned owner while
            // retaining the incoming destination/camera. Reuse the same snapshot factors
            // as effect projection rather than rebasing this sample into the private source
            // or multiplying the composition's inverse-owner raster transform a second time.
            const auto& source_layer =
                *m_nodeDataMap.at(draw.DataKey()).effect_layer_projection.layer;
            layer_snapshot = ResolveEffectProjectionSnapshot(
                source_layer, effect_snapshot_camera, reflection_snapshot);
        }
        Matrix4d modelTrans = geometry_space
            ? Matrix4d::Identity()
            : (layer_snapshot ? layer_snapshot->layer_model
                              : transformResolver.ResolveRawModelTransform(draw));
        Vector3d destinationOffset = Vector3d::Zero();
        const bool composition_draw = !geometry_space && !layer_snapshot &&
            !use_active_camera_for_uniforms &&
            has_named_camera_override && overrides->use_active_camera_for_parallax &&
            camera_node != nullptr;
        const bool suppress_destination_parallax =
            overrides != nullptr && overrides->suppress_destination_parallax;
        // Private material rasterization uses identity destination state; root displacement
        // belongs to the restored scene destination. Composition children also retain identity
        // destination state for the entire child walk. A source's nonzero owner depth must never
        // move its pixels inside a private target.
        if (!geometry_space && !layer_snapshot && !composition_draw &&
            !suppress_destination_parallax && hasNodeData &&
            (camera == m_scene->activeCamera || use_active_parallax_camera) &&
            m_nodeDataMap.at(draw.DataKey()).AppliesModelParallax()) {
            destinationOffset =
                transformResolver.ResolveParallaxOffset(draw, model_parallax_camera).cast<double>();
        }
        if (composition_draw) {
            // The composition child walk replaces I with the inverse raw owner matrix,
            // destination with identity, and camera with a centered orthographic projection. Each
            // child's draw then multiplies that I by its own raw object matrix. Keep those exact
            // uniform factors: cancelling an attached view matrix only in MVP leaves M/MI and
            // normal-model uniforms in the wrong coordinate system. The enclosing draw restores
            // its saved state after this walk, so no parent/child parallax subtraction or owner
            // transform mutation is needed here.
            modelTrans = transformResolver.ResolveRawModelTransform(camera_node.get()).inverse() *
                modelTrans;
            viewProTrans = camera->GetProjectionMatrix();
        }

        if (image_prelighting_source) {
            // The prelighting source appends this local translation to incoming I. Its
            // independent AM upload below retains the raw owner transform. In particular, the
            // private source's texture-space placement must never reach scripts or bones.
            const auto& size = layer->SourceTextureContentSize();
            modelTrans = (modelTrans * Affine3d(Translation3d(
                size[0] * 0.5, size[1] * 0.5, 0.0)).matrix()).eval();
        }
        modelTrans = ApplyMeshGeometryTransform(modelTrans, draw.Mesh());
        const Matrix4d destinationTrans =
            Affine3d(Eigen::Translation3d(destinationOffset)).matrix();
        viewProTrans = layer_snapshot ? layer_snapshot->incoming_view_projection
                                     : (viewProTrans * destinationTrans).eval();

        const auto* model_owner = m_scene->FindSceneObject(draw.LayerId(*m_scene));
        if (model_owner != nullptr && model_owner->Kind() == SceneObjectKind::Model &&
            model_owner->ModelPerspective()) {
            Matrix4d projection = camera->GetProjectionMatrix();
            Matrix4d destination = composition_draw
                ? Matrix4d::Identity() : camera->GetViewMatrix();
            const bool reflected_destination = !composition_draw && overrides != nullptr &&
                                                overrides->reflection_raster;
            if (reflected_destination) {
                projection(1, 1) = -projection(1, 1);
                destination = destination * Affine3d(Eigen::Scaling(1.0, -1.0, 1.0)).matrix();
            }
            destination = destination * destinationTrans;
            viewProTrans = ModelPerspectiveViewProjection(*m_scene, projection, destination, composition_draw,
                model_owner->Id(), m_puppet_frame_serial, reflected_destination);
        }

        // Opt-in per-layer diagnostics follow every material phase, including private source
        // draws and the authored last pass. Comparing the raw model with the submitted model
        // identifies whether placement was changed by routing, parallax, or camera rebasing;
        // the clip-space origin then separates those changes from projection differences.
        const char* trace_layer = std::getenv("WESCENE_TRACE_TRANSFORM_LAYER");
        const int32_t layer_id = draw.LayerId(*m_scene);
        if (trace_layer != nullptr && std::to_string(layer_id) == trace_layer) {
            const auto raw_model = transformResolver.ResolveRawModelTransform(draw);
            const auto offset = transformResolver.ResolveParallaxOffset(draw, model_parallax_camera);
            const Vector4d clip_origin = viewProTrans * modelTrans.col(3);
            const auto* data = GetNodeData(draw.DataKey());
            const auto* effect_layer = data != nullptr ? data->effect_layer_projection.layer : nullptr;
            const Matrix4d local_model = pNode != nullptr ? pNode->GetLocalTrans() : Matrix4d::Identity();
            const auto* owner = m_scene->FindSceneObject(layer_id);
            const auto* owner_transform = owner != nullptr ? owner->RuntimeTransform().get() : nullptr;
            LOG_INFO("SceneTransformDraw: frame=%llu layer=%d node='%s' material='%s' "
                     "reflection=%s "
                     "camera='%.*s' suppress=%s parallax-root=%d owner-transform=%s "
                     "owner-revision=%llu stored-local=[%.6f %.6f %.6f] raw=[%.6f %.6f %.6f] "
                     "parallax=[%.6f %.6f %.6f] model=[%.6f %.6f %.6f] "
                     "destination=[%.6f %.6f %.6f] "
                     "clip=[%.6f %.6f %.6f %.6f]",
                     static_cast<unsigned long long>(m_puppet_frame_serial), layer_id,
                     draw.Name().c_str(), material != nullptr ? material->name.c_str() : "",
                     reflection_pass ? "true" : "false",
                     static_cast<int>(uniform_cam_name.size()), uniform_cam_name.data(),
                     data != nullptr && data->suppress_model_parallax ? "true" : "false",
                     transformResolver.ParallaxRoot(draw) != nullptr
                         ? transformResolver.ParallaxRoot(draw)->Id() : 0,
                     ((draw.Phase() != nullptr &&
                       draw.Phase()->PlacementSpace() == SceneDrawPhase::Space::Layer) ||
                      (effect_layer != nullptr && effect_layer->UsesOwnerTransform(pNode)))
                         ? "true" : "false",
                     static_cast<unsigned long long>(
                         owner_transform != nullptr ? owner_transform->Revision() : 0),
                     local_model(0, 3), local_model(1, 3), local_model(2, 3),
                     raw_model(0, 3), raw_model(1, 3), raw_model(2, 3),
                     offset.x(), offset.y(), offset.z(),
                     modelTrans(0, 3), modelTrans(1, 3), modelTrans(2, 3),
                     destinationOffset.x(), destinationOffset.y(), destinationOffset.z(),
                     clip_origin.x(), clip_origin.y(), clip_origin.z(), clip_origin.w());
            if (m_puppet_frame_serial == 0 || m_puppet_frame_serial == 120 ||
                m_puppet_frame_serial == 600) {
                const Matrix4d clip_model = viewProTrans * modelTrans;
                LOG_INFO("SceneDrawClipBasis: frame=%llu layer=%d node='%s' "
                         "x=[%.9f %.9f %.9f %.9f] y=[%.9f %.9f %.9f %.9f]",
                         static_cast<unsigned long long>(m_puppet_frame_serial), layer_id,
                         draw.Name().c_str(), clip_model(0, 0), clip_model(1, 0),
                         clip_model(2, 0), clip_model(3, 0), clip_model(0, 1), clip_model(1, 1),
                         clip_model(2, 1), clip_model(3, 1));
            }
            if (material != nullptr) {
                // Keep modulation beside the selected phase's placement trace. A visible owner
                // can still produce transparent pixels after a script alpha write; logging both
                // material overrides and shader defaults distinguishes that from clipping or a
                // missing publication draw without changing the submitted uniform values.
                const auto log_modulation = [&](const ShaderValues& values, const char* source) {
                    for (const auto& [name, value] : values) {
                        const bool standard_modulation = name == "g_Alpha" ||
                            name == "g_UserAlpha" || name == "g_Color4";
                        if (!standard_modulation && m_puppet_frame_serial != 0 &&
                            m_puppet_frame_serial != 120 && m_puppet_frame_serial != 600) continue;
                        std::string components;
                        for (size_t index = 0; index < value.size(); ++index) {
                            if (index != 0) components += ' ';
                            components += std::to_string(value[index]);
                        }
                        LOG_INFO("SceneDrawModulation: frame=%llu layer=%d node='%s' "
                                 "material='%s' source=%s uniform='%s' value=[%s] texture0='%s'",
                                 static_cast<unsigned long long>(m_puppet_frame_serial), layer_id,
                                 draw.Name().c_str(), material->name.c_str(), source, name.c_str(),
                                 components.c_str(),
                                 material->textures.empty() ? "" : material->Texture(0).c_str());
                    }
                };
                log_modulation(material->customShader.constValues, "material");
                if (material->customShader.shader != nullptr) {
                    log_modulation(material->customShader.shader->default_uniforms, "shader");
                }
            }
        }

        if (info.has_NORMAL_MODEL_MATRIX) {
            const auto packed = NormalizedModelBasis(modelTrans);
            updateOp(G_NORMAL_MODEL_MATRIX,
                     std::span<const float> { packed.data(), packed.size() });
        }

        if (reqM) updateOp(G_M, ToDxcCBufferMatrixUniform(modelTrans));
        // The alternate model has a separate publication site from I and the destination stack.
        if (reqAM && !image_prelighting_source) {
            updateOp(G_AM, ToDxcCBufferMatrixUniform(destinationTrans * modelTrans));
        }
        if (reqMI) updateOp(G_MI, ToDxcCBufferMatrixUniform(modelTrans.inverse()));
        if (reqMVP || reqMVPI) {
            const Matrix4d mvpTrans = viewProTrans * modelTrans;
            if (reqMVP) updateOp(G_MVP, ToDxcCBufferMatrixUniform(mvpTrans));
            if (reqMVPI) updateOp(G_MVPI, ToDxcCBufferMatrixUniform(mvpTrans.inverse()));
        }
        if (reqLMM || reqEM || reqEMVP || reqEMVPI || reqETVP || reqETVPI) {
            const WPShaderValueData* nodeDataPtr = hasNodeData ? &m_nodeDataMap.at(draw.DataKey()) : nullptr;
            const auto* layer = nodeDataPtr != nullptr
                ? nodeDataPtr->effect_layer_projection.layer : nullptr;
            Matrix4d layerModel = modelTrans;
            Matrix4d effectModel = modelTrans;
            Matrix4d effectMvp = viewProTrans * modelTrans;
            Matrix4d etvpTrans;
            if (layer != nullptr) {
                const auto snapshot = ResolveEffectProjectionSnapshot(*layer,
                    effect_snapshot_camera, reflection_snapshot);
                const auto& projectionCameraName = effect_snapshot_camera;
                // All effect matrices come from the owner's I and placement snapshots, before
                // private passes substitute their local camera and fullscreen mesh. LayerModel
                // always retains I. EffectModel/EffectMVP scale X/Y by the stored half-size in
                // intermediate segments and retain the raw snapshots in the final segment;
                // EffectTextureProjection always uses the scaled placement snapshot. Mesh
                // calibration, puppet animation bounds and render-target padding are not part
                // of this size domain. Fullscreen layers retain these owner snapshots and use
                // their source-content pixel extent, not the separate 2x2 raster card.
                layerModel = snapshot.layer_model;
                const auto& placedModel = snapshot.placed_model;
                const auto& cardSize = layer->EffectMatrixSize();
                const Matrix4d cardScale = Affine3d(Eigen::Scaling(
                    static_cast<double>(cardSize[0]) * 0.5,
                    static_cast<double>(cardSize[1]) * 0.5, 1.0)).matrix();
                effectModel = layerModel;
                effectMvp = snapshot.view_projection * placedModel;
                etvpTrans = effectMvp * cardScale;
                if (!layer->UsesLayerSpaceEffectMatrices(pNode)) {
                    effectModel = (effectModel * cardScale).eval();
                    effectMvp = etvpTrans;
                }
                if (std::getenv("WESCENE_TRACE_EFFECT_PROJECTION") != nullptr &&
                    (trace_layer == nullptr || std::to_string(layer_id) == trace_layer)) {
                    LOG_INFO("SceneEffectMatrixSnapshot: frame=%llu layer=%d node='%s' "
                             "snapshot-source=%s "
                             "camera='%s' composition=%s fullscreen=%s layer-space=%s "
                             "projection-diagonal=[%.9f %.9f %.9f] "
                             "card=[%.6f %.6f] raster-card=[%.6f %.6f] raw=[%.6f %.6f %.6f] "
                             "placed=[%.6f %.6f %.6f] clip=[%.6f %.6f %.6f %.6f]",
                             static_cast<unsigned long long>(m_puppet_frame_serial), layer_id,
                             draw.Name().c_str(),
                             layer->UsesShapeDraw() ? "shape-constructor" : "image-destination",
                             projectionCameraName.c_str(),
                             !projectionCameraName.empty() ? "true" : "false",
                             layer->IsFullscreen() ? "true" : "false",
                             layer->UsesLayerSpaceEffectMatrices(pNode) ? "true" : "false",
                             snapshot.view_projection(0, 0), snapshot.view_projection(1, 1),
                             snapshot.view_projection(2, 2), cardSize[0], cardSize[1],
                             layer->CardSize()[0], layer->CardSize()[1],
                             layerModel(0, 3), layerModel(1, 3), layerModel(2, 3),
                             placedModel(0, 3), placedModel(1, 3), placedModel(2, 3),
                             etvpTrans(0, 3), etvpTrans(1, 3), etvpTrans(2, 3), etvpTrans(3, 3));
                }
            } else if (reqETVP || reqETVPI) {
                etvpTrans = ComputeEffectTextureProjection(draw.Mesh(),
                                                           modelTrans, viewProTrans);
            }
            if (reqLMM) {
                updateOp(G_LMM, ToDxcCBufferMatrixUniform(layerModel));
                if (trace_layer != nullptr && std::to_string(layer_id) == trace_layer) {
                    LOG_INFO("SceneLayerModelDraw: frame=%llu layer=%d node='%s' "
                             "origin=[%.6f %.6f %.6f]",
                             static_cast<unsigned long long>(m_puppet_frame_serial), layer_id,
                             draw.Name().c_str(), layerModel(0, 3), layerModel(1, 3), layerModel(2, 3));
                }
            }
            if (reqEM && layer != nullptr) updateOp(G_EM, ToDxcCBufferMatrixUniform(effectModel));
            if (reqEMVP) updateOp(G_EMVP, ToDxcCBufferMatrixUniform(effectMvp));
            if (reqEMVPI) updateOp(G_EMVPI, ToDxcCBufferMatrixUniform(effectMvp.inverse()));
            if (reqETVP) updateOp(G_ETVP, ToDxcCBufferMatrixUniform(etvpTrans));
            if (reqETVPI) {
                const double determinant = etvpTrans.determinant();
                Matrix4d inverse = Matrix4d::Identity();
                if (std::abs(determinant) > 1e-12) inverse = etvpTrans.inverse();
                updateOp(G_ETVPI, ToDxcCBufferMatrixUniform(inverse));
                if (std::getenv("WESCENE_TRACE_EFFECT_PROJECTION") != nullptr &&
                    (trace_layer == nullptr || std::to_string(layer_id) == trace_layer)) {
                    // Evaluate the screen-to-effect mapping using the float matrix actually
                    // uploaded to the shader. Screen-space pointer effects unproject clip Z=0
                    // and divide XY by homogeneous W before restoring top-down texture UVs.
                    // Reporting both W and the divided coordinates distinguishes an off-card
                    // impulse from a pass that stopped drawing or a strength that became zero.
                    const Matrix4f uploaded_inverse = inverse.cast<float>();
                    const auto unproject = [&](const std::array<float, 2>& pointer) -> Vector4f {
                        return uploaded_inverse * Vector4f(
                            pointer[0] * 2.0f - 1.0f, 1.0f - pointer[1] * 2.0f, 0.0f, 1.0f);
                    };
                    const Vector4f current = unproject(m_pointerPos);
                    const Vector4f previous = unproject(m_pointerPosLast);
                    LOG_INFO("SceneEffectPointerProjection: frame=%llu layer=%d node='%s' "
                             "determinant=%.12g pointer=[%.6f %.6f] last=[%.6f %.6f] "
                             "left-down=%s homogeneous=[%.9f %.9f %.9f %.9f] "
                             "uv=[%.9f %.9f] last-uv=[%.9f %.9f] "
                             "inverse-rows=[%.9f %.9f %.9f %.9f; %.9f %.9f %.9f %.9f; "
                             "%.9f %.9f %.9f %.9f; %.9f %.9f %.9f %.9f]",
                             static_cast<unsigned long long>(m_puppet_frame_serial), layer_id,
                             draw.Name().c_str(), determinant, m_pointerPos[0], m_pointerPos[1],
                             m_pointerPosLast[0], m_pointerPosLast[1],
                             m_scene->cursorLeftDown ? "true" : "false",
                             current.x(), current.y(), current.z(), current.w(),
                             0.5f + 0.5f * current.x() / current.w(),
                             0.5f - 0.5f * current.y() / current.w(),
                             0.5f + 0.5f * previous.x() / previous.w(),
                             0.5f - 0.5f * previous.y() / previous.w(),
                             uploaded_inverse(0, 0), uploaded_inverse(0, 1),
                             uploaded_inverse(0, 2), uploaded_inverse(0, 3),
                             uploaded_inverse(1, 0), uploaded_inverse(1, 1),
                             uploaded_inverse(1, 2), uploaded_inverse(1, 3),
                             uploaded_inverse(2, 0), uploaded_inverse(2, 1),
                             uploaded_inverse(2, 2), uploaded_inverse(2, 3),
                             uploaded_inverse(3, 0), uploaded_inverse(3, 1),
                             uploaded_inverse(3, 2), uploaded_inverse(3, 3));
                }
            }
        }
    }

    if (info.has_VP) {
        updateOp(G_VP, ToDxcCBufferMatrixUniform(viewProTrans));
    }

    if (image_prelighting_source && (info.has_AM || info.has_ANM || info.has_AVP)) {
        // AM is raw owner I, optionally scaled by the instanced card/content ratio. AVP snapshots
        // the incoming camera and destination before private rasterization, excluding the owner
        // matrix. The volumetric writer below has its own contract and never enters this image
        // branch.
        const auto snapshot = ResolveEffectProjectionSnapshot(*layer,
            effect_snapshot_camera, reflection_snapshot);
        const auto& source = *layer->GetPrelightingSource();
        Matrix4d alternate_model = snapshot.layer_model;
        if (source.instanced) {
            const auto& card = layer->EffectMatrixSize();
            const auto& content = layer->SourceTextureContentSize();
            alternate_model.col(0) *= card[0] / content[0];
            alternate_model.col(1) *= card[1] / content[1];
        }
        if (info.has_AM) updateOp(G_AM, ToDxcCBufferMatrixUniform(alternate_model));
        if (info.has_ANM) {
            const auto basis = NormalizedModelBasis(alternate_model);
            updateOp(G_ANM, std::span<const float> {basis.data(), basis.size()});
        }
        if (info.has_AVP) {
            updateOp(G_AVP, ToDxcCBufferMatrixUniform(snapshot.incoming_view_projection));
        }
        if (std::getenv("WESCENE_TRACE_PRELIGHTING") != nullptr) {
            LOG_INFO("SceneImagePrelightingMatrices: frame=%llu layer=%d name='%s' "
                     "am-origin=[%.6f %.6f %.6f] am-scale=[%.6f %.6f %.6f] "
                     "avp-origin=[%.6f %.6f %.6f %.6f] incoming-camera='%s' "
                     "uniforms=[am=%d normal=%d avp=%d]",
                     static_cast<unsigned long long>(m_puppet_frame_serial),
                     layer->Owner().Id(), layer->Owner().Name().c_str(),
                     alternate_model(0, 3), alternate_model(1, 3), alternate_model(2, 3),
                     alternate_model.col(0).head<3>().norm(),
                     alternate_model.col(1).head<3>().norm(),
                     alternate_model.col(2).head<3>().norm(),
                     snapshot.incoming_view_projection(0, 3),
                     snapshot.incoming_view_projection(1, 3),
                     snapshot.incoming_view_projection(2, 3),
                     snapshot.incoming_view_projection(3, 3),
                     effect_snapshot_camera.c_str(), info.has_AM, info.has_ANM, info.has_AVP);
        }
    }

    if (hasNodeData) {
        const auto& vol = m_nodeDataMap.at(draw.DataKey());
        if (vol.volumetric_pass && vol.volumetric_light != nullptr) {
            const SceneLight& light = *vol.volumetric_light;
            const Matrix4d    alt_vp = light.AltViewProjection().cast<double>();
            // The camera projection is already reversed-depth, which is the convention the
            // volumetric shaders expect under REVERSEDEPTH for both the hull window Z and the
            // g_EffectModelMatrix unprojection.
            if (info.has_VP) updateOp(G_VP, ToDxcCBufferMatrixUniform(viewProTrans));
            if (info.has_AVP) updateOp(G_AVP, ToDxcCBufferMatrixUniform(alt_vp));
            if (info.has_EM) {
                updateOp(G_EM, ToDxcCBufferMatrixUniform(viewProTrans.inverse()));
            }
            if (info.has_AM) {
                if (light.type() == SceneLightType::Point) {
                    updateOp(G_AM, ToDxcCBufferMatrixUniform(Matrix4d::Identity()));
                } else {
                    updateOp(G_AM,
                             ToDxcCBufferMatrixUniform(light.WorldToLightClip().cast<double>()));
                }
            }
            const Vector3f origin  = light.WorldOrigin();
            const Vector3f forward = light.WorldForward();
            const Vector3f color   = light.color();
            // This opt-in trace ties the authored light animation to an executed volume
            // pass and its actual camera projection. Sampling every ten rendered frames
            // keeps short lightning peaks observable without flooding ordinary captures.
            if (std::getenv("WESCENE_TRACE_VOLUMETRICS") != nullptr &&
                m_puppet_frame_serial % 10 == 0) {
                const Vector4d clip = viewProTrans *
                    Vector4d(origin.x(), origin.y(), origin.z(), 1.0);
                LOG_INFO("SceneVolumetricUniforms: frame=%llu elapsed=%.6f light-layer=%d "
                         "pass='%s' intensity=%.6f radius=%.6f density=%.6f "
                         "origin=[%.6f %.6f %.6f] clip=[%.6f %.6f %.6f %.6f] "
                         "has-rv1=%d has-em=%d",
                         static_cast<unsigned long long>(m_puppet_frame_serial),
                         m_scene->elapsingTime, m_scene->LayerIdForNode(light.node()),
                         draw.Name().c_str(), light.intensity(), light.radius(), light.density(),
                         origin.x(), origin.y(), origin.z(),
                         clip.x(), clip.y(), clip.z(), clip.w(), info.has_RV1, info.has_EM);
                if (m_puppet_frame_serial <= 10) {
                    const Vector3d eye = camera->GetPosition();
                    const Vector3f frame_eye = m_scene->FrameEyePosition();
                    const Matrix4d inverse_vp = viewProTrans.inverse();
                    const Vector4d near_point = inverse_vp * Vector4d(0, 0, 1, 1);
                    const Vector4d far_point = inverse_vp * Vector4d(0, 0, 0, 1);
                    LOG_INFO("SceneVolumetricProjection: light-layer=%d pass='%s' "
                             "has-vp=%d has-avp=%d hull-scale=[%.6f %.6f %.6f] "
                             "hull-origin=[%.6f %.6f %.6f] "
                             "camera=[%.6f %.6f %.6f] ray-z=[%.6f %.6f] "
                             "frame-eye=[%.6f %.6f %.6f] volume-inside=%d",
                             m_scene->LayerIdForNode(light.node()), draw.Name().c_str(),
                             info.has_VP, info.has_AVP,
                             alt_vp(0, 0), alt_vp(1, 1), alt_vp(2, 2),
                             alt_vp(0, 3), alt_vp(1, 3), alt_vp(2, 3),
                             eye.x(), eye.y(), eye.z(),
                             near_point.z() / near_point.w(), far_point.z() / far_point.w(),
                             frame_eye.x(), frame_eye.y(), frame_eye.z(),
                             light.CameraInsideVolume(frame_eye,
                                 m_scene->activeCamera->GetDirection().cast<float>()));
                }
            }
            // g_RenderVar1: radius*0.99, cos(inner), cos(outer), intensity.
            if (info.has_RV0) {
                const auto atlas = light.ShadowAtlasUv();
                updateOp(G_RV0, std::array<float, 4> { atlas.x(), atlas.y(), atlas.z(), atlas.w() });
            }
            if (info.has_RV1) {
                updateOp(G_RV1,
                         std::array<float, 4> { light.radius() * 0.9900000095367432f,
                                                std::cos(light.innerCone() * SceneLight::Deg2Rad()),
                                                std::cos(light.outerCone() * SceneLight::Deg2Rad()),
                                                light.intensity() });
            }
            if (info.has_RV2) {
                updateOp(G_RV2,
                         std::array<float, 4> {
                             origin.x(), origin.y(), origin.z(), light.density() });
            }
            if (info.has_RV3) {
                if (light.type() == SceneLightType::Point && light.castsShadows() &&
                    m_scene->shadows.quality != 0) {
                    const auto proj = light.ShadowProjectionInfo();
                    updateOp(G_RV3,
                             std::array<float, 4> { proj.x(), proj.y(), proj.z(), proj.w() });
                } else {
                    updateOp(G_RV3,
                             std::array<float, 4> { forward.x(), forward.y(), forward.z(), 0.0f });
                }
            }
            if (info.has_RV4) {
                updateOp(G_RV4,
                         std::array<float, 4> { color.x(),
                                                color.y(),
                                                color.z(),
                                                light.volumetricsExponent() });
            }
        }
    }

    //	g_EffectTextureProjectionMatrix
    // shadervs.push_back({"g_EffectTextureProjectionMatrixInverse",
    // ShaderValue::ValueOf(Eigen::Matrix4f::Identity())});
    if (info.has_TIME) updateOp(G_TIME, (float)m_scene->elapsingTime);

    if (info.has_DAYTIME) updateOp(G_DAYTIME, (float)m_dayTime);

    if (info.has_POINTERPOSITION) updateOp(G_POINTERPOSITION, m_pointerPos);
    if (info.has_POINTERPOSITIONLAST) updateOp(G_POINTERPOSITIONLAST, m_pointerPosLast);
    if (info.has_POINTERSTATE) {
        // Wallpaper Engine cursor ripple shaders treat `.z` as the left-button impulse term. Keep
        // the other lanes neutral because their exact editor-side meanings are effect-specific, and
        // writing arbitrary non-zero values would inject force into authored feedback buffers.
        updateOp(G_POINTERSTATE,
                 std::array<float, 4> { 0.0f, 0.0f, m_scene->cursorLeftDown ? 1.0f : 0.0f, 0.0f });
    }
    if (info.has_FRAMETIME) {
        // Feedback effects such as cursor ripple integrate per-frame decay from this uniform. The
        // parser already exposes the authored default, but runtime updates must overwrite it so the
        // simulation sees the same frame delta that drives timers and scripts.
        updateOp(G_FRAMETIME, static_cast<float>(std::max(m_scene->frameTime, 0.0)));
    }

    if (info.has_TEXELSIZE) updateOp(G_TEXELSIZE, m_texelSize);

    if (info.has_TEXELSIZEHALF)
        updateOp(G_TEXELSIZEHALF, std::array { m_texelSize[0] / 2.0f, m_texelSize[1] / 2.0f });

    if (info.has_SCREEN)
        updateOp(G_SCREEN,
                 std::array<float, 3> {
                     m_screen_size[0], m_screen_size[1], m_screen_size[0] / m_screen_size[1] });

    if (info.has_EYE_POSITION) {
        // Image prelighting and the following effect passes consume the eye selected for the
        // scene frame. Their local raster cameras can change M/VP/MVP, but cannot replace that
        // world-space eye. Scene resolves orthographic canvas framing independently of path
        // look-at state, while perspective cameras retain the selected position on all axes.
        Vector3f eye = m_scene->FrameEyePosition();
        // The reflected walk changes this same frame eye. It is not a raster-camera replacement
        // and must not leak into the following ordinary pass.
        if (reflection_pass) eye.y() = -eye.y();
        updateOp(G_EYE_POSITION, std::array<float, 3> { eye.x(), eye.y(), eye.z() });

        if ((std::getenv("WESCENE_TRACE_EYE_POSITION") != nullptr ||
             std::getenv("WESCENE_TRACE_REFLECTION") != nullptr) &&
            (m_puppet_frame_serial <= 2 ||
             std::getenv("WESCENE_TRACE_DRAW_EVERY_FRAME") != nullptr)) {
            const auto raster_eye = camera->GetPosition();
            LOG_INFO("SceneEyePositionDraw: frame=%llu layer=%d node='%s' material='%s' "
                     "raster-camera='%.*s' reflection=%s orthographic=%s eye=[%.6f %.6f %.6f] "
                     "raster-eye=[%.6f %.6f %.6f] time=%.6f",
                     static_cast<unsigned long long>(m_puppet_frame_serial),
                     draw.LayerId(*m_scene), draw.Name().c_str(),
                     material != nullptr ? material->name.c_str() : "",
                     static_cast<int>(uniform_cam_name.size()), uniform_cam_name.data(),
                     reflection_pass ? "true" : "false",
                     m_scene->cameraOrthographic ? "true" : "false",
                     eye.x(), eye.y(), eye.z(),
                     raster_eye.x(), raster_eye.y(), raster_eye.z(), m_scene->elapsingTime);
        }
    }

    if (info.has_VIEWUP || info.has_VIEWRIGHT || info.has_VIEWFORWARD) {
        // These vectors describe the selected scene frame, like the eye position above. A
        // private source or composition camera changes raster projection only; deriving the
        // billboard basis from that temporary camera would replace the reflected frame state.
        const auto* frame_camera = m_scene->activeCamera;
        Vector3f forward = frame_camera->GetDirection().cast<float>();
        if (forward.norm() > 1e-6f) forward.normalize();
        Vector3f up = frame_camera->GetUp().cast<float>();
        if (up.norm() > 1e-6f) up.normalize();
        Vector3f right = forward.cross(up);
        if (right.norm() > 1e-6f) right.normalize();
        // Reflection changes the stored right/up vectors independently, leaving g_ViewForward
        // untouched. Recomputing a cross product after the mirror would silently undo that
        // handedness choice for billboard and model shader consumers.
        if (reflection_pass) {
            right.y() = -right.y();
            up.y() = -up.y();
        }

        if (info.has_VIEWUP)
            updateOp(G_VIEWUP, std::array<float, 3> { up.x(), up.y(), up.z() });
        if (info.has_VIEWRIGHT)
            updateOp(G_VIEWRIGHT, std::array<float, 3> { right.x(), right.y(), right.z() });
        if (info.has_VIEWFORWARD)
            updateOp(G_VIEWFORWARD,
                     std::array<float, 3> { forward.x(), forward.y(), forward.z() });
    }

    if (info.has_PARALLAXPOSITION) {
        Vector2f para { 0.5f, 0.5f };
        if (m_parallax.enable) {
            const Vector2f mouseCentered =
                Vector2f(&m_parallaxPointerPos[0]) - Vector2f { 0.5f, 0.5f };
            para = Vector2f { 0.5f, 0.5f } +
                   (Scaling(1.0f, -1.0f) * mouseCentered) * m_parallax.mouseinfluence;
        }
        updateOp(G_PARALLAXPOSITION, std::array { para[0], para[1] });
    }

    for (size_t index = 0; index < kAudioSpectrumResolutions.size(); index++) {
        if (!info.has_audio_spectrum_left[index] && !info.has_audio_spectrum_right[index]) continue;

        const uint32_t     resolution = kAudioSpectrumResolutions[index];
        std::vector<float> left;
        std::vector<float> right;
        std::vector<float> average;
        if (m_scene->scriptHost == nullptr ||
            ! m_scene->scriptHost->GetAudioSpectrum(resolution, &left, &right, &average)) {
            left.assign(resolution, 0.0f);
            right.assign(resolution, 0.0f);
            average.assign(resolution, 0.0f);
        }

        if (info.has_audio_spectrum_left[index]) {
            updateOp(kAudioSpectrumLeftUniforms[index],
                     std::span<const float> { left.data(), left.size() });
        }
        if (info.has_audio_spectrum_right[index]) {
            updateOp(kAudioSpectrumRightUniforms[index],
                     std::span<const float> { right.data(), right.size() });
        }

    }

    if (m_scene->scriptHost && pNode != nullptr) {
        m_scene->scriptHost->ApplyTextureAnimations(pNode, sprites, m_scene->frameTime);
    }

    for (auto& [i, sp] : sprites) {
        const auto& f      = sp.GetAnimateFrame(m_scene->frameTime);
        auto        grot   = WE_GLTEX_ROTATION_NAMES[i];
        auto        gtrans = WE_GLTEX_TRANSLATION_NAMES[i];
        updateOp(grot, std::array { f.xAxis[0], f.xAxis[1], f.yAxis[0], f.yAxis[1] });
        updateOp(gtrans, std::array { f.x, f.y });
    }

    if (info.has_LP || info.has_model_LCP || info.has_LCR) {
        std::array<float, 16> lights { 0 };
        std::array<float, 12> lights_color { 0 };
        std::array<float, 16> lights_color_radius { 0 };
        uint                  i = 0;
        for (auto& l : m_scene->lights) {
            if (i == 4) break;
            assert(l->node() != nullptr);
            const auto modelTrans = l->WorldTransform();
            lights[i * 4 + 0]     = (float)modelTrans(0, 3);
            lights[i * 4 + 1]     = (float)modelTrans(1, 3);
            lights[i * 4 + 2]     = (float)modelTrans(2, 3);
            // g_LightsColorRadius is distinct from g_LightsColorPremultiplied: Demon Core's
            // core.frag feeds rgb directly into ComputeLightSpecular and keeps the falloff radius in
            // w. Sending the radius-squared premultiplied payload here overdrives the sphere into a
            // clipped red/white blob, while color*intensity matches the shader's authored contract.
            const auto color_radius = l->colorIntensity();
            lights_color_radius[i * 4 + 0] = color_radius[0];
            lights_color_radius[i * 4 + 1] = color_radius[1];
            lights_color_radius[i * 4 + 2] = color_radius[2];
            lights_color_radius[i * 4 + 3] = l->radius();
            if (i < 3) {
                const auto& color = l->premultipliedColor();
                std::copy(color.begin(), color.end(), lights_color.begin() + i * 4);
            }
            i++;
        }
        if (info.has_LP) updateOp(G_LP, lights);
        if (info.has_LP || info.has_model_LCP) updateOp(G_LCP, lights_color);
        if (info.has_LCR) updateOp(G_LCR, lights_color_radius);
    }

    const bool has_lighting_v1 =
        info.has_LPOINT_ORIGIN || info.has_LPOINT_COLOR || info.has_LSPOT_ORIGIN ||
        info.has_LSPOT_COLOR || info.has_LSPOT_DIRECTION || info.has_LSPOT_EXPONENT ||
        info.has_LDIR_COLOR || info.has_LDIR_DIRECTION || info.has_LTUBE_ORIGINA ||
        info.has_LTUBE_ORIGINB || info.has_LTUBE_COLOR || info.has_LFEAT_SHADOW_POINT_PROJ ||
        info.has_LFEAT_SHADOW_POINT_XFORM || info.has_LFEAT_SHADOW_PROJ ||
        info.has_LFEAT_SHADOW_PROJ_XFORM;
    if (has_lighting_v1) {
        const bool shadows_on = m_scene->shadows.quality != 0;
        const DirectionalShadowView cascade_view = m_scene->ShadowCascadeView();
        std::vector<float> point_origin;
        std::vector<float> point_color;
        std::vector<float> point_proj;
        std::vector<float> point_xform;
        std::vector<float> spot_origin;
        std::vector<float> spot_color;
        std::vector<float> spot_direction;
        std::vector<float> spot_exponent;
        std::vector<float> dir_color;
        std::vector<float> dir_direction;
        std::vector<float> tube_a;
        std::vector<float> tube_b;
        std::vector<float> tube_color;
        std::vector<float> feat_proj;
        std::vector<float> feat_xform;

        auto append_vec4 = [](std::vector<float>& dst, float x, float y, float z, float w) {
            dst.insert(dst.end(), { x, y, z, w });
        };
        auto append_mat4 = [](std::vector<float>& dst, const Matrix4f& mat) {
            dst.insert(dst.end(), mat.data(), mat.data() + 16);
        };

        for (auto& light_ptr : m_scene->lights) {
            if (! light_ptr) continue;
            SceneLight& light = *light_ptr;
            const Vector3f origin  = light.WorldOrigin();
            const Vector3f forward = light.WorldForward();
            const Vector3f color   = light.colorIntensity();
            if (light.type() == SceneLightType::Point) {
                // LightingV1 array contract: the falloff radius rides the color vector's w and
                // the origin vector's w carries the falloff exponent. Crossing these slots makes
                // saturate(1 - distance/radius) collapse to zero for any scene whose lights sit
                // farther than a few units, which blacks out every lit surface.
                append_vec4(point_origin, origin.x(), origin.y(), origin.z(), light.exponent());
                append_vec4(point_color, color.x(), color.y(), color.z(), light.radius());
                if (shadows_on && light.castsShadows()) {
                    const auto proj = light.ShadowProjectionInfo();
                    const auto uv   = light.ShadowAtlasUv();
                    append_vec4(point_proj, proj.x(), proj.y(), proj.z(), proj.w());
                    append_vec4(point_xform, uv.x(), uv.y(), uv.z(), uv.w());
                } else if (shadows_on) {
                    append_vec4(point_proj, 0, 0, 0, 0);
                    append_vec4(point_xform, 0, 0, 0, 0);
                }
            } else if (light.type() == SceneLightType::Spot) {
                append_vec4(spot_origin, origin.x(), origin.y(), origin.z(),
                            std::cos(light.outerCone() * SceneLight::Deg2Rad()));
                append_vec4(spot_color, color.x(), color.y(), color.z(), light.radius());
                append_vec4(spot_direction, forward.x(), forward.y(), forward.z(),
                            std::cos(light.innerCone() * SceneLight::Deg2Rad()));
                append_vec4(spot_exponent, light.exponent(), 0, 0, 0);
                if (shadows_on && (light.castsShadows() || light.hasCookie())) {
                    // Shadow sampling compares against the reversed-depth atlas; cookie lookups
                    // only consume the xy of the same projection.
                    append_mat4(feat_proj, light.ShadowSpotWorldToLightClip(false));
                    const auto uv = light.ShadowAtlasUv();
                    append_vec4(feat_xform, uv.x(), uv.y(), uv.z(), uv.w());
                }
            } else if (light.type() == SceneLightType::Directional) {
                append_vec4(dir_color, color.x(), color.y(), color.z(), light.intensity());
                // SceneLight::WorldForward() is the direction in which a directional light
                // travels. LightingV1 consumes the opposite convention: its PBR helper takes the
                // vector from the shaded surface toward the light. Keep the authored forward axis
                // unchanged for cascade shadow cameras, but reverse it at this lighting-uniform
                // boundary so direct illumination and the shadow projection describe the same sun.
                append_vec4(dir_direction, -forward.x(), -forward.y(), -forward.z(), 0);
                if (shadows_on && light.castsShadows()) {
                    for (int cascade = 0; cascade < 3; ++cascade) {
                        append_mat4(feat_proj,
                                    light.ShadowCascadeWorldToLightClip(
                                        cascade,
                                        cascade_view,
                                        light.cascadeAtlasSlot(cascade).size,
                                        false));
                        const auto uv = light.cascadeAtlasSlot(cascade).packed
                                            ? Eigen::Vector4f(
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).x) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_w, 1)),
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).y) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_h, 1)),
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).size) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_w, 1)),
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).size) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_h, 1)))
                                            : Eigen::Vector4f::Zero();
                        append_vec4(feat_xform, uv.x(), uv.y(), uv.z(), uv.w());
                    }
                }
            } else if (light.type() == SceneLightType::Tube) {
                append_vec4(tube_a, origin.x(), origin.y(), origin.z(), light.exponent());
                append_vec4(tube_b, origin.x(), origin.y(), origin.z(), 0);
                append_vec4(tube_color, color.x(), color.y(), color.z(), light.radius());
            }
        }

        auto push_if = [&](bool enabled, std::string_view name, const std::vector<float>& values) {
            if (! enabled || values.empty()) return;
            updateOp(name, std::span<const float> { values.data(), values.size() });
        };
        push_if(info.has_LPOINT_ORIGIN, G_LPOINT_ORIGIN, point_origin);
        push_if(info.has_LPOINT_COLOR, G_LPOINT_COLOR, point_color);
        push_if(info.has_LSPOT_ORIGIN, G_LSPOT_ORIGIN, spot_origin);
        push_if(info.has_LSPOT_COLOR, G_LSPOT_COLOR, spot_color);
        push_if(info.has_LSPOT_DIRECTION, G_LSPOT_DIRECTION, spot_direction);
        push_if(info.has_LSPOT_EXPONENT, G_LSPOT_EXPONENT, spot_exponent);
        push_if(info.has_LDIR_COLOR, G_LDIR_COLOR, dir_color);
        push_if(info.has_LDIR_DIRECTION, G_LDIR_DIRECTION, dir_direction);
        push_if(info.has_LTUBE_ORIGINA, G_LTUBE_ORIGINA, tube_a);
        push_if(info.has_LTUBE_ORIGINB, G_LTUBE_ORIGINB, tube_b);
        push_if(info.has_LTUBE_COLOR, G_LTUBE_COLOR, tube_color);
        push_if(info.has_LFEAT_SHADOW_POINT_PROJ, G_LFEAT_SHADOW_POINT_PROJ, point_proj);
        push_if(info.has_LFEAT_SHADOW_POINT_XFORM, G_LFEAT_SHADOW_POINT_XFORM, point_xform);
        push_if(info.has_LFEAT_SHADOW_PROJ, G_LFEAT_SHADOW_PROJ, feat_proj);
        push_if(info.has_LFEAT_SHADOW_PROJ_XFORM, G_LFEAT_SHADOW_PROJ_XFORM, feat_xform);
    }
}

void WPShaderValueUpdater::SetNodeData(void* nodeAddr, const WPShaderValueData& data) {
    m_nodeDataMap[nodeAddr] = data;
}

void WPShaderValueUpdater::RemoveDrawData(const SceneDraw& draw) {
    // Removing an owner ends its runtime resources. These records can own puppet runtimes and
    // reference the owner's effect bridge even after the node has left the physical graph. Retire
    // both node and publication-phase keys before their objects are freed, so the next puppet
    // advance cannot notify a deleted handle.
    const auto key = draw.DataKey();
    m_nodeDataMap.erase(key);
    m_nodeUniformInfoMap.erase(key);
    // Surviving children have just lost an ancestor. Their derived placement must be evaluated
    // from the new hierarchy even if it was queried earlier in this frame's script work.
    m_modelTransformCache.clear();
    m_parallaxOffsetCache.clear();
}


const WPShaderValueData* WPShaderValueUpdater::GetNodeData(const void* node_addr) const {
    auto it = m_nodeDataMap.find(const_cast<void*>(node_addr));
    return it == m_nodeDataMap.end() ? nullptr : std::addressof(it->second);
}

WPShaderValueData* WPShaderValueUpdater::GetNodeData(const void* node_addr) {
    auto it = m_nodeDataMap.find(const_cast<void*>(node_addr));
    return it == m_nodeDataMap.end() ? nullptr : std::addressof(it->second);
}

std::optional<ShaderSkinningPose>
WPShaderValueUpdater::SkinningPose(SceneNode* node) const {
    const auto* node_data = GetNodeData(node);
    if (node_data == nullptr || !node_data->puppet_layer.hasPuppet()) return std::nullopt;

    const auto pose = node_data->puppet_layer.PoseSnapshot();
    if (pose.frame_serial != m_puppet_frame_serial) {
        LOG_ERROR("SkinningPose: node='%s' pose frame=%llu current frame=%llu",
                  node != nullptr ? node->Name().c_str() : "<null>",
                  static_cast<unsigned long long>(pose.frame_serial),
                  static_cast<unsigned long long>(m_puppet_frame_serial));
        return std::nullopt;
    }
    return ShaderSkinningPose {
        .matrices     = pose.skinning,
        .revision     = pose.revision,
        .frame_serial = pose.frame_serial,
    };
}

void WPShaderValueUpdater::SetTexelSize(float x, float y) { m_texelSize = { x, y }; }

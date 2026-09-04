#include "WPShaderValueUpdater.hpp"
#include "WPNodeTransformResolver.hpp"
#include "SkinningShaderContract.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Transform.h"
#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "SpriteAnimation.hpp"
#include "SpecTexs.hpp"
#include "Core/ArrayHelper.hpp"
#include "Utils/Algorism.h"
#include "Utils/Logging.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <ctime>
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

bool IsModelRenderNode(SceneNode* node) {
    auto* mesh = node != nullptr ? node->Mesh() : nullptr;
    const auto* material = mesh != nullptr ? mesh->Material() : nullptr;
    // `g_EyePosition` updates are scoped to materials explicitly marked by WPModelObject
    // materialization. This prevents the new 3D camera uniform support from changing any legacy 2D
    // image, effect, text, or particle shader that happens to declare the same uniform name.
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

Matrix4d ComputeEffectTextureProjection(const SceneNode* projectionNode,
                                        const SceneMesh* projectionMesh,
                                        const Matrix4d&  projectionModelTrans,
                                        const Matrix4d&  viewProjectionTrans) {
    if (projectionNode == nullptr || projectionMesh == nullptr) return Matrix4d::Identity();

    const auto bounds = ComputeMeshBounds2D(projectionMesh);
    if (!bounds.valid) return viewProjectionTrans * projectionModelTrans;

    const auto localFromNormalized =
        (Affine3d(Eigen::Translation3d(bounds.center)) *
         Eigen::Scaling(bounds.halfExtent.x(), bounds.halfExtent.y(), 1.0))
            .matrix();
    return viewProjectionTrans * projectionModelTrans * localFromNormalized;
}

std::string_view ResolveEffectiveNodeCameraName(const SceneNode* node) {
    // Effect-backed text still contributes intermediate bridge-source quads to the generic image
    // path while the logical text owner carries the camera binding. Walking ancestors here lets
    // those bridge-source quads inherit the same offscreen camera contract as the owning text
    // primitive, so text effects stay synchronized without any text-specific fallback camera path.
    for (auto* current = node; current != nullptr; current = current->Parent()) {
        if (!current->Camera().empty()) return current->Camera();
    }
    return {};
}

} // namespace

void WPShaderValueUpdater::PrepareFrame() {
    m_puppet_frame_serial++;
    m_modelTransformCache.clear();
    m_parallaxOffsetCache.clear();
    m_attachmentTransformCache.clear();
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
                                               m_attachmentTransformCache,
                                               nullptr,
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

    // Disabling camera parallax bypasses the complete official look-at update, so retain the last
    // filtered target while disabled. Raw pointer uniforms above must continue advancing every
    // frame independently; otherwise cursor feedback inherits camera-only delay semantics.
    if (! m_parallax.enable) return;

    if (!(m_parallax.delay > 0.0f) || ! std::isfinite(m_parallax.delay)) {
        m_parallaxPointerPos = m_pointerPosInput;
        return;
    }

    const double frameTime = std::max(m_scene->frameTime, 0.0);
    // Wallpaper Engine maps the authored 0..3 delay setting to a response rate instead of
    // treating it as a settling duration. Keep that curve intact: scene authors tune the slider
    // against this exact relationship, and the per-frame clamp preserves the native fast path.
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
        nodeData.puppet_layer.AdvanceIfNeeded(frame_time, m_puppet_frame_serial);
        notification_nodes.push_back(static_cast<SceneNode*>(addr));
    }

    // Surface synchronization is a frame-preparation transaction. Every binding reads the pose
    // snapshot cached above, while SceneImageEffectLayer deduplicates multiple consumers of the
    // same private surface and owns all camera, target and publication-mesh changes.
    for (auto& [addr, nodeData] : m_nodeDataMap) {
        (void)addr;
        if (!nodeData.puppet_layer.hasPuppet() || nodeData.puppet_surface.layer == nullptr ||
            nodeData.puppet_surface.skinned_mesh == nullptr) {
            continue;
        }
        nodeData.puppet_surface.layer->PreparePuppetSurface(
            *m_scene,
            *nodeData.puppet_surface.skinned_mesh,
            nodeData.puppet_layer.PoseSnapshot(),
            m_puppet_frame_serial);
    }

    if (m_scene->scriptHost != nullptr) {
        for (auto* node : notification_nodes) {
            m_scene->scriptHost->NotifyAnimationLayersAdvanced(node);
        }
    }
}

void WPShaderValueUpdater::FrameEnd() {}

Matrix4d WPShaderValueUpdater::ResolveModelTransformForProjection(
    SceneNode* node, const SceneCamera* camera, bool apply_parallax) {
    if (m_scene == nullptr || node == nullptr) return Matrix4d::Identity();

    // Projection can run before render-graph refresh while ordinary uniform updates happen during
    // draw. Isolated caches make this query observe the current node graph without consuming a
    // matrix cached before a script changed an origin, scale, parent, or attachment in this frame.
    Map<void*, Matrix4d> local_model_cache;
    Map<void*, Vector3f> local_parallax_cache;
    Map<void*, Affine3f> local_attachment_cache;
    WPNodeTransformResolver transform_resolver(*m_scene,
                                               m_parallax,
                                               m_nodeDataMap,
                                               local_model_cache,
                                               local_parallax_cache,
                                               local_attachment_cache,
                                               camera,
                                               m_parallaxPointerPos,
                                               m_puppet_frame_serial);

    if (const auto* node_data = GetNodeData(node); node_data != nullptr) {
        transform_resolver.UpdateAttachmentParentIfNeeded(*node_data);
        if (const auto local_transform =
                transform_resolver.ResolveAttachmentLocalTransform(node);
            local_transform.has_value()) {
            node->SetLocalAffine(*local_transform);
        }
    }

    return transform_resolver.ResolveParallaxedModelTransform(node, camera, apply_parallax);
}

void WPShaderValueUpdater::MouseInput(double x, double y) {
    m_pointerPosInput[0] = SanitizeMouseCoord(x);
    m_pointerPosInput[1] = SanitizeMouseCoord(y);
}

void WPShaderValueUpdater::InitUniforms(SceneNode* pNode, const ExistsUniformOp& existsOp) {
    m_nodeUniformInfoMap[pNode] = WPUniformInfo();
    auto& info                  = m_nodeUniformInfoMap[pNode];
    info.has_MI                 = existsOp(G_MI);
    info.has_M                  = existsOp(G_M);
    info.has_AM                 = existsOp(G_AM);
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
    info.has_model_LCP        = IsModelRenderNode(pNode) && existsOp(G_LCP);
    info.has_LCR              = IsModelRenderNode(pNode) && existsOp(G_LCR);
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
    // Particle shaders in a 3D scene are fed by the same camera/destination chain as model
    // shaders. Their eye and view basis must therefore follow the named model camera as it
    // moves; leaving the parse-time orthographic constants in place rotates trail ribbons and
    // view-dependent particle quads against a different camera than their projection matrix.
    const bool follows_scene_camera =
        IsModelRenderNode(pNode) ||
        (m_scene != nullptr && ! m_scene->modelPerspectiveCameraName.empty() && pNode != nullptr &&
         pNode->Camera() == m_scene->modelPerspectiveCameraName);
    info.has_EYE_POSITION     = follows_scene_camera && existsOp(G_EYE_POSITION);
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

void WPShaderValueUpdater::UpdateUniforms(SceneNode* pNode, sprite_map_t& sprites,
                                          const UpdateUniformOp& updateOp,
                                          const ShaderUniformOverrides* overrides) {
    const auto node_cam_name = ResolveEffectiveNodeCameraName(pNode);
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
                      pNode != nullptr ? pNode->Name().c_str() : "<null>");
            camera = m_scene->activeCamera;
        }
    } else {
        camera = m_scene->activeCamera;
    }

    if (! camera) return;

    const bool use_active_parallax_camera =
        has_camera_override && overrides->use_active_camera_for_parallax &&
        m_scene->activeCamera != nullptr;
    const SceneCamera* model_parallax_camera =
        use_active_parallax_camera ? m_scene->activeCamera : camera;
    const bool use_camera_local_transform_caches =
        has_camera_override && model_parallax_camera != m_scene->activeCamera;

    Map<void*, Matrix4d> localModelTransformCache;
    Map<void*, Vector3f> localParallaxOffsetCache;
    Map<void*, Affine3f> localAttachmentTransformCache;
    auto& modelTransformCache =
        use_camera_local_transform_caches ? localModelTransformCache : m_modelTransformCache;
    auto& parallaxOffsetCache =
        use_camera_local_transform_caches ? localParallaxOffsetCache : m_parallaxOffsetCache;
    auto& attachmentTransformCache =
        use_camera_local_transform_caches ? localAttachmentTransformCache
                                          : m_attachmentTransformCache;

    WPNodeTransformResolver transformResolver(*m_scene,
                                              m_parallax,
                                              m_nodeDataMap,
                                              modelTransformCache,
                                              parallaxOffsetCache,
                                              attachmentTransformCache,
                                              use_camera_local_transform_caches
                                                  ? model_parallax_camera
                                                  : m_scene->activeCamera,
                                              m_parallaxPointerPos,
                                              m_puppet_frame_serial);

    if (exists(m_nodeDataMap, pNode)) {
        auto& nodeData = m_nodeDataMap.at(pNode);
        transformResolver.UpdateAttachmentParentIfNeeded(nodeData);
        auto localTransform = transformResolver.ResolveAttachmentLocalTransform(pNode);
        if (localTransform.has_value()) {
            SceneImageEffectLayer* effectLayer { nullptr };
            if (!node_cam_name.empty()) {
                // pNode renders through its layer's private bridge camera exactly when it is the
                // effect-source route; resolve the bridge through the owning layer instead of a
                // camera back-reference.
                auto* candidate = m_scene->FindImageEffectLayer(m_scene->LayerIdForNode(pNode));
                if (candidate != nullptr &&
                    std::string_view(candidate->BridgeCameraName()) == node_cam_name) {
                    effectLayer = candidate;
                }
            }

            if (effectLayer != nullptr) {
                if (auto* worldNode = effectLayer->WorldNode()) {
                    worldNode->SetLocalAffine(*localTransform);
                    worldNode->UpdateTrans();
                    effectLayer->SyncResolvedNodeToWorld();
                }
            } else {
                pNode->SetLocalAffine(*localTransform);
            }
        }
    }

    pNode->UpdateTrans();

    if (! node_cam_name.empty()) {
        auto* effectLayer = m_scene->FindImageEffectLayer(m_scene->LayerIdForNode(pNode));
        if (effectLayer != nullptr &&
            std::string_view(effectLayer->BridgeCameraName()) == node_cam_name) {
            auto* worldNode   = effectLayer->WorldNode();
            if (worldNode != nullptr && exists(m_nodeDataMap, worldNode)) {
                auto& worldNodeData = m_nodeDataMap.at(worldNode);
                transformResolver.UpdateAttachmentParentIfNeeded(worldNodeData);
                if (worldNodeData.IsBoneAttached()) {
                    // Effect-backed layers draw their source into a private camera, then composite a
                    // detached final node back into the visible scene. That world node does not get a
                    // normal SceneNode tree update carrying the puppet bone and inherited parallax
                    // into the detached writer. Resolve the attachment here before synchronizing the
                    // effect output matrix so visibility changes preserve the same camera parallax.
                    auto localTransform = transformResolver.ResolveAttachmentLocalTransform(worldNode);
                    if (localTransform.has_value()) {
                        worldNode->SetLocalAffine(*localTransform);
                        worldNode->UpdateTrans();
                    }
                }
                if (worldNodeData.InheritsSceneParentTransform() || worldNodeData.IsBoneAttached()) {
                    const SceneCamera* displayCamera =
                        m_scene->activeCamera != nullptr ? m_scene->activeCamera : camera;
                    // Composition source routes publish a child layer's private authored effect
                    // through the neutral final composite. Keep that publisher on the raw routed
                    // world transform here; the actual compose-source pass applies the chosen
                    // source camera and camera-parallax exactly once. Syncing it to the active
                    // screen camera here would bake one parallax offset into the node transform,
                    // then the compose pass would add another one, which pulls effect-backed body
                    // parts away from direct siblings such as faces and eyes.
                    const auto worldModel =
                        effectLayer->PublishesPrivateFinalComposite()
                            ? transformResolver.ResolveRawModelTransform(worldNode)
                            : transformResolver.ResolveParallaxedModelTransform(
                                  worldNode, displayCamera, displayCamera != nullptr);
                    effectLayer->SyncResolvedNodeToMatrix(Affine3f(worldModel.cast<float>()));
                }
            }
        }
    }

    // Text is now allowed to be a first-class renderable without a backing SceneMesh material.
    // The old updater returned early here, which made transform uniforms unavailable to any
    // render path that was not disguised as a mesh/custom-shader node. Keeping material access
    // optional lets the dedicated text pass reuse the same attachment/parallax/camera transform
    // logic while still skipping mesh-only material uniform work when no mesh exists.
    auto* material = pNode->Mesh() != nullptr ? pNode->Mesh()->Material() : nullptr;
    // auto& shadervs = material->customShader.updateValueList;
    // const auto& valueSet = material->customShader.valueSet;

    assert(exists(m_nodeUniformInfoMap, pNode));
    const auto& info = m_nodeUniformInfoMap[pNode];

    bool hasNodeData = exists(m_nodeDataMap, pNode);
    if (hasNodeData) {
        auto& nodeData = m_nodeDataMap.at(pNode);
        for (const auto& el : nodeData.renderTargets) {
            if (m_scene->renderTargets.count(el.second) == 0) continue;
            const auto& rt = m_scene->renderTargets[el.second];

            const auto& unifrom_tex = info.texs[el.first];
            const auto  resolution  = rt.ResolutionVector();

            if (unifrom_tex.has_resolution) {
                // Runtime render targets expose one canonical resolution contract through
                // `ResolutionVector()`: physical size in `.xy`, logical content size in `.zw`.
                // Uniform updates should always forward that authoritative scene-side contract
                // directly instead of layering text-specific interpretation on top of it.
                updateOp(WE_GLTEX_RESOLUTION_NAMES[el.first],
                         ShaderValue(array_cast<float>(resolution)));
            }
            if (unifrom_tex.has_texel) {
                updateOp(WE_GLTEX_TEXEL_NAMES[el.first], TextureTexelUniform(resolution));
            }
            if (unifrom_tex.has_mipmap) {
                updateOp(WE_GLTEX_MIPMAPINFO_NAMES[el.first], (float)rt.mipmap_level);
            }
        }
        if (nodeData.puppet_layer.hasPuppet() && info.has_BONES) {
            const auto pose = nodeData.puppet_layer.PoseSnapshot();
            // PrepareFrame() is the sole pose-advance boundary. Uniform consumers only publish the
            // immutable snapshot selected for this frame, so mask pre-passes, clipped main passes
            // and effect writers cannot independently advance animation or mutate render topology.
            assert(pose.frame_serial == m_puppet_frame_serial);
            updateOp(G_BONES, ToDxcRowVectorSkinningUniform(pose.skinning));
        }
    }

    if (material != nullptr) {
        const auto tex_count = std::min(material->textures.size(), info.texs.size());
        for (size_t i = 0; i < tex_count; i++) {
            const auto& texture_uniforms = info.texs[i];
            if (! texture_uniforms.has_resolution && ! texture_uniforms.has_texel) continue;
            const auto& name = material->textures[i];
            if (name.empty() || m_scene->renderTargets.count(name) != 0) continue;
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
    bool reqEMVP  = info.has_EMVP;
    bool reqMVPI  = info.has_MVPI;
    bool reqETVP  = info.has_ETVP;
    bool reqETVPI = info.has_ETVPI;

    Matrix4d viewProTrans = camera->GetViewProjectionMatrix();
    const auto camera_node = camera->GetAttachedNode();

    if (info.has_VP) {
        updateOp(G_VP, ToDxcCBufferMatrixUniform(viewProTrans));
    }
    if (reqM || reqAM || reqMVP || reqLMM || reqEMVP || reqMI || reqMVPI || reqETVP ||
        reqETVPI) {
        Matrix4d modelTrans =
            transformResolver.ResolveParallaxedModelTransform(
                pNode, model_parallax_camera, uniform_cam_name != "effect");
        if (use_active_parallax_camera) {
            const auto source_camera_node = camera->GetAttachedNode();
            const auto source_camera_data =
                source_camera_node != nullptr && exists(m_nodeDataMap, source_camera_node.get())
                    ? &m_nodeDataMap.at(source_camera_node.get())
                    : nullptr;
            if (source_camera_data != nullptr && source_camera_data->AppliesModelParallax()) {
                // Composition-source routes project a child through the source camera, then the
                // parent composition layer publishes that source texture through its own final
                // scene-space writer. The child model therefore needs active-camera parallax minus
                // the parallax already represented by the source camera's attached layer; otherwise
                // parent depth is added twice for authored parallax groups, while source-camera
                // relative parallax drops the root fallback movement for groups whose final writer
                // deliberately suppresses its own model parallax.
                const auto source_parallax =
                    transformResolver.ResolveParallaxOffset(source_camera_node.get(),
                                                            model_parallax_camera);
                modelTrans =
                    Affine3d(Eigen::Translation3d((-source_parallax).cast<double>())).matrix() *
                    modelTrans;
            }
        }

        if (has_named_camera_override && overrides->use_active_camera_for_parallax &&
            camera_node != nullptr) {
            // A composition source is rasterized in the owning layer's local coordinate system and
            // its neutral final composite applies the owning layer's world transform afterwards.
            // Routed children, however, resolve directly to world space because their physical
            // SceneNode parent is only an ordering proxy. Rebase those world matrices through the
            // exact source-camera view used by the pass: VP * inverse(V) cancels the camera pose,
            // while inverse(cameraWorld) removes the composition root and every routed ancestor.
            // The resulting source texture therefore contains child-local geometry, leaving the
            // final publisher as the sole owner of parent translation, rotation, and scale.
            const auto source_camera_world =
                transformResolver.ResolveRawModelTransform(camera_node.get());
            modelTrans = camera->GetViewMatrix().inverse() * source_camera_world.inverse() *
                modelTrans;
        }

        modelTrans = ApplyMeshGeometryTransform(modelTrans, pNode->Mesh());

        if (info.has_NORMAL_MODEL_MATRIX) {
            // Wallpaper Engine's stock model vertex shaders normalize the transformed vertex
            // normal, but pass the transformed tangent and bitangent to the fragment shader
            // without normalizing either one. Consequently g_NormalModelMatrix is a direction
            // basis, not an arbitrary scaled inverse-transpose: retaining a uniform model scale of
            // 0.01 here produces tangent vectors of length 100, and a normal-map XY perturbation
            // then inflates N.L by the same factor in the PBR helpers. Compute the mathematically
            // correct inverse-transpose first, then remove only the per-axis magnitude while
            // preserving rotation, reflection, and the directional effect of non-uniform scale.
            Eigen::Matrix3d linear = modelTrans.topLeftCorner<3, 3>();
            if (std::abs(linear.determinant()) < 1e-18) {
                linear = Eigen::Matrix3d::Identity();
            } else {
                linear = linear.inverse().transpose().eval();
            }
            for (int column = 0; column < 3; ++column) {
                const double magnitude = linear.col(column).norm();
                if (magnitude > 1e-12) {
                    linear.col(column) /= magnitude;
                }
            }
            const Eigen::Matrix3f normal_matrix = linear.cast<float>();
            std::array<float, 12> packed {};
            for (int column = 0; column < 3; ++column) {
                packed[static_cast<size_t>(column) * 4 + 0] = normal_matrix(0, column);
                packed[static_cast<size_t>(column) * 4 + 1] = normal_matrix(1, column);
                packed[static_cast<size_t>(column) * 4 + 2] = normal_matrix(2, column);
            }
            updateOp(G_NORMAL_MODEL_MATRIX,
                     std::span<const float> { packed.data(), packed.size() });
        }

        if (reqM) updateOp(G_M, ToDxcCBufferMatrixUniform(modelTrans));
        if (reqAM) updateOp(G_AM, ToDxcCBufferMatrixUniform(modelTrans));
        if (reqLMM) updateOp(G_LMM, ToDxcCBufferMatrixUniform(modelTrans));
        if (reqMI) updateOp(G_MI, ToDxcCBufferMatrixUniform(modelTrans.inverse()));
        if (reqMVP || reqEMVP) {
            Matrix4d mvpTrans = viewProTrans * modelTrans;
            if (reqMVP) updateOp(G_MVP, ToDxcCBufferMatrixUniform(mvpTrans));
            if (reqEMVP) updateOp(G_EMVP, ToDxcCBufferMatrixUniform(mvpTrans));
            if (reqMVPI) updateOp(G_MVPI, ToDxcCBufferMatrixUniform(mvpTrans.inverse()));
        }
        if (reqETVP || reqETVPI) {
            const SceneNode* projectionNode      = pNode;
            const SceneMesh* projectionMesh      = pNode->Mesh();
            Matrix4d         projectionModelTrans = modelTrans;
            Matrix4d         projectionViewPro    = viewProTrans;

            const WPShaderValueData* nodeDataPtr = hasNodeData ? &m_nodeDataMap.at(pNode) : nullptr;
            if (nodeDataPtr != nullptr &&
                nodeDataPtr->effect_texture_projection.node != nullptr &&
                nodeDataPtr->effect_texture_projection.mesh != nullptr &&
                m_scene->activeCamera != nullptr) {
                projectionNode = nodeDataPtr->effect_texture_projection.node;
                projectionMesh = nodeDataPtr->effect_texture_projection.mesh;
                const_cast<SceneNode*>(projectionNode)->UpdateTrans();
                projectionModelTrans = ApplyMeshGeometryTransform(
                    projectionNode->ModelTrans(), projectionMesh);
                projectionViewPro    = m_scene->activeCamera->GetViewProjectionMatrix();
            }

            const auto etvpTrans = ComputeEffectTextureProjection(projectionNode,
                                                                  projectionMesh,
                                                                  projectionModelTrans,
                                                                  projectionViewPro);
            if (reqETVP) updateOp(G_ETVP, ToDxcCBufferMatrixUniform(etvpTrans));
            if (reqETVPI) {
                if (std::abs(etvpTrans.determinant()) > 1e-12) {
                    updateOp(G_ETVPI, ToDxcCBufferMatrixUniform(etvpTrans.inverse()));
                } else {
                    updateOp(G_ETVPI, ToDxcCBufferMatrixUniform(Matrix4d::Identity()));
                }
            }
        }
    }

    if (hasNodeData) {
        const auto& vol = m_nodeDataMap.at(pNode);
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

    if (info.has_EYE_POSITION || info.has_VIEWUP || info.has_VIEWRIGHT || info.has_VIEWFORWARD) {
        // InitUniforms restricts these updates to model materials and nodes explicitly routed
        // through the scene's 3D camera. Canvas-space image/effect particles keep their authored
        // constants while perspective particles receive one coherent projection and view basis.
        const auto eye = camera->GetPosition().cast<float>();
        Vector3f forward = camera->GetDirection().cast<float>();
        if (forward.norm() > 1e-6f) forward.normalize();
        Vector3f up = camera->GetUp().cast<float>();
        if (up.norm() > 1e-6f) up.normalize();
        Vector3f right = forward.cross(up);
        if (right.norm() > 1e-6f) right.normalize();

        if (info.has_EYE_POSITION)
            updateOp(G_EYE_POSITION, std::array<float, 3> { eye.x(), eye.y(), eye.z() });
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

    if (m_scene->scriptHost) {
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

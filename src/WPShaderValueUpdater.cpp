#include "WPShaderValueUpdater.hpp"
#include "WPNodeTransformResolver.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Transform.h"
#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "Audio/SoundManager.h"
#include "SpriteAnimation.hpp"
#include "SpecTexs.hpp"
#include "Core/ArrayHelper.hpp"
#include "Utils/Algorism.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <numeric>

using namespace wallpaper;
using namespace Eigen;

namespace
{
constexpr float kDefaultMouseCoord = 0.5f;
constexpr double kParallaxSettleRatio = std::log(100.0);
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

struct MeshBounds2D {
    bool     valid { false };
    Vector3d center { Vector3d::Zero() };
    Vector2d halfExtent { Vector2d::Ones() };
};

float SanitizeMouseCoord(double value) {
    if (! std::isfinite(value)) return kDefaultMouseCoord;
    return std::clamp(static_cast<float>(value), 0.0f, 1.0f);
}

MeshBounds2D ComputeMeshBounds2D(const SceneMesh* mesh) {
    if (mesh == nullptr || mesh->VertexCount() == 0) return {};

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

void WPShaderValueUpdater::FrameBegin() {
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
    /*
        using namespace std::chrono;
        auto nowTime = system_clock::to_time_t(system_clock::now());
        auto cTime   = std::localtime(&nowTime);
        m_dayTime =
            (((cTime->tm_hour * 60) + cTime->tm_min) * 60 + cTime->tm_sec) / (24.0f * 60.0f
       * 60.0f);
    */
    const std::array<float, 2> previousMousePos = m_mousePos;
    if (!(m_parallax.delay > 0.0f) || !std::isfinite(m_parallax.delay)) {
        m_mousePosLast     = previousMousePos;
        m_mousePos         = m_mousePosInput;
        return;
    }

    const double frameTime = std::max(m_scene->frameTime, 0.0);
    const double t =
        1.0 - std::exp(-(frameTime * kParallaxSettleRatio) / static_cast<double>(m_parallax.delay));
    m_mousePosLast     = previousMousePos;
    m_mousePos         = std::array { (float)algorism::lerp(t, m_mousePos[0], m_mousePosInput[0]),
                                      (float)algorism::lerp(t, m_mousePos[1], m_mousePosInput[1]) };
}

void WPShaderValueUpdater::FrameEnd() {}

void WPShaderValueUpdater::MouseInput(double x, double y) {
    m_mousePosInput[0] = SanitizeMouseCoord(x);
    m_mousePosInput[1] = SanitizeMouseCoord(y);
}

void WPShaderValueUpdater::InitUniforms(SceneNode* pNode, const ExistsUniformOp& existsOp) {
    m_nodeUniformInfoMap[pNode] = WPUniformInfo();
    auto& info                  = m_nodeUniformInfoMap[pNode];
    info.has_MI                 = existsOp(G_MI);
    info.has_M                  = existsOp(G_M);
    info.has_AM                 = existsOp(G_AM);
    info.has_MVP                = existsOp(G_MVP);
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
    info.has_EYE_POSITION     = IsModelRenderNode(pNode) && existsOp(G_EYE_POSITION);
    info.has_VIEWUP           = IsModelRenderNode(pNode) && existsOp(G_VIEWUP);
    info.has_VIEWRIGHT        = IsModelRenderNode(pNode) && existsOp(G_VIEWRIGHT);
    info.has_VIEWFORWARD      = IsModelRenderNode(pNode) && existsOp(G_VIEWFORWARD);
    for (size_t index = 0; index < kAudioSpectrumResolutions.size(); index++) {
        info.has_audio_spectrum_left[index] = existsOp(kAudioSpectrumLeftUniforms[index]);
        info.has_audio_spectrum_right[index] = existsOp(kAudioSpectrumRightUniforms[index]);
    }

    if (std::any_of(info.has_audio_spectrum_left.begin(),
                    info.has_audio_spectrum_left.end(),
                    [](bool value) { return value; }) ||
        std::any_of(info.has_audio_spectrum_right.begin(),
                    info.has_audio_spectrum_right.end(),
                    [](bool value) { return value; })) {
        LOG_INFO("SceneAudioUniformInit: layer=%d name='%s' has16=(%s,%s) has32=(%s,%s) has64=(%s,%s)",
                 pNode->ID(),
                 pNode->Name().c_str(),
                 info.has_audio_spectrum_left[0] ? "true" : "false",
                 info.has_audio_spectrum_right[0] ? "true" : "false",
                 info.has_audio_spectrum_left[1] ? "true" : "false",
                 info.has_audio_spectrum_right[1] ? "true" : "false",
                 info.has_audio_spectrum_left[2] ? "true" : "false",
                 info.has_audio_spectrum_right[2] ? "true" : "false");
    }

    std::accumulate(begin(info.texs), end(info.texs), 0, [&existsOp](uint index, auto& value) {
        value.has_resolution = existsOp(WE_GLTEX_RESOLUTION_NAMES[index]);
        value.has_mipmap     = existsOp(WE_GLTEX_MIPMAPINFO_NAMES[index]);
        return index + 1;
    });
}

void WPShaderValueUpdater::UpdateUniforms(SceneNode* pNode, sprite_map_t& sprites,
                                          const UpdateUniformOp& updateOp) {
    WPNodeTransformResolver transformResolver(*m_scene,
                                              m_parallax,
                                              m_nodeDataMap,
                                              m_modelTransformCache,
                                              m_parallaxOffsetCache,
                                              m_attachmentTransformCache,
                                              m_scene->activeCamera,
                                              m_mousePos,
                                              m_puppet_frame_serial);

    if (exists(m_nodeDataMap, pNode)) {
        auto& nodeData = m_nodeDataMap.at(pNode);
        transformResolver.UpdateAttachmentParentIfNeeded(nodeData);
        auto localTransform = transformResolver.ResolveAttachmentLocalTransform(pNode);
        if (localTransform.has_value()) {
            SceneImageEffectLayer* effectLayer { nullptr };
            const auto effective_camera = ResolveEffectiveNodeCameraName(pNode);
            if (!effective_camera.empty()) {
                auto camera_it = m_scene->cameras.find(effective_camera.data());
                if (camera_it != m_scene->cameras.end() && camera_it->second->HasImgEffect()) {
                    effectLayer = camera_it->second->GetImgEffect().get();
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

    const SceneCamera* camera;
    const auto         cam_name = ResolveEffectiveNodeCameraName(pNode);
    if (! cam_name.empty()) {
        camera = m_scene->cameras.at(cam_name.data()).get();
    } else
        camera = m_scene->activeCamera;

    if (! camera) return;

    if (! cam_name.empty()) {
        auto camera_it = m_scene->cameras.find(cam_name.data());
        if (camera_it != m_scene->cameras.end() && camera_it->second->HasImgEffect()) {
            auto* effectLayer = camera_it->second->GetImgEffect().get();
            auto* worldNode   = effectLayer->WorldNode();
            if (worldNode != nullptr && exists(m_nodeDataMap, worldNode)) {
                auto& worldNodeData = m_nodeDataMap.at(worldNode);
                transformResolver.UpdateAttachmentParentIfNeeded(worldNodeData);
                if (worldNodeData.InheritsSceneParentTransform() &&
                    ! worldNodeData.IsBoneAttached()) {
                    const SceneCamera* displayCamera =
                        m_scene->activeCamera != nullptr ? m_scene->activeCamera : camera;
                    const auto worldModel = transformResolver.ResolveParallaxedModelTransform(
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

            if (unifrom_tex.has_resolution) {
                // Runtime render targets expose one canonical resolution contract through
                // `ResolutionVector()`: physical size in `.xy`, logical content size in `.zw`.
                // Uniform updates should always forward that authoritative scene-side contract
                // directly instead of layering text-specific interpretation on top of it.
                std::array<i32, 4> resolution_uint(rt.ResolutionVector());
                updateOp(WE_GLTEX_RESOLUTION_NAMES[el.first],
                         ShaderValue(array_cast<float>(resolution_uint)));
            }
            if (unifrom_tex.has_mipmap) {
                updateOp(WE_GLTEX_MIPMAPINFO_NAMES[el.first], (float)rt.mipmap_level);
            }
        }
        if (nodeData.puppet_layer.hasPuppet() && info.has_BONES) {
            auto data = nodeData.puppet_layer.AdvanceIfNeeded(m_scene->frameTime, m_puppet_frame_serial);
            if (m_scene->scriptHost) {
                m_scene->scriptHost->NotifyAnimationLayersAdvanced(pNode);
            }
            updateOp(G_BONES, std::span<const float> { data[0].data(), data.size() * 16 });
        }
    }

    bool reqMI    = info.has_MI;
    bool reqM     = info.has_M;
    bool reqAM    = info.has_AM;
    bool reqMVP   = info.has_MVP;
    bool reqMVPI  = info.has_MVPI;
    bool reqETVP  = info.has_ETVP;
    bool reqETVPI = info.has_ETVPI;

    Matrix4d viewProTrans = camera->GetViewProjectionMatrix();

    if (info.has_VP) {
        updateOp(G_VP, ShaderValue::fromMatrix(viewProTrans));
    }
    if (reqM || reqMVP || reqMI || reqMVPI || reqETVP || reqETVPI) {
        Matrix4d modelTrans =
            transformResolver.ResolveParallaxedModelTransform(pNode, camera, cam_name != "effect");

        if (reqM) updateOp(G_M, ShaderValue::fromMatrix(modelTrans));
        if (reqAM) updateOp(G_AM, ShaderValue::fromMatrix(modelTrans));
        if (reqMI) updateOp(G_MI, ShaderValue::fromMatrix(modelTrans.inverse()));
        if (reqMVP) {
            Matrix4d mvpTrans = viewProTrans * modelTrans;
            updateOp(G_MVP, ShaderValue::fromMatrix(mvpTrans));
            if (reqMVPI) updateOp(G_MVPI, ShaderValue::fromMatrix(mvpTrans.inverse()));
        }
        if (reqETVP || reqETVPI) {
            const SceneNode* projectionNode      = pNode;
            const SceneMesh* projectionMesh      = pNode->Mesh();
            Matrix4d         projectionModelTrans = modelTrans;
            Matrix4d         projectionViewPro    = viewProTrans;

            const WPShaderValueData* nodeDataPtr = hasNodeData ? &m_nodeDataMap.at(pNode) : nullptr;
            if (nodeDataPtr != nullptr && nodeDataPtr->effect_projection_node != nullptr &&
                nodeDataPtr->effect_projection_mesh != nullptr && m_scene->activeCamera != nullptr) {
                projectionNode = nodeDataPtr->effect_projection_node;
                projectionMesh = nodeDataPtr->effect_projection_mesh;
                const_cast<SceneNode*>(projectionNode)->UpdateTrans();
                projectionModelTrans = projectionNode->ModelTrans();
                projectionViewPro    = m_scene->activeCamera->GetViewProjectionMatrix();
            }

            const auto etvpTrans = ComputeEffectTextureProjection(projectionNode,
                                                                  projectionMesh,
                                                                  projectionModelTrans,
                                                                  projectionViewPro);
            if (reqETVP) updateOp(G_ETVP, ShaderValue::fromMatrix(etvpTrans));
            if (reqETVPI) {
                if (std::abs(etvpTrans.determinant()) > 1e-12) {
                    updateOp(G_ETVPI, ShaderValue::fromMatrix(etvpTrans.inverse()));
                } else {
                    updateOp(G_ETVPI, ShaderValue::fromMatrix(Matrix4d::Identity()));
                }
            }
        }
    }

    //	g_EffectTextureProjectionMatrix
    // shadervs.push_back({"g_EffectTextureProjectionMatrixInverse",
    // ShaderValue::ValueOf(Eigen::Matrix4f::Identity())});
    if (info.has_TIME) updateOp(G_TIME, (float)m_scene->elapsingTime);

    if (info.has_DAYTIME) updateOp(G_DAYTIME, (float)m_dayTime);

    if (info.has_POINTERPOSITION) updateOp(G_POINTERPOSITION, m_mousePos);
    if (info.has_POINTERPOSITIONLAST) updateOp(G_POINTERPOSITIONLAST, m_mousePosLast);
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
        // These camera basis uniforms are gated by IsModelRenderNode() during InitUniforms. Updating
        // them here gives 3D model shaders coherent camera-path lighting/reflection data without
        // introducing a new uniform contract for unrelated 2D image/effect/particle shaders.
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
            const Vector2f mouseCentered = Vector2f(&m_mousePos[0]) - Vector2f { 0.5f, 0.5f };
            para = Vector2f { 0.5f, 0.5f } +
                   (Scaling(1.0f, -1.0f) * mouseCentered) * m_parallax.mouseinfluence;
        }
        updateOp(G_PARALLAXPOSITION, std::array { para[0], para[1] });
    }

    for (size_t index = 0; index < kAudioSpectrumResolutions.size(); index++) {
        if (!info.has_audio_spectrum_left[index] && !info.has_audio_spectrum_right[index]) continue;

        std::vector<float> left;
        std::vector<float> right;
        std::vector<float> average;
        bool has_audio = false;
        if (m_scene->scriptHost != nullptr) {
            has_audio = m_scene->scriptHost->GetAudioSpectrum(kAudioSpectrumResolutions[index],
                                                              &left,
                                                              &right,
                                                              &average);
        } else if (m_scene->soundManager != nullptr) {
            m_scene->soundManager->GetSpectrum(kAudioSpectrumResolutions[index], &left, &right, &average);
            has_audio = !left.empty() || !right.empty() || !average.empty();
        }
        if (!has_audio) {
            static std::unordered_map<int32_t, double> last_audio_missing_log_at;
            const auto layer_id = pNode->ID();
            const auto last_it = last_audio_missing_log_at.find(layer_id);
            if (last_it == last_audio_missing_log_at.end() ||
                m_scene->elapsingTime - last_it->second >= 5.0) {
                last_audio_missing_log_at[layer_id] = m_scene->elapsingTime;
                LOG_INFO("SceneAudioUniform: layer=%d name='%s' res=%u missing-audio-data",
                         layer_id,
                         pNode->Name().c_str(),
                         kAudioSpectrumResolutions[index]);
            }
            continue;
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
            l->node()->UpdateTrans();
            const auto modelTrans = l->node()->ModelTrans();
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

void WPShaderValueUpdater::SetTexelSize(float x, float y) { m_texelSize = { x, y }; }

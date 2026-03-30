#include "WPShaderValueUpdater.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Transform.h"
#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "SpriteAnimation.hpp"
#include "SpecTexs.hpp"
#include "Core/ArrayHelper.hpp"
#include "Utils/Algorism.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <numeric>

using namespace wallpaper;
using namespace Eigen;

namespace
{
void ApplyLocalAffine(SceneNode* node, const Affine3f& affine) {
    if (node == nullptr) return;

    Matrix3f linear = affine.linear();
    Vector3f scale(linear.col(0).norm(), linear.col(1).norm(), linear.col(2).norm());
    for (int i = 0; i < 3; ++i) {
        if (scale[i] > 1e-6f) {
            linear.col(i) /= scale[i];
        } else {
            linear.col(i).setZero();
            linear(i, i) = 1.0f;
            scale[i] = 1.0f;
        }
    }

    const auto zyx = linear.eulerAngles(2, 1, 0);
    node->SetScale(scale);
    node->SetRotation(Vector3f(zyx[2], zyx[1], zyx[0]));
    node->SetTranslate(affine.translation());
}

Vector3f ComputeParallaxOffset(SceneNode*                           node,
                               const WPShaderValueData&             nodeData,
                               const Scene&                         scene,
                               const WPCameraParallax&              parallax,
                               const wallpaper::Map<void*, WPShaderValueData>& nodeDataMap,
                               const SceneCamera*                   camera,
                               const std::array<float, 2>&          mousePos) {
    if (node == nullptr || camera == nullptr || !parallax.enable) return Vector3f::Zero();

    if (nodeData.scene_parent != nullptr && exists(nodeDataMap, nodeData.scene_parent)) {
        return ComputeParallaxOffset(nodeData.scene_parent,
                                     nodeDataMap.at(nodeData.scene_parent),
                                     scene,
                                     parallax,
                                     nodeDataMap,
                                     camera,
                                     mousePos);
    }

    node->UpdateTrans();
    const auto modelTrans = node->ModelTrans();
    Vector3f   nodePos((float)modelTrans(0, 3), (float)modelTrans(1, 3), (float)modelTrans(2, 3));
    Vector2f   depth(nodeData.parallaxDepth[0], nodeData.parallaxDepth[1]);

    Vector2f ortho { (float)scene.ortho[0], (float)scene.ortho[1] };
    Vector2f mouseVec =
        Scaling(1.0f, -1.0f) * (Vector2f { 0.5f, 0.5f } - Vector2f(&mousePos[0]));
    mouseVec = mouseVec.cwiseProduct(ortho) * parallax.mouseinfluence;

    Vector3f camPos = camera->GetPosition().cast<float>();
    Vector2f paraVec =
        (nodePos.head<2>() - camPos.head<2>() + mouseVec).cwiseProduct(depth) * parallax.amount;
    return Vector3f(paraVec.x(), paraVec.y(), 0.0f);
}

void ApplyParentParallaxToAttachment(SceneNode*                           parentNode,
                                     const WPShaderValueData&             parentData,
                                     const Scene&                         scene,
                                     const WPCameraParallax&              parallax,
                                     const wallpaper::Map<void*, WPShaderValueData>& nodeDataMap,
                                     const SceneCamera*                   camera,
                                     const std::array<float, 2>&          mousePos,
                                     Affine3f&                            localTransform) {
    if (parentNode == nullptr || camera == nullptr || !parallax.enable) return;

    parentNode->UpdateTrans();
    const auto parentParallax =
        ComputeParallaxOffset(parentNode, parentData, scene, parallax, nodeDataMap, camera, mousePos);
    const auto parentModel  = parentNode->ModelTrans();
    Matrix3f    parentLinear = parentModel.block<3, 3>(0, 0).cast<float>();
    Vector3f    parentParallaxLocal = parentParallax;
    if (std::abs(parentLinear.determinant()) > 1e-6f) {
        parentParallaxLocal = parentLinear.inverse() * parentParallax;
    }
    localTransform.translation() += parentParallaxLocal;
}
} // namespace

void WPShaderValueUpdater::FrameBegin() {
    m_puppet_frame_serial++;
    /*
        using namespace std::chrono;
        auto nowTime = system_clock::to_time_t(system_clock::now());
        auto cTime   = std::localtime(&nowTime);
        m_dayTime =
            (((cTime->tm_hour * 60) + cTime->tm_min) * 60 + cTime->tm_sec) / (24.0f * 60.0f
       * 60.0f);
    */
    double new_time    = m_mouseDelayedTime + m_scene->frameTime;
    new_time           = new_time > m_parallax.delay ? m_parallax.delay : new_time;
    m_mouseDelayedTime = new_time;
    double t           = new_time / m_parallax.delay;
    m_mousePos         = std::array { (float)algorism::lerp(t, m_mousePos[0], m_mousePosInput[0]),
                              (float)algorism::lerp(t, m_mousePos[1], m_mousePosInput[1]) };
}

void WPShaderValueUpdater::FrameEnd() {}

void WPShaderValueUpdater::MouseInput(double x, double y) {
    using namespace std::chrono;

    auto   now_time = steady_clock::now();
    double new_time = m_mouseDelayedTime -
                      duration_cast<duration<double>>(now_time - m_last_mouse_input_time).count();
    m_mouseDelayedTime = new_time < 0.0f ? 0.0f : new_time;

    m_mousePosInput[0] = (float)x;
    m_mousePosInput[1] = (float)y;

    m_last_mouse_input_time = now_time;
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
    info.has_POINTERPOSITION  = existsOp(G_POINTERPOSITION);
    info.has_PARALLAXPOSITION = existsOp(G_PARALLAXPOSITION);
    info.has_TEXELSIZE        = existsOp(G_TEXELSIZE);
    info.has_TEXELSIZEHALF    = existsOp(G_TEXELSIZEHALF);
    info.has_SCREEN           = existsOp(G_SCREEN);
    info.has_LP               = existsOp(G_LP);

    std::accumulate(begin(info.texs), end(info.texs), 0, [&existsOp](uint index, auto& value) {
        value.has_resolution = existsOp(WE_GLTEX_RESOLUTION_NAMES[index]);
        value.has_mipmap     = existsOp(WE_GLTEX_MIPMAPINFO_NAMES[index]);
        return index + 1;
    });
}

void WPShaderValueUpdater::UpdateUniforms(SceneNode* pNode, sprite_map_t& sprites,
                                          const UpdateUniformOp& updateOp) {
    if (exists(m_nodeDataMap, pNode)) {
        auto& nodeData = m_nodeDataMap.at(pNode);
        if (nodeData.attach_to_bone && nodeData.scene_parent != nullptr &&
            exists(m_nodeDataMap, nodeData.scene_parent)) {
            auto& parentData = m_nodeDataMap.at(nodeData.scene_parent);
            if (parentData.puppet_layer.hasPuppet()) {
                parentData.puppet_layer.AdvanceIfNeeded(m_scene->frameTime, m_puppet_frame_serial);
                const auto* parentPuppet = parentData.puppet_layer.Puppet();
                if (parentPuppet != nullptr) {
                    const auto& boneTransform = parentPuppet->BoneModelTransform(nodeData.attach_bone);
                    Affine3f    localTransform =
                        boneTransform * nodeData.attach_transform * nodeData.attach_local_transform;
                    ApplyParentParallaxToAttachment(nodeData.scene_parent,
                                                    parentData,
                                                    *m_scene,
                                                    m_parallax,
                                                    m_nodeDataMap,
                                                    m_scene->activeCamera,
                                                    m_mousePos,
                                                    localTransform);

                    SceneImageEffectLayer* effectLayer { nullptr };
                    if (!pNode->Camera().empty()) {
                        auto camera_it = m_scene->cameras.find(pNode->Camera());
                        if (camera_it != m_scene->cameras.end() && camera_it->second->HasImgEffect()) {
                            effectLayer = camera_it->second->GetImgEffect().get();
                        }
                    }

                    if (effectLayer != nullptr) {
                        if (auto* worldNode = effectLayer->WorldNode()) {
                            ApplyLocalAffine(worldNode, localTransform);
                            worldNode->UpdateTrans();
                            effectLayer->SyncResolvedNodeToWorld();
                        }
                    } else {
                        ApplyLocalAffine(pNode, localTransform);
                    }
                }
            }
        }
    }

    if (! pNode->Mesh()) return;

    pNode->UpdateTrans();

    const SceneCamera* camera;
    std::string_view   cam_name = pNode->Camera();
    if (! pNode->Camera().empty()) {
        camera = m_scene->cameras.at(cam_name.data()).get();
    } else
        camera = m_scene->activeCamera;

    if (! camera) return;

    auto* material = pNode->Mesh()->Material();
    if (! material) return;
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
                std::array<i32, 4> resolution_uint({ rt.width, rt.height, rt.width, rt.height });
                updateOp(WE_GLTEX_RESOLUTION_NAMES[el.first],
                         ShaderValue(array_cast<float>(resolution_uint)));
            }
            if (unifrom_tex.has_mipmap) {
                updateOp(WE_GLTEX_MIPMAPINFO_NAMES[el.first], (float)rt.mipmap_level);
            }
        }
        if (nodeData.puppet_layer.hasPuppet() && info.has_BONES) {
            auto data = nodeData.puppet_layer.AdvanceIfNeeded(m_scene->frameTime, m_puppet_frame_serial);
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
    if (reqM || reqMVP || reqMI || reqMVPI) {
        Matrix4d modelTrans = pNode->ModelTrans();
        if (hasNodeData && cam_name != "effect") {
            const auto& nodeData = m_nodeDataMap.at(pNode);
            if (m_parallax.enable && !nodeData.skip_model_parallax) {
                const auto parallaxOffset =
                    ComputeParallaxOffset(pNode,
                                          nodeData,
                                          *m_scene,
                                          m_parallax,
                                          m_nodeDataMap,
                                          camera,
                                          m_mousePos);
                const Vector3d parallaxOffsetD = parallaxOffset.cast<double>();
                modelTrans =
                    Affine3d(Eigen::Translation3d(parallaxOffsetD)).matrix() * modelTrans;
            }
        }

        if (reqM) updateOp(G_M, ShaderValue::fromMatrix(modelTrans));
        if (reqAM) updateOp(G_AM, ShaderValue::fromMatrix(modelTrans));
        if (reqMI) updateOp(G_MI, ShaderValue::fromMatrix(modelTrans.inverse()));
        if (reqMVP) {
            Matrix4d mvpTrans = viewProTrans * modelTrans;
            updateOp(G_MVP, ShaderValue::fromMatrix(mvpTrans));
            if (reqMVPI) updateOp(G_MVPI, ShaderValue::fromMatrix(mvpTrans.inverse()));
        }
        if (reqETVP || reqETVPI) {
            /*
            Vector3d nodePos = pNode->Translate().cast<double>();
            nodePos.z()      = 1.0f;
            Matrix4d etvpTrans =
                viewProTrans * modelTrans * Affine3d(Eigen::Scaling(nodePos)).matrix();
            if (reqETVPI) updateOp(G_ETVP, ShaderValue::fromMatrix(etvpTrans));
            if (reqETVPI) updateOp(G_ETVPI, ShaderValue::fromMatrix(etvpTrans.inverse()));
            */
        }
    }

    //	g_EffectTextureProjectionMatrix
    // shadervs.push_back({"g_EffectTextureProjectionMatrixInverse",
    // ShaderValue::ValueOf(Eigen::Matrix4f::Identity())});
    if (info.has_TIME) updateOp(G_TIME, (float)m_scene->elapsingTime);

    if (info.has_DAYTIME) updateOp(G_DAYTIME, (float)m_dayTime);

    if (info.has_POINTERPOSITION) updateOp(G_POINTERPOSITION, m_mousePos);

    if (info.has_TEXELSIZE) updateOp(G_TEXELSIZE, m_texelSize);

    if (info.has_TEXELSIZEHALF)
        updateOp(G_TEXELSIZEHALF, std::array { m_texelSize[0] / 2.0f, m_texelSize[1] / 2.0f });

    if (info.has_SCREEN)
        updateOp(G_SCREEN,
                 std::array<float, 3> {
                     m_screen_size[0], m_screen_size[1], m_screen_size[0] / m_screen_size[1] });

    if (info.has_PARALLAXPOSITION) {
        Vector2f para =
            Vector2f { 0.5f, 0.5f } +
            (Scaling(1.0f, -1.0f) * (Vector2f(&m_mousePos[0])) - Vector2f { 0.5f, 0.5f }) *
                m_parallax.mouseinfluence;
        updateOp(G_PARALLAXPOSITION, std::array { para[0], para[1] });
    }

    for (auto& [i, sp] : sprites) {
        const auto& f      = sp.GetAnimateFrame(m_scene->frameTime);
        auto        grot   = WE_GLTEX_ROTATION_NAMES[i];
        auto        gtrans = WE_GLTEX_TRANSLATION_NAMES[i];
        updateOp(grot, std::array { f.xAxis[0], f.xAxis[1], f.yAxis[0], f.yAxis[1] });
        updateOp(gtrans, std::array { f.x, f.y });
    }

    if (info.has_LP) {
        std::array<float, 16> lights { 0 };
        std::array<float, 12> lights_color { 0 };
        uint                  i = 0;
        for (auto& l : m_scene->lights) {
            if (i == 4) break;
            assert(l->node() != nullptr);
            l->node()->UpdateTrans();
            const auto modelTrans = l->node()->ModelTrans();
            lights[i * 4 + 0]     = (float)modelTrans(0, 3);
            lights[i * 4 + 1]     = (float)modelTrans(1, 3);
            lights[i * 4 + 2]     = (float)modelTrans(2, 3);
            if (i < 3) {
                const auto& color = l->premultipliedColor();
                std::copy(color.begin(), color.end(), lights_color.begin() + i * 4);
            }
            i++;
        }
        updateOp(G_LP, lights);
        updateOp(G_LCP, lights_color);
    }
}

void WPShaderValueUpdater::SetNodeData(void* nodeAddr, const WPShaderValueData& data) {
    m_nodeDataMap[nodeAddr] = data;
}

void WPShaderValueUpdater::SetTexelSize(float x, float y) { m_texelSize = { x, y }; }

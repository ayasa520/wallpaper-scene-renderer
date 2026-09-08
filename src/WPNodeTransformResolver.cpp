#include "WPNodeTransformResolver.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "WPImageAlignment.hpp"

#include <Eigen/Geometry>

using namespace wallpaper;
using namespace Eigen;

WPNodeTransformResolver::WPNodeTransformResolver(
    Scene& scene, const WPCameraParallax& parallax,
    Map<void*, WPShaderValueData>& node_data_map,
    Map<void*, Matrix4d>& model_transform_cache,
    Map<void*, Vector3f>& parallax_offset_cache,
    const std::array<float, 2>& mouse_pos, uint64_t puppet_frame_serial)
    : m_scene(scene),
      m_parallax(parallax),
      m_node_data_map(node_data_map),
      m_model_transform_cache(model_transform_cache),
      m_parallax_offset_cache(parallax_offset_cache),
      m_mouse_pos(mouse_pos),
      m_puppet_frame_serial(puppet_frame_serial) {}

Matrix4d WPNodeTransformResolver::ResolveParallaxedModelTransform(const SceneDraw& draw,
                                                                const SceneCamera* camera,
                                                                bool apply_parallax) {
    const auto* data = FindNodeData(draw);
    Matrix4d model = ResolveModelTransform(draw, data);
    if (data != nullptr && apply_parallax && data->AppliesModelParallax()) {
        const auto offset = ComputeParallaxOffset(draw, *data, camera);
        model = Affine3d(Translation3d(offset.cast<double>())).matrix() * model;
    }
    return model;
}

Matrix4d WPNodeTransformResolver::ResolveRawModelTransform(const SceneDraw& draw) {
    return ResolveModelTransform(draw, FindNodeData(draw));
}

Vector3f WPNodeTransformResolver::ResolveParallaxOffset(const SceneDraw& draw,
                                                       const SceneCamera* camera) {
    const auto* data = FindNodeData(draw);
    return data != nullptr ? ComputeParallaxOffset(draw, *data, camera) : Vector3f::Zero();
}

const WPShaderValueData* WPNodeTransformResolver::FindNodeData(const SceneDraw& draw) const {
    const auto it = m_node_data_map.find(draw.DataKey());
    return it == m_node_data_map.end() ? nullptr : std::addressof(it->second);
}

Matrix4d WPNodeTransformResolver::ResolveObjectTransform(const SceneObject& object) {
    // Authored TRS is input to matrix evaluation, never its output. In particular, a live bone
    // must not decompose its result back into the owner's origin/angles/scale or freeze a copy of
    // those properties at attachment time. Repeated script/projection/draw queries read the same
    // authored record.
    auto* key = const_cast<SceneObject*>(&object);
    if (const auto it = m_model_transform_cache.find(key); it != m_model_transform_cache.end()) {
        return it->second;
    }
    const auto& transform = object.RuntimeTransform();
    if (transform == nullptr) return Matrix4d::Identity();
    Matrix4d local = transform->LocalMatrix();
    const auto* parent = m_scene.FindSceneObject(object.ParentId());
    Matrix4d model = local;
    if (parent != nullptr && parent->RuntimeTransform() != nullptr) {
        const auto* data = FindNodeData(object.LayerNode());
        if (data != nullptr && data->IsBoneAttached()) {
            auto* parent_data = m_node_data_map.contains(parent->LayerNode())
                ? &m_node_data_map.at(parent->LayerNode()) : nullptr;
            if (parent_data != nullptr && parent_data->puppet_layer.hasPuppet()) {
                parent_data->puppet_layer.AdvanceIfNeeded(m_scene.frameTime, m_puppet_frame_serial);
                const auto* puppet = parent_data->puppet_layer.Puppet();
                const auto attachment =
                    puppet->BoneModelTransform(data->transform_binding.bone_index) *
                    data->transform_binding.bind_transform;
                local = attachment.matrix().cast<double>() * local;
            }
        }
        // Card alignment is a local drawing offset, not the pivot inherited by another object.
        // The parent relation is authored identity data; physical render-order proxies and
        // detached source resources cannot change this multiplication chain.
        const Matrix4d parent_model = RemoveImageAlignmentOffsetFromModel(
            ResolveObjectTransform(*parent), parent->RuntimeTransform()->AlignmentOffset());
        model = parent_model * local;
    }
    m_model_transform_cache[key] = model;
    return model;
}

Matrix4d WPNodeTransformResolver::ResolveModelTransform(const SceneDraw& draw,
                                                       const WPShaderValueData* data) {
    if (const auto* phase = draw.Phase(); phase != nullptr) {
        return phase->PlacementSpace() == SceneDrawPhase::Space::Camera
            ? Matrix4d(Matrix4d::Identity()) : ResolveObjectTransform(phase->Owner());
    }
    auto* node = draw.Node();
    if (node == nullptr) return Matrix4d::Identity();
    if (auto* layer = data != nullptr ? data->effect_layer_projection.layer : nullptr;
        layer != nullptr && layer->UsesOwnerTransform(node)) {
        return ResolveObjectTransform(layer->Owner());
    }
    if (const auto* owner = m_scene.FindSceneObject(draw.LayerId(m_scene));
        owner != nullptr && owner->LayerNode() == node) {
        return ResolveObjectTransform(*owner);
    }
    if (const auto it = m_model_transform_cache.find(node); it != m_model_transform_cache.end()) {
        return it->second;
    }

    // Internal model/particle resources still carry local geometry transforms. Their parent can
    // be an authored object, so resolve it through the same query instead of depending on a
    // SceneNode cache that a previous attachment draw happened to mutate.
    Matrix4d model = node->GetLocalTrans();
    if (data != nullptr && data->InheritsSceneParentTransform() &&
        data->TransformParent() != nullptr) {
        auto* parent = data->TransformParent();
        model = RemoveImageAlignmentOffsetFromModel(
                    ResolveRawModelTransform(parent), parent->AlignmentOffset()) * model;
    } else if (auto* parent = node->Parent(); parent != nullptr) {
        model = ResolveRawModelTransform(parent) * model;
    }
    m_model_transform_cache[node] = model;
    return model;
}

const SceneObject* WPNodeTransformResolver::ParallaxRoot(const SceneDraw& draw) const {
    const auto* root = m_scene.FindSceneObject(draw.LayerId(m_scene));
    if (root == nullptr) return nullptr;
    while (const auto* parent = m_scene.FindSceneObject(root->ParentId())) {
        root = parent;
    }
    return root;
}

Vector3f WPNodeTransformResolver::ComputeParallaxOffset(const SceneDraw& draw,
                                                       const WPShaderValueData& data,
                                                       const SceneCamera* camera) {
    if (!draw.Valid() || camera == nullptr || !m_parallax.enable) return Vector3f::Zero();
    const auto key = draw.DataKey();
    if (const auto it = m_parallax_offset_cache.find(key); it != m_parallax_offset_cache.end()) {
        return it->second;
    }

    // Destination displacement uses the authored origin and depth of the object's root. Every
    // descendant receives the same displacement, including nested bone attachments. Neither a
    // transformed child position nor a name-matched peer supplies this input, and it never enters
    // raw M.
    Vector2f position;
    Vector2f depth(data.parallaxDepth.data());
    if (const auto* root = ParallaxRoot(draw);
        root != nullptr && root->RuntimeTransform() != nullptr) {
        position = root->RuntimeTransform()->Translate().head<2>();
        if (const auto* root_data = FindNodeData(root->LayerNode()); root_data != nullptr) {
            depth = Vector2f(root_data->parallaxDepth.data());
        }
    } else {
        position = ResolveModelTransform(draw, &data).block<2, 1>(0, 3).cast<float>();
    }

    const Vector2f ortho((float)m_scene.ortho[0], (float)m_scene.ortho[1]);
    Vector2f mouse = Scaling(1.0f, -1.0f) *
        (Vector2f(0.5f, 0.5f) - Vector2f(m_mouse_pos.data()));
    mouse = mouse.cwiseProduct(ortho) * m_parallax.mouseinfluence;
    const Vector2f offset =
        (position - camera->GetPosition().head<2>().cast<float>() + mouse)
            .cwiseProduct(depth) * m_parallax.amount;
    const Vector3f result(offset.x(), offset.y(), 0.0f);
    m_parallax_offset_cache[key] = result;
    return result;
}

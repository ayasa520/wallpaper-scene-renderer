#pragma once
#include <cstdint>
#include <list>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include "SceneMesh.h"
#include "SceneCamera.h"
#include "SceneTransform.h"
#include "SceneDraw.h"

#include "Core/Literals.hpp"
#include "Core/NoCopyMove.hpp"

namespace wallpaper
{

class SceneTextPrimitive;

class SceneNode : NoCopy, NoMove {
public:
    SceneNode() : m_transform(std::make_shared<SceneTransform>()) {}
    SceneNode(const Eigen::Vector3f& translate, const Eigen::Vector3f& scale,
              const Eigen::Vector3f& rotation, const std::string& name = "")
        : m_name(name),
          m_transform(std::make_shared<SceneTransform>(translate, scale, rotation)) {};

    const auto& Camera() const { return m_cameraName; }
    void        SetCamera(const std::string& name) { m_cameraName = name; }
    const auto& Name() const { return m_name; }
    void        SetName(std::string name) { m_name = std::move(name); }
    void        AddMesh(std::shared_ptr<SceneMesh> mesh) { m_mesh = mesh; }
    // Text is now a first-class scene renderable. The node keeps an optional text primitive next
    // to the mesh slot so the render graph can emit a dedicated text pass directly.
    void        AddText(std::shared_ptr<SceneTextPrimitive> text) { m_text = std::move(text); }
    void        AppendChild(std::shared_ptr<SceneNode> sub) {
               sub->m_parent = this;
               // Reparenting invalidates this model. Its next revision propagates through child
               // queries, including when no local transform value changed anywhere in the branch.
               sub->m_dirty = true;
               m_children.push_back(sub);
    }
    bool        RemoveChild(SceneNode* child) {
               for (auto it = m_children.begin(); it != m_children.end(); ++it) {
                   if (it->get() == child) {
                       auto removed = *it;
                       removed->m_parent = nullptr;
                       removed->m_dirty = true;
                       m_children.erase(it);
                       return true;
                   }
               }
               return false;
    }
    Eigen::Matrix4d GetLocalTrans() const { return m_transform->LocalMatrix(); }
    const std::shared_ptr<SceneTransform>& TransformState() const { return m_transform; }

    const auto& Translate() const { return m_transform->Translate(); }
    const auto& Rotation() const { return m_transform->Rotation(); }
    const auto& Scale() const { return m_transform->Scale(); }
    const auto& AlignmentOffset() const { return m_transform->AlignmentOffset(); }
    void SetRotation(Eigen::Vector3f value) { m_transform->SetRotation(value); }
    void SetScale(Eigen::Vector3f value) { m_transform->SetScale(value); }
    void SetTranslate(Eigen::Vector3f value) { m_transform->SetTranslate(value); }
    void SetAlignmentOffset(Eigen::Vector3f value) { m_transform->SetAlignmentOffset(value); }
    void SetLocalAffine(const Eigen::Affine3f& affine) { m_transform->SetLocalAffine(affine); }
    void CopyTrans(const SceneNode& node) { m_transform->CopyFrom(*node.m_transform); }

    bool Visible() const noexcept {
        // Effective visibility now combines the node-local flag with the layer-level flag that the
        // scene graph propagates. Keeping those concerns separate lets runtime-only support nodes,
        // such as effect internals, preserve their own local visibility contract without fighting
        // the generic layer visibility system every time a script or parent layer updates a layer.
        return m_localVisible && m_layerVisible &&
               (m_parent == nullptr || m_parent->Visible());
    }
    bool LocalVisible() const noexcept { return m_localVisible; }
    bool LayerVisible() const noexcept { return m_layerVisible; }
    void SetVisible(bool value) noexcept { SetLocalVisible(value); }
    void SetLocalVisible(bool value) noexcept { m_localVisible = value; }
    void SetLayerVisible(bool value) noexcept { m_layerVisible = value; }
    bool CastsShadows() const noexcept { return m_castsShadows; }
    void SetCastsShadows(bool value) noexcept { m_castsShadows = value; }

    // update self modle trans (will update parent before)
    void            UpdateTrans();
    Eigen::Matrix4d ModelTrans() const { return m_trans; };

    SceneMesh* Mesh() { return m_mesh.get(); }
    const SceneMesh* Mesh() const { return m_mesh.get(); }
    // Callers use the text slot to decide whether a node should go through the dedicated text
    // pipeline or the traditional mesh/custom-shader pipeline. Returning raw pointers here keeps
    // the public API aligned with `Mesh()` while ownership still lives in `m_text`.
    SceneTextPrimitive* Text() { return m_text.get(); }
    const SceneTextPrimitive* Text() const { return m_text.get(); }
    bool       HasText() const { return m_text != nullptr; }
    bool       HasMaterial() const { return m_mesh && m_mesh->Material() != nullptr; };

    const auto& GetChildren() const { return m_children; }
    auto&       GetChildren() { return m_children; }
    SceneNode*  Parent() const noexcept { return m_parent; }

    i32& ID() { return m_id; }
    i32  ID() const { return m_id; }

    // Render-graph residency can outlive the SceneNode object that originally described a pass:
    // topology rebuilds replace synthetic effect nodes before the old Vulkan pass is retired. A
    // raw address is therefore not an identity -- the allocator may hand the same address to the
    // next node and make a newly compiled shader pass look reusable. Keep a process-local monotonic
    // identity on each node so residency matching remains valid across destruction and allocation.
    uint64_t RenderIdentity() const noexcept { return m_render_identity; }

private:
    // 0 means "no authored layer": detached helper nodes keep the default so layer-id resolution
    // deterministically treats them as unowned instead of reading uninitialized memory.
    i32         m_id { 0 };
    uint64_t    m_render_identity { AllocateSceneDrawIdentity() };
    std::string m_name;

    std::shared_ptr<SceneTransform> m_transform;
    bool            m_dirty { true };
    Eigen::Matrix4d m_trans { Eigen::Matrix4d::Identity() };
    uint64_t m_local_revision { 0 };
    uint64_t m_parent_revision { 0 };
    uint64_t m_model_revision { 0 };

    std::shared_ptr<SceneMesh> m_mesh;
    // The text primitive holds canonical text geometry, atlas resources, and optional bridge
    // metadata. Storing it directly on the node is what makes text a scene-native renderable.
    std::shared_ptr<SceneTextPrimitive> m_text;
    bool m_localVisible { true };
    bool m_layerVisible { true };
    bool m_castsShadows { false };

    // specific a camera not active, used for image effect
    std::string m_cameraName;

    SceneNode* m_parent { nullptr };

    std::list<std::shared_ptr<SceneNode>> m_children;
};
} // namespace wallpaper

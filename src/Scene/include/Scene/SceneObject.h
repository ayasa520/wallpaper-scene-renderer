#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "Core/NoCopyMove.hpp"

namespace wallpaper
{

// Authored object kind straight from scene.json. One entry of objects[] maps to exactly one
// SceneObject; render-time SceneNodes are draw handles owned by passes, not layer identities.
enum class SceneObjectKind
{
    Empty,
    Image,
    Particle,
    Text,
    Light,
    Sound,
    Camera,
    Model,
    Shape,
};

// A hidden authored layer may stay a lightweight logical placeholder until it is first needed.
// The deferred kind records which materializer owns the layer; None means fully materialized.
// A layer is deferred for at most one kind (its authored kind), so a single field carries the
// same information as the former per-kind Scene sets.
enum class SceneDeferredRuntimeKind
{
    None,
    Image,
    Particle,
    Text,
};

// Image layers keep a small always-available runtime state (authored quad size plus alignment)
// that scripts read and write independently from any render node. It lives on the SceneObject so
// the object is the single source for layer-level state.
struct SceneImageLayerRuntimeState {
    std::array<float, 2> size { 0.0f, 0.0f };
    std::string          alignment { "center" };
};

// One scene.json object. This is the only authored layer identity: id, name, kind, authored
// transform, local visibility, parent binding, and image-layer runtime state all live here.
// Draw phases (source draw, effect passes, final composite) reference this object through its id
// instead of duplicating the identity onto multiple SceneNodes.
//
// Behavior contract: the fields that drive runtime behavior (local_visible, parent_id/attachment,
// image runtime state) keep exactly the semantics of the former Scene-level maps
// (layerLocalVisibility, layerParentBindings, imageLayers). Authored identity fields (kind, name,
// origin/scale/angles, effect_count, passthrough) are snapshots of the parsed scene.json values.
class SceneObject : NoCopy, NoMove {
public:
    explicit SceneObject(int32_t id) : m_id(id) {}

    int32_t Id() const { return m_id; }

    SceneObjectKind Kind() const { return m_kind; }
    void            SetKind(SceneObjectKind kind) { m_kind = kind; }

    const std::string& Name() const { return m_name; }
    void               SetName(std::string name) { m_name = std::move(name); }

    const std::array<float, 3>& Origin() const { return m_origin; }
    const std::array<float, 3>& Scale() const { return m_scale; }
    const std::array<float, 3>& Angles() const { return m_angles; }
    void SetAuthoredTransform(const std::array<float, 3>& origin, const std::array<float, 3>& scale,
                              const std::array<float, 3>& angles) {
        m_origin = origin;
        m_scale  = scale;
        m_angles = angles;
    }

    // Layer-local visibility. Same default as the former layerLocalVisibility map: a layer that
    // was never explicitly set is visible.
    bool LocalVisible() const { return m_local_visible; }
    void SetLocalVisible(bool visible) { m_local_visible = visible; }

    // Authored parent binding by layer id. parent_id 0 means "no parent". Ids are stored instead
    // of object pointers so a deleted parent leaves exactly the same dangling-id semantics the
    // former layerParentBindings map had.
    int32_t            ParentId() const { return m_parent_id; }
    const std::string& Attachment() const { return m_attachment; }
    void               SetParentBinding(int32_t parent_id, std::string attachment) {
        m_parent_id  = parent_id;
        m_attachment = std::move(attachment);
    }
    void ClearParentBinding() {
        m_parent_id = 0;
        m_attachment.clear();
    }

    // Authored effect count from scene.json (before parse-time pruning).
    int32_t EffectCount() const { return m_effect_count; }
    void    SetEffectCount(int32_t count) { m_effect_count = count; }

    // Passthrough compose helpers publish nothing while their final effect is hidden. The flag is
    // authored per object, so it belongs here instead of being re-read from parse-time JSON.
    bool Passthrough() const { return m_passthrough; }
    void SetPassthrough(bool passthrough) { m_passthrough = passthrough; }

    SceneDeferredRuntimeKind DeferredRuntimeKind() const { return m_deferred_runtime_kind; }
    void SetDeferredRuntimeKind(SceneDeferredRuntimeKind kind) { m_deferred_runtime_kind = kind; }

    // Image runtime state exists only for materialized image layers (concrete or logical); other
    // kinds return nullptr, which is what tells scripts "this is not an image layer".
    SceneImageLayerRuntimeState*       ImageRuntimeState() {
        return m_has_image_runtime_state ? &m_image_runtime_state : nullptr;
    }
    const SceneImageLayerRuntimeState* ImageRuntimeState() const {
        return m_has_image_runtime_state ? &m_image_runtime_state : nullptr;
    }
    void SetImageRuntimeState(SceneImageLayerRuntimeState state) {
        m_image_runtime_state     = std::move(state);
        m_has_image_runtime_state = true;
    }
    void ClearImageRuntimeState() {
        m_image_runtime_state     = SceneImageLayerRuntimeState {};
        m_has_image_runtime_state = false;
    }

private:
    int32_t         m_id { 0 };
    SceneObjectKind m_kind { SceneObjectKind::Empty };
    std::string     m_name;

    std::array<float, 3> m_origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> m_scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> m_angles { 0.0f, 0.0f, 0.0f };

    bool        m_local_visible { true };
    int32_t     m_parent_id { 0 };
    std::string m_attachment;

    int32_t m_effect_count { 0 };
    bool    m_passthrough { false };

    SceneDeferredRuntimeKind m_deferred_runtime_kind { SceneDeferredRuntimeKind::None };

    bool                        m_has_image_runtime_state { false };
    SceneImageLayerRuntimeState m_image_runtime_state;
};

} // namespace wallpaper

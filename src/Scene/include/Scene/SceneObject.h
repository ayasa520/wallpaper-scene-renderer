#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Core/NoCopyMove.hpp"

namespace wallpaper
{

class SceneNode;
class SceneImageEffectLayer;
class SceneLight;
class ParticleSubSystem;
struct TextLayerRuntimeState;

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

// Wallpaper Engine camera layers are represented in scene.json as transform-only objects with
// camera-specific properties. Keep the authored values beside the render node so scripts and
// keyframe animations can round-trip the WE-facing origin/zoom values while Hanabi stores the
// attached SceneCamera node in renderer coordinates. Like the image runtime state, the record
// lives on the owning SceneObject; the scene-wide camera-layer precedence order stays on Scene.
struct SceneCameraLayerRuntimeState {
    std::string                camera_name { "global" };
    std::shared_ptr<SceneNode> node;
    std::array<float, 3>       origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3>       angles { 0.0f, 0.0f, 0.0f };
    double                     zoom { 1.0 };
    float                      fov { 50.0f };
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
    // Out-of-line (Scene.cpp): the text runtime state is held through a pointer to a parser-side
    // type this header only forward-declares.
    ~SceneObject();

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

    // Sound layers mount a SoundManager stream; the handle is that layer's runtime resource.
    // nullopt means no stream was mounted. A stored 0 is kept as a value on purpose: it is what a
    // failed mount recorded in the former Scene map, and presence checks must keep matching it.
    std::optional<uint32_t> SoundHandle() const { return m_sound_handle; }
    void                    SetSoundHandle(uint32_t handle) { m_sound_handle = handle; }
    void                    ClearSoundHandle() { m_sound_handle.reset(); }

    // The layer's script-visible handle node. The slot is tri-state to preserve the former
    // Scene::layerNodes semantics: no slot means the layer is not registered, a slot holding
    // nullptr means the layer is registered but its handle is temporarily absent
    // (mid-rematerialization, or node-less sound layers), and otherwise the slot is the live
    // handle. Registration checks must use HasLayerNodeSlot, not the node value.
    bool       HasLayerNodeSlot() const { return m_has_layer_node_slot; }
    SceneNode* LayerNode() const { return m_layer_node; }
    void       SetLayerNode(SceneNode* node) {
        m_layer_node          = node;
        m_has_layer_node_slot = true;
    }
    void ClearLayerNodeSlot() {
        m_layer_node          = nullptr;
        m_has_layer_node_slot = false;
    }

    // The layer's image-effect bridge. The SceneObject owns it: the bridge is a per-layer
    // runtime resource exactly like the sound handle, and it lives as long as the authored
    // identity. Cameras and render targets the bridge materialized stay plain named resources
    // in the Scene pools; nothing else holds an owning reference.
    const std::shared_ptr<SceneImageEffectLayer>& ImageEffectLayer() const {
        return m_image_effect_layer;
    }
    void SetImageEffectLayer(std::shared_ptr<SceneImageEffectLayer> effect_layer) {
        m_image_effect_layer = std::move(effect_layer);
    }

    // Runtime resources this layer mounted into the Scene-level pools: scene lights
    // (Scene::lights) and particle subsystems (ParticleSystem::subsystems). The object holds the
    // back-references so property reads/writes and layer destroy resolve them through the layer
    // instead of Scene-level per-layer registries. Registration only ever appends live pointers,
    // so an empty list means "nothing mounted" exactly like the former absent map entry.
    void AddRuntimeLight(SceneLight* light) { m_runtime_lights.push_back(light); }
    const std::vector<SceneLight*>& RuntimeLights() const { return m_runtime_lights; }
    void AddRuntimeParticleSubsystem(ParticleSubSystem* subsystem) {
        m_runtime_particle_subsystems.push_back(subsystem);
    }
    const std::vector<ParticleSubSystem*>& RuntimeParticleSubsystems() const {
        return m_runtime_particle_subsystems;
    }

    // The layer's live draw handles: world/placeholder node, detached effect-source node, model
    // material nodes, particle renderer nodes. Same append-only contract as the lists above.
    // The list must be cleared at the exact point the nodes are freed (layer destroy and
    // re-materialization), because readers walk these pointers for visibility, residency, and
    // material updates.
    void AddRuntimeNode(SceneNode* node) { m_runtime_nodes.push_back(node); }
    const std::vector<SceneNode*>& RuntimeNodes() const { return m_runtime_nodes; }
    void ClearRuntimeNodes() { m_runtime_nodes.clear(); }

    // Text layers keep their runtime text record (authored object snapshot, live primitive,
    // render contract, applied alignment) on the identity. nullptr means "not a registered text
    // layer"; registration overwrites the record exactly like the former Scene::textLayers map
    // entry. Held by pointer so this header does not include the parser-side type; the setter is
    // defined out-of-line in Scene.cpp for the same reason.
    TextLayerRuntimeState* TextRuntimeState() const { return m_text_runtime_state.get(); }
    void                   SetTextRuntimeState(TextLayerRuntimeState state);

    // Camera layers keep their runtime record here; nullptr means "not a camera layer".
    // Registration overwrites the record exactly like the former Scene::cameraLayers map entry.
    SceneCameraLayerRuntimeState* CameraRuntimeState() {
        return m_has_camera_runtime_state ? &m_camera_runtime_state : nullptr;
    }
    const SceneCameraLayerRuntimeState* CameraRuntimeState() const {
        return m_has_camera_runtime_state ? &m_camera_runtime_state : nullptr;
    }
    void SetCameraRuntimeState(SceneCameraLayerRuntimeState state) {
        m_camera_runtime_state     = std::move(state);
        m_has_camera_runtime_state = true;
    }

    // The authored scene.json record for this object, normalized to its parse-time id. Deferred
    // layers re-parse it on materialization and scripts read originalOrigin and the initial
    // config from it. nullptr means no record (sound-only and some helper layers).
    const std::string* InitialConfigJson() const {
        return m_initial_config_json.has_value() ? &*m_initial_config_json : nullptr;
    }
    void SetInitialConfigJson(std::string config_json) {
        m_initial_config_json = std::move(config_json);
    }
    void ClearInitialConfigJson() { m_initial_config_json.reset(); }

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

    SceneDeferredRuntimeKind   m_deferred_runtime_kind { SceneDeferredRuntimeKind::None };
    std::optional<uint32_t>    m_sound_handle;
    std::optional<std::string> m_initial_config_json;

    bool       m_has_layer_node_slot { false };
    SceneNode* m_layer_node { nullptr };

    std::shared_ptr<SceneImageEffectLayer> m_image_effect_layer;

    std::vector<SceneLight*>        m_runtime_lights;
    std::vector<ParticleSubSystem*> m_runtime_particle_subsystems;
    std::vector<SceneNode*>         m_runtime_nodes;

    std::unique_ptr<TextLayerRuntimeState> m_text_runtime_state;

    bool                         m_has_camera_runtime_state { false };
    SceneCameraLayerRuntimeState m_camera_runtime_state;

    bool                        m_has_image_runtime_state { false };
    SceneImageLayerRuntimeState m_image_runtime_state;
};

} // namespace wallpaper

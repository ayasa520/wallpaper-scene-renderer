#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Core/NoCopyMove.hpp"
#include "SceneTransform.h"

namespace wallpaper
{

class Scene;
class SceneNode;
class SceneImageEffectLayer;
class SceneLight;
class SceneModelData;
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

// Image layers keep a small always-available runtime state (authored quad size plus alignment)
// that scripts read and write independently from any render node. It lives on the SceneObject so
// the object is the single source for layer-level state.
struct SceneImageLayerRuntimeState {
    std::array<float, 2> size { 0.0f, 0.0f };
    std::string          alignment { "center" };
    // The runtime property belongs to the authored image, including logical-only compositions.
    // Source selection and destination blending read this same value after script writes; they
    // must not retain separate parse-time copies of copybackground.
    bool                 copy_background { true };
};

// These are owner properties, not material controls. A shape retains them even without a source
// mesh or any effects, and its typed property setters do not visit or invalidate authored effect
// materials. Keeping the record on the identity prevents a draw's current camera/output from
// selecting the storage used by scripts. Other owner kinds can adopt this record at their own
// staging boundary without changing the shape property contract.
struct SceneLayerModulationState {
    std::array<float, 3> color { 1.0f, 1.0f, 1.0f };
    float               alpha { 1.0f };
    float               brightness { 1.0f };
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
    // Live placement is independent from the original JSON snapshots above and from the lifetime
    // of a drawing handle. Registration adopts the canonical handle's existing record without a
    // transform copy; scripts and attachment/layout writers subsequently share this single state.
    const std::shared_ptr<SceneTransform>& RuntimeTransform() const { return m_runtime_transform; }
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

    // Authored parent binding by layer id. parent_id 0 means "no parent". Parent destruction
    // clears this relation and its attachment while retaining this child's local transform;
    // owning a child in the scene hierarchy does not own that child's authored lifetime.
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

    // Parsed effect count from the authored scene entry. Visibility never changes this count.
    int32_t EffectCount() const { return m_effect_count; }
    void    SetEffectCount(int32_t count) { m_effect_count = count; }

    // The model's passthrough flag gives this owner a composition child phase. With zero visible
    // effect steps, only an empty non-private owner skips drawing; nonempty compositions still
    // publish their child source.
    bool Passthrough() const { return m_passthrough; }
    void SetPassthrough(bool passthrough) { m_passthrough = passthrough; }

    // Reflection membership belongs to the authored owner, not to a cloned mesh or material. A
    // model with even one active receiver material excludes all of its chunks from the producer
    // walk. Material setup records that fact independently from the authored opt-out.
    bool Reflected() const { return m_reflected; }
    void SetReflected(bool reflected) { m_reflected = reflected; }
    bool ReceivesReflection() const { return m_receives_reflection; }
    void SetReceivesReflection(bool receives) { m_receives_reflection = receives; }

    // Set when another layer's authored effect samples this layer's private offscreen output
    // (`_rt_imageLayerComposite_<id>`). Derived once at parse time from the scene-wide dependency
    // scan; hidden dependency sources keep rendering into private targets and stay GPU-resident.
    bool IsOffscreenDependencySource() const { return m_offscreen_dependency_source; }
    void MarkOffscreenDependencySource() { m_offscreen_dependency_source = true; }

    // Sound layers mount a SoundManager stream; the handle is that layer's runtime resource.
    // nullopt means no stream was mounted. A stored 0 is kept as a value on purpose: it is what a
    // failed mount recorded in the former Scene map, and presence checks must keep matching it.
    std::optional<uint32_t> SoundHandle() const { return m_sound_handle; }
    void                    SetSoundHandle(uint32_t handle) { m_sound_handle = handle; }
    void                    ClearSoundHandle() { m_sound_handle.reset(); }

    // The layer's script-visible handle node. The slot is tri-state to preserve the former
    // Scene::layerNodes semantics: no slot means the layer is not registered, a slot holding
    // nullptr means the layer is registered but has no drawable handle (for example a node-less
    // sound layer), and otherwise the slot is the live handle. Registration checks must use
    // HasLayerNodeSlot, not the node value.
    bool       HasLayerNodeSlot() const { return m_has_layer_node_slot; }
    SceneNode* LayerNode() const { return m_layer_node; }
    void       SetLayerNode(SceneNode* node);
    void ClearLayerNodeSlot() {
        m_layer_node          = nullptr;
        m_has_layer_node_slot = false;
        m_runtime_transform.reset();
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

    // The concrete layer registers its setup operation after materialization. Hierarchy edits
    // dispatch through the owner, independent of passthrough and without teaching Scene about
    // parser-side text layout or shape material controls. Identity-only prepass entries have no
    // resources to refresh; their eventual constructor consumes the recorded ancestry.
    using ResourceSetupCallback = void (*)(Scene&, SceneObject&);
    void SetResourceSetupCallback(ResourceSetupCallback callback) {
        m_resource_setup = callback;
    }
    void RefreshResources(Scene& scene) {
        if (m_resource_setup != nullptr) m_resource_setup(scene, *this);
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

    // The layer's live resource handles: its authored node, model material nodes,
    // and particle renderer nodes. Same append-only contract as the lists above. The list must be
    // cleared at the exact point the nodes are freed because readers walk these pointers for
    // visibility, resource ownership, and material updates.
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

    // The authored scene.json record for this object, normalized to its parse-time id. Scripts read
    // originalOrigin and the initial config from it. nullptr means no record (sound-only and some
    // helper layers).
    const std::string* InitialConfigJson() const {
        return m_initial_config_json.has_value() ? &*m_initial_config_json : nullptr;
    }
    void SetInitialConfigJson(std::string config_json) {
        m_initial_config_json = std::move(config_json);
    }
    void ClearInitialConfigJson() { m_initial_config_json.reset(); }

    // Image runtime state exists only for parsed image layers; other kinds return nullptr, which is
    // what tells scripts "this is not an image layer".
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

    SceneLayerModulationState* ModulationState() {
        return m_modulation_state ? &*m_modulation_state : nullptr;
    }
    const SceneLayerModulationState* ModulationState() const {
        return m_modulation_state ? &*m_modulation_state : nullptr;
    }
    void SetModulationState(SceneLayerModulationState state) { m_modulation_state = state; }

    // Each model owner retains its generated resource independently from the script handle.
    // Materials live on the owner's draw nodes; geometry/revisions live in this shared model.
    const std::shared_ptr<SceneModelData>& ModelData() const { return m_model_data; }
    void SetModelData(std::shared_ptr<SceneModelData> data) { m_model_data = std::move(data); }
    uint64_t ModelDataRevision() const { return m_model_data_revision; }
    void SetModelDataRevision(uint64_t revision) { m_model_data_revision = revision; }

    // Auxiliary projection is a model-owner property. Shared geometry may be drawn by owners
    // with different projection choices, without replacing either the scene camera or mesh.
    bool ModelPerspective() const { return m_model_perspective; }
    void SetModelPerspective(bool perspective) { m_model_perspective = perspective; }

private:
    int32_t         m_id { 0 };
    SceneObjectKind m_kind { SceneObjectKind::Empty };
    std::string     m_name;

    std::array<float, 3> m_origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> m_scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> m_angles { 0.0f, 0.0f, 0.0f };
    std::shared_ptr<SceneTransform> m_runtime_transform;

    bool        m_local_visible { true };
    int32_t     m_parent_id { 0 };
    std::string m_attachment;

    int32_t m_effect_count { 0 };
    bool    m_passthrough { false };
    bool    m_offscreen_dependency_source { false };
    bool    m_reflected { false };
    bool    m_receives_reflection { false };

    std::optional<uint32_t>    m_sound_handle;
    std::optional<std::string> m_initial_config_json;

    bool       m_has_layer_node_slot { false };
    SceneNode* m_layer_node { nullptr };

    std::shared_ptr<SceneImageEffectLayer> m_image_effect_layer;
    ResourceSetupCallback                m_resource_setup { nullptr };

    std::vector<SceneLight*>        m_runtime_lights;
    std::vector<ParticleSubSystem*> m_runtime_particle_subsystems;
    std::vector<SceneNode*>         m_runtime_nodes;

    std::unique_ptr<TextLayerRuntimeState> m_text_runtime_state;

    bool                         m_has_camera_runtime_state { false };
    SceneCameraLayerRuntimeState m_camera_runtime_state;

    bool                        m_has_image_runtime_state { false };
    SceneImageLayerRuntimeState m_image_runtime_state;
    std::optional<SceneLayerModulationState> m_modulation_state;
    std::shared_ptr<SceneModelData> m_model_data;
    uint64_t m_model_data_revision { 0 };
    bool                          m_model_perspective { false };
};

} // namespace wallpaper

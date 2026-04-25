#pragma once
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "SceneTexture.h"
#include "SceneRenderTarget.h"
#include "SceneNode.h"
#include "SceneLight.hpp"
#include "WPSceneScriptHost.hpp"
#include "WPTextLayer.hpp"
#include "WPUserProperties.hpp"

#include "Core/NoCopyMove.hpp"

namespace wallpaper
{
class ParticleSystem;
class ParticleSubSystem;
class IShaderValueUpdater;
class IImageParser;
struct SceneImageEffect;
namespace audio
{
class SoundManager;
}

namespace fs
{
class VFS;
}
class Scene : NoCopy, NoMove {
public:
    struct ImageLayerRuntimeState {
        std::array<float, 2> size { 0.0f, 0.0f };
        std::string          alignment { "center" };
    };

    struct CameraLayerRuntimeState {
        // Wallpaper Engine camera layers are represented in scene.json as transform-only objects
        // with camera-specific properties. Keep the authored values beside the render node so
        // scripts and keyframe animations can round-trip the WE-facing origin/zoom values while
        // Hanabi stores the attached SceneCamera node in renderer coordinates.
        std::string                camera_name { "global" };
        std::shared_ptr<SceneNode> node;
        std::array<float, 3>       origin { 0.0f, 0.0f, 0.0f };
        std::array<float, 3>       angles { 0.0f, 0.0f, 0.0f };
        double                     zoom { 1.0 };
        float                      fov { 50.0f };
    };

    struct LayerParentBinding {
        int32_t     parent_id { 0 };
        std::string attachment;
    };

    struct BloomSettings {
        // Bloom is a scene option in Wallpaper Engine. It must run after the whole layer tree has
        // produced `_rt_default`, so the render graph owns it as a dedicated global post-process
        // node instead of attaching it to any authored layer.
        bool                       enabled { false };
        float                      strength { 0.0f };
        float                      threshold { 1.0f };
        std::array<float, 3>       tint { 1.0f, 1.0f, 1.0f };
        bool                       hdr { false };
        float                      hdrStrength { 0.0f };
        float                      hdrThreshold { 1.0f };
        float                      hdrScatter { 1.0f };
        float                      hdrFeather { 0.0f };
        int32_t                    hdrIterations { 0 };
        std::shared_ptr<SceneNode> node;
        // Scene Bloom is a small post-process chain in Wallpaper Engine assets, not a single
        // shader. Keep the legacy `node` as the runtime-uniform anchor, and store the ordered
        // pass nodes plus their output targets so SceneToRenderGraph can reproduce the chain.
        std::vector<std::shared_ptr<SceneNode>> nodes;
        std::vector<std::string>                outputs;
    };

    Scene();
    ~Scene();

    void                SetLayerParentBinding(int32_t layer_id, int32_t parent_id,
                                              std::string attachment = {});
    LayerParentBinding  GetLayerParentBinding(int32_t layer_id) const;
    void                ClearLayerParentBinding(int32_t layer_id);
    std::vector<int32_t> GetLayerChildren(int32_t layer_id) const;

    void SetLayerLocalVisibility(int32_t layer_id, bool visible);
    bool GetLayerLocalVisibility(int32_t layer_id) const;
    bool IsLayerVisible(int32_t layer_id) const;
    void ApplyLayerVisibility(int32_t layer_id);
    void ApplyAllLayerVisibility();
    Eigen::Vector3f ResolveCameraLayerNodeTranslation(
        const std::array<float, 3>& authored_origin) const;
    void UpdateActiveCameraLayer();

    SceneImageEffect*       FindImageEffect(int32_t owner_layer_id, uint32_t effect_index);
    const SceneImageEffect* FindImageEffect(int32_t owner_layer_id, uint32_t effect_index) const;
    SceneImageEffect*       FindImageEffectById(int32_t owner_layer_id, int32_t effect_id);
    const SceneImageEffect* FindImageEffectById(int32_t owner_layer_id, int32_t effect_id) const;
    bool                    SetEffectLocalVisibility(int32_t owner_layer_id,
                                                     uint32_t effect_index, bool visible);
    bool                    SetEffectLocalVisibilityById(int32_t owner_layer_id,
                                                         int32_t effect_id, bool visible);

    std::unordered_map<std::string, SceneTexture>      textures;
    std::unordered_map<std::string, SceneRenderTarget> renderTargets;

    std::unordered_map<std::string, std::shared_ptr<SceneCamera>> cameras;
    std::unordered_map<std::string, std::vector<std::string>>     linkedCameras;

    std::vector<std::unique_ptr<SceneLight>> lights;

    std::shared_ptr<SceneNode>           sceneGraph;
    std::unique_ptr<IShaderValueUpdater> shaderValueUpdater;
    std::unique_ptr<IImageParser>        imageParser;
    std::unique_ptr<fs::VFS>             vfs;
    std::unique_ptr<WPSceneScriptHost>   scriptHost;
    std::vector<WPSceneScriptRegistration> bindingRegistrations;
    std::vector<WPSceneScriptRegistration> scriptRegistrations;
    std::vector<WPSceneScriptRegistration> propertyAnimationRegistrations;
    std::vector<int32_t>                 layerOrder;
    std::unordered_map<int32_t, SceneNode*> layerNodes;
    std::unordered_map<int32_t, LayerParentBinding> layerParentBindings;
    std::unordered_map<int32_t, bool>    layerLocalVisibility;
    std::unordered_map<int32_t, std::vector<SceneNode*>> objectRuntimeNodes;
    std::unordered_map<int32_t, std::vector<std::string>> objectRuntimeCameraNames;
    std::unordered_map<int32_t, std::vector<std::string>> objectRuntimeRenderTargets;
    std::unordered_map<int32_t, std::vector<SceneLight*>> objectRuntimeLights;
    std::unordered_map<int32_t, std::vector<ParticleSubSystem*>> objectRuntimeParticleSubsystems;
    std::unordered_set<int32_t>                          deferredRuntimeParticleLayerIds;
    std::unordered_set<int32_t>                          deferredRuntimeTextLayerIds;
    std::unordered_map<int32_t, uint32_t>                 objectRuntimeSoundHandles;
    std::unordered_map<int32_t, ImageLayerRuntimeState>   imageLayers;
    std::unordered_map<int32_t, TextLayerRuntimeState>    textLayers;
    std::unordered_map<int32_t, CameraLayerRuntimeState>  cameraLayers;
    std::vector<int32_t>                                  cameraLayerOrder;
    std::unordered_map<SceneNode*, int32_t> nodeOwners;
    std::unordered_map<int32_t, std::string> initialLayerConfigJson;
    std::unordered_map<std::string, int32_t> layerNameToId;
    std::unordered_set<int32_t>              offscreenDependencyLayerIds;
    // Some runtime nodes must stay root-owned for transform correctness, effect-camera routing, or
    // deferred materialization, but Wallpaper Engine still orders them as children of their
    // authored parent layer. These maps keep physical ownership separate from authored render
    // order so the render graph can emit passes at the correct sibling position without changing
    // the node's transform/output ownership model.
    std::unordered_map<SceneNode*, std::vector<SceneNode*>> renderOrderProxyChildren;
    std::unordered_set<SceneNode*>                          renderOrderProxyNodes;
    // Effect-backed image/text layers split into a visible world node plus a root-owned source
    // node that draws through the effect camera. The source must render at the world node's
    // authored position in sibling order, then be skipped when the physical root traversal reaches
    // the root-owned source node later.
    std::unordered_map<SceneNode*, std::vector<SceneNode*>> detachedEffectSourceNodesByWorldNode;
    std::unordered_set<SceneNode*>                          detachedEffectSourceNodes;
    UserPropertyMap                      userProperties;
    std::set<std::string>                dirtyImportedTextureKeys;
    std::unordered_set<std::string>      dirtyRenderTargetKeys;
    // Direct text rerastering changes pass-owned atlas and mesh resources without naming a
    // render-target dependency, so text layers need their own resource-refresh dirty key set.
    std::unordered_set<int32_t>          dirtyTextLayerIds;
    bool                                 renderGraphDirty { false };
    bool                                 renderGraphResourcesDirty { false };
    bool                                 renderGraphTopologyDirty { false };
    bool                                 renderGraphAllResourcesDirty { false };
    // Wallpaper Engine scene scripts can pause and resume video textures independently from layer
    // visibility. Keep the desired playback state on the scene so the script host can update it
    // during the QuickJS tick while the Vulkan video cache consumes the same state on the render
    // thread before polling GStreamer.
    std::unordered_map<std::string, bool> videoTexturePaused;

    std::string scene_id { "unknown_id" };

    bool first_frame_ok { false };

    SceneMesh default_effect_mesh;

    std::unique_ptr<ParticleSystem> paritileSys;
    audio::SoundManager*            soundManager { nullptr };
    std::array<float, 2>            mousePositionNormalized { 0.5f, 0.5f };
    bool                            cursorLeftDown { false };

    SceneCamera* activeCamera;
    std::shared_ptr<SceneNode> defaultGlobalCameraNode;
    double                     defaultGlobalCameraZoom { 1.0 };
    int32_t                    activeCameraLayerId { 0 };

    i32                  ortho[2] { 1920, 1080 }; // w, h
    std::array<float, 3> clearColor { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> ambientColor { 0.2f, 0.2f, 0.2f };
    std::array<float, 3> skylightColor { 0.3f, 0.3f, 0.3f };
    BloomSettings        bloom;
    bool                 cameraParallax { false };
    float                cameraParallaxAmount { 0.0f };
    float                cameraParallaxDelay { 0.0f };
    float                cameraParallaxMouseInfluence { 0.0f };
    double               textRenderScale { 1.0 };

    double elapsingTime { 0.0f }, frameTime { 0.0f };
    void   PassFrameTime(double t) {
          frameTime = t;
          elapsingTime += t;
    }

    void MarkRenderGraphResourcesDirty() {
        // Global resource refreshes are still available for broad image-layer edits whose affected
        // render-target set is not known locally. Clearing the selective target set here makes the
        // renderer intentionally refresh every prepared pass instead of accidentally treating a
        // previous text-bridge target as the only dirty dependency.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        renderGraphAllResourcesDirty = true;
        dirtyRenderTargetKeys.clear();
        dirtyTextLayerIds.clear();
    }

    void MarkRenderTargetResourcesDirty(std::string render_target_key) {
        // Text bridges resize like particle-owned resources: the graph topology stays stable and
        // only passes that write or sample the changed render target need their framebuffer and
        // descriptor handles refreshed. Keeping this target list explicit prevents one clock tick
        // from refreshing every unrelated wallpaper effect pass.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        if (!renderGraphAllResourcesDirty && !render_target_key.empty()) {
            dirtyRenderTargetKeys.insert(std::move(render_target_key));
        }
    }

    void MarkTextLayerResourcesDirty(int32_t layer_id) {
        // Direct text layers can change glyph atlas textures and vertex meshes while keeping the
        // same render target. Track the owning layer separately so the renderer refreshes the exact
        // TextPass before the next frame upload instead of discovering the atlas change during draw.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        if (!renderGraphAllResourcesDirty && layer_id != 0) {
            dirtyTextLayerIds.insert(layer_id);
        }
    }

    void MarkRenderGraphTopologyDirty() {
        // Topology rebuilds are the heavy path: the set of render passes or runtime scene nodes
        // changed, so the renderer must rebuild the graph structure and recreate its resources.
        // A topology change also implies a resource refresh, so both flags rise together here.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        renderGraphTopologyDirty = true;
    }

    void ClearRenderGraphDirty() {
        renderGraphDirty = false;
        renderGraphResourcesDirty = false;
        renderGraphTopologyDirty = false;
        renderGraphAllResourcesDirty = false;
        dirtyRenderTargetKeys.clear();
        dirtyTextLayerIds.clear();
    }

    void UpdateLinkedCamera(const std::string& name) {
        if (linkedCameras.count(name) != 0) {
            auto& cams = linkedCameras.at(name);
            for (auto& cam : cams) {
                if (cameras.count(cam) != 0) {
                    cameras.at(cam)->Clone(*cameras.at(name));
                    cameras.at(cam)->Update();
                }
            }
        }
    }
};
} // namespace wallpaper

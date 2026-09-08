#pragma once
#include <vector>
#include <list>
#include <memory>
#include <cstdint>
#include <string>
#include <string_view>
#include <array>
#include <optional>
#include <unordered_map>
#include <limits>
#include <Eigen/Geometry>
#include "Core/Literals.hpp"
#include "Type.hpp"
#include "WPPuppet.hpp"
#include "SceneDraw.h"

namespace wallpaper
{

class SceneNode;
class SceneObject;
class SceneMesh;
class Scene;
struct SceneShader;

std::string_view FinalOutputCapabilityName(FinalOutputCapability capability);

struct SceneImageEffectNode {
    // Authored effect passes use symbolic ping-pong targets. ResolveEffect() maps those symbols to
    // concrete render targets every time the graph is built, so keep the parsed template separate
    // from the mutable runtime output to avoid carrying one build's A/B swap into the next build.
    std::string                authored_output; // parsed render target template
    std::string                output;          // resolved render target for the current graph
    std::vector<std::string>   authored_textures;
    // Swap rewrites effect FBO indices, not the material's literal texture array. Retain which
    // slots came from the effect's explicit binding list.
    std::vector<usize>         fbo_texture_slots;
    bool                      output_is_fbo { false };
    std::shared_ptr<SceneNode> sceneNode;
    bool advances_composition { false };
    // The final material is selected while parsing the ordered records, independently of trailing
    // commands or an explicit FBO output. Shapes use their retained card for this material alone
    // and preserve other materials' blend state.
    bool is_final_material { false };
    BlendMode authored_blend { BlendMode::Normal };
    // Matrix phase belongs to the deferred destination segment. Auxiliary FBO passes in that
    // segment use the same unscaled layer snapshots as its visible output material, even though
    // their raster mesh remains a fullscreen quad. ResolveEffect() recalculates this phase.
    bool uses_layer_space_effect_matrices { false };
    // Several materials can draw the deferred destination segment. Placement and live card
    // updates belong to each such draw, rather than to one arbitrarily selected final node.
    bool uses_owner_transform { false };
    bool mesh_follows_final_mesh { false };
    // Effect nodes are render-graph internals, not authored scene owners. When an internal pass
    // needs a layer-local camera, store it as a pass override instead of assigning it to
    // SceneNode::Camera(); otherwise graph traversal may see the image-effect camera attached to
    // that layer and recursively resolve the same effect chain a second time.
    std::string camera_override;
    bool        use_active_camera_for_parallax { false };
    bool        clear_before_draw { false };
    AlphaWritePolicy alpha_write_policy { AlphaWritePolicy::Preserve };
};

struct SceneImageEffect {
    enum class VisibilityPolicy { Instance, OwnerOnly };
    explicit SceneImageEffect(VisibilityPolicy policy = VisibilityPolicy::Instance)
        : m_visibility_policy(policy) {}

    std::size_t CompositionStepCount() const;

    enum class CmdType
    {
        Copy,
        Swap,
    };
    struct Command {
        CmdType     cmd { CmdType::Copy };
        // Only declared FBOs have command indices. A missing source means the active draw
        // destination, while a missing target performs no copy. Neither case uses a material's
        // special `previous` input binding. Keep the absence explicit so a later graph rebuild
        // cannot mistake a resolved target for authored data.
        std::optional<std::string> authored_dst;
        std::optional<std::string> authored_src;
        std::string dst;
        std::string src;
        i32         afterpos { 0 }; // start at 1, 0 for begin at all
        bool        advances_composition { false };
        bool        uses_final_destination { false };
    };
    std::vector<Command>            commands;
    std::list<SceneImageEffectNode> nodes;

    using FboBindings = std::unordered_map<std::string, std::string>;
    void RegisterFbo(const std::string& target) { m_fbo_bindings.emplace(target, target); }
    bool IsDeclaredFbo(const std::string& target) const {
        return m_fbo_bindings.contains(target);
    }
    const FboBindings& CurrentFboBindings() const { return m_fbo_bindings; }
    const FboBindings& NextFboBindings() const { return m_next_fbo_bindings; }
    void SetNextFboBindings(FboBindings bindings) { m_next_fbo_bindings = std::move(bindings); }
    bool HasPendingFboSwap() const { return m_fbo_bindings != m_next_fbo_bindings; }
    bool CommitFboBindings(const FboBindings& bindings);
    static void SwapFboBindings(FboBindings&, const std::string& source,
                                const std::string& target);
    void ResolveCommand(Command&, FboBindings&, std::string_view current_destination);

    // Effect visibility is a first-class runtime contract. Wallpaper Engine allows an effect to
    // start hidden and later become visible through a script, user property, or animation. The
    // effect therefore needs its own local visibility state instead of borrowing the owner layer's
    // visibility or being pruned while parsing.
    void SetIdentity(int32_t owner_layer_id, int32_t effect_id, uint32_t effect_index,
                     std::string effect_name);
    void SetLocalVisible(bool visible);
    bool LocalVisible() const { return m_local_visible; }

    int32_t            OwnerLayerId() const { return m_owner_layer_id; }
    int32_t            EffectId() const { return m_effect_id; }
    uint32_t           EffectIndex() const { return m_effect_index; }
    const std::string& EffectName() const { return m_effect_name; }

private:
    // A shape retains the authored visibility bit for scripts, but its draw callback ignores
    // that bit when dispatching the physical last effect. Image/text effects use instance gates.
    const VisibilityPolicy m_visibility_policy;
    // A graph build evaluates the ordered stream on a copy of this persistent index table.
    // Only a submitted frame commits the result: warm-up, repeated graph construction and
    // skipped frames must not advance feedback history. Swap commands retain their fixed
    // authored operands while non-swap references use the current table at each boundary.
    FboBindings m_fbo_bindings;
    FboBindings m_next_fbo_bindings;
    int32_t     m_owner_layer_id { 0 };
    int32_t     m_effect_id { 0 };
    uint32_t    m_effect_index { 0 };
    std::string m_effect_name;
    bool        m_local_visible { true };
};

class SceneImageEffectLayer {
public:
    struct PrelightingSource {
        // Source variants change the executable and vertex stream while retaining the authored
        // material controls. Keep those programs beside the source geometry; live constants,
        // aliases and texture assignments stay on the owner's one material, including when effect
        // visibility switches this branch at runtime.
        std::shared_ptr<SceneShader> ordinary_shader;
        std::shared_ptr<SceneShader> prelighting_shader;
        std::shared_ptr<SceneMesh> mesh;
        std::array<float, 2> content_size;
        bool sprite { false };
        bool instanced { false };
    };

    struct DirectPuppetSource {
        // Direct skinning selects a skinned executable for the authored material. Keep controls
        // on the owner's material; switching execution must not replace values already written by
        // scripts. Promotion to private publication persists for this owner, even when its
        // effects are hidden again before the next graph build.
        std::shared_ptr<SceneShader> ordinary_shader;
        std::shared_ptr<SceneShader> skinned_shader;
        bool private_publication { false };
    };

    enum class SourcePolicy
    {
        None,
        OwnerNode,
        OwnerNodeAndProxyChildren,
        ProxyChildrenOnly,
    };
    SceneImageEffectLayer(SceneObject& owner, float w, float h, std::string_view pingpong_a,
                          std::string_view pingpong_b);

    void AddEffect(const std::shared_ptr<SceneImageEffect>& node) { m_effects.push_back(node); }
    std::size_t EffectCount() const { return m_effects.size(); }
    std::size_t VisibleCompositionStepCount() const;
    bool HasVisibleEffects() const;
    bool UsesShapeDraw() const;
    bool ShouldExecuteEffect(const SceneImageEffect& effect) const;
    auto&       GetEffect(std::size_t index) { return m_effects.at(index); }
    std::size_t SourceSlot() const;
    const std::string& SourceTarget() const;
    void SetDestinationTargets(std::string first_target, std::string second_target);
    void RefreshDestinationTargets(
        Scene& scene, std::optional<std::array<int32_t, 2>> destination_extent = std::nullopt);
    void SetDestinationUsesCardSize(bool enabled) { m_destination_uses_card_size = enabled; }
    SceneMesh&  SourceMesh() const { return *m_source_mesh; }
    void SetDirectDrawMesh(const SceneMesh& mesh) { m_direct_draw_mesh = &mesh; }
    void SetDirectPuppetSource(DirectPuppetSource source) {
        m_direct_puppet_source = std::move(source);
    }
    void RefreshPuppetPublicationState();
    bool UsesDirectDraw() const;
    void SetPrelightingSource(PrelightingSource source) {
        m_prelighting_source = std::move(source);
    }
    const PrelightingSource* GetPrelightingSource() const {
        return m_prelighting_source ? &*m_prelighting_source : nullptr;
    }
    bool UsesPrelightingSource() const;
    void ResolveOwnerDraw(Scene& scene);
    SceneMesh&  FinalMesh() const { return *m_final_mesh; }
    SceneDrawPhase& FinalCompositeDraw() { return m_final_composite.draw; }
    const SceneDrawPhase& FinalCompositeDraw() const { return m_final_composite.draw; }
    const std::array<float, 2>& CardSize() const { return m_card_size; }
    void SetCardSize(const std::array<float, 2>& size) { m_card_size = size; }
    bool UsesLayerSpaceEffectMatrices(const SceneNode* draw_node) const;
    // Destination draws resolve the owning object's current transform at uniform evaluation.
    // Private source/effect draws retain their local camera space. This phase selection carries
    // no copied transform, so script writes and graph reconstruction observe the same owner.
    bool UsesOwnerTransform(const SceneNode* draw_node) const;
    SceneObject& Owner() const { return m_owner; }
    SourcePolicy SourceContributionPolicy() const;
    FinalOutputCapability DeclaredFinalOutputCapability() const {
        return m_final_output_capability;
    }
    FinalOutputCapability ResolveFinalOutputCapability() const;
    bool        HasFinalComposite() const;
    bool        ShouldRunFinalComposite() const;
    bool        PublishesPrivateFinalComposite() const {
        return m_final_composite.publishes_private_output;
    }
    void        SetFinalCompositeSource(std::string source);
    void        SetFullscreen(bool fullscreen) { m_fullscreen = fullscreen; }
    bool        IsFullscreen() const { return m_fullscreen; }
    void        SetFinalOutputCapability(FinalOutputCapability capability) {
        m_final_output_capability = capability;
    }
    // Projection resource selected by this owner's source pass. It is not stored on the
    // authored node, so descendants and script queries retain their scene camera selection.
    void SetBridgeCameraName(std::string camera_name) {
        m_bridge_camera_name = std::move(camera_name);
    }
    const std::string& BridgeCameraName() const { return m_bridge_camera_name; }

    // Effect snapshots retain the camera and destination that were active on entry to the owner,
    // before its private source draw. An empty name selects scene camera/destination state; a
    // composition name selects its centered projection and identity destination. The snapshot
    // replaces incoming I with the raw owner matrix, so this context must not use the publication
    // draw's camera or the composition camera's attached inverse-owner transform.
    void SetEffectSnapshotCamera(std::string camera_name) {
        m_effect_snapshot_camera = std::move(camera_name);
    }
    const std::string& EffectSnapshotCamera() const { return m_effect_snapshot_camera; }

    // Names of the Scene::cameras entries this bridge materialized. These projection resources
    // belong to the bridge; geometry updates and layer destroy resolve them through the owner's
    // bridge instead of a Scene-level per-layer registry.
    void AddRuntimeCameraName(std::string camera_name) {
        m_runtime_camera_names.push_back(std::move(camera_name));
    }
    const std::vector<std::string>& RuntimeCameraNames() const { return m_runtime_camera_names; }

    // Names of the Scene::renderTargets entries this bridge materialized (ping-pong pair, effect
    // FBOs). Same ownership contract as the runtime camera names.
    void AddRuntimeRenderTargetName(std::string render_target_name) {
        m_runtime_render_target_names.push_back(std::move(render_target_name));
    }
    const std::vector<std::string>& RuntimeRenderTargetNames() const {
        return m_runtime_render_target_names;
    }
    void AddEffectRenderTarget(std::string name, uint32_t scale, uint32_t fit);
    bool ResizeEffectRenderTargets(Scene& scene, std::array<float, 2> source_extent);
    bool        CopyBackground() const;
    AlphaWritePolicy CompositionChildAlphaWritePolicy() const {
        return CopyBackground() ? AlphaWritePolicy::Preserve : AlphaWritePolicy::Max;
    }
    void        SetFinalBlend(BlendMode m) { m_final_blend = m; }
    void SetTransparentCompositionBlend(BlendMode blend) {
        m_transparent_composition_blend = blend;
    }
    BlendMode   FinalBlend() const;
    void        SyncResolvedOutputMesh();

    void ResolveEffect(const SceneMesh& defualt_mesh, std::string_view effect_cam,
                       std::string_view final_output);

private:
    struct FinalCompositeState {
        explicit FinalCompositeState(SceneObject& owner) : draw(owner) {}
        SceneDrawPhase draw;
        // An independent publisher is required by private output ownership. Ordinary visible
        // chains use their authored destination segment; the absence of visible effects instead
        // exposes the source.
        bool publishes_private_output { false };
        void ResetForResolve() {
            publishes_private_output = false;
        }
    };

    struct FinalOutputResolveDecision {
        bool keep_authored_final_private { false };
    };

    struct EffectRenderTarget {
        std::string name;
        uint32_t scale;
        uint32_t fit;
    };

    SceneObject& m_owner;
    std::string m_pingpong_a;
    std::string m_pingpong_b;
    // Effect texture projection maps a normalized card through the layer's authored size.
    // Mesh bounds may instead describe a cropped static mesh or an animated puppet envelope;
    // neither is the card domain sampled by the authored effect. Parser and layout/size updates
    // keep this geometry value current independently of resource setup. Only the card-derived
    // destination policy consumes it when a later resource boundary rebuilds the targets.
    std::array<float, 2> m_card_size;
    bool m_destination_uses_card_size { false };

    // Fullscreen utility layers, such as Wallpaper Engine's postprocess framebuffer layer, are
    // authored in clip-space sized 2x2 quads. Their final effect pass must therefore stay on the
    // effect camera/fullscreen mesh path; resolving that pass through the active scene camera turns
    // a shader such as godrays_combine into a tiny world-space quad and makes the rays disappear.
    bool m_fullscreen { false };
    FinalOutputCapability m_final_output_capability {
        FinalOutputCapability::PrivateThenPublish
    };
    std::string m_bridge_camera_name;
    std::string m_effect_snapshot_camera;
    std::vector<std::string> m_runtime_camera_names;
    std::vector<std::string> m_runtime_render_target_names;
    // Each authored FBO record retains its sizing rule even when its name is shared with
    // another layer. A later setup resizes that same named resource in authored record order.
    std::vector<EffectRenderTarget> m_effect_render_targets;
    //    std::vector<float> m_size;
    std::unique_ptr<SceneMesh> m_source_mesh;
    std::optional<PrelightingSource> m_prelighting_source;
    std::optional<DirectPuppetSource> m_direct_puppet_source;
    std::unique_ptr<SceneMesh> m_final_mesh;
    // The image destination selects one of this layer's existing mesh resources. Ordinary
    // images use the source card's file UVs; static imported images use their authored final
    // geometry. Keep a reference so live size edits update both routes through the same data.
    const SceneMesh* m_direct_draw_mesh;
    FinalCompositeState        m_final_composite;
    BlendMode                  m_final_blend { BlendMode::Normal };
    // Select the transparent-source blend at draw time. Keep its parsed colorBlendMode precedence
    // separate from the authored blend so toggling copybackground in either direction restores
    // the correct destination state without recompiling shaders.
    BlendMode m_transparent_composition_blend { BlendMode::Translucent };

    std::vector<std::shared_ptr<SceneImageEffect>> m_effects;

    void ResolveEffectMatrixPhases(bool keep_final_private);
    void ResolveShapeEffect(const SceneMesh& default_mesh, std::string_view final_output);
    SceneImageEffectNode* ResolveEffectPingPongChain(const SceneMesh& default_mesh,
                                                     SceneNode& default_node,
                                                     std::string_view effect_cam,
                                                     std::string_view final_output,
                                                     std::string_view& ppong_a,
                                                     std::string_view& ppong_b);
    FinalOutputResolveDecision ResolveFinalOutputDecision(
        FinalOutputCapability output_capability);
    void ResolveFinalComposite(const SceneMesh& default_mesh,
                                   std::string_view effect_cam,
                                   std::string_view final_output,
                                   std::string_view final_composite_source);
    void ResolveVisibleFinalOutput(SceneImageEffectNode& final_output_node,
                                   const SceneMesh& default_mesh,
                                   SceneNode& default_node,
                                   std::string_view effect_cam,
                                   std::string_view final_output);
    void ResolvePrivateFinalOutput(SceneImageEffectNode& final_output_node,
                                   const SceneMesh& default_mesh,
                                   SceneNode& default_node,
                                   std::string_view effect_cam);
};
} // namespace wallpaper

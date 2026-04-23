#pragma once
#include <vector>
#include <list>
#include <memory>
#include <cstdint>
#include <string>
#include <Eigen/Geometry>
#include "Core/Literals.hpp"
#include "Type.hpp"

namespace wallpaper
{

class SceneNode;
class SceneMesh;

struct SceneImageEffectNode {
    std::string                output; // render target
    std::shared_ptr<SceneNode> sceneNode;
};

struct SceneImageEffect {
    enum class CmdType
    {
        Copy,
    };
    struct Command {
        CmdType     cmd { CmdType::Copy };
        std::string dst;
        std::string src;
        i32         afterpos { 0 }; // start at 1, 0 for begin at all
    };
    std::vector<Command>            commands;
    std::list<SceneImageEffectNode> nodes;

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

    // ResolveEffect() rewrites authored ping-pong aliases to concrete render targets. These two
    // names describe the stable bypass copy used while the effect is hidden: copy the input
    // ping-pong target to the output ping-pong target so downstream effects never sample a stale
    // frame left by the last visible execution.
    void SetBypassTargets(std::string src, std::string dst);
    const std::string& BypassSource() const { return m_bypass_src; }
    const std::string& BypassTarget() const { return m_bypass_dst; }

private:
    int32_t     m_owner_layer_id { 0 };
    int32_t     m_effect_id { 0 };
    uint32_t    m_effect_index { 0 };
    std::string m_effect_name;
    bool        m_local_visible { true };
    std::string m_bypass_src;
    std::string m_bypass_dst;
};

class SceneImageEffectLayer {
public:
    SceneImageEffectLayer(SceneNode* node, float w, float h, std::string_view pingpong_a,
                          std::string_view pingpong_b);

    void AddEffect(const std::shared_ptr<SceneImageEffect>& node) { m_effects.push_back(node); }
    std::size_t EffectCount() const { return m_effects.size(); }
    auto&       GetEffect(std::size_t index) { return m_effects.at(index); }
    const auto& FirstTarget() const { return m_pingpong_a; }
    SceneMesh&  FinalMesh() const { return *m_final_mesh; }
    SceneNode&  FinalNode() const { return *m_final_node; }
    bool        HasFinalComposite() const;
    bool        ShouldRunFinalCompositeFallback() const;
    void        SetFinalCompositeSource(std::string source);
    SceneNode*  WorldNode() const { return m_worldNode; }
    void        SetFinalBlend(BlendMode m) { m_final_blend = m; }
    void        SyncResolvedOutputMesh();
    void        SyncResolvedNodeToWorld();
    void        SyncResolvedNodeToMatrix(const Eigen::Affine3f& world_affine);

    void ResolveEffect(const SceneMesh& defualt_mesh, std::string_view effect_cam,
                       bool keep_final_output_private = false,
                       const Eigen::Affine3f* resolved_world_affine = nullptr);

private:
    SceneNode*  m_worldNode;
    std::string m_pingpong_a;
    std::string m_pingpong_b;

    bool fullscreen { false };
    //    std::vector<float> m_size;
    std::unique_ptr<SceneMesh> m_final_mesh;
    std::unique_ptr<SceneNode> m_final_node;
    SceneNode*                 m_resolved_output_node { nullptr };
    // Visible effect layers resolve their final authored pass in scene space, so runtime transform
    // updates must keep that node synchronized with the layer world node. Hidden dependency sources
    // resolve into a private offscreen texture instead; their final pass must stay in the effect
    // camera's local fullscreen space or `_rt_imageLayerComposite_<id>` samples a shifted source.
    bool                       m_resolved_output_follows_world { true };
    // The synthetic final composite is a narrow fallback, not the normal output path. Keep the
    // authored final effect as the resolved screen writer while it is visible, and only enable the
    // passthrough composite when that exact final effect becomes runtime-hidden.
    SceneImageEffect* m_final_output_effect { nullptr };
    BlendMode                  m_final_blend;

    std::vector<std::shared_ptr<SceneImageEffect>> m_effects;
};
} // namespace wallpaper

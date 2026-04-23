#include "SceneToRenderGraph.hpp"

#include "Scene/Scene.h"
#include "RenderGraph/RenderGraph.hpp"
#include "SpecTexs.hpp"
#include "Core/MapSet.hpp"
#include "Utils/Logging.h"

#include "VulkanRender/AllPasses.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace wallpaper;
namespace wallpaper::rg
{

void doCopy(RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc, TexNode* in, TexNode* out) {
    builder.read(in);
    builder.write(out);

    desc.src = in->key();
    desc.dst = out->key();
}
void addCopyPass(RenderGraph& rgraph, TexNode* in, TexNode* out,
                 std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [in, out, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc) {
            doCopy(builder, desc, in, out);
            desc.should_execute = should_execute;
        });
}

void addCopyPass(RenderGraph& rgraph, const TexNode::Desc& in, const TexNode::Desc& out,
                 std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [in, out, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc) {
            auto* in_node  = builder.createTexNode(in);
            auto* out_node = builder.createTexNode(out, true);
            doCopy(builder, desc, in_node, out_node);
            desc.should_execute = should_execute;
        });
}

TexNode* addCopyPass(RenderGraph& rgraph, TexNode* in, TexNode::Desc* out_desc = nullptr,
                     std::function<bool()> should_execute = {}) {
    TexNode* copy { nullptr };
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [&copy, in, out_desc, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& pdesc) {
            auto desc = out_desc == nullptr ? in->genDesc() : *out_desc;
            if (out_desc == nullptr) {
                desc.key += "_" + std::to_string(in->version()) + "_copy";
                desc.name += "_" + std::to_string(in->version()) + "_copy";
            }
            copy = builder.createTexNode(desc, true);
            doCopy(builder, pdesc, in, copy);
            pdesc.should_execute = should_execute;
        });
    return copy;
}

static TexNode::Desc createTexDesc(std::string path) {
    return TexNode::Desc { .name = path,
                           .key  = path,
                           .type = IsSpecTex(path) ? TexNode::TexType::Temp
                                                   : TexNode::TexType::Imported };
}
} // namespace wallpaper::rg

static void CheckAndSetSprite(Scene& scene, vulkan::CustomShaderPass::Desc& desc,
                              std::span<const std::string> texs) {
    for (usize i = 0; i < texs.size(); i++) {
        auto& tex = texs[i];
        if (! tex.empty() && ! IsSpecTex(tex) && scene.textures.count(tex) != 0) {
            const auto& stex = scene.textures.at(tex);
            if (stex.isSprite) {
                desc.sprites_map[i] = stex.spriteAnim;
            }
        }
    }
}

static bool IsAudioBarShaderNode(SceneNode* node) {
    if (node == nullptr || node->Mesh() == nullptr || node->Mesh()->Material() == nullptr) {
        return false;
    }
    const auto* shader = node->Mesh()->Material()->customShader.shader.get();
    return shader != nullptr && shader->name.find("Simple_Audio_Bars") != std::string::npos;
}

static bool IsSyntheticCompositeNode(SceneNode* node) {
    if (node == nullptr || node->Mesh() == nullptr || node->Mesh()->Material() == nullptr) {
        return false;
    }
    const auto* shader   = node->Mesh()->Material()->customShader.shader.get();
    return shader != nullptr && shader->name == "genericimage3";
}

static bool ShouldExecuteHiddenDependency(Scene& scene, SceneNode* node, std::string_view output) {
    const auto owner_it = scene.nodeOwners.find(node);
    if (owner_it == scene.nodeOwners.end()) return false;
    if (scene.offscreenDependencyLayerIds.count(owner_it->second) == 0) return false;

    // Hidden dependency layers are allowed to keep rendering only into private offscreen targets that
    // another effect samples. They must never use that exemption for `_rt_default`, because that
    // would make an invisible helper layer composite directly onto the wallpaper and create the large
    // tinted rectangles seen in xray-style scenes.
    return output != SpecTex_Default;
}

struct DelayLinkInfo {
    rg::NodeID id;
    rg::NodeID link_id;
    i32        tex_index;
};

struct ExtraInfo {
    Map<size_t, rg::TexNode*>  id_link_map {};
    std::vector<DelayLinkInfo> link_info {};
    rg::RenderGraph*           rgraph { nullptr };
    Scene*                     scene { nullptr };
    std::unordered_map<int32_t, size_t> layer_order_index {};
    bool                       use_mipmap_framebuffer { false };
};

static bool IsOffscreenDependencyLayer(const ExtraInfo& extra, i32 imgId) {
    return extra.scene != nullptr && imgId != 0 &&
        extra.scene->offscreenDependencyLayerIds.count(imgId) != 0;
}

static bool ShouldPublishLayerLinkOutput(const ExtraInfo& extra, i32 imgId,
                                         std::string_view output) {
    if (!IsOffscreenDependencyLayer(extra, imgId)) return true;
    if (output != SpecTex_Default) return true;

    // `_rt_imageLayerComposite_<id>` is a source-texture contract, not a screen-composite contract.
    // Hidden dependency layers may still contain historical final passes that target `_rt_default`,
    // but those passes are deliberately blocked from executing while the layer is invisible. Letting
    // such a skipped screen pass replace the layer's link source makes the consumer sample the
    // wallpaper/default target instead of the dependency's raw or effect-resolved offscreen image.
    return false;
}

struct OrderedRenderGraphChild {
    SceneNode* node { nullptr };
    bool       proxy { false };
    size_t     sequence { 0 };
};

static int32_t NodeLayerId(const Scene& scene, SceneNode* node) {
    if (node == nullptr) return 0;
    if (auto owner_it = scene.nodeOwners.find(node); owner_it != scene.nodeOwners.end()) {
        return owner_it->second;
    }
    return node->ID();
}

static size_t NodeLayerOrderIndex(SceneNode* node, const ExtraInfo& extra) {
    if (extra.scene == nullptr || node == nullptr) return std::numeric_limits<size_t>::max();
    const int32_t layer_id = NodeLayerId(*extra.scene, node);
    if (auto it = extra.layer_order_index.find(layer_id); it != extra.layer_order_index.end()) {
        return it->second;
    }
    return std::numeric_limits<size_t>::max();
}

static bool IsEffectLocalProxyDependency(SceneNode* node, const ExtraInfo& extra) {
    if (extra.scene == nullptr || node == nullptr) return false;
    const int32_t layer_id = NodeLayerId(*extra.scene, node);
    return layer_id != 0 && extra.scene->offscreenDependencyLayerIds.count(layer_id) != 0;
}

static Eigen::Matrix4d ResolveRouteModel(SceneNode* node,
                                         const std::optional<Eigen::Matrix4d>& route_model) {
    if (route_model.has_value()) return *route_model;
    if (node == nullptr) return Eigen::Matrix4d::Identity();

    node->UpdateTrans();
    return node->ModelTrans();
}

static std::optional<Eigen::Matrix4d> BuildChildRouteModel(
    SceneNode* parent, SceneNode* child, bool routed_child,
    const std::optional<Eigen::Matrix4d>& parent_route_model) {
    if (parent == nullptr || child == nullptr) return std::nullopt;
    if (!routed_child && !parent_route_model.has_value()) return std::nullopt;

    // Proxy routing is order-only in the physical SceneNode tree, but render-time transforms still
    // need to follow Wallpaper Engine's authored parent. Propagating a resolved route matrix lets
    // effect final passes use the same virtual parent chain that shader uniforms use later.
    return ResolveRouteModel(parent, parent_route_model) * child->GetLocalTrans();
}

static std::vector<OrderedRenderGraphChild> OrderedRenderGraphChildren(SceneNode* node,
                                                                       ExtraInfo& extra) {
    std::vector<OrderedRenderGraphChild> children;
    if (node == nullptr || extra.scene == nullptr) return children;

    std::unordered_set<SceneNode*> seen;
    size_t sequence = 0;
    for (auto& child : node->GetChildren()) {
        if (!child || !seen.insert(child.get()).second) continue;
        children.push_back(OrderedRenderGraphChild {
            .node = child.get(),
            .proxy = false,
            .sequence = sequence++,
        });
    }

    if (auto proxy_it = extra.scene->renderOrderProxyChildren.find(node);
        proxy_it != extra.scene->renderOrderProxyChildren.end()) {
        for (auto* proxy_child : proxy_it->second) {
            if (proxy_child == nullptr || !seen.insert(proxy_child).second) continue;
            children.push_back(OrderedRenderGraphChild {
                .node = proxy_child,
                .proxy = true,
                .sequence = sequence++,
            });
        }
    }

    // The scene tree still owns lifetime and transforms, but the render graph needs Wallpaper
    // Engine's authored layer order. Sorting only by known layer order and then by insertion
    // sequence keeps helper nodes deterministic without forcing every runtime node to have an
    // authored layer id.
    std::stable_sort(children.begin(), children.end(), [&extra](const auto& lhs, const auto& rhs) {
        const auto lhs_index = NodeLayerOrderIndex(lhs.node, extra);
        const auto rhs_index = NodeLayerOrderIndex(rhs.node, extra);
        if (lhs_index != rhs_index) return lhs_index < rhs_index;
        return lhs.sequence < rhs.sequence;
    });
    return children;
}

static void AddNodePass(SceneNode* node, std::string_view output, i32 imgId, ExtraInfo& extra,
                        std::function<bool()> should_execute = {}) {
    auto& rgraph = *extra.rgraph;
    auto& scene  = *extra.scene;

    if (node->Mesh() == nullptr) {
        return;
    }
    auto* mesh = node->Mesh();
    if (mesh->Material() == nullptr) {
        return;
    }
    auto* material = mesh->Material();

    std::string passName = material->name;
    rgraph.addPass<vulkan::CustomShaderPass>(
        passName,
        rg::PassNode::Type::CustomShader,
        [material, node, &output, &imgId, &rgraph, &scene, &extra,
         should_execute = std::move(should_execute)](
            rg::RenderGraphBuilder& builder, vulkan::CustomShaderPass::Desc& pdesc) {
            const auto& pass = builder.workPassNode();
            // Passing the live scene into the prepared pass lets resource refreshes resolve current
            // render-target dependencies directly, which is what keeps first-class text bridges and
            // ordinary effect passes on the same stable render-graph contract.
            pdesc.scene      = &scene;
            pdesc.node       = node;
            pdesc.execute_when_hidden = ShouldExecuteHiddenDependency(scene, node, output);
            pdesc.should_execute      = should_execute;
            pdesc.output     = output;
            if (IsAudioBarShaderNode(node)) {
                LOG_INFO("SceneAudioGraphBind: pass-id=%zu source-layer=%d node-ptr=%p node-id=%d name='%s' camera='%s' output='%s' execute-when-hidden=%s gated=%s",
                         static_cast<size_t>(pass.ID()),
                         imgId,
                         static_cast<void*>(node),
                         node->ID(),
                         node->Name().c_str(),
                         node->Camera().c_str(),
                         output.data(),
                         pdesc.execute_when_hidden ? "true" : "false",
                         pdesc.should_execute ? "true" : "false");
            } else if (IsSyntheticCompositeNode(node)) {
                const auto* bind_material = node->Mesh()->Material();
                const auto texture0 = bind_material != nullptr && !bind_material->textures.empty()
                    ? bind_material->textures[0].c_str()
                    : "<none>";
                LOG_INFO("SceneAudioCompositeGraphBind: pass-id=%zu source-layer=%d node-ptr=%p node-id=%d name='%s' camera='%s' output='%s' execute-when-hidden=%s gated=%s",
                         static_cast<size_t>(pass.ID()),
                         imgId,
                         static_cast<void*>(node),
                         node->ID(),
                         node->Name().c_str(),
                         node->Camera().c_str(),
                         output.data(),
                         pdesc.execute_when_hidden ? "true" : "false",
                         pdesc.should_execute ? "true" : "false");
                LOG_INFO("SceneAudioCompositeGraphBind: texture0='%s'", texture0);
            }
            CheckAndSetSprite(scene, pdesc, material->textures);
            for (usize i = 0; i < material->textures.size(); i++) {
                const auto&  url = material->textures[i];
                rg::TexNode* input { nullptr };
                if (url.empty()) {
                    pdesc.textures.emplace_back("");
                    continue;
                } else if (IsSpecLinkTex(url)) {
                    auto id = ParseLinkTex(url);
                    extra.link_info.push_back(
                        DelayLinkInfo { .id = pass.ID(), .link_id = id, .tex_index = (i32)i });
                    pdesc.textures.emplace_back("");
                    continue;
                } else {
                    rg::TexNode::Desc desc;
                    desc.key  = url;
                    desc.name = url;
                    desc.type = ! IsSpecTex(url) ? rg::TexNode::TexType::Imported
                                                 : rg::TexNode::TexType::Temp;
                    input     = builder.createTexNode(desc);
                    if (IsSpecTex(url)) builder.markVirtualWrite(input);
                    if (sstart_with(url, WE_MIP_MAPPED_FRAME_BUFFER))
                        extra.use_mipmap_framebuffer = true;
                }

                if (url == output) {
                    builder.markSelfWrite(input);
                    input = rg::addCopyPass(rgraph, input);
                }
                builder.read(input);
                pdesc.textures.emplace_back(input->key());
            }

            rg::TexNode* output_node { nullptr };
            output_node =
                builder.createTexNode(rg::TexNode::Desc { .name = output.data(),
                                                          .key  = output.data(),
                                                          .type = rg::TexNode::TexType::Temp },
                                      true);
            builder.write(output_node);
            if (ShouldPublishLayerLinkOutput(extra, imgId, output)) {
                extra.id_link_map[(usize)imgId] = output_node;
            }
        });
}

static void AddTextNodePass(SceneNode* node, std::string_view output, i32 imgId, ExtraInfo& extra) {
    auto& rgraph = *extra.rgraph;
    auto& scene  = *extra.scene;

    if (node == nullptr || node->Text() == nullptr) {
        return;
    }

    std::string pass_name = node->Name().empty() ? std::string("text") : node->Name();
    rgraph.addPass<vulkan::TextPass>(
        pass_name,
        rg::PassNode::Type::Text,
        [node, &output, imgId, &scene, &extra](
            rg::RenderGraphBuilder& builder, vulkan::TextPass::Desc& pdesc) {
            const auto& pass = builder.workPassNode();
            // Text is now emitted as its own render-graph pass. It shares the same constrained
            // hidden-dependency rule as mesh passes: invisible helper layers may render private
            // offscreen sources, but they must not composite text directly into `_rt_default`.
            pdesc.scene = &scene;
            pdesc.node = node;
            // Keep the authored layer id on the prepared pass so runtime text rerasters can
            // refresh the exact Clock/TextPass resources without broadening the dirty target set.
            pdesc.layer_id = imgId;
            pdesc.execute_when_hidden = ShouldExecuteHiddenDependency(scene, node, output);
            pdesc.output = output;

            auto* output_node =
                builder.createTexNode(rg::TexNode::Desc { .name = output.data(),
                                                          .key = output.data(),
                                                          .type = rg::TexNode::TexType::Temp },
                                      true);
            builder.write(output_node);
            if (ShouldPublishLayerLinkOutput(extra, imgId, output)) {
                extra.id_link_map[(usize)imgId] = output_node;
            }
            (void)pass;
        });
}

static void ToGraphPass(SceneNode* node, std::string_view inherited_output, i32 imgId,
                        ExtraInfo& extra, std::function<bool()> node_execute_gate = {},
                        bool routed_node = false,
                        std::optional<Eigen::Matrix4d> route_model = std::nullopt) {
    auto& scene = *extra.scene;

    if (node != nullptr && !routed_node) {
        const bool proxy_node = scene.renderOrderProxyNodes.count(node) != 0;
        const bool detached_source_node = scene.detachedEffectSourceNodes.count(node) != 0;
        if (proxy_node || detached_source_node) {
            // Root-owned proxy/source nodes are emitted through explicit authored-order routes.
            // Reaching them through the physical tree means the root traversal is at the wrong
            // sibling position, so skip this visit to avoid late duplicate composites.
            LOG_INFO("SceneRenderGraphNodeRouteSkip: layer=%d name='%s' reason='%s'",
                     NodeLayerId(scene, node),
                     node->Name().c_str(),
                     detached_source_node ? "detached-source" : "proxy");
            return;
        }
    }

    std::string_view         output = inherited_output;
    SceneImageEffectLayer*   imgeff { nullptr };
    const auto resolved_route_model = ResolveRouteModel(node, route_model);
    if (node != nullptr && !node->Camera().empty()) {
        auto camera_it = scene.cameras.find(node->Camera());
        if (camera_it != scene.cameras.end() && camera_it->second->HasImgEffect()) {
            imgeff = camera_it->second->GetImgEffect().get();
            output = imgeff->FirstTarget();
        }
    }

    if (node != nullptr && node->Mesh() != nullptr && node->Mesh()->Material() != nullptr) {
        AddNodePass(node, output, imgId, extra, node_execute_gate);
    }
    // Text is now a first-class scene primitive. Whenever a node owns text we emit the dedicated
    // text pass directly from that primitive, keeping the render graph aligned with the same
    // authoritative text object that parser and runtime updates mutate.
    if (node != nullptr && node->Text() != nullptr) {
        AddTextNodePass(node, output, imgId, extra);
    }

    if (node != nullptr) {
        if (auto detached_it = scene.detachedEffectSourceNodesByWorldNode.find(node);
            detached_it != scene.detachedEffectSourceNodesByWorldNode.end()) {
            for (auto* source_node : detached_it->second) {
                if (source_node == nullptr) continue;
                LOG_INFO("SceneRenderGraphDetachedSourceRoute: world-layer=%d source-layer=%d "
                         "world-name='%s' source-name='%s' output='%.*s'",
                         NodeLayerId(scene, node),
                         NodeLayerId(scene, source_node),
                         node->Name().c_str(),
                         source_node->Name().c_str(),
                         static_cast<int>(output.size()),
                         output.data());
                ToGraphPass(source_node,
                            output,
                            NodeLayerId(scene, source_node),
                            extra,
                            {},
                            true,
                            // Detached source nodes render through their own effect camera, but
                            // the image-effect final writer belongs to the visible world node.
                            // Forward the world route matrix so the final pass inherits virtual
                            // render-order parents while intermediate effect passes stay local.
                            std::optional<Eigen::Matrix4d> { resolved_route_model });
            }
        }
    }

    std::vector<OrderedRenderGraphChild> deferred_proxy_children;
    if (node != nullptr) {
        for (const auto& child : OrderedRenderGraphChildren(node, extra)) {
            if (child.node == nullptr) continue;
            if (child.proxy && imgeff != nullptr &&
                !IsEffectLocalProxyDependency(child.node, extra)) {
                // A render-order proxy edge only restores authored sibling order. It does not make
                // the proxied world-space node a real child of this image-effect source target, so
                // defer ordinary proxies until the parent effect has resolved back to the inherited
                // output space.
                LOG_INFO("SceneRenderGraphProxyChildDefer: parent-layer=%d proxy-layer=%d "
                         "inherited-output='%.*s' parent-effect-output='%.*s'",
                         NodeLayerId(scene, node),
                         NodeLayerId(scene, child.node),
                         static_cast<int>(inherited_output.size()),
                         inherited_output.data(),
                         static_cast<int>(output.size()),
                         output.data());
                deferred_proxy_children.push_back(child);
                continue;
            }

            if (child.proxy) {
                if (imgeff != nullptr) {
                    // Wallpaper Engine `dependencies` are effect-local inputs. These proxies must
                    // stay inside the parent effect phase because the parent shader samples their
                    // private source target while resolving the effect chain.
                    LOG_INFO("SceneRenderGraphProxyInlineEffectRoute: parent-layer=%d "
                             "proxy-layer=%d output='%.*s'",
                             NodeLayerId(scene, node),
                             NodeLayerId(scene, child.node),
                             static_cast<int>(output.size()),
                             output.data());
                } else {
                    LOG_INFO("SceneRenderGraphProxyChildRoute: parent-layer=%d proxy-layer=%d "
                             "name='%s' output='%.*s'",
                             NodeLayerId(scene, node),
                             NodeLayerId(scene, child.node),
                             child.node->Name().c_str(),
                             static_cast<int>(output.size()),
                             output.data());
                }
            }
            ToGraphPass(child.node,
                        output,
                        NodeLayerId(scene, child.node),
                        extra,
                        {},
                        child.proxy,
                        BuildChildRouteModel(node, child.node, child.proxy, route_model));
        }
    }

    if (imgeff != nullptr) {
        // Composite source nodes may now be transform-only containers whose children draw into the
        // effect source target. Resolving the effect after all descendants have emitted their passes
        // keeps those composite layers correct without requiring each child to manage effect timing.
        // Hidden dependency layers are not visible screen compositors. When a later effect samples
        // `_rt_imageLayerComposite_<id>`, the dependency's final authored effect must still resolve
        // into its private ping-pong target; otherwise the generic visible-layer rewrite would move
        // that final pass to `_rt_default`, where the hidden-layer execution gate correctly skips it.
        const auto resolved_effect_world_affine =
            Eigen::Affine3f(resolved_route_model.cast<float>());
        imgeff->ResolveEffect(scene.default_effect_mesh,
                              "effect",
                              IsOffscreenDependencyLayer(extra, imgId),
                              &resolved_effect_world_affine);

        for (usize i = 0; i < imgeff->EffectCount(); i++) {
            auto& eff     = imgeff->GetEffect(i);
            auto  cmdItor = eff->commands.begin();
            auto  cmdEnd  = eff->commands.end();
            int   nodePos = 0;
            auto  effect_visible_gate = [eff]() {
                return eff == nullptr || eff->LocalVisible();
            };
            auto effect_hidden_gate = [eff]() {
                return eff != nullptr && !eff->LocalVisible();
            };
            for (auto& effect_node : eff->nodes) {
                if (cmdItor != cmdEnd && nodePos == cmdItor->afterpos) {
                    rg::addCopyPass(*extra.rgraph,
                                    rg::createTexDesc(cmdItor->src),
                                    rg::createTexDesc(cmdItor->dst),
                                    effect_visible_gate);
                    cmdItor++;
                }
                ToGraphPass(effect_node.sceneNode.get(), effect_node.output, imgId, extra);
                nodePos++;
            }
            if (!eff->BypassSource().empty() && !eff->BypassTarget().empty() &&
                eff->BypassSource() != eff->BypassTarget()) {
                // Hidden effects must still advance the ping-pong chain. The shader and authored
                // effect commands become no-ops through their local visibility, then this gated copy
                // forwards the current input target to the effect output target so downstream
                // effects sample the current frame, not the stale texture from the last visible
                // frame.
                rg::addCopyPass(*extra.rgraph,
                                rg::createTexDesc(eff->BypassSource()),
                                rg::createTexDesc(eff->BypassTarget()),
                                effect_hidden_gate);
            }
        }

        if (imgeff->HasFinalComposite()) {
            // The final composite is deliberately a runtime fallback, not a replacement for visible
            // authored final effects. It only draws when the final effect is hidden; otherwise the
            // historical direct-to-output shader path remains the only screen writer for this chain.
            auto hidden_final_fallback_gate = [imgeff]() {
                return imgeff != nullptr && imgeff->ShouldRunFinalCompositeFallback();
            };
            ToGraphPass(&imgeff->FinalNode(),
                        inherited_output,
                        imgId,
                        extra,
                        hidden_final_fallback_gate,
                        false,
                        // The synthetic fallback is another detached final writer for the same
                        // image effect, so it must share the resolved route matrix used by the
                        // authored final output instead of recomputing from its physical tree.
                        std::optional<Eigen::Matrix4d> { resolved_route_model });
        }
    }

    for (const auto& child : deferred_proxy_children) {
        if (child.node == nullptr) continue;
        LOG_INFO("SceneRenderGraphProxyOutputRoute: parent-layer=%d proxy-layer=%d output='%.*s'",
                 NodeLayerId(scene, node),
                 NodeLayerId(scene, child.node),
                 static_cast<int>(inherited_output.size()),
                 inherited_output.data());
        ToGraphPass(child.node,
                    inherited_output,
                    NodeLayerId(scene, child.node),
                    extra,
                    {},
                    true,
                    BuildChildRouteModel(node, child.node, true, route_model));
    }
}

std::unique_ptr<rg::RenderGraph> wallpaper::sceneToRenderGraph(Scene& scene) {
    std::unique_ptr<rg::RenderGraph> rgraph = std::make_unique<rg::RenderGraph>();
    ExtraInfo                        extra { .rgraph = rgraph.get(), .scene = &scene };
    for (size_t index = 0; index < scene.layerOrder.size(); index++) {
        extra.layer_order_index[scene.layerOrder[index]] = index;
    }
    LOG_INFO("SceneRenderGraphOrderInit: layer-count=%zu proxy-parent-count=%zu proxy-node-count=%zu "
             "detached-anchor-count=%zu detached-source-count=%zu",
             scene.layerOrder.size(),
             scene.renderOrderProxyChildren.size(),
             scene.renderOrderProxyNodes.size(),
             scene.detachedEffectSourceNodesByWorldNode.size(),
             scene.detachedEffectSourceNodes.size());
    ToGraphPass(scene.sceneGraph.get(), SpecTex_Default, scene.sceneGraph->ID(), extra);

    for (auto& info : extra.link_info) {
        if (! exists(extra.id_link_map, info.link_id)) {
            LOG_ERROR("link tex %d not found", info.link_id);
            continue;
        }
        rgraph->afterBuild(
            info.id, [&rgraph, &extra, &info](rg::RenderGraphBuilder& builder, rg::Pass& rgpass) {
                auto& pass = static_cast<vulkan::CustomShaderPass&>(rgpass);

                auto* link_tex_node = extra.id_link_map.at(info.link_id);
                auto  copy_desc     = link_tex_node->genDesc();
                copy_desc.key       = GenLinkTex((idx)info.link_id);
                copy_desc.name      = copy_desc.key;

                auto new_in = rg::addCopyPass(*rgraph, link_tex_node, &copy_desc);
                builder.read(new_in);
                pass.setDescTex((u32)info.tex_index, new_in->key());
                return true;
            });
    }

    if (extra.use_mipmap_framebuffer) {
        rg::addCopyPass(*rgraph,
                        rg::TexNode::Desc { .name = SpecTex_Default.data(),
                                            .key  = SpecTex_Default.data(),
                                            .type = rg::TexNode::TexType::Temp },
                        rg::TexNode::Desc { .name = WE_MIP_MAPPED_FRAME_BUFFER.data(),
                                            .key  = WE_MIP_MAPPED_FRAME_BUFFER.data(),
                                            .type = rg::TexNode::TexType::Temp });
    }

    return rgraph;
}

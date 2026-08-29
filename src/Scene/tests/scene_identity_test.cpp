// Contract tests for the SceneObject identity model: one authored layer owns all of its
// per-layer state, node ids are the only layer back-reference, and render-order proxy
// routing derives from authored parent bindings. These are the invariants the golden-frame
// acceptance runs rely on; they are plain assert tests in the style of the daemon tests.
#include "Scene/Scene.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"

#include <cassert>
#include <cstdio>
#include <memory>

using namespace wallpaper;

static void TestLayerIdForNode() {
    Scene scene;
    assert(scene.LayerIdForNode(nullptr) == 0);

    SceneNode helper;
    assert(scene.LayerIdForNode(&helper) == 0); // helper nodes keep the default id 0

    SceneNode handle;
    handle.ID() = 42;
    assert(scene.LayerIdForNode(&handle) == 42);
}

static void TestIdentityLifecycle() {
    Scene scene;
    assert(scene.FindSceneObject(7) == nullptr);

    auto& object = scene.EnsureSceneObject(7);
    assert(&scene.EnsureSceneObject(7) == &object); // idempotent
    assert(scene.FindSceneObject(7) == &object);

    // Per-layer records live on the object and die with it.
    scene.SetLayerSoundHandle(7, 0); // a failed mount's 0 handle stays "present"
    assert(scene.GetLayerSoundHandle(7).has_value());
    scene.SetLayerInitialConfigJson(7, "{}");
    assert(scene.GetLayerInitialConfigJson(7) != nullptr);
    scene.MarkLayerOffscreenDependencySource(7);
    assert(scene.IsLayerOffscreenDependencySource(7));

    scene.DestroySceneObject(7);
    assert(scene.FindSceneObject(7) == nullptr);
    assert(!scene.GetLayerSoundHandle(7).has_value());
    assert(scene.GetLayerInitialConfigJson(7) == nullptr);
    assert(!scene.IsLayerOffscreenDependencySource(7));
}

static void TestLayerNodeSlotTriState() {
    Scene scene;
    SceneNode handle;
    handle.ID() = 3;

    assert(!scene.HasLayerNodeSlot(3));
    assert(scene.GetLayerNode(3) == nullptr);

    // Registered-but-null: the layer exists while its handle is temporarily absent.
    scene.SetLayerNode(3, nullptr);
    assert(scene.HasLayerNodeSlot(3));
    assert(scene.GetLayerNode(3) == nullptr);
    assert(scene.FindLayerIdByNode(nullptr) == 0); // null queries must not match null slots

    scene.SetLayerNode(3, &handle);
    assert(scene.GetLayerNode(3) == &handle);
    assert(scene.FindLayerIdByNode(&handle) == 3);

    scene.ClearAllLayerNodeSlots();
    assert(!scene.HasLayerNodeSlot(3));
    assert(scene.FindLayerIdByNode(&handle) == 0);
}

static void TestDeferredRuntimeKind() {
    Scene scene;
    assert(!scene.IsLayerDeferredRuntime(5));

    scene.MarkLayerDeferredRuntime(5, SceneDeferredRuntimeKind::Text);
    assert(scene.IsLayerDeferredRuntime(5));
    assert(scene.IsLayerDeferredRuntime(5, SceneDeferredRuntimeKind::Text));
    assert(!scene.IsLayerDeferredRuntime(5, SceneDeferredRuntimeKind::Image));

    // A stale clear for another kind is a no-op.
    scene.ClearLayerDeferredRuntime(5, SceneDeferredRuntimeKind::Image);
    assert(scene.IsLayerDeferredRuntime(5, SceneDeferredRuntimeKind::Text));

    scene.ClearLayerDeferredRuntime(5, SceneDeferredRuntimeKind::Text);
    assert(!scene.IsLayerDeferredRuntime(5));

    scene.MarkLayerDeferredRuntime(6, SceneDeferredRuntimeKind::Image);
    assert(scene.DeferredRuntimeLayerCount(SceneDeferredRuntimeKind::Image) == 1);
    assert(scene.DeferredRuntimeLayerIds(SceneDeferredRuntimeKind::Image).size() == 1);
}

static void TestRuntimeResourceLists() {
    Scene scene;
    assert(scene.GetLayerRuntimeLights(9).empty());
    assert(scene.GetLayerRuntimeNodes(9).empty());
    assert(scene.GetLayerRuntimeParticleSubsystems(9).empty());

    SceneNode draw_handle;
    draw_handle.ID() = 9;
    scene.AddLayerRuntimeNode(9, &draw_handle);
    assert(scene.GetLayerRuntimeNodes(9).size() == 1);

    // The list is cleared at the exact point the nodes are freed; the identity survives.
    scene.ClearLayerRuntimeNodes(9);
    assert(scene.GetLayerRuntimeNodes(9).empty());
    assert(scene.FindSceneObject(9) != nullptr);
}

static void TestCameraLayerState() {
    Scene scene;
    assert(!scene.HasCameraLayers());
    assert(scene.FindCameraLayerState(4) == nullptr);

    Scene::CameraLayerRuntimeState state;
    state.zoom = 2.0;
    const bool first = scene.FindCameraLayerState(4) == nullptr;
    scene.SetCameraLayerState(4, state);
    if (first) scene.cameraLayerOrder.push_back(4);

    assert(scene.HasCameraLayers());
    assert(scene.FindCameraLayerState(4) != nullptr);
    assert(scene.FindCameraLayerState(4)->zoom == 2.0);

    // Registration overwrites the record without duplicating the order entry.
    state.zoom = 3.0;
    const bool again = scene.FindCameraLayerState(4) == nullptr;
    scene.SetCameraLayerState(4, state);
    if (again) scene.cameraLayerOrder.push_back(4);
    assert(scene.cameraLayerOrder.size() == 1);
    assert(scene.FindCameraLayerState(4)->zoom == 3.0);
}

static void TestEffectBridgeOwnership() {
    Scene scene;
    SceneNode world;
    world.ID() = 11;

    auto bridge = std::make_shared<SceneImageEffectLayer>(&world, 100.0f, 100.0f, "a", "b");
    bridge->SetBridgeCameraName("cam11");
    bridge->AddRuntimeCameraName("cam11");
    scene.EnsureSceneObject(11).SetImageEffectLayer(bridge);

    assert(scene.FindImageEffectLayer(11) == bridge.get());
    assert(scene.FindImageEffectLayer(11)->BridgeCameraName() == "cam11");
    assert(scene.FindImageEffectLayer(12) == nullptr);

    // The object owns the bridge: dropping the identity drops the resolution result.
    std::weak_ptr<SceneImageEffectLayer> watch = bridge;
    bridge.reset();
    assert(!watch.expired());
    scene.DestroySceneObject(11);
    assert(watch.expired());
}

static void TestRenderOrderProxyDerivation() {
    Scene scene;
    auto parent_node = std::make_shared<SceneNode>();
    auto routed_child = std::make_shared<SceneNode>();
    auto attached_child = std::make_shared<SceneNode>();
    parent_node->ID()    = 1;
    routed_child->ID()   = 2;
    attached_child->ID() = 3;

    // Routed children stay root-owned for transform correctness; attached children live
    // under their physical parent.
    scene.sceneGraph->AppendChild(parent_node);
    scene.sceneGraph->AppendChild(routed_child);
    parent_node->AppendChild(attached_child);

    scene.SetLayerNode(1, parent_node.get());
    scene.SetLayerNode(2, routed_child.get());
    scene.SetLayerNode(3, attached_child.get());
    scene.SetLayerParentBinding(2, 1, "");
    scene.SetLayerParentBinding(3, 1, "");
    scene.layerOrder = { 1, 2, 3 };

    assert(!scene.IsRenderOrderProxyNode(parent_node.get()));  // no parent binding
    assert(scene.IsRenderOrderProxyNode(routed_child.get()));  // root-owned + routed binding
    assert(!scene.IsRenderOrderProxyNode(attached_child.get())); // physically parented

    const auto children = scene.RenderOrderProxyChildrenOf(parent_node.get());
    assert(children.size() == 1 && children[0] == routed_child.get());

    // An attachment binding routes through the puppet machinery, not render order.
    scene.SetLayerParentBinding(2, 1, "bone_3");
    assert(!scene.IsRenderOrderProxyNode(routed_child.get()));
    assert(scene.RenderOrderProxyChildrenOf(parent_node.get()).empty());

    // Reparenting by binding alone re-routes the derived answer with no table upkeep.
    scene.SetLayerParentBinding(2, 1, "");
    assert(scene.IsRenderOrderProxyNode(routed_child.get()));
}

int main() {
    TestLayerIdForNode();
    TestIdentityLifecycle();
    TestLayerNodeSlotTriState();
    TestDeferredRuntimeKind();
    TestRuntimeResourceLists();
    TestCameraLayerState();
    TestEffectBridgeOwnership();
    TestRenderOrderProxyDerivation();
    std::printf("scene_identity_test: all checks passed\n");
    return 0;
}

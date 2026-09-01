# Runtime visibility and named render-target design

本文记录 Scene 的对象可见性、effect 可见性和 named render target 行为。隐藏状态不得改变 authored 拓扑。

## 契约

- 每个成功解析的 authored object 都按 JSON 顺序进入 Scene。
- layer `visible` 是 draw-time 状态；隐藏 layer 仍保留对象和资源。
- effect resource、pass 和 FBO 先完整创建，随后才应用 scene effect instance JSON；effect `visible` 只改变执行位和 effect 计数。
- destination RT 按生成字符串命名并按名字全局 intern，第一次创建的描述符生效。
- 普通 effect FBO 使用 authored name；`unique=true` 才追加 effect id。

## Layer visibility

解析阶段不会因为 layer 初始隐藏而跳过 image、text、particle、model、shape、light 或其 effect chain。`SceneObject::LocalVisible()` 保存 layer-local 位，`Scene::ApplyLayerVisibility()` 把父链结果传播到所有 runtime draw node。

RenderGraph 始终包含完整 authored pass 集。每个 destination draw、clear 和 copy 在执行时检查 layer gate：

- 可见 layer 正常提交 destination work。
- 普通隐藏 layer 跳过属于自己的 destination work。
- 被可见内容采样的隐藏 dependency layer 仍可写私有 offscreen target，但不能写主输出。

因此，`visible` 更新不标记 topology dirty，不创建或销毁 pass，也不释放该 layer 的纹理、FBO、mesh 或 pipeline。

## Effect instance visibility

`WPImageEffect::FromJson()` 先读取共享 effect resource 及 scene pass override，再保存 instance `visible`。解析器随后完成全部 `SceneImageEffect` node、command 和 FBO，最后解析初始 user condition 并调用 `SetLocalVisible()`。

对象 materialization 完成后，`WPSceneParserBindings` 为 object-form `visible` 注册 effect target。后续 user property、SceneScript 或 property animation 只更新 `SceneImageEffect::LocalVisible()`；不修改 effect 数量或资源所有权。

RenderGraph 为每个 effect 保留两条互斥执行 gate：

- visible gate 执行 authored commands 和 shader passes；
- hidden gate 只推进 effect ping-pong 输入到输出，使后续 effect 读取当前帧内容。

source-less compose helper 的最终发布还会检查是否存在当前可见的 runtime-controlled source effect，防止隐藏 generator 的空目标或旧内容被合成到画面。

## Named render targets

Image/text destination target 的基础名字是 `sc.W.H.b` 与 `sc.W.H.n`。只统计 parent chain 中与该基础名字相同的 destination slot；计数非零时把十进制计数追加到名字。这样同尺寸 sibling 会共享 target，而嵌套冲突按父链同名计数改名。

Effect FBO 使用以下名字：

- `unique=false`：原始 FBO name；
- `unique=true`：`<name>_<effect-id>`。

`InternNamedRenderTarget()` 使用名字作为唯一 key，并保留第一次注册的尺寸、采样和绑定描述。后出现的多语言分支不会用自己的描述符覆盖已有条目。

Named target 可以被多个完整 materialized layer 共同引用。动态 layer 删除只有在 retained-resource 扫描确认没有其他 owner 时才移除该名字；layer hide 不参与资源回收。场景切换和 renderer destroy 仍负责整场资源清理。

## 图与资源生命周期

稳定可见性更新只改 gate。真正会改变 RenderGraph 结构的操作包括动态 layer 创建/删除，以及 volumetrics、shadow、postprocessing 等渲染功能重配置。结构变化仍可按 pass identity 复用已有 Vulkan pass；这与 layer hide 无关。

Texture upload 和 render-target clear 仍随 render command buffer 提交，已有 pipeline/resource cache 仍服务于场景加载和真正的结构更新。它们不再被解释为隐藏 layer 的 materialization 或剪枝机制。

## 主要代码入口

- `WPSceneParser::Parse()` / `AddWPObject()` / `FillSceneObjectIdentity()`
- `ParseImageObj()` / `ParseTextObj()` / `ParseParticleObj()` / `ParseModelObj()`
- `WPImageEffect::FromJson()`
- `RegisterEffectVisibilityBindings()`
- `Scene::SetLayerLocalVisibility()` / `Scene::SetEffectLocalVisibilityById()`
- `SceneImageEffectLayer::ResolveEffect()` / `ShouldRunFinalComposite()`
- `sceneToRenderGraph()` / `ShouldExecuteLayerDestination()`
- `SceneDestinationRenderTargetNames()` / `InternNamedRenderTarget()`

## 心智模型

Layer 或 effect 的 `visible` 是已经存在对象上的执行位：对象、effect、pass、FBO 和 named target 先完整存在，帧执行时再决定哪些 destination work 可以提交。资源身份来自 authored 名称与 destination / FBO 命名规则，不来自当前是否可见。

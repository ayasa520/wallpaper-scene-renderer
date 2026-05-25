# Runtime visibility residency design

本文总结当前未提交实现中，Scene 子模块如何把运行时可见 / 不可见切换改成非阻塞路径，并分析它和游戏引擎渲染架构之间的相似性。

## 结论

这套实现的核心不是简单地“隐藏时不画，显示时再画”，而是把一个层拆成三种不同的生命周期：

- 逻辑生命周期：SceneScript、用户属性、动画、父子层级和 authored visibility 仍然存在并继续更新。
- 图生命周期：RenderGraph 只保留当前必须参与渲染或被可见层采样的 pass；普通隐藏层会从 resident graph 中剪掉。
- 资源驻留生命周期：纹理、视频解码器、render target、framebuffer、buffer suballocation 可以按层精确释放；pipeline state 可以继续保温。

因此，false -> true 不再等价于“重新加载整张场景”，true -> false 也不再等价于“清空所有缓存”。它更像一次小规模的 streaming residency transition：可见性改变只提交资源驻留意图，渲染线程在后续帧按预算推进准备工作。

## 旧阻塞点

原来的重路径大致是：

1. 可见性或图结构改变后标记 render graph dirty。
2. 拓扑重建时销毁旧 pass。
3. 清空 TextureCache / VideoTextureCache。
4. 重新创建大量 texture、video、framebuffer、pipeline、descriptor、vertex/index/UBO buffer。
5. 额外提交 upload command，并 `WaitIdle()` 等待设备完成。

这个模型对场景切换是保守的，但对运行时可见性切换太重。一个隐藏分支变可见时，会把大量无关资源也拖入同一帧同步准备；一个分支变隐藏时，也可能因为全局 cache clear 破坏无关层的 warm state。

## 新的切换链路

### 1. Scene 只记录逻辑状态和资源意图

Scene 现在新增了异步解析缓存和释放队列：

- `m_parsed_image_cache`
- `m_pending_parsed_images`
- `m_failed_parsed_images`
- `pendingStaticTextureReleaseKeys`
- `pendingVideoTextureReleaseKeys`
- `pendingRenderTargetReleaseKeys`

静态图片解析通过 `RequestParsedImageAsync()` 发起后台解析，`GetParsedImageIfReady()` 只做非阻塞轮询，`ParseImageBlockingCached()` 保留给必须同步完成的路径。

可见性写入发生在 SceneScript host 的 layer `visible` 分支。流程是：

1. 保存切换前的 effective visibility。
2. 写入 layer-local visibility。
3. 如果变为 visible，先 materialize deferred 子树。
4. 重新应用 layer visibility。
5. 如果 effective visibility 发生变化，标记 render graph topology dirty。
6. 变 hidden 时，把该层树独占的资源 key 放入 pending release 队列。
7. 变 visible 时，从 pending release 队列取消该层树的资源释放。

这里的关键是“独占资源”判断。实现会遍历该层的 runtime nodes、effect nodes、text primitive glyph pages、object runtime render targets，把材质纹理、视频纹理、render target 分类收集出来。然后再收集仍然可见、或作为 offscreen dependency 需要保留驻留的层资源。只有不被这些 retained resources 引用的 key 才会进入释放队列。

这避免了两个问题：

- 共享资源不会因为某个隐藏分支被误删。
- 同一帧 hidden 后又 visible 的层可以取消 pending release，不会出现先删后用。

### 2. Deferred layer 先 materialize 逻辑对象，但不强制 GPU 驻留

启动时，hidden image / particle / text layer 仍可先作为 placeholder 存在。初始化脚本、用户属性、媒体状态都应用后，会调用 `MaterializeDeferredRuntimeLayersForResidency()`。

这个 warm-up 做的是 CPU / runtime identity materialization：

- 把 deferred image / particle / text layer 从初始 JSON 构建成真实 runtime object。
- 保留 placeholder 上已经被脚本修改过的 transform / name / parent binding。
- 解析时临时用 `visible=true` 满足 parser 的构建入口，然后恢复 Scene 中真实的 logical visibility。
- 注册新生成的 binding / animation / script registration。

这样后续 visible=true 不必在同一帧重新从 JSON 构建完整对象。与此同时，RenderGraph 的普通构建仍会剪掉不可见层，所以这些隐藏层不会因为已经 materialize 就立刻占满 GPU 资源。

### 3. RenderGraph 按 residency 剪枝，而不是全量常驻

`sceneToRenderGraph()` 现在会调用 `ShouldEmitLayerNodeForResidency()`：

- root、无 layer id 节点保留。
- visible layer 保留。
- offscreen dependency layer 即使 hidden 也保留，因为可见 effect 可能采样它的私有输出。
- 普通 hidden layer 直接跳过，不发出该分支 pass。

另外有一个 `sceneToPipelineWarmupRenderGraph()` 变体，它会包含 hidden layer。这个图不用于实际驻留，而是用于 pipeline warm-up。

### 4. VulkanRender 对新旧图做 pass diff

拓扑 dirty 时不再把它当成场景切换。`compileRenderGraph()` 会：

1. 从旧 resident graph 中取出已有 pass，并按 `residencyKey()` 建索引。
2. 对新 graph 中的 pass 尝试复用旧 pass。
3. 复用成功时只吸收新的 declarative graph state，例如 runtime gate、texture key、output。
4. 旧图中不再出现的 pass 才调用 `destory()` 退休。
5. 退休 pass 释放 framebuffer、descriptor-bound image refs、buffer suballocation，但不会清空 pipeline cache。
6. 旧 pass 都解除引用后，再 drain Scene 的 pending release 队列。

这个流程把“graph topology 改变”和“全局资源销毁”解耦。普通 visible toggle 只影响相关 pass 和相关资源。

### 5. 新 pass 分帧 deferred prepare

如果已经存在 resident graph，运行时拓扑变化中新出现的非 CopyPass pass 不会立即全部 prepare。它们会进入 `m_deferred_prepare_indices` 队列。

每帧 `drawFrame()` 前调用 `processDeferredGraphPreparation()`，受两个预算限制：

- 每帧最多准备 `kDeferredPrepareMaxPassesPerFrame` 个 pass。
- 每批最多花 `kDeferredPrepareFrameBudgetMs` 毫秒。

pass 可以先调用 `requestDeferredPrepareResources()`，如果静态纹理还没有准备好，就返回 `Waiting`。这会让渲染线程继续画已经 prepared 的旧内容，而不是卡在解析或上传上。

### 6. TextureCache 从同步上传改成 staged upload

TextureCache 的变化是非阻塞切换的另一半：

- `CreateTex()` 不再马上提交 copy command 并 `WaitIdle()`。
- 静态图片上传变成 `TextureCachePendingImageUpload`，在下一帧主 render command buffer 中通过 `RecordUploads()` 录入。
- upload staging buffer 会留在 `m_inflight_image_uploads`，等 frame fence 完成后由 `RetireCompletedUploads()` 释放。
- render target 的透明黑初始化也从立即提交改成 `TextureCachePendingRenderTargetClear`，随下一帧 command buffer 执行。
- `StageTexUploads()` 支持按 slot / byte budget 推进 sprite 或多 slot 纹理的 GPU 驻留。
- `FindTex()` 允许 pass 在不触发同步解析 / 创建的情况下查询现有驻留。

对 deferred prepare 来说，CustomShaderPass 会先请求 Scene 解析图片，再让 TextureCache 按预算 stage upload。如果纹理没有 ready，它不会掉回阻塞式 `Parse()` / `CreateTex()` 路径。

### 7. Pipeline state cache 保温不可变 PSO

GraphicsPipeline 新增了 `GraphicsPipelineStateCache`。pass 提供 render-pass compatibility key，GraphicsPipeline 再把 shader SPIR-V、descriptor layout、vertex input、blend、depth、raster、multisample 等状态混进最终 cache key。

结果是：

- hidden 时可以释放 framebuffer、texture residency、buffer suballocation。
- pipeline、pipeline layout、descriptor set layout、render pass object 可以通过 cache 保留。
- visible 时如果状态匹配，可以 cache hit，避免重新 `vkCreateGraphicsPipelines()`。

启动时还会构建包含隐藏层的 warmup graph，只 warm up pipeline，不绑定 layer-owned texture / framebuffer / mesh buffer。这和实际驻留图分离，避免隐藏层占 GPU 内存。

## 为什么它是非阻塞的

非阻塞来自几个边界同时成立：

- CPU 解析可异步：图片解析进入 Scene 的 future cache，render thread 轮询 ready 状态。
- GPU 上传可延迟：texture upload、render target clear 进入下一帧 command buffer，而不是 compile 阶段 submit + WaitIdle。
- pass prepare 可分帧：新 pass 进入 deferred queue，每帧按预算推进。
- 旧内容可继续画：已经 prepared 的 pass 保持 resident，未准备好的新 pass 暂时不执行。
- 资源释放有顺序：先让旧 pass destory 并掉引用，再释放 pending texture / video / render target。
- 同帧反转可取消：visible=true 可以取消 hidden 时排队的 release intent。

所以运行时切换不再把整个场景拖到“加载屏幕式”的同步路径上，而是变成逐步收敛到新驻留集合。

## 与游戏引擎的关系

这套设计明显受到现代游戏引擎思路启发，尤其是以下模式：

- Resource residency / streaming：资源不是由逻辑对象存在与否直接决定，而是按可见性、依赖关系和预算进入或退出 GPU 驻留。
- PSO cache：不可变 pipeline state 与纹理 / framebuffer / mesh buffer 驻留分开，隐藏对象释放内存但保留 pipeline warm state。
- Render graph diffing：运行时图变化通过新旧 graph 对比复用稳定 pass，而不是全量 teardown。
- Frame-budgeted work：昂贵的 prepare / upload work 被拆进后续帧，防止单帧尖峰。
- Transient resource lifetime：render target 的 share-ready 延迟到 graph activation 完成，避免 deferred graph 尚未完全 resident 时发生错误 alias。
- Visibility culling as residency policy：不可见分支被剪出 resident graph，但 offscreen dependency 这种被可见内容采样的隐藏分支仍保留。

它不是照搬某个特定引擎 API，而是把 Unreal / Unity / Frostbite / id Tech / 自研引擎中常见的渲染资产流式加载、PSO 预热、render graph transient resource 管理思想，移植到 Wallpaper Engine scene renderer 的约束下。

更准确地说，这不是游戏引擎式“场景对象 culling”本身，而是游戏引擎式“可见性驱动的资源驻留管理”。区别在于 Hanabi 的输入是 Wallpaper Engine scene.json、SceneScript 和 Vulkan render graph，不是 ECS 或传统关卡 streaming volume。

## 边界条件

这套实现仍然保留了几条安全边界：

- 场景切换和 renderer destroy 仍可以 full cache teardown。
- ordinary topology rebuild 不清空所有 texture / video / pipeline cache。
- CopyPass 作为轻量依赖会优先 prepare，避免复用 shader pass 时采样的 copy render target 尚未注册。
- offscreen dependency hidden layer 不能直接写 `_rt_default`，只允许保持私有 offscreen 输出。
- render target alias 的 `MarkShareReady()` 在 deferred graph activation 期间会延迟 replay。
- video texture release 走 VideoTextureCache 的 per-key release，而不是全局 clear。
- synthetic image parser 加锁，避免异步解析缓存访问到注册表时发生竞争。

## 主要代码入口

- `Scene::RequestParsedImageAsync()` / `GetParsedImageIfReady()` / `ParseImageBlockingCached()`
- `ApplyLayerPropertyValue(..., "visible", ...)`
- `QueueHiddenLayerTreeResourceRelease()` / `CancelLayerTreeResourceRelease()`
- `WPSceneScriptHost::MaterializeDeferredRuntimeLayersForResidency()`
- `sceneToRenderGraph()` / `sceneToPipelineWarmupRenderGraph()`
- `VulkanRender::compileRenderGraph()`
- `VulkanRender::processDeferredGraphPreparation()`
- `TextureCache::StageTexUploads()` / `RecordUploads()` / `RetireCompletedUploads()`
- `GraphicsPipelineStateCache`
- `CustomShaderPass::requestDeferredPrepareResources()` / `prepareDeferred()` / `warmupPipeline()`

## 简短心智模型

可以把一次可见性切换理解成：

1. SceneScript 改变 logical visibility。
2. Scene 计算哪些资源应该退出或重新进入 resident set。
3. RenderGraph 生成新的可见驻留图。
4. VulkanRender 复用旧 pass，退休消失的 pass，排队新 pass。
5. TextureCache / VideoTextureCache 按 key 释放不再被引用的资源。
6. 新资源按帧预算解析、分配、上传、prepare。
7. pipeline cache 尽量让重新显示的第一帧避开 PSO 编译成本。

这就是“非阻塞切换”的本质：可见性变化立刻改变逻辑和目标拓扑，但昂贵资源工作被拆成可取消、可复用、可分帧推进的驻留任务。

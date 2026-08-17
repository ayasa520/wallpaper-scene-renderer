#pragma once

#include "Interface/IShaderValueUpdater.h"
#include "Resource.hpp"
#include "Scene/Scene.h"
#include "SpriteAnimation.hpp"
#include "Vulkan/Device.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/Shader.hpp"
#include "Vulkan/StagingBuffer.hpp"
#include "VulkanPass.hpp"

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace wallpaper::vulkan
{

struct ShaderDrawRequest {
    Scene*                scene { nullptr };
    SceneNode*            node { nullptr };
    int32_t               layer_id { 0 };
    bool                  execute_when_hidden { false };
    std::function<bool()> should_execute;
    std::vector<std::string> textures;
    std::string              output;
    AlphaWritePolicy         alpha_write_policy { AlphaWritePolicy::Preserve };
    bool                     premultiplied_source_blend { false };
    bool                     clear_before_draw { false };
    std::string              camera_override;
    bool                     use_active_camera_for_uniforms { false };
    bool                     use_active_camera_for_parallax { false };
    sprite_map_t             sprites_map;
    bool                     model_pass { false };
    bool                     depth_test { false };
    bool                     depth_write { false };
    bool                     clear_depth { true };
    bool                     depth_greater { false };
    float                    depth_clear { 1.0f };
};

struct ShaderDrawState {
    std::vector<ImageSlotsRef> vk_textures;
    std::vector<i32>           vk_tex_binding;
    ImageParameters            vk_output;
    ImageParameters            vk_resolve;
    VkSampleCountFlagBits      sample_count { VK_SAMPLE_COUNT_1_BIT };
    bool                       resolve_msaa { false };
    bool                       alpha_to_coverage { false };

    bool                          dyn_vertex { false };
    bool                          force_dyn_upload { false };
    std::vector<StagingBufferRef> vertex_bufs;
    StagingBufferRef              index_buf;
    StagingBufferRef              ubo_buf;

    VkClearValue        clear_value;
    VkClearValue        depth_clear_value;
    bool                blending { false };
    vvk::Framebuffer    fb;
    VmaImageParameters* depth_stencil_image_ref { nullptr };
    PipelineParameters  pipeline;
    u32                 draw_count { 0 };
    u32                 index_element_bytes { 2 };

    std::function<void()> update_dynamic_mesh_op;
    std::function<void()> update_op;
};

// The graph-facing request and prepared Vulkan state are deliberately separate. The core keeps a
// combined internal object only so preparation helpers can receive one stable reference without
// leaking prepared resources back through RenderGraph::addPass().
struct ShaderDrawData : ShaderDrawRequest, ShaderDrawState {};

struct ShaderDrawRenderState {
    VkPipelineColorBlendAttachmentState color_blend {};
    VkAttachmentLoadOp                  color_load_op { VK_ATTACHMENT_LOAD_OP_DONT_CARE };
};

struct ShaderDrawAttachmentDescription {
    VkFormat            format { VK_FORMAT_UNDEFINED };
    VkAttachmentLoadOp  depth_load_op { VK_ATTACHMENT_LOAD_OP_DONT_CARE };
    VkAttachmentStoreOp depth_store_op { VK_ATTACHMENT_STORE_OP_DONT_CARE };
    VkAttachmentLoadOp  stencil_load_op { VK_ATTACHMENT_LOAD_OP_DONT_CARE };
    VkAttachmentStoreOp stencil_store_op { VK_ATTACHMENT_STORE_OP_DONT_CARE };
    VkImageLayout       initial_layout { VK_IMAGE_LAYOUT_UNDEFINED };
    VkImageLayout       final_layout { VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    std::string_view    cache_tag;

    bool enabled() const { return format != VK_FORMAT_UNDEFINED; }
};

struct ShaderDrawPipelineContext {
    const ShaderDrawData& data;
    const SceneMesh&      mesh;
    const SceneMaterial&  material;
    const DescriptorSetInfo& descriptor_info;
    const std::vector<VkVertexInputBindingDescription>& binding_descriptions;
    const std::vector<VkVertexInputAttributeDescription>& attribute_descriptions;
    const ShaderDrawRenderState& render_state;
};

struct ShaderDrawRecordContext {
    const ShaderDrawData& data;
    const Device&         device;
    RenderingResources&   resources;
    std::function<void(VkPipelineLayout)> push_visible_descriptors;
};

std::optional<vvk::RenderPass> CreateShaderDrawRenderPass(
    const vvk::Device&, VkFormat, VkAttachmentLoadOp, VkImageLayout,
    const ShaderDrawAttachmentDescription& = {},
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, bool resolve_msaa = false);
std::string ShaderDrawPipelineCompatibilityKey(
    VkAttachmentLoadOp, bool model_pass, VkAttachmentLoadOp model_depth_load_op,
    const ShaderDrawAttachmentDescription& = {},
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, bool resolve_msaa = false);
void UpdateShaderDrawUniform(StagingBuffer*, const StagingBufferRef&,
                             const ShaderReflected::Block&, std::string_view,
                             const ShaderValue&);

class ShaderDrawExtension {
public:
    virtual ~ShaderDrawExtension() = default;

    virtual bool configure(const Device&, const ShaderDrawData&, const SceneMesh&) = 0;
    virtual std::vector<std::string_view> resourceTextures(const SceneMesh&) const = 0;
    virtual bool refreshTextures(Scene&, const Device&, const ShaderDrawData&) = 0;
    virtual ShaderDrawAttachmentDescription attachmentDescription() const = 0;
    virtual VmaImageParameters* acquireAttachment(const Device&, RenderingResources&,
                                                  const ShaderDrawData&) = 0;
    virtual bool preparePipelines(const Device&, RenderingResources&,
                                  const ShaderDrawPipelineContext&) = 0;
    virtual void updateUniform(StagingBuffer*, std::string_view, const ShaderValue&) = 0;
    virtual void initializeUniforms(StagingBuffer*) = 0;
    virtual void recordIndexed(const ShaderDrawRecordContext&) = 0;
    virtual void destroy(RenderingResources&) = 0;
};

class ShaderDrawCore {
public:
    explicit ShaderDrawCore(const ShaderDrawRequest&);

    void setExtension(ShaderDrawExtension* extension) { m_extension = extension; }

    bool prepare(Scene&, const Device&, RenderingResources&);
    bool prepareDeferred(Scene&, const Device&, RenderingResources&);
    bool refreshResources(Scene&, const Device&, RenderingResources&);
    void dropOutputFramebuffers();
    void updateBeforeUpload();
    DeferredPrepareResourcesState requestDeferredPrepareResources(Scene&, const Device&);
    bool warmupPipeline(Scene&, const Device&, RenderingResources&);
    void execute(const Device&, RenderingResources&);
    void destroy(RenderingResources&);

    std::string residencyKey(std::string_view pass_kind) const;
    bool canReuseForResidency(const ShaderDrawCore&) const;
    void absorbResidencyGraphState(const ShaderDrawCore&);
    bool referencesRenderTarget(std::string_view) const;
    void setTexture(u32 index, std::string_view texture_key);

    const ShaderDrawData& data() const { return m_desc; }

private:
    ShaderDrawData       m_desc;
    ShaderDrawExtension* m_extension { nullptr };
};

} // namespace wallpaper::vulkan

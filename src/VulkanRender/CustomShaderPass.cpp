#include "CustomShaderPass.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneNode.h"
#include "SpecTexs.hpp"
#include "Type.hpp"

using namespace wallpaper::vulkan;

namespace
{
const char* BlendModeName(wallpaper::BlendMode mode) {
    switch (mode) {
    case wallpaper::BlendMode::Disable:     return "disable";
    case wallpaper::BlendMode::Translucent: return "translucent";
    case wallpaper::BlendMode::Additive:    return "additive";
    case wallpaper::BlendMode::Normal:      return "normal";
    }
    return "unknown";
}
} // namespace

CustomShaderPass::CustomShaderPass(const Desc& desc)
    : m_core(desc) {}

CustomShaderPass::~CustomShaderPass() = default;

void CustomShaderPass::prepare(Scene& scene, const Device& device, RenderingResources& resources) {
    setPrepared(m_core.prepare(scene, device, resources));
}

void CustomShaderPass::prepareDeferred(Scene& scene, const Device& device,
                                       RenderingResources& resources) {
    setPrepared(m_core.prepareDeferred(scene, device, resources));
}

void CustomShaderPass::refreshResources(Scene& scene, const Device& device,
                                        RenderingResources& resources) {
    if (! m_core.refreshResources(scene, device, resources)) setPrepared(false);
}

void CustomShaderPass::updateBeforeUpload() { m_core.updateBeforeUpload(); }

DeferredPrepareResourcesState
CustomShaderPass::requestDeferredPrepareResources(Scene& scene, const Device& device) {
    return m_core.requestDeferredPrepareResources(scene, device);
}

void CustomShaderPass::execute(const Device& device, RenderingResources& resources) {
    m_core.execute(device, resources);
    releaseFinalReadTexs(device);
}

void CustomShaderPass::destory(const Device&, RenderingResources& resources) {
    m_core.destroy(resources);
    setPrepared(false);
}

bool CustomShaderPass::warmupPipeline(Scene& scene, const Device& device,
                                      RenderingResources& resources) {
    return m_core.warmupPipeline(scene, device, resources);
}

std::string CustomShaderPass::residencyKey() const {
    return m_core.residencyKey("CustomShaderPass");
}

bool CustomShaderPass::canReuseForResidency(const VulkanPass& next_pass) const {
    const auto* next = dynamic_cast<const CustomShaderPass*>(&next_pass);
    return next != nullptr && m_core.canReuseForResidency(next->m_core);
}

void CustomShaderPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto* next = dynamic_cast<const CustomShaderPass*>(&next_pass);
    if (next != nullptr) m_core.absorbResidencyGraphState(next->m_core);
}

bool CustomShaderPass::referencesRenderTarget(std::string_view target) const {
    return m_core.referencesRenderTarget(target);
}

GpuPassDiagInfo CustomShaderPass::gpuDiagInfo() const {
    GpuPassDiagInfo info;
    info.category = GpuPassCategory::Effect;
    SceneNode* node = m_core.data().node;
    Scene* scene = m_core.data().scene;
    info.layer_id = m_core.data().layer_id;

    SceneMesh* mesh = node != nullptr ? node->Mesh() : nullptr;
    if (mesh != nullptr) {
        info.primitive = mesh->Primitive() == MeshPrimitive::POINT ? "point" : "triangle";
        if (const auto* material = mesh->Material(); material != nullptr) {
            info.blend = BlendModeName(material->blenmode);
        }
    }

    const bool particle_layer =
        scene != nullptr &&
        scene->objectRuntimeParticleSubsystems.find(info.layer_id) !=
            scene->objectRuntimeParticleSubsystems.end();
    const bool particle_mesh = m_core.data().dyn_vertex && mesh != nullptr &&
                               mesh->Primitive() == MeshPrimitive::POINT;
    if (particle_layer || particle_mesh) {
        info.category = GpuPassCategory::Particle;
        return info;
    }

    if (scene != nullptr) {
        for (const auto& tex : m_core.data().textures) {
            if (tex.empty() || IsSpecTex(tex)) continue;
            const auto texture_it = scene->textures.find(tex);
            if (texture_it != scene->textures.end() && texture_it->second.isVideo) {
                info.category = GpuPassCategory::Video;
                return info;
            }
        }
    }
    return info;
}

void CustomShaderPass::setDescTex(u32 index, std::string_view tex_key) {
    m_core.setTexture(index, tex_key);
}

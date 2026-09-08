#include "SceneDraw.h"
#include "Scene.h"
#include "SceneNode.h"
#include "SceneObject.h"

namespace wallpaper
{
void* SceneDraw::DataKey() const {
    return m_phase != nullptr ? const_cast<SceneDrawPhase*>(m_phase) : static_cast<void*>(m_node);
}

uint64_t SceneDraw::RenderIdentity() const {
    return m_phase != nullptr ? m_phase->RenderIdentity()
                             : (m_node != nullptr ? m_node->RenderIdentity() : 0);
}

int32_t SceneDraw::LayerId(const Scene& scene) const {
    return m_phase != nullptr ? m_phase->Owner().Id() : scene.LayerIdForNode(m_node);
}

SceneMesh* SceneDraw::Mesh() const {
    return m_phase != nullptr ? m_phase->Mesh() : (m_node != nullptr ? m_node->Mesh() : nullptr);
}

const std::string& SceneDraw::Name() const {
    static const std::string empty;
    return m_phase != nullptr ? m_phase->Name() : (m_node != nullptr ? m_node->Name() : empty);
}

const std::string& SceneDraw::Camera() const {
    static const std::string empty;
    return m_phase != nullptr ? m_phase->Camera() : (m_node != nullptr ? m_node->Camera() : empty);
}

bool SceneDraw::LocalVisible() const {
    return m_node == nullptr || m_node->LocalVisible();
}

bool SceneDraw::Visible(const Scene& scene) const {
    return m_phase != nullptr ? scene.IsLayerVisible(m_phase->Owner().Id())
                             : (m_node == nullptr || m_node->Visible());
}
} // namespace wallpaper

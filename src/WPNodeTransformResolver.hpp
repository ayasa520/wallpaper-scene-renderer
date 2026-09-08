#pragma once

#include "WPShaderValueUpdater.hpp"

#include <optional>

namespace wallpaper
{

class Scene;
class SceneNode;
class SceneCamera;

class WPNodeTransformResolver {
public:
    WPNodeTransformResolver(Scene& scene, const WPCameraParallax& parallax,
                            Map<void*, WPShaderValueData>& node_data_map,
                            Map<void*, Eigen::Matrix4d>& model_transform_cache,
                            Map<void*, Eigen::Vector3f>& parallax_offset_cache,
                            const std::array<float, 2>& mouse_pos,
                            uint64_t puppet_frame_serial);

    Eigen::Matrix4d ResolveParallaxedModelTransform(const SceneDraw& draw,
                                                    const SceneCamera* camera,
                                                    bool apply_parallax);
    Eigen::Matrix4d ResolveRawModelTransform(const SceneDraw& draw);
    Eigen::Vector3f ResolveParallaxOffset(const SceneDraw& draw, const SceneCamera* camera);
    const SceneObject* ParallaxRoot(const SceneDraw& draw) const;

private:
    const WPShaderValueData* FindNodeData(const SceneDraw& draw) const;
    Eigen::Matrix4d          ResolveModelTransform(const SceneDraw& draw, const WPShaderValueData* node_data);
    Eigen::Vector3f          ComputeParallaxOffset(const SceneDraw& draw,
                                                   const WPShaderValueData& node_data,
                                                   const SceneCamera* camera);
    Eigen::Matrix4d ResolveObjectTransform(const SceneObject& object);

    Scene&                         m_scene;
    const WPCameraParallax&        m_parallax;
    Map<void*, WPShaderValueData>& m_node_data_map;
    Map<void*, Eigen::Matrix4d>&   m_model_transform_cache;
    Map<void*, Eigen::Vector3f>&   m_parallax_offset_cache;
    const std::array<float, 2>&    m_mouse_pos;
    uint64_t                       m_puppet_frame_serial;
};

} // namespace wallpaper

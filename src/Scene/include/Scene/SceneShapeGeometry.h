#pragma once

namespace wallpaper
{
class Scene;
class SceneObject;

// Construction already created destinations and effect FBOs. Its final geometry step must
// not resize a shared FBO that this owner merely found in the initial name lookup.
void RebuildShapeLayerGeometry(SceneObject& owner);

// Runtime setup additionally reselects destinations and reruns the owner's retained FBOs.
void RefreshShapeLayerResources(Scene& scene, SceneObject& owner);
} // namespace wallpaper

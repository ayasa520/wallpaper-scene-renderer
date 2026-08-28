#include "Scene.h"

#include "Image.hpp"
#include "SceneCamera.h"

#include "Fs/VFS.h"
#include "Interface/IImageParser.h"
#include "Interface/IShaderValueUpdater.h"
#include "Particle/ParticleSystem.h"
#include "Utils/Logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <string_view>
#include <unordered_set>

namespace wallpaper 
{

namespace
{
std::size_t EstimateParsedImageBytes(const std::shared_ptr<Image>& image) {
    if (image == nullptr) return 0;

    std::size_t total = 0;
    for (const auto& slot : image->slots) {
        for (const auto& mipmap : slot.mipmaps) {
            if (mipmap.size > 0) {
                total += static_cast<std::size_t>(mipmap.size);
                continue;
            }
            total += static_cast<std::size_t>(std::max(mipmap.width, 0)) *
                     static_cast<std::size_t>(std::max(mipmap.height, 0)) * 4u;
        }
    }
    return total;
}

bool IsLayerVisibleImpl(const Scene& scene, int32_t layer_id, std::unordered_set<int32_t>& visiting) {
    if (layer_id == 0) return true;
    if (!visiting.insert(layer_id).second) return true;

    const auto* object = scene.FindSceneObject(layer_id);
    if (object == nullptr) return true;
    if (!object->LocalVisible()) return false;
    if (object->ParentId() == 0) return true;

    return IsLayerVisibleImpl(scene, object->ParentId(), visiting);
}

Eigen::Vector3d ToVector3d(const std::array<float, 3>& value) {
    return Eigen::Vector3d(value[0], value[1], value[2]);
}

std::array<float, 3> LerpArray3(const std::array<float, 3>& lhs,
                                const std::array<float, 3>& rhs,
                                double ratio) {
    const auto t = static_cast<float>(std::clamp(ratio, 0.0, 1.0));
    return {
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    };
}

bool ResolveCameraPathSample(const Scene::CameraPathSegment& segment,
                             double local_time,
                             Scene::CameraPathKeyframe& out) {
    if (segment.keyframes.empty()) return false;
    if (segment.keyframes.size() == 1) {
        out = segment.keyframes.front();
        return true;
    }

    const auto clamped_time = std::clamp(local_time, 0.0, std::max(0.0, segment.duration));
    const auto& first = segment.keyframes.front();
    const auto& last = segment.keyframes.back();
    if (clamped_time <= first.timestamp) {
        out = first;
        return true;
    }
    if (clamped_time >= last.timestamp) {
        out = last;
        return true;
    }

    for (size_t index = 1; index < segment.keyframes.size(); index++) {
        const auto& lhs = segment.keyframes[index - 1];
        const auto& rhs = segment.keyframes[index];
        if (clamped_time > rhs.timestamp) continue;

        const auto span = rhs.timestamp - lhs.timestamp;
        const auto ratio = span > 1e-9 ? (clamped_time - lhs.timestamp) / span : 0.0;
        out.timestamp = clamped_time;
        out.eye = LerpArray3(lhs.eye, rhs.eye, ratio);
        out.center = LerpArray3(lhs.center, rhs.center, ratio);
        out.up = LerpArray3(lhs.up, rhs.up, ratio);
        return true;
    }

    out = last;
    return true;
}

void CollectLayerEffectNodes(const Scene& scene, int32_t layer_id, std::vector<SceneNode*>& nodes) {
    auto camera_names_it = scene.objectRuntimeCameraNames.find(layer_id);
    if (camera_names_it == scene.objectRuntimeCameraNames.end()) return;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = scene.cameras.find(camera_name);
        if (camera_it == scene.cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr) continue;

        if (effect_layer->HasFinalComposite()) {
            // Final composite nodes are owned by the image-effect bridge instead of the authored
            // scene tree, so they will not be reached by normal parent/child propagation. Treat
            // them as layer-owned runtime nodes here to keep layer visibility authoritative while
            // preserving effect-local visibility on the internal shader nodes.
            nodes.push_back(&effect_layer->FinalNode());
        }

        for (size_t effect_index = 0; effect_index < effect_layer->EffectCount(); effect_index++) {
            auto& effect = effect_layer->GetEffect(effect_index);
            for (auto& effect_node : effect->nodes) {
                if (effect_node.sceneNode) nodes.push_back(effect_node.sceneNode.get());
            }
        }
    }
}

void ApplyLayerVisibilityRecursive(Scene& scene, int32_t layer_id, std::unordered_set<int32_t>& visited) {
    if (layer_id == 0 || !visited.insert(layer_id).second) return;

    std::unordered_set<int32_t> visiting;
    const bool effective_visible = IsLayerVisibleImpl(scene, layer_id, visiting);

    if (auto runtime_nodes_it = scene.objectRuntimeNodes.find(layer_id);
        runtime_nodes_it != scene.objectRuntimeNodes.end()) {
        for (auto* node : runtime_nodes_it->second) {
            if (node != nullptr) {
                // Layer visibility propagation must not overwrite a node's own local visibility
                // contract. Runtime-owned support nodes may intentionally stay hidden even while
                // their authored layer is visible, so the scene system only updates the
                // layer-level flag.
                node->SetLayerVisible(effective_visible);
            }
        }
    }

    std::vector<SceneNode*> effect_nodes;
    CollectLayerEffectNodes(scene, layer_id, effect_nodes);
    for (auto* node : effect_nodes) {
        if (node != nullptr) {
            // Effect nodes are also owned by the layer-visibility system, but they still need to
            // preserve any explicit local visibility decisions that the effect pipeline may make.
            node->SetLayerVisible(effective_visible);
        }
    }

    for (const auto& [child_id, object] : scene.sceneObjects) {
        if (object != nullptr && object->ParentId() == layer_id) {
            ApplyLayerVisibilityRecursive(scene, child_id, visited);
        }
    }
}

std::pair<int32_t, Scene::CameraLayerRuntimeState*> FindActiveCameraLayer(Scene& scene) {
    // Wallpaper Engine uses the bottom-most visible camera layer as the active view. Scene JSON is
    // parsed in layer order, so walking the recorded camera layer order backwards gives later
    // camera layers precedence while still letting user/script visibility changes disable them.
    for (auto it = scene.cameraLayerOrder.rbegin(); it != scene.cameraLayerOrder.rend(); ++it) {
        auto layer_it = scene.cameraLayers.find(*it);
        if (layer_it == scene.cameraLayers.end() || !layer_it->second.node) continue;
        if (!scene.IsLayerVisible(*it)) continue;
        return { *it, &layer_it->second };
    }
    return { 0, nullptr };
}

constexpr uint64_t kOfficialTextureResolutionAutoArea = 1969920ull;

const char* TextureResolutionRequestedName(int quality) {
    switch (quality) {
    case 1:
        return "half";
    case 2:
        return "auto";
    default:
        return "full";
    }
}

bool TextureResolutionShouldDropMip0(int quality, uint32_t width, uint32_t height) {
    if (quality == 1) return true;
    if (quality != 2) return false;
    // Official Wallpaper Engine 2.8.42 auto: one global bool for the whole
    // wallpaper, not a per-texture 1920x1080 test. Compare output pixel area
    // (floatA * floatB) against 1969920 (1920 * 1080 * 0.95). Below → half.
    const uint64_t area =
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    return area < kOfficialTextureResolutionAutoArea;
}

void DropImageMip0(Image& image) {
    for (auto& slot : image.slots) {
        if (slot.mipmaps.size() <= 1) continue;
        slot.mipmaps.erase(slot.mipmaps.begin());
        if (!slot.mipmaps.empty()) {
            slot.width  = slot.mipmaps.front().width;
            slot.height = slot.mipmaps.front().height;
        }
    }
}

double SanitizeCameraZoom(double zoom, int32_t layer_id) {
    if (std::isfinite(zoom) && zoom > 0.0001) return zoom;

    // Invalid authored/user zoom values would collapse the orthographic projection to infinity.
    // Log the offending camera layer and keep a neutral zoom so the wallpaper remains visible.
    LOG_ERROR("SceneCameraLayer: invalid zoom %.6f on layer=%d, using 1.0", zoom, layer_id);
    return 1.0;
}

void ApplyCameraProjectionState(Scene& scene,
                                const std::string& camera_name,
                                SceneCamera& camera,
                                double zoom,
                                float fov,
                                int32_t layer_id) {
    if (camera.IsPerspective()) {
        camera.SetFov(fov);
    } else {
        const double safe_zoom = SanitizeCameraZoom(zoom, layer_id);
        camera.SetWidth(std::max(1.0, static_cast<double>(scene.ortho[0]) / safe_zoom));
        camera.SetHeight(std::max(1.0, static_cast<double>(scene.ortho[1]) / safe_zoom));
    }

    camera.Update();
    scene.UpdateLinkedCamera(camera_name);
}
} // namespace

Scene::Scene(): sceneGraph(std::make_shared<SceneNode>()) ,paritileSys(std::make_unique<ParticleSystem>(*this)) {}

Scene::~Scene() {
    ClearParsedImageCache();
}

std::shared_ptr<Image> Scene::CacheParsedImageResultLocked(
    const std::string& texture_key,
    std::shared_ptr<Image> image,
    std::chrono::steady_clock::time_point started_at,
    const char* success_event,
    const char* failure_event) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started_at)
                                .count();
    if (image != nullptr) {
        PrepareParsedImageForGpu(*image);
        m_parsed_image_cache[texture_key] = image;
        LOG_INFO("%s: key='%s' bytes=%zu duration=%.2fms",
                 success_event,
                 texture_key.c_str(),
                 EstimateParsedImageBytes(image),
                 static_cast<double>(elapsed_us) / 1000.0);
        return image;
    }

    m_failed_parsed_images.insert(texture_key);
    LOG_ERROR("%s: key='%s' duration=%.2fms",
              failure_event,
              texture_key.c_str(),
              static_cast<double>(elapsed_us) / 1000.0);
    return {};
}

std::shared_ptr<Image> Scene::GetParsedImageIfReady(const std::string& texture_key) {
    if (texture_key.empty()) return {};

    std::lock_guard lock(m_parsed_image_mutex);
    if (const auto cached_it = m_parsed_image_cache.find(texture_key);
        cached_it != m_parsed_image_cache.end()) {
        return cached_it->second;
    }
    if (m_failed_parsed_images.count(texture_key) != 0) return {};

    const auto pending_it = m_pending_parsed_images.find(texture_key);
    if (pending_it == m_pending_parsed_images.end()) return {};
    if (pending_it->second.future.wait_for(std::chrono::milliseconds(0)) !=
        std::future_status::ready) {
        return {};
    }

    const auto started_at = pending_it->second.started_at;
    auto       image      = pending_it->second.future.get();
    m_pending_parsed_images.erase(pending_it);

    return CacheParsedImageResultLocked(texture_key,
                                        std::move(image),
                                        started_at,
                                        "SceneImageAsyncParseComplete",
                                        "SceneImageAsyncParseFailed");
}

std::shared_ptr<Image> Scene::ParseImageBlockingCached(const std::string& texture_key) {
    if (texture_key.empty() || imageParser == nullptr) return {};

    {
        std::lock_guard lock(m_parsed_image_mutex);
        if (const auto cached_it = m_parsed_image_cache.find(texture_key);
            cached_it != m_parsed_image_cache.end()) {
            return cached_it->second;
        }
        if (m_failed_parsed_images.count(texture_key) != 0) return {};

        const auto pending_it = m_pending_parsed_images.find(texture_key);
        if (pending_it != m_pending_parsed_images.end()) {
            const auto started_at = pending_it->second.started_at;
            auto       image      = pending_it->second.future.get();
            m_pending_parsed_images.erase(pending_it);
            return CacheParsedImageResultLocked(texture_key,
                                                std::move(image),
                                                started_at,
                                                "SceneImageAsyncParseJoined",
                                                "SceneImageAsyncParseFailed");
        }
    }

    const auto started_at = std::chrono::steady_clock::now();
    auto       image      = imageParser->Parse(texture_key);

    std::lock_guard lock(m_parsed_image_mutex);
    return CacheParsedImageResultLocked(texture_key,
                                        std::move(image),
                                        started_at,
                                        "SceneImageParseBlocking",
                                        "SceneImageParseBlockingFailed");
}

Scene::ParsedImageRequest Scene::RequestParsedImageAsync(const std::string& texture_key) {
    if (texture_key.empty() || imageParser == nullptr) {
        return { ParsedImageRequestState::Failed, {} };
    }

    if (auto image = GetParsedImageIfReady(texture_key); image != nullptr) {
        return { ParsedImageRequestState::Ready, image };
    }

    {
        std::lock_guard lock(m_parsed_image_mutex);
        if (const auto cached_it = m_parsed_image_cache.find(texture_key);
            cached_it != m_parsed_image_cache.end()) {
            return { ParsedImageRequestState::Ready, cached_it->second };
        }
        if (m_failed_parsed_images.count(texture_key) != 0) {
            return { ParsedImageRequestState::Failed, {} };
        }
        if (m_pending_parsed_images.count(texture_key) != 0) {
            return { ParsedImageRequestState::Pending, {} };
        }

        auto*       parser   = imageParser.get();
        std::string key_copy = texture_key;
        PendingParsedImageRequest pending;
        pending.started_at = std::chrono::steady_clock::now();
        pending.future     = std::async(std::launch::async, [parser, key_copy]() {
            return parser != nullptr ? parser->Parse(key_copy) : std::shared_ptr<Image> {};
        });
        m_pending_parsed_images.emplace(texture_key, std::move(pending));
    }

    LOG_INFO("SceneImageAsyncParseQueued: key='%s'", texture_key.c_str());
    return { ParsedImageRequestState::Pending, {} };
}

void Scene::DropParsedImageCache(std::string_view texture_key) {
    if (texture_key.empty()) return;

    const std::string key(texture_key);
    std::future<std::shared_ptr<Image>> pending_future;
    std::size_t dropped_bytes = 0;
    bool dropped_cached_image = false;
    bool dropped_pending_parse = false;
    {
        std::lock_guard lock(m_parsed_image_mutex);
        if (auto cached_it = m_parsed_image_cache.find(key);
            cached_it != m_parsed_image_cache.end()) {
            dropped_bytes = EstimateParsedImageBytes(cached_it->second);
            m_parsed_image_cache.erase(cached_it);
            dropped_cached_image = true;
        }
        if (auto pending_it = m_pending_parsed_images.find(key);
            pending_it != m_pending_parsed_images.end()) {
            pending_future = std::move(pending_it->second.future);
            m_pending_parsed_images.erase(pending_it);
            dropped_pending_parse = true;
        }
        m_failed_parsed_images.erase(key);
    }
    if (dropped_cached_image || dropped_pending_parse) {
        LOG_INFO("SceneImageCacheDrop: key='%s' cached=%s bytes=%zu pending=%s",
                 key.c_str(),
                 dropped_cached_image ? "true" : "false",
                 dropped_bytes,
                 dropped_pending_parse ? "true" : "false");
    }
    if (pending_future.valid()) pending_future.wait();
}

void Scene::ClearParsedImageCache() {
    std::vector<std::future<std::shared_ptr<Image>>> pending_futures;
    {
        std::lock_guard lock(m_parsed_image_mutex);
        pending_futures.reserve(m_pending_parsed_images.size());
        for (auto& [_, request] : m_pending_parsed_images) {
            if (request.future.valid()) pending_futures.emplace_back(std::move(request.future));
        }
        m_parsed_image_cache.clear();
        m_pending_parsed_images.clear();
        m_failed_parsed_images.clear();
    }

    if (!pending_futures.empty()) {
        LOG_INFO("SceneImageAsyncParseJoin: pending=%zu", pending_futures.size());
        for (auto& future : pending_futures) {
            if (future.valid()) future.wait();
        }
    }
}

void Scene::ApplyTextureResolution(int quality, uint32_t output_width, uint32_t output_height) {
    quality = std::clamp(quality, 0, 2);
    const bool next_drop = TextureResolutionShouldDropMip0(quality, output_width, output_height);
    const bool quality_changed = textureResolution.quality != quality;
    const bool drop_changed    = textureResolution.drop_mip0 != next_drop;
    const bool first_apply     = textureResolution.output_width == 0 && output_width != 0;

    textureResolution.quality       = quality;
    textureResolution.drop_mip0     = next_drop;
    textureResolution.output_width  = output_width;
    textureResolution.output_height = output_height;

    if (quality_changed || drop_changed || first_apply) {
        const uint64_t area =
            static_cast<uint64_t>(output_width) * static_cast<uint64_t>(output_height);
        LOG_INFO("texture-resolution requested=%s drop-mip0=%s output=%ux%u area=%llu",
                 TextureResolutionRequestedName(quality),
                 next_drop ? "true" : "false",
                 output_width,
                 output_height,
                 static_cast<unsigned long long>(area));
    }

    if (!drop_changed) return;

    textureResolution.epoch++;
    for (auto& [key, texture] : textures) {
        texture.gpuWidth  = 0;
        texture.gpuHeight = 0;
        if (texture.isVideo || key.empty()) continue;
        // 1-mip / video / synthetic images cannot drop mip0. Unknown mip
        // counts are refreshed so a later parse can apply the new policy.
        if (texture.mipmapCount == 1) continue;
        dirtyImportedTextureKeys.insert(key);
    }
    ClearParsedImageCache();
    MarkRenderGraphResourcesDirty();
}

void Scene::ApplyTextureResolutionForCurrentOutput() {
    if (physicalOutputExtent[0] == 0 || physicalOutputExtent[1] == 0) return;
    ApplyTextureResolution(textureResolution.quality,
                           physicalOutputExtent[0],
                           physicalOutputExtent[1]);
}

void Scene::PrepareParsedImageForGpu(Image& image) {
    image.textureResolutionEpoch = textureResolution.epoch;
    const auto texture_it = textures.find(image.key);
    const bool video =
        image.header.isVideoTexture ||
        (texture_it != textures.end() && texture_it->second.isVideo);
    if (!video && textureResolution.drop_mip0) {
        DropImageMip0(image);
    }
    if (texture_it == textures.end() || image.slots.empty()) return;
    texture_it->second.gpuWidth  = image.slots[0].width;
    texture_it->second.gpuHeight = image.slots[0].height;
}

std::array<i32, 4>
Scene::EffectiveImportedTextureResolution(const SceneTexture& texture) const {
    const bool drop = textureResolution.drop_mip0 && !texture.isVideo &&
                      texture.mipmapCount > 1;
    if (texture.gpuWidth > 0 && texture.gpuHeight > 0) {
        // Official bind path: g_TextureNResolution follows the uploaded GPU
        // extent, not the authored .tex header, once mip0 has been dropped.
        if (!drop) {
            if (texture.mipmap_larger) {
                return { texture.width, texture.height, texture.mapWidth, texture.mapHeight };
            }
            return { texture.mapWidth, texture.mapHeight, texture.mapWidth, texture.mapHeight };
        }
        if (texture.mipmap_larger) {
            return { texture.gpuWidth,
                     texture.gpuHeight,
                     std::max<i32>(1, texture.mapWidth / 2),
                     std::max<i32>(1, texture.mapHeight / 2) };
        }
        return { texture.gpuWidth, texture.gpuHeight, texture.gpuWidth, texture.gpuHeight };
    }
    if (!drop) {
        if (texture.mipmap_larger) {
            return { texture.width, texture.height, texture.mapWidth, texture.mapHeight };
        }
        return { texture.mapWidth, texture.mapHeight, texture.mapWidth, texture.mapHeight };
    }
    if (texture.mipmap_larger) {
        return { std::max<i32>(1, texture.width / 2),
                 std::max<i32>(1, texture.height / 2),
                 std::max<i32>(1, texture.mapWidth / 2),
                 std::max<i32>(1, texture.mapHeight / 2) };
    }
    const i32 half_w = std::max<i32>(1, texture.mapWidth / 2);
    const i32 half_h = std::max<i32>(1, texture.mapHeight / 2);
    return { half_w, half_h, half_w, half_h };
}

SceneObject* Scene::FindSceneObject(int32_t layer_id) {
    auto it = sceneObjects.find(layer_id);
    return it == sceneObjects.end() ? nullptr : it->second.get();
}

const SceneObject* Scene::FindSceneObject(int32_t layer_id) const {
    auto it = sceneObjects.find(layer_id);
    return it == sceneObjects.end() ? nullptr : it->second.get();
}

SceneObject& Scene::EnsureSceneObject(int32_t layer_id) {
    if (auto* existing = FindSceneObject(layer_id)) return *existing;
    auto object = std::make_unique<SceneObject>(layer_id);
    auto* raw   = object.get();
    sceneObjects.emplace(layer_id, std::move(object));
    return *raw;
}

void Scene::DestroySceneObject(int32_t layer_id) { sceneObjects.erase(layer_id); }

void Scene::SetLayerParentBinding(int32_t layer_id, int32_t parent_id, std::string attachment) {
    if (layer_id == 0) return;
    if (parent_id == 0 && attachment.empty()) {
        if (auto* object = FindSceneObject(layer_id)) object->ClearParentBinding();
        return;
    }
    EnsureSceneObject(layer_id).SetParentBinding(parent_id, std::move(attachment));
}

Scene::LayerParentBinding Scene::GetLayerParentBinding(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    if (object == nullptr) return LayerParentBinding {};
    return LayerParentBinding {
        .parent_id  = object->ParentId(),
        .attachment = object->Attachment(),
    };
}

void Scene::ClearLayerParentBinding(int32_t layer_id) {
    if (auto* object = FindSceneObject(layer_id)) object->ClearParentBinding();
}

std::vector<int32_t> Scene::GetLayerChildren(int32_t layer_id) const {
    std::vector<int32_t> children;
    if (layer_id == 0) return children;
    for (const auto& [child_id, object] : sceneObjects) {
        if (object != nullptr && object->ParentId() == layer_id) children.push_back(child_id);
    }
    return children;
}

void Scene::SetLayerLocalVisibility(int32_t layer_id, bool visible) {
    if (layer_id == 0) return;

    EnsureSceneObject(layer_id).SetLocalVisible(visible);
}

bool Scene::GetLayerLocalVisibility(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? true : object->LocalVisible();
}

bool Scene::IsLayerVisible(int32_t layer_id) const {
    std::unordered_set<int32_t> visiting;
    return IsLayerVisibleImpl(*this, layer_id, visiting);
}

void Scene::ApplyLayerVisibility(int32_t layer_id) {
    std::unordered_set<int32_t> visited;
    ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    if (!cameraLayers.empty()) UpdateActiveCameraLayer();
}

void Scene::ApplyAllLayerVisibility() {
    std::unordered_set<int32_t> visited;
    for (const auto layer_id : layerOrder) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    for (const auto& [layer_id, _] : layerNodes) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    if (!cameraLayers.empty()) UpdateActiveCameraLayer();
}

void Scene::UpdateModelCameraPath() {
    if (!modelCameraPathEnabled || modelCameraPathSegments.empty() ||
        modelPerspectiveCameraName.empty()) {
        return;
    }

    auto camera_it = cameras.find(modelPerspectiveCameraName);
    if (camera_it == cameras.end() || !camera_it->second) return;

    double total_duration = 0.0;
    for (const auto& segment : modelCameraPathSegments) {
        total_duration += std::max(0.0, segment.duration);
    }
    if (total_duration <= 1e-9) return;

    double path_time = std::fmod(std::max(0.0, elapsingTime), total_duration);
    if (path_time < 0.0) path_time += total_duration;

    int32_t active_segment = -1;
    double local_time = path_time;
    for (size_t index = 0; index < modelCameraPathSegments.size(); index++) {
        const auto duration = std::max(0.0, modelCameraPathSegments[index].duration);
        if (local_time <= duration || index + 1 == modelCameraPathSegments.size()) {
            active_segment = static_cast<int32_t>(index);
            break;
        }
        local_time -= duration;
    }
    if (active_segment < 0 ||
        active_segment >= static_cast<int32_t>(modelCameraPathSegments.size())) {
        return;
    }

    Scene::CameraPathKeyframe sample;
    if (!ResolveCameraPathSample(modelCameraPathSegments[active_segment], local_time, sample)) {
        return;
    }

    // Camera path playback is bound to the model-only camera name installed by WPModelObject
    // parsing. This deliberately avoids `global_perspective`, which is a legacy 2D particle camera.
    camera_it->second->SetExplicitView(ToVector3d(sample.eye),
                                       ToVector3d(sample.center),
                                       ToVector3d(sample.up));
    UpdateLinkedCamera(modelPerspectiveCameraName);

    if (activeModelCameraPathSegment != active_segment) {
        const auto& segment = modelCameraPathSegments[active_segment];
        LOG_INFO("Scene3DModelCameraPathActive: previous=%d active=%d duration=%.3f "
                 "local-time=%.3f eye=[%.3f, %.3f, %.3f] center=[%.3f, %.3f, %.3f]",
                 activeModelCameraPathSegment,
                 active_segment,
                 segment.duration,
                 local_time,
                 sample.eye[0],
                 sample.eye[1],
                 sample.eye[2],
                 sample.center[0],
                 sample.center[1],
                 sample.center[2]);
        activeModelCameraPathSegment = active_segment;
    }
}

namespace
{

// The same translation is added to eye and center so look direction stays unchanged.
constexpr float kCameraShakeYFrequency     = 1.3329999446868896f;
constexpr float kCameraShakeAmplitudeScale = 0.1f;
constexpr float kCameraShakeRoughnessPow   = 3.0f;
constexpr float kCameraShakeRoughnessEps   = 0.001f;

Eigen::Vector3d ComputeSceneCameraShakeOffset(bool enabled, bool orthographic, float amplitude,
                                              float roughness, float speed, double time_seconds,
                                              int32_t ortho_height) {
    if (!enabled) return Eigen::Vector3d::Zero();

    const float p     = std::pow(roughness, kCameraShakeRoughnessPow);
    const float t     = speed * speed * static_cast<float>(time_seconds);
    float       x     = std::sin(t);
    float       y     = std::cos(t * kCameraShakeYFrequency);
    float       z     = std::cos(t);
    float       scale = amplitude * kCameraShakeAmplitudeScale;
    if (orthographic) {
        z = 0.0f;
        scale *= static_cast<float>(ortho_height) * kCameraShakeAmplitudeScale;
    }
    if (p > kCameraShakeRoughnessEps && p != 1.0f) {
        const float len2 = x * x + y * y + z * z;
        if (len2 > 0.0f) {
            const float len = std::sqrt(len2);
            const float n   = std::pow(len, p) / len;
            x *= n;
            y *= n;
            z *= n;
        }
    }
    return Eigen::Vector3d(static_cast<double>(x * scale),
                           static_cast<double>(y * scale),
                           static_cast<double>(z * scale));
}

void ApplyCameraShakeOffset(Scene& scene, std::string_view camera_name,
                            const Eigen::Vector3d& offset) {
    auto camera_it = scene.cameras.find(std::string(camera_name));
    if (camera_it == scene.cameras.end() || !camera_it->second) return;
    camera_it->second->SetShakeOffset(offset);
    camera_it->second->Update();
}

} // namespace

void Scene::UpdateCameraShake() {
    const Eigen::Vector3d offset =
        ComputeSceneCameraShakeOffset(cameraShake,
                                      cameraOrthographic,
                                      cameraShakeAmplitude,
                                      cameraShakeRoughness,
                                      cameraShakeSpeed,
                                      elapsingTime,
                                      ortho[1]);
    ApplyCameraShakeOffset(*this, "global", offset);
    ApplyCameraShakeOffset(*this, "global_perspective", offset);
    if (activeCamera != nullptr) {
        activeCamera->SetShakeOffset(offset);
        activeCamera->Update();
    }
    if (!cameraOrthographic && !modelPerspectiveCameraName.empty()) {
        ApplyCameraShakeOffset(*this, modelPerspectiveCameraName, offset);
    }
}

Eigen::Vector3f Scene::ResolveCameraLayerNodeTranslation(
    const std::array<float, 3>& authored_origin) const {
    // WE 2D camera origins are authored around the static camera origin, where 0/0 means the
    // default centered wallpaper view. Hanabi's orthographic camera node is centered in render
    // coordinates, so add the canvas half-size before attaching the SceneCamera to this layer.
    return Eigen::Vector3f {
        static_cast<float>(ortho[0]) * 0.5f + authored_origin[0],
        static_cast<float>(ortho[1]) * 0.5f + authored_origin[1],
        authored_origin[2],
    };
}

void Scene::UpdateActiveCameraLayer() {
    auto [next_layer_id, camera_layer] = FindActiveCameraLayer(*this);

    std::string camera_name = "global";
    std::shared_ptr<SceneNode> camera_node = defaultGlobalCameraNode;
    double zoom = defaultGlobalCameraZoom;
    float fov = 50.0f;

    if (camera_layer != nullptr) {
        camera_name = camera_layer->camera_name.empty() ? "global" : camera_layer->camera_name;
        camera_node = camera_layer->node;
        zoom = camera_layer->zoom;
        fov = camera_layer->fov;
    }

    auto camera_it = cameras.find(camera_name);
    if (camera_it == cameras.end() || !camera_it->second) {
        LOG_ERROR("SceneCameraLayer: target camera '%s' for layer=%d is missing",
                  camera_name.c_str(),
                  next_layer_id);
        camera_it = cameras.find("global");
        camera_name = "global";
    }
    if (camera_it == cameras.end() || !camera_it->second || !camera_node) return;

    camera_it->second->AttatchNode(camera_node);
    ApplyCameraProjectionState(*this,
                               camera_name,
                               *camera_it->second,
                               zoom,
                               fov,
                               next_layer_id);
    activeCamera = camera_it->second.get();

    if (activeCameraLayerId != next_layer_id) {
        // This transition log is intentionally sparse: it proves which authored camera layer owns
        // the view without flooding frame logs while keyframed zoom/origin values animate.
        LOG_INFO("SceneCameraLayerActive: previous=%d active=%d camera='%s' zoom=%.3f origin=[%.3f, %.3f, %.3f]",
                 activeCameraLayerId,
                 next_layer_id,
                 camera_name.c_str(),
                 zoom,
                 camera_layer != nullptr ? camera_layer->origin[0] : 0.0f,
                 camera_layer != nullptr ? camera_layer->origin[1] : 0.0f,
                 camera_layer != nullptr ? camera_layer->origin[2] : 0.0f);
        activeCameraLayerId = next_layer_id;
    }
}

SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id, uint32_t effect_index) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr || effect_index >= effect_layer->EffectCount()) continue;
        return effect_layer->GetEffect(effect_index).get();
    }

    return nullptr;
}

const SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id,
                                               uint32_t effect_index) const {
    return const_cast<Scene*>(this)->FindImageEffect(owner_layer_id, effect_index);
}

SceneImageEffectLayer* Scene::FindImageEffectLayer(int32_t owner_layer_id) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer != nullptr) return effect_layer;
    }

    return nullptr;
}

SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id, int32_t effect_id) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr) continue;

        for (std::size_t effect_index = 0; effect_index < effect_layer->EffectCount();
             effect_index++) {
            auto& effect = effect_layer->GetEffect(effect_index);
            if (effect != nullptr && effect->EffectId() == effect_id) return effect.get();
        }
    }

    return nullptr;
}

const SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id,
                                                   int32_t effect_id) const {
    return const_cast<Scene*>(this)->FindImageEffectById(owner_layer_id, effect_id);
}

bool Scene::SetEffectLocalVisibility(int32_t owner_layer_id, uint32_t effect_index,
                                     bool visible) {
    auto* effect = FindImageEffect(owner_layer_id, effect_index);
    if (effect == nullptr) return false;

    // Only the effect-local bit changes here. The render graph topology remains valid because
    // hidden effects are handled by conditional execution and a bypass copy, while layer visibility
    // propagation still owns parent/child effective visibility.
    effect->SetLocalVisible(visible);
    ApplyLayerVisibility(owner_layer_id);
    return true;
}

bool Scene::SetEffectLocalVisibilityById(int32_t owner_layer_id, int32_t effect_id,
                                         bool visible) {
    auto* effect = FindImageEffectById(owner_layer_id, effect_id);
    if (effect == nullptr) return false;

    effect->SetLocalVisible(visible);
    ApplyLayerVisibility(owner_layer_id);
    return true;
}

}

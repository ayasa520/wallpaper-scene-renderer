#include "Scene.h"

#include "Image.hpp"
#include "SpecTexs.hpp"
#include "SceneCamera.h"
#include "SceneImageEffectLayer.h"
#include "SceneImageSource.h"
#include "SceneResidency.h"

#include "Fs/VFS.h"
#include "Interface/IImageParser.h"
#include "Interface/IShaderValueUpdater.h"
#include "Particle/ParticleSystem.h"
#include "Utils/Logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace wallpaper 
{

// Defined here instead of the header: SceneObject.h only forward-declares the parser-side
// TextLayerRuntimeState, and this translation unit sees the complete type through Scene.h.
SceneObject::~SceneObject() = default;

std::shared_ptr<const std::string> Scene::GetSystemTextureBinding(const std::string& name) {
    const auto [it, inserted] = m_system_texture_bindings.try_emplace(name);
    if (inserted) it->second = std::make_shared<std::string>();
    return it->second;
}

void Scene::SetSystemTextureBinding(const std::string& name, const std::string& texture_key) {
    GetSystemTextureBinding(name);
    auto& current = *m_system_texture_bindings.at(name);
    if (current == texture_key) return;
    const auto previous = CollectRetainedResidencyResources(*this);
    if (wallpaper::diagnostics::Options().trace_media_state) {
        LOG_INFO("SceneSystemTextureChange: property='%s' previous='%s' current='%s'",
                 name.c_str(), current.c_str(), texture_key.c_str());
    }
    current = texture_key;
    QueueReplacedImportedTextures(*this, previous);
    // Selecting a system image can replace a render-target input, so this changes graph
    // dependencies as well as descriptors. All handles are updated on the scene thread before the
    // next graph build. Pixel replacement under an unchanged key continues to use the
    // imported-resource path.
    MarkRenderGraphTopologyDirty();
}

std::shared_ptr<const std::optional<std::string>> Scene::RegisterUserTextureBinding(
    std::string property, std::string authored_key, std::optional<std::string> selected_key,
    TextureKeyResolver resolve) {
    auto selection = std::make_shared<std::optional<std::string>>(std::move(selected_key));
    m_user_texture_bindings.push_back({ std::move(property), std::move(authored_key),
                                       selection, std::move(resolve) });
    return selection;
}

void Scene::RefreshUserTextureBindings() {
    std::optional<LayerResidencyResources> previous;
    for (auto it = m_user_texture_bindings.begin(); it != m_user_texture_bindings.end();) {
        const auto selection = it->selection.lock();
        if (!selection) {
            it = m_user_texture_bindings.erase(it);
            continue;
        }
        const auto* property = LookupUserPropertyString(&userProperties, it->property);
        std::optional<std::string> next;
        if (property != nullptr && !property->empty()) next = it->resolve(*this, *property);
        if (next != *selection) {
            // Snapshot once before the first changed binding. Material copies share selection
            // handles, and system inputs have lower precedence than user overrides, so the old
            // resolved material slots are the ownership source for the entire property batch.
            if (!previous) previous = CollectRetainedResidencyResources(*this);
            // Resolve metadata before publishing the selected key. The next source refresh and
            // graph build must see one coherent image, sampler and extent even when this texture
            // has never been used in the scene. Empty resets retain each material's own input;
            // they must not reuse the nondefault selection that happened to be active at load.
            const auto& key = next ? *next : it->authoredKey;
            const bool register_texture = !key.empty() && !IsSpecTex(key) &&
                !renderTargets.contains(key) && !textures.contains(key);
            if (register_texture) {
                RegisterTextureFromHeader(key, imageParser->ParseHeader(key));
            }
            LOG_INFO("SceneUserTextureChange: property='%s' authored='%s' previous='%s' "
                     "current='%s' override=%s registered=%s",
                     it->property.c_str(), it->authoredKey.c_str(),
                     selection->has_value() ? (**selection).c_str() : it->authoredKey.c_str(),
                     key.c_str(), next ? "true" : "false", register_texture ? "true" : "false");
            *selection = std::move(next);
            // A material selection changes the graph's imported or named-target read, not just
            // its pixels. Diff the resident graph after all property writes, while keeping the
            // material, script instances and retained image-source resource policy unchanged.
            MarkRenderGraphTopologyDirty();
        }
        ++it;
    }
    if (previous) QueueReplacedImportedTextures(*this, *previous);
}

void Scene::RegisterTextureFromHeader(const std::string& name, const ImageHeader& header) {
    if (textures.contains(name)) return;

    SceneTexture texture;
    texture.sample    = header.sample;
    texture.url       = name;
    texture.format    = header.format;
    texture.isVideo   = header.isVideoTexture;
    texture.width     = header.width;
    texture.height    = header.height;
    texture.mapWidth  = header.mapWidth;
    texture.mapHeight = header.mapHeight;
    texture.mipmapCount   = header.mipmapCount;
    texture.mipmap_larger = header.mipmap_larger;
    if (header.isSprite) {
        texture.isSprite   = true;
        texture.spriteAnim = header.spriteAnim;
    }
    textures.emplace(name, std::move(texture));
}

void Scene::RefreshImageSourceTextures() {
    // Named sources can occur after their readers in the authored draw list. Prepare the
    // source descriptors before dependent image setup so one graph compilation sees coherent
    // dimensions and filtering. This walk performs no draws and never changes painter order.
    std::vector<SceneImageSource*> sources;
    std::unordered_map<std::string, SceneImageSource*> publishers;
    for (const auto layer_id : layerOrder) {
        const auto* owner = FindSceneObject(layer_id);
        if (owner == nullptr || !owner->ImageSource()) continue;
        auto* source = owner->ImageSource().get();
        sources.push_back(source);
        if (const auto& layer = owner->ImageEffectLayer(); layer &&
            layer->DeclaredFinalOutputCapability() != FinalOutputCapability::SceneAuthoredWriter) {
            publishers.emplace(GenImageLayerCompositeTex(layer_id), source);
        }
    }
    std::unordered_set<SceneImageSource*> visited;
    const auto refresh = [&](const auto& self, SceneImageSource* source) -> void {
        // Shared sources are prepared once per resource boundary. Mark entry before following
        // source references so a repeated reference cannot recursively re-enter owner setup.
        if (!visited.insert(source).second) return;
        const auto publisher = publishers.find(std::string(source->TextureName()));
        if (publisher != publishers.end()) self(self, publisher->second);
        source->Refresh(*this);
    };
    for (auto* source : sources) refresh(refresh, source);
}

void SceneObject::SetLayerNode(SceneNode* node) {
    m_layer_node = node;
    m_has_layer_node_slot = true;
    // Sound-only registrations have no spatial draw handle. A structural layer replacement
    // installs the new record atomically with its handle; retiring passes can still own the old
    // record until their draw resources are released, without accessing a destroyed SceneObject.
    m_runtime_transform = node != nullptr ? node->TransformState() : nullptr;
}

void SceneObject::SetTextRuntimeState(TextLayerRuntimeState state) {
    m_text_runtime_state = std::make_unique<TextLayerRuntimeState>(std::move(state));
}

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
        // Pose components and zoom share cubic Hermite interpolation with both endpoint
        // tangents equal to half of the component difference. Evaluate the common weight
        // from the retained path cursor before constructing the view: interpolating an
        // already normalized direction or matrix would change the authored pose curve.
        const float t = static_cast<float>(ratio);
        const float weight = 0.5f * t + 1.5f * t * t - t * t * t;
        out.eye = LerpArray3(lhs.eye, rhs.eye, weight);
        out.center = LerpArray3(lhs.center, rhs.center, weight);
        out.up = LerpArray3(lhs.up, rhs.up, weight);
        out.zoom = lhs.zoom + (rhs.zoom - lhs.zoom) * weight;
        return true;
    }

    out = last;
    return true;
}

void CollectLayerEffectNodes(const Scene& scene, int32_t layer_id, std::vector<SceneNode*>& nodes) {
    auto* effect_layer = const_cast<Scene&>(scene).FindImageEffectLayer(layer_id);
    if (effect_layer == nullptr) return;

    for (size_t effect_index = 0; effect_index < effect_layer->EffectCount(); effect_index++) {
        auto& effect = effect_layer->GetEffect(effect_index);
        for (auto& effect_node : effect->nodes) {
            if (effect_node.sceneNode) nodes.push_back(effect_node.sceneNode.get());
        }
    }
}

void ApplyLayerVisibilityRecursive(Scene& scene, int32_t layer_id, std::unordered_set<int32_t>& visited) {
    if (layer_id == 0 || !visited.insert(layer_id).second) return;

    std::unordered_set<int32_t> visiting;
    const bool effective_visible = IsLayerVisibleImpl(scene, layer_id, visiting);

    for (auto* node : scene.GetLayerRuntimeNodes(layer_id)) {
        if (node != nullptr) {
            // Layer visibility propagation must not overwrite a node's own local visibility
            // contract. Runtime-owned support nodes may intentionally stay hidden even while
            // their authored layer is visible, so the scene system only updates the
            // layer-level flag.
            node->SetLayerVisible(effective_visible);
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
        auto* camera_state = scene.FindCameraLayerState(*it);
        if (camera_state == nullptr || !camera_state->node) continue;
        if (!scene.IsLayerVisible(*it)) continue;
        return { *it, camera_state };
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
    // Auto resolution is one global bool for the whole wallpaper, not a
    // per-texture 1920x1080 test. Compare output pixel area (floatA * floatB)
    // against 1969920 (1920 * 1080 * 0.95). Below → half.
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
    // Destroy callbacks may release generated models and still query their layer owners. Run
    // them while all scene registries, transforms and VFS resources are alive; reverse member
    // destruction order would otherwise dispose of those registries before the script host.
    scriptHost.reset();
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
        // g_TextureNResolution follows the uploaded GPU extent, not the authored .tex header,
        // once mip0 has been dropped.
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

void Scene::DestroySceneObject(int32_t layer_id) {
    auto it = sceneObjects.find(layer_id);
    if (it == sceneObjects.end()) return;

    // Parent notifications may immediately acquire this owner's former destination name.
    // Release its slots before hierarchy cleanup so that allocation observes the new lifetime.
    ReleaseLayerDestinationTargets(layer_id);

    // Deletion ends one authored owner's lifetime. Its direct children become roots with their
    // existing local transforms and identities. Clear those bindings directly, without running
    // resource setup: calling SetLayerParentBinding for each child here would rebuild the dying
    // owner's released effect targets. The script host detaches the surviving drawing resources
    // before this identity cleanup.
    for (const auto child_id : GetLayerChildren(layer_id)) {
        FindSceneObject(child_id)->ClearParentBinding();
        ApplyLayerVisibility(child_id);
    }
    // This notification belongs to the owner's still-live parent. Its final child can be the
    // object being deleted, so it must observe the empty boundary and reselect its destinations.
    ClearLayerParentBinding(layer_id);

    if (const auto* object = it->second.get();
        object != nullptr && object->LayerNode() != nullptr) {
        auto index_it = layerNodeIndex.find(object->LayerNode());
        if (index_it != layerNodeIndex.end() && index_it->second == layer_id) {
            layerNodeIndex.erase(index_it);
        }
    }
    sceneObjects.erase(it);
    modelData.Prune();
}

int32_t Scene::LayerIdForNode(const SceneNode* node) const {
    // The node id is the single back-reference: layer handles carry their authored id from
    // parse, drawing-phase nodes (detached sources, effect passes, the final composite) carry
    // their owner's id, and helper nodes keep the default 0, which callers treat as "no owner".
    return node == nullptr ? 0 : node->ID();
}

void Scene::SetLayerSoundHandle(int32_t layer_id, uint32_t handle) {
    if (layer_id == 0) return;
    EnsureSceneObject(layer_id).SetSoundHandle(handle);
}

std::optional<uint32_t> Scene::GetLayerSoundHandle(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? std::nullopt : object->SoundHandle();
}

void Scene::SetLayerInitialConfigJson(int32_t layer_id, std::string config_json) {
    if (layer_id == 0) return;
    EnsureSceneObject(layer_id).SetInitialConfigJson(std::move(config_json));
}

const std::string* Scene::GetLayerInitialConfigJson(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? nullptr : object->InitialConfigJson();
}

void Scene::ClearAllLayerInitialConfigJson() {
    for (auto& [layer_id, object] : sceneObjects) {
        (void)layer_id;
        if (object != nullptr) object->ClearInitialConfigJson();
    }
}

void Scene::SetLayerNode(int32_t layer_id, SceneNode* node) {
    if (layer_id == 0) return;
    auto& object = EnsureSceneObject(layer_id);
    if (SceneNode* previous = object.LayerNode(); previous != nullptr && previous != node) {
        auto index_it = layerNodeIndex.find(previous);
        if (index_it != layerNodeIndex.end() && index_it->second == layer_id) {
            layerNodeIndex.erase(index_it);
        }
    }
    object.SetLayerNode(node);
    if (node != nullptr) layerNodeIndex[node] = layer_id;
}

SceneNode* Scene::GetLayerNode(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? nullptr : object->LayerNode();
}

bool Scene::HasLayerNodeSlot(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object != nullptr && object->HasLayerNodeSlot();
}

void Scene::ClearAllLayerNodeSlots() {
    for (auto& [layer_id, object] : sceneObjects) {
        (void)layer_id;
        if (object != nullptr) object->ClearLayerNodeSlot();
    }
    layerNodeIndex.clear();
}

void Scene::AddLayerRuntimeLight(int32_t layer_id, SceneLight* light) {
    if (layer_id == 0 || light == nullptr) return;
    EnsureSceneObject(layer_id).AddRuntimeLight(light);
}

namespace
{
// Absent-object lookups return a shared empty list so callers keep the former map's
// "no entry" iteration behavior without a per-call allocation.
const std::vector<SceneLight*>        kNoRuntimeLights;
const std::vector<ParticleSubSystem*> kNoRuntimeParticleSubsystems;
} // namespace

const std::vector<SceneLight*>& Scene::GetLayerRuntimeLights(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? kNoRuntimeLights : object->RuntimeLights();
}

void Scene::AddLayerRuntimeParticleSubsystem(int32_t layer_id, ParticleSubSystem* subsystem) {
    if (layer_id == 0 || subsystem == nullptr) return;
    EnsureSceneObject(layer_id).AddRuntimeParticleSubsystem(subsystem);
}

const std::vector<ParticleSubSystem*>&
Scene::GetLayerRuntimeParticleSubsystems(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? kNoRuntimeParticleSubsystems : object->RuntimeParticleSubsystems();
}

void Scene::AddLayerRuntimeNode(int32_t layer_id, SceneNode* node) {
    if (layer_id == 0 || node == nullptr) return;
    EnsureSceneObject(layer_id).AddRuntimeNode(node);
}

namespace
{
const std::vector<SceneNode*> kNoRuntimeNodes;
} // namespace

const std::vector<SceneNode*>& Scene::GetLayerRuntimeNodes(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? kNoRuntimeNodes : object->RuntimeNodes();
}

void Scene::ClearLayerRuntimeNodes(int32_t layer_id) {
    if (auto* object = FindSceneObject(layer_id)) object->ClearRuntimeNodes();
}

void Scene::SetTextLayerState(int32_t layer_id, TextLayerRuntimeState state) {
    if (layer_id == 0) return;
    EnsureSceneObject(layer_id).SetTextRuntimeState(std::move(state));
}

void Scene::SetCameraLayerState(int32_t layer_id, CameraLayerRuntimeState state) {
    if (layer_id == 0) return;
    EnsureSceneObject(layer_id).SetCameraRuntimeState(std::move(state));
}

void Scene::MarkLayerOffscreenDependencySource(int32_t layer_id) {
    if (layer_id == 0) return;
    EnsureSceneObject(layer_id).MarkOffscreenDependencySource();
}

bool Scene::IsLayerOffscreenDependencySource(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object != nullptr && object->IsOffscreenDependencySource();
}

bool Scene::IsRenderOrderProxyNode(const SceneNode* node) const {
    if (node == nullptr || sceneGraph == nullptr || node->Parent() != sceneGraph.get()) {
        return false;
    }
    const int32_t layer_id = LayerIdForNode(node);
    if (layer_id == 0 || GetLayerNode(layer_id) != node) return false;
    const auto* object = FindSceneObject(layer_id);
    return object != nullptr && object->ParentId() != 0 && object->Attachment().empty();
}

std::vector<SceneNode*> Scene::RenderOrderProxyChildrenOf(const SceneNode* parent_node) const {
    std::vector<SceneNode*> children;
    if (parent_node == nullptr || sceneGraph == nullptr) return children;
    const int32_t parent_layer_id = LayerIdForNode(parent_node);
    if (parent_layer_id == 0 || GetLayerNode(parent_layer_id) != parent_node) return children;
    // layerOrder is the authored z-order, which keeps the enumeration deterministic; the render
    // graph re-sorts by the same order anyway.
    for (const auto child_layer_id : layerOrder) {
        if (child_layer_id == parent_layer_id) continue;
        const auto* object = FindSceneObject(child_layer_id);
        if (object == nullptr || object->ParentId() != parent_layer_id ||
            ! object->Attachment().empty()) {
            continue;
        }
        SceneNode* child_node = GetLayerNode(child_layer_id);
        if (child_node == nullptr || child_node->Parent() != sceneGraph.get()) continue;
        children.push_back(child_node);
    }
    return children;
}

Scene::CameraLayerRuntimeState* Scene::FindCameraLayerState(int32_t layer_id) {
    auto* object = FindSceneObject(layer_id);
    return object == nullptr ? nullptr : object->CameraRuntimeState();
}

const Scene::CameraLayerRuntimeState* Scene::FindCameraLayerState(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? nullptr : object->CameraRuntimeState();
}

TextLayerRuntimeState* Scene::FindTextLayerState(int32_t layer_id) {
    auto* object = FindSceneObject(layer_id);
    return object == nullptr ? nullptr : object->TextRuntimeState();
}

const TextLayerRuntimeState* Scene::FindTextLayerState(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? nullptr : object->TextRuntimeState();
}

int32_t Scene::FindLayerIdByNode(const SceneNode* node) const {
    // A null query must not match registered-but-null slots; the index never stores null keys.
    if (node == nullptr) return 0;
    // Scene scripts resolve a handle node back to its layer id on every property write, so this
    // must stay O(1): a linear scan over sceneObjects dominates CPU frames on object-heavy scenes.
    const auto it = layerNodeIndex.find(node);
    return it == layerNodeIndex.end() ? 0 : it->second;
}

void Scene::SetLayerParentBinding(int32_t layer_id, int32_t parent_id, std::string attachment) {
    if (layer_id == 0) return;
    const auto* previous_object = FindSceneObject(layer_id);
    const int32_t previous_parent = previous_object != nullptr ? previous_object->ParentId() : 0;
    if (parent_id == 0 && attachment.empty()) {
        if (auto* object = FindSceneObject(layer_id)) object->ClearParentBinding();
    } else {
        EnsureSceneObject(layer_id).SetParentBinding(parent_id, std::move(attachment));
    }
    if (previous_parent == parent_id) return;

    // Resource setup belongs to the parent whose membership crossed the empty/nonempty boundary.
    // This canonical mutation serves both dynamic creation's identity registration and script
    // reparenting; updating only the script host would miss children created with a parent
    // already present in their config. The initial identity prepass has no materialized bridges
    // yet and only records ancestry.
    const auto refresh_boundary = [&](int32_t owner_id, size_t boundary_count) {
        auto* owner = FindSceneObject(owner_id);
        if (owner != nullptr && GetLayerChildren(owner_id).size() == boundary_count) {
            owner->RefreshResources(*this);
        }
    };
    refresh_boundary(previous_parent, 0);
    refresh_boundary(parent_id, 1);
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
    SetLayerParentBinding(layer_id, 0, {});
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

    auto& object = EnsureSceneObject(layer_id);
    if (object.LocalVisible() == visible) return;
    object.SetLocalVisible(visible);
    if (effectCommandPlanUsesVisibility) MarkRenderGraphTopologyDirty();
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
    if (HasCameraLayers()) UpdateActiveCameraLayer();
}

void Scene::ApplyAllLayerVisibility() {
    std::unordered_set<int32_t> visited;
    for (const auto layer_id : layerOrder) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    for (const auto& [layer_id, object] : sceneObjects) {
        if (object != nullptr && object->HasLayerNodeSlot()) {
            ApplyLayerVisibilityRecursive(*this, layer_id, visited);
        }
    }
    if (HasCameraLayers()) UpdateActiveCameraLayer();
}

void Scene::UpdateModelCameraPath() {
    if (!modelCameraPathEnabled || modelCameraPathSegments.empty() ||
        modelPerspectiveCameraName.empty()) {
        return;
    }

    // A camera layer temporarily owns the shared view and pauses path playback. Resolve this
    // frame's effective visibility, including ancestors, rather than the previous active id:
    // scripts may have hidden or restored a parent just before frame preparation. Neither the
    // path pose nor its clock changes while any selected camera layer owns the view.
    if (FindActiveCameraLayer(*this).second != nullptr) return;

    auto camera_it = cameras.find(modelPerspectiveCameraName);
    if (camera_it == cameras.end() || !camera_it->second) return;

    double total_duration = 0.0;
    for (const auto& segment : modelCameraPathSegments) {
        total_duration += std::max(0.0, segment.duration);
    }
    if (total_duration <= 1e-9) return;

    const double sample_time = modelCameraPathTime;
    double path_time = std::fmod(std::max(0.0, sample_time), total_duration);
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

    // Publish the full sample before either frame projection consumes it. The separate retained
    // pose survives camera-layer selection; the perspective camera also supplies auxiliary
    // consumers. Neither publication advances the cursor, and layer ownership retains both.
    cameraPathPose = { .eye = sample.eye, .center = sample.center, .up = sample.up };
    cameraPathZoom = sample.zoom;
    camera_it->second->SetExplicitView(ToVector3d(sample.eye),
                                       ToVector3d(sample.center),
                                       ToVector3d(sample.up));
    UpdateLinkedCamera(modelPerspectiveCameraName);

    // Sample before advancing, using the same scaled frame delta supplied to scripts. The first
    // release therefore publishes time zero, and later releases resume the retained cursor instead
    // of including time spent under camera-layer ownership. Frame preparation is the sole caller;
    // visibility setters and repeated camera/uniform consumers must not advance playback again.
    modelCameraPathTime += frameTime;
    if (wallpaper::diagnostics::Options().trace_scene_projection) {
        LOG_INFO("SceneCameraPathSample: scene-time=%.6f path-time=%.9f next-time=%.9f "
                 "segment=%d local-time=%.9f delta=%.9f zoom=%.9f",
                 elapsingTime,
                 sample_time,
                 modelCameraPathTime,
                 active_segment,
                 local_time,
                 frameTime,
                 cameraPathZoom);
    }

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

Eigen::Vector3f Scene::FrameEyePosition() const {
    // Lighting uniforms and volume containment share the frame eye, independently of any private
    // raster camera. A node-driven orthographic camera already carries the canvas half-size in
    // X/Y. A path camera keeps its raw eye for look-at and centers the destination separately, so
    // only that selected path needs the canvas adjustment here. The frame eye's fixed Z never
    // replaces the eye used to construct the actual view matrix.
    constexpr float kOrthographicSceneEyeZ = 2000.0f;
    Eigen::Vector3f eye = activeCamera->GetPosition().cast<float>();
    if (cameraOrthographic) {
        if (activeCameraLayerId == 0 && modelCameraPathEnabled) {
            eye.x() += static_cast<float>(ortho[0]) * 0.5f;
            eye.y() += static_cast<float>(ortho[1]) * 0.5f;
        }
        eye.z() = kOrthographicSceneEyeZ;
    }
    return eye;
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

double Scene::ResolveOrthographicCameraZoom() {
    // Perspective scenes still use the shared orthographic camera for auxiliary draws. General
    // zoom belongs to the orthographic scene projection and must not scale those consumers.
    if (!cameraOrthographic) return 1.0;
    // Scene zoom and camera zoom are independent factors. Resolve the same effective layer
    // selection for immediate property updates and later framebuffer/aspect framing, so neither
    // consumer can replace the scene factor or discard the path sample. This is a read of the
    // current sample only: repeated projection work must never advance path playback.
    auto [layer_id, layer] = FindActiveCameraLayer(*this);
    const double selected_zoom =
        layer != nullptr ? layer->zoom : modelCameraPathEnabled ? cameraPathZoom : 1.0;
    return SanitizeCameraZoom(defaultGlobalCameraZoom * selected_zoom, layer_id);
}

void Scene::UpdateActiveCameraLayer() {
    auto [next_layer_id, camera_layer] = FindActiveCameraLayer(*this);

    if (!cameraOrthographic) {
        // Perspective scenes have a single shared view. A camera layer, when present, owns that
        // view per frame: the layer node's world translation is the eye, the node's local -Z axis
        // is the look direction, and its +Y axis is the up vector. Without a camera layer,
        // select the authored scene pose or this frame's camera-path sample explicitly.
        if (modelPerspectiveCameraName.empty()) return;
        auto camera_it = cameras.find(modelPerspectiveCameraName);
        if (camera_it == cameras.end() || !camera_it->second) return;
        auto& camera = *camera_it->second;

        if (camera_layer != nullptr && camera_layer->node) {
            auto& node = *camera_layer->node;
            // Routed parent layers are intentionally kept as root-owned proxy nodes in the
            // physical scene graph. Their authored parent transform is resolved by the shader
            // updater's WPNodeTransformResolver, which also handles nested routed parents and
            // attachments. Reuse that resolver here so the active camera consumes the same world
            // matrix as the model draw path; reading node.ModelTrans() alone would discard the
            // CameraBoneMoveMesh roll even though the scene script has already written it.
            const Eigen::Matrix4d model =
                shaderValueUpdater->ResolveModelTransformForProjection(&node, nullptr, false);
            const Eigen::Vector3d eye   = model.block<3, 1>(0, 3);
            Eigen::Vector3d       zaxis = model.block<3, 1>(0, 2);
            Eigen::Vector3d       yaxis = model.block<3, 1>(0, 1);
            if (zaxis.norm() > 1e-12 && yaxis.norm() > 1e-12) {
                zaxis.normalize();
                yaxis.normalize();
                camera.SetExplicitView(eye, eye - zaxis, yaxis);
            }
        } else if (camera_layer == nullptr) {
            // Selection may also run from a visibility setter before the next playback tick.
            // Restore all three vectors from scene-owned state so layer release cannot retain
            // the hidden layer's view or advance the path just to reconstruct its current pose.
            const auto& pose = modelCameraPathEnabled ? cameraPathPose : authoredCameraPose;
            camera.SetExplicitView(ToVector3d(pose.eye),
                                   ToVector3d(pose.center),
                                   ToVector3d(pose.up));
        }

        // Every frame selects the layer's FOV when a camera layer is active, otherwise the latest
        // general FOV. Apply scene clip planes in either case. Updating only inside the layer
        // branch leaves scripts without a layer writing an unused camera, and retains an old
        // layer FOV after the layer becomes hidden. The FOV clamp affects only the effective
        // angle; raw general/layer getters still return their stored values.
        constexpr float kMinimumFrameFov = 0.1f;
        constexpr float kMaximumFrameFov = 179.9f;
        const float selected_fov = camera_layer != nullptr ? camera_layer->fov
                                                          : generalProjection.fov;
        const float effective_fov =
            std::max(kMinimumFrameFov, std::min(kMaximumFrameFov, selected_fov));
        camera.SetNearClip(generalProjection.nearClip);
        camera.SetFarClip(generalProjection.farClip);
        ApplyCameraProjectionState(*this,
                                   modelPerspectiveCameraName,
                                   camera,
                                   defaultGlobalCameraZoom,
                                   effective_fov,
                                   next_layer_id);

        activeCamera = camera_it->second.get();
        if (activeCameraLayerId != next_layer_id) {
            LOG_INFO("SceneCameraLayerActive: previous=%d active=%d camera='%s' perspective=true "
                     "origin=[%.3f, %.3f, %.3f]",
                     activeCameraLayerId,
                     next_layer_id,
                     modelPerspectiveCameraName.c_str(),
                     camera_layer != nullptr ? camera_layer->origin[0] : 0.0f,
                     camera_layer != nullptr ? camera_layer->origin[1] : 0.0f,
                     camera_layer != nullptr ? camera_layer->origin[2] : 0.0f);
            activeCameraLayerId = next_layer_id;
        }
        return;
    }

    std::string camera_name = "global";
    std::shared_ptr<SceneNode> camera_node = defaultGlobalCameraNode;
    const double zoom = ResolveOrthographicCameraZoom();
    float fov = 50.0f;

    if (camera_layer != nullptr) {
        camera_name = camera_layer->camera_name.empty() ? "global" : camera_layer->camera_name;
        camera_node = camera_layer->node;
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

    auto& camera = *camera_it->second;
    if (camera_layer == nullptr && modelCameraPathEnabled) {
        // Orthographic paths use the same sampled look-at as the perspective frame. The
        // symmetric canvas projection additionally needs its anchor in destination space,
        // after the look-at basis, while lighting keeps the separately adjusted frame eye.
        camera.SetExplicitView(ToVector3d(cameraPathPose.eye),
                               ToVector3d(cameraPathPose.center),
                               ToVector3d(cameraPathPose.up),
                               Eigen::Vector3d(-static_cast<double>(ortho[0]) * 0.5,
                                               -static_cast<double>(ortho[1]) * 0.5,
                                               0.0));
    } else {
        // Attaching a node retains explicit-view state. Clear the path's view mode when a
        // layer takes over, or its node pose would remain hidden behind the old path sample.
        camera.ClearExplicitView();
        camera.AttatchNode(camera_node);
    }
    ApplyCameraProjectionState(*this,
                               camera_name,
                               camera,
                               zoom,
                               fov,
                               next_layer_id);
    activeCamera = camera_it->second.get();

    if (activeCameraLayerId != next_layer_id) {
        // Log camera-layer transitions only. Keyframed zoom/origin animation would flood
        // the log if every frame were recorded.
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
    auto* effect_layer = FindImageEffectLayer(owner_layer_id);
    if (effect_layer == nullptr || effect_index >= effect_layer->EffectCount()) return nullptr;
    return effect_layer->GetEffect(effect_index).get();
}

const SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id,
                                               uint32_t effect_index) const {
    return const_cast<Scene*>(this)->FindImageEffect(owner_layer_id, effect_index);
}

SceneImageEffectLayer* Scene::FindImageEffectLayer(int32_t owner_layer_id) {
    // The owning SceneObject owns its effect bridge directly; the returned raw pointer is backed
    // by that owning reference and stays valid until the layer identity is destroyed. A destroyed
    // layer resolves to "no effect layer" because the object itself is gone.
    const auto* object = FindSceneObject(owner_layer_id);
    if (object == nullptr) return nullptr;
    return object->ImageEffectLayer().get();
}

const SceneImageEffectLayer* Scene::FindImageEffectLayer(int32_t owner_layer_id) const {
    return const_cast<Scene*>(this)->FindImageEffectLayer(owner_layer_id);
}

SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id, int32_t effect_id) {
    auto* effect_layer = FindImageEffectLayer(owner_layer_id);
    if (effect_layer == nullptr) return nullptr;

    for (std::size_t effect_index = 0; effect_index < effect_layer->EffectCount();
         effect_index++) {
        auto& effect = effect_layer->GetEffect(effect_index);
        if (effect != nullptr && effect->EffectId() == effect_id) return effect.get();
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
    return effect != nullptr && ApplyEffectLocalVisibility(*effect, visible);
}

void Scene::QueueRenderTargetClear(std::string target, std::array<float, 4> color,
                                    int32_t owner_layer_id, int32_t effect_id, bool setup) {
    const auto sequence = ++m_render_target_clear_sequence;
    m_pending_render_target_clears.push_back(
        { sequence, std::move(target), color, owner_layer_id, effect_id, setup });
    MarkRenderGraphTopologyDirty();
    if (wallpaper::diagnostics::Options().trace_render_commands) {
        LOG_INFO("SceneRenderTargetClearQueue: sequence=%llu layer=%d effect=%d reason=%s "
                 "target='%s' color=[%.9g %.9g %.9g %.9g]",
                 static_cast<unsigned long long>(sequence), owner_layer_id, effect_id,
                 setup ? "setup" : "function", m_pending_render_target_clears.back().target.c_str(),
                 color[0], color[1], color[2], color[3]);
    }
}

void Scene::CommitRenderTargetClears(uint64_t through_sequence) {
    const auto end = std::find_if(m_pending_render_target_clears.begin(),
                                  m_pending_render_target_clears.end(),
                                  [through_sequence](const auto& request) {
                                      return request.sequence > through_sequence;
                                  });
    if (end == m_pending_render_target_clears.begin()) return;
    const auto count = static_cast<std::size_t>(end - m_pending_render_target_clears.begin());
    // A queued request may have outlived its owner. Once submitted, nominate its target for the
    // ordinary retained-resource census; actual storage release still waits for completed GPU
    // use. A surviving owner or a later request retains the same target there.
    for (auto request = m_pending_render_target_clears.begin(); request != end; ++request) {
        pendingRenderTargetRetirementKeys.insert(request->target);
    }
    m_pending_render_target_clears.erase(m_pending_render_target_clears.begin(), end);
    MarkRenderGraphTopologyDirty();
    if (wallpaper::diagnostics::Options().trace_render_commands) {
        LOG_INFO("SceneRenderTargetClearCommit: through=%llu count=%zu pending=%zu",
                 static_cast<unsigned long long>(through_sequence), count,
                 m_pending_render_target_clears.size());
    }
}

bool Scene::SetEffectLocalVisibilityById(int32_t owner_layer_id, int32_t effect_id,
                                         bool visible) {
    auto* effect = FindImageEffectById(owner_layer_id, effect_id);
    return effect != nullptr && ApplyEffectLocalVisibility(*effect, visible);
}

bool Scene::ApplyEffectLocalVisibility(SceneImageEffect& effect, bool visible) {
    if (effect.LocalVisible() == visible) return true;
    const int32_t layer_id = effect.OwnerLayerId();
    effect.SetLocalVisible(visible);
    if (auto* layer = FindImageEffectLayer(layer_id)) layer->RefreshPuppetPublicationState();
    ApplyLayerVisibility(layer_id);
    // Skipping an effect changes the next input, the final authored writer and its matrix phase.
    // Rebuild that sequence before drawing; the renderer's residency diff keeps unaffected GPU
    // resources alive. Repeated script writes of the same value never enter this path.
    MarkRenderGraphTopologyDirty();
    LOG_INFO("SceneEffectVisibilityChange: layer=%d effect-id=%d visible=%s",
             layer_id, effect.EffectId(), visible ? "true" : "false");
    if (FindTextLayerState(layer_id) != nullptr) {
        return SyncTextLayerEffectVisibility(*this, layer_id);
    }
    // Changing the visible composition count reruns the retained targets' setup clears even
    // when the image dimensions stay the same. Hiding a layer alone does not enter this path.
    if (auto* layer = FindImageEffectLayer(layer_id)) layer->QueueSetupClears(*this);
    return true;
}

}

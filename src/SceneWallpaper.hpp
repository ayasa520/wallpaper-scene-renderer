#pragma once
#include <memory>
#include <string_view>
#include <functional>
#include <vector>
#include "Type.hpp"
#include "Swapchain/ExSwapchain.hpp"
#include "VulkanRender/OffscreenFrameReleaseCallback.hpp"

namespace wallpaper
{

using FirstFrameCallback = std::function<void()>;

constexpr std::string_view PROPERTY_SOURCE               = "source";
constexpr std::string_view PROPERTY_ASSETS               = "assets";
constexpr std::string_view PROPERTY_FPS                  = "fps";
constexpr std::string_view PROPERTY_FILLMODE             = "fillmode";
constexpr std::string_view PROPERTY_SPEED                = "speed";
constexpr std::string_view PROPERTY_GRAPHIVZ             = "graphivz";
constexpr std::string_view PROPERTY_VOLUME               = "volume";
constexpr std::string_view PROPERTY_MUTED                = "muted";
constexpr std::string_view PROPERTY_CACHE_PATH           = "cache_path";
constexpr std::string_view PROPERTY_FIRST_FRAME_CALLBACK = "first_frame_callback";
// Load-time user properties update the MainHandler state that the next scene
// parse consumes, but they deliberately do not send live script updates to the
// scene that is currently still on screen during a reused-backend project switch.
constexpr std::string_view PROPERTY_LOAD_USER_PROPERTIES = "load_user_properties";
constexpr std::string_view PROPERTY_USER_PROPERTIES      = "user_properties";
constexpr std::string_view PROPERTY_MEDIA_STATE          = "media_state";
constexpr std::string_view PROPERTY_AUDIO_SAMPLES        = "audio_samples";

#include "Core/NoCopyMove.hpp"
class MainHandler;
struct RenderInitInfo;

class SceneWallpaper : NoCopy {
public:
    SceneWallpaper();
    ~SceneWallpaper();
    bool init();
    bool inited() const;

    void initVulkan(const RenderInitInfo&);
    void setOffscreenFrameReleaseCallback(
        vulkan::OffscreenFrameReleaseCallback callback);
    bool reconfigureOffscreenExport(uint32_t width,
                                    uint32_t height,
                                    TexTiling tiling,
                                    ExternalFrameExportMode export_mode,
                                    uint32_t export_drm_fourcc,
                                    const std::vector<uint64_t>& export_drm_modifiers,
                                    ExternalFrameMemoryPreference memory_preference);

    void play();
    void pause();
    void requestFrame();
    void mouseInput(double x, double y);
    void mouseLeftButton(bool down);

    void setPropertyBool(std::string_view, bool);
    void setPropertyInt32(std::string_view, int32_t);
    void setPropertyFloat(std::string_view, float);
    void setPropertyString(std::string_view, std::string);
    void setPropertyObject(std::string_view, std::shared_ptr<void>);

    ExSwapchain* exSwapchain() const;

private:
    bool m_inited { false };

private:
    friend class MainHandler;

    bool                         m_offscreen { false };
    std::shared_ptr<MainHandler> m_main_handler;
};
} // namespace wallpaper

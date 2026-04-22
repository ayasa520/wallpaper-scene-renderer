find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
pkg_check_modules(PANGOCAIRO REQUIRED pangocairo)
pkg_check_modules(FONTCONFIG REQUIRED fontconfig)
pkg_check_modules(FREETYPE2 REQUIRED freetype2)

option(ENABLE_RENDERDOC "Build with renderdoc api" OFF)

if(ENABLE_RENDERDOC)
  add_compile_definitions(ENABLE_RENDERDOC_API=1)
endif()

include(TestBigEndian)
test_big_endian(ENDIAN)
if(ENDIAN)
  add_compile_definitions(WP_BIG_ENDIAN)
endif()

set(warn_opts
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wno-unused-variable
    CACHE INTERNAL "")

include_directories(.)

add_subdirectory(Fs)
add_subdirectory(Utils)
add_subdirectory(Timer)
add_subdirectory(Audio)
add_subdirectory(Scene)
add_subdirectory(Looper)
add_subdirectory(Vulkan)
add_subdirectory(Particle)
add_subdirectory(VulkanRender)
add_subdirectory(RenderGraph)

add_library(
  ${PROJECT_NAME} STATIC
  WPJson.cpp
  WPDynamicValue.cpp
  WPPropertyAnimation.cpp
  WPSceneScriptMedia.cpp
  WPScriptRuntime.cpp
  WPPkgFs.cpp
  Type.cpp
  WPUserSetting.cpp
  # Shared effect parsing is used by both image objects and first-class text objects, so it is
  # compiled as its own unit instead of being hidden behind WPImageObject.
  wpscene/WPEffect.cpp
  wpscene/WPImageObject.cpp
  wpscene/WPParticleObject.cpp
  wpscene/WPMaterial.cpp
  wpscene/WPScene.cpp
  wpscene/WPLightObject.cpp
  WPParticleParser.cpp
  WPShaderParser.cpp
  WPSceneParser.cpp
  WPSceneScriptHost.cpp
  WPNodeTransformResolver.cpp
  WPShaderValueUpdater.cpp
  WPTexImageParser.cpp
  WPSyntheticImageParser.cpp
  WPSoundParser.cpp
  WPTextLayer.cpp
  WPMdlParser.cpp
  WPPuppet.cpp
  SceneWallpaper.cpp)

set(InteralLib
    wpAudio
    wpLooper
    wpVulkanRender
    wpFs
    wpScene
    wpVulkan
    wpRGraph
    wpParticle
    wpTimer)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC wpUtils
  PRIVATE ${InteralLib} nlohmann_json ${LZ4_LIBRARIES} wpquickjs
          ${PANGOCAIRO_LIBRARIES} ${FONTCONFIG_LIBRARIES} ${FREETYPE2_LIBRARIES}
          ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

target_include_directories(${PROJECT_NAME}
                           PUBLIC .
                                  Swapchain
                           PRIVATE ${PANGOCAIRO_INCLUDE_DIRS}
                                   ${FONTCONFIG_INCLUDE_DIRS}
                                   ${FREETYPE2_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME}
                       PRIVATE ${PANGOCAIRO_CFLAGS_OTHER}
                               ${FONTCONFIG_CFLAGS_OTHER}
                               ${FREETYPE2_CFLAGS_OTHER})
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

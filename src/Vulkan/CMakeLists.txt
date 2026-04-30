set(LIB_NAME wpVulkan)

find_package(Vulkan REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-allocators-1.0
    gstreamer-video-1.0
    gstreamer-va-1.0
    gstreamer-cuda-1.0
    libdrm
    libva
    gio-2.0)

add_library(${LIB_NAME}
STATIC
Instance.cpp
Device.cpp
GraphicsPipeline.cpp
Shader.cpp
StagingBuffer.cpp
Swapchain.cpp
TextureCache.cpp
VideoTextureCache.cpp
Parameters.cpp
Vma.cpp
vulkan_wrapper.cpp
)

target_link_libraries(${LIB_NAME}
PUBLIC
	glslang
    spirv-reflect-static
	${Vulkan_LIBRARIES}
    wpScene
PRIVATE
    ${GSTREAMER_LIBRARIES}
	SPIRV
)
target_include_directories(${LIB_NAME} PUBLIC include PRIVATE include/Vulkan include/vkk ${GSTREAMER_INCLUDE_DIRS})
target_compile_definitions(${LIB_NAME} PRIVATE GST_USE_UNSTABLE_API)
target_compile_options(${LIB_NAME} PRIVATE ${warn_opts} -Wno-missing-field-initializers ${GSTREAMER_CFLAGS_OTHER})
set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

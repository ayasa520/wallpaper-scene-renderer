set(LIB_NAME wpScene)

add_library(${LIB_NAME}
STATIC

Scene.cpp
SceneCamera.cpp
SceneImageEffectLayer.cpp
SceneIndexArray.cpp
SceneNode.cpp
SceneVertexArray.cpp
SceneShader.cpp
)

target_link_libraries(${LIB_NAME}
PUBLIC
    wpUtils
    nlohmann_json
PRIVATE
    wpFs
    wpParticle
)
target_include_directories(${LIB_NAME} PUBLIC include PRIVATE include/Scene)
target_compile_options(${LIB_NAME} PRIVATE ${warn_opts})
set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
  add_executable(scene_identity_test tests/scene_identity_test.cpp)
  # Scene.h reaches into parser-side types (script host, text runtime state), so the test
  # links the full renderer library; wescene-renderer is defined later in src/CMakeLists.txt,
  # which is fine because CMake resolves target references at generate time.
  target_link_libraries(scene_identity_test PRIVATE ${LIB_NAME} wescene-renderer)
  target_include_directories(scene_identity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  add_test(NAME scene_identity COMMAND scene_identity_test)
endif()

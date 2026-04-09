cmake_minimum_required(VERSION 3.14)
project(wescene-renderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

include_directories(SYSTEM third_party)

link_libraries(Eigen3::Eigen)

add_library(
    wpquickjs STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs/dtoa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs/libregexp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs/libunicode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs/quickjs.c
)
target_compile_definitions(wpquickjs PRIVATE QUICKJS_NG_BUILD _GNU_SOURCE)
target_include_directories(wpquickjs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs)
find_library(M_LIBRARIES m)
if(M_LIBRARIES)
    target_link_libraries(wpquickjs PUBLIC ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} m)
else()
    target_link_libraries(wpquickjs PUBLIC ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()
set_property(TARGET wpquickjs PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(third_party)
add_subdirectory(src)

if(BUILD_QML)
    find_package(Qt6 COMPONENTS Gui Quick Qml REQUIRED)

    set(CMAKE_AUTOMOC ON) 
    set(CMAKE_AUTORCC ON) 
    set(CMAKE_AUTOUIC ON)

    add_library(${PROJECT_NAME}-qml
    STATIC
        qml_helper/SceneBackend.cpp
        qml_helper/glExtra.cpp
    )
    target_link_libraries(${PROJECT_NAME}-qml
    PUBLIC
        Qt::Quick
        Qt::Gui
        glad
        ${PROJECT_NAME}
    )
    target_include_directories(${PROJECT_NAME}-qml PUBLIC qml_helper third_party)

    set(CMAKE_AUTOMOC OFF) 
    set(CMAKE_AUTORCC OFF) 
    set(CMAKE_AUTOUIC OFF) 
endif()

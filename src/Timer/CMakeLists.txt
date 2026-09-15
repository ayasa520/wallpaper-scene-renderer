set(LIB_NAME wpTimer)

add_library(${LIB_NAME}
STATIC
ThreadTimer.cpp
FrameTimer.cpp
)

target_link_libraries(${LIB_NAME} PUBLIC wpUtils)
target_include_directories(${LIB_NAME} PUBLIC include PRIVATE include/Timer)
target_compile_options(${LIB_NAME} PRIVATE ${warn_opts})
set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
  # Lockstep capture contracts: frame-timer request gate, swapchain publish numbering and the
  # determinism helpers. Header-only dependencies apart from wpTimer, so no GPU is needed.
  add_executable(lockstep_contract_test tests/lockstep_contract_test.cpp)
  target_link_libraries(lockstep_contract_test PRIVATE ${LIB_NAME} ${CMAKE_THREAD_LIBS_INIT})
  target_include_directories(lockstep_contract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  add_test(NAME lockstep_contract COMMAND lockstep_contract_test)
endif()

set(LIB_NAME wpUtils)

find_package(spdlog 1.14 REQUIRED)

add_library(${LIB_NAME}
STATIC
Logging.cpp
FpsCounter.cpp
Algorism.cpp	
Sha.cpp
DynamicLibrary.cpp
)

target_link_libraries(${LIB_NAME} PUBLIC spdlog::spdlog)
target_include_directories(${LIB_NAME} PUBLIC include PRIVATE include/Utils)
target_compile_options(${LIB_NAME} PRIVATE ${warn_opts})
# Debug is the only instrumented configuration. Propagate its compile-time policy through
# every renderer archive and embedding target because public headers contain diagnostic state.
# Generator expressions also preserve the boundary with multi-configuration generators.
target_compile_definitions(${LIB_NAME} PUBLIC
    WESCENE_ENABLE_DIAGNOSTICS=$<CONFIG:Debug>
    SPDLOG_ACTIVE_LEVEL=$<IF:$<CONFIG:Debug>,SPDLOG_LEVEL_INFO,SPDLOG_LEVEL_WARN>)
target_sources(${LIB_NAME} PRIVATE
    "$<$<CONFIG:Debug>:${CMAKE_CURRENT_SOURCE_DIR}/Diagnostics.cpp>")

# Keep the utility archive on the baseline ISA. The optimized functions use a compiler target
# attribute and are selected only after a runtime CPU check, so accepting an AVX compiler flag
# never turns AVX2/FMA into a process-wide hardware requirement.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles([=[
#if !defined(__i386__) && !defined(__x86_64__)
#error "x86 target required"
#endif
#if !defined(__GNUC__) && !defined(__clang__)
#error "GNU target attributes required"
#endif
#include <immintrin.h>
__attribute__((target("avx2,fma")))
static float probe(const float* values) {
    const __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256 gathered = _mm256_i32gather_ps(values, indices, 4);
    const __m256 result = _mm256_fmadd_ps(gathered, gathered, gathered);
    return _mm_cvtss_f32(_mm256_castps256_ps128(result));
}
int main() {
    float values[8] {};
    __builtin_cpu_init();
    return (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        ? static_cast<int>(probe(values))
        : 0;
}
]=] WPUTILS_HAS_X86_AVX2_DISPATCH)
if(WPUTILS_HAS_X86_AVX2_DISPATCH)
    target_compile_definitions(${LIB_NAME} PRIVATE WPUTILS_HAS_X86_AVX2_DISPATCH=1)
endif()
set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

#include "Algorism.h"
#include "Eigen.h"

#include <bit>
#include <cmath>
#include <cstdint>

#if defined(WPUTILS_HAS_X86_AVX2_DISPATCH)
#include <immintrin.h>
#endif

using namespace wallpaper;
using namespace Eigen;

double algorism::CalculatePersperctiveDistance(double fov, double height) noexcept {
    double k = std::tan(Radians(fov / 2.0f)) * 2.0f;
    return height / k;
}

double algorism::CalculatePersperctiveFov(double distence, double height) noexcept {
    double k     = height / distence / 2.0f;
    double angle = std::atan(k) * 2;
    return angle / Radians(180.0f) * 180.0f;
}

namespace
{
constexpr double grad(int hash, double x, double y, double z) noexcept {
    // http://riven8192.blogspot.com/2010/08/calculate-perlinnoise-twice-as-fast.html
    switch (hash & 0xF) {
    case 0x0: return x + y;
    case 0x1: return -x + y;
    case 0x2: return x - y;
    case 0x3: return -x - y;
    case 0x4: return x + z;
    case 0x5: return -x + z;
    case 0x6: return x - z;
    case 0x7: return -x - z;
    case 0x8: return y + z;
    case 0x9: return -y + z;
    case 0xA: return y - z;
    case 0xB: return -y - z;
    case 0xC: return y + x;
    case 0xD: return -y + z;
    case 0xE: return y - x;
    case 0xF: return -y - z;
    default: return 0; // never happens
    }
    /*
    int    h = hash & 15;     // CONVERT LO 4 BITS OF HASH CODE
    double u = h < 8 ? x : y, // INTO 12 GRADIENT DIRECTIONS.
    v    = h < 4                ? y
           : h == 12 || h == 14 ? x
                                : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    */
}

struct GradVec {
    double a, b, c;
};

// The constant gradient vector selected by grad() for each hash value. grad(hash, x, y, z) is the
// dot product of this vector with (x, y, z), which is what makes the analytic noise derivative
// below possible: d(grad)/dx is simply the .a component and so on.
constexpr GradVec grad_vector(int hash) noexcept {
    switch (hash & 0xF) {
    case 0x0: return { 1, 1, 0 };
    case 0x1: return { -1, 1, 0 };
    case 0x2: return { 1, -1, 0 };
    case 0x3: return { -1, -1, 0 };
    case 0x4: return { 1, 0, 1 };
    case 0x5: return { -1, 0, 1 };
    case 0x6: return { 1, 0, -1 };
    case 0x7: return { -1, 0, -1 };
    case 0x8: return { 0, 1, 1 };
    case 0x9: return { 0, -1, 1 };
    case 0xA: return { 0, 1, -1 };
    case 0xB: return { 0, -1, -1 };
    case 0xC: return { 1, 1, 0 };
    case 0xD: return { 0, -1, 1 };
    case 0xE: return { -1, 1, 0 };
    case 0xF: return { 0, -1, -1 };
    default: return { 0, 0, 0 }; // never happens
    }
}

// d/dt of PerlinEase(t) = 6t^5 - 15t^4 + 10t^3.
constexpr double PerlinEaseDerivative(double t) noexcept {
    return 30.0 * t * t * (t - 1.0) * (t - 1.0);
}

constexpr unsigned char perlin_perm[] = {
    151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,   225, 140, 36,
    103, 30,  69,  142, 8,   99,  37,  240, 21,  10,  23,  190, 6,   148, 247, 120, 234, 75,
    0,   26,  197, 62,  94,  252, 219, 203, 117, 35,  11,  32,  57,  177, 33,  88,  237, 149,
    56,  87,  174, 20,  125, 136, 171, 168, 68,  175, 74,  165, 71,  134, 139, 48,  27,  166,
    77,  146, 158, 231, 83,  111, 229, 122, 60,  211, 133, 230, 220, 105, 92,  41,  55,  46,
    245, 40,  244, 102, 143, 54,  65,  25,  63,  161, 1,   216, 80,  73,  209, 76,  132, 187,
    208, 89,  18,  169, 200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186,
    3,   64,  52,  217, 226, 250, 124, 123, 5,   202, 38,  147, 118, 126, 255, 82,  85,  212,
    207, 206, 59,  227, 47,  16,  58,  17,  182, 189, 28,  42,  223, 183, 170, 213, 119, 248,
    152, 2,   44,  154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,   129, 22,  39,  253,
    19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104, 218, 246, 97,  228, 251, 34,
    242, 193, 238, 210, 144, 12,  191, 179, 162, 241, 81,  51,  145, 235, 249, 14,  239, 107,
    49,  192, 214, 31,  181, 199, 106, 157, 184, 84,  204, 176, 115, 121, 50,  45,  127, 4,
    150, 254, 138, 236, 205, 93,  222, 114, 67,  29,  24,  72,  243, 141, 128, 195, 78,  66,
    215, 61,  156, 180,

    151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,   225, 140, 36,
    103, 30,  69,  142, 8,   99,  37,  240, 21,  10,  23,  190, 6,   148, 247, 120, 234, 75,
    0,   26,  197, 62,  94,  252, 219, 203, 117, 35,  11,  32,  57,  177, 33,  88,  237, 149,
    56,  87,  174, 20,  125, 136, 171, 168, 68,  175, 74,  165, 71,  134, 139, 48,  27,  166,
    77,  146, 158, 231, 83,  111, 229, 122, 60,  211, 133, 230, 220, 105, 92,  41,  55,  46,
    245, 40,  244, 102, 143, 54,  65,  25,  63,  161, 1,   216, 80,  73,  209, 76,  132, 187,
    208, 89,  18,  169, 200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186,
    3,   64,  52,  217, 226, 250, 124, 123, 5,   202, 38,  147, 118, 126, 255, 82,  85,  212,
    207, 206, 59,  227, 47,  16,  58,  17,  182, 189, 28,  42,  223, 183, 170, 213, 119, 248,
    152, 2,   44,  154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,   129, 22,  39,  253,
    19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104, 218, 246, 97,  228, 251, 34,
    242, 193, 238, 210, 144, 12,  191, 179, 162, 241, 81,  51,  145, 235, 249, 14,  239, 107,
    49,  192, 214, 31,  181, 199, 106, 157, 184, 84,  204, 176, 115, 121, 50,  45,  127, 4,
    150, 254, 138, 236, 205, 93,  222, 114, 67,  29,  24,  72,  243, 141, 128, 195, 78,  66,
    215, 61,  156, 180
};
} // namespace

// from https://mrl.cs.nyu.edu/~perlin/noise/
double algorism::PerlinNoise(double x, double y, double z) noexcept {
    const unsigned char* perm = perlin_perm;
    int X = (int)floor(x) & 255, // FIND UNIT CUBE THAT
        Y = (int)floor(y) & 255, // CONTAINS POINT.
        Z = (int)floor(z) & 255;

    x -= floor(x); // FIND RELATIVE X,Y,Z
    y -= floor(y); // OF POINT IN CUBE.
    z -= floor(z);

    double u = PerlinEase(x), // COMPUTE FADE CURVES
        v    = PerlinEase(y), // FOR EACH OF X,Y,Z.
        w    = PerlinEase(z);

    int A = perm[X] + Y, AA = perm[A] + Z, AB = perm[A + 1] + Z,     // HASH COORDINATES OF
        B = perm[X + 1] + Y, BA = perm[B] + Z, BB = perm[B + 1] + Z; // THE 8 CUBE CORNERS,

    return lerp(
        w,
        lerp(v,
             lerp(u,
                  grad(perm[AA], x, y, z),      // AND ADD
                  grad(perm[BA], x - 1, y, z)), // BLENDED
             lerp(u,
                  grad(perm[AB], x, y - 1, z),       // RESULTS
                  grad(perm[BB], x - 1, y - 1, z))), // FROM  8
        lerp(
            v,
            lerp(u,
                 grad(perm[AA + 1], x, y, z - 1),      // CORNERS
                 grad(perm[BA + 1], x - 1, y, z - 1)), // OF CUBE
            lerp(u, grad(perm[AB + 1], x, y - 1, z - 1), grad(perm[BB + 1], x - 1, y - 1, z - 1))));
}

/*
 * Analytic spatial gradient of PerlinNoise(). CurlNoise() previously evaluated the noise field at
 * six offset points per vector component (18 PerlinNoise calls per particle per frame) to build a
 * central-difference curl; profiling the turbulence particle operator showed PerlinNoise dominating
 * whole-process CPU time on emitter-heavy wallpapers. The trilinear blend of corner dot products is
 * differentiable in closed form: each corner contributes its constant gradient vector directly plus
 * an interpolation-weight term through the fade-curve derivative. One evaluation therefore yields
 * the exact d(noise)/d(x,y,z), and CurlNoise needs only three of these instead of 18 noise calls.
 */
Eigen::Vector3d algorism::PerlinNoiseGradient(double x, double y, double z) noexcept {
    const unsigned char* perm = perlin_perm;
    int X = (int)floor(x) & 255, Y = (int)floor(y) & 255, Z = (int)floor(z) & 255;

    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    const double u = PerlinEase(x), v = PerlinEase(y), w = PerlinEase(z);
    const double du = PerlinEaseDerivative(x), dv = PerlinEaseDerivative(y),
                 dw = PerlinEaseDerivative(z);

    int A = perm[X] + Y, AA = perm[A] + Z, AB = perm[A + 1] + Z;
    int B = perm[X + 1] + Y, BA = perm[B] + Z, BB = perm[B + 1] + Z;

    const GradVec g000 = grad_vector(perm[AA]);
    const GradVec g100 = grad_vector(perm[BA]);
    const GradVec g010 = grad_vector(perm[AB]);
    const GradVec g110 = grad_vector(perm[BB]);
    const GradVec g001 = grad_vector(perm[AA + 1]);
    const GradVec g101 = grad_vector(perm[BA + 1]);
    const GradVec g011 = grad_vector(perm[AB + 1]);
    const GradVec g111 = grad_vector(perm[BB + 1]);

    const double d000 = g000.a * x + g000.b * y + g000.c * z;
    const double d100 = g100.a * (x - 1) + g100.b * y + g100.c * z;
    const double d010 = g010.a * x + g010.b * (y - 1) + g010.c * z;
    const double d110 = g110.a * (x - 1) + g110.b * (y - 1) + g110.c * z;
    const double d001 = g001.a * x + g001.b * y + g001.c * (z - 1);
    const double d101 = g101.a * (x - 1) + g101.b * y + g101.c * (z - 1);
    const double d011 = g011.a * x + g011.b * (y - 1) + g011.c * (z - 1);
    const double d111 = g111.a * (x - 1) + g111.b * (y - 1) + g111.c * (z - 1);

    const auto trilerp = [&](double c000, double c100, double c010, double c110, double c001,
                             double c101, double c011, double c111) {
        return lerp(w,
                    lerp(v, lerp(u, c000, c100), lerp(u, c010, c110)),
                    lerp(v, lerp(u, c001, c101), lerp(u, c011, c111)));
    };

    // Direct term: every corner dot product varies linearly in x/y/z with its gradient component.
    const double gx = trilerp(g000.a, g100.a, g010.a, g110.a, g001.a, g101.a, g011.a, g111.a);
    const double gy = trilerp(g000.b, g100.b, g010.b, g110.b, g001.b, g101.b, g011.b, g111.b);
    const double gz = trilerp(g000.c, g100.c, g010.c, g110.c, g001.c, g101.c, g011.c, g111.c);

    // Weight term: derivative of the trilinear interpolation weights via the fade-curve slope.
    const double x00 = lerp(u, d000, d100), x10 = lerp(u, d010, d110);
    const double x01 = lerp(u, d001, d101), x11 = lerp(u, d011, d111);
    const double dndu =
        lerp(w, lerp(v, d100 - d000, d110 - d010), lerp(v, d101 - d001, d111 - d011));
    const double dndv = lerp(w, x10 - x00, x11 - x01);
    const double dndw = lerp(v, x01, x11) - lerp(v, x00, x10);

    return { gx + du * dndu, gy + dv * dndv, gz + dw * dndw };
}

namespace
{
constexpr float PerlinEaseF(float t) noexcept { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
constexpr float PerlinEaseDerivativeF(float t) noexcept {
    return 30.0f * t * t * (t - 1.0f) * (t - 1.0f);
}

struct GradVecF {
    float a, b, c;
};

// Same 16 Ken Perlin corner vectors as grad_vector(), stored densely so the turbulence hot path
// does not switch on the low four hash bits for every lattice corner.
constexpr GradVecF kGradTableF[16] = {
    { 1, 1, 0 },  { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 }, { 1, 0, 1 },  { -1, 0, 1 },
    { 1, 0, -1 }, { -1, 0, -1 }, { 0, 1, 1 },  { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 },
    { 1, 1, 0 },  { 0, -1, 1 },  { -1, 1, 0 }, { 0, -1, -1 },
};

constexpr float lerp_f(float t, float a, float b) noexcept { return a + t * (b - a); }
} // namespace

Eigen::Vector3f algorism::PerlinNoiseGradient(float x, float y, float z) noexcept {
    const unsigned char* perm = perlin_perm;
    const int            X    = static_cast<int>(std::floor(x)) & 255;
    const int            Y    = static_cast<int>(std::floor(y)) & 255;
    const int            Z    = static_cast<int>(std::floor(z)) & 255;

    x -= std::floor(x);
    y -= std::floor(y);
    z -= std::floor(z);

    const float u = PerlinEaseF(x), v = PerlinEaseF(y), w = PerlinEaseF(z);
    const float du = PerlinEaseDerivativeF(x), dv = PerlinEaseDerivativeF(y),
                dw = PerlinEaseDerivativeF(z);

    const int A = perm[X] + Y, AA = perm[A] + Z, AB = perm[A + 1] + Z;
    const int B = perm[X + 1] + Y, BA = perm[B] + Z, BB = perm[B + 1] + Z;

    const GradVecF g000 = kGradTableF[perm[AA] & 0xF];
    const GradVecF g100 = kGradTableF[perm[BA] & 0xF];
    const GradVecF g010 = kGradTableF[perm[AB] & 0xF];
    const GradVecF g110 = kGradTableF[perm[BB] & 0xF];
    const GradVecF g001 = kGradTableF[perm[AA + 1] & 0xF];
    const GradVecF g101 = kGradTableF[perm[BA + 1] & 0xF];
    const GradVecF g011 = kGradTableF[perm[AB + 1] & 0xF];
    const GradVecF g111 = kGradTableF[perm[BB + 1] & 0xF];

    const float d000 = g000.a * x + g000.b * y + g000.c * z;
    const float d100 = g100.a * (x - 1.0f) + g100.b * y + g100.c * z;
    const float d010 = g010.a * x + g010.b * (y - 1.0f) + g010.c * z;
    const float d110 = g110.a * (x - 1.0f) + g110.b * (y - 1.0f) + g110.c * z;
    const float d001 = g001.a * x + g001.b * y + g001.c * (z - 1.0f);
    const float d101 = g101.a * (x - 1.0f) + g101.b * y + g101.c * (z - 1.0f);
    const float d011 = g011.a * x + g011.b * (y - 1.0f) + g011.c * (z - 1.0f);
    const float d111 = g111.a * (x - 1.0f) + g111.b * (y - 1.0f) + g111.c * (z - 1.0f);

    const auto trilerp = [&](float c000, float c100, float c010, float c110, float c001, float c101,
                             float c011, float c111) {
        return lerp_f(w,
                      lerp_f(v, lerp_f(u, c000, c100), lerp_f(u, c010, c110)),
                      lerp_f(v, lerp_f(u, c001, c101), lerp_f(u, c011, c111)));
    };

    const float gx = trilerp(g000.a, g100.a, g010.a, g110.a, g001.a, g101.a, g011.a, g111.a);
    const float gy = trilerp(g000.b, g100.b, g010.b, g110.b, g001.b, g101.b, g011.b, g111.b);
    const float gz = trilerp(g000.c, g100.c, g010.c, g110.c, g001.c, g101.c, g011.c, g111.c);

    const float x00 = lerp_f(u, d000, d100), x10 = lerp_f(u, d010, d110);
    const float x01 = lerp_f(u, d001, d101), x11 = lerp_f(u, d011, d111);
    const float dndu =
        lerp_f(w, lerp_f(v, d100 - d000, d110 - d010), lerp_f(v, d101 - d001, d111 - d011));
    const float dndv = lerp_f(w, x10 - x00, x11 - x01);
    const float dndw = lerp_f(v, x01, x11) - lerp_f(v, x00, x10);

    return { gx + du * dndu, gy + dv * dndv, gz + dw * dndw };
}

#if defined(WPUTILS_HAS_X86_AVX2_DISPATCH)
namespace
{
#define WPUTILS_AVX2_FMA_TARGET __attribute__((target("avx2,fma")))

alignas(64) int32_t kPerm32[512];
alignas(32) float   kGradA[16];
alignas(32) float   kGradB[16];
alignas(32) float   kGradC[16];

struct PerlinSimdTables {
    PerlinSimdTables() noexcept {
        for (int i = 0; i < 512; i++) kPerm32[i] = perlin_perm[i];
        for (int i = 0; i < 16; i++) {
            kGradA[i] = kGradTableF[i].a;
            kGradB[i] = kGradTableF[i].b;
            kGradC[i] = kGradTableF[i].c;
        }
    }
};

const PerlinSimdTables kPerlinSimdTables {};

WPUTILS_AVX2_FMA_TARGET inline __m256 Fade8(__m256 t) noexcept {
    const __m256 t2    = _mm256_mul_ps(t, t);
    const __m256 t3    = _mm256_mul_ps(t2, t);
    const __m256 inner = _mm256_fmadd_ps(
        t, _mm256_fmadd_ps(t, _mm256_set1_ps(6.0f), _mm256_set1_ps(-15.0f)), _mm256_set1_ps(10.0f));
    return _mm256_mul_ps(t3, inner);
}

WPUTILS_AVX2_FMA_TARGET inline __m256 FadeDerivative8(__m256 t) noexcept {
    const __m256 tm1 = _mm256_sub_ps(t, _mm256_set1_ps(1.0f));
    return _mm256_mul_ps(_mm256_set1_ps(30.0f),
                         _mm256_mul_ps(_mm256_mul_ps(t, t), _mm256_mul_ps(tm1, tm1)));
}

WPUTILS_AVX2_FMA_TARGET inline __m256 Lerp8(__m256 t, __m256 a, __m256 b) noexcept {
    return _mm256_fmadd_ps(t, _mm256_sub_ps(b, a), a);
}

WPUTILS_AVX2_FMA_TARGET inline __m256
Trilerp8(__m256 u, __m256 v, __m256 w, __m256 c000, __m256 c100, __m256 c010, __m256 c110,
         __m256 c001, __m256 c101, __m256 c011, __m256 c111) noexcept {
    return Lerp8(w,
                 Lerp8(v, Lerp8(u, c000, c100), Lerp8(u, c010, c110)),
                 Lerp8(v, Lerp8(u, c001, c101), Lerp8(u, c011, c111)));
}

struct Grad8 {
    __m256 x;
    __m256 y;
    __m256 z;
};

WPUTILS_AVX2_FMA_TARGET Grad8
PerlinNoiseGradientVec8(__m256 x, __m256 y, __m256 z) noexcept {
    (void)kPerlinSimdTables;
    const __m256  x_floor = _mm256_floor_ps(x);
    const __m256  y_floor = _mm256_floor_ps(y);
    const __m256  z_floor = _mm256_floor_ps(z);
    const __m256i mask255 = _mm256_set1_epi32(255);
    const __m256i Xi      = _mm256_and_si256(_mm256_cvtps_epi32(x_floor), mask255);
    const __m256i Yi      = _mm256_and_si256(_mm256_cvtps_epi32(y_floor), mask255);
    const __m256i Zi      = _mm256_and_si256(_mm256_cvtps_epi32(z_floor), mask255);

    x = _mm256_sub_ps(x, x_floor);
    y = _mm256_sub_ps(y, y_floor);
    z = _mm256_sub_ps(z, z_floor);

    const __m256 u  = Fade8(x);
    const __m256 v  = Fade8(y);
    const __m256 w  = Fade8(z);
    const __m256 du = FadeDerivative8(x);
    const __m256 dv = FadeDerivative8(y);
    const __m256 dw = FadeDerivative8(z);

    const __m256i one = _mm256_set1_epi32(1);
    const __m256i pX  = _mm256_i32gather_epi32(kPerm32, Xi, 4);
    const __m256i pXp = _mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(Xi, one), 4);
    const __m256i A   = _mm256_add_epi32(pX, Yi);
    const __m256i B   = _mm256_add_epi32(pXp, Yi);
    const __m256i pA  = _mm256_i32gather_epi32(kPerm32, A, 4);
    const __m256i pAp = _mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(A, one), 4);
    const __m256i pB  = _mm256_i32gather_epi32(kPerm32, B, 4);
    const __m256i pBp = _mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(B, one), 4);
    const __m256i AA  = _mm256_add_epi32(pA, Zi);
    const __m256i AB  = _mm256_add_epi32(pAp, Zi);
    const __m256i BA  = _mm256_add_epi32(pB, Zi);
    const __m256i BB  = _mm256_add_epi32(pBp, Zi);

    const __m256i low4 = _mm256_set1_epi32(15);
    const __m256i h000 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, AA, 4), low4);
    const __m256i h100 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, BA, 4), low4);
    const __m256i h010 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, AB, 4), low4);
    const __m256i h110 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, BB, 4), low4);
    const __m256i h001 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(AA, one), 4), low4);
    const __m256i h101 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(BA, one), 4), low4);
    const __m256i h011 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(AB, one), 4), low4);
    const __m256i h111 = _mm256_and_si256(_mm256_i32gather_epi32(kPerm32, _mm256_add_epi32(BB, one), 4), low4);

    const __m256 g000a = _mm256_i32gather_ps(kGradA, h000, 4);
    const __m256 g000b = _mm256_i32gather_ps(kGradB, h000, 4);
    const __m256 g000c = _mm256_i32gather_ps(kGradC, h000, 4);
    const __m256 g100a = _mm256_i32gather_ps(kGradA, h100, 4);
    const __m256 g100b = _mm256_i32gather_ps(kGradB, h100, 4);
    const __m256 g100c = _mm256_i32gather_ps(kGradC, h100, 4);
    const __m256 g010a = _mm256_i32gather_ps(kGradA, h010, 4);
    const __m256 g010b = _mm256_i32gather_ps(kGradB, h010, 4);
    const __m256 g010c = _mm256_i32gather_ps(kGradC, h010, 4);
    const __m256 g110a = _mm256_i32gather_ps(kGradA, h110, 4);
    const __m256 g110b = _mm256_i32gather_ps(kGradB, h110, 4);
    const __m256 g110c = _mm256_i32gather_ps(kGradC, h110, 4);
    const __m256 g001a = _mm256_i32gather_ps(kGradA, h001, 4);
    const __m256 g001b = _mm256_i32gather_ps(kGradB, h001, 4);
    const __m256 g001c = _mm256_i32gather_ps(kGradC, h001, 4);
    const __m256 g101a = _mm256_i32gather_ps(kGradA, h101, 4);
    const __m256 g101b = _mm256_i32gather_ps(kGradB, h101, 4);
    const __m256 g101c = _mm256_i32gather_ps(kGradC, h101, 4);
    const __m256 g011a = _mm256_i32gather_ps(kGradA, h011, 4);
    const __m256 g011b = _mm256_i32gather_ps(kGradB, h011, 4);
    const __m256 g011c = _mm256_i32gather_ps(kGradC, h011, 4);
    const __m256 g111a = _mm256_i32gather_ps(kGradA, h111, 4);
    const __m256 g111b = _mm256_i32gather_ps(kGradB, h111, 4);
    const __m256 g111c = _mm256_i32gather_ps(kGradC, h111, 4);

    const __m256 one_f = _mm256_set1_ps(1.0f);
    const __m256 xm1   = _mm256_sub_ps(x, one_f);
    const __m256 ym1   = _mm256_sub_ps(y, one_f);
    const __m256 zm1   = _mm256_sub_ps(z, one_f);

    const __m256 d000 = _mm256_fmadd_ps(g000a, x, _mm256_fmadd_ps(g000b, y, _mm256_mul_ps(g000c, z)));
    const __m256 d100 = _mm256_fmadd_ps(g100a, xm1, _mm256_fmadd_ps(g100b, y, _mm256_mul_ps(g100c, z)));
    const __m256 d010 = _mm256_fmadd_ps(g010a, x, _mm256_fmadd_ps(g010b, ym1, _mm256_mul_ps(g010c, z)));
    const __m256 d110 = _mm256_fmadd_ps(g110a, xm1, _mm256_fmadd_ps(g110b, ym1, _mm256_mul_ps(g110c, z)));
    const __m256 d001 = _mm256_fmadd_ps(g001a, x, _mm256_fmadd_ps(g001b, y, _mm256_mul_ps(g001c, zm1)));
    const __m256 d101 = _mm256_fmadd_ps(g101a, xm1, _mm256_fmadd_ps(g101b, y, _mm256_mul_ps(g101c, zm1)));
    const __m256 d011 = _mm256_fmadd_ps(g011a, x, _mm256_fmadd_ps(g011b, ym1, _mm256_mul_ps(g011c, zm1)));
    const __m256 d111 = _mm256_fmadd_ps(g111a, xm1, _mm256_fmadd_ps(g111b, ym1, _mm256_mul_ps(g111c, zm1)));

    const __m256 gx = Trilerp8(u, v, w, g000a, g100a, g010a, g110a, g001a, g101a, g011a, g111a);
    const __m256 gy = Trilerp8(u, v, w, g000b, g100b, g010b, g110b, g001b, g101b, g011b, g111b);
    const __m256 gz = Trilerp8(u, v, w, g000c, g100c, g010c, g110c, g001c, g101c, g011c, g111c);

    const __m256 x00 = Lerp8(u, d000, d100);
    const __m256 x10 = Lerp8(u, d010, d110);
    const __m256 x01 = Lerp8(u, d001, d101);
    const __m256 x11 = Lerp8(u, d011, d111);
    const __m256 dndu =
        Lerp8(w, Lerp8(v, _mm256_sub_ps(d100, d000), _mm256_sub_ps(d110, d010)),
              Lerp8(v, _mm256_sub_ps(d101, d001), _mm256_sub_ps(d111, d011)));
    const __m256 dndv = Lerp8(w, _mm256_sub_ps(x10, x00), _mm256_sub_ps(x11, x01));
    const __m256 dndw = _mm256_sub_ps(Lerp8(v, x01, x11), Lerp8(v, x00, x10));

    return { _mm256_fmadd_ps(du, dndu, gx), _mm256_fmadd_ps(dv, dndv, gy),
             _mm256_fmadd_ps(dw, dndw, gz) };
}

WPUTILS_AVX2_FMA_TARGET void
PerlinNoiseGradient8Avx2(const float x[8], const float y[8], const float z[8], float gx[8],
                         float gy[8], float gz[8]) noexcept {
    const Grad8 g = PerlinNoiseGradientVec8(_mm256_loadu_ps(x), _mm256_loadu_ps(y),
                                            _mm256_loadu_ps(z));
    _mm256_storeu_ps(gx, g.x);
    _mm256_storeu_ps(gy, g.y);
    _mm256_storeu_ps(gz, g.z);
}

WPUTILS_AVX2_FMA_TARGET void
CurlNoise8Avx2(const float px[8], const float py[8], const float pz[8], float cx[8], float cy[8],
               float cz[8]) noexcept {
    const __m256 x  = _mm256_loadu_ps(px);
    const __m256 y  = _mm256_loadu_ps(py);
    const __m256 z  = _mm256_loadu_ps(pz);
    const Grad8  gx = PerlinNoiseGradientVec8(x, y, z);
    const Grad8  gy = PerlinNoiseGradientVec8(_mm256_add_ps(x, _mm256_set1_ps(89.2f)),
                                             _mm256_add_ps(y, _mm256_set1_ps(33.1f)),
                                             _mm256_add_ps(z, _mm256_set1_ps(57.3f)));
    const Grad8  gz = PerlinNoiseGradientVec8(_mm256_add_ps(x, _mm256_set1_ps(100.3f)),
                                             _mm256_add_ps(y, _mm256_set1_ps(120.1f)),
                                             _mm256_add_ps(z, _mm256_set1_ps(142.2f)));
    _mm256_storeu_ps(cx, _mm256_sub_ps(gz.y, gy.z));
    _mm256_storeu_ps(cy, _mm256_sub_ps(gx.z, gz.x));
    _mm256_storeu_ps(cz, _mm256_sub_ps(gy.x, gx.y));
}

bool CpuSupportsAvx2Fma() noexcept {
    // Cache the dispatch decision, but keep the check in baseline code. The AVX functions above
    // are never entered on unsupported machines.
    static const bool supported = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return supported;
}

#undef WPUTILS_AVX2_FMA_TARGET
} // namespace
#endif

void algorism::PerlinNoiseGradient8(const float x[8], const float y[8], const float z[8],
                                    float gx[8], float gy[8], float gz[8]) noexcept {
#if defined(WPUTILS_HAS_X86_AVX2_DISPATCH)
    if (CpuSupportsAvx2Fma()) {
        PerlinNoiseGradient8Avx2(x, y, z, gx, gy, gz);
        return;
    }
#endif
    for (int i = 0; i < 8; i++) {
        const Eigen::Vector3f g = PerlinNoiseGradient(x[i], y[i], z[i]);
        gx[i]                   = g.x();
        gy[i]                   = g.y();
        gz[i]                   = g.z();
    }
}

void algorism::CurlNoise8(const float px[8], const float py[8], const float pz[8], float cx[8],
                          float cy[8], float cz[8]) noexcept {
#if defined(WPUTILS_HAS_X86_AVX2_DISPATCH)
    if (CpuSupportsAvx2Fma()) {
        CurlNoise8Avx2(px, py, pz, cx, cy, cz);
        return;
    }
#endif
    for (int i = 0; i < 8; i++) {
        const Eigen::Vector3f c = CurlNoise({ px[i], py[i], pz[i] });
        cx[i]                   = c.x();
        cy[i]                   = c.y();
        cz[i]                   = c.z();
    }
}

float algorism::HashNoise1D(float x) noexcept {
    /*
     * The permutation is the same 256-byte Ken Perlin table already used by PerlinNoise(); the 1D
     * interpolant only consumes the low four bits as a signed magnitude in 1..8 and weights
     * neighbouring lattices with (1-t^2)^4.
     */
    const int lattice = static_cast<int>(std::floor(x));
    const float t     = x - static_cast<float>(lattice);
    const float tm1   = t - 1.0f;
    const unsigned char hashed0 = perlin_perm[static_cast<unsigned char>(lattice)];
    const unsigned char hashed1 =
        perlin_perm[static_cast<unsigned char>(static_cast<unsigned char>(lattice) + 1)];
    auto gradient = [](unsigned char hashed) {
        float value = static_cast<float>(hashed & 7u) + 1.0f;
        if ((hashed & 8u) != 0) value = -value;
        return value;
    };
    const float g0  = gradient(hashed0);
    const float g1  = gradient(hashed1);
    const float w0  = 1.0f - t * t;
    const float w0s = w0 * w0;
    const float w1  = 1.0f - tm1 * tm1;
    const float w1s = w1 * w1;
    constexpr float kHashNoise1DScale = 0.395f;
    return (g1 * tm1 * w1s * w1s + g0 * t * w0s * w0s) * kHashNoise1DScale;
}

namespace
{
constexpr float    kSimplexF2       = 0.366025388f;
constexpr float    kSimplexG2       = 0.211324871f;
constexpr float    kSimplexCorner2  = -0.577350259f;
constexpr float    kSimplexGradSkew = 2.41421366f;
constexpr float    kSimplexNorm     = 38.2836876f;
constexpr uint32_t kSimplexHashI    = 501125321u;
constexpr uint32_t kSimplexHashJ    = 0x43c42e4du;
constexpr uint32_t kSimplexHashMix  = 0x27d4eb2du;

int32_t FloorTowardNegInf(float value) noexcept {
    const int32_t truncated = static_cast<int32_t>(value);
    return value < static_cast<float>(truncated) ? truncated - 1 : truncated;
}

uint32_t MixLatticeHash(int32_t i, int32_t j, uint32_t seed) noexcept {
    const uint32_t hashed = kSimplexHashMix * ((kSimplexHashI * static_cast<uint32_t>(i)) ^
                                               (kSimplexHashJ * static_cast<uint32_t>(j)) ^ seed);
    return (hashed >> 15) ^ hashed;
}

float SimplexCorner(float x, float y, uint32_t hash) noexcept {
    float falloff = 0.5f - x * x - y * y;
    if (falloff <= 0.0f) return 0.0f;
    falloff *= falloff;
    falloff *= falloff;

    const uint32_t signed_x = std::bit_cast<uint32_t>(x) ^ (hash << 31);
    const uint32_t signed_y = std::bit_cast<uint32_t>(y) ^ ((hash >> 1) << 31);
    const uint32_t select =
        static_cast<uint32_t>(static_cast<int32_t>(hash << 29) >> 31) & (signed_x ^ signed_y);
    const float gx = std::bit_cast<float>(select ^ signed_x);
    const float gy = std::bit_cast<float>(select ^ signed_y);
    return (gx * kSimplexGradSkew + gy) * falloff;
}
} // namespace

float algorism::SimplexNoise2D(float x, float y, uint32_t seed) noexcept {
    const float   skew = (x + y) * kSimplexF2;
    const int32_t i    = FloorTowardNegInf(x + skew);
    const int32_t j    = FloorTowardNegInf(y + skew);
    const float   unskew =
        (static_cast<float>(i) + static_cast<float>(j)) * kSimplexG2;
    const float x0 = x - (static_cast<float>(i) - unskew);
    const float y0 = y - (static_cast<float>(j) - unskew);
    const int32_t i1 = y0 < x0 ? 1 : 0;
    const int32_t j1 = y0 < x0 ? 0 : 1;
    const float   x1 = x0 - static_cast<float>(i1) + kSimplexG2;
    const float   y1 = y0 - static_cast<float>(j1) + kSimplexG2;
    const float   x2 = x0 + kSimplexCorner2;
    const float   y2 = y0 + kSimplexCorner2;
    return (SimplexCorner(x0, y0, MixLatticeHash(i, j, seed)) +
            SimplexCorner(x1, y1, MixLatticeHash(i + i1, j + j1, seed)) +
            SimplexCorner(x2, y2, MixLatticeHash(i + 1, j + 1, seed))) *
           kSimplexNorm;
}

float algorism::FbmNoise2D(float x, float y, uint32_t seed, int octaves) noexcept {
    if (octaves < 1) octaves = 1;
    float weight_sum = 1.0f;
    float persist    = 0.5f;
    for (int octave = 1; octave < octaves; ++octave) {
        weight_sum += persist;
        persist *= 0.5f;
    }
    float amp  = 1.0f / weight_sum;
    float last = SimplexNoise2D(x, y, seed);
    float sum  = last * amp;
    for (int octave = 1; octave < octaves; ++octave) {
        x *= 2.0f;
        y *= 2.0f;
        seed += 1u;
        amp *= 0.5f;
        last = SimplexNoise2D(x, y, seed);
        sum += last * amp;
    }
    return sum;
}

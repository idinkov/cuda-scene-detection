// tests/test_mad.cu - stripped down for debugging
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "../cpu_mad.h"

extern "C" float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA,
                                   const uint8_t* frameB_dev, int pitchB,
                                   int width, int height, int downscale);

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_TRUE(cond) do { if (!(cond)) { \
    fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, #cond); \
    g_failed++; return; } } while(0)

#define ASSERT_NEAR(a, b, eps) do { if (fabs((double)(a) - (double)(b)) > (eps)) { \
    fprintf(stderr, "  FAIL line %d: got %f vs %f\n", __LINE__, (double)(a), (double)(b)); \
    g_failed++; return; } } while(0)

static uint8_t* alloc_dev_frame(int width, int height, size_t* out_pitch, uint8_t fill) {
    uint8_t* d_ptr = nullptr;
    cudaError_t err = cudaMallocPitch((void**)&d_ptr, out_pitch, (size_t)width, (size_t)height);
    if (err != cudaSuccess) return nullptr;
    std::vector<uint8_t> host((*out_pitch) * height, fill);
    cudaMemcpy2D(d_ptr, *out_pitch, host.data(), *out_pitch, width, height, cudaMemcpyHostToDevice);
    return d_ptr;
}

static void test_identical() {
    int w = 64, h = 64;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 128);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 128);
    ASSERT_TRUE(a && b);
    float mad = compute_mad_cuda(a, (int)pA, b, (int)pB, w, h, 1);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

static void test_known_diff() {
    int w = 64, h = 64;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 100);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 150);
    ASSERT_TRUE(a && b);
    float mad = compute_mad_cuda(a, (int)pA, b, (int)pB, w, h, 1);
    ASSERT_NEAR(mad, 50.0f, 0.5f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

static void test_null() {
    float mad = compute_mad_cuda(nullptr, 0, nullptr, 0, 64, 64, 1);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    g_passed++;
}

static void test_zero_dim() {
    size_t p;
    uint8_t* a = alloc_dev_frame(64, 64, &p, 100);
    ASSERT_TRUE(a);
    float mad = compute_mad_cuda(a, (int)p, a, (int)p, 0, 0, 1);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    cudaFree(a);
    g_passed++;
}

static void test_large() {
    int w = 1920, h = 1080;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 50);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 75);
    ASSERT_TRUE(a && b);
    float mad = compute_mad_cuda(a, (int)pA, b, (int)pB, w, h, 2);
    ASSERT_NEAR(mad, 25.0f, 0.5f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

static void test_downscale() {
    int w = 128, h = 128;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 0);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 255);
    ASSERT_TRUE(a && b);
    float m1 = compute_mad_cuda(a, (int)pA, b, (int)pB, w, h, 1);
    float m2 = compute_mad_cuda(a, (int)pA, b, (int)pB, w, h, 2);
    float m4 = compute_mad_cuda(a, (int)pA, b, (int)pB, w, h, 4);
    ASSERT_NEAR(m1, 255.0f, 0.5f);
    ASSERT_NEAR(m2, 255.0f, 0.5f);
    ASSERT_NEAR(m4, 255.0f, 0.5f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

// ---------------------------------------------------------------------------
// CPU helper tests (sample_luma + compute_mad_cpu from cpu_mad.h)
// ---------------------------------------------------------------------------

static void test_sample_luma_basic() {
    // 4x4 source plane with known values, ds=2 → 2x2 output
    // Row stride == width (no extra padding for this host buffer)
    uint8_t src[4 * 4] = {
         10,  20,  30,  40,
         50,  60,  70,  80,
         90, 100, 110, 120,
        130, 140, 150, 160
    };
    uint8_t dst[2 * 2] = {};
    // ds=2: sample (x=0,y=0),(x=2,y=0),(x=0,y=2),(x=2,y=2) → 10, 30, 90, 110
    sample_luma(src, /*srcPitch=*/4, /*w=*/4, /*h=*/4, /*ds=*/2, dst, /*dstW=*/2, /*dstH=*/2);
    ASSERT_TRUE(dst[0] == 10);
    ASSERT_TRUE(dst[1] == 30);
    ASSERT_TRUE(dst[2] == 90);
    ASSERT_TRUE(dst[3] == 110);
    g_passed++;
}

static void test_sample_luma_ds1() {
    // ds=1 → output identical to input (no downscale)
    uint8_t src[4] = { 1, 2, 3, 4 };
    uint8_t dst[4] = {};
    sample_luma(src, 4, 4, 1, 1, dst, 4, 1);
    ASSERT_TRUE(dst[0] == 1);
    ASSERT_TRUE(dst[1] == 2);
    ASSERT_TRUE(dst[2] == 3);
    ASSERT_TRUE(dst[3] == 4);
    g_passed++;
}

static void test_cpu_mad_identical() {
    uint8_t a[128], b[128];
    memset(a, 128, 128);
    memset(b, 128, 128);
    float mad = compute_mad_cpu(a, b, 128);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    g_passed++;
}

static void test_cpu_mad_known_diff() {
    // All pixels differ by 50; MAD must equal 50.
    uint8_t a[128], b[128];
    memset(a, 100, 128);
    memset(b, 150, 128);
    float mad = compute_mad_cpu(a, b, 128);
    ASSERT_NEAR(mad, 50.0f, 0.5f);
    g_passed++;
}

static void test_cpu_mad_null() {
    // Null inputs → 0.
    float mad = compute_mad_cpu(nullptr, nullptr, 64);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    g_passed++;
}

static void test_cpu_mad_tail() {
    // 33 bytes: exercises the AVX2 main loop (32 bytes) plus the scalar tail (1 byte).
    // Also tests the scalar fallback when __AVX2__ is not defined.
    uint8_t a[33], b[33];
    memset(a, 0,   33);
    memset(b, 255, 33);
    float mad = compute_mad_cpu(a, b, 33);
    ASSERT_NEAR(mad, 255.0f, 0.5f);
    g_passed++;
}

int main() {
    fprintf(stderr, "=== nvdec_scene_detect Unit Tests ===\n");

    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) { fprintf(stderr, "No CUDA devices!\n"); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    fprintf(stderr, "  test_identical...\n"); test_identical();
    fprintf(stderr, "  test_known_diff...\n"); test_known_diff();
    fprintf(stderr, "  test_null...\n"); test_null();
    fprintf(stderr, "  test_zero_dim...\n"); test_zero_dim();
    fprintf(stderr, "  test_large...\n"); test_large();
    fprintf(stderr, "  test_downscale...\n"); test_downscale();

    fprintf(stderr, "\n  -- CPU helpers (cpu_mad.h) --\n");
    fprintf(stderr, "  test_sample_luma_basic...\n"); test_sample_luma_basic();
    fprintf(stderr, "  test_sample_luma_ds1...\n"); test_sample_luma_ds1();
    fprintf(stderr, "  test_cpu_mad_identical...\n"); test_cpu_mad_identical();
    fprintf(stderr, "  test_cpu_mad_known_diff...\n"); test_cpu_mad_known_diff();
    fprintf(stderr, "  test_cpu_mad_null...\n"); test_cpu_mad_null();
    fprintf(stderr, "  test_cpu_mad_tail...\n"); test_cpu_mad_tail();

    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

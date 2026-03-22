// tests/test_mad.cu - stripped down for debugging
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA,
                                   const uint8_t* frameB_dev, int pitchB,
                                   int width, int height, int downscale);

extern "C" bool  downscale_luma_box(const uint8_t* src, int srcPitch,
                                     uint8_t* dst, int dstPitch,
                                     int srcWidth, int srcHeight, int downscale);

extern "C" float compute_mad_cuda_box(const uint8_t* frameA_dev, int pitchA,
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

// ─── Box-filter downscale tests ───────────────────────────────────────────────

// compute_mad_cuda_box: identical frames → MAD ≈ 0
static void test_box_identical() {
    int w = 64, h = 64;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 128);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 128);
    ASSERT_TRUE(a && b);
    float mad = compute_mad_cuda_box(a, (int)pA, b, (int)pB, w, h, 2);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

// compute_mad_cuda_box: uniform frames with known difference
static void test_box_known_diff() {
    int w = 64, h = 64;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 80);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 130);
    ASSERT_TRUE(a && b);
    // Box filter over a uniform frame leaves every pixel unchanged, so MAD == 50.
    float mad = compute_mad_cuda_box(a, (int)pA, b, (int)pB, w, h, 2);
    ASSERT_NEAR(mad, 50.0f, 0.5f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

// compute_mad_cuda_box: null inputs return 0
static void test_box_null() {
    float mad = compute_mad_cuda_box(nullptr, 0, nullptr, 0, 64, 64, 2);
    ASSERT_NEAR(mad, 0.0f, 0.001f);
    g_passed++;
}

// compute_mad_cuda_box: downscale==1 path delegates to compute_mad_cuda
static void test_box_downscale_one() {
    int w = 64, h = 64;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 60);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 100);
    ASSERT_TRUE(a && b);
    float mad_box    = compute_mad_cuda_box(a, (int)pA, b, (int)pB, w, h, 1);
    float mad_stride = compute_mad_cuda    (a, (int)pA, b, (int)pB, w, h, 1);
    ASSERT_NEAR(mad_box, mad_stride, 0.5f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

// downscale_luma_box: verify that a checkerboard (0/255 alternating) averages
// to ≈ 127 after a 2×2 box downscale.
static void test_box_downscale_averaging() {
    int w = 64, h = 64;
    // Build a host checkerboard (each 2×2 block is half 0, half 255 → average 127)
    size_t pitchH = (size_t)w; // for host buffer (no padding)
    std::vector<uint8_t> host_src(pitchH * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            host_src[y * pitchH + x] = (uint8_t)(((x + y) & 1) ? 255 : 0);

    // Upload to device
    uint8_t* d_src = nullptr; size_t d_pitchSrc = 0;
    cudaError_t err = cudaMallocPitch((void**)&d_src, &d_pitchSrc, (size_t)w, (size_t)h);
    ASSERT_TRUE(err == cudaSuccess);
    cudaMemcpy2D(d_src, d_pitchSrc, host_src.data(), pitchH, (size_t)w, (size_t)h, cudaMemcpyHostToDevice);

    // Allocate output for downscaled image (ds = 2 → 32×32)
    int dsW = w / 2, dsH = h / 2;
    uint8_t* d_dst = nullptr; size_t d_pitchDst = 0;
    err = cudaMallocPitch((void**)&d_dst, &d_pitchDst, (size_t)dsW, (size_t)dsH);
    ASSERT_TRUE(err == cudaSuccess);

    bool ok = downscale_luma_box(d_src, (int)d_pitchSrc, d_dst, (int)d_pitchDst, w, h, 2);
    ASSERT_TRUE(ok);
    cudaDeviceSynchronize();

    // Copy result back
    std::vector<uint8_t> host_dst(d_pitchDst * dsH);
    cudaMemcpy2D(host_dst.data(), d_pitchDst, d_dst, d_pitchDst, (size_t)dsW, (size_t)dsH, cudaMemcpyDeviceToHost);

    // Every output pixel should be ~128 (round((0+255+255+0)/4) = 128)
    bool all_ok = true;
    for (int y = 0; y < dsH && all_ok; ++y)
        for (int x = 0; x < dsW && all_ok; ++x) {
            uint8_t v = host_dst[y * d_pitchDst + x];
            if (v < 120 || v > 136) { // allow ±8 from expected 128
                fprintf(stderr, "  FAIL: checkerboard avg at (%d,%d) = %d (expected ~128)\n", x, y, (int)v);
                all_ok = false;
            }
        }
    ASSERT_TRUE(all_ok);

    cudaFree(d_src); cudaFree(d_dst);
    g_passed++;
}

// Box vs stride: for uniform frames both should agree closely
static void test_box_vs_stride_uniform() {
    int w = 128, h = 128;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 40);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 90);
    ASSERT_TRUE(a && b);
    float mad_stride = compute_mad_cuda    (a, (int)pA, b, (int)pB, w, h, 4);
    float mad_box    = compute_mad_cuda_box(a, (int)pA, b, (int)pB, w, h, 4);
    // Both should be ≈ 50; box filter averages a uniform block so result is identical
    ASSERT_NEAR(mad_stride, 50.0f, 0.5f);
    ASSERT_NEAR(mad_box,    50.0f, 0.5f);
    cudaFree(a); cudaFree(b);
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
    fprintf(stderr, "  test_box_identical...\n"); test_box_identical();
    fprintf(stderr, "  test_box_known_diff...\n"); test_box_known_diff();
    fprintf(stderr, "  test_box_null...\n"); test_box_null();
    fprintf(stderr, "  test_box_downscale_one...\n"); test_box_downscale_one();
    fprintf(stderr, "  test_box_downscale_averaging...\n"); test_box_downscale_averaging();
    fprintf(stderr, "  test_box_vs_stride_uniform...\n"); test_box_vs_stride_uniform();

    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

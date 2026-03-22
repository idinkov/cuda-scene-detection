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

    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

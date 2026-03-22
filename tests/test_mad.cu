// tests/test_mad.cu - stripped down for debugging
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

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
// Minimal GpuWorkQueue re-implementation for testing the queue/thread logic.
// This mirrors the class in ffmpeg_nvdec_scene_detect.cpp.
// ---------------------------------------------------------------------------

struct GpuTask_t {
    const uint8_t* a_dev;  int pitch_a;
    const uint8_t* b_dev;  int pitch_b;
    int width, height, downscale;
    std::promise<float> result;
};

class GpuWorkQueue_t {
public:
    std::future<float> submit(const uint8_t* a, int pa, const uint8_t* b, int pb,
                               int w, int h, int ds) {
        GpuTask_t task;
        task.a_dev = a; task.pitch_a = pa;
        task.b_dev = b; task.pitch_b = pb;
        task.width = w; task.height = h; task.downscale = ds;
        auto fut = task.result.get_future();
        { std::unique_lock<std::mutex> lk(mtx_); queue_.push(std::move(task)); }
        cv_.notify_one();
        return fut;
    }
    void shutdown() {
        { std::unique_lock<std::mutex> lk(mtx_); done_ = true; }
        cv_.notify_all();
    }
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]{ return !queue_.empty() || done_; });
            if (queue_.empty()) break;
            GpuTask_t task = std::move(queue_.front()); queue_.pop();
            lk.unlock();
            float mad = compute_mad_cuda(task.a_dev, task.pitch_a,
                                         task.b_dev, task.pitch_b,
                                         task.width, task.height, task.downscale);
            task.result.set_value(mad);
        }
    }
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<GpuTask_t> queue_;
    bool done_ = false;
};

// Test: GpuWorkQueue produces the same MAD result as a direct call.
static void test_gpu_work_queue_single() {
    int w = 64, h = 64;
    size_t pA, pB;
    uint8_t* a = alloc_dev_frame(w, h, &pA, 80);
    uint8_t* b = alloc_dev_frame(w, h, &pB, 140);
    ASSERT_TRUE(a && b);

    GpuWorkQueue_t q;
    std::thread worker([&q]{ q.run(); });

    auto fut = q.submit(a, (int)pA, b, (int)pB, w, h, 1);
    float queue_mad = fut.get();

    q.shutdown();
    worker.join();

    // Expected MAD = |80-140| = 60
    ASSERT_NEAR(queue_mad, 60.0f, 0.5f);
    cudaFree(a); cudaFree(b);
    g_passed++;
}

// Test: multiple tasks submitted by concurrent threads all resolve correctly.
static void test_gpu_work_queue_concurrent() {
    const int N = 4;
    int w = 64, h = 64;

    GpuWorkQueue_t q;
    std::thread worker([&q]{ q.run(); });

    // Each "stream" submits one task from its own thread.
    std::vector<std::thread> threads;

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&, i]{
            size_t pA, pB;
            uint8_t val_a = (uint8_t)(i * 20);
            uint8_t val_b = (uint8_t)(i * 20 + 10);
            uint8_t* a = alloc_dev_frame(w, h, &pA, val_a);
            uint8_t* b = alloc_dev_frame(w, h, &pB, val_b);
            if (!a || !b) {
                fprintf(stderr, "  FAIL (concurrent stream %d): frame alloc failed\n", i);
                g_failed++;
                if (a) cudaFree(a);
                if (b) cudaFree(b);
                return;
            }
            auto fut = q.submit(a, (int)pA, b, (int)pB, w, h, 1);
            float mad = fut.get();
            // MAD should equal 10 for every stream
            if (fabsf(mad - 10.0f) > 0.5f) {
                fprintf(stderr, "  FAIL (concurrent stream %d): got %f vs 10.0\n", i, mad);
                g_failed++;
            }
            cudaFree(a); cudaFree(b);
        });
    }

    for (auto& t : threads) t.join();
    q.shutdown();
    worker.join();

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
    fprintf(stderr, "  test_gpu_work_queue_single...\n"); test_gpu_work_queue_single();
    fprintf(stderr, "  test_gpu_work_queue_concurrent...\n"); test_gpu_work_queue_concurrent();

    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

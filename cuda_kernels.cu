// cuda_kernels.cu
// CUDA kernels + host helpers to compute Mean Absolute Difference (MAD)
// between two 8-bit luma planes (pitch-aware). Compiles with nvcc.
//
// Exported functions:
//   compute_mad_cuda      - MAD with stride sampling (original)
//   downscale_luma_box    - Box-filter (average-pool) luma downscale kernel
//   compute_mad_cuda_box  - MAD after proper box-filter downscale (better quality)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA, const uint8_t* frameB_dev, int pitchB, int width, int height, int downscale);
bool  downscale_luma_box(const uint8_t* src, int srcPitch, uint8_t* dst, int dstPitch, int srcWidth, int srcHeight, int downscale);
float compute_mad_cuda_box(const uint8_t* frameA_dev, int pitchA, const uint8_t* frameB_dev, int pitchB, int width, int height, int downscale);
}

// ─── Stride-sampling MAD kernel ──────────────────────────────────────────────
// Each thread accumulates a local sum across a strided set of pixels and
// atomically adds to a global 64-bit accumulator.
static __global__ void mad_kernel(const uint8_t* __restrict__ a, int pitchA, const uint8_t* __restrict__ b, int pitchB, int width, int height, int downscale, unsigned long long* out_sum) {
    if (downscale < 1) downscale = 1;
    int dsWidth = (width + downscale - 1) / downscale;
    int dsHeight = (height + downscale - 1) / downscale;
    unsigned int total = (unsigned int)dsWidth * (unsigned int)dsHeight;
    unsigned long long local = 0ULL;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < total; i += stride) {
        unsigned int y_ds = i / dsWidth;
        unsigned int x_ds = i - y_ds * dsWidth;
        unsigned int y = y_ds * downscale; if (y >= (unsigned)height) y = height - 1;
        unsigned int x = x_ds * downscale; if (x >= (unsigned)width) x = width - 1;
        int va = a[y * pitchA + x];
        int vb = b[y * pitchB + x];
        local += (unsigned long long)(va > vb ? va - vb : vb - va);
    }

    if (local) atomicAdd(out_sum, local);
}

// Host wrapper
extern "C" float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA, const uint8_t* frameB_dev, int pitchB, int width, int height, int downscale) {
    if (!frameA_dev || !frameB_dev || width <= 0 || height <= 0) return 0.0f;
    if (downscale < 1) downscale = 1;
    int dsWidth = (width + downscale - 1) / downscale;
    int dsHeight = (height + downscale - 1) / downscale;
    unsigned int total = (unsigned int)dsWidth * (unsigned int)dsHeight;
    unsigned long long* d_accum = nullptr;
    cudaError_t cerr = cudaMalloc((void**)&d_accum, sizeof(unsigned long long));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cerr));
        return 0.0f;
    }
    cerr = cudaMemset(d_accum, 0, sizeof(unsigned long long));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_accum);
        return 0.0f;
    }

    const int threads = 256;
    unsigned int blocks = (total + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    // Clamp grid size to a reasonable value (avoid extremely large grids)
    if (blocks > 65535) blocks = 65535;

    mad_kernel<<<blocks, threads>>>(frameA_dev, pitchA, frameB_dev, pitchB, width, height, downscale, d_accum);
    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_accum);
        return 0.0f;
    }

    // Wait for kernel
    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_accum);
        return 0.0f;
    }

    unsigned long long host_sum = 0ULL;
    cerr = cudaMemcpy(&host_sum, d_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_accum);
        return 0.0f;
    }

    cudaFree(d_accum);

    float mad = 0.0f;
    if (total > 0) mad = (float)host_sum / (float)total;
    return mad;
}

// ─── Box-filter downscale kernel ─────────────────────────────────────────────
// Each thread computes one output pixel by averaging all source pixels in its
// (downscale × downscale) block (average pooling / box filter).  Partial
// blocks at the right/bottom edges are handled correctly.
//
// Grid: 2-D, one thread per output pixel (x_ds, y_ds).
// Uses __ldg for read-only cached loads from global memory.
static __global__ void downscale_box_kernel(
        const uint8_t* __restrict__ src, int srcPitch,
        uint8_t* __restrict__ dst,       int dstPitch,
        int srcWidth, int srcHeight, int downscale)
{
    int x_ds = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y_ds = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int dsWidth  = (srcWidth  + downscale - 1) / downscale;
    int dsHeight = (srcHeight + downscale - 1) / downscale;
    if (x_ds >= dsWidth || y_ds >= dsHeight) return;

    int sum   = 0;
    int count = 0;
    int y_src_base = y_ds * downscale;
    int x_src_base = x_ds * downscale;
    for (int dy = 0; dy < downscale; ++dy) {
        int y = y_src_base + dy;
        if (y >= srcHeight) break;
        for (int dx = 0; dx < downscale; ++dx) {
            int x = x_src_base + dx;
            if (x >= srcWidth) break;
            sum += (int)__ldg(&src[y * srcPitch + x]);
            ++count;
        }
    }
    dst[y_ds * dstPitch + x_ds] = (uint8_t)((sum + count / 2) / count);
}

// Host helper: launch box-filter downscale for one luma plane.
// dst must be a device buffer with dstPitch >= dsWidth bytes and dsHeight rows.
// Returns true on success.
extern "C" bool downscale_luma_box(
        const uint8_t* src, int srcPitch,
        uint8_t*       dst, int dstPitch,
        int srcWidth, int srcHeight, int downscale)
{
    if (!src || !dst || srcWidth <= 0 || srcHeight <= 0) return false;
    if (downscale < 1) downscale = 1;
    if (downscale == 1) {
        // Identity copy
        cudaError_t ce = cudaMemcpy2D(dst, (size_t)dstPitch, src, (size_t)srcPitch,
                                      (size_t)srcWidth, (size_t)srcHeight,
                                      cudaMemcpyDeviceToDevice);
        return ce == cudaSuccess;
    }
    int dsWidth  = (srcWidth  + downscale - 1) / downscale;
    int dsHeight = (srcHeight + downscale - 1) / downscale;
    dim3 block(16, 16);
    dim3 grid((unsigned int)(dsWidth  + (int)block.x - 1) / block.x,
              (unsigned int)(dsHeight + (int)block.y - 1) / block.y);
    downscale_box_kernel<<<grid, block>>>(src, srcPitch, dst, dstPitch,
                                         srcWidth, srcHeight, downscale);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "downscale_box_kernel launch failed: %s\n", cudaGetErrorString(ce));
        return false;
    }
    return true;
}

// Host wrapper: box-filter downscale both luma planes then compute MAD.
// Compared with compute_mad_cuda (stride sampling), this averages each
// downscale×downscale block so the result is less noisy.
extern "C" float compute_mad_cuda_box(
        const uint8_t* frameA_dev, int pitchA,
        const uint8_t* frameB_dev, int pitchB,
        int width, int height, int downscale)
{
    if (!frameA_dev || !frameB_dev || width <= 0 || height <= 0) return 0.0f;
    if (downscale < 1) downscale = 1;
    // downscale==1: no actual shrink needed, delegate to existing MAD function.
    if (downscale == 1)
        return compute_mad_cuda(frameA_dev, pitchA, frameB_dev, pitchB, width, height, 1);

    int dsWidth  = (width  + downscale - 1) / downscale;
    int dsHeight = (height + downscale - 1) / downscale;

    // Allocate pitched device buffers for the two downscaled planes.
    uint8_t* ds_a = nullptr; size_t ds_pitchA = 0;
    uint8_t* ds_b = nullptr; size_t ds_pitchB = 0;
    cudaError_t ce;
    ce = cudaMallocPitch((void**)&ds_a, &ds_pitchA, (size_t)dsWidth, (size_t)dsHeight);
    if (ce != cudaSuccess) {
        fprintf(stderr, "compute_mad_cuda_box: cudaMallocPitch A failed: %s\n", cudaGetErrorString(ce));
        return 0.0f;
    }
    ce = cudaMallocPitch((void**)&ds_b, &ds_pitchB, (size_t)dsWidth, (size_t)dsHeight);
    if (ce != cudaSuccess) {
        fprintf(stderr, "compute_mad_cuda_box: cudaMallocPitch B failed: %s\n", cudaGetErrorString(ce));
        cudaFree(ds_a);
        return 0.0f;
    }

    // Downscale both planes using the box filter.
    // Both kernels run in the default stream, so launch ordering guarantees the
    // MAD kernel below sees completed writes.  compute_mad_cuda() calls its own
    // cudaDeviceSynchronize(), so no extra sync is needed here.
    bool ok = downscale_luma_box(frameA_dev, pitchA, ds_a, (int)ds_pitchA, width, height, downscale)
           && downscale_luma_box(frameB_dev, pitchB, ds_b, (int)ds_pitchB, width, height, downscale);
    if (!ok) {
        cudaFree(ds_a); cudaFree(ds_b);
        return 0.0f;
    }

    // Compute MAD on the downscaled frames (no further downscaling inside).
    float mad = compute_mad_cuda(ds_a, (int)ds_pitchA, ds_b, (int)ds_pitchB,
                                 dsWidth, dsHeight, 1);

    cudaFree(ds_a);
    cudaFree(ds_b);
    return mad;
}

/*
CMakeLists.txt (example)

cmake_minimum_required(VERSION 3.18)
project(nvdec_scene_detect LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# Find FFmpeg via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED libavformat libavcodec libavutil libswscale)

# Add executable: the main C++ file (ffmpeg_nvdec_scene_detect.cpp) and this CUDA file
add_executable(nvdec_scene_detect ffmpeg_nvdec_scene_detect.cpp cuda_kernels.cu)

# Include FFmpeg includes
target_include_directories(nvdec_scene_detect PRIVATE ${AV_INCLUDE_DIRS})
# Link FFmpeg libs
target_link_libraries(nvdec_scene_detect PRIVATE ${AV_LIBRARIES} cuda cudart)

# Add include/link flags from pkg-config
target_compile_options(nvdec_scene_detect PRIVATE ${AV_CFLAGS_OTHER})

# Ensure CUDA architecture config (optional)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

# Example build commands (from repo root):
# mkdir build && cd build
# cmake ..
# cmake --build . -j

Notes:
- You may need to adjust link flags on your system (sometimes pkg-config names are libavformat ... and AV_LIBRARIES contains -lavformat ... already).
- On some systems it is necessary to link against -lcuda -lcudart explicitly. If link fails, add them to target_link_libraries.
- Ensure your FFmpeg was built with CUDA/NVDEC support. If not, decoding will fall back to software.
*/

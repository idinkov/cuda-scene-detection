// cuda_kernels.cu
// Simple CUDA kernel + host helper to compute Mean Absolute Difference (MAD)
// between two 8-bit luma planes (pitch-aware). Compiles with nvcc.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA, const uint8_t* frameB_dev, int pitchB, int width, int height);
}

// Kernel: each thread accumulates a local sum across a strided set of pixels and atomically adds to a global 64-bit accumulator.
static __global__ void mad_kernel(const uint8_t* __restrict__ a, int pitchA, const uint8_t* __restrict__ b, int pitchB, int width, int height, unsigned long long* out_sum) {
    unsigned long long local = 0ULL;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int total = (unsigned int)width * (unsigned int)height;

    for (unsigned int i = tid; i < total; i += stride) {
        unsigned int y = i / width;
        unsigned int x = i - y * width;
        int va = a[y * pitchA + x];
        int vb = b[y * pitchB + x];
        local += (unsigned long long) (va > vb ? va - vb : vb - va);
    }

    if (local) atomicAdd(out_sum, local);
}

// Host wrapper
extern "C" float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA, const uint8_t* frameB_dev, int pitchB, int width, int height) {
    if (!frameA_dev || !frameB_dev || width <= 0 || height <= 0) return 0.0f;

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

    const unsigned int total = (unsigned int)width * (unsigned int)height;
    const int threads = 256;
    unsigned int blocks = (total + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    // Clamp grid size to a reasonable value (avoid extremely large grids)
    if (blocks > 65535) blocks = 65535;

    dim3 grid(blocks);
    dim3 block(threads);

    mad_kernel<<<grid, block>>>(frameA_dev, pitchA, frameB_dev, pitchB, width, height, d_accum);
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

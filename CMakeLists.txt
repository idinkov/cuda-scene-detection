cmake_minimum_required(VERSION 3.20)
project(nvdec_scene_detect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Adjust to your FFmpeg build location
set(FFMPEG_DIR "dependencies/ffmpeg/")   # root folder where include/lib are
include_directories(${FFMPEG_DIR}/include)
link_directories(${FFMPEG_DIR}/lib)

add_executable(nvdec_scene_detect
        ffmpeg_nvdec_scene_detect.cpp
        cuda_kernels.cu
)

# Link against FFmpeg import libs + CUDA
if(WIN32)
    target_link_libraries(nvdec_scene_detect
            avformat.lib
            avcodec.lib
            avutil.lib
            swscale.lib
            cuda.lib
            cudart.lib
    )
else()
    target_link_libraries(nvdec_scene_detect
            avformat
            avcodec
            avutil
            swscale
            cudart
    )
endif()

# Ensure CUDA architecture flags (adjust if you know your GPU SM version)
# RTX 5070 = SM 120 (Blackwell)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# === Unit tests ===
enable_testing()

add_executable(test_mad
        tests/test_mad.cu
        cuda_kernels.cu
)

# nvcc uses shared cudart to avoid CRT mismatch (LNK4098)
set_target_properties(test_mad PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(test_mad
        CUDA::cudart
        CUDA::cuda_driver
)

add_test(NAME mad_tests COMMAND test_mad)

# === Gradual detection unit tests (CPU only, no CUDA device needed) ===
add_executable(test_gradual
        tests/test_gradual.cpp
)
target_include_directories(test_gradual PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME gradual_tests COMMAND test_gradual)

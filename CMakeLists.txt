cmake_minimum_required(VERSION 3.20)
project(nvdec_scene_detect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Path to your FFmpeg installation (root folder containing include/ and lib/).
# Override on the command line with -DFFMPEG_DIR=/path/to/ffmpeg, e.g.:
#   cmake -B build -DFFMPEG_DIR=/usr
set(FFMPEG_DIR "dependencies/ffmpeg/" CACHE PATH "Root directory of the FFmpeg installation (must contain include/ and lib/)")
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
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86)
endif()

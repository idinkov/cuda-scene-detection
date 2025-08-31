# CUDA NVDEC Scene Detection

Lightweight demo tool that decodes video with FFmpeg + NVIDIA NVDEC (CUDA) and detects hard scene cuts on the GPU using a simple Mean Absolute Difference (MAD) over the luma plane. Frames stay on the GPU when possible; only a small CUDA kernel runs per frame pair.

## Features
- Hardware accelerated H.264 / HEVC decode via FFmpeg (CUVID / AVHWDeviceType CUDA).
- Zero‑copy path for NV12 luma when available.
- CUDA kernel computes MAD with optional spatial downscale sampling.
- Optional CSV output of detected cuts.
- Graceful fallback to CPU MAD when frames are not on the GPU (still functional, slower).

## Algorithm (summary)
1. Decode frames (prefer NVDEC). Obtain NV12 / YUV420 luma plane.
2. Downsample logically by strided sampling factor `--downscale` (no resize kernel, just sampling points).
3. Compute sum(|Y_t - Y_{t-1}|) and divide by sampled pixel count => MAD.
4. Declare a cut if MAD > threshold AND time since last cut > min gap.

GPU path: two pitched device buffers (previous/current). Kernel launches with 256 threads per block; each thread iterates over a strided subset accumulating into a 64‑bit global atomic counter. Result is copied back and normalized.

## Build Requirements
- NVIDIA GPU + driver supporting CUDA and NVDEC.
- CUDA Toolkit (tested conceptually with 11+/12+).
- FFmpeg built with NVDEC/CUDA support (enable flags like `--enable-cuda --enable-nvdec --enable-cuvid` depending on build scripts). Provided Windows `dependencies/ffmpeg` folder can be used directly.
- CMake 3.20+ (project sets C++20 / CUDA 20).

## Repository Layout
```
ffmpeg_nvdec_scene_detect.cpp   Main program
cuda_kernels.cu                 MAD CUDA kernel
CMakeLists.txt                  Build script (Windows oriented, static path to dependencies/ffmpeg)
build.bat                       Convenience Windows build script
install_ffmpeg.bat              (Optional helper if you add logic) placeholder
dependencies/ffmpeg/            Prebuilt FFmpeg (bin/include/lib)
```

## Windows Build (Visual Studio + CMake)
Option 1 (recommended):
```
build.bat
```
Produces `build/Release/nvdec_scene_detect.exe` and copies required DLLs.

Option 2 (manual):
```
mkdir build & cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release -j
```
Ensure `dependencies/ffmpeg/bin` and CUDA runtime DLLs are on PATH when running.

## Linux Build (example quick compile)
Install FFmpeg (with CUDA/NVDEC) and CUDA toolkit. Example:
```
nvcc -O2 -arch=sm_75 -c cuda_kernels.cu -o cuda_kernels.o
 g++ -O2 -std=c++17 ffmpeg_nvdec_scene_detect.cpp cuda_kernels.o -o nvdec_scene_detect \
    `pkg-config --cflags --libs libavformat libavcodec libavutil libswscale` -lcuda -ldl -lpthread
```
Or use CMake by adapting `CMakeLists.txt` (add pkg-config discovery as commented in cuda_kernels.cu).

## Usage
```
nvdec_scene_detect <input> [--threshold <val>] [--min-gap-ms <ms>] [--downscale <n>] [--csv <file>] [--verbose]
```
Options:
- `--threshold` (float, default 18.0): MAD cut threshold.
- `--min-gap-ms` (int, default 400): Minimum time between reported cuts (debounce).
- `--downscale` (int, default 2): Spatial sampling stride (1 = full res). Higher = faster, noisier.
- `--csv` (path): Write `timestamp,frame_idx,mad` lines for each detected cut.
- `--verbose`: Extra diagnostic logs.

Example:
```
nvdec_scene_detect sample.mp4 --threshold 20 --min-gap-ms 500 --downscale 4 --csv cuts.csv
```
Console output line example:
```
12.4667, frame 374, mad=37.21
```
Meaning: scene cut at 12.4667s on frame 374 with MAD 37.21.

## Determining Proper Threshold
Start with default (18) and inspect a few sample outputs. Increase if you see false positives; decrease if cuts are missed. Because MAD uses only Y plane and simple sampling, optimal values vary by content and downscale factor.

## Verifying Hardware Acceleration
Check FFmpeg supports CUDA/NVDEC:
```
ffmpeg -hwaccels
ffmpeg -decoders | findstr cuvid       (Windows)
ffmpeg -decoders | grep cuvid          (Linux/macOS)
```
If CUDA device creation fails the program logs a warning and continues in software decode mode.

## CSV Output
When `--csv file.csv` is specified, only detected cuts are written (not every frame). Header: `timestamp,frame_idx,mad`.

## Limitations / TODO
- Only uses luma plane of NV12 / NV21 / YUV420P. Other pixel formats are skipped (could add swscale path).
- Only detects hard cuts (no gradual dissolve detection).
- CPU fallback path currently copies full luma each frame; could be optimized or moved fully to GPU with an upload.
- No multi-stream batch processing.
- No adaptive thresholding.

Future enhancements (ideas):
- Add GPU downscale kernel for better sampling quality.
- Support additional pixel formats via on-GPU conversion.
- Sliding window variance / histogram metrics for more robust detection.
- Optional output of all frame MAD values for offline analysis.

## Troubleshooting
- Decoder not found: ensure FFmpeg build includes the codec and CUDA support (look for `h264_cuvid`, `hevc_cuvid`).
- HW device creation fails: driver/CUDA mismatch or missing FFmpeg configuration.
- Missing DLLs at runtime (Windows): copy FFmpeg & CUDA runtime DLLs next to the executable or add to PATH.
- Very low MAD values: maybe downscale too large; try `--downscale 1`.
- High false positives: raise threshold or increase `--min-gap-ms`.

## Performance Notes
Downscaling factor tradeoff: approximate runtime proportional to sampled pixels ( ~ 1 / ds^2 ). For 4K content, `--downscale 4` often sufficient for hard cuts.

## License
SPDX-License-Identifier: MIT (see file headers).

## Disclaimer
Demo-quality code for educational / prototyping use. Not production hardened.


# CUDA NVDEC Scene Detection

Lightweight demo tool that decodes video with FFmpeg + NVIDIA NVDEC (CUDA) and detects hard scene cuts on the GPU using a simple Mean Absolute Difference (MAD) over the luma plane. Frames stay on the GPU when possible; only a small CUDA kernel runs per frame pair.

## Features
- Hardware accelerated H.264 / HEVC decode via FFmpeg (CUVID / AVHWDeviceType CUDA).
- Zero‑copy path for NV12 luma when available.
- CUDA kernel computes MAD with optional spatial downscale sampling.
- Optional CSV output of detected cuts.
- Graceful fallback to CPU MAD when frames are not on the GPU (still functional, slower).

## Algorithm (summary)
1. Decode frames (prefer NVDEC). Obtain luma plane from any supported pixel format.
2. `FrameNormalizer` extracts 8-bit luma: returns data[0] directly for planar 8-bit YUV formats, or converts via `sws_scale` for 10/12-bit, RGB, packed-YUV, and similar formats.
3. Downsample logically by strided sampling factor `--downscale` (no resize kernel, just sampling points).
4. Compute sum(|Y_t - Y_{t-1}|) and divide by sampled pixel count => MAD.
5. Declare a cut if MAD > threshold AND time since last cut > min gap.

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

## Supported Pixel Formats
Scene detection works on a broad range of pixel formats via the `FrameNormalizer` component:

| Format group | Examples | Handling |
|---|---|---|
| 8-bit planar YUV | NV12, NV21, YUV420P, YUV422P, YUV444P, GRAY8, and JPEG/alpha variants | Direct – luma plane returned as-is (zero copy) |
| 10/12-bit planar YUV | YUV420P10, YUV422P10, YUV444P10, YUV420P12, … | CPU swscale → YUV420P, luma extracted |
| 10-bit semi-planar | P010LE, P010BE, P016LE, P016BE | CPU swscale → YUV420P, luma extracted |
| Packed RGB / RGBA | RGB24, BGR24, RGBA, BGRA, ARGB, ABGR, RGB48 | CPU swscale → YUV420P, luma extracted |
| Packed YUV | YUYV422, UYVY422, YVYU422 | CPU swscale → YUV420P, luma extracted |
| Planar RGB | GBRP, GBRAP | CPU swscale → YUV420P, luma extracted |
| High-bit-depth gray | GRAY10LE, GRAY12LE, GRAY16 | CPU swscale → YUV420P, luma extracted |
| CUDA device surface | AV_PIX_FMT_CUDA | GPU path (direct device pointer, no CPU conversion) |

The swscale conversion is done once per format/resolution change; the `SwsContext` is cached and reused for subsequent frames of the same type, minimising overhead.

**Performance impact of the swscale path:** For formats that require conversion, each frame incurs an additional CPU pass proportional to frame area. On typical 1080p content with `--downscale 2` the extra cost is dominated by the swscale call rather than the CUDA MAD kernel, so enabling a software decoder together with an RGB or 10-bit source will be noticeably slower than an NV12 NVDEC path. When latency matters, prefer hardware-decoded NV12/CUDA output.

## Limitations / TODO
- Only detects hard cuts (no gradual dissolve detection).
- CPU fallback path currently copies full luma each frame; could be optimized or moved fully to GPU with an upload (the `dev_luma_ptr` from `FrameNormalizer` can be passed directly to `cudaMemcpy`).
- No multi-stream batch processing.
- No adaptive thresholding.

Future enhancements (ideas):
- Add GPU downscale kernel for better sampling quality.
- Implement on-GPU conversion kernels for common high-bit-depth formats (P010, YUV420P10) to skip the CPU swscale pass.
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


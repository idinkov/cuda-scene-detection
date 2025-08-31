// ffmpeg_nvdec_scene_detect.cpp
// SPDX-License-Identifier: MIT
// Requires: FFmpeg built with CUDA/NVDEC support, CUDA Toolkit, NVIDIA driver

/*
Overview
--------
This program decodes video using FFmpeg's NVDEC/CUVID/CUDA hardware acceleration and keeps frames in GPU memory.
It computes a simple mean absolute difference (MAD) between consecutive frames using a small CUDA kernel to
detect hard scene cuts. The goal is low CPU overhead and high throughput on NVIDIA GPUs.

Notes / Requirements
--------------------
- FFmpeg must be built with NVDEC/CUVID/CUDA support (e.g. decoders h264_cuvid/hevc_cuvid or AV_HWDEVICE_TYPE_CUDA).
- Link against FFmpeg libraries: avformat, avcodec, avutil, swscale (if needed).
- Link against CUDA (nvcc) to build the CUDA kernel.
- Tested conceptually against FFmpeg hw_decode.c example and NVIDIA Video Codec SDK docs.

Build (example on Linux):
  nvcc -ccbin g++ -O2 -arch=sm_60 -c cuda_kernels.cu -o cuda_kernels.o
  g++ -O2 -std=c++17 ffmpeg_nvdec_scene_detect.cpp cuda_kernels.o -o nvdec_scene_detect \
    `pkg-config --cflags --libs libavformat libavcodec libavutil libswscale` -lcuda -ldl -lpthread

Run:
  ./nvdec_scene_detect input.mp4 --threshold 18 --min-gap-ms 400

References:
- FFmpeg hw_decode.c example (hw accelerated decode). See ffmpeg.org examples.
- NVIDIA Video Codec SDK: docs on FFmpeg integration and NVDEC.
*/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// CUDA kernel declaration (implemented in cuda_kernels.cu)
extern "C" float compute_mad_cuda(const uint8_t* frameA_dev, int pitchA, const uint8_t* frameB_dev, int pitchB, int width, int height, int downscale);

struct Args { std::string input; double threshold=18.0; int minGapMs=400; bool verbose=false; std::optional<std::string> csv; int downscale=2; };

static std::optional<Args> parse_args(int argc, char** argv){
    if (argc < 2) return std::nullopt;
    Args a; a.input = argv[1];
    for (int i=2;i<argc;++i){ std::string k=argv[i]; if(k=="--threshold" && i+1<argc) a.threshold=std::stod(argv[++i]);
        else if(k=="--min-gap-ms" && i+1<argc) a.minGapMs=std::stoi(argv[++i]);
        else if(k=="--csv" && i+1<argc) a.csv=argv[++i];
        else if(k=="--downscale" && i+1<argc) a.downscale=std::max(1, std::stoi(argv[++i]));
        else if(k=="--verbose") a.verbose=true;
        else { std::cerr<<"Unknown arg: "<<k<<"\n"; return std::nullopt; }
    }
    if (a.downscale < 1) a.downscale = 1; // safety
    return a;
}

// Helper to get a string representation of AVError
static std::string av_err2str_wrap(int err){ char buf[256]; av_strerror(err, buf, sizeof(buf)); return std::string(buf); }

int main(int argc, char** argv){
    av_log_set_level(AV_LOG_ERROR);
    auto parsed = parse_args(argc, argv);
    if(!parsed){ std::cerr<<"Usage: "<<argv[0]<<" <input> [--threshold <val>] [--min-gap-ms <ms>] [--downscale <n>] [--csv <path>] [--verbose]\n"; return 2; }
    Args args = *parsed;

    avformat_network_init();
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, args.input.c_str(), nullptr, nullptr) < 0){ std::cerr<<"Failed to open input\n"; return 3; }
    if (avformat_find_stream_info(fmt, nullptr) < 0){ std::cerr<<"Failed to find stream info\n"; return 4; }

    int video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream < 0){ std::cerr<<"No video stream\n"; return 5; }
    AVStream* st = fmt->streams[video_stream];

    // Find a hardware-accelerated decoder if possible
    // FFmpeg newer API returns const AVCodec* from find functions
    const AVCodec* dec = nullptr;
    AVCodecParameters* par = st->codecpar;

    // Try codec name with cuvid (h264_cuvid / hevc_cuvid) first for NVDEC/CUVID decoders
    if (par->codec_id == AV_CODEC_ID_H264) dec = avcodec_find_decoder_by_name("h264_cuvid");
    else if (par->codec_id == AV_CODEC_ID_HEVC) dec = avcodec_find_decoder_by_name("hevc_cuvid");
    if (!dec) {
        // fallback to generic decoder (FFmpeg may internally map to hw accelerated backend if configured)
        dec = avcodec_find_decoder(par->codec_id);
    }
    if (!dec){ std::cerr<<"Decoder not found\n"; return 6; }

    AVCodecContext* dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) { std::cerr<<"Failed alloc codec ctx\n"; return 7; }
    if (avcodec_parameters_to_context(dec_ctx, par) < 0){ std::cerr<<"Failed to copy codec params\n"; return 8; }

    // Try to create a CUDA HW device context so FFmpeg can output frames in GPU memory.
    AVBufferRef* hw_device_ctx = nullptr;
    int err = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (err < 0){
        std::cerr << "Could not create HW device ctx (CUDA): " << av_err2str_wrap(err) << "\n";
        std::cerr << "Make sure FFmpeg was built with --enable-cuda --enable-nvdec / Video Codec SDK integration.\n";
        // proceed without hw accel (will decode to system memory)
        hw_device_ctx = nullptr;
    } else {
        dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    }

    if (avcodec_open2(dec_ctx, dec, nullptr) < 0){ std::cerr<<"Failed to open codec\n"; return 9; }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc(); // for software-transferred frames if needed
    if (!pkt || !frame || !sw_frame) { std::cerr<<"Alloc fail\n"; return 10; }

    std::ofstream csv;
    if (args.csv) { csv.open(*args.csv); if(!csv) { std::cerr<<"Could not open CSV\n"; } else csv<<"timestamp,frame_idx,mad\n"; }

    int64_t frame_idx = 0;
    double last_cut_time = -1e9;
    int video_fps = st->avg_frame_rate.num>0 ? (int)(av_q2d(st->avg_frame_rate)+0.5) : 30;
    double frame_time = 1.0 / (video_fps>0?video_fps:30);

    // We will hold previous frame device pointers (NV12 luma plane) to compare.
    // Implementation detail: Many HW-decoded frames for NV12 have two planes: Y (luma) and UV (chroma).
    // For MAD on intensity we only need luma (Y). We'll attempt to retrieve the device pointer for plane 0.

    // previous device buffer info
    const uint8_t* prev_dev_ptr = nullptr;
    int prev_linesize = 0;
    int width = dec_ctx->width;
    int height = dec_ctx->height;
    bool host_prev_alloc = false; // tracks if prev_dev_ptr was malloc'd (CPU path)

    // Owned CUDA buffers for previous & current frame luma (device memory we control)
    uint8_t* prev_dev_owned = nullptr; size_t prev_pitch = 0;
    uint8_t* curr_dev_owned = nullptr; size_t curr_pitch = 0;
    bool cuda_buffers_inited = false;

    auto ensure_cuda_buffers = [&](int w, int h) -> bool {
        if (cuda_buffers_inited) return true;
        // Allocate two pitched buffers for ping-pong
        cudaError_t ce;
        ce = cudaMallocPitch((void**)&prev_dev_owned, &prev_pitch, (size_t)w, (size_t)h);
        if (ce != cudaSuccess){ std::cerr << "cudaMallocPitch prev failed: " << cudaGetErrorString(ce) << "\n"; return false; }
        ce = cudaMallocPitch((void**)&curr_dev_owned, &curr_pitch, (size_t)w, (size_t)h);
        if (ce != cudaSuccess){ std::cerr << "cudaMallocPitch curr failed: " << cudaGetErrorString(ce) << "\n"; cudaFree(prev_dev_owned); prev_dev_owned=nullptr; return false; }
        cuda_buffers_inited = true; return true;
    };

    auto copy_luma_to_owned = [&](const uint8_t* srcPtr, int srcPitch, bool srcOnDevice, uint8_t* dstPtr, size_t dstPitch, int w, int h) -> bool {
        cudaMemcpyKind kind = srcOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        cudaError_t ce = cudaMemcpy2D(dstPtr, dstPitch, srcPtr, (size_t)srcPitch, (size_t)w, (size_t)h, kind);
        if (ce != cudaSuccess){ std::cerr << "cudaMemcpy2D luma failed: " << cudaGetErrorString(ce) << "\n"; return false; }
        return true;
    };

    bool have_prev = false;
    // Main loop: read packets and send to decoder
    while (av_read_frame(fmt, pkt) >= 0){
        if (pkt->stream_index != video_stream){ av_packet_unref(pkt); continue; }
        err = avcodec_send_packet(dec_ctx, pkt);
        if (err < 0){ std::cerr<<"Error sending packet: "<<av_err2str_wrap(err)<<"\n"; av_packet_unref(pkt); break; }
        av_packet_unref(pkt);

        // receive frames
        while (true){
            err = avcodec_receive_frame(dec_ctx, frame);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
            if (err < 0){ std::cerr<<"Error while decoding: "<<av_err2str_wrap(err)<<"\n"; goto end; }

            AVFrame* processing_frame = nullptr;

            if (frame->hw_frames_ctx) {
                // Frame is in HW (e.g., AV_PIX_FMT_CUDA or AV_PIX_FMT_NV12 with hwctx). Transfer to a CUDA-accessible frame if supported.
                // We'll try av_hwframe_transfer_data() to a frame with format AV_PIX_FMT_CUDA or AV_PIX_FMT_NV12 and get device pointers.

                // Create a software-frame that will contain the transferred data in device memory format
                av_frame_unref(sw_frame);
                if ((err = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
                    // Transfer failed; fall back to original frame (system memory if already there) and skip GPU MAD.
                    std::cerr << "av_hwframe_transfer_data failed: " << av_err2str_wrap(err) << ". Using original frame.\n";
                    processing_frame = frame;
                } else {
                    processing_frame = sw_frame;
                }
            } else {
                // Frame already in system memory
                processing_frame = frame;
            }

            // We expect processing_frame->data[0] to point to either host memory or device memory depending on transfer target.
            // FFmpeg uses AV_PIX_FMT_CUDA to indicate CUDA device memory; however availability depends on build. For portability,
            // we will handle two cases: device memory pointer accessible via processing_frame->data[0] (cuda) or host memory (sw).

            // Only use luma plane for MAD. We handle both packed RGB and planar YUV by converting if needed.
            // For simplicity: if pixel format is YUV420/420P (AV_PIX_FMT_NV12/AV_PIX_FMT_YUV420P), take data[0] as luma.

            int pix_fmt = processing_frame->format;
            const uint8_t* dev_luma_ptr = nullptr; // pointer to device or host luma plane
            int luma_linesize = 0;
            bool luma_on_device = false;

            if (pix_fmt == AV_PIX_FMT_CUDA) {
                // If AV_PIX_FMT_CUDA, FFmpeg wraps a GPU surface. According to docs, data[] may contain device pointers.
                // We'll assume data[0] contains a device pointer to Y (NV12) or packed data depending on decoder.
                dev_luma_ptr = processing_frame->data[0];
                luma_linesize = processing_frame->linesize[0];
                luma_on_device = true;
            } else if (pix_fmt == AV_PIX_FMT_NV12 || pix_fmt == AV_PIX_FMT_NV21 || pix_fmt == AV_PIX_FMT_YUV420P) {
                dev_luma_ptr = processing_frame->data[0];
                luma_linesize = processing_frame->linesize[0];
                luma_on_device = false; // treat as host memory; upload before GPU MAD if desired
            } else {
                // For packed formats like RGB, we need to convert to Y plane; fallback: use sw conversion to YUV on CPU (slow path)
                // Convert to YUV420 on CPU then upload to CUDA and compute MAD. For brevity we skip this path in this example.
                std::cerr<<"Unsupported pixel format (need Y plane). Consider building FFmpeg with CUDA hw frames in NV12/YUV420 formats.\n";
                continue;
            }

            // If we don't have device luma pointer, perform a software copy to CUDA device using cudaMemcpy (user must have built with CUDA; compute_mad_cuda expects device pointers)
            // For simplicity we assume dev_luma_ptr points to device memory when hw_device_ctx != null.
            if (!luma_on_device) {
                // Upload host luma plane to CUDA device memory. We'll use compute_mad_cuda which will internally cudaMalloc and free for temporary buffers.
                // For simplicity we call compute_mad_cuda with host pointer but that function expects device pointer; in a production app you'd cudaMalloc & cudaMemcpy here.
                // We'll skip this complex path in this demo; instead, we fall back to CPU-based MAD for this frame.

                // CPU fallback MAD (very simple)
                if (prev_dev_ptr == nullptr) {
                    // store host luma pointer for next iteration via a copied host buffer
                    // allocate and copy
                    size_t hsize = luma_linesize * height;
                    uint8_t* host_copy = (uint8_t*)malloc(hsize);
                    if (!host_copy) continue;
                    memcpy(host_copy, processing_frame->data[0], hsize);
                    prev_dev_ptr = host_copy;
                    prev_linesize = luma_linesize;
                    host_prev_alloc = true;
                } else {
                    // compute MAD between prev_dev_ptr (host) and current host pointer with optional downscale sampling
                    uint8_t* cur = (uint8_t*)processing_frame->data[0];
                    int ds = args.downscale < 1 ? 1 : args.downscale;
                    int sampW = (width + ds - 1) / ds;
                    int sampH = (height + ds - 1) / ds;
                    double sum = 0.0;
                    for (int y_ds=0;y_ds<sampH;++y_ds){
                        int y = y_ds * ds; if (y >= height) y = height - 1;
                        const uint8_t* r0 = prev_dev_ptr + y*prev_linesize;
                        const uint8_t* r1 = cur + y*luma_linesize;
                        for (int x_ds=0;x_ds<sampW;++x_ds){
                            int x = x_ds * ds; if (x >= width) x = width - 1;
                            sum += fabs((double)r0[x] - (double)r1[x]);
                        }
                    }
                    double mad = sum / (sampW * (double)sampH);
                    double ts = frame_idx * frame_time;
                    bool isCut = mad > args.threshold && (ts - last_cut_time)*1000.0 > args.minGapMs;
                    if (isCut){ last_cut_time = ts; std::cout<<ts<<", frame "<<frame_idx<<", mad="<<mad<<"\n"; if(csv) csv<<ts<<","<<frame_idx<<","<<mad<<"\n"; }
                    // replace previous buffer (full-resolution copy for next comparison)
                    memcpy((void*)prev_dev_ptr, cur, luma_linesize*height);
                }

                ++frame_idx;
                av_frame_unref(frame);
                continue;
            }

            // At this point we have device-accessible luma plane pointer in dev_luma_ptr.
            // Compute MAD with previous device buffer.
            // REPLACED unsafe raw pointer reuse logic with owned device buffers.
            if (!ensure_cuda_buffers(width, height)) {
                av_frame_unref(frame);
                ++frame_idx;
                continue;
            }
            if (!copy_luma_to_owned(dev_luma_ptr, luma_linesize, luma_on_device, have_prev ? curr_dev_owned : prev_dev_owned, have_prev ? curr_pitch : prev_pitch, width, height)) {
                av_frame_unref(frame);
                ++frame_idx;
                continue;
            }
            if (!have_prev) {
                have_prev = true; // stored first frame into prev_dev_owned
            } else {
                // current frame just copied into curr_dev_owned, compute MAD
                float mad = compute_mad_cuda(prev_dev_owned, (int)prev_pitch, curr_dev_owned, (int)curr_pitch, width, height, args.downscale);
                double ts = frame_idx * frame_time;
                bool isCut = mad > args.threshold && (ts - last_cut_time)*1000.0 > args.minGapMs;
                if (isCut){ last_cut_time = ts; std::cout<<ts<<", frame "<<frame_idx<<", mad="<<mad<<"\n"; if(csv) csv<<ts<<","<<frame_idx<<","<<mad<<"\n"; }
                // swap buffers for next iteration (so prev holds last frame)
                std::swap(prev_dev_owned, curr_dev_owned);
                std::swap(prev_pitch, curr_pitch);
            }

            ++frame_idx;
            av_frame_unref(frame);
        }
    }

    end:
    if (csv) csv.close();
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
    avformat_network_deinit();
    if (cuda_buffers_inited){ cudaFree(prev_dev_owned); cudaFree(curr_dev_owned); }
    if (host_prev_alloc && prev_dev_ptr) free((void*)prev_dev_ptr);
    return 0;
}

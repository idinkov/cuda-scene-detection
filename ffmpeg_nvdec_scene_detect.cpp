// ffmpeg_nvdec_scene_detect.cpp
// SPDX-License-Identifier: MIT
// Requires: FFmpeg built with CUDA/NVDEC support, CUDA Toolkit, NVIDIA driver

/*
Overview
--------
This program decodes video using FFmpeg's NVDEC/CUVID/CUDA hardware acceleration and keeps frames in GPU memory.
It computes a simple mean absolute difference (MAD) between consecutive frames using a small CUDA kernel to
detect hard scene cuts. The goal is low CPU overhead and high throughput on NVIDIA GPUs.

Batch / multi-stream mode
-------------------------
Pass multiple input files as positional arguments or use --input-list to specify a file containing one path
per line. Each stream runs in its own decoder thread; all streams share a single GPU work queue so MAD
computations are serialised through the GPU worker thread. Per-stream latency and throughput metrics are
printed on completion.

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

Run (single stream):
  ./nvdec_scene_detect input.mp4 --threshold 18 --min-gap-ms 400

Run (batch):
  ./nvdec_scene_detect a.mp4 b.mp4 c.mp4 --threshold 18
  ./nvdec_scene_detect --input-list videos.txt --threshold 18 --csv cuts.csv

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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <algorithm>
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

// ---------------------------------------------------------------------------
// GPU work queue: serialises compute_mad_cuda calls across decoder threads
// ---------------------------------------------------------------------------

struct GpuTask {
    const uint8_t* a_dev;
    int pitch_a;
    const uint8_t* b_dev;
    int pitch_b;
    int width, height, downscale;
    std::promise<float> result;
};

// GpuWorkQueue: thread-safe queue for MAD computation tasks.
// A single GPU worker thread drains this queue, keeping GPU access serialised
// while decoder threads run concurrently on their own CPU threads.
class GpuWorkQueue {
public:
    // Submit a MAD task; returns a future that resolves to the MAD value.
    std::future<float> submit(const uint8_t* a, int pa, const uint8_t* b, int pb,
                               int w, int h, int ds) {
        GpuTask task;
        task.a_dev = a; task.pitch_a = pa;
        task.b_dev = b; task.pitch_b = pb;
        task.width = w; task.height = h; task.downscale = ds;
        auto fut = task.result.get_future();
        {
            std::unique_lock<std::mutex> lk(mtx_);
            queue_.push(std::move(task));
        }
        cv_.notify_one();
        return fut;
    }

    // Signal the worker to drain the queue and exit.
    void shutdown() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            done_ = true;
        }
        cv_.notify_all();
    }

    // Worker loop: run this in a dedicated thread.
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]{ return !queue_.empty() || done_; });
            if (queue_.empty()) break; // done_ is true and queue is empty
            GpuTask task = std::move(queue_.front());
            queue_.pop();
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
    std::queue<GpuTask> queue_;
    bool done_ = false;
};

// ---------------------------------------------------------------------------
// Arguments and per-stream metrics
// ---------------------------------------------------------------------------

struct Args {
    std::vector<std::string> inputs;          // one or more input video paths
    double threshold = 18.0;
    int minGapMs = 400;
    bool verbose = false;
    std::optional<std::string> csv;           // base CSV output path
    int downscale = 2;
    std::optional<std::string> inputList;     // file containing one path per line
    int jobs = 0;                             // max parallel streams; 0 = all parallel
};

struct StreamMetrics {
    std::string path;
    int64_t frames_decoded = 0;
    int cuts_detected = 0;
    double wall_time_sec = 0.0;
    double throughput_fps = 0.0;
    int exit_code = 0;
    std::string error_msg;
};

// Mutex protecting stdout/stderr from concurrent writes across decoder threads.
static std::mutex g_print_mtx;

static std::optional<Args> parse_args(int argc, char** argv) {
    if (argc < 2) return std::nullopt;
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--threshold" && i+1 < argc) a.threshold = std::stod(argv[++i]);
        else if (k == "--min-gap-ms" && i+1 < argc) a.minGapMs = std::stoi(argv[++i]);
        else if (k == "--csv" && i+1 < argc) a.csv = argv[++i];
        else if (k == "--downscale" && i+1 < argc) a.downscale = std::max(1, std::stoi(argv[++i]));
        else if (k == "--verbose") a.verbose = true;
        else if (k == "--input-list" && i+1 < argc) a.inputList = argv[++i];
        else if (k == "--jobs" && i+1 < argc) a.jobs = std::max(1, std::stoi(argv[++i]));
        else if (k.rfind("--", 0) != 0) a.inputs.push_back(k); // positional: treat as input file
        else { std::cerr << "Unknown arg: " << k << "\n"; return std::nullopt; }
    }
    // Load additional paths from --input-list file
    if (a.inputList) {
        std::ifstream lf(*a.inputList);
        if (!lf) { std::cerr << "Cannot open input list: " << *a.inputList << "\n"; return std::nullopt; }
        std::string line;
        while (std::getline(lf, line)) {
            auto s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            line = line.substr(s);
            auto e = line.find_last_not_of(" \t\r\n");
            if (e != std::string::npos) line = line.substr(0, e + 1);
            if (!line.empty() && line[0] != '#') a.inputs.push_back(line);
        }
    }
    if (a.inputs.empty()) return std::nullopt;
    if (a.downscale < 1) a.downscale = 1;
    return a;
}

// Helper to get a string representation of AVError
static std::string av_err2str_wrap(int err) {
    char buf[256]; av_strerror(err, buf, sizeof(buf)); return std::string(buf);
}

// Derive a per-stream CSV path from the base path and stream index.
// When there is only one stream the base path is used unchanged.
static std::string derive_csv_path(const std::string& base, int idx, int total) {
    if (total == 1) return base;
    auto dot = base.rfind('.');
    if (dot == std::string::npos) return base + "_" + std::to_string(idx);
    return base.substr(0, dot) + "_" + std::to_string(idx) + base.substr(dot);
}

// ---------------------------------------------------------------------------
// Single-stream processing (runs in its own decoder thread)
// ---------------------------------------------------------------------------
static StreamMetrics process_stream(int stream_idx, const std::string& path,
                                    const Args& args, GpuWorkQueue& gpu_queue,
                                    int total_streams) {
    StreamMetrics m;
    m.path = path;
    auto wall_start = std::chrono::steady_clock::now();

    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) {
        m.exit_code = 3; m.error_msg = "Failed to open input: " + path; return m;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt);
        m.exit_code = 4; m.error_msg = "Failed to find stream info"; return m;
    }

    int video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream < 0) {
        avformat_close_input(&fmt);
        m.exit_code = 5; m.error_msg = "No video stream"; return m;
    }
    AVStream* st = fmt->streams[video_stream];

    // Find a hardware-accelerated decoder if possible
    const AVCodec* dec = nullptr;
    AVCodecParameters* par = st->codecpar;
    if (par->codec_id == AV_CODEC_ID_H264) dec = avcodec_find_decoder_by_name("h264_cuvid");
    else if (par->codec_id == AV_CODEC_ID_HEVC) dec = avcodec_find_decoder_by_name("hevc_cuvid");
    if (!dec) dec = avcodec_find_decoder(par->codec_id);
    if (!dec) {
        avformat_close_input(&fmt);
        m.exit_code = 6; m.error_msg = "Decoder not found"; return m;
    }

    AVCodecContext* dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) {
        avformat_close_input(&fmt);
        m.exit_code = 7; m.error_msg = "Failed alloc codec ctx"; return m;
    }
    if (avcodec_parameters_to_context(dec_ctx, par) < 0) {
        avcodec_free_context(&dec_ctx); avformat_close_input(&fmt);
        m.exit_code = 8; m.error_msg = "Failed to copy codec params"; return m;
    }

    // Try to create a CUDA HW device context so FFmpeg can output frames in GPU memory.
    AVBufferRef* hw_device_ctx = nullptr;
    int err = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (err < 0) {
        if (args.verbose) {
            std::lock_guard<std::mutex> lk(g_print_mtx);
            std::cerr << "[stream " << stream_idx << "] Could not create HW device ctx (CUDA): "
                      << av_err2str_wrap(err) << "\n";
        }
        hw_device_ctx = nullptr;
    } else {
        dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    }

    if (avcodec_open2(dec_ctx, dec, nullptr) < 0) {
        avcodec_free_context(&dec_ctx); avformat_close_input(&fmt);
        if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
        m.exit_code = 9; m.error_msg = "Failed to open codec"; return m;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();
    if (!pkt || !frame || !sw_frame) {
        av_packet_free(&pkt); av_frame_free(&frame); av_frame_free(&sw_frame);
        avcodec_free_context(&dec_ctx); avformat_close_input(&fmt);
        if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
        m.exit_code = 10; m.error_msg = "Alloc fail"; return m;
    }

    // Per-stream CSV output
    std::ofstream csv_file;
    if (args.csv) {
        std::string csv_path = derive_csv_path(*args.csv, stream_idx, total_streams);
        csv_file.open(csv_path);
        if (!csv_file) {
            std::lock_guard<std::mutex> lk(g_print_mtx);
            std::cerr << "[stream " << stream_idx << "] Could not open CSV: " << csv_path << "\n";
        } else {
            csv_file << "timestamp,frame_idx,mad\n";
        }
    }

    int64_t frame_idx = 0;
    double last_cut_time = -1e9;
    int video_fps = st->avg_frame_rate.num > 0 ? (int)(av_q2d(st->avg_frame_rate) + 0.5) : 30;
    double frame_time = 1.0 / (video_fps > 0 ? video_fps : 30);
    int width = dec_ctx->width;
    int height = dec_ctx->height;

    // Helper: print a detected cut (thread-safe).
    // In single-stream mode the output matches the original format exactly.
    auto report_cut = [&](double ts, int64_t fidx, double mad) {
        ++m.cuts_detected;
        std::lock_guard<std::mutex> lk(g_print_mtx);
        if (total_streams > 1) std::cout << "[stream " << stream_idx << "] ";
        std::cout << ts << ", frame " << fidx << ", mad=" << mad << "\n";
        if (csv_file) csv_file << ts << "," << fidx << "," << mad << "\n";
    };

    // Owned CUDA buffers for previous & current frame luma (device memory we control)
    uint8_t* prev_dev_owned = nullptr; size_t prev_pitch = 0;
    uint8_t* curr_dev_owned = nullptr; size_t curr_pitch = 0;
    bool cuda_buffers_inited = false;

    auto ensure_cuda_buffers = [&](int w, int h) -> bool {
        if (cuda_buffers_inited) return true;
        cudaError_t ce;
        ce = cudaMallocPitch((void**)&prev_dev_owned, &prev_pitch, (size_t)w, (size_t)h);
        if (ce != cudaSuccess) {
            if (args.verbose) {
                std::lock_guard<std::mutex> lk(g_print_mtx);
                std::cerr << "[stream " << stream_idx << "] cudaMallocPitch (prev) failed: "
                          << cudaGetErrorString(ce) << "\n";
            }
            return false;
        }
        ce = cudaMallocPitch((void**)&curr_dev_owned, &curr_pitch, (size_t)w, (size_t)h);
        if (ce != cudaSuccess) {
            if (args.verbose) {
                std::lock_guard<std::mutex> lk(g_print_mtx);
                std::cerr << "[stream " << stream_idx << "] cudaMallocPitch (curr) failed: "
                          << cudaGetErrorString(ce) << "\n";
            }
            cudaFree(prev_dev_owned); prev_dev_owned = nullptr; return false;
        }
        cuda_buffers_inited = true; return true;
    };

    auto copy_luma_to_owned = [&](const uint8_t* srcPtr, int srcPitch, bool srcOnDevice,
                                   uint8_t* dstPtr, size_t dstPitch, int w, int h) -> bool {
        cudaMemcpyKind kind = srcOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        return cudaMemcpy2D(dstPtr, dstPitch, srcPtr, (size_t)srcPitch,
                            (size_t)w, (size_t)h, kind) == cudaSuccess;
    };

    bool have_prev = false;

    // CPU fallback: previous luma host buffer
    const uint8_t* host_prev_ptr = nullptr;
    int host_prev_linesize = 0;
    bool host_prev_alloc = false;

    // Main decode loop
    bool decode_error = false;
    while (!decode_error && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != video_stream) { av_packet_unref(pkt); continue; }
        err = avcodec_send_packet(dec_ctx, pkt);
        if (err < 0) { av_packet_unref(pkt); break; }
        av_packet_unref(pkt);

        while (!decode_error) {
            err = avcodec_receive_frame(dec_ctx, frame);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
            if (err < 0) { decode_error = true; break; }

            AVFrame* processing_frame = nullptr;
            if (frame->hw_frames_ctx) {
                av_frame_unref(sw_frame);
                if (av_hwframe_transfer_data(sw_frame, frame, 0) < 0) {
                    processing_frame = frame;
                } else {
                    processing_frame = sw_frame;
                }
            } else {
                processing_frame = frame;
            }

            int pix_fmt = processing_frame->format;
            const uint8_t* dev_luma_ptr = nullptr;
            int luma_linesize = 0;
            bool luma_on_device = false;

            if (pix_fmt == AV_PIX_FMT_CUDA) {
                dev_luma_ptr = processing_frame->data[0];
                luma_linesize = processing_frame->linesize[0];
                luma_on_device = true;
            } else if (pix_fmt == AV_PIX_FMT_NV12 || pix_fmt == AV_PIX_FMT_NV21 ||
                       pix_fmt == AV_PIX_FMT_YUV420P) {
                dev_luma_ptr = processing_frame->data[0];
                luma_linesize = processing_frame->linesize[0];
                luma_on_device = false;
            } else {
                if (args.verbose) {
                    std::lock_guard<std::mutex> lk(g_print_mtx);
                    std::cerr << "[stream " << stream_idx
                              << "] Unsupported pixel format; skipping frame.\n";
                }
                ++frame_idx;
                av_frame_unref(frame);
                continue;
            }

            if (!luma_on_device) {
                // CPU fallback MAD path
                if (host_prev_ptr == nullptr) {
                    size_t hsize = (size_t)luma_linesize * height;
                    uint8_t* host_copy = (uint8_t*)malloc(hsize);
                    if (!host_copy) { ++frame_idx; av_frame_unref(frame); continue; }
                    memcpy(host_copy, processing_frame->data[0], hsize);
                    host_prev_ptr = host_copy;
                    host_prev_linesize = luma_linesize;
                    host_prev_alloc = true;
                } else {
                    uint8_t* cur = (uint8_t*)processing_frame->data[0];
                    int ds = args.downscale < 1 ? 1 : args.downscale;
                    int sampW = (width + ds - 1) / ds;
                    int sampH = (height + ds - 1) / ds;
                    double sum = 0.0;
                    for (int y_ds = 0; y_ds < sampH; ++y_ds) {
                        int y = y_ds * ds; if (y >= height) y = height - 1;
                        const uint8_t* r0 = host_prev_ptr + y * host_prev_linesize;
                        const uint8_t* r1 = cur + y * luma_linesize;
                        for (int x_ds = 0; x_ds < sampW; ++x_ds) {
                            int x = x_ds * ds; if (x >= width) x = width - 1;
                            sum += fabs((double)r0[x] - (double)r1[x]);
                        }
                    }
                    double mad = sum / (sampW * (double)sampH);
                    double ts = frame_idx * frame_time;
                    if (mad > args.threshold && (ts - last_cut_time) * 1000.0 > args.minGapMs) {
                        last_cut_time = ts;
                        report_cut(ts, frame_idx, mad);
                    }
                    memcpy((void*)host_prev_ptr, cur, (size_t)luma_linesize * height);
                }
                ++frame_idx;
                av_frame_unref(frame);
                continue;
            }

            // GPU path: copy luma into owned device buffers, then submit to shared GPU work queue.
            if (!ensure_cuda_buffers(width, height)) {
                ++frame_idx; av_frame_unref(frame); continue;
            }
            if (!copy_luma_to_owned(dev_luma_ptr, luma_linesize, luma_on_device,
                                    have_prev ? curr_dev_owned : prev_dev_owned,
                                    have_prev ? curr_pitch    : prev_pitch,
                                    width, height)) {
                ++frame_idx; av_frame_unref(frame); continue;
            }
            if (!have_prev) {
                have_prev = true;
            } else {
                // Submit MAD computation to the shared GPU work queue and wait for result.
                auto fut = gpu_queue.submit(prev_dev_owned, (int)prev_pitch,
                                            curr_dev_owned, (int)curr_pitch,
                                            width, height, args.downscale);
                float mad = fut.get();
                double ts = frame_idx * frame_time;
                if (mad > args.threshold && (ts - last_cut_time) * 1000.0 > args.minGapMs) {
                    last_cut_time = ts;
                    report_cut(ts, frame_idx, (double)mad);
                }
                std::swap(prev_dev_owned, curr_dev_owned);
                std::swap(prev_pitch, curr_pitch);
            }
            ++frame_idx;
            av_frame_unref(frame);
        }
    }

    if (csv_file) csv_file.close();
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
    if (cuda_buffers_inited) { cudaFree(prev_dev_owned); cudaFree(curr_dev_owned); }
    if (host_prev_alloc && host_prev_ptr) free((void*)host_prev_ptr);

    auto wall_end = std::chrono::steady_clock::now();
    m.frames_decoded = frame_idx;
    m.wall_time_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    m.throughput_fps = m.wall_time_sec > 0.0 ? (double)m.frames_decoded / m.wall_time_sec : 0.0;
    return m;
}

int main(int argc, char** argv) {
    av_log_set_level(AV_LOG_ERROR);
    auto parsed = parse_args(argc, argv);
    if (!parsed) {
        std::cerr << "Usage: " << argv[0]
                  << " <input> [<input2> ...] [--threshold <val>] [--min-gap-ms <ms>]"
                     " [--downscale <n>] [--csv <path>] [--verbose]"
                     " [--input-list <file>] [--jobs <n>]\n";
        return 2;
    }
    Args args = *parsed;

    avformat_network_init();

    // Create the shared GPU work queue and start the GPU worker thread.
    GpuWorkQueue gpu_queue;
    std::thread gpu_worker([&gpu_queue]{ gpu_queue.run(); });

    int total_streams = (int)args.inputs.size();
    int max_parallel = (args.jobs > 0) ? std::min(args.jobs, total_streams) : total_streams;

    std::vector<StreamMetrics> results(total_streams);
    std::vector<std::thread> threads;
    threads.reserve(total_streams);

    // Launch decoder threads, respecting the --jobs concurrency limit.
    std::mutex jobs_mtx;
    std::condition_variable jobs_cv;
    int running = 0;

    for (int i = 0; i < total_streams; ++i) {
        {
            std::unique_lock<std::mutex> lk(jobs_mtx);
            jobs_cv.wait(lk, [&]{ return running < max_parallel; });
            ++running;
        }
        threads.emplace_back([&, i]{
            results[i] = process_stream(i, args.inputs[i], args, gpu_queue, total_streams);
            {
                std::lock_guard<std::mutex> lk(jobs_mtx);
                --running;
            }
            jobs_cv.notify_one();
        });
    }

    for (auto& t : threads) t.join();

    // Drain remaining GPU tasks and stop the GPU worker thread.
    gpu_queue.shutdown();
    gpu_worker.join();

    avformat_network_deinit();

    // Print per-stream metrics (always for batch; only when verbose for single stream).
    if (total_streams > 1 || args.verbose) {
        std::cout << "\n=== Batch Processing Summary ===\n";
        int64_t total_frames = 0;
        int total_cuts = 0;
        for (int i = 0; i < total_streams; ++i) {
            const auto& r = results[i];
            std::cout << "Stream " << i << ": " << r.path << "\n";
            if (r.exit_code != 0) {
                std::cout << "  ERROR: " << r.error_msg << " (code " << r.exit_code << ")\n";
            } else {
                std::cout << "  Frames decoded : " << r.frames_decoded << "\n";
                std::cout << "  Cuts detected  : " << r.cuts_detected  << "\n";
                std::cout << "  Wall time      : " << r.wall_time_sec  << " s\n";
                std::cout << "  Throughput     : " << r.throughput_fps << " fps\n";
                total_frames += r.frames_decoded;
                total_cuts   += r.cuts_detected;
            }
        }
        if (total_streams > 1) {
            std::cout << "\nTotal: " << total_streams << " streams, "
                      << total_frames << " frames, " << total_cuts << " cuts\n";
        }
    }

    // Return a non-zero exit code if any stream encountered an error.
    for (const auto& r : results) {
        if (r.exit_code != 0) return r.exit_code;
    }
    return 0;
}

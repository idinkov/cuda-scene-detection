// frame_normalizer.h
// SPDX-License-Identifier: MIT
//
// FrameNormalizer: extracts an 8-bit planar luma (Y) plane from a wide range of
// AVPixelFormats so that the MAD kernel always receives a consistent input.
//
// Supported format groups:
//   1. Direct 8-bit luma  – formats whose data[0] is already an 8-bit Y plane
//      (NV12, NV21, YUV420P, YUV422P, YUV444P, GRAY8, and their JPEG/alpha
//       variants).  get_luma() simply returns data[0] with no copying.
//
//   2. Swscale conversion – all other formats (10/12/16-bit YUV, P010, RGB24,
//      BGR24, RGBA, packed YUV, etc.) are converted to AV_PIX_FMT_YUV420P
//      via libswscale and the Y plane of the result is returned.  The SwsContext
//      and the temporary output frame are cached and reused across calls as long
//      as width/height/format stay the same.
//
// Note: AV_PIX_FMT_CUDA (device memory) is intentionally NOT handled here;
// the caller must manage that path separately before invoking get_luma().

#pragma once

#include <cstdint>
#include <string>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns true when data[0] of a frame in this format is an 8-bit Y plane.
static inline bool fn_has_direct_8bit_luma(AVPixelFormat fmt) {
    switch (fmt) {
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_NV21:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVJ422P:
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P:
        case AV_PIX_FMT_YUV440P:
        case AV_PIX_FMT_YUVJ440P:
        case AV_PIX_FMT_YUVA420P:
        case AV_PIX_FMT_YUVA422P:
        case AV_PIX_FMT_YUVA444P:
        case AV_PIX_FMT_GRAY8:
            return true;
        default:
            return false;
    }
}

/// Returns true when the format requires a swscale conversion to extract
/// 8-bit luma.
static inline bool fn_needs_swscale(AVPixelFormat fmt) {
    switch (fmt) {
        // 10-bit semi-planar (HDR)
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_P010BE:
        case AV_PIX_FMT_P016LE:
        case AV_PIX_FMT_P016BE:
        // 10-bit planar YUV
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV420P10BE:
        case AV_PIX_FMT_YUV422P10LE:
        case AV_PIX_FMT_YUV422P10BE:
        case AV_PIX_FMT_YUV444P10LE:
        case AV_PIX_FMT_YUV444P10BE:
        // 12-bit planar YUV
        case AV_PIX_FMT_YUV420P12LE:
        case AV_PIX_FMT_YUV420P12BE:
        case AV_PIX_FMT_YUV422P12LE:
        case AV_PIX_FMT_YUV422P12BE:
        case AV_PIX_FMT_YUV444P12LE:
        case AV_PIX_FMT_YUV444P12BE:
        // RGB / BGR packed
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_BGR24:
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_ARGB:
        case AV_PIX_FMT_ABGR:
        case AV_PIX_FMT_RGB48LE:
        case AV_PIX_FMT_RGB48BE:
        case AV_PIX_FMT_BGR48LE:
        case AV_PIX_FMT_BGR48BE:
        // Planar RGB
        case AV_PIX_FMT_GBRP:
        case AV_PIX_FMT_GBRAP:
        // Packed YUV
        case AV_PIX_FMT_YUYV422:
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YVYU422:
        // High-bit-depth gray
        case AV_PIX_FMT_GRAY10LE:
        case AV_PIX_FMT_GRAY12LE:
        case AV_PIX_FMT_GRAY16LE:
        case AV_PIX_FMT_GRAY16BE:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// FrameNormalizer
// ---------------------------------------------------------------------------

class FrameNormalizer {
public:
    FrameNormalizer() = default;

    ~FrameNormalizer() {
        if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
        if (sws_frame_) { av_frame_free(&sws_frame_); }
    }

    // Non-copyable (SwsContext and AVFrame are not safely copyable)
    FrameNormalizer(const FrameNormalizer&) = delete;
    FrameNormalizer& operator=(const FrameNormalizer&) = delete;

    /// Returns true if the pixel format can be handled by this normalizer.
    /// AV_PIX_FMT_CUDA is intentionally excluded – that path is managed by
    /// the caller.
    static bool is_supported(AVPixelFormat fmt) {
        return fn_has_direct_8bit_luma(fmt) || fn_needs_swscale(fmt);
    }

    /// Human-readable description of how a format is processed.
    static std::string format_description(AVPixelFormat fmt) {
        if (fn_has_direct_8bit_luma(fmt)) return "direct 8-bit luma";
        if (fn_needs_swscale(fmt))        return "swscale->YUV420P";
        return "unsupported";
    }

    /// Extracts an 8-bit luma plane from *frame*.
    ///
    /// On success returns true and sets:
    ///   *out_data      – pointer to the Y plane (host memory)
    ///   *out_linesize  – stride in bytes
    ///
    /// The returned pointer is valid until the next call to get_luma() or
    /// until this FrameNormalizer is destroyed.  Callers that need to retain
    /// the data across calls must copy it.
    ///
    /// Returns false when the pixel format is not supported.
    bool get_luma(const AVFrame* frame, const uint8_t** out_data, int* out_linesize) {
        if (!frame || !out_data || !out_linesize) return false;

        AVPixelFormat fmt = static_cast<AVPixelFormat>(frame->format);

        if (fn_has_direct_8bit_luma(fmt)) {
            *out_data      = frame->data[0];
            *out_linesize  = frame->linesize[0];
            return true;
        }

        if (fn_needs_swscale(fmt)) {
            return convert_via_swscale(frame, out_data, out_linesize);
        }

        return false;
    }

private:
    SwsContext* sws_ctx_         = nullptr;
    AVFrame*    sws_frame_       = nullptr;
    int         sws_src_w_       = 0;
    int         sws_src_h_       = 0;
    AVPixelFormat sws_src_fmt_   = AV_PIX_FMT_NONE;

    /// Lazily creates (or reuses) the SwsContext and output AVFrame.
    bool ensure_sws(int w, int h, AVPixelFormat src_fmt) {
        if (sws_ctx_ && sws_src_w_ == w && sws_src_h_ == h && sws_src_fmt_ == src_fmt)
            return true;

        // Dimensions or format changed – tear down the old context.
        if (sws_ctx_)   { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
        if (sws_frame_) { av_frame_free(&sws_frame_); sws_frame_ = nullptr; }

        sws_ctx_ = sws_getContext(w, h, src_fmt,
                                  w, h, AV_PIX_FMT_YUV420P,
                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx_) return false;

        sws_frame_ = av_frame_alloc();
        if (!sws_frame_) {
            sws_freeContext(sws_ctx_); sws_ctx_ = nullptr;
            return false;
        }
        sws_frame_->format = AV_PIX_FMT_YUV420P;
        sws_frame_->width  = w;
        sws_frame_->height = h;
        if (av_frame_get_buffer(sws_frame_, 0) < 0) {
            sws_freeContext(sws_ctx_);  sws_ctx_   = nullptr;
            av_frame_free(&sws_frame_); sws_frame_ = nullptr;
            return false;
        }

        sws_src_w_   = w;
        sws_src_h_   = h;
        sws_src_fmt_ = src_fmt;
        return true;
    }

    bool convert_via_swscale(const AVFrame* frame,
                             const uint8_t** out_data, int* out_linesize) {
        if (!ensure_sws(frame->width, frame->height,
                        static_cast<AVPixelFormat>(frame->format)))
            return false;

        int rows = sws_scale(sws_ctx_,
                             // Cast adds const at both pointer levels; no data is lost.
                             (const uint8_t * const *)frame->data,
                             frame->linesize,
                             0, frame->height,
                             sws_frame_->data, sws_frame_->linesize);
        if (rows <= 0) return false;

        *out_data     = sws_frame_->data[0];
        *out_linesize = sws_frame_->linesize[0];
        return true;
    }
};

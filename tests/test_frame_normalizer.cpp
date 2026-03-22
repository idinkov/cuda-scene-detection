// tests/test_frame_normalizer.cpp
// Unit tests for the FrameNormalizer component.
// Tests cover: is_supported(), direct 8-bit luma extraction, and the
// swscale conversion path (RGB24, P010LE).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>
}

#include "../frame_normalizer.h"

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, #cond); \
        g_failed++; return; \
    } } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabs((double)(a) - (double)(b)) > (eps)) { \
        fprintf(stderr, "  FAIL line %d: got %f vs %f (eps=%f)\n", \
                __LINE__, (double)(a), (double)(b), (double)(eps)); \
        g_failed++; return; \
    } } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Allocate an AVFrame with the given format and fill data[0] (luma / first
/// plane) with `luma_fill`.  Chroma planes are left at whatever value
/// av_frame_get_buffer sets them to (tests that use this helper only examine
/// the luma plane, so chroma content does not matter).
/// Returns nullptr on failure; caller must av_frame_free() it.
static AVFrame* make_frame(int w, int h, AVPixelFormat fmt, uint8_t luma_fill) {
    AVFrame* f = av_frame_alloc();
    if (!f) return nullptr;
    f->width  = w;
    f->height = h;
    f->format = static_cast<int>(fmt);
    if (av_frame_get_buffer(f, 0) < 0) { av_frame_free(&f); return nullptr; }
    if (av_frame_make_writable(f) < 0) { av_frame_free(&f); return nullptr; }

    // Fill the luma / first plane only (active pixels only, not stride padding)
    for (int y = 0; y < h; y++)
        memset(f->data[0] + y * f->linesize[0], luma_fill, (size_t)w);

    return f;
}

/// Allocate a P010LE frame and fill it with a specific 10-bit luma value.
/// In P010LE each sample is a 16-bit word with the 10-bit value in the high bits.
static AVFrame* make_p010_frame(int w, int h, uint16_t luma10) {
    AVFrame* f = av_frame_alloc();
    if (!f) return nullptr;
    f->width  = w;
    f->height = h;
    f->format = static_cast<int>(AV_PIX_FMT_P010LE);
    if (av_frame_get_buffer(f, 0) < 0) { av_frame_free(&f); return nullptr; }
    if (av_frame_make_writable(f) < 0) { av_frame_free(&f); return nullptr; }

    // Luma plane: each pixel is 16-bit LE; value is luma10 << 6
    uint16_t stored = static_cast<uint16_t>(luma10 << 6);
    int ls = f->linesize[0]; // bytes per row
    for (int y = 0; y < h; y++) {
        uint16_t* row = reinterpret_cast<uint16_t*>(f->data[0] + y * ls);
        for (int x = 0; x < w; x++) row[x] = stored;
    }
    // Chroma (UV interleaved, half height): fill with mid-point
    uint16_t chroma_mid = static_cast<uint16_t>(512 << 6);
    if (f->data[1]) {
        int cls = f->linesize[1];
        for (int y = 0; y < h / 2; y++) {
            uint16_t* row = reinterpret_cast<uint16_t*>(f->data[1] + y * cls);
            for (int x = 0; x < w; x++) row[x] = chroma_mid;
        }
    }
    return f;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

static void test_is_supported_known_formats() {
    // Direct 8-bit luma formats
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_NV12));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_NV21));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUV420P));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUVJ420P));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUV422P));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUV444P));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_GRAY8));
    // Swscale conversion formats
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_P010LE));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_P010BE));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_RGB24));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_BGR24));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_RGBA));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_BGRA));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUV420P10LE));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUV422P10LE));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUV444P10LE));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_YUYV422));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_UYVY422));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_GBRP));
    ASSERT_TRUE(FrameNormalizer::is_supported(AV_PIX_FMT_GRAY16LE));
    // AV_PIX_FMT_CUDA is intentionally excluded (device memory path)
    ASSERT_FALSE(FrameNormalizer::is_supported(AV_PIX_FMT_CUDA));
    // Unknown/none
    ASSERT_FALSE(FrameNormalizer::is_supported(AV_PIX_FMT_NONE));
    g_passed++;
}

static void test_format_description() {
    ASSERT_TRUE(FrameNormalizer::format_description(AV_PIX_FMT_NV12)    == "direct 8-bit luma");
    ASSERT_TRUE(FrameNormalizer::format_description(AV_PIX_FMT_YUV420P) == "direct 8-bit luma");
    ASSERT_TRUE(FrameNormalizer::format_description(AV_PIX_FMT_RGB24)   == "swscale->YUV420P");
    ASSERT_TRUE(FrameNormalizer::format_description(AV_PIX_FMT_P010LE)  == "swscale->YUV420P");
    ASSERT_TRUE(FrameNormalizer::format_description(AV_PIX_FMT_CUDA)    == "unsupported");
    g_passed++;
}

static void test_direct_luma_nv12() {
    int w = 64, h = 32;
    AVFrame* f = make_frame(w, h, AV_PIX_FMT_NV12, 120);
    ASSERT_TRUE(f);

    FrameNormalizer norm;
    const uint8_t* data = nullptr;
    int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &data, &ls));
    ASSERT_TRUE(data == f->data[0]);     // should be the same pointer
    ASSERT_TRUE(ls   == f->linesize[0]);
    ASSERT_TRUE(data[0] == 120);         // pixel value preserved

    av_frame_free(&f);
    g_passed++;
}

static void test_direct_luma_yuv420p() {
    int w = 80, h = 48;
    AVFrame* f = make_frame(w, h, AV_PIX_FMT_YUV420P, 200);
    ASSERT_TRUE(f);

    FrameNormalizer norm;
    const uint8_t* data = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &data, &ls));
    ASSERT_TRUE(data == f->data[0]);
    ASSERT_TRUE(data[0] == 200);

    av_frame_free(&f);
    g_passed++;
}

static void test_direct_luma_yuv422p() {
    int w = 64, h = 32;
    AVFrame* f = make_frame(w, h, AV_PIX_FMT_YUV422P, 77);
    ASSERT_TRUE(f);

    FrameNormalizer norm;
    const uint8_t* data = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &data, &ls));
    ASSERT_TRUE(data == f->data[0]);
    ASSERT_TRUE(data[0] == 77);

    av_frame_free(&f);
    g_passed++;
}

static void test_direct_luma_gray8() {
    int w = 32, h = 32;
    AVFrame* f = make_frame(w, h, AV_PIX_FMT_GRAY8, 42);
    ASSERT_TRUE(f);

    FrameNormalizer norm;
    const uint8_t* data = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &data, &ls));
    ASSERT_TRUE(data == f->data[0]);
    ASSERT_TRUE(data[0] == 42);

    av_frame_free(&f);
    g_passed++;
}

static void test_swscale_rgb24_gray() {
    // Mid-gray RGB (128,128,128) should produce a luma value close to 128.
    // BT.601: Y = 0.299*R + 0.587*G + 0.114*B → for R=G=B=128, Y≈128.
    int w = 32, h = 32;
    AVFrame* f = av_frame_alloc();
    ASSERT_TRUE(f);
    f->width = w; f->height = h;
    f->format = static_cast<int>(AV_PIX_FMT_RGB24);
    ASSERT_TRUE(av_frame_get_buffer(f, 0) >= 0);
    ASSERT_TRUE(av_frame_make_writable(f) >= 0);
    // Fill with (128, 128, 128)
    for (int y = 0; y < h; y++) {
        uint8_t* row = f->data[0] + y * f->linesize[0];
        for (int x = 0; x < w * 3; x++) row[x] = 128;
    }

    FrameNormalizer norm;
    const uint8_t* luma = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &luma, &ls));
    ASSERT_TRUE(luma != nullptr);
    ASSERT_TRUE(ls > 0);
    // Y for mid-gray should be roughly in [100, 160] – allow some swscale variance
    ASSERT_TRUE(luma[0] >= 100 && luma[0] <= 160);

    av_frame_free(&f);
    g_passed++;
}

static void test_swscale_rgba() {
    // Opaque white RGBA (255,255,255,255) → Y should be near 235 (BT.601 full range 255 or limited 235)
    int w = 32, h = 32;
    AVFrame* f = av_frame_alloc();
    ASSERT_TRUE(f);
    f->width = w; f->height = h;
    f->format = static_cast<int>(AV_PIX_FMT_RGBA);
    ASSERT_TRUE(av_frame_get_buffer(f, 0) >= 0);
    ASSERT_TRUE(av_frame_make_writable(f) >= 0);
    for (int y = 0; y < h; y++) {
        uint8_t* row = f->data[0] + y * f->linesize[0];
        for (int x = 0; x < w; x++) {
            row[x * 4 + 0] = 255; // R
            row[x * 4 + 1] = 255; // G
            row[x * 4 + 2] = 255; // B
            row[x * 4 + 3] = 255; // A
        }
    }

    FrameNormalizer norm;
    const uint8_t* luma = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &luma, &ls));
    ASSERT_TRUE(luma != nullptr);
    // Y for white: BT.601 limited range gives 235; full range gives 255; allow [200,255]
    ASSERT_TRUE(luma[0] >= 200);

    av_frame_free(&f);
    g_passed++;
}

static void test_swscale_p010le() {
    // P010LE mid-gray (luma10 = 512) → after conversion to 8-bit Y ≈ 128
    int w = 32, h = 32;
    AVFrame* f = make_p010_frame(w, h, 512);
    ASSERT_TRUE(f);

    FrameNormalizer norm;
    const uint8_t* luma = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &luma, &ls));
    ASSERT_TRUE(luma != nullptr);
    // Accept a broad tolerance due to limited→limited range mapping
    ASSERT_TRUE(luma[0] >= 100 && luma[0] <= 160);

    av_frame_free(&f);
    g_passed++;
}

static void test_swscale_yuyv422() {
    // YUYV422 packed: Y0 U Y1 V pattern
    int w = 32, h = 32;
    AVFrame* f = av_frame_alloc();
    ASSERT_TRUE(f);
    f->width = w; f->height = h;
    f->format = static_cast<int>(AV_PIX_FMT_YUYV422);
    ASSERT_TRUE(av_frame_get_buffer(f, 0) >= 0);
    ASSERT_TRUE(av_frame_make_writable(f) >= 0);
    // Fill: Y=100, U=128, V=128
    for (int y = 0; y < h; y++) {
        uint8_t* row = f->data[0] + y * f->linesize[0];
        for (int x = 0; x < w / 2; x++) {
            row[x * 4 + 0] = 100; // Y0
            row[x * 4 + 1] = 128; // U
            row[x * 4 + 2] = 100; // Y1
            row[x * 4 + 3] = 128; // V
        }
    }

    FrameNormalizer norm;
    const uint8_t* luma = nullptr; int ls = 0;
    ASSERT_TRUE(norm.get_luma(f, &luma, &ls));
    ASSERT_TRUE(luma != nullptr);
    // Y=100 should survive the packed→planar conversion approximately
    ASSERT_TRUE(luma[0] >= 80 && luma[0] <= 120);

    av_frame_free(&f);
    g_passed++;
}

static void test_unsupported_format_returns_false() {
    // AV_PIX_FMT_CUDA is managed by the caller; normalizer should reject it.
    FrameNormalizer norm;
    // We cannot create a real CUDA frame here, so just test is_supported().
    ASSERT_FALSE(FrameNormalizer::is_supported(AV_PIX_FMT_CUDA));
    ASSERT_FALSE(FrameNormalizer::is_supported(AV_PIX_FMT_NONE));
    g_passed++;
}

static void test_null_frame_returns_false() {
    FrameNormalizer norm;
    const uint8_t* data = nullptr; int ls = 0;
    ASSERT_FALSE(norm.get_luma(nullptr, &data, &ls));
    g_passed++;
}

static void test_sws_context_reuse() {
    // Calling get_luma() multiple times with the same format should reuse
    // the SwsContext (no crash or incorrect results).
    int w = 16, h = 16;
    FrameNormalizer norm;
    for (int i = 0; i < 3; i++) {
        AVFrame* f = av_frame_alloc();
        ASSERT_TRUE(f);
        f->width = w; f->height = h;
        f->format = static_cast<int>(AV_PIX_FMT_RGB24);
        ASSERT_TRUE(av_frame_get_buffer(f, 0) >= 0);
        ASSERT_TRUE(av_frame_make_writable(f) >= 0);
        uint8_t fill = static_cast<uint8_t>(50 * (i + 1));
        for (int y = 0; y < h; y++) {
            uint8_t* row = f->data[0] + y * f->linesize[0];
            for (int x = 0; x < w * 3; x++) row[x] = fill;
        }
        const uint8_t* luma = nullptr; int ls = 0;
        ASSERT_TRUE(norm.get_luma(f, &luma, &ls));
        ASSERT_TRUE(luma != nullptr && ls > 0);
        av_frame_free(&f);
    }
    g_passed++;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    fprintf(stderr, "=== FrameNormalizer Unit Tests ===\n\n");

    fprintf(stderr, "  test_is_supported_known_formats...\n");
    test_is_supported_known_formats();

    fprintf(stderr, "  test_format_description...\n");
    test_format_description();

    fprintf(stderr, "  test_direct_luma_nv12...\n");
    test_direct_luma_nv12();

    fprintf(stderr, "  test_direct_luma_yuv420p...\n");
    test_direct_luma_yuv420p();

    fprintf(stderr, "  test_direct_luma_yuv422p...\n");
    test_direct_luma_yuv422p();

    fprintf(stderr, "  test_direct_luma_gray8...\n");
    test_direct_luma_gray8();

    fprintf(stderr, "  test_swscale_rgb24_gray...\n");
    test_swscale_rgb24_gray();

    fprintf(stderr, "  test_swscale_rgba...\n");
    test_swscale_rgba();

    fprintf(stderr, "  test_swscale_p010le...\n");
    test_swscale_p010le();

    fprintf(stderr, "  test_swscale_yuyv422...\n");
    test_swscale_yuyv422();

    fprintf(stderr, "  test_unsupported_format_returns_false...\n");
    test_unsupported_format_returns_false();

    fprintf(stderr, "  test_null_frame_returns_false...\n");
    test_null_frame_returns_false();

    fprintf(stderr, "  test_sws_context_reuse...\n");
    test_sws_context_reuse();

    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

// cpu_mad.h
// SPDX-License-Identifier: MIT
// Optimized CPU helpers for luma sampling and Mean Absolute Difference (MAD).
// Works as a standalone header included by both the main application (.cpp)
// and the CUDA test file (.cu, host code only).

#pragma once

#include <cstdint>
#include <cstdlib>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// sample_luma
// ---------------------------------------------------------------------------
// Extracts stride-sampled luma pixels from a source plane into a compact flat
// buffer. Only dstW * dstH bytes are written, avoiding the need to store the
// full-resolution luma plane between frames.
//
//   src      - pointer to the luma plane (host memory, row-major)
//   srcPitch - row stride of src in bytes (may be > width due to alignment)
//   width    - logical frame width in pixels
//   height   - logical frame height in pixels
//   ds       - downscale stride (1 = full res, 2 = every 2nd pixel, …)
//   dst      - output compact buffer of size dstW * dstH bytes
//   dstW     - ceil(width  / ds)
//   dstH     - ceil(height / ds)
static inline void sample_luma(
    const uint8_t* src, int srcPitch,
    int width, int height, int ds,
    uint8_t* dst, int dstW, int dstH)
{
    for (int yds = 0; yds < dstH; ++yds) {
        int y = yds * ds;
        if (y >= height) y = height - 1;
        const uint8_t* row = src + (size_t)y * srcPitch;
        uint8_t* drow = dst + (size_t)yds * dstW;
        for (int xds = 0; xds < dstW; ++xds) {
            int x = xds * ds;
            if (x >= width) x = width - 1;
            drow[xds] = row[x];
        }
    }
}

// ---------------------------------------------------------------------------
// compute_mad_scalar
// ---------------------------------------------------------------------------
// Reference scalar MAD over two flat pixel arrays of length n.
static inline float compute_mad_scalar(const uint8_t* a, const uint8_t* b, int n)
{
    unsigned long long total = 0;
    for (int i = 0; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        total += (unsigned long long)(diff < 0 ? -diff : diff);
    }
    return n > 0 ? (float)total / (float)n : 0.0f;
}

#ifdef __AVX2__
// ---------------------------------------------------------------------------
// compute_mad_avx2
// ---------------------------------------------------------------------------
// AVX2-accelerated MAD using _mm256_sad_epu8 (processes 32 bytes per cycle).
// Falls back to scalar for any remaining tail bytes.
static inline float compute_mad_avx2(const uint8_t* a, const uint8_t* b, int n)
{
    unsigned long long total = 0;
    int i = 0;

    // Accumulate partial SAD sums in an AVX2 register to reduce store traffic.
    __m256i acc = _mm256_setzero_si256();
    // _mm256_sad_epu8 splits the 256-bit vector into four 64-bit chunks of 8 bytes
    // each. Each chunk yields a partial SAD sum of up to 8*255=2040, stored in a
    // 64-bit lane. Accumulating into a 64-bit acc, overflow requires > 2^64/2040
    // ≈ 9e15 iterations — impossible in practice.
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i sad = _mm256_sad_epu8(va, vb);
        acc = _mm256_add_epi64(acc, sad);
    }

    // Horizontal reduce the four 64-bit lanes.
    total  = (unsigned long long)_mm256_extract_epi64(acc, 0);
    total += (unsigned long long)_mm256_extract_epi64(acc, 1);
    total += (unsigned long long)_mm256_extract_epi64(acc, 2);
    total += (unsigned long long)_mm256_extract_epi64(acc, 3);

    // Scalar tail (< 32 remaining bytes).
    for (; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        total += (unsigned long long)(diff < 0 ? -diff : diff);
    }

    return n > 0 ? (float)total / (float)n : 0.0f;
}
#endif // __AVX2__

// ---------------------------------------------------------------------------
// compute_mad_cpu
// ---------------------------------------------------------------------------
// Dispatcher: uses AVX2 when compiled with __AVX2__, otherwise scalar.
static inline float compute_mad_cpu(const uint8_t* a, const uint8_t* b, int n)
{
    if (!a || !b || n <= 0) return 0.0f;
#ifdef __AVX2__
    return compute_mad_avx2(a, b, n);
#else
    return compute_mad_scalar(a, b, n);
#endif
}

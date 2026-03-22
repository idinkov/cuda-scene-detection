// tests/test_adaptive.cu
// SPDX-License-Identifier: MIT
// Unit tests for AdaptiveStats (adaptive thresholding).
// No CUDA device is required; tests run on the host.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../adaptive_stats.h"

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_TRUE(cond) do { if (!(cond)) { \
    fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, #cond); \
    g_failed++; return; } } while(0)

#define ASSERT_NEAR(a, b, eps) do { if (fabs((double)(a) - (double)(b)) > (eps)) { \
    fprintf(stderr, "  FAIL line %d: got %f vs expected %f (eps %f)\n", \
            __LINE__, (double)(a), (double)(b), (double)(eps)); \
    g_failed++; return; } } while(0)

// Before any samples are provided the adaptive threshold must fall back to the static value.
static void test_empty_returns_static() {
    AdaptiveStats s;
    s.enabled      = true;
    s.k            = 3.0;
    s.warmupFrames = 30;
    ASSERT_NEAR(s.threshold(18.0), 18.0, 1e-6);
    g_passed++;
}

// During warmup the static threshold is returned regardless of accumulated samples.
static void test_warmup_uses_static_threshold() {
    AdaptiveStats s;
    s.enabled      = true;
    s.k            = 3.0;
    s.warmupFrames = 30;
    for (int i = 0; i < 15; ++i) s.update(20.0);
    ASSERT_TRUE(!s.warmedUp());
    ASSERT_NEAR(s.threshold(18.0), 18.0, 1e-6);
    g_passed++;
}

// Once warmup is complete warmedUp() returns true.
static void test_warmed_up_flag() {
    AdaptiveStats s;
    s.enabled      = true;
    s.warmupFrames = 10;
    for (int i = 0; i < 9; ++i) s.update(5.0);
    ASSERT_TRUE(!s.warmedUp());
    s.update(5.0);
    ASSERT_TRUE(s.warmedUp());
    g_passed++;
}

// Mean and variance are computed correctly for a small dataset.
static void test_statistics_correctness() {
    AdaptiveStats s;
    s.warmupFrames = 1;
    s.update(10.0);
    s.update(20.0);
    s.update(30.0);
    // mean = 20, sample variance = ((10-20)^2+(20-20)^2+(30-20)^2)/(3-1) = 200/2 = 100
    ASSERT_NEAR(s.mean_,    20.0, 1e-6);
    ASSERT_NEAR(s.variance(), 100.0, 1e-6);
    ASSERT_NEAR(s.stdev(),   10.0, 1e-6);
    g_passed++;
}

// Adaptive threshold matches the formula mean + k * stdev.
static void test_adaptive_threshold_formula() {
    AdaptiveStats s;
    s.enabled      = true;
    s.k            = 2.0;
    s.warmupFrames = 1;
    s.update(10.0);
    s.update(20.0);
    s.update(30.0);
    // mean=20, stdev=10, threshold = 20 + 2*10 = 40
    ASSERT_NEAR(s.threshold(0.0), 40.0, 1e-6);
    g_passed++;
}

// When adaptive is disabled, static threshold is always returned.
static void test_disabled_uses_static_threshold() {
    AdaptiveStats s;
    s.enabled      = false;
    s.warmupFrames = 1;
    for (int i = 0; i < 100; ++i) s.update(50.0);
    ASSERT_NEAR(s.threshold(18.0), 18.0, 1e-6);
    g_passed++;
}

// With uniform inputs, stdev is zero and threshold equals the mean.
static void test_uniform_values() {
    AdaptiveStats s;
    s.enabled      = true;
    s.k            = 3.0;
    s.warmupFrames = 5;
    for (int i = 0; i < 10; ++i) s.update(15.0);
    ASSERT_NEAR(s.mean_,  15.0, 1e-6);
    ASSERT_NEAR(s.stdev(), 0.0, 1e-6);
    // threshold = max(15 + 3*0, 1) = 15
    ASSERT_NEAR(s.threshold(0.0), 15.0, 1e-6);
    g_passed++;
}

// Adaptive threshold is clamped to at least 1.0 to stay meaningful.
static void test_minimum_threshold_clamp() {
    AdaptiveStats s;
    s.enabled      = true;
    s.k            = 0.0;
    s.warmupFrames = 1;
    s.update(0.0);
    s.update(0.0);
    // mean=0, stdev=0, adaptive=0 -> clamped to 1.0
    ASSERT_NEAR(s.threshold(99.0), 1.0, 1e-6);
    g_passed++;
}

// Single-sample variance is 0 (no division by zero).
static void test_single_sample_variance() {
    AdaptiveStats s;
    s.update(42.0);
    ASSERT_NEAR(s.variance(), 0.0, 1e-6);
    ASSERT_NEAR(s.stdev(),    0.0, 1e-6);
    g_passed++;
}

// count_ increments correctly.
static void test_count() {
    AdaptiveStats s;
    ASSERT_TRUE(s.count_ == 0);
    for (int i = 0; i < 7; ++i) s.update(1.0);
    ASSERT_TRUE(s.count_ == 7);
    g_passed++;
}

int main() {
    fprintf(stderr, "=== AdaptiveStats Unit Tests ===\n\n");

    fprintf(stderr, "  test_empty_returns_static...\n");         test_empty_returns_static();
    fprintf(stderr, "  test_warmup_uses_static_threshold...\n"); test_warmup_uses_static_threshold();
    fprintf(stderr, "  test_warmed_up_flag...\n");               test_warmed_up_flag();
    fprintf(stderr, "  test_statistics_correctness...\n");       test_statistics_correctness();
    fprintf(stderr, "  test_adaptive_threshold_formula...\n");   test_adaptive_threshold_formula();
    fprintf(stderr, "  test_disabled_uses_static_threshold...\n"); test_disabled_uses_static_threshold();
    fprintf(stderr, "  test_uniform_values...\n");               test_uniform_values();
    fprintf(stderr, "  test_minimum_threshold_clamp...\n");      test_minimum_threshold_clamp();
    fprintf(stderr, "  test_single_sample_variance...\n");       test_single_sample_variance();
    fprintf(stderr, "  test_count...\n");                        test_count();

    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

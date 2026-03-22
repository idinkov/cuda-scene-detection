// tests/test_gradual.cpp
// SPDX-License-Identifier: MIT
// Unit tests for the gradual transition detection algorithm in gradual_detection.h.
// Pure C++ — no CUDA device required.

#include <cstdio>
#include <cmath>
#include "gradual_detection.h"

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_TRUE(cond) do { if (!(cond)) { \
    fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, #cond); \
    g_failed++; return; } } while(0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_NEAR(a, b, eps) do { if (fabs((double)(a) - (double)(b)) > (eps)) { \
    fprintf(stderr, "  FAIL line %d: got %f vs %f\n", __LINE__, (double)(a), (double)(b)); \
    g_failed++; return; } } while(0)

// No detection on quiet frames (MAD below low threshold)
static void test_no_detection_on_quiet_frames() {
    GradualDetector det(3.0, 15, 5);
    for (int i = 0; i < 20; ++i) {
        bool detected = det.update(1.0f, i * 0.033, 400, false);
        ASSERT_FALSE(detected);
    }
    g_passed++;
}

// Detection triggered after exactly minConsecutive frames above low threshold
static void test_detection_after_min_consecutive() {
    GradualDetector det(3.0, 15, 5);
    for (int i = 0; i < 4; ++i) {
        bool detected = det.update(5.0f, i * 0.033, 0, false);
        ASSERT_FALSE(detected);
    }
    // Fifth frame triggers detection
    bool detected = det.update(5.0f, 4 * 0.033, 0, false);
    ASSERT_TRUE(detected);
    g_passed++;
}

// A quiet frame resets the counter, requiring a fresh run of minConsecutive
static void test_reset_on_quiet_frame() {
    GradualDetector det(3.0, 15, 5);
    for (int i = 0; i < 4; ++i) det.update(5.0f, i * 0.033, 0, false);
    // Quiet frame resets counter
    det.update(1.0f, 4 * 0.033, 0, false);
    // Need another full run of 5 frames before detection
    for (int i = 0; i < 4; ++i) {
        bool detected = det.update(5.0f, (5 + i) * 0.033, 0, false);
        ASSERT_FALSE(detected);
    }
    bool detected = det.update(5.0f, 9 * 0.033, 0, false);
    ASSERT_TRUE(detected);
    g_passed++;
}

// A hard cut resets the gradual accumulator
static void test_hard_cut_resets_accumulator() {
    GradualDetector det(3.0, 15, 5);
    for (int i = 0; i < 4; ++i) det.update(5.0f, i * 0.033, 0, false);
    // Hard cut on this frame resets state and suppresses gradual report
    bool detected = det.update(5.0f, 4 * 0.033, 0, /*hardCutDetected=*/true);
    ASSERT_FALSE(detected);
    // After reset, need another full run of minConsecutive
    for (int i = 0; i < 4; ++i) {
        bool d = det.update(5.0f, (5 + i) * 0.033, 0, false);
        ASSERT_FALSE(d);
    }
    bool d2 = det.update(5.0f, 9 * 0.033, 0, false);
    ASSERT_TRUE(d2);
    g_passed++;
}

// minGapMs prevents back-to-back detections
static void test_min_gap_prevents_duplicates() {
    GradualDetector det(3.0, 15, 3);
    // Trigger first detection at frame 2 (t ≈ 0.066s)
    for (int i = 0; i < 3; ++i) det.update(5.0f, i * 0.033, 400, false);
    // Immediately after, more above-threshold frames — still within the 400ms gap
    for (int i = 3; i < 6; ++i) {
        bool detected = det.update(5.0f, i * 0.033, 400, false);
        ASSERT_FALSE(detected); // gap from first detection (0.066s) to frame 5 (0.165s) is 99ms < 400ms
    }
    // Well after the gap, a new detection should fire
    bool detected = det.update(5.0f, 2.0, 400, false); // 2.0s - 0.066s >> 400ms
    ASSERT_TRUE(detected);
    g_passed++;
}

// windowAvg() returns the correct rolling average
static void test_window_avg() {
    GradualDetector det(3.0, 4, 5); // window capped at 4 frames
    det.update(10.0f, 0.0, 0, false);
    det.update(20.0f, 1.0, 0, false);
    det.update(30.0f, 2.0, 0, false);
    ASSERT_NEAR(det.windowAvg(), 20.0, 0.01); // avg(10,20,30)=20

    det.update(40.0f, 3.0, 0, false);
    ASSERT_NEAR(det.windowAvg(), 25.0, 0.01); // avg(10,20,30,40)=25

    // Window is full (size 4): adding 50 evicts 10 → avg(20,30,40,50)=35
    det.update(50.0f, 4.0, 0, false);
    ASSERT_NEAR(det.windowAvg(), 35.0, 0.01);
    g_passed++;
}

// Frames reported as hard cuts never trigger gradual detection
static void test_hard_cut_frames_not_gradual() {
    GradualDetector det(3.0, 15, 5);
    for (int i = 0; i < 10; ++i) {
        bool detected = det.update(50.0f, i * 0.033, 0, /*hardCutDetected=*/true);
        ASSERT_FALSE(detected);
    }
    g_passed++;
}

int main() {
    fprintf(stderr, "=== Gradual Transition Detection Unit Tests ===\n\n");
    fprintf(stderr, "  test_no_detection_on_quiet_frames...\n");
    test_no_detection_on_quiet_frames();
    fprintf(stderr, "  test_detection_after_min_consecutive...\n");
    test_detection_after_min_consecutive();
    fprintf(stderr, "  test_reset_on_quiet_frame...\n");
    test_reset_on_quiet_frame();
    fprintf(stderr, "  test_hard_cut_resets_accumulator...\n");
    test_hard_cut_resets_accumulator();
    fprintf(stderr, "  test_min_gap_prevents_duplicates...\n");
    test_min_gap_prevents_duplicates();
    fprintf(stderr, "  test_window_avg...\n");
    test_window_avg();
    fprintf(stderr, "  test_hard_cut_frames_not_gradual...\n");
    test_hard_cut_frames_not_gradual();
    fprintf(stderr, "\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

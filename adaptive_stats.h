// adaptive_stats.h
// SPDX-License-Identifier: MIT
// Online running statistics (mean, variance) for MAD values using Welford's
// numerically-stable algorithm. Used to implement adaptive scene-cut thresholds.

#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

// AdaptiveStats tracks a running mean and standard deviation of MAD values
// and computes a per-frame adaptive threshold: mean + k * stdev.
//
// During the initial warmup period the static threshold is returned unchanged
// so the detector behaves identically to the non-adaptive mode until it has
// collected enough samples to produce a reliable estimate.
struct AdaptiveStats {
    // Running state (Welford's online algorithm)
    double mean_   = 0.0;
    double M2_     = 0.0; // accumulated squared deviations from mean
    int64_t count_ = 0;

    // Configuration
    bool   enabled      = false; // adaptive mode enabled?
    double k            = 3.0;   // threshold = mean + k * stdev
    int    warmupFrames = 30;    // frames collected before adaptive kicks in

    // Feed one new MAD sample into the running statistics.
    void update(double mad) {
        ++count_;
        double delta  = mad - mean_;
        mean_        += delta / static_cast<double>(count_);
        double delta2 = mad - mean_;
        M2_          += delta * delta2;
    }

    // Sample variance (Bessel-corrected).
    double variance() const {
        if (count_ < 2) return 0.0;
        return M2_ / static_cast<double>(count_ - 1);
    }

    double stdev() const { return std::sqrt(variance()); }

    // Returns the effective detection threshold for the current frame.
    // Falls back to staticThreshold when adaptive is disabled or during warmup.
    double threshold(double staticThreshold) const {
        if (!enabled || count_ < static_cast<int64_t>(warmupFrames))
            return staticThreshold;
        double adaptive = mean_ + k * stdev();
        // Clamp to a small positive value so the detector never becomes degenerate.
        return std::max(adaptive, 1.0);
    }

    // True once enough samples have been collected to use the adaptive threshold.
    bool warmedUp() const { return count_ >= static_cast<int64_t>(warmupFrames); }
};

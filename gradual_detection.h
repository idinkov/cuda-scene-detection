// gradual_detection.h
// SPDX-License-Identifier: MIT
// Sliding-window twin-comparison algorithm for detecting gradual scene transitions
// (dissolves and fades) by tracking consecutive sub-threshold MAD values over a
// configurable frame window. Header-only for easy inclusion and unit testing.

#pragma once
#include <deque>

// GradualDetector implements a twin-comparison sliding-window algorithm:
//
//   Tb (lowThreshold)  < MAD < Tn (hard-cut threshold) => "candidate" gradual frame
//
// When minConsecutive or more consecutive candidate frames accumulate a gradual
// transition is declared.  Hard cuts (reported separately by the caller) reset the
// internal accumulator so their boundary does not bleed into gradual detection.
//
// Typical values:
//   lowThreshold     3.0   (below hard-cut default of 18.0)
//   windowSize       15    frames kept for windowAvg()
//   minConsecutive   5     frames needed to declare a transition
struct GradualDetector {
    double lowThreshold;    // Tb: frames with MAD > lowThreshold (but below hard-cut
                            //     threshold) contribute to gradual transition evidence
    int windowSize;         // Maximum number of recent MAD values kept in the sliding window
    int minConsecutive;     // Minimum consecutive candidate frames required to declare
                            // a gradual transition

    std::deque<float> window;     // Sliding window of recent per-frame MAD values
    int consecutiveAboveLow = 0;  // Count of consecutive candidate frames
    double lastEventTime = -1e9;  // Timestamp (seconds) of last reported transition

    GradualDetector(double tb = 3.0, int wsize = 15, int minc = 5)
        : lowThreshold(tb), windowSize(wsize), minConsecutive(minc) {}

    // Call once per frame. Returns true when a gradual transition is detected.
    //   mad             : per-frame MAD value
    //   timestamp       : frame timestamp in seconds
    //   minGapMs        : minimum milliseconds between reported gradual transitions
    //   hardCutDetected : true if this frame was already classified as a hard cut
    bool update(float mad, double timestamp, int minGapMs, bool hardCutDetected) {
        // Maintain sliding window regardless of frame type
        window.push_back(mad);
        if ((int)window.size() > windowSize) window.pop_front();

        if (hardCutDetected) {
            // Hard cuts are already reported by the caller; reset the accumulator so the
            // hard-cut boundary does not bleed into the gradual transition detector.
            consecutiveAboveLow = 0;
            lastEventTime = timestamp;
            return false;
        }

        // Twin-comparison: count consecutive candidate frames (Tb < MAD < Tn).
        // The caller is responsible for ensuring MAD < Tn before passing hardCutDetected=false.
        if (mad > lowThreshold) {
            consecutiveAboveLow++;
        } else {
            consecutiveAboveLow = 0;
        }

        // Declare a gradual transition when enough evidence has accumulated and the
        // minimum time gap since the last event has elapsed.
        bool gapOk = (timestamp - lastEventTime) * 1000.0 > (double)minGapMs;
        if (consecutiveAboveLow >= minConsecutive && gapOk) {
            lastEventTime = timestamp;
            consecutiveAboveLow = 0; // reset to prevent duplicate detections in same transition
            return true;
        }
        return false;
    }

    // Average MAD of all frames currently in the sliding window.
    double windowAvg() const {
        if (window.empty()) return 0.0;
        double sum = 0.0;
        for (float v : window) sum += v;
        return sum / (double)window.size();
    }
};

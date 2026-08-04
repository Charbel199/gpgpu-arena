#pragma once

#include <vector>

// Pure measurement policy: no CUDA, no I/O, no global state.
// Everything here is unit-testable without a GPU. Keep it that way.

namespace arena::measure {

// True median. Even-length input averages the two middle samples.
// Empty input returns 0.
float median(std::vector<float> v);

struct DriftCheck {
    bool  converged  = false;
    float rel_change = 0.0f;
};

// Compares the median of the last `window` samples against the median of the
// `window` samples immediately before them.
//
// Detects monotonic drift (a GPU ramping its clock produces steadily falling
// times whose local variance is small), which a sliding standard-deviation
// test would incorrectly report as stable.
//
// Returns converged == false when:
//   - window <= 0
//   - history.size() < 2 * window
//   - the earlier median is non-positive
//   - rel_change >= threshold  (strict: exactly at threshold is not converged)
DriftCheck check_drift(const std::vector<float>& history, int window, float threshold);

// quantity per second. Returns 0 when ms is zero, negative, or NaN, rather
// than propagating inf into throughput figures.
double rate_per_sec(double quantity, float ms);

}

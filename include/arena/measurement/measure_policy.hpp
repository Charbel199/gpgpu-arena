#pragma once

#include <vector>

// Pure measurement policy: no CUDA, no I/O, no global state.
// Everything here is unit-testable without a GPU. Keep it that way.

namespace arena::measure {

// True median. Even-length input averages the two middle samples.
// Empty input returns 0.
float median(std::vector<float> v);

// Mean of the middle half (25th to 75th percentile). Short kernels' event
// timings cluster near multiples of about 2 us on some GPUs, so the median
// snaps from one cluster to the next while this moves smoothly, and outliers
// still cannot drag it the way they drag a plain mean.
float trimmed_mean(std::vector<float> v);

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

// Summary of the NVML samples taken across one result's timed runs.
struct ClockPowerStats {
    int      samples = 0;          // clock samples
    double   sm_mean_mhz = 0.0;
    unsigned sm_min_mhz = 0;
    unsigned sm_max_mhz = 0;
    double   power_mean_w = 0.0;   // 0 when no power sample was taken
    double   power_max_w = 0.0;
};

ClockPowerStats summarize_clock_power(const std::vector<unsigned>& sm_mhz,
                                      const std::vector<double>& watts);

// The clock --lock-clocks auto asks for: the lower of the board's default
// application clock (its rated boost, which it is meant to sustain) and 85%
// of the max SM clock, snapped down to a clock the board supports. The 85%
// figure is the fallback when the default clock is unknown and a cap when it
// is implausibly high. default_app_mhz may be 0 and supported may be empty.
// Returns 0 if max_mhz is 0.
unsigned pick_auto_lock_clock(unsigned max_mhz, unsigned default_app_mhz,
                              const std::vector<unsigned>& supported);

}

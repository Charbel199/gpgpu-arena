#pragma once

#include "arena/measurement/measure_policy.hpp"
#include "arena/runner_config.hpp"
#include "arena/measurement/power.hpp"

#include <cuda.h>
#include <functional>
#include <vector>

namespace arena::measure {

using LaunchFn = std::function<void()>;

struct WarmupResult {
    // Why warmup stopped. Not converging is not automatically a bad kernel:
    // TooFewSamples means the drift check never got to run at all, which is a
    // budget problem rather than an unsteady GPU, and the two used to be
    // reported identically.
    enum class Stop { Converged, Drifting, TooFewSamples, IterationCap, Fixed };

    int   iterations      = 0;
    bool  converged       = false;
    float first_launch_ms = 0.0f;   // iteration 0, measured separately
    float last_drift      = 0.0f;   // relative change at the final check
    Stop  stop            = Stop::TooFewSamples;
};

const char* warmup_stop_name(WarmupResult::Stop s);

// Runs launch_fn until timings stabilise (Auto) or a fixed count (Fixed).
// reset_fn, if provided, runs before each iteration and is never timed.
//
// Auto stops when check_drift reports convergence, bounded by warmup_min,
// warmup_max, and warmup_max_ms. Hitting a bound without converging is not an
// error: converged is false and the caller records that.
WarmupResult warmup(const LaunchFn& launch_fn, const LaunchFn& reset_fn,
                    const RunConfig& config);

struct TimingResult {
    float median_ms = 0.0f;
    std::vector<float> all_times_ms;
};

// Times `runs` isolated iterations. reset_fn runs before each and is excluded
// from the measured interval, so each run starts from a drained queue.
TimingResult time_operation(const LaunchFn& launch_fn, const LaunchFn& reset_fn,
                            int runs);

struct EnergyResult {
    bool   available  = false;
    double mj_per_op  = 0.0;
    double avg_watts  = 0.0;
    int    iterations = 0;
    float  total_ms   = 0.0f;   // actual interval, including the drain
};

// Sustained-load energy measurement.
//
// Deliberately does NOT reset between iterations: it launches back-to-back so
// launches pipeline, which is the only regime where energy is meaningful. The
// consequence is that the output buffer accumulates garbage, so this pass must
// run last and its output must never be verified.
//
// Energy and elapsed time are measured across the same interval, including the
// final queue drain, so mj_per_op and avg_watts stay correct despite the
// window overshooting slightly.
//
// Two caveats on interpreting mj_per_op:
//
//  1. NVML reports whole-board energy, so it includes idle draw for the
//     duration of each operation, not just the kernel's marginal cost.
//
//  2. The loop is only GPU-bound if the host can issue launches faster than
//     the GPU retires them. For short kernels it often cannot -- the launch
//     path costs tens of microseconds, so the GPU idles between launches and
//     that idle energy is charged to the operation. Compare
//     (iterations * gpu_ms) against total_ms to see how saturated the device
//     actually was; Runner logs this as a saturation percentage.
EnergyResult measure_energy(const LaunchFn& launch_fn, PowerMonitor& power,
                            float window_ms);

}

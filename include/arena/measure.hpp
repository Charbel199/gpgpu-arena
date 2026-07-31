#pragma once

#include "arena/measure_policy.hpp"
#include "arena/runner_config.hpp"

#include <cuda.h>
#include <functional>
#include <vector>

namespace arena::measure {

using LaunchFn = std::function<void()>;

struct WarmupResult {
    int   iterations      = 0;
    bool  converged       = false;
    float first_launch_ms = 0.0f;   // iteration 0, measured separately
};

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

}

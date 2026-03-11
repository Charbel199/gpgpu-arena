#pragma once

#include <cuda.h>
#include <functional>
#include <vector>

namespace arena {

using KernelLaunchFn = std::function<void()>;

struct BenchmarkResult {
    float median_ms = 0.0f;
    std::vector<float> all_times_ms;
};

class Benchmark {
public:
    // Time a kernel over multiple runs.
    // reset_fn is called before each run to reinitialize GPU state.
    BenchmarkResult run(KernelLaunchFn launch_fn, int number_of_runs,
                        KernelLaunchFn reset_fn = nullptr);
};

}

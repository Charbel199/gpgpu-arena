#include "arena/benchmark.hpp"
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <string>

namespace arena {

BenchmarkResult Benchmark::run(KernelLaunchFn launch_fn, int number_of_runs,
                                KernelLaunchFn reset_fn) {
    BenchmarkResult result;
    result.all_times_ms.reserve(number_of_runs);

    for (int i = 0; i < number_of_runs; i++) {
        if (reset_fn) reset_fn();

        nvtxRangePushA(("Run " + std::to_string(i)).c_str());

        CUevent start, stop;
        cuEventCreate(&start, CU_EVENT_DEFAULT);
        cuEventCreate(&stop, CU_EVENT_DEFAULT);

        cuEventRecord(start, nullptr);
        nvtxRangePushA("Kernel");
        launch_fn();
        cuEventRecord(stop, nullptr);
        cuEventSynchronize(stop);
        nvtxRangePop();

        float elapsed_ms = 0.0f;
        cuEventElapsedTime(&elapsed_ms, start, stop);

        cuEventDestroy(start);
        cuEventDestroy(stop);

        result.all_times_ms.push_back(elapsed_ms);

        nvtxRangePop();
    }

    auto sorted = result.all_times_ms;
    std::sort(sorted.begin(), sorted.end());
    result.median_ms = sorted[sorted.size() / 2];

    return result;
}

}

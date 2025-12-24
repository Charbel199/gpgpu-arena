#include "arena/profiler.hpp"
#include "arena/utils.hpp"
#include <iostream>

namespace arena {

namespace {
    // CUPTI activity callback
    void CUPTIAPI activity_callback(
        CUpti_CallbackDomain domain,
        CUpti_CallbackId callback_id,
        const void* callback_info
    ) {
        // This will be expanded to capture kernel metrics
        // For now, just a placeholder
    }
}

Profiler::Profiler() {
    init_cupti();
}

Profiler::~Profiler() {
    cleanup_cupti();
}

void Profiler::init_cupti() {
    if (initialized_) return;

    // Subscribe to CUPTI callbacks
    CUptiResult result = cuptiSubscribe(
        &subscriber_,
        reinterpret_cast<CUpti_CallbackFunc>(activity_callback),
        nullptr
    );
    
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Warning: CUPTI initialization failed. "
                  << "Profiling will be limited.\n";
        return;
    }

    // Enable activity tracking for kernels
    check_cupti(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL),
        "cuptiActivityEnable (kernel)"
    );

    // Enable activity tracking for memory operations
    check_cupti(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),
        "cuptiActivityEnable (memcpy)"
    );

    initialized_ = true;
}

void Profiler::cleanup_cupti() {
    if (!initialized_) return;

    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiUnsubscribe(subscriber_);
    
    initialized_ = false;
}

Profiler::KernelMetrics Profiler::profile(KernelLaunchFn launch_fn) {
    KernelMetrics metrics;

    // For Phase 1, we just do basic timing
    // CUPTI metric collection will be added in Phase 2

    // Create CUDA events for timing
    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop, CU_EVENT_DEFAULT);

    // Time the kernel
    cuEventRecord(start, nullptr);
    launch_fn();
    cuEventRecord(stop, nullptr);
    cuEventSynchronize(stop);

    cuEventElapsedTime(&metrics.elapsed_ms, start, stop);

    cuEventDestroy(start);
    cuEventDestroy(stop);

    // TODO: Collect CUPTI metrics here in Phase 2
    // Use cuptiProfilerBeginPass / cuptiProfilerEndPass
    // Query specific metrics like sm__throughput.avg

    return metrics;
}

std::vector<std::string> Profiler::available_metrics() const {
    // This would query CUPTI for available metrics on the current device
    // For now, return commonly available metrics
    return {
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    };
}

void Profiler::enable_metric(const std::string& metric_name) {
    // TODO: Enable specific CUPTI metrics
}

void Profiler::disable_metric(const std::string& metric_name) {
    // TODO: Disable specific CUPTI metrics
}

}


#include "arena/profiler.hpp"
#include "arena/utils.hpp"
#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <string>

namespace arena {

struct KernelActivityData {
    uint64_t duration_ns = 0;
    int registers_per_thread = 0;
    int shared_memory = 0;
};

// Map of kernel name -> activity data
static std::unordered_map<std::string, KernelActivityData> s_activity_results;

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size = 16 * 1024;
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, 
                                      uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = nullptr;
    
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4*)record;
            
            KernelActivityData& data = s_activity_results[kernel->name];
            data.duration_ns = kernel->end - kernel->start;
            data.registers_per_thread = kernel->registersPerThread;
            data.shared_memory = kernel->staticSharedMemory + kernel->dynamicSharedMemory;
        }
    }
    
    free(buffer);
}

Profiler::Profiler() {
    init_cupti();
}

Profiler::~Profiler() {
    cleanup_cupti();
}

void Profiler::init_cupti() {
    if (initialized_) return;

    CUptiResult result = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Warning: CUPTI Activity registration failed.\n";
        return;
    }

    initialized_ = true;
}

void Profiler::cleanup_cupti() {
    if (!initialized_) return;
    
    cuptiActivityFlushAll(0);
    initialized_ = false;
}

Profiler::KernelMetrics Profiler::profile(KernelLaunchFn launch_fn, ProfilerConfig config) {
    KernelMetrics metrics;

    // Clear previous results TODO: Do we need to clear everything ?
    s_activity_results.clear();

    // Enable activity tracking
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

    // Create CUDA events for timing (backup if Activity API fails)
    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop, CU_EVENT_DEFAULT);

    cuEventRecord(start, nullptr);

    // Run the kernel
    launch_fn();

    cuEventRecord(stop, nullptr);
    cuEventSynchronize(stop);

    // Get elapsed time from CUDA events
    cuEventElapsedTime(&metrics.elapsed_ms, start, stop);

    cuEventDestroy(start);
    cuEventDestroy(stop);

    // Flush activity buffers to trigger callbacks
    cuptiActivityFlushAll(0);

    // Disable activity tracking
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);

    // Collect results from activity map (safe: flush blocks until callbacks done)
    if (!s_activity_results.empty()) {
        // Get the first (usually only) kernel's data
        const auto& [name, data] = *s_activity_results.begin();
        metrics.elapsed_ms = data.duration_ns / 1e6f;
        metrics.registers_per_thread = data.registers_per_thread;
        metrics.shared_memory_per_block = data.shared_memory;
        
    }

    return metrics;
}

std::vector<std::string> Profiler::available_metrics() const {
    return {
        "elapsed_ms",
        "registers_per_thread",
        "shared_memory_per_block"
    };
}

void Profiler::enable_metric(const std::string& metric_name) {
    // Activity API metrics are always collected
}

void Profiler::disable_metric(const std::string& metric_name) {
    // Activity API metrics are always collected
}

}

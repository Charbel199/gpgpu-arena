#pragma once

#include <cuda.h>
#include <cupti.h>
#include <cupti_target.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace arena {

// Logical metric keys (chip-agnostic) used as keys in ProfileResult::metric_values
namespace metric {
    constexpr const char* DRAM_READ  = "dram_read_bytes";
    constexpr const char* DRAM_WRITE = "dram_write_bytes";
    constexpr const char* OCCUPANCY  = "occupancy";
    constexpr const char* IPC        = "ipc";
}

class Profiler {
public:
    Profiler();
    ~Profiler();

    using KernelLaunchFn = std::function<void()>;

    struct SubKernelInfo {
        std::string name;
        float duration_ms = 0.0f;
        int registers = 0;
        int shared_memory = 0;
    };

    struct ProfileResult {
        float kernel_time_ms = 0.0f;   // GPU-only kernel execution time
        int registers_per_thread = 0;
        int shared_memory_per_block = 0;
        std::vector<SubKernelInfo> sub_kernels;  // per-kernel breakdown
        std::map<std::string, double> metric_values;  // keyed by metric:: constants
    };

    // kernel-only time + registers + shmem (Activity API, one kernel launch)
    ProfileResult collect_activity(KernelLaunchFn launch_fn);

    // full profiling: Activity API + Range Profiler hardware counters
    ProfileResult profile(KernelLaunchFn launch_fn, KernelLaunchFn reset_fn = nullptr);

private:
    void init_activity();
    void cleanup_activity();

    // CUPTI range profiler API (hardware counters)
    void init_range_profiler();
    void cleanup_range_profiler();
    CUpti_Profiler_Host_Object* create_host_object();
    void destroy_host_object();
    std::map<std::string, double> collect_counters(
        KernelLaunchFn launch_fn, KernelLaunchFn reset_fn = nullptr);

    // CUPTI metric names for the current chip
    struct ChipMetrics {
        std::string dram_read;
        std::string dram_write;
        std::string occupancy;
        std::string ipc;
    };
    ChipMetrics chip_metrics_;

    bool activity_initialized_ = false;
    bool range_profiler_initialized_ = false;

    std::string chip_name_;
    CUpti_Profiler_Host_Object* host_object_ = nullptr;
    std::vector<uint8_t> counter_availability_image_;
};

}

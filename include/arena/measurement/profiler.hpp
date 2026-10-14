#pragma once

#include <cuda.h>
#include <cupti.h>
#include <cupti_target.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include "arena/measurement/launch_summary.hpp"
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

    using SubKernelInfo = arena::SubKernelInfo;
    using ProfileResult = LaunchSummary;

    // kernel-only time + registers + shmem (Activity API, one kernel launch)
    ProfileResult collect_activity(KernelLaunchFn launch_fn);

    // Hardware counters via the Range Profiler (occupancy, IPC, DRAM bytes).
    // Runner sequences this separately from collect_activity because activity
    // runs unconditionally while counters are opt-in.
    std::map<std::string, double> collect_counters(
        KernelLaunchFn launch_fn, KernelLaunchFn reset_fn = nullptr);

private:
    void init_activity();
    void cleanup_activity();

    // CUPTI range profiler API (hardware counters)
    void init_range_profiler();
    void cleanup_range_profiler();
    CUpti_Profiler_Host_Object* create_host_object();
    void destroy_host_object();

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

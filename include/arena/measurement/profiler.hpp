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

    // --counters full only. Keys with these prefixes carry a pipe or stall
    // reason name after the dot.
    constexpr const char* ISSUE_ACTIVE    = "issue_active";
    constexpr const char* ELIGIBLE_WARPS  = "eligible_warps";
    constexpr const char* SHARED_WAVEFRONTS = "shared_wavefronts";
    constexpr const char* SHARED_BANK_CONFLICTS = "shared_bank_conflicts";
    constexpr const char* L2_SECTORS      = "l2_sectors";
    // Summed launch time of the operation. Used to pick the launch that
    // dominates when rates are read from one launch of several.
    constexpr const char* GPU_TIME_NS     = "gpu_time_ns";
    constexpr const char* PIPE_PREFIX     = "pipe.";
    constexpr const char* STALL_PREFIX    = "stall.";
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
    //
    // launches_per_op is how many kernel launches one call of launch_fn
    // makes (from collect_activity), so ranges can be grouped back into
    // whole operations.
    //
    // full adds pipe utilisation, stall reasons, shared memory and L2
    // traffic. Metrics this chip does not have are dropped once, on first
    // use, so the result only carries the keys that were measured.
    std::map<std::string, double> collect_counters(
        KernelLaunchFn launch_fn, KernelLaunchFn reset_fn = nullptr,
        int launches_per_op = 1, bool full = false);

    // Replay passes the last collect_counters call needed.
    int last_num_passes() const { return last_num_passes_; }

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

    // --counters full: logical key, CUPTI name, and whether it adds up
    // across an operation's launches. Resolved against the chip once.
    struct ExtraMetric {
        std::string key;
        std::string cupti_name;
        bool additive = false;
    };
    void resolve_full_metrics();
    std::vector<ExtraMetric> full_metrics_;
    bool full_metrics_resolved_ = false;
    int last_num_passes_ = 0;

    bool activity_initialized_ = false;
    bool range_profiler_initialized_ = false;

    std::string chip_name_;
    CUpti_Profiler_Host_Object* host_object_ = nullptr;
    std::vector<uint8_t> counter_availability_image_;
};

}

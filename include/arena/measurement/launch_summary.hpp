#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Pure aggregation of per-launch activity records. Kept out of profiler.hpp so
// it can be unit-tested without CUPTI.

namespace arena {

// One kernel launch as CUPTI's Activity API reports it.
struct KernelLaunchRecord {
    std::string name;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    int registers_per_thread = 0;
    int shared_memory = 0;
};

struct SubKernelInfo {
    std::string name;
    float duration_ms = 0.0f;
    int registers = 0;
    int shared_memory = 0;
    float start_offset_ms = 0.0f;   // from the first launch's start
};

struct LaunchSummary {
    float kernel_time_ms = 0.0f;     // sum of launch durations
    int launch_count = 0;
    int registers_per_thread = 0;    // of the longest launch
    int shared_memory_per_block = 0; // of the longest launch
    std::vector<SubKernelInfo> sub_kernels;   // one per launch, in start order
};

// Records can arrive out of order (they are flushed per buffer), so they are
// sorted by start time. Every launch is kept, including repeats of the same
// kernel, and their durations are summed.
LaunchSummary summarize_launches(std::vector<KernelLaunchRecord> records);

}

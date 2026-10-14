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

// Counter values for one operation, picked from a run of profiled ranges.
//
// The range profiler gives one range per launch, so an operation that makes
// several launches (two_stage, CUB) spans several consecutive ranges. They
// have to be grouped back into operations: taking the minimum over single
// ranges used to pick the tiny second-stage launch and report 0 MB of DRAM
// traffic for the whole operation.
//
// ranges[i][m] is metric m of range i. additive[m] says whether metric m adds
// up across an operation's launches (bytes) or is a rate that does not
// (occupancy, IPC). Additive metrics are summed over the group; the others
// come from the group's range with the largest value of metric 0, the launch
// that dominates. The first operation is skipped because it starts cold, and
// of the rest the one with the smallest metric 0 is returned: instrumentation
// can only add DRAM traffic, so the minimum is the closest to the real loop.
// Falls back to operation 0 when it is the only complete one.
std::vector<double> select_counter_op(const std::vector<std::vector<double>>& ranges,
                                      int launches_per_op,
                                      const std::vector<bool>& additive);

}

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

// One distinct device kernel an operation launched, with how often and for
// how long. This is how a library call such as cublasSgemm shows which
// kernel it dispatched at a given size.
struct LaunchedKernel {
    std::string name;        // demangled, without return type or parameters
    int count_per_op = 0;
    float gpu_ms = 0.0f;     // mean per operation, all its launches summed
};

// Readable name for a CUPTI kernel record. Itanium-mangled names (_Z...) are
// demangled, then the parameter list and any return type are dropped, so
// "void cutlass::Kernel2<X>(X::Params)" becomes "cutlass::Kernel2<X>".
// Names that are not mangled (extern "C" kernels) come back unchanged.
std::string short_kernel_name(const std::string& raw);

// Groups the launches of `ops` operations by short name. Counts and times are
// divided by `ops`, so they read per operation. Ordered by gpu_ms, largest
// first.
std::vector<LaunchedKernel> group_launched_kernels(const std::vector<SubKernelInfo>& launches,
                                                   int ops = 1);

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
// come from the group's range with the largest value of metric
// dominant_metric, the launch that dominates. That is metric 0 unless a
// launch duration was collected: DRAM bytes are noise for an L2-resident
// operation, and keying on them can pick a 1 us init kernel over the main one. The first operation is skipped because it starts cold, and
// of the rest the one with the smallest metric 0 is returned: instrumentation
// can only add DRAM traffic, so the minimum is the closest to the real loop.
// Falls back to operation 0 when it is the only complete one.
std::vector<double> select_counter_op(const std::vector<std::vector<double>>& ranges,
                                      int launches_per_op,
                                      const std::vector<bool>& additive,
                                      size_t dominant_metric = 0);

}

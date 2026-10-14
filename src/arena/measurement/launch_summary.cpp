#include "arena/measurement/launch_summary.hpp"

#include <algorithm>

namespace arena {

LaunchSummary summarize_launches(std::vector<KernelLaunchRecord> records) {
    LaunchSummary out;
    if (records.empty()) return out;

    std::stable_sort(records.begin(), records.end(),
        [](const auto& a, const auto& b) { return a.start_ns < b.start_ns; });

    const uint64_t t0 = records.front().start_ns;
    uint64_t total_ns = 0;
    uint64_t longest_ns = 0;

    for (const auto& r : records) {
        const uint64_t d = r.end_ns > r.start_ns ? r.end_ns - r.start_ns : 0;
        total_ns += d;

        SubKernelInfo sk;
        sk.name = r.name;
        sk.duration_ms = d / 1e6f;
        sk.registers = r.registers_per_thread;
        sk.shared_memory = r.shared_memory;
        sk.start_offset_ms = (r.start_ns - t0) / 1e6f;
        out.sub_kernels.push_back(std::move(sk));

        // Metadata comes from the longest launch, the one that dominates.
        if (d > longest_ns || &r == &records.front()) {
            longest_ns = d;
            out.registers_per_thread = r.registers_per_thread;
            out.shared_memory_per_block = r.shared_memory;
        }
    }

    out.kernel_time_ms = total_ns / 1e6f;
    out.launch_count = static_cast<int>(records.size());
    return out;
}

}

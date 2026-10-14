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

std::vector<double> select_counter_op(const std::vector<std::vector<double>>& ranges,
                                      int launches_per_op,
                                      const std::vector<bool>& additive) {
    if (ranges.empty()) return {};
    const size_t per_op = launches_per_op > 0 ? static_cast<size_t>(launches_per_op) : 1;
    const size_t n_metrics = additive.size();
    const size_t n_ops = ranges.size() / per_op;
    if (n_ops == 0) return {};

    auto reduce_op = [&](size_t op) {
        std::vector<double> out(n_metrics, 0.0);
        size_t dominant = op * per_op;
        for (size_t i = op * per_op; i < (op + 1) * per_op; i++) {
            if (ranges[i].size() < n_metrics) continue;
            for (size_t m = 0; m < n_metrics; m++) {
                if (additive[m]) out[m] += ranges[i][m];
            }
            if (ranges[i][0] > ranges[dominant][0]) dominant = i;
        }
        for (size_t m = 0; m < n_metrics; m++) {
            if (!additive[m] && ranges[dominant].size() >= n_metrics) out[m] = ranges[dominant][m];
        }
        return out;
    };

    if (n_ops == 1) return reduce_op(0);

    std::vector<double> best;
    for (size_t op = 1; op < n_ops; op++) {
        auto v = reduce_op(op);
        if (best.empty() || v[0] < best[0]) best = std::move(v);
    }
    return best;
}

}

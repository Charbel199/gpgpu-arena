#include "arena/measurement/launch_summary.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cxxabi.h>
#include <map>

namespace arena {

std::string short_kernel_name(const std::string& raw) {
    if (raw.size() < 2 || raw.compare(0, 2, "_Z") != 0) return raw;

    int status = 0;
    char* d = abi::__cxa_demangle(raw.c_str(), nullptr, nullptr, &status);
    if (status != 0 || !d) {
        std::free(d);
        return raw;
    }
    std::string s(d);
    std::free(d);

    // Drop the trailing parameter list: walk back from the final ')' to the
    // '(' that opens it.
    if (!s.empty() && s.back() == ')') {
        int depth = 0;
        for (size_t i = s.size(); i-- > 0;) {
            if (s[i] == ')') {
                depth++;
            } else if (s[i] == '(' && --depth == 0) {
                if (i > 0) s.erase(i);
                break;
            }
        }
    }

    // Drop a return type, which the demangler prints only for template
    // functions. It ends at the last space outside any <> or ().
    int depth = 0;
    size_t cut = std::string::npos;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (c == '<' || c == '(') depth++;
        else if ((c == '>' || c == ')') && depth > 0) depth--;
        else if (c == ' ' && depth == 0) cut = i;
    }
    if (cut != std::string::npos && cut + 1 < s.size()) s.erase(0, cut + 1);
    return s;
}

std::vector<LaunchedKernel> group_launched_kernels(const std::vector<SubKernelInfo>& launches,
                                                   int ops) {
    if (ops < 1) ops = 1;
    struct Acc { int count = 0; double ms = 0.0; };
    std::map<std::string, Acc> by_name;
    for (const auto& sk : launches) {
        auto& a = by_name[short_kernel_name(sk.name)];
        a.count++;
        a.ms += sk.duration_ms;
    }

    std::vector<LaunchedKernel> out;
    for (const auto& [name, a] : by_name) {
        LaunchedKernel k;
        k.name = name;
        // A kernel that ran on only some of the operations still reports at
        // least one launch, so it does not disappear from the list.
        k.count_per_op = std::max(1, static_cast<int>(std::lround(double(a.count) / ops)));
        k.gpu_ms = static_cast<float>(a.ms / ops);
        out.push_back(std::move(k));
    }
    std::stable_sort(out.begin(), out.end(),
        [](const auto& a, const auto& b) { return a.gpu_ms > b.gpu_ms; });
    return out;
}

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
                                      const std::vector<bool>& additive,
                                      size_t dominant_metric) {
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
            if (dominant_metric < n_metrics &&
                ranges[i][dominant_metric] > ranges[dominant][dominant_metric]) dominant = i;
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

#include "arena/measurement/measure_policy.hpp"

#include <algorithm>
#include <cmath>

namespace arena::measure {

float median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

DriftCheck check_drift(const std::vector<float>& history, int window, float threshold) {
    DriftCheck out;
    if (window <= 0) return out;
    if (history.size() < static_cast<size_t>(2 * window)) return out;

    const auto end = history.end();
    std::vector<float> recent(end - window, end);
    std::vector<float> prev(end - 2 * window, end - window);

    const float m_recent = median(std::move(recent));
    const float m_prev   = median(std::move(prev));
    if (!(m_prev > 0.0f)) return out;

    out.rel_change = std::fabs(m_recent - m_prev) / m_prev;
    out.converged  = out.rel_change < threshold;
    return out;
}

double rate_per_sec(double quantity, float ms) {
    if (!(ms > 0.0f)) return 0.0;   // also rejects NaN
    return quantity / (static_cast<double>(ms) / 1000.0);
}

}

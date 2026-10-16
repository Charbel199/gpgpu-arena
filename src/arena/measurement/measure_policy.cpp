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

float trimmed_mean(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    size_t lo = n / 4, hi = n - n / 4;
    if (hi <= lo) { lo = 0; hi = n; }
    double s = 0.0;
    for (size_t i = lo; i < hi; i++) s += v[i];
    return static_cast<float>(s / (hi - lo));
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

ClockPowerStats summarize_clock_power(const std::vector<unsigned>& sm_mhz,
                                      const std::vector<double>& watts) {
    ClockPowerStats out;
    if (!sm_mhz.empty()) {
        double sum = 0.0;
        out.sm_min_mhz = sm_mhz.front();
        out.sm_max_mhz = sm_mhz.front();
        for (unsigned c : sm_mhz) {
            sum += c;
            out.sm_min_mhz = std::min(out.sm_min_mhz, c);
            out.sm_max_mhz = std::max(out.sm_max_mhz, c);
        }
        out.samples = static_cast<int>(sm_mhz.size());
        out.sm_mean_mhz = sum / sm_mhz.size();
    }
    if (!watts.empty()) {
        double sum = 0.0;
        for (double w : watts) {
            sum += w;
            out.power_max_w = std::max(out.power_max_w, w);
        }
        out.power_mean_w = sum / watts.size();
    }
    return out;
}

unsigned pick_auto_lock_clock(unsigned max_mhz, unsigned default_app_mhz,
                              const std::vector<unsigned>& supported) {
    if (max_mhz == 0) return 0;
    unsigned target = static_cast<unsigned>(max_mhz * 0.85);
    if (default_app_mhz > 0) target = std::min(target, default_app_mhz);

    unsigned best = 0;
    for (unsigned c : supported) {
        if (c <= target && c > best) best = c;
    }
    return best > 0 ? best : target;
}

}

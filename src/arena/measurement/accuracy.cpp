#include "arena/measurement/accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace arena {

double relative_error(double got, double expected) {
    if (std::isnan(got) || std::isnan(expected)) {
        return std::numeric_limits<double>::infinity();
    }
    if (got == expected) return 0.0;   // catches inf == inf

    const double denom = std::abs(expected);
    const double diff  = std::abs(got - expected);
    if (denom < 1e-30) return diff;    // expected is zero: report absolute
    return diff / denom;
}

void ErrorAccumulator::add(double got, double expected_quantized, double expected_exact) {
    const double e = relative_error(got, expected_quantized);
    if (e > max_) max_ = e;
    sum_ += e;

    const double t = relative_error(got, expected_exact);
    if (t > max_total_) max_total_ = t;
    sum_total_ += t;

    count_++;
}

void ErrorAccumulator::add_scaled(double got, double expected, double scale) {
    double e;
    if (std::isnan(got) || std::isnan(expected)) {
        e = std::numeric_limits<double>::infinity();
    } else if (got == expected) {
        e = 0.0;
    } else {
        const double s = std::max(std::abs(scale), std::abs(expected));
        e = s < 1e-30 ? std::abs(got - expected) : std::abs(got - expected) / s;
    }
    if (e > max_) max_ = e;
    sum_ += e;
    if (e > max_total_) max_total_ = e;
    sum_total_ += e;
    count_++;
}

VerifyResult ErrorAccumulator::finish(double tolerance) const {
    VerifyResult r;
    r.elements_checked = count_;
    r.tolerance = tolerance;
    if (count_ == 0) return r;         // passed stays false

    r.max_rel_error  = max_;
    r.mean_rel_error = sum_ / count_;
    r.max_total_error  = max_total_;
    r.mean_total_error = sum_total_ / count_;

    // Judged on arithmetic error: a kernel should not fail for input rounding
    // that its declared storage type made unavoidable.
    r.passed = max_ <= tolerance;
    return r;
}

}

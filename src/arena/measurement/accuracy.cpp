#include "arena/measurement/accuracy.hpp"

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

void ErrorAccumulator::add(double got, double expected) {
    const double e = relative_error(got, expected);
    if (e > max_) max_ = e;
    sum_ += e;
    count_++;
}

VerifyResult ErrorAccumulator::finish(double tolerance) const {
    VerifyResult r;
    r.elements_checked = count_;
    r.tolerance = tolerance;
    if (count_ == 0) return r;         // passed stays false

    r.max_rel_error  = max_;
    r.mean_rel_error = sum_ / count_;
    r.passed = max_ <= tolerance;
    return r;
}

}

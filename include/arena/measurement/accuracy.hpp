#pragma once

#include <cstddef>

// Numerical accuracy as a measured quantity. No CUDA here, so the error maths
// is unit-testable without a GPU.

namespace arena {

struct VerifyResult {
    bool   passed = false;
    double max_rel_error = 0.0;
    double mean_rel_error = 0.0;
    int    elements_checked = 0;
    double tolerance = 0.0;    // what max_rel_error was judged against
};

// |got - expected| / |expected|, falling back to absolute error when expected
// is at or near zero so the ratio does not blow up on a legitimate zero.
//
// Returns infinity if either value is NaN, so a kernel that produces NaN
// fails rather than silently scoring well.
double relative_error(double got, double expected);

// Accumulates per-element errors so a category's verify() can report a
// distribution instead of a single boolean.
class ErrorAccumulator {
public:
    void add(double got, double expected);

    // elements_checked of zero yields passed=false: a check that examined
    // nothing has not demonstrated anything.
    VerifyResult finish(double tolerance) const;

private:
    double max_ = 0.0;
    double sum_ = 0.0;
    int    count_ = 0;
};

}

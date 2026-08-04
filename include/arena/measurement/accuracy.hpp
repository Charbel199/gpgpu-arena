#pragma once

#include <cstddef>

// Numerical accuracy as a measured quantity. No CUDA here, so the error maths
// is unit-testable without a GPU.

namespace arena {

struct VerifyResult {
    bool passed = false;

    // Arithmetic error: measured against a reference built from the values the
    // kernel actually received, so inputs already rounded to its storage type.
    // This is the kernel's own doing, so it is what pass/fail is judged on, and
    // it is comparable between kernels of the same dtype.
    double max_rel_error = 0.0;
    double mean_rel_error = 0.0;

    // Total error: measured against the original fp32 data, so it includes the
    // cost of storing inputs in a narrower type. This is the one that is
    // comparable across dtypes. For an fp32 kernel the two are identical.
    double max_total_error = 0.0;
    double mean_total_error = 0.0;

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
    // Two references: what the kernel could have produced given the inputs it
    // was handed, and what the original data called for. Reporting only the
    // first makes a narrow-dtype kernel look more accurate than an fp32 one,
    // because the input rounding is excluded from both the number and the
    // reader's attention.
    void add(double got, double expected_quantized, double expected_exact);

    // Same reference for both, which is the fp32 case.
    void add(double got, double expected) { add(got, expected, expected); }

    // elements_checked of zero yields passed=false: a check that examined
    // nothing has not demonstrated anything.
    VerifyResult finish(double tolerance) const;

private:
    double max_ = 0.0, sum_ = 0.0;
    double max_total_ = 0.0, sum_total_ = 0.0;
    int    count_ = 0;
};

}

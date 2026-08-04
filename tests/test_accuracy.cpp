#include <doctest/doctest.h>
#include "arena/measurement/accuracy.hpp"
#include "arena/dtype.hpp"

#include <cmath>
#include <limits>

using namespace arena;

TEST_CASE("relative_error") {
    SUBCASE("exact match is zero") {
        CHECK(relative_error(1.5, 1.5) == 0.0);
    }
    SUBCASE("ordinary case") {
        CHECK(relative_error(1.1, 1.0) == doctest::Approx(0.1));
    }
    SUBCASE("sign of the difference does not matter") {
        CHECK(relative_error(0.9, 1.0) == doctest::Approx(0.1));
    }
    SUBCASE("expected zero falls back to absolute error") {
        // A ratio would be infinite here even for a tiny miss, which would
        // make any kernel with a legitimate zero output look broken.
        CHECK(relative_error(1e-8, 0.0) == doctest::Approx(1e-8));
    }
    SUBCASE("NaN scores infinity so it cannot pass") {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        CHECK(std::isinf(relative_error(nan, 1.0)));
        CHECK(std::isinf(relative_error(1.0, nan)));
    }
    SUBCASE("matching infinities count as equal") {
        const double inf = std::numeric_limits<double>::infinity();
        CHECK(relative_error(inf, inf) == 0.0);
    }
    SUBCASE("fp32 accumulator saturation is a large error, not a NaN") {
        // reduce_baseline at 64M: the accumulator stops advancing at 2^24
        // while the true sum keeps going.
        CHECK(relative_error(16777216.0, 64000000.0) == doctest::Approx(0.738).epsilon(0.01));
    }
}

TEST_CASE("ErrorAccumulator") {
    SUBCASE("reports max and mean separately") {
        ErrorAccumulator a;
        a.add(1.0, 1.0);      // 0.0
        a.add(1.2, 1.0);      // 0.2
        auto r = a.finish(0.5);
        CHECK(r.elements_checked == 2);
        CHECK(r.max_rel_error == doctest::Approx(0.2));
        CHECK(r.mean_rel_error == doctest::Approx(0.1));
        CHECK(r.passed);
    }
    SUBCASE("judged on max, not mean") {
        // One bad element out of many must fail even though the mean is fine.
        ErrorAccumulator a;
        for (int i = 0; i < 99; i++) a.add(1.0, 1.0);
        a.add(2.0, 1.0);      // single 100% error
        auto r = a.finish(0.5);
        CHECK(r.mean_rel_error < 0.02);
        CHECK_FALSE(r.passed);
    }
    SUBCASE("checking nothing does not pass") {
        ErrorAccumulator a;
        auto r = a.finish(1.0);
        CHECK(r.elements_checked == 0);
        CHECK_FALSE(r.passed);
    }
    SUBCASE("tolerance is recorded so a result can be re-judged") {
        ErrorAccumulator a;
        a.add(1.1, 1.0);
        CHECK(a.finish(0.05).tolerance == doctest::Approx(0.05));
    }
    SUBCASE("error exactly at tolerance passes") {
        ErrorAccumulator a;
        a.add(1.5, 1.0);      // 0.5
        CHECK(a.finish(0.5).passed);
    }
}

TEST_CASE("dtype") {
    SUBCASE("sizes") {
        CHECK(dtype_size(DType::FP32) == 4);
        CHECK(dtype_size(DType::FP16) == 2);
        CHECK(dtype_size(DType::BF16) == 2);
    }
    SUBCASE("names round-trip into result files") {
        CHECK(std::string(dtype_name(DType::FP32)) == "fp32");
        CHECK(std::string(dtype_name(DType::FP16)) == "fp16");
        CHECK(std::string(dtype_name(DType::BF16)) == "bf16");
    }
    SUBCASE("tolerance loosens as the mantissa shrinks") {
        const double f32 = default_tolerance(DType::FP32);
        const double f16 = default_tolerance(DType::FP16);
        const double b16 = default_tolerance(DType::BF16);
        CHECK(f32 < f16);
        CHECK(f16 < b16);
    }
    SUBCASE("tf32 compute loosens fp32 storage") {
        // This is what matmul_base's hand-picked 5e-2 was standing in for.
        CHECK(default_tolerance(DType::FP32, ComputeMode::TF32)
              > default_tolerance(DType::FP32, ComputeMode::Default));
    }
    SUBCASE("per-element budget stays tight enough to catch saturation") {
        // Even scaled by sqrt(64e6), the fp32 budget must not admit the
        // 0.738 error a saturated accumulator produces.
        const double scaled = default_tolerance(DType::FP32) * std::sqrt(64e6);
        CHECK(scaled < 0.738);
    }
}

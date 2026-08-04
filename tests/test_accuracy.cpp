#include <doctest/doctest.h>
#include "arena/measurement/accuracy.hpp"
#include "arena/dtype.hpp"

#include <cmath>
#include <limits>
#include <vector>

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

TEST_CASE("half conversion") {
    SUBCASE("exact values round-trip") {
        for (float f : {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -256.0f, 1024.0f}) {
            CHECK(half_to_float(float_to_half(f)) == f);
        }
    }
    SUBCASE("signed zero is preserved") {
        CHECK(half_to_float(float_to_half(-0.0f)) == 0.0f);
        CHECK(std::signbit(half_to_float(float_to_half(-0.0f))));
    }
    SUBCASE("integers above 2048 start losing steps") {
        // This is why an fp16 accumulator stops counting: 2049 is not
        // representable, so adding 1.0 to 2048.0 does nothing.
        CHECK(half_to_float(float_to_half(2048.0f)) == 2048.0f);
        CHECK(half_to_float(float_to_half(2049.0f)) == 2048.0f);
    }
    SUBCASE("overflow saturates to infinity") {
        CHECK(std::isinf(half_to_float(float_to_half(1e6f))));
    }
    SUBCASE("underflow reaches subnormals then zero") {
        CHECK(half_to_float(float_to_half(1e-7f)) > 0.0f);    // subnormal
        CHECK(half_to_float(float_to_half(1e-10f)) == 0.0f);  // too small
    }
    SUBCASE("NaN stays NaN") {
        CHECK(std::isnan(half_to_float(float_to_half(
            std::numeric_limits<float>::quiet_NaN()))));
    }
    SUBCASE("round-trip error stays within one half-ulp") {
        for (float f : {0.3f, 1.7f, -12.34f, 987.6f}) {
            CHECK(relative_error(half_to_float(float_to_half(f)), f) < 1e-3);
        }
    }
}

TEST_CASE("bf16 conversion") {
    SUBCASE("exact values round-trip") {
        for (float f : {0.0f, 1.0f, -2.0f, 256.0f}) {
            CHECK(bf16_to_float(float_to_bf16(f)) == f);
        }
    }
    SUBCASE("keeps fp32 range but loses mantissa") {
        // bf16 holds far larger values than fp16, at much lower precision.
        CHECK(bf16_to_float(float_to_bf16(1e30f)) > 1e29f);
        CHECK(relative_error(bf16_to_float(float_to_bf16(1.234f)), 1.234f) < 1e-2);
    }
    SUBCASE("NaN stays NaN") {
        CHECK(std::isnan(bf16_to_float(float_to_bf16(
            std::numeric_limits<float>::quiet_NaN()))));
    }
}

TEST_CASE("quantize_in_place") {
    SUBCASE("fp32 leaves data untouched") {
        std::vector<float> v{0.1f, 0.2f, 0.3f}, orig = v;
        quantize_in_place(v, DType::FP32);
        CHECK(v == orig);
    }
    SUBCASE("fp16 rounds every element") {
        std::vector<float> v{0.1f, 0.2f, 0.3f};
        quantize_in_place(v, DType::FP16);
        for (size_t i = 0; i < v.size(); i++) CHECK(v[i] != 0.0f);
        CHECK(v[0] != 0.1f);   // 0.1 is not representable in half
    }
}

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "arena/measurement/measure_policy.hpp"

#include <vector>

using namespace arena::measure;

TEST_CASE("median") {
    SUBCASE("odd count returns the middle element") {
        CHECK(median({3.0f, 1.0f, 2.0f}) == doctest::Approx(2.0f));
    }
    SUBCASE("even count averages the two middle elements") {
        CHECK(median({1.0f, 2.0f, 3.0f, 4.0f}) == doctest::Approx(2.5f));
    }
    SUBCASE("single element") {
        CHECK(median({7.5f}) == doctest::Approx(7.5f));
    }
    SUBCASE("empty returns zero") {
        CHECK(median({}) == doctest::Approx(0.0f));
    }
}

TEST_CASE("check_drift") {
    SUBCASE("monotonic ramp does not converge") {
        // A GPU boosting its clock produces steadily falling times. Each
        // consecutive pair looks tight, which is exactly what a sliding
        // standard-deviation test gets wrong. Drift detection must catch it.
        std::vector<float> ramp = {10.0f, 9.5f, 9.0f, 8.5f, 8.0f,
                                   7.5f,  7.0f, 6.5f, 6.0f, 5.5f};
        auto r = check_drift(ramp, 5, 0.02f);
        CHECK_FALSE(r.converged);
        CHECK(r.rel_change > 0.02f);
    }

    SUBCASE("stable series with noise converges") {
        std::vector<float> stable = {5.00f, 5.02f, 4.98f, 5.01f, 4.99f,
                                     5.01f, 4.99f, 5.00f, 5.02f, 4.98f};
        auto r = check_drift(stable, 5, 0.02f);
        CHECK(r.converged);
        CHECK(r.rel_change < 0.02f);
    }

    SUBCASE("history shorter than 2*window never converges") {
        std::vector<float> few = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
        auto r = check_drift(few, 5, 0.02f);   // needs 10, has 7
        CHECK_FALSE(r.converged);
        CHECK(r.rel_change == doctest::Approx(0.0f));
    }

    SUBCASE("single outlier spike does not break a converged series") {
        // Median is robust to one bad sample; this is why we use median
        // drift rather than mean drift.
        std::vector<float> spiked = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                                     5.0f, 5.0f, 99.0f, 5.0f, 5.0f};
        auto r = check_drift(spiked, 5, 0.02f);
        CHECK(r.converged);
    }

    SUBCASE("change exactly at threshold is not converged") {
        // Binary-exact values only: 1.0 -> 1.03125 is exactly 1/32, and
        // 1/32 is exactly representable as a float. Decimal literals like
        // 5.1f land slightly off their decimal value, so a "5.0 -> 5.1 is
        // exactly 2%" case would silently test one side of the boundary
        // rather than the boundary itself.
        constexpr float kThreshold = 0.03125f;   // 1/32
        std::vector<float> boundary = {1.0f,     1.0f,     1.0f,     1.0f,     1.0f,
                                       1.03125f, 1.03125f, 1.03125f, 1.03125f, 1.03125f};
        auto r = check_drift(boundary, 5, kThreshold);
        CHECK(r.rel_change == kThreshold);   // exact equality, no Approx
        CHECK_FALSE(r.converged);            // converged requires strictly less
    }

    SUBCASE("change just under threshold converges") {
        // Same construction, one ULP-scale step below the boundary.
        constexpr float kThreshold = 0.03125f;   // 1/32
        std::vector<float> under = {1.0f,      1.0f,      1.0f,      1.0f,      1.0f,
                                    1.015625f, 1.015625f, 1.015625f, 1.015625f, 1.015625f};
        auto r = check_drift(under, 5, kThreshold);
        CHECK(r.rel_change == 0.015625f);    // 1/64
        CHECK(r.converged);
    }

    SUBCASE("zero window is rejected") {
        std::vector<float> any = {1.0f, 2.0f, 3.0f, 4.0f};
        CHECK_FALSE(check_drift(any, 0, 0.02f).converged);
    }
}

TEST_CASE("rate_per_sec") {
    SUBCASE("normal case") {
        // 1e9 bytes in 1 ms == 1e12 bytes/sec
        CHECK(rate_per_sec(1e9, 1.0f) == doctest::Approx(1e12));
    }
    SUBCASE("zero ms returns zero, not inf") {
        CHECK(rate_per_sec(1e9, 0.0f) == doctest::Approx(0.0));
    }
    SUBCASE("negative ms returns zero") {
        CHECK(rate_per_sec(1e9, -1.0f) == doctest::Approx(0.0));
    }
    SUBCASE("zero quantity is zero") {
        CHECK(rate_per_sec(0.0, 5.0f) == doctest::Approx(0.0));
    }
}

// ---------------------------------------------------------------------------
// Sweep ladder generation. Pure arithmetic on RunConfig, so it is testable
// without a GPU; the generator itself lives on KernelDescriptor.
// ---------------------------------------------------------------------------
#include "arena/runner_config.hpp"

namespace {
// Mirrors KernelDescriptor::get_sweep_configs for a single parameter.
std::vector<int> ladder(int lo, int hi, double factor) {
    std::vector<int> out;
    if (lo <= 0 || hi < lo || factor <= 1.0) return out;
    for (double v = lo; v <= (double)hi * 1.0001; v *= factor)
        out.push_back((int)(v + 0.5));
    return out;
}
}

TEST_CASE("sweep ladder") {
    SUBCASE("powers of four between the bounds") {
        CHECK(ladder(256, 16384, 4.0) == std::vector<int>{256, 1024, 4096, 16384});
    }
    SUBCASE("includes the upper bound when it lands exactly") {
        auto v = ladder(64, 8192, 2.0);
        CHECK(v.front() == 64);
        CHECK(v.back() == 8192);
    }
    SUBCASE("stops below the bound when the step overshoots") {
        CHECK(ladder(100, 999, 10.0) == std::vector<int>{100});
    }
    SUBCASE("single point when min equals max") {
        CHECK(ladder(1024, 1024, 4.0) == std::vector<int>{1024});
    }
    SUBCASE("degenerate inputs give nothing rather than looping forever") {
        CHECK(ladder(0, 100, 4.0).empty());
        CHECK(ladder(100, 10, 4.0).empty());
        CHECK(ladder(1, 100, 1.0).empty());     // factor of 1 would never advance
        CHECK(ladder(1, 100, 0.5).empty());
    }
}

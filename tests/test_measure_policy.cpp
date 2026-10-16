#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "arena/measurement/measure_policy.hpp"
#include "arena/categories/ideal_bytes.hpp"

#include <vector>

using namespace arena::measure;
using arena::IdealBytes;
using arena::DType;
namespace ideal = arena::ideal;

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

TEST_CASE("trimmed_mean") {
    SUBCASE("drops the outer quarters") {
        // sorted: 1 2 3 4 5 6 7 100 -> middle half 3 4 5 6
        CHECK(trimmed_mean({100, 1, 2, 3, 4, 5, 6, 7}) == doctest::Approx(4.5f));
    }
    SUBCASE("moves smoothly across clustered values where the median jumps") {
        // half the runs at 4.096, half at 6.144: the median sits on a cluster
        // edge, the trimmed mean between them
        std::vector<float> v;
        for (int i = 0; i < 10; i++) v.push_back(4.096f);
        for (int i = 0; i < 10; i++) v.push_back(6.144f);
        CHECK(trimmed_mean(v) == doctest::Approx(5.12f));
    }
    SUBCASE("small inputs") {
        CHECK(trimmed_mean({}) == doctest::Approx(0.0f));
        CHECK(trimmed_mean({3.0f}) == doctest::Approx(3.0f));
        CHECK(trimmed_mean({1.0f, 3.0f}) == doctest::Approx(2.0f));
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

TEST_CASE("summarize_clock_power") {
    SUBCASE("no samples gives zeros") {
        auto s = summarize_clock_power({}, {});
        CHECK(s.samples == 0);
        CHECK(s.sm_mean_mhz == doctest::Approx(0.0));
        CHECK(s.power_max_w == doctest::Approx(0.0));
    }
    SUBCASE("mean, min and max") {
        auto s = summarize_clock_power({2400, 2600, 2500}, {200.0, 300.0});
        CHECK(s.samples == 3);
        CHECK(s.sm_mean_mhz == doctest::Approx(2500.0));
        CHECK(s.sm_min_mhz == 2400);
        CHECK(s.sm_max_mhz == 2600);
        CHECK(s.power_mean_w == doctest::Approx(250.0));
        CHECK(s.power_max_w == doctest::Approx(300.0));
    }
    SUBCASE("clock without power") {
        auto s = summarize_clock_power({1000}, {});
        CHECK(s.samples == 1);
        CHECK(s.power_mean_w == doctest::Approx(0.0));
    }
}

TEST_CASE("pick_auto_lock_clock") {
    const std::vector<unsigned> supported = {3090, 2625, 2617, 2610, 2595, 1500};
    SUBCASE("default application clock below 85% of max wins, snapped down") {
        CHECK(pick_auto_lock_clock(3090, 2612, supported) == 2610);
    }
    SUBCASE("85% of max caps an implausible default") {
        CHECK(pick_auto_lock_clock(3090, 3090, supported) == 2625);   // 85% = 2626
    }
    SUBCASE("unknown default falls back to 85% of max") {
        CHECK(pick_auto_lock_clock(3090, 0, supported) == 2625);
    }
    SUBCASE("no supported list returns the target itself") {
        CHECK(pick_auto_lock_clock(3090, 2617, {}) == 2617);
    }
    SUBCASE("no supported clock at or below the target returns the target") {
        CHECK(pick_auto_lock_clock(3090, 2400, {3090, 2800}) == 2400);
    }
    SUBCASE("max unknown") {
        CHECK(pick_auto_lock_clock(0, 2617, supported) == 0);
    }
}

TEST_CASE("ClockTally") {
    SUBCASE("nothing sampled is unknown, not zero") {
        ClockTally t;
        CHECK_FALSE(t.mean().has_value());
        t.add({});
        CHECK_FALSE(t.mean().has_value());
    }
    SUBCASE("pools every window, weighted by sample count") {
        // a long window at 2800 and a short one at 3000: the long one dominates
        ClockTally t;
        t.add({2800, 2800, 2800});
        t.add({3000});
        REQUIRE(t.mean().has_value());
        CHECK(*t.mean() == doctest::Approx(2850.0));
        CHECK(t.samples == 4);
    }
}

TEST_CASE("clock_below_request") {
    SUBCASE("the RTX PRO 6000 case: 3090 asked, 2817 achieved") {
        CHECK(clock_below_request(2817.0, 3090));
    }
    SUBCASE("within 3% is fine") {
        CHECK_FALSE(clock_below_request(3000.0, 3090));   // 2.9% below
        CHECK_FALSE(clock_below_request(3090.0, 3090));
        CHECK_FALSE(clock_below_request(3100.0, 3090));
    }
    SUBCASE("just past the tolerance") {
        CHECK(clock_below_request(2990.0, 3090));          // 3.2% below
    }
    SUBCASE("custom tolerance") {
        CHECK_FALSE(clock_below_request(2817.0, 3090, 0.10));
    }
    SUBCASE("unknown values never warn") {
        CHECK_FALSE(clock_below_request(0.0, 3090));
        CHECK_FALSE(clock_below_request(2817.0, 0));
    }
}

// --- ideal bytes per category (include/arena/categories/ideal_bytes.hpp) ---
// Each split must add up to what the category's calculate_bytes_accessed()
// returns, since bandwidth_gbps is built on that total. The totals below are
// those formulas written out.

TEST_CASE("ideal bytes: unknown by default") {
    IdealBytes b;
    CHECK_FALSE(b.known());
    CHECK(b.total() == doctest::Approx(0.0));
}

TEST_CASE("ideal bytes: reduce") {
    const auto b = ideal::reduce(1000, DType::FP32, DType::FP32);
    CHECK(b.read == doctest::Approx(4000.0));
    CHECK(b.write == doctest::Approx(4.0));
    CHECK(b.known());
    // fp16 in, fp32 out: the input type sets the read, the output the write
    const auto h = ideal::reduce(1000, DType::FP16, DType::FP32);
    CHECK(h.read == doctest::Approx(2000.0));
    CHECK(h.write == doctest::Approx(4.0));
    CHECK(h.total() == doctest::Approx(arena::dtype_buffer_bytes(1000, DType::FP16) +
                                       arena::dtype_buffer_bytes(1, DType::FP32)));
}

TEST_CASE("ideal bytes: scan") {
    const auto b = ideal::scan(4096);
    CHECK(b.read == doctest::Approx(4096.0 * 4));
    CHECK(b.write == doctest::Approx(4096.0 * 4));
    CHECK(b.total() == doctest::Approx(2.0 * 4096 * 4));
}

TEST_CASE("ideal bytes: softmax") {
    const auto b = ideal::softmax(64, 1024);
    CHECK(b.read == doctest::Approx(64.0 * 1024 * 4));
    CHECK(b.write == doctest::Approx(64.0 * 1024 * 4));
    CHECK(b.total() == doctest::Approx(2.0 * 64 * 1024 * 4));
}

TEST_CASE("ideal bytes: layernorm") {
    const auto b = ideal::layernorm(64, 1024);
    CHECK(b.read == doctest::Approx(270336.0));    // x plus gamma and beta
    CHECK(b.write == doctest::Approx(262144.0));   // y
    CHECK(b.total() == doctest::Approx((2.0 * 64 * 1024 + 2.0 * 1024) * 4));
    // the large shapes do not overflow
    const auto big = ideal::layernorm(16384, 4096);
    CHECK(big.write == doctest::Approx(16384.0 * 4096 * 4));
}

TEST_CASE("ideal bytes: matmul") {
    const auto b = ideal::matmul(256, 128, 512);   // M, K, N
    CHECK(b.read == doctest::Approx((256.0 * 128 + 128.0 * 512) * 4));
    CHECK(b.write == doctest::Approx(256.0 * 512 * 4));
}

TEST_CASE("ideal bytes: attention") {
    const auto b = ideal::attention(16, 1024, 64);
    const double t = 16.0 * 1024 * 64 * 4;
    CHECK(b.read == doctest::Approx(3.0 * t));
    CHECK(b.write == doctest::Approx(t));
    CHECK(b.total() == doctest::Approx(4.0 * t));
}

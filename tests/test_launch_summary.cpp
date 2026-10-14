#include <doctest/doctest.h>
#include "arena/measurement/launch_summary.hpp"

using namespace arena;

namespace {
KernelLaunchRecord rec(const char* name, uint64_t start, uint64_t end, int regs = 0, int shmem = 0) {
    KernelLaunchRecord r;
    r.name = name;
    r.start_ns = start;
    r.end_ns = end;
    r.registers_per_thread = regs;
    r.shared_memory = shmem;
    return r;
}
}

TEST_CASE("summarize_launches") {
    SUBCASE("empty") {
        auto s = summarize_launches({});
        CHECK(s.launch_count == 0);
        CHECK(s.kernel_time_ms == doctest::Approx(0.0f));
        CHECK(s.sub_kernels.empty());
    }

    SUBCASE("repeated launches of one kernel are all counted") {
        // The old map keyed on name kept only the last of these, reporting
        // one launch of 1 us instead of three totalling 6 us.
        auto s = summarize_launches({
            rec("k", 0, 3000), rec("k", 5000, 7000), rec("k", 9000, 10000)});
        CHECK(s.launch_count == 3);
        CHECK(s.kernel_time_ms == doctest::Approx(0.006f));
        CHECK(s.sub_kernels.size() == 3);
    }

    SUBCASE("sub-kernels come out in start order with offsets") {
        auto s = summarize_launches({
            rec("final", 20000, 21000), rec("blocks", 1000, 19000)});
        REQUIRE(s.sub_kernels.size() == 2);
        CHECK(s.sub_kernels[0].name == "blocks");
        CHECK(s.sub_kernels[0].start_offset_ms == doctest::Approx(0.0f));
        CHECK(s.sub_kernels[1].name == "final");
        CHECK(s.sub_kernels[1].start_offset_ms == doctest::Approx(0.019f));
    }

    SUBCASE("gaps between launches are not kernel time") {
        auto s = summarize_launches({rec("a", 0, 1000), rec("b", 100000, 101000)});
        CHECK(s.kernel_time_ms == doctest::Approx(0.002f));
    }

    SUBCASE("metadata comes from the longest launch") {
        auto s = summarize_launches({
            rec("short", 0, 100, 16, 0), rec("long", 200, 5200, 64, 4096)});
        CHECK(s.registers_per_thread == 64);
        CHECK(s.shared_memory_per_block == 4096);
    }

    SUBCASE("a tie keeps the first launch") {
        auto s = summarize_launches({rec("a", 0, 100, 10), rec("b", 200, 300, 20)});
        CHECK(s.registers_per_thread == 10);
    }

    SUBCASE("a malformed record with end before start counts as zero") {
        auto s = summarize_launches({rec("bad", 500, 100)});
        CHECK(s.launch_count == 1);
        CHECK(s.kernel_time_ms == doctest::Approx(0.0f));
    }
}

TEST_CASE("select_counter_op") {
    const std::vector<bool> additive = {true, true, false, false};

    SUBCASE("single-launch operations: skip the cold first, keep the minimum") {
        std::vector<std::vector<double>> r = {
            {86, 1, 0.5, 1.0},   // cold first launch, skipped
            {4.7, 1, 0.6, 1.1},
            {0.05, 1, 0.7, 1.2},
            {85, 1, 0.8, 1.3},
        };
        auto v = select_counter_op(r, 1, additive);
        REQUIRE(v.size() == 4);
        CHECK(v[0] == doctest::Approx(0.05));
        CHECK(v[2] == doctest::Approx(0.7));
    }

    SUBCASE("multi-launch operations are summed, not split") {
        // Two-stage reduce: a big stage 1, then a one-block stage 2 that reads
        // almost nothing. Per-range minimum used to report the stage-2 range.
        std::vector<std::vector<double>> r = {
            {268, 0.1, 0.90, 2.0}, {0.0, 0.0, 0.01, 0.1},   // op 0, cold
            {268, 0.1, 0.92, 2.1}, {0.0, 0.0, 0.01, 0.1},   // op 1
            {270, 0.2, 0.91, 2.2}, {0.0, 0.0, 0.01, 0.1},   // op 2
        };
        auto v = select_counter_op(r, 2, additive);
        REQUIRE(v.size() == 4);
        CHECK(v[0] == doctest::Approx(268.0));
        CHECK(v[1] == doctest::Approx(0.1));
        // rates come from the dominant launch, not the tiny one
        CHECK(v[2] == doctest::Approx(0.92));
        CHECK(v[3] == doctest::Approx(2.1));
    }

    SUBCASE("a trailing partial operation is ignored") {
        std::vector<std::vector<double>> r = {
            {10, 0, 0.5, 1}, {1, 0, 0.1, 1},
            {9, 0, 0.5, 1},  {1, 0, 0.1, 1},
            {0.5, 0, 0.5, 1},                 // incomplete op 2
        };
        auto v = select_counter_op(r, 2, additive);
        CHECK(v[0] == doctest::Approx(10.0));
    }

    SUBCASE("only one complete operation falls back to it") {
        auto v = select_counter_op({{50, 1, 0.4, 1}}, 1, additive);
        CHECK(v[0] == doctest::Approx(50.0));
    }

    SUBCASE("nothing to select") {
        CHECK(select_counter_op({}, 1, additive).empty());
        CHECK(select_counter_op({{1, 1, 1, 1}}, 2, additive).empty());
    }
}

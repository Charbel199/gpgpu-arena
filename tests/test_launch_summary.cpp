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

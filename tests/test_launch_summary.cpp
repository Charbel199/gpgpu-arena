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

TEST_CASE("short_kernel_name") {
    // extern "C" names are not mangled and pass through
    CHECK(short_kernel_name("matmul_naive") == "matmul_naive");
    // void cutlass::Kernel2<Foo>(Foo::Params)
    CHECK(short_kernel_name("_ZN7cutlass7Kernel2I3FooEEvNT_6ParamsE") == "cutlass::Kernel2<Foo>");
    // foo(int, float*)
    CHECK(short_kernel_name("_Z3fooiPf") == "foo");
    // not valid mangling: left alone
    CHECK(short_kernel_name("_Zgarbage") == "_Zgarbage");

    // the cuBLAS SIMT kernel this was written for
    CHECK(short_kernel_name("_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_256x128_8x4_nn_align1EEvNT_6ParamsE")
          == "cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>");
    // template function: return type dropped, value argument kept
    CHECK(short_kernel_name("_Z6reduceILi256EEvPKfPfi") == "reduce<256>");
    // spaces and commas inside nested template arguments stay
    CHECK(short_kernel_name("_Z2nkI1PIiS0_IfcEELi4EEvT_PKf") == "nk<P<int, P<float, char> >, 4>");
    // parentheses inside a template argument are not the parameter list
    CHECK(short_kernel_name("_Z2tkIFviEEvPT_i") == "tk<void (int)>");
    // anonymous namespace: its space is inside parentheses
    CHECK(short_kernel_name("_ZN12_GLOBAL__N_14kernEPf") == "(anonymous namespace)::kern");
    // a lambda as template argument (extended device lambdas look like this)
    CHECK(short_kernel_name("_ZN12_GLOBAL__N_12lkIZ3usevEUlifE_EEvT_")
          == "(anonymous namespace)::lk<use()::{lambda(int, float)#1}>");
    // Triton, cuTile and Warp entry points are plain C names
    CHECK(short_kernel_name("triton_poi_fused_add_0") == "triton_poi_fused_add_0");
    CHECK(short_kernel_name("wp_add_kernel_cuda_kernel_forward") == "wp_add_kernel_cuda_kernel_forward");
    CHECK(short_kernel_name("") == "");
    CHECK(short_kernel_name("_Z") == "_Z");
}

TEST_CASE("group_launched_kernels") {
    std::vector<SubKernelInfo> launches;
    auto add = [&](const char* n, float ms) {
        SubKernelInfo sk;
        sk.name = n;
        sk.duration_ms = ms;
        launches.push_back(sk);
    };
    // two operations, each running "small" twice and "big" once
    add("small", 0.1f); add("big", 1.0f); add("small", 0.1f);
    add("small", 0.1f); add("big", 3.0f); add("small", 0.1f);

    auto g = group_launched_kernels(launches, 2);
    REQUIRE(g.size() == 2);
    CHECK(g[0].name == "big");
    CHECK(g[0].count_per_op == 1);
    CHECK(g[0].gpu_ms == doctest::Approx(2.0f));
    CHECK(g[1].name == "small");
    CHECK(g[1].count_per_op == 2);
    CHECK(g[1].gpu_ms == doctest::Approx(0.2f));

    CHECK(group_launched_kernels({}).empty());

    SUBCASE("mangled names group under their short name") {
        std::vector<SubKernelInfo> l;
        auto push = [&](const char* n, float ms) {
            SubKernelInfo sk; sk.name = n; sk.duration_ms = ms; l.push_back(sk);
        };
        push("_Z3fooiPf", 1.0f);
        push("_Z3foofPf", 2.0f);   // overload, same short name
        auto k = group_launched_kernels(l);
        REQUIRE(k.size() == 1);
        CHECK(k[0].name == "foo");
        CHECK(k[0].count_per_op == 2);
        CHECK(k[0].gpu_ms == doctest::Approx(3.0f));
    }

    SUBCASE("a kernel seen in only some operations keeps one launch") {
        std::vector<SubKernelInfo> l;
        SubKernelInfo a; a.name = "main"; a.duration_ms = 1.0f;
        SubKernelInfo b; b.name = "init"; b.duration_ms = 0.5f;
        l = {b, a, a, a, a};   // init only on the first of four operations
        auto k = group_launched_kernels(l, 4);
        REQUIRE(k.size() == 2);
        CHECK(k[0].name == "main");
        CHECK(k[0].count_per_op == 1);
        CHECK(k[0].gpu_ms == doctest::Approx(1.0f));
        CHECK(k[1].name == "init");
        CHECK(k[1].count_per_op == 1);
        CHECK(k[1].gpu_ms == doctest::Approx(0.125f));
    }

    SUBCASE("ops below one reads as one") {
        auto k = group_launched_kernels(launches, 0);
        REQUIRE(k.size() == 2);
        CHECK(k[1].count_per_op == 4);
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

    SUBCASE("rates can come from the longest launch instead of the most DRAM") {
        // An L2-resident scan: a 1 us init kernel, then the scan. DRAM bytes
        // are a few KB of noise either way; metric 4 is the launch time.
        const std::vector<bool> add5 = {true, true, false, false, true};
        std::vector<std::vector<double>> r = {
            {2816, 0, 7.9, 0.04, 1000}, {4e6, 0, 24.6, 0.10, 9000},   // op 0, cold
            {768, 0, 7.9, 0.04, 1000},  {256, 0, 24.3, 0.09, 9000},   // op 1
        };
        auto by_dram = select_counter_op(r, 2, add5);
        CHECK(by_dram[2] == doctest::Approx(7.9));    // the init kernel
        auto by_time = select_counter_op(r, 2, add5, 4);
        CHECK(by_time[0] == doctest::Approx(1024.0)); // same operation picked
        CHECK(by_time[2] == doctest::Approx(24.3));
        CHECK(by_time[3] == doctest::Approx(0.09));
        CHECK(by_time[4] == doctest::Approx(10000.0));
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

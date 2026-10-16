#include <doctest/doctest.h>
#include "arena/cli_args.hpp"

#include <map>
#include <string>
#include <vector>

using namespace arena::cli;

TEST_CASE("parse_param") {
    SUBCASE("normal key=value") {
        auto kv = parse_param("n=1000000");
        REQUIRE(kv.has_value());
        CHECK(kv->first == "n");
        CHECK(kv->second == 1000000);
    }
    SUBCASE("zero is valid") {
        auto kv = parse_param("n=0");
        REQUIRE(kv.has_value());
        CHECK(kv->second == 0);
    }
    SUBCASE("multi-character key") {
        auto kv = parse_param("rows=512");
        REQUIRE(kv.has_value());
        CHECK(kv->first == "rows");
        CHECK(kv->second == 512);
    }
    SUBCASE("missing equals is rejected") {
        CHECK_FALSE(parse_param("n1000").has_value());
    }
    SUBCASE("empty key is rejected") {
        CHECK_FALSE(parse_param("=1000").has_value());
    }
    SUBCASE("empty value is rejected") {
        CHECK_FALSE(parse_param("n=").has_value());
    }
    SUBCASE("non-numeric value is rejected") {
        CHECK_FALSE(parse_param("n=abc").has_value());
    }
    SUBCASE("trailing junk is rejected, not silently truncated") {
        // strtol would happily return 1000 here; that would turn a typo into
        // a silently wrong problem size.
        CHECK_FALSE(parse_param("n=1000x").has_value());
        CHECK_FALSE(parse_param("n=1000.5").has_value());
    }
    SUBCASE("negative is rejected") {
        CHECK_FALSE(parse_param("n=-5").has_value());
    }
}

TEST_CASE("apply_params") {
    SUBCASE("accumulates across calls") {
        std::map<std::string, int> p;
        CHECK(apply_params("M=256", p));
        CHECK(apply_params("N=512", p));
        CHECK(p.size() == 2);
        CHECK(p["M"] == 256);
        CHECK(p["N"] == 512);
    }
    SUBCASE("later value wins") {
        std::map<std::string, int> p;
        CHECK(apply_params("n=1", p));
        CHECK(apply_params("n=2", p));
        CHECK(p["n"] == 2);
    }
    SUBCASE("malformed spec leaves the map untouched") {
        std::map<std::string, int> p{{"n", 42}};
        CHECK_FALSE(apply_params("bogus", p));
        CHECK(p.size() == 1);
        CHECK(p["n"] == 42);
    }
}

TEST_CASE("tuning_variants_for") {
    const std::vector<int> blocks{64, 128, 256, 512, 1024};
    const std::vector<std::map<std::string, int>> defines{
        {{"BLOCK_SIZE", 256}}, {{"BLOCK_SIZE", 1024}},
    };
    const std::map<std::string, int> none{};

    SUBCASE("no sweep gives one run at the descriptor default") {
        auto v = tuning_variants_for(false, 0, none, blocks, {});
        REQUIRE(v.size() == 1);
        CHECK(v[0].block_size == 0);
        CHECK(v[0].defines.empty());
    }
    SUBCASE("no sweep honours an explicit --block") {
        auto v = tuning_variants_for(false, 256, none, blocks, {});
        REQUIRE(v.size() == 1);
        CHECK(v[0].block_size == 256);
    }
    SUBCASE("no sweep honours explicit --define") {
        const std::map<std::string, int> d{{"BLOCK_SIZE", 512}};
        auto v = tuning_variants_for(false, 0, d, {}, defines);
        REQUIRE(v.size() == 1);
        CHECK(v[0].defines == d);
    }
    SUBCASE("sweep expands over block sizes for a CUDA kernel") {
        auto v = tuning_variants_for(true, 0, none, blocks, {});
        REQUIRE(v.size() == blocks.size());
        for (size_t i = 0; i < blocks.size(); i++) CHECK(v[i].block_size == blocks[i]);
    }
    SUBCASE("sweep expands over compile configs for a DSL kernel") {
        // A DSL kernel declares no tunable block sizes, because changing its
        // block means recompiling rather than relaunching.
        auto v = tuning_variants_for(true, 0, none, {}, defines);
        REQUIRE(v.size() == 2);
        CHECK(v[0].defines.at("BLOCK_SIZE") == 256);
        CHECK(v[1].defines.at("BLOCK_SIZE") == 1024);
    }
    SUBCASE("block sizes win when a kernel somehow declares both") {
        auto v = tuning_variants_for(true, 0, none, blocks, defines);
        CHECK(v.size() == blocks.size());
    }
    SUBCASE("a kernel with nothing tunable still runs once") {
        // Returning an empty list here would silently drop it from the sweep.
        auto v = tuning_variants_for(true, 0, none, {}, {});
        REQUIRE(v.size() == 1);
        CHECK(v[0].block_size == 0);
    }
}

TEST_CASE("split_csv") {
    CHECK(split_csv("a,b,c") == std::vector<std::string>{"a", "b", "c"});
    CHECK(split_csv("single") == std::vector<std::string>{"single"});
    CHECK(split_csv("a,,b,") == std::vector<std::string>{"a", "b"});
    CHECK(split_csv("").empty());
}

TEST_CASE("parse_param_list") {
    SUBCASE("single value is a one-element list") {
        auto kv = parse_param_list("n=4096");
        REQUIRE(kv.has_value());
        CHECK(kv->first == "n");
        CHECK(kv->second == std::vector<int>{4096});
    }
    SUBCASE("several values keep their order") {
        auto kv = parse_param_list("n=1000003,256,67108864");
        REQUIRE(kv.has_value());
        CHECK(kv->second == std::vector<int>{1000003, 256, 67108864});
    }
    SUBCASE("empty items are malformed rather than skipped") {
        // A stray comma is a typo, and silently running fewer sizes than
        // asked for would be worse than refusing.
        CHECK_FALSE(parse_param_list("n=1,,2").has_value());
        CHECK_FALSE(parse_param_list("n=1,").has_value());
        CHECK_FALSE(parse_param_list("n=,1").has_value());
    }
    SUBCASE("any bad value rejects the whole list") {
        CHECK_FALSE(parse_param_list("n=1,abc").has_value());
        CHECK_FALSE(parse_param_list("n=1,-5").has_value());
    }
    SUBCASE("missing key or values") {
        CHECK_FALSE(parse_param_list("=1,2").has_value());
        CHECK_FALSE(parse_param_list("n=").has_value());
        CHECK_FALSE(parse_param_list("n").has_value());
    }
}

TEST_CASE("apply_param_list") {
    std::map<std::string, std::vector<int>> m;
    REQUIRE(apply_param_list("M=128,256", m));
    REQUIRE(apply_param_list("N=64", m));
    CHECK(m["M"] == std::vector<int>{128, 256});
    CHECK(m["N"] == std::vector<int>{64});

    SUBCASE("a later spec for the same key replaces it") {
        REQUIRE(apply_param_list("M=512", m));
        CHECK(m["M"] == std::vector<int>{512});
    }
    SUBCASE("a malformed spec leaves the map untouched") {
        auto before = m;
        CHECK_FALSE(apply_param_list("M=1,x", m));
        CHECK(m == before);
    }
}

TEST_CASE("parse_u64_list") {
    auto v = parse_u64_list("1,2,18446744073709551615");
    REQUIRE(v.has_value());
    CHECK(v->size() == 3);
    CHECK((*v)[2] == 18446744073709551615ULL);

    CHECK_FALSE(parse_u64_list("").has_value());
    CHECK_FALSE(parse_u64_list("1,x").has_value());
    CHECK_FALSE(parse_u64_list("-1").has_value());
    CHECK_FALSE(parse_u64_list("7z").has_value());
}

TEST_CASE("expand_grid") {
    SUBCASE("no parameters still runs once") {
        auto g = expand_grid({});
        REQUIRE(g.size() == 1);
        CHECK(g[0].empty());
    }
    SUBCASE("one parameter, several values") {
        auto g = expand_grid({{"n", {1, 2, 3}}});
        REQUIRE(g.size() == 3);
        CHECK(g[0].at("n") == 1);
        CHECK(g[2].at("n") == 3);
    }
    SUBCASE("cartesian product, last key fastest") {
        auto g = expand_grid({{"M", {1, 2}}, {"N", {10, 20, 30}}});
        REQUIRE(g.size() == 6);
        CHECK(g[0] == std::map<std::string, int>{{"M", 1}, {"N", 10}});
        CHECK(g[1] == std::map<std::string, int>{{"M", 1}, {"N", 20}});
        CHECK(g[3] == std::map<std::string, int>{{"M", 2}, {"N", 10}});
        CHECK(g[5] == std::map<std::string, int>{{"M", 2}, {"N", 30}});
    }
    SUBCASE("an empty value list yields nothing") {
        CHECK(expand_grid({{"n", {}}}).empty());
    }
}

TEST_CASE("parse_shape") {
    auto s = parse_shape("rows=64,cols=1024");
    REQUIRE(s.has_value());
    CHECK(s->size() == 2);
    CHECK(s->at("rows") == 64);
    CHECK(s->at("cols") == 1024);
    CHECK(parse_shape("n=7")->at("n") == 7);
    CHECK_FALSE(parse_shape("").has_value());
    CHECK_FALSE(parse_shape("rows=64,rows=8").has_value());
    CHECK_FALSE(parse_shape("rows=64,cols").has_value());
    CHECK_FALSE(parse_shape("rows=-1").has_value());
}

TEST_CASE("parse_scratch_policy") {
    CHECK(parse_scratch_policy("legacy") == arena::ScratchPolicy::Legacy);
    CHECK(parse_scratch_policy("declared") == arena::ScratchPolicy::Declared);
    CHECK_FALSE(parse_scratch_policy("").has_value());
    CHECK_FALSE(parse_scratch_policy("Declared").has_value());
    CHECK_FALSE(parse_scratch_policy("zero").has_value());
    CHECK(std::string(arena::scratch_policy_name(arena::ScratchPolicy::Declared)) == "declared");
    CHECK(std::string(arena::scratch_policy_name(arena::RunConfig{}.scratch_policy)) == "legacy");
}

TEST_CASE("plan_args_error") {
    CHECK(plan_args_error(true, false, false, false).empty());
    CHECK(plan_args_error(false, false, false, false).find("--manifest") != std::string::npos);
    CHECK_FALSE(plan_args_error(true, true, false, false).empty());
    CHECK_FALSE(plan_args_error(true, false, true, false).empty());
    CHECK_FALSE(plan_args_error(true, false, false, true).empty());
}

TEST_CASE("parse_lock_clocks") {
    CHECK(parse_lock_clocks("2400") == 2400u);
    CHECK(parse_lock_clocks("auto") == 0u);
    CHECK_FALSE(parse_lock_clocks("").has_value());
    CHECK_FALSE(parse_lock_clocks("0").has_value());
    CHECK_FALSE(parse_lock_clocks("-5").has_value());
    CHECK_FALSE(parse_lock_clocks("24x").has_value());
    CHECK_FALSE(parse_lock_clocks("AUTO").has_value());
}

TEST_CASE("parse_timing_mode") {
    CHECK(parse_timing_mode("v1") == arena::TimingMode::V1);
    CHECK(parse_timing_mode("v2") == arena::TimingMode::V2);
    CHECK_FALSE(parse_timing_mode("").has_value());
    CHECK_FALSE(parse_timing_mode("V2").has_value());
    CHECK_FALSE(parse_timing_mode("v3").has_value());
}

TEST_CASE("take_timing_mode") {
    SUBCASE("removes the pair and keeps the rest in order") {
        char a0[] = "arena", a1[] = "--run", a2[] = "k", a3[] = "--timing-mode",
             a4[] = "v2", a5[] = "--runs", a6[] = "5";
        char* argv[] = {a0, a1, a2, a3, a4, a5, a6, nullptr};
        int argc = 7;
        auto v = take_timing_mode(argc, argv);
        REQUIRE(v.has_value());
        CHECK(*v == "v2");
        REQUIRE(argc == 5);
        CHECK(std::string(argv[1]) == "--run");
        CHECK(std::string(argv[2]) == "k");
        CHECK(std::string(argv[3]) == "--runs");
        CHECK(std::string(argv[4]) == "5");
        CHECK(argv[5] == nullptr);
    }
    SUBCASE("absent leaves argv alone") {
        char a0[] = "arena", a1[] = "--list";
        char* argv[] = {a0, a1, nullptr};
        int argc = 2;
        CHECK_FALSE(take_timing_mode(argc, argv).has_value());
        CHECK(argc == 2);
    }
    SUBCASE("the last one wins") {
        char a0[] = "arena", a1[] = "--timing-mode", a2[] = "v2", a3[] = "--timing-mode", a4[] = "v1";
        char* argv[] = {a0, a1, a2, a3, a4, nullptr};
        int argc = 5;
        CHECK(take_timing_mode(argc, argv) == std::string("v1"));
        CHECK(argc == 1);
    }
    SUBCASE("a trailing flag without a value is left for the CLI to reject") {
        char a0[] = "arena", a1[] = "--timing-mode";
        char* argv[] = {a0, a1, nullptr};
        int argc = 2;
        CHECK_FALSE(take_timing_mode(argc, argv).has_value());
        CHECK(argc == 2);
    }
}

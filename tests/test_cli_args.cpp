#include <doctest/doctest.h>
#include "arena/cli_args.hpp"

#include <map>
#include <string>

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

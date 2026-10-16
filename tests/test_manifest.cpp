#include <doctest/doctest.h>
#include "arena/manifest/expr.hpp"
#include "arena/manifest/args.hpp"
#include "arena/warp_abi.hpp"

#include <cstring>

using namespace arena;
using namespace arena::manifest;
using nlohmann::json;

namespace {
template <typename T>
T as(const PackedArgs& p, size_t i) {
    T v{};
    REQUIRE(p.bytes(i).size() == sizeof(T));
    std::memcpy(&v, p.bytes(i).data(), sizeof(T));
    return v;
}
}

TEST_CASE("manifest expressions") {
    const Vars v = {{"rows", 4096}, {"cols", 4097}, {"ROWS_PER_PROGRAM", 4}, {"sm_count", 188}};

    CHECK(eval("42", v) == 42);
    CHECK(eval("rows", v) == 4096);
    CHECK(eval("2 + 3 * 4", v) == 14);
    CHECK(eval("(2 + 3) * 4", v) == 20);
    CHECK(eval("-3 + 10", v) == 7);
    CHECK(eval("cols / 2", v) == 2048);
    CHECK(eval("cols % 2", v) == 1);
    CHECK(eval("cdiv(rows, ROWS_PER_PROGRAM)", v) == 1024);
    CHECK(eval("cdiv(cols, 1024)", v) == 5);
    CHECK(eval("min(sm_count * 4, rows)", v) == 752);
    CHECK(eval("max(1, 2, 3)", v) == 3);
    CHECK(eval("64 + 16 * cdiv(rows * cols, 32)", v) == 64 + 16 * ((4096LL * 4097 + 31) / 32));

    SUBCASE("large products do not overflow 32 bits") {
        CHECK(eval("65536 * 65536", v) == 4294967296LL);
    }
    SUBCASE("errors name the problem") {
        CHECK_THROWS_WITH(eval("rows / 0", v), doctest::Contains("division by zero"));
        CHECK_THROWS_WITH(eval("cdiv(rows, 0)", v), doctest::Contains("division by zero"));
        CHECK_THROWS_WITH(eval("nope + 1", v), doctest::Contains("unknown name 'nope'"));
        CHECK_THROWS_WITH(eval("system(1)", v), doctest::Contains("unknown function"));
        CHECK_THROWS_WITH(eval("(rows", v), doctest::Contains("missing ')'"));
        CHECK_THROWS_WITH(eval("rows rows", v), doctest::Contains("unexpected"));
        CHECK_THROWS(eval("", v));
        CHECK_THROWS(eval("cdiv(1)", v));
    }
    SUBCASE("json numbers and strings") {
        CHECK(eval_json(json(256), v) == 256);
        CHECK(eval_json(json("rows * 2"), v) == 8192);
        CHECK_THROWS(eval_json(json::array(), v));
    }
}

TEST_CASE("manifest argument packing") {
    ArgContext ctx;
    ctx.buffers = {{"input", 0x1000}, {"output", 0x2000}, {"ws", 0x3000}};
    ctx.vars = {{"rows", 4}, {"cols", 6}};

    SUBCASE("buffers, sizes and null") {
        auto p = pack_args(json::parse(R"(["input", "output", "rows", "cols", "null", "ws"])"), ctx);
        REQUIRE(p.size() == 6);
        CHECK(as<uint64_t>(p, 0) == 0x1000);
        CHECK(as<uint64_t>(p, 1) == 0x2000);
        CHECK(as<int32_t>(p, 2) == 4);
        CHECK(as<int32_t>(p, 3) == 6);
        CHECK(as<uint64_t>(p, 4) == 0);
        CHECK(as<uint64_t>(p, 5) == 0x3000);
        CHECK(p.pointers().size() == 6);
    }
    SUBCASE("explicit widths") {
        auto p = pack_args(json::parse(R"([{"i32": "rows * cols"}, {"u32": 7}, {"i64": "rows"}])"), ctx);
        CHECK(as<int32_t>(p, 0) == 24);
        CHECK(as<uint32_t>(p, 1) == 7);
        CHECK(as<int64_t>(p, 2) == 4);
    }
    SUBCASE("cuTile array expands to pointer, shapes, then strides") {
        auto p = pack_args(json::parse(R"([{"cutile_array": "input", "shape": ["rows", "cols"]}])"), ctx);
        REQUIRE(p.size() == 5);
        CHECK(as<uint64_t>(p, 0) == 0x1000);
        CHECK(as<uint32_t>(p, 1) == 4);
        CHECK(as<uint32_t>(p, 2) == 6);
        CHECK(as<uint32_t>(p, 3) == 6);   // row-major: stride of dim 0 is cols
        CHECK(as<uint32_t>(p, 4) == 1);
    }
    SUBCASE("Warp array is one 56-byte struct with byte strides") {
        auto p = pack_args(json::parse(R"([{"warp_array": "output", "shape": ["rows", "cols"]}])"), ctx);
        REQUIRE(p.size() == 1);
        const auto a = as<WarpArrayF32>(p, 0);
        CHECK(a.data == 0x2000);
        CHECK(a.grad == 0);
        CHECK(a.ndim == 2);
        CHECK(a.shape[0] == 4);
        CHECK(a.shape[1] == 6);
        CHECK(a.strides[0] == 24);
        CHECK(a.strides[1] == 4);
    }
    SUBCASE("Warp launch bounds") {
        auto p = pack_args(json::parse(R"([{"warp_bounds": "rows"}])"), ctx);
        const auto b = as<WarpLaunchBounds>(p, 0);
        CHECK(b.shape[0] == 4);
        CHECK(b.size == 4);
        CHECK(b.coord_mult == 1);
    }
    SUBCASE("padding to the kernel's parameter count") {
        ctx.param_sizes = {8, 8, 4, 4, 8, 8};
        auto p = pack_args(json::parse(R"(["input", "output", "rows", "cols", "..."])"), ctx);
        CHECK(p.size() == 6);
        CHECK(as<uint64_t>(p, 5) == 0);
    }
    SUBCASE("a layout that does not match the kernel is refused") {
        ctx.param_sizes = {8, 8, 4, 4};
        CHECK_THROWS_WITH(pack_args(json::parse(R"(["input", "output", "rows"])"), ctx),
                          doctest::Contains("takes 4 parameters but the manifest passes 3"));
        CHECK_THROWS_WITH(pack_args(json::parse(R"(["input", "output", "rows", {"i64": "cols"}])"), ctx),
                          doctest::Contains("argument 3 is 8 bytes"));
        CHECK_THROWS_WITH(pack_args(json::parse(R"(["input", {"warp_array": "output", "shape": [1]}, "rows", "cols"])"), ctx),
                          doctest::Contains("56 bytes"));
    }
    SUBCASE("described arguments, for --plan") {
        ctx.vars["BLOCK"] = 128;
        ctx.param_sizes = {8, 4, 4, 8, 8, 8, 4, 8, 4, 4, 4, 4, 56, 24, 8, 8};
        json d;
        pack_args(json::parse(R"J(["input", "rows", {"i32": "cdiv(cols, BLOCK)"}, "null", "ws",
                                   {"i64": 5}, {"u32": "rows"}, {"cutile_array": "output", "shape": ["rows", "cols"]},
                                   {"warp_array": "input", "shape": ["rows * cols"]}, {"warp_bounds": "rows"},
                                   "..."])J"), ctx, &d);
        REQUIRE(d.size() == 11);
        CHECK(d[0] == json{{"buffer", "input"}});
        CHECK(d[1] == json{{"name", "rows"}, {"type", "i32"}, {"value", 4}});
        CHECK(d[2] == json{{"type", "i32"}, {"expr", "cdiv(cols, BLOCK)"}, {"value", 1}});
        CHECK(d[3] == json{{"null", true}});
        CHECK(d[4] == json{{"buffer", "ws"}});
        CHECK(d[5] == json{{"type", "i64"}, {"expr", 5}, {"value", 5}});
        CHECK(d[6]["value"] == 4);
        CHECK(d[7] == json{{"cutile_array", "output"}, {"shape", {4, 6}}, {"strides", {6, 1}}});
        CHECK(d[8] == json{{"warp_array", "input"}, {"shape", {24}}, {"strides", {4}}});
        CHECK(d[9] == json{{"warp_bounds", 4}});
        CHECK(d[10] == json{{"padding", 2}});

        // reused for the next launch: replaced, not appended to
        ctx.param_sizes = {8, 8};
        pack_args(json::parse(R"(["ws", "..."])"), ctx, &d);
        REQUIRE(d.size() == 2);
        CHECK(d[0] == json{{"buffer", "ws"}});
        CHECK(d[1] == json{{"padding", 1}});
        ctx.param_sizes = {8, 4};
        pack_args(json::parse(R"(["ws", "rows", "..."])"), ctx, &d);
        CHECK(d[2] == json{{"padding", 0}});
    }
    SUBCASE("unknown names and kinds are errors, not zeros") {
        CHECK_THROWS_WITH(pack_args(json::parse(R"(["inptu"])"), ctx), doctest::Contains("unknown name"));
        CHECK_THROWS_WITH(pack_args(json::parse(R"([{"cutile_array": "nope", "shape": [1]}])"), ctx),
                          doctest::Contains("unknown buffer 'nope'"));
        CHECK_THROWS(pack_args(json::parse(R"([{"f32": 1}])"), ctx));
        CHECK_THROWS(pack_args(json::parse(R"(["...", "input"])"), ctx));
        CHECK_THROWS(pack_args(json::parse(R"({"not": "a list"})"), ctx));
        CHECK_THROWS(pack_args(json::parse(R"([{"cutile_array": "input", "shape": [1, 2], "strides": [1]}])"), ctx));
    }
}

TEST_CASE("manifest expressions: comparisons for when") {
    const Vars v = {{"rows", 4096}, {"cols", 1000}};
    CHECK(eval("rows * cols < 1048576", v) == 0);
    CHECK(eval("rows * cols >= 1048576", v) == 1);
    CHECK(eval("cols <= 1024 && rows > 1", v) == 1);
    CHECK(eval("cols > 1024 || rows == 4096", v) == 1);
    CHECK(eval("cols != 1000", v) == 0);
    CHECK(eval("!(cols < 1024)", v) == 0);
    CHECK(eval("1 + 2 < 4", v) == 1);           // arithmetic binds tighter
    CHECK(eval("(rows < cols) * 7 + 1", v) == 1);
    CHECK(eval("max(cols < 2000, 0)", v) == 1);
}

TEST_CASE("manifest expressions: conditional") {
    const Vars v = {{"M", 2048}, {"N", 2048}, {"a", 0}, {"b", 10}, {"c", 1}, {"d", 20}, {"e", 30}};
    CHECK(eval("M*N <= 2048*2048 ? 8 : 4", v) == 8);
    CHECK(eval("M*N < 2048*2048 ? 8 : 4", v) == 4);

    SUBCASE("right-associative") {
        CHECK(eval("a ? b : c ? d : e", v) == 20);        // a ? b : (c ? d : e)
        CHECK(eval("c ? b : a ? d : e", v) == 10);
        CHECK(eval("a ? b : a ? d : e", v) == 30);
        CHECK(eval("c ? a ? 1 : 2 : 3", v) == 2);         // nested in the middle
    }
    SUBCASE("lowest precedence") {
        CHECK(eval("1 + c ? 5 : 6", v) == 5);             // (1 + c) ? 5 : 6
        CHECK(eval("a || c ? 5 : 6", v) == 5);            // (a || c) ? 5 : 6
        CHECK(eval("c ? 2 : 3 + 4", v) == 2);             // c ? 2 : (3 + 4)
        CHECK(eval("a ? 2 : 3 + 4", v) == 7);
        CHECK(eval("(c ? 2 : 3) * 4", v) == 8);
        CHECK(eval("max(a ? 1 : 9, 3)", v) == 9);
    }
    SUBCASE("only the taken branch is evaluated") {
        CHECK(eval("a > 0 ? b / a : 0", v) == 0);
        CHECK(eval("a ? b % a : cdiv(b, a + 1)", v) == 10);
        CHECK_THROWS_WITH(eval("c ? b / a : 0", v), doctest::Contains("division by zero"));
        // && and || short-circuit the same way
        CHECK(eval("a > 0 && b / a > 1", v) == 0);
        CHECK(eval("a == 0 || b / a > 1", v) == 1);
        // a typo is still an error in the branch not taken
        CHECK_THROWS_WITH(eval("c ? 1 : nope", v), doctest::Contains("unknown name 'nope'"));
        CHECK_THROWS_WITH(eval("a ? nope : 1", v), doctest::Contains("unknown name 'nope'"));
        CHECK_THROWS_WITH(eval("a && nope", v), doctest::Contains("unknown name 'nope'"));
        CHECK_THROWS_WITH(eval("c || nope", v), doctest::Contains("unknown name 'nope'"));
        CHECK_THROWS_WITH(eval("a ? foo(1) : 1", v), doctest::Contains("unknown function 'foo'"));
        CHECK_THROWS_WITH(eval("a ? cdiv(1) : 1", v), doctest::Contains("cdiv takes 2 arguments"));
        CHECK_THROWS_WITH(eval("a && (b + )", v), doctest::Contains("unexpected ')'"));
        // inside an untaken branch nothing counts, even a nested taken one
        CHECK(eval("a ? (c ? b / a : 0) : 5", v) == 5);
        CHECK(eval("a && (c ? b % a : 1)", v) == 0);
        CHECK(eval("c || (c ? cdiv(b, a) : 1)", v) == 1);
        // ...and the skip ends with the branch
        CHECK_THROWS_WITH(eval("(a ? b / a : 1) + b / a", v), doctest::Contains("division by zero"));
        CHECK_THROWS_WITH(eval("(a && b / a) + b % a", v), doctest::Contains("modulo by zero"));
        // && binds tighter than ||, and both give 1 or 0
        CHECK(eval("c || a && b / a", v) == 1);
        CHECK(eval("b && d", v) == 1);
        CHECK(eval("a || b", v) == 1);
    }
    SUBCASE("errors") {
        CHECK_THROWS_WITH(eval("c ? 1", v), doctest::Contains("missing ':'"));
        CHECK_THROWS_WITH(eval("c ? 1 2", v), doctest::Contains("missing ':'"));
        CHECK_THROWS_WITH(eval("a ? b : c ? d", v), doctest::Contains("missing ':'"));
        CHECK_THROWS_WITH(eval("c ? : 2", v), doctest::Contains("unexpected ':'"));
        CHECK_THROWS_WITH(eval("c : 2", v), doctest::Contains("unexpected ':'"));
        CHECK_THROWS_WITH(eval("c ?", v), doctest::Contains("unexpected end"));
        CHECK_THROWS(eval("? 1 : 2", v));
    }
}

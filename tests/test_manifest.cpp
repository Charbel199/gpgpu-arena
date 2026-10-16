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

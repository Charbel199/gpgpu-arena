#include <doctest/doctest.h>
#include "arena/compilers/sass.hpp"

#include <string>

using namespace arena::sass;

namespace {

// Trimmed from `cuobjdump -sass` on a two-kernel cubin (sm_120, CUDA 13.1).
const char* kDump = R"SASS(
	code for sm_120
	.target	sm_120

		Function : reduce_sum_blocks
	.headerflags	@"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                                  /* 0x0000df00ff017b82 */
                                                                                            /* 0x000fe20000000800 */
        /*0010*/                   S2R R5, SR_TID.X ;                                       /* 0x0000000000057919 */
                                                                                            /* 0x000e2e0000002100 */
        /*0020*/                   LDG.E.128.CONSTANT R8, desc[UR4][R2.64] ;                /* 0x0000000402087981 */
        /*0030*/                   LDG.E.128.CONSTANT R12, desc[UR4][R2.64+0x10] ;          /* 0x0000100402097981 */
        /*0040*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                    /* 0x0000000402037981 */
        /*0050*/                   FADD R0, R8, R9 ;                                        /* 0x0000000908007221 */
        /*0060*/                   SHFL.DOWN PT, R4, R0, 0x10, 0x1f ;                       /* 0x0a001f0000047f89 */
        /*0070*/               @!P0 STS [UR4], R0 ;                                         /* 0x00000000ff008988 */
        /*0080*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;                            /* 0x0000000000007b1d */
        /*0090*/              @P1 REDG.E.ADD.F32.FTZ.RN.STRONG.GPU desc[UR4][R2.64], R0 ;  /* 0x000000000200798e */
        /*00a0*/                   EXIT ;                                                   /* 0x000000000000794d */
        /*00b0*/                   BRA 0xb0;                                                /* 0xfffffffc00fc7947 */
        /*00c0*/                   NOP;                                                     /* 0x0000000000007918 */
        /*00d0*/                   NOP;                                                     /* 0x0000000000007918 */
		..........

		Function : reduce_sum_final
	.headerflags	@"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                                  /* 0x0000df00ff017b82 */
        /*0010*/                   STG.E desc[UR4][R2.64], R0 ;                             /* 0x0000000002007986 */
        /*0020*/                   EXIT ;                                                   /* 0x000000000000794d */
		..........
)SASS";

const char* kResUsage = R"RES(
Resource usage:
 Common:
  GLOBAL:0
 Function reduce_sum_final:
  REG:12 STACK:0 SHARED:132 LOCAL:0 CONSTANT[0]:908 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function reduce_sum_blocks:
  REG:40 STACK:16 SHARED:132 LOCAL:24 CONSTANT[0]:916 TEXTURE:0 SURFACE:0 SAMPLER:0
)RES";

}

TEST_CASE("function_text") {
    SUBCASE("extracts one function and stops at the next") {
        auto t = function_text(kDump, "reduce_sum_blocks");
        CHECK(t.find("Function : reduce_sum_blocks") != std::string::npos);
        CHECK(t.find("SHFL.DOWN") != std::string::npos);
        CHECK(t.find("reduce_sum_final") == std::string::npos);
    }
    SUBCASE("last function runs to the end") {
        auto t = function_text(kDump, "reduce_sum_final");
        CHECK(t.find("STG.E") != std::string::npos);
        CHECK(t.find("SHFL") == std::string::npos);
    }
    SUBCASE("unknown function falls back to the whole dump") {
        CHECK(function_text(kDump, "nope") == std::string(kDump));
    }
    SUBCASE("a name that is a prefix of another does not match it") {
        // "reduce_sum" must not select "reduce_sum_blocks".
        CHECK(function_text(kDump, "reduce_sum") == std::string(kDump));
    }
}

TEST_CASE("opcode_histogram") {
    auto h = opcode_histogram(function_text(kDump, "reduce_sum_blocks"));

    CHECK(h["LDG.E.128.CONSTANT"] == 2);
    CHECK(h["LDG.E.CONSTANT"] == 1);
    CHECK(h["FADD"] == 1);
    CHECK(h["SHFL.DOWN"] == 1);
    CHECK(h["BAR.SYNC.DEFER_BLOCKING"] == 1);

    SUBCASE("predicates are stripped, so a guarded op counts as itself") {
        CHECK(h["STS"] == 1);
        CHECK(h["REDG.E.ADD.F32.FTZ.RN.STRONG.GPU"] == 1);
    }
    SUBCASE("NOP padding is not counted") {
        CHECK(h.count("NOP") == 0);
    }
    SUBCASE("encoding-only lines are not instructions") {
        int total = 0;
        for (const auto& [op, n] : h) total += n;
        CHECK(total == 12);
    }
}

TEST_CASE("classify") {
    auto c = classify(opcode_histogram(function_text(kDump, "reduce_sum_blocks")));
    CHECK(c["global_load_128"] == 2);
    CHECK(c["global_load_32"] == 1);
    CHECK(c["global_atomic"] == 1);
    CHECK(c["warp_shuffle"] == 1);
    CHECK(c["shared_mem"] == 1);
    CHECK(c["sync"] == 1);
    CHECK(c["fp32"] == 1);
    CHECK(c["control"] == 2);

    SUBCASE("every instruction lands in exactly one class") {
        auto h = opcode_histogram(kDump);
        int by_op = 0, by_class = 0;
        for (const auto& [op, n] : h) by_op += n;
        for (const auto& [cls, n] : classify(h)) by_class += n;
        CHECK(by_op == by_class);
    }
    SUBCASE("unknown opcodes go to other") {
        auto o = classify({{"FUTUREOP.X", 3}});
        CHECK(o["other"] == 3);
    }
    SUBCASE("a .64 store is not mistaken for a .128 one") {
        auto o = classify({{"STG.E.64", 1}, {"STG.E.128", 2}});
        CHECK(o["global_store_64"] == 1);
        CHECK(o["global_store_128"] == 2);
    }
}

TEST_CASE("mean_global_load_bytes") {
    auto h = opcode_histogram(function_text(kDump, "reduce_sum_blocks"));
    // two 16-byte loads and one 4-byte load
    CHECK(mean_global_load_bytes(h) == doctest::Approx(36.0 / 3.0));
    CHECK(mean_global_load_bytes({}) == doctest::Approx(0.0));
    CHECK(mean_global_load_bytes({{"LDG.E.64", 2}}) == doctest::Approx(8.0));
    CHECK(mean_global_load_bytes({{"LDG.E.U16", 1}}) == doctest::Approx(2.0));
}

TEST_CASE("parse_resource_usage") {
    SUBCASE("picks the named function") {
        auto r = parse_resource_usage(kResUsage, "reduce_sum_blocks");
        REQUIRE(r.found);
        CHECK(r.registers == 40);
        CHECK(r.stack == 16);
        CHECK(r.shared == 132);
        CHECK(r.local == 24);
        CHECK(r.constant0 == 916);
    }
    SUBCASE("falls back to the first function") {
        auto r = parse_resource_usage(kResUsage, "missing");
        REQUIRE(r.found);
        CHECK(r.registers == 12);
    }
    SUBCASE("nothing to parse") {
        CHECK_FALSE(parse_resource_usage("garbage", "x").found);
    }
}

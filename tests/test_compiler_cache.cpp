#include <doctest/doctest.h>
#include "arena/compilers/cache_key.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include "arena/compilers/kernel_compiler.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace arena;

namespace {

void ensure_logger() {
    if (!spdlog::get("compiler")) spdlog::null_logger_mt("compiler");
}

// A scratch directory removed at the end of the test.
struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<int> counter{0};
        path = fs::temp_directory_path() /
               ("arena_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

void write(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

// Stands in for nvcc: writes a cubin whose bytes are the source text, and
// counts how often it was asked to compile.
struct FakeBackend : CompilerBackend {
    int* compiles;
    std::string* toolchain;
    std::vector<std::string> deps;

    FakeBackend(int* c, std::string* t, std::vector<std::string> d = {})
        : compiles(c), toolchain(t), deps(std::move(d)) {}

    CompileResult compile(const std::string& source_path, const std::string& output_name,
                          const std::string& cache_dir, const CompileDefines&) override {
        (*compiles)++;
        auto cubin = cache_dir + "/" + output_name + ".cubin";
        std::ofstream(cubin) << read_file(source_path);
        CompileResult r;
        r.module_path = cubin;
        r.kernel_name = "k";
        return r;
    }
    std::string toolchain_id() override { return *toolchain; }
    std::vector<std::string> dependencies() const override { return deps; }
};

}

TEST_CASE("fnv1a64 and hex64") {
    // Reference values for 64-bit FNV-1a.
    CHECK(fnv1a64("") == 0xcbf29ce484222325ULL);
    CHECK(fnv1a64("a") == 0xaf63dc4c8601ec8cULL);
    CHECK(hex64(0xabcULL) == "0000000000000abc");
    CHECK(hex64(fnv1a64("foobar")).size() == 16);
}

TEST_CASE("build_fingerprint") {
    BuildInputs base;
    base.source_text = "__global__ void k() {}";
    base.dependency_texts = {"base module"};
    base.toolchain = "nvcc 13.1 -arch=native";
    base.arch = "sm_120";
    base.defines = {{"BLOCK", 256}};
    const auto fp = build_fingerprint(base);

    CHECK(build_fingerprint(base) == fp);

    SUBCASE("each input changes it") {
        auto v = base; v.source_text += " ";
        CHECK(build_fingerprint(v) != fp);
        v = base; v.dependency_texts[0] = "edited base";
        CHECK(build_fingerprint(v) != fp);
        v = base; v.toolchain = "nvcc 13.2 -arch=native";
        CHECK(build_fingerprint(v) != fp);
        v = base; v.arch = "sm_90";
        CHECK(build_fingerprint(v) != fp);
        v = base; v.defines["BLOCK"] = 512;
        CHECK(build_fingerprint(v) != fp);
        v = base; v.defines.clear();
        CHECK(build_fingerprint(v) != fp);
    }
    SUBCASE("moving bytes between fields does not collide") {
        BuildInputs a; a.source_text = "ab"; a.toolchain = "c";
        BuildInputs b; b.source_text = "a";  b.toolchain = "bc";
        CHECK(build_fingerprint(a) != build_fingerprint(b));
    }
}

TEST_CASE("derive_output_name and get_extension") {
    CHECK(KernelCompiler::derive_output_name("reduce/fp32/baseline.cu") == "reduce_fp32_baseline");
    CHECK(KernelCompiler::derive_output_name("reduce/fp32/reduce.triton.py") ==
          "reduce_fp32_triton_reduce");
    CHECK(KernelCompiler::derive_output_name("reduce/fp16/accum.cu") == "reduce_fp16_accum");

    SUBCASE("absolute paths get a short path hash instead of the full path") {
        auto a = KernelCompiler::derive_output_name("/tmp/run1/cand.cu");
        auto b = KernelCompiler::derive_output_name("/tmp/run2/cand.cu");
        CHECK(a.rfind("ext_", 0) == 0);
        CHECK(a.find("_cand") != std::string::npos);
        CHECK(a.find('/') == std::string::npos);
        CHECK(a != b);
    }

    CHECK(KernelCompiler::get_extension("reduce/fp32/reduce.triton.py") == ".triton.py");
    CHECK(KernelCompiler::get_extension("reduce/fp32/baseline.cu") == ".cu");
    CHECK(KernelCompiler::get_extension("/abs/x.cutile.py") == ".cutile.py");
    CHECK(KernelCompiler::get_extension("noext") == "");
}

TEST_CASE("compiler_utils helpers") {
    CHECK(resolve_source("/k", "reduce/a.cu") == "/k/reduce/a.cu");
    CHECK(resolve_source("/k", "/abs/a.cu") == "/abs/a.cu");
    CHECK(shell_quote("plain") == "'plain'");
    CHECK(shell_quote("it's") == "'it'\\''s'");
    CHECK(read_file("/definitely/not/here").empty());
}

TEST_CASE("KernelCompiler cache") {
    ensure_logger();
    TempDir tmp;
    const auto src = (tmp.path / "src" / "k.cu").string();
    const auto dep = (tmp.path / "src" / "base.txt").string();
    const auto cache = (tmp.path / "cache").string();
    write(src, "version 1");
    write(dep, "base 1");

    int compiles = 0;
    std::string toolchain = "fake 1.0";

    auto make = [&](const std::string& arch) {
        KernelCompiler c(cache);
        c.set_target_arch(arch);
        c.register_compiler(".cu", std::make_unique<FakeBackend>(
            &compiles, &toolchain, std::vector<std::string>{dep}));
        return c;
    };

    auto c = make("sm_120");
    auto r1 = c.compile(src);
    CHECK(compiles == 1);
    CHECK_FALSE(r1.cache_hit);
    CHECK(read_file(r1.module_path) == "version 1");

    SUBCASE("same inputs hit memory, then disk in a fresh compiler") {
        CHECK(c.compile(src).cache_hit);
        auto fresh = make("sm_120");
        auto r = fresh.compile(src);
        CHECK(r.cache_hit);
        CHECK(compiles == 1);
    }

    SUBCASE("an edited source rebuilds, even with the same mtime") {
        // mtime used to be the whole key. Restoring it after an edit, as
        // git checkout or a copy with preserved times can, kept the old cubin.
        const auto t = fs::last_write_time(src);
        write(src, "version 2");
        fs::last_write_time(src, t);

        auto r = c.compile(src);
        CHECK_FALSE(r.cache_hit);
        CHECK(compiles == 2);
        CHECK(read_file(r.module_path) == "version 2");
    }

    SUBCASE("an edited dependency rebuilds") {
        write(dep, "base 2");
        CHECK_FALSE(c.compile(src).cache_hit);
        CHECK(compiles == 2);
    }

    SUBCASE("a new toolchain rebuilds") {
        toolchain = "fake 2.0";
        auto fresh = make("sm_120");
        CHECK_FALSE(fresh.compile(src).cache_hit);
        CHECK(compiles == 2);
    }

    SUBCASE("another GPU arch rebuilds") {
        auto other = make("sm_90");
        CHECK_FALSE(other.compile(src).cache_hit);
        CHECK(compiles == 2);
    }

    SUBCASE("defines are cached separately") {
        auto a = c.compile(src, {{"BLOCK", 128}});
        auto b = c.compile(src, {{"BLOCK", 256}});
        CHECK(compiles == 3);
        CHECK(a.module_path != b.module_path);
        CHECK(c.compile(src, {{"BLOCK", 128}}).cache_hit);
        CHECK(compiles == 3);
    }

    SUBCASE("a variant path carries its defines") {
        const auto v = KernelCompiler::variant_path(src, {{"ROWS", 4}, {"BLOCK", 256}});
        CHECK(v == src + "|BLOCK=256,ROWS=4");
        CHECK(KernelCompiler::variant_path(src, {}) == src);
        const auto [path, d] = KernelCompiler::split_variant(v);
        CHECK(path == src);
        CHECK(d == CompileDefines{{"BLOCK", 256}, {"ROWS", 4}});
        CHECK(KernelCompiler::split_variant(src).second.empty());
        CHECK(KernelCompiler::get_extension(v) == ".cu");
        CHECK(KernelCompiler::derive_output_name(v) ==
              KernelCompiler::derive_output_name(src) + "_BLOCK256_ROWS4");

        // the same build as passing the defines, and cached with it
        auto a = c.compile(v);
        CHECK(compiles == 2);
        auto b = c.compile(src, {{"BLOCK", 256}, {"ROWS", 4}});
        CHECK(b.cache_hit);
        CHECK(a.module_path == b.module_path);
        // a variant's own defines win over the ones passed
        CHECK(c.compile(v, {{"BLOCK", 128}}).cache_hit);
        CHECK(compiles == 2);
        CHECK(c.compile(KernelCompiler::variant_path(src, {{"BLOCK", 128}})).module_path != a.module_path);
        CHECK(compiles == 3);
    }

    SUBCASE("metadata from before fingerprints is treated as stale") {
        auto meta = fs::path(cache) / (KernelCompiler::derive_output_name(src) + ".json");
        REQUIRE(fs::exists(meta));
        write(meta, R"({"kernel_name": "k", "source_mtime": 1})");
        auto fresh = make("sm_120");
        CHECK_FALSE(fresh.compile(src).cache_hit);
        CHECK(compiles == 2);
    }

    SUBCASE("a missing source is an error, not an empty build") {
        CHECK_THROWS(c.compile((tmp.path / "nope.cu").string()));
    }

    SUBCASE("unknown extension is an error") {
        write(tmp.path / "src" / "k.metal", "x");
        CHECK_THROWS(c.compile((tmp.path / "src" / "k.metal").string()));
    }
}

TEST_CASE("parse_arena_directives") {
    SUBCASE("several keys on one line, several lines") {
        auto d = parse_arena_directives(
            "// arena: BLOCK_M=128 BLOCK_N=64\n"
            "#include <x>\n"
            "// arena: THREADS = 256\n");
        CHECK(d.size() == 3);
        CHECK(d["BLOCK_M"] == 128);
        CHECK(d["BLOCK_N"] == 64);
        CHECK(d["THREADS"] == 256);
    }
    SUBCASE("lower-case keys belong to pgaio and are ignored") {
        auto d = parse_arena_directives("// arena: block=512 TILE=4\n");
        CHECK(d.count("block") == 0);
        CHECK(d["TILE"] == 4);
    }
    SUBCASE("a later value wins") {
        CHECK(parse_arena_directives("// arena: X=1\n// arena: X=2\n")["X"] == 2);
    }
    SUBCASE("no directive, or malformed values") {
        CHECK(parse_arena_directives("int BLOCK_M = 128;\n").empty());
        CHECK(parse_arena_directives("// arena: BLOCK_M=-4 N=abc\n").empty());
        CHECK(parse_arena_directives("").empty());
    }
}

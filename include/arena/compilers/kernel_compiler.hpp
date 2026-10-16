#pragma once

#include "arena/compilers/backend.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>
#include <vector>

namespace arena {

// dispatcher routes to the right compiler by file extension, handles caching
class KernelCompiler {
public:
    explicit KernelCompiler(const std::string& cache_dir);

    void register_compiler(const std::string& extension, std::unique_ptr<CompilerBackend> compiler);

    // GPU the cubins are built for, e.g. "sm_120". Part of the cache key, so a
    // cache shared between machines (the Docker volume) cannot hand one GPU a
    // cubin built for another.
    void set_target_arch(const std::string& arch) { arch_ = arch; }

    // Compiles with the source's own defaults unless defines overrides them.
    // Each distinct set of defines is cached separately, on disk and in
    // memory, since it produces a different cubin.
    //
    // source_path is relative to the kernel dir, or absolute for a source
    // outside it.
    CompileResult compile(const std::string& source_path,
                          const CompileDefines& defines = {});

    // PTX for the same build, or "" when the DSL does not expose it. Compiles
    // first if needed, so the PTX always matches the cubin.
    std::string emit_ptx(const std::string& source_path,
                         const CompileDefines& defines = {});

    // Files a compiler wrote next to the cubin for this build, absolute: PTX
    // (Triton), TTGIR (Triton), the CUDA C++ Warp generated. Only those that
    // exist; CUDA PTX comes from emit_ptx() instead.
    std::vector<std::string> sidecars(const std::string& source_path,
                                      const CompileDefines& defines = {}) const;

    // The package a source's DSL needs but this environment lacks ("triton"),
    // or "" when it can be compiled here. A native build usually has CUDA but
    // not the Python DSLs, which only Docker installs.
    std::string missing_toolchain(const std::string& source_path);

    // clear all cached cubins and metadata (forces recompilation)
    void clear_cache();

    // "reduce/fp32/baseline.cu" -> "reduce_fp32_baseline". Public so tests and
    // the inspect command can predict cache file names. A variant path (below)
    // gets its defines suffix, as its cubin does.
    static std::string derive_output_name(const std::string& source_path);
    static std::string get_extension(const std::string& source_path);

    // A source path with defines folded in, "/abs/ln.cu|BLOCK=256,ROWS=4", for
    // callers that pass sources around as plain paths: a manifest launch with
    // its own "defines" is one of the descriptor's extra sources. compile(),
    // emit_ptx() and sidecars() take it in place of a path and apply its
    // defines over the ones passed. No defines gives the path unchanged.
    static std::string variant_path(const std::string& source_path, const CompileDefines& defines);
    static std::pair<std::string, CompileDefines> split_variant(const std::string& source_path);

private:
    CompilerBackend& backend_for(const std::string& source_path);
    std::string fingerprint(const std::string& source_path, const CompileDefines& defines);
    bool try_disk_cache(const std::string& output_name, const std::string& fingerprint,
                        CompileResult& result);
    void save_disk_cache(const std::string& source_path, const std::string& output_name,
                         const std::string& fingerprint, const CompileResult& result);

    std::string cache_dir_;
    std::string arch_;
    std::unordered_map<std::string, std::unique_ptr<CompilerBackend>> compilers_;
    std::unordered_map<std::string, CompileResult> cache_;
};

}

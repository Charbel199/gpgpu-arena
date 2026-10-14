#pragma once

#include "arena/compilers/backend.hpp"
#include <string>
#include <unordered_map>
#include <memory>

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

    // clear all cached cubins and metadata (forces recompilation)
    void clear_cache();

    // "reduce/fp32/baseline.cu" -> "reduce_fp32_baseline". Public so tests and
    // the inspect command can predict cache file names.
    static std::string derive_output_name(const std::string& source_path);
    static std::string get_extension(const std::string& source_path);

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

#pragma once

#include "arena/compilers/compiler.hpp"
#include <string>
#include <unordered_map>
#include <memory>

namespace arena {

// dispatcher routes to the right compiler by file extension, handles caching
class KernelCompiler {
public:
    explicit KernelCompiler(const std::string& cache_dir);

    void register_compiler(const std::string& extension, std::unique_ptr<Compiler> compiler);

    CompileResult compile(const std::string& source_path);

    // clear all cached cubins and metadata (forces recompilation)
    void clear_cache();

private:
    std::string derive_output_name(const std::string& source_path);
    std::string get_extension(const std::string& source_path);
    bool try_disk_cache(const std::string& source_path, const std::string& output_name,
                        CompileResult& result);
    void save_disk_cache(const std::string& source_path, const std::string& output_name,
                         const CompileResult& result);

    std::string cache_dir_;
    std::unordered_map<std::string, std::unique_ptr<Compiler>> compilers_;
    std::unordered_map<std::string, CompileResult> cache_;
};

}

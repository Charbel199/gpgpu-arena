#pragma once

#include <string>
#include <unordered_map>
#include <memory>

namespace arena {

struct CompileResult {
    std::string ptx_path;
    std::string kernel_name;
    int num_warps = 4;
    int shared_memory = 0;
    int num_params = 3;
};

// base class, each DSL implements its own compilation logic
class Compiler {
public:
    virtual ~Compiler() = default;
    virtual CompileResult compile(const std::string& source_path,
                                  const std::string& output_name,
                                  const std::string& cache_dir) = 0;
};

// triton DSL compiler (runs Python script, parses JSON metadata from stdout)
class TritonCompiler : public Compiler {
public:
    explicit TritonCompiler(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir) override;
private:
    std::string kernel_dir_;
};

// dispatcher routes to the right compiler by file extension, handles caching
class KernelCompiler {
public:
    explicit KernelCompiler(const std::string& cache_dir);

    void register_compiler(const std::string& extension, std::unique_ptr<Compiler> compiler);

    CompileResult compile(const std::string& source_path);

private:
    std::string derive_output_name(const std::string& source_path);
    std::string get_extension(const std::string& source_path);

    std::string cache_dir_;
    std::unordered_map<std::string, std::unique_ptr<Compiler>> compilers_;
    std::unordered_map<std::string, CompileResult> cache_;
};

}

#pragma once

#include "arena/compilers/compiler.hpp"
#include <string>

namespace arena {

// triton DSL compiler produces cubin via Triton's built-in compilation
class TritonCompiler : public Compiler {
public:
    explicit TritonCompiler(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir) override;
private:
    std::string kernel_dir_;
};

}

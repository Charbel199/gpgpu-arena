#pragma once

#include "arena/compilers/compiler.hpp"
#include <string>

namespace arena {

// CUDA C++ compiler - nvcc -cubin, parses kernel name from source
class CudaCompiler : public Compiler {
public:
    explicit CudaCompiler(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir) override;
private:
    std::string kernel_dir_;
};

}

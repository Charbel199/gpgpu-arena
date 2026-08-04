#pragma once

#include "arena/compilers/backend.hpp"
#include <string>

namespace arena {

// CUDA C++ compiler - nvcc -cubin, parses kernel name from source
class CudaBackend : public CompilerBackend {
public:
    explicit CudaBackend(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir) override;
private:
    std::string kernel_dir_;
};

}

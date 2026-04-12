#pragma once

#include "arena/compilers/compiler.hpp"
#include <string>

namespace arena {

// NVIDIA Warp compiler produces cubin via wp.Module
class WarpCompiler : public Compiler {
public:
    explicit WarpCompiler(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir) override;
private:
    std::string kernel_dir_;
};

}

#pragma once

#include "arena/compilers/compiler.hpp"
#include <string>

namespace arena {

// cuTile DSL compiler produces cubin via tileiras
class CuTileCompiler : public Compiler {
public:
    explicit CuTileCompiler(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir) override;
private:
    std::string kernel_dir_;
};

}

#pragma once

#include "arena/compilers/backend.hpp"
#include <string>

namespace arena {

// cuTile DSL compiler produces cubin via tileiras
class CuTileBackend : public CompilerBackend {
public:
    explicit CuTileBackend(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir,
                          const CompileDefines& defines) override;
private:
    std::string kernel_dir_;
};

}

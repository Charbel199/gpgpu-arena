#pragma once

#include "arena/compilers/backend.hpp"
#include <string>

namespace arena {

// NVIDIA Warp compiler produces cubin via wp.Module
class WarpBackend : public CompilerBackend {
public:
    explicit WarpBackend(const std::string& kernel_dir);
    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir,
                          const CompileDefines& defines) override;
private:
    std::string kernel_dir_;
};

}

#pragma once

#include "arena/compilers/python_backend.hpp"

namespace arena {

// NVIDIA Warp compiler, produces a cubin via wp.Module
class WarpBackend : public PythonBackend {
public:
    explicit WarpBackend(const std::string& kernel_dir)
        : PythonBackend(kernel_dir, "Warp", "warp-lang", "warp_base.py") {}
};

}

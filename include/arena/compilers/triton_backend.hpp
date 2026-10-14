#pragma once

#include "arena/compilers/python_backend.hpp"

namespace arena {

// Triton DSL compiler, produces a cubin via Triton's built-in compilation
class TritonBackend : public PythonBackend {
public:
    explicit TritonBackend(const std::string& kernel_dir)
        : PythonBackend(kernel_dir, "Triton", "triton", "triton_base.py") {}
};

}

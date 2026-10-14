#pragma once

#include "arena/compilers/python_backend.hpp"

namespace arena {

// cuTile DSL compiler, produces a cubin via tileiras
class CuTileBackend : public PythonBackend {
public:
    explicit CuTileBackend(const std::string& kernel_dir)
        : PythonBackend(kernel_dir, "cuTile", "cuda-tile", "cutile_base.py") {}
};

}

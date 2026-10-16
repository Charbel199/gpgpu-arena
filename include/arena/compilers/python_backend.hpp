#pragma once

#include "arena/compilers/backend.hpp"
#include <string>

namespace arena {

// Shared driver for the Python DSLs. Each one runs the kernel's own .py file,
// which imports <dsl>_base.py, compiles, writes a cubin and prints one line of
// JSON metadata. The DSLs differ only in what they are called, which pip
// package versions them and which base module they import.
class PythonBackend : public CompilerBackend {
public:
    PythonBackend(const std::string& kernel_dir, std::string label,
                  std::string package, std::string base_module);

    CompileResult compile(const std::string& source_path,
                          const std::string& output_name,
                          const std::string& cache_dir,
                          const CompileDefines& defines) override;

    // "<package> <version> <python>". Read with importlib.metadata, which
    // does not import the DSL, so it costs one fast subprocess per process.
    std::string toolchain_id() override;

    std::vector<std::string> dependencies() const override;

    // The base module writes <output>.ptx next to the cubin when the DSL
    // exposes it. Nothing is recompiled here.
    std::string emit_ptx(const std::string& source_path,
                         const std::string& output_name,
                         const std::string& cache_dir,
                         const CompileDefines& defines) override;

private:
    std::string kernel_dir_;
    std::string label_;
    std::string package_;
    std::string base_module_;
    std::string toolchain_;
};

}

#pragma once

#include <string>
#include <map>

namespace arena {

struct CompileResult {
    std::string module_path;    // cubin path loaded by cuModuleLoad
    std::string kernel_name;
    int num_warps = 4;
    int shared_memory = 0;
    int num_params = 3;
    int block_dim = 0;          // cuTile: threads per block (0 = not set)
    std::map<std::string, int> constants;
    bool  cache_hit  = false;
    float compile_ms = 0.0f;   // the compiler's own time, self-reported
    float import_ms  = 0.0f;   // Python import cost (DSL compilers only)
    float invoke_ms  = 0.0f;   // full subprocess wall time as seen from C++
};

// One backend per DSL. KernelCompiler owns the set of these and picks one by
// file extension; a backend only has to turn a source file into a cubin plus
// metadata. Caching, output naming and timing all live in KernelCompiler.
class CompilerBackend {
public:
    virtual ~CompilerBackend() = default;
    virtual CompileResult compile(const std::string& source_path,
                                  const std::string& output_name,
                                  const std::string& cache_dir) = 0;
};

}

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

// Compile-time knobs handed to the compiler, not to the launch. A DSL bakes
// its block size into the cubin, so tuning one means recompiling: Triton's
// BLOCK_SIZE and num_warps, cuTile's TILE_SIZE. Empty is the source's own
// defaults, which is what an ordinary run uses.
using CompileDefines = std::map<std::string, int>;

// A stable, filesystem-safe suffix for a set of defines, so two configs of the
// same source get two cubins instead of overwriting each other. Empty defines
// give an empty suffix, which keeps default builds at their old cache names.
std::string defines_suffix(const CompileDefines& defines);

// One backend per DSL. KernelCompiler owns the set of these and picks one by
// file extension; a backend only has to turn a source file into a cubin plus
// metadata. Caching, output naming and timing all live in KernelCompiler.
class CompilerBackend {
public:
    virtual ~CompilerBackend() = default;
    virtual CompileResult compile(const std::string& source_path,
                                  const std::string& output_name,
                                  const std::string& cache_dir,
                                  const CompileDefines& defines) = 0;
};

}

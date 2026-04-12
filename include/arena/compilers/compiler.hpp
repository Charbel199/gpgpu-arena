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
    std::map<std::string, int> constants;
};

// base class each DSL implements its own compilation logic
class Compiler {
public:
    virtual ~Compiler() = default;
    virtual CompileResult compile(const std::string& source_path,
                                  const std::string& output_name,
                                  const std::string& cache_dir) = 0;
};

}

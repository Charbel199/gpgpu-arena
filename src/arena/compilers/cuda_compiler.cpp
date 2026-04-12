#include "arena/compilers/cuda_compiler.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include <fstream>
#include <regex>

namespace arena {

CudaCompiler::CudaCompiler(const std::string& kernel_dir)
    : kernel_dir_(kernel_dir) {}

CompileResult CudaCompiler::compile(const std::string& source_path,
                                     const std::string& output_name,
                                     const std::string& cache_dir) {
    auto full_source = kernel_dir_ + "/" + source_path;
    auto cubin_path = cache_dir + "/" + output_name + ".cubin";

    std::string cmd =
        std::string(ARENA_NVCC) + " -cubin -arch=native"
        " -o " + cubin_path + " " + full_source +
        " 2>&1";

    run_command(cmd, "nvcc for " + source_path);

    // parse kernel name from source: find 'extern "C" __global__ void <name>('
    std::string kernel_name;
    {
        std::ifstream file(full_source);
        std::string line;
        std::regex kernel_re(R"(extern\s+"C"\s+__global__\s+\w+\s+(\w+)\s*\()");
        while (std::getline(file, line)) {
            std::smatch match;
            if (std::regex_search(line, match, kernel_re)) {
                kernel_name = match[1].str();
                break;
            }
        }
    }

    // count kernel params from cubin
    int num_params = 0;
    {
        auto dump = run_command("cuobjdump --dump-elf " + cubin_path + " 2>&1",
                                "cuobjdump for " + source_path);
        std::regex kparam_re("EIATTR_KPARAM_INFO");
        auto begin = std::sregex_iterator(dump.begin(), dump.end(), kparam_re);
        auto end = std::sregex_iterator();
        num_params = std::distance(begin, end);
    }

    CompileResult result;
    result.module_path = cubin_path;
    result.kernel_name = kernel_name;
    result.num_params = num_params;
    return result;
}

}

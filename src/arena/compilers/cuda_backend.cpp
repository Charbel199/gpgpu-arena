#include "arena/compilers/cuda_backend.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include <chrono>
#include <fstream>
#include <regex>

namespace arena {

CudaBackend::CudaBackend(const std::string& kernel_dir)
    : kernel_dir_(kernel_dir) {}

CompileResult CudaBackend::compile(const std::string& source_path,
                                     const std::string& output_name,
                                     const std::string& cache_dir,
                                     const CompileDefines& defines) {
    auto full_source = kernel_dir_ + "/" + source_path;
    auto cubin_path = cache_dir + "/" + output_name + ".cubin";

    std::string nvcc_defines;
    for (const auto& [key, val] : defines) {
        nvcc_defines += " -D" + key + "=" + std::to_string(val);
    }

    std::string cmd =
        std::string(ARENA_NVCC) + " -cubin -arch=native" + nvcc_defines +
        " -o " + cubin_path + " " + full_source +
        " 2>&1";

    auto nvcc_t0 = std::chrono::steady_clock::now();
    run_command(cmd, "nvcc for " + source_path);
    const float nvcc_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - nvcc_t0).count();

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
    result.compile_ms = nvcc_ms;   // for nvcc the subprocess is the compiler
    result.import_ms = 0.0f;
    return result;
}

}

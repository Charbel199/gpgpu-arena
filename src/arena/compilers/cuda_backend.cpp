#include "arena/compilers/cuda_backend.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include <chrono>
#include <fstream>
#include <regex>

namespace arena {

namespace {

// Flags shared by the cubin and the PTX build, so both describe one program.
// Kept in one place because they are part of the cache key.
const char* kNvccFlags = "-arch=native -lineinfo";

std::string nvcc_defines(const CompileDefines& defines) {
    std::string out;
    for (const auto& [key, val] : defines) {
        out += " -D" + key + "=" + std::to_string(val);
    }
    return out;
}

}

CudaBackend::CudaBackend(const std::string& kernel_dir)
    : kernel_dir_(kernel_dir) {}

std::string CudaBackend::toolchain_id() {
    if (toolchain_.empty()) {
        // The last line of `nvcc --version` is the build string, which changes
        // with every toolkit release, unlike the marketing version above it.
        std::string version;
        try {
            version = run_command(std::string(ARENA_NVCC) + " --version 2>&1", "nvcc --version");
        } catch (...) {
            version = "unknown";
        }
        while (!version.empty() && (version.back() == '\n' || version.back() == '\r'))
            version.pop_back();
        const auto nl = version.rfind('\n');
        if (nl != std::string::npos) version = version.substr(nl + 1);
        toolchain_ = "nvcc " + version + " " + kNvccFlags;
    }
    return toolchain_;
}

CompileResult CudaBackend::compile(const std::string& source_path,
                                     const std::string& output_name,
                                     const std::string& cache_dir,
                                     const CompileDefines& defines) {
    auto full_source = resolve_source(kernel_dir_, source_path);
    auto cubin_path = cache_dir + "/" + output_name + ".cubin";

    // -lineinfo costs nothing at run time and lets nvdisasm map SASS back to
    // source lines, which is what makes a SASS dump readable.
    std::string cmd =
        std::string(ARENA_NVCC) + " -cubin " + kNvccFlags + nvcc_defines(defines) +
        " -o " + shell_quote(cubin_path) + " " + shell_quote(full_source) +
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
        auto dump = run_command("cuobjdump --dump-elf " + shell_quote(cubin_path) + " 2>&1",
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

std::string CudaBackend::emit_ptx(const std::string& source_path,
                                  const std::string& output_name,
                                  const std::string& cache_dir,
                                  const CompileDefines& defines) {
    auto ptx_path = cache_dir + "/" + output_name + ".ptx";
    std::string cmd =
        std::string(ARENA_NVCC) + " -ptx " + kNvccFlags + nvcc_defines(defines) +
        " -o " + shell_quote(ptx_path) + " " +
        shell_quote(resolve_source(kernel_dir_, source_path)) + " 2>&1";
    run_command(cmd, "nvcc -ptx for " + source_path);
    return ptx_path;
}

}

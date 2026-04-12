#include "arena/kernel_loader.hpp"
#include "arena/utils.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>

namespace arena {

KernelLoader::~KernelLoader() {
    unload_all();
}

CUmodule KernelLoader::load_module(const std::string& path) {
    CUmodule module;
    
    // check if it's a PTX file (text) or CUBIN (binary, TODO: not implemented yet)
    bool is_ptx = path.find(".ptx") != std::string::npos;
    
    if (is_ptx) {
        // read PTX source and JIT compile with error logging
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("PTX file not found: " + path);
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        std::string ptx_source = ss.str();

        CUjit_option options[] = {
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER
        };

        char info_log[1024] = {0};
        char error_log[1024] = {0};

        void* option_values[] = {
            reinterpret_cast<void*>(sizeof(info_log)),
            info_log,
            reinterpret_cast<void*>(sizeof(error_log)),
            error_log
        };

        CUresult result = cuModuleLoadDataEx(&module, ptx_source.c_str(),
            4, options, option_values);

        if (result != CUDA_SUCCESS) {
            const char* err_str;
            cuGetErrorString(result, &err_str);
            throw std::runtime_error(
                "Failed to load PTX module: " + path +
                "\nCUDA error: " + (err_str ? err_str : "unknown") +
                "\nJIT error log: " + error_log
            );
        }
    } else {
        // load CUBIN directly
        check_cuda(cuModuleLoad(&module, path.c_str()), "cuModuleLoad");
    }
    
    loaded_modules_.push_back(module);
    spdlog::get("loader")->debug("Loaded module: {}", path);
    return module;
}

CUfunction KernelLoader::get_function(CUmodule module, const std::string& kernel_name) {
    CUfunction func;
    check_cuda(
        cuModuleGetFunction(&func, module, kernel_name.c_str()),
        ("cuModuleGetFunction: " + kernel_name).c_str()
    );
    return func;
}

KernelLoader::LaunchResult KernelLoader::launch(
    CUfunction func, 
    const LaunchConfig& config, 
    void** args
) {
    // TODO: I don't like this, we will probably only use the reference output result to check its validity
    // BenchmarkResult contains all of the meaningful benchmarking results, so might removed this struct and return values 
    LaunchResult result;
    result.result = cuLaunchKernel(
        func,
        config.grid_x, config.grid_y, config.grid_z,
        config.block_x, config.block_y, config.block_z,
        config.shared_mem_bytes,
        config.stream,
        args,
        nullptr
    );
    if (result.result != CUDA_SUCCESS) {
        const char* errstr;
        cuGetErrorString(result.result, &errstr);
        spdlog::get("loader")->error("cuLaunchKernel failed: {}", errstr);
    }
    return result;
}

void KernelLoader::unload_all() {
    for (auto& module : loaded_modules_) {
        cuModuleUnload(module);
    }
    loaded_modules_.clear();
}

}


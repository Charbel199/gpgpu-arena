#include "arena/kernel_loader.hpp"
#include "arena/utils.hpp"
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
        // load PTX and JIT compile
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
        
        CUresult result = cuModuleLoadDataEx(
            &module, 
            nullptr,
            0, nullptr, nullptr
        );
        result = cuModuleLoad(&module, path.c_str());
        
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to load PTX module: " + path + "\nError: " + error_log
            );
        }
    } else {
        // load CUBIN directly
        check_cuda(cuModuleLoad(&module, path.c_str()), "cuModuleLoad");
    }
    
    loaded_modules_.push_back(module);
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
        nullptr  // extra params
    );
    return result;
}

void KernelLoader::unload_all() {
    for (auto& module : loaded_modules_) {
        cuModuleUnload(module);
    }
    loaded_modules_.clear();
}

}


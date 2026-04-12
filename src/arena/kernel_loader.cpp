#include "arena/kernel_loader.hpp"
#include "arena/utils.hpp"
#include <spdlog/spdlog.h>

namespace arena {

KernelLoader::~KernelLoader() {
    unload_all();
}

CUmodule KernelLoader::load_module(const std::string& path) {
    auto log = spdlog::get("loader");
    log->debug("Loading module: {} ...", path);

    CUmodule module;
    CUresult result = cuModuleLoad(&module, path.c_str());
    if (result != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(result, &err_str);
        log->error("Failed to load module {}: {}", path, err_str ? err_str : "unknown");
        throw std::runtime_error(
            "Failed to load module: " + path +
            "\nCUDA error: " + (err_str ? err_str : "unknown"));
    }
    loaded_modules_.push_back(module);
    log->debug("Loaded module: {}", path);
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
    // opt into dynamic shared memory >48KB (required for large tile sizes)
    if (config.shared_mem_bytes > 48 * 1024) {
        cuFuncSetAttribute(func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            config.shared_mem_bytes);
    }

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
        spdlog::get("loader")->error("Kernel launch failed: {}", errstr ? errstr : "unknown");
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


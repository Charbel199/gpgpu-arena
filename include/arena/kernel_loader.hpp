#pragma once

#include <cuda.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace arena {

class KernelLoader {
public:
    KernelLoader() = default;
    ~KernelLoader();

    // Load a kernel module from PTX or CUBIN file
    // Returns a module handle for later reference
    CUmodule load_module(const std::string& path);

    // Get a kernel function from a loaded module
    CUfunction get_function(CUmodule module, const std::string& kernel_name);

    // Launch a kernel with timing
    struct LaunchConfig {
        unsigned int grid_x = 1, grid_y = 1, grid_z = 1;
        unsigned int block_x = 1, block_y = 1, block_z = 1;
        unsigned int shared_mem_bytes = 0;
        CUstream stream = nullptr;
    };

    struct LaunchResult {
        CUresult result;
    };

    LaunchResult launch(CUfunction func, const LaunchConfig& config, void** args);

    // Unload all modules
    void unload_all();

private:
    std::vector<CUmodule> loaded_modules_;
};

}


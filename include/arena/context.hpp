#pragma once

#include <cuda.h>
#include <string>
#include <vector>
#include <memory>

namespace arena {


class Context {
public:
    Context(int device_id = 0);
    ~Context();

    // Prevent copying (CUDA context is not copyable)
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Allocate device memory for kernel inputs/outputs
    CUdeviceptr allocate(size_t bytes);
    void free(CUdeviceptr ptr);

    // Copy data to/from device
    void copy_to_device(CUdeviceptr dst, const void* src, size_t bytes);
    void copy_to_host(void* dst, CUdeviceptr src, size_t bytes);

    // Get device properties
    std::string device_name() const { return device_name_; }
    int compute_capability_major() const { return cc_major_; }
    int compute_capability_minor() const { return cc_minor_; }
    size_t total_memory() const { return total_mem_; }

    // Access the raw CUDA context (for advanced usage)
    CUcontext handle() const { return context_; }

private:
    CUdevice device_;
    CUcontext context_;
    std::string device_name_;
    int cc_major_, cc_minor_;
    size_t total_mem_;
};

}


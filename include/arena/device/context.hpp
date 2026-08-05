#pragma once

#include <cuda.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

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

    // Device allocation accounting. Descriptors must allocate through this
    // class for the numbers to be meaningful.
    size_t bytes_allocated() const { return current_bytes_; }
    size_t peak_bytes() const { return peak_bytes_; }
    void   reset_peak() { peak_bytes_ = current_bytes_; }

    // Copy data to/from device
    void copy_to_device(CUdeviceptr dst, const void* src, size_t bytes);
    void copy_to_host(void* dst, CUdeviceptr src, size_t bytes);

    // Zeroes device memory without staging a host buffer of zeros.
    void zero_device(CUdeviceptr dst, size_t bytes);

    // Get device properties
    std::string device_name() const { return device_name_; }
    int compute_capability_major() const { return cc_major_; }
    int compute_capability_minor() const { return cc_minor_; }
    size_t total_memory() const { return total_mem_; }
    int sm_count() const { return sm_count_; }
    int clock_rate_khz() const { return clock_rate_khz_; }
    int memory_clock_khz() const { return memory_clock_khz_; }
    int memory_bus_width() const { return memory_bus_width_; }

    // destroy and recreate the CUDA context (recovers from sticky errors like illegal memory access)
    void reset();

    // Access the raw CUDA context (for advanced usage)
    CUcontext handle() const { return context_; }

private:
    CUdevice device_;
    CUcontext context_;
    std::string device_name_;
    int cc_major_, cc_minor_;
    int sm_count_;
    int clock_rate_khz_;
    int memory_clock_khz_;
    int memory_bus_width_;
    size_t total_mem_;

    size_t current_bytes_ = 0;
    size_t peak_bytes_ = 0;
    std::unordered_map<CUdeviceptr, size_t> allocation_sizes_;
};

}


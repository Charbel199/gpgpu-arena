#pragma once

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/profiler.hpp"
#include "arena/kernel_descriptor.hpp"
#include <string>
#include <vector>
#include <map>

namespace arena {

struct BenchmarkResult {
    std::string kernel_name;
    std::string category;
    std::string description;
    

    unsigned int grid_x = 0, grid_y = 0, grid_z = 0;
    unsigned int block_x = 0, block_y = 0, block_z = 0;
    unsigned int shared_mem_bytes = 0;
    

    float elapsed_ms = 0.0f;
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;
    

    int registers_per_thread = 0;
    int shared_memory_bytes = 0;
    
    // CUPTI Profiler API (from documentation)
    double achieved_occupancy = 0.0;
    double dram_read_gbps = 0.0;
    double dram_write_gbps = 0.0;
    

    bool verified = false;
    
    bool success = false;
    std::string error;
};


struct BenchmarkConfig {
    int warmup_runs = 10;
    int number_of_runs = 10;
    
    // dict of kernel parameters
    std::map<std::string, int> params;
};

class Benchmark {
public:
    Benchmark(Context& ctx, KernelLoader& loader, Profiler& profiler);

    BenchmarkResult run(KernelDescriptor& descriptor, const BenchmarkConfig& config);

    std::vector<BenchmarkResult> run_category(
        const std::string& category,
        const BenchmarkConfig& config
    );

    std::vector<std::string> get_categories() const;
    
    std::vector<KernelDescriptor*> get_kernels_by_category(const std::string& category) const;
    
    std::vector<KernelDescriptor*> get_all_kernels() const;

    const Context& context() const { return ctx_; }

private:
    Context& ctx_;
    KernelLoader& loader_;
    Profiler& profiler_;
};

}

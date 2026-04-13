#pragma once

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/benchmark.hpp"
#include "arena/profiler.hpp"
#include "arena/kernel_descriptor.hpp"
#include <string>
#include <vector>
#include <map>

namespace arena {

struct RunConfig {
    int warmup_runs = 10;
    int number_of_runs = 10;
    bool collect_metrics = false;

    std::map<std::string, int> params;
};

struct RunResult {
    std::string kernel_name;
    std::string category;
    std::string description;

    // launch type
    bool uses_module = false;       // true = cubin (driver API), false = compiled (runtime API, e.g. CUB)

    // launch config
    unsigned int grid_x = 0, grid_y = 0, grid_z = 0;
    unsigned int block_x = 0, block_y = 0, block_z = 0;
    unsigned int shared_mem_bytes = 0;

    // benchmark results (timing)
    float elapsed_ms = 0.0f;       // wall time (CUDA events, median over N runs)
    std::vector<float> all_times_ms;  // individual run times from benchmark
    float kernel_ms = 0.0f;        // GPU-only time (CUPTI Activity API, sum of all kernels)
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;

    // GPU sub-kernel breakdown (from activity API)
    std::vector<Profiler::SubKernelInfo> sub_kernels;

    // profile results (hardware counters)
    int registers_per_thread = 0;
    int shared_memory_bytes = 0;
    double achieved_occupancy = 0.0;
    double dram_read_gbps = 0.0;
    double dram_write_gbps = 0.0;
    double ipc = 0.0;

    bool verified = false;
    bool success = false;
    std::string error;
};

class Runner {
public:
    Runner(Context& ctx, KernelLoader& loader, KernelCompiler& compiler,
           Benchmark& benchmark, Profiler& profiler);

    RunResult run(KernelDescriptor& descriptor, const RunConfig& config);

    std::vector<RunResult> run_category(const std::string& category, const RunConfig& config);


    std::vector<std::string> get_categories() const;
    std::vector<KernelDescriptor*> get_kernels_by_category(const std::string& category) const;
    std::vector<KernelDescriptor*> get_all_kernels() const;

    const Context& context() const { return ctx_; }
    KernelCompiler& compiler() { return compiler_; }

private:
    Context& ctx_;
    KernelLoader& loader_;
    KernelCompiler& compiler_;
    Benchmark& benchmark_;
    Profiler& profiler_;
};

}

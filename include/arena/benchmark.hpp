#pragma once

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/profiler.hpp"
#include <string>
#include <vector>
#include <functional>

namespace arena {

struct KernelInfo {
    std::string name;
    std::string path;
};

struct BenchmarkResult {
    std::string kernel_name;
    float elapsed_ms = 0.0f;
    double gflops = 0.0;
    double achieved_occupancy = 0.0;
    double dram_read_gbps = 0.0;
    double dram_write_gbps = 0.0;
    bool success = false;
    std::string error;
};

struct BenchmarkConfig {
    int matrix_size = 1024;
    int warmup_runs = 10;
};

class Benchmark {
public:
    Benchmark(Context& ctx, KernelLoader& loader, Profiler& profiler);

    // Scan for available kernels in a directory
    std::vector<KernelInfo> scan_kernels(const std::string& dir = "kernels");

    // Run a single kernel benchmark
    BenchmarkResult run(const KernelInfo& kernel, const BenchmarkConfig& config);

    // Run multiple kernels
    std::vector<BenchmarkResult> run_all(const std::vector<KernelInfo>& kernels, 
                                          const BenchmarkConfig& config);

    // Access context (for device info display)
    const Context& context() const { return ctx_; }

private:
    Context& ctx_;
    KernelLoader& loader_;
    Profiler& profiler_;
};

}

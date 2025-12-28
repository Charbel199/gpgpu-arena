#include "arena/benchmark.hpp"
#include <filesystem>
#include <algorithm>

namespace arena {

Benchmark::Benchmark(Context& ctx, KernelLoader& loader, Profiler& profiler)
    : ctx_(ctx), loader_(loader), profiler_(profiler) {}

std::vector<KernelInfo> Benchmark::scan_kernels(const std::string& dir) {
    std::vector<KernelInfo> kernels;

    if (!std::filesystem::exists(dir)) {
        return kernels;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".ptx") {
            KernelInfo info;
            info.name = entry.path().stem().string();
            info.path = entry.path().string();
            kernels.push_back(info);
        }
    }

    std::sort(kernels.begin(), kernels.end(),
        [](const auto& a, const auto& b) { return a.name < b.name; });

    return kernels;
}

BenchmarkResult Benchmark::run(const KernelInfo& kernel, const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.kernel_name = kernel.name;

    const int M = config.matrix_size;
    const int K = config.matrix_size;
    const int N = config.matrix_size;

    try {
        CUmodule module = loader_.load_module(kernel.path);
        CUfunction func = loader_.get_function(module, "matmul"); //TODO: MAke this dynamic

        // Allocate
        size_t size_a = M * K * sizeof(float);
        size_t size_b = K * N * sizeof(float);
        size_t size_c = M * N * sizeof(float);

        CUdeviceptr d_a = ctx_.allocate(size_a);
        CUdeviceptr d_b = ctx_.allocate(size_b);
        CUdeviceptr d_c = ctx_.allocate(size_c);

        // Initialize
        std::vector<float> h_a(M * K, 1.0f);
        std::vector<float> h_b(K * N, 1.0f);
        ctx_.copy_to_device(d_a, h_a.data(), size_a);
        ctx_.copy_to_device(d_b, h_b.data(), size_b);
        
        // Distributing benchmark config to all the components
        // Launch config
        KernelLoader::LaunchConfig launch_config;
        launch_config.block_x = 16;
        launch_config.block_y = 16;
        launch_config.grid_x = (N + launch_config.block_x - 1) / launch_config.block_x;
        launch_config.grid_y = (M + launch_config.block_y - 1) / launch_config.block_y;
        // PRofiler config
        Profiler::ProfilerConfig profiler_config;
        profiler_config.number_of_runs = config.number_of_runs;

        void* args[] = { &d_a, &d_b, &d_c, (void*)&M, (void*)&K, (void*)&N };

        // Warmup
        for (int i = 0; i < config.warmup_runs; i++) {
            loader_.launch(func, launch_config, args);
        }

        // Benchmark, pass the profiler config to the profiler
        auto metrics = profiler_.profile([&]() {
            loader_.launch(func, launch_config, args);
        }, profiler_config);

        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;
        result.elapsed_ms = metrics.elapsed_ms;
        result.gflops = (flops / (metrics.elapsed_ms / 1000.0)) / 1e9;
        
        // From CUPTI Activity API
        result.registers_per_thread = metrics.registers_per_thread;
        result.shared_memory_bytes = metrics.shared_memory_per_block;
        
        // From CUPTI Profiler API (Phase 2)
        result.achieved_occupancy = metrics.achieved_occupancy;
        result.dram_read_gbps = metrics.dram_read_throughput_gbps;
        result.dram_write_gbps = metrics.dram_write_throughput_gbps;
        
        result.success = true;

        // Cleanup
        ctx_.free(d_a);
        ctx_.free(d_b);
        ctx_.free(d_c);

    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
    }

    return result;
}

std::vector<BenchmarkResult> Benchmark::run_all(
    const std::vector<KernelInfo>& kernels,
    const BenchmarkConfig& config
) {
    std::vector<BenchmarkResult> results;
    for (const auto& kernel : kernels) {
        results.push_back(run(kernel, config));
    }
    return results;
}

}


#include "arena/benchmark.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace arena {

Benchmark::Benchmark(Context& ctx, KernelLoader& loader, Profiler& profiler)
    : ctx_(ctx), loader_(loader), profiler_(profiler) {}

BenchmarkResult Benchmark::run(KernelDescriptor& desc, const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.kernel_name = desc.name();
    result.category = desc.category();
    result.description = desc.description();

    spdlog::get("benchmark")->info("Benchmarking {} [{}]", result.kernel_name, result.category);

    try {
        desc.set_problem_size(config.params);

        CUmodule module = nullptr;
        CUfunction func = nullptr;
        if (desc.uses_ptx()) {
            module = loader_.load_module(desc.ptx_path());
            func = loader_.get_function(module, desc.function_name());
        }
        desc.allocate(ctx_);
        desc.initialize(ctx_);

        auto launch_config = desc.get_launch_config();
        spdlog::get("benchmark")->debug("Launch config: grid=({},{},{}), block=({},{},{}), shmem={} B",
            launch_config.grid_x, launch_config.grid_y, launch_config.grid_z,
            launch_config.block_x, launch_config.block_y, launch_config.block_z,
            launch_config.shared_mem_bytes);
        
        result.grid_x = launch_config.grid_x;
        result.grid_y = launch_config.grid_y;
        result.grid_z = launch_config.grid_z;
        result.block_x = launch_config.block_x;
        result.block_y = launch_config.block_y;
        result.block_z = launch_config.block_z;
        result.shared_mem_bytes = launch_config.shared_mem_bytes;

        auto args = desc.get_kernel_args();

        // Warmup
        for (int i = 0; i < config.warmup_runs; i++) {
            if (desc.uses_ptx()) {
                loader_.launch(func, launch_config, args.data());
            } else {
                desc.execute(ctx_);
            }
        }
        // TODO: This works but very inefficient
        desc.cleanup(ctx_);
        desc.allocate(ctx_);
        desc.initialize(ctx_);
        
        // profile multiple runs with reset between runs
        std::vector<float> times;
        times.reserve(config.number_of_runs);

        Profiler::ProfilerConfig profiler_config;

        Profiler::KernelMetrics metrics;
        for (int i = 0; i < config.number_of_runs; i++) {
            // reset output before each run
            desc.initialize(ctx_);

            auto run_metrics = profiler_.profile([&]() {
                if (desc.uses_ptx()) {
                    loader_.launch(func, launch_config, args.data());
                } else {
                    desc.execute(ctx_);
                }
            }, profiler_config);

            // ensure kernel completes before next iteration
            cuCtxSynchronize();

            times.push_back(run_metrics.elapsed_ms);
            metrics = run_metrics; // Keep last metrics for register/shmem info
        }

        // compute median time (better than mean for outlier resistance TODO: double check best way to benchmark)
        std::sort(times.begin(), times.end());
        metrics.elapsed_ms = times[times.size() / 2];

        double flops = desc.calculate_flops();
        double bytes = desc.calculate_bytes_accessed();
        
        result.elapsed_ms = metrics.elapsed_ms;
        result.gflops = (flops / (metrics.elapsed_ms / 1000.0)) / 1e9;
        result.bandwidth_gbps = (bytes / (metrics.elapsed_ms / 1000.0)) / 1e9;

        result.registers_per_thread = metrics.registers_per_thread;
        result.shared_memory_bytes = metrics.shared_memory_per_block;
        result.achieved_occupancy = metrics.achieved_occupancy;
        result.dram_read_gbps = metrics.dram_read_throughput_gbps;
        result.dram_write_gbps = metrics.dram_write_throughput_gbps;

        result.verified = desc.verify(ctx_);
        if (!result.verified) {
            spdlog::get("benchmark")->warn("{} failed verification", result.kernel_name);
        }
        desc.cleanup(ctx_);
        result.success = true;

        spdlog::get("benchmark")->debug("{}: {:.3f} ms, {:.2f} GFLOPS, {:.2f} GB/s",
            result.kernel_name, result.elapsed_ms, result.gflops, result.bandwidth_gbps);

    } catch (const std::exception& e) {
        spdlog::get("benchmark")->error("{} failed: {}", result.kernel_name, e.what());
        result.success = false;
        result.error = e.what();
        try { desc.cleanup(ctx_); } catch (...) {}
    }

    return result;
}

std::vector<BenchmarkResult> Benchmark::run_category(
    const std::string& category,
    const BenchmarkConfig& config
) {
    std::vector<BenchmarkResult> results;
    for (auto* kernel : get_kernels_by_category(category)) {
        results.push_back(run(*kernel, config));
    }
    return results;
}


// TODO: Move to kernel registry as internal functions
std::vector<std::string> Benchmark::get_categories() const {
    return KernelRegistry::instance().get_categories();
}

std::vector<KernelDescriptor*> Benchmark::get_kernels_by_category(const std::string& category) const {
    return KernelRegistry::instance().get_by_category(category);
}

std::vector<KernelDescriptor*> Benchmark::get_all_kernels() const {
    return KernelRegistry::instance().get_all();
}

}

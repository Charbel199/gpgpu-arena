#include "arena/runner.hpp"
#include "arena/utils.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <nvtx3/nvToolsExt.h>

namespace arena {

Runner::Runner(Context& ctx, KernelLoader& loader, Benchmark& benchmark, Profiler& profiler)
    : ctx_(ctx), loader_(loader), benchmark_(benchmark), profiler_(profiler) {}

RunResult Runner::run(KernelDescriptor& desc, const RunConfig& config) {
    RunResult result;
    result.kernel_name = desc.name();
    result.category = desc.category();
    result.description = desc.description();

    auto log = spdlog::get("runner");

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
        log->debug("Launch config: grid=({},{},{}), block=({},{},{}), shmem={} B",
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

        auto launch_kernel = [&]() {
            if (desc.uses_ptx()) {
                loader_.launch(func, launch_config, args.data());
            } else {
                desc.execute(ctx_);
            }
        };

        // warmup
        log->debug("Warmup: {} runs", config.warmup_runs);
        for (int i = 0; i < config.warmup_runs; i++) {
            launch_kernel();
        }
        check_cuda(cuCtxSynchronize(), "warmup sync");
        desc.cleanup(ctx_);
        desc.allocate(ctx_);
        desc.initialize(ctx_);

        // benchmark
        log->info("[BENCHMARK] {} - {} runs", result.kernel_name, config.number_of_runs);

        nvtxRangePushA(("BENCHMARK: " + result.kernel_name).c_str());
        auto bench_result = benchmark_.run(launch_kernel, config.number_of_runs,
            [&]() { desc.initialize(ctx_); }); // TODO: too complicated, why not simply pass in the DESCRIPTOR + LOADER maybe and internally clean up
        nvtxRangePop();

        result.elapsed_ms = bench_result.median_ms;

        // GPU-only kernel time via Activity API (single run, sums all sub-kernels)
        desc.initialize(ctx_);
        auto activity = profiler_.collect_activity(launch_kernel);
        result.kernel_ms = activity.kernel_time_ms;
        result.sub_kernels = activity.sub_kernels;
        result.uses_ptx = desc.uses_ptx();

        double flops = desc.calculate_flops();
        double bytes = desc.calculate_bytes_accessed();
        result.gflops = (flops / (result.elapsed_ms / 1000.0)) / 1e9;
        result.bandwidth_gbps = (bytes / (result.elapsed_ms / 1000.0)) / 1e9;

        log->info("[BENCHMARK] {} - wall={:.3f} ms kernel={:.3f} ms | {:.2f} GFLOPS | {:.2f} GB/s",
            result.kernel_name, result.elapsed_ms, result.kernel_ms, result.gflops, result.bandwidth_gbps);

        // profile
        if (config.collect_metrics) {
            log->info("[PROFILER]  {} - collecting hardware counters", result.kernel_name);
            nvtxRangePushA(("PROFILER: " + result.kernel_name).c_str());
            desc.initialize(ctx_);

            auto profile_result = profiler_.profile(launch_kernel,
                [&]() { desc.initialize(ctx_); });

            result.registers_per_thread = profile_result.registers_per_thread;
            result.shared_memory_bytes = profile_result.shared_memory_per_block;

            auto& mv = profile_result.metric_values;
            if (mv.count(metric::OCCUPANCY)) {
                result.achieved_occupancy = mv.at(metric::OCCUPANCY) / 100.0;
            }
            if (mv.count(metric::DRAM_READ)) {
                result.dram_read_gbps = (mv.at(metric::DRAM_READ) / (result.elapsed_ms / 1000.0)) / 1e9;
            }
            if (mv.count(metric::DRAM_WRITE)) {
                result.dram_write_gbps = (mv.at(metric::DRAM_WRITE) / (result.elapsed_ms / 1000.0)) / 1e9;
            }
            if (mv.count(metric::IPC)) {
                result.ipc = mv.at(metric::IPC);
            }

            log->info("[PROFILER]  {} - {} regs | {} B shmem | occupancy={:.1f}% | DRAM R={:.2f} W={:.2f} GB/s | IPC={:.2f}",
                result.kernel_name, result.registers_per_thread, result.shared_memory_bytes,
                result.achieved_occupancy * 100.0,
                result.dram_read_gbps, result.dram_write_gbps, result.ipc);
            nvtxRangePop();
        }

        // verification (per kernel)
        result.verified = desc.verify(ctx_);
        if (!result.verified) {
            log->warn("{} failed verification", result.kernel_name);
        }
        desc.cleanup(ctx_);
        result.success = true;

    } catch (const std::exception& e) {
        log->error("{} failed: {}", result.kernel_name, e.what());
        result.success = false;
        result.error = e.what();
        try { desc.cleanup(ctx_); } catch (...) {}

        // reset context in case of error
        try {
            CUresult ctx_status = cuCtxSynchronize();
            if (ctx_status != CUDA_SUCCESS) {
                loader_.unload_all();
                ctx_.reset();
            }
        } catch (...) {
            log->error("Failed to recover CUDA context, subsequent kernels may fail");
        }
    }

    return result;
}

std::vector<RunResult> Runner::run_category(
    const std::string& category, const RunConfig& config)
{
    std::vector<RunResult> results;
    for (auto* kernel : get_kernels_by_category(category)) {
        results.push_back(run(*kernel, config));
    }
    return results;
}

std::vector<std::string> Runner::get_categories() const {
    return KernelRegistry::instance().get_categories();
}

std::vector<KernelDescriptor*> Runner::get_kernels_by_category(const std::string& category) const {
    return KernelRegistry::instance().get_by_category(category);
}

std::vector<KernelDescriptor*> Runner::get_all_kernels() const {
    return KernelRegistry::instance().get_all();
}

}

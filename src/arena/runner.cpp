#include "arena/runner.hpp"
#include "arena/utils.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <nvtx3/nvToolsExt.h>

namespace arena {

Runner::Runner(Context& ctx, KernelLoader& loader, KernelCompiler& compiler,
               Profiler& profiler)
    : ctx_(ctx), loader_(loader), compiler_(compiler), profiler_(profiler) {}

RunResult Runner::run(KernelDescriptor& desc, const RunConfig& config) {
    RunResult result;
    result.kernel_name = desc.name();
    result.category = desc.category();
    result.description = desc.description();

    auto log = spdlog::get("runner");
    log->info("-------- {} --------", result.kernel_name);

    try {
        desc.set_problem_size(config.params);

        // runtime compilation for DSL kernels
        if (desc.needs_compilation()) {
            auto cr = compiler_.compile(desc.source_path());
            result.cache_hit = cr.cache_hit;
            result.compile_ms = cr.compile_time_ms;
            desc.set_compile_result(cr);
        }

        CUmodule module = nullptr;
        CUfunction func = nullptr;
        if (desc.uses_module()) {
            module = loader_.load_module(desc.module_path());
            func = loader_.get_function(module, desc.function_name());
        }
        desc.allocate(ctx_);
        desc.initialize(ctx_);

        auto launch_config = desc.get_launch_config();
        log->debug("{} launch config: grid({},{},{}) block({},{},{}) shmem={}B",
            result.kernel_name,
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

        auto launch_kernel = [&]() {
            if (desc.uses_module()) {
                auto args = desc.get_kernel_args();
                loader_.launch(func, launch_config, args.data());
            } else {
                desc.execute(ctx_);
            }
        };

        auto reset_fn = [&]() { desc.initialize(ctx_); };

        log->info("  warmup ...");
        auto warm = measure::warmup(launch_kernel, reset_fn, config);
        log->info("  warmup: {} iterations, {}", warm.iterations,
            warm.converged ? "converged" : "DID NOT CONVERGE");

        log->info("  benchmark: {} runs ...", config.number_of_runs);
        nvtxRangePushA(("BENCHMARK: " + result.kernel_name).c_str());
        auto bench_result = measure::time_operation(launch_kernel, reset_fn,
                                                    config.number_of_runs);
        nvtxRangePop();

        result.op_ms = bench_result.median_ms;
        result.all_times_ms = bench_result.all_times_ms;

        // GPU-only kernel time via Activity API (single run, sums all sub-kernels)
        desc.initialize(ctx_);
        auto activity = profiler_.collect_activity(launch_kernel);
        result.gpu_ms = activity.kernel_time_ms;
        result.sub_kernels = activity.sub_kernels;
        result.uses_module = desc.uses_module();

        double flops = desc.calculate_flops();
        double bytes = desc.calculate_bytes_accessed();
        result.gflops = (flops / (result.op_ms / 1000.0)) / 1e9;
        result.bandwidth_gbps = (bytes / (result.op_ms / 1000.0)) / 1e9;

        log->info("  result: wall={:.3f} ms  kernel={:.3f} ms  {:.2f} GFLOPS  {:.2f} GB/s",
            result.op_ms, result.gpu_ms, result.gflops, result.bandwidth_gbps);

        // profile
        if (config.collect_metrics) {
            log->info("  profiling: collecting hardware counters ...");
            nvtxRangePushA(("PROFILER: " + result.kernel_name).c_str());
            // registers and shared memory come from the activity pass, which
            // already ran above; no need to re-collect them here
            result.counters.regs_per_thread = activity.registers_per_thread;
            result.counters.shared_mem_bytes = activity.shared_memory_per_block;

            auto mv = profiler_.collect_counters(launch_kernel, reset_fn);
            if (mv.count(metric::OCCUPANCY)) {
                result.counters.occupancy = mv.at(metric::OCCUPANCY) / 100.0;
            }
            if (mv.count(metric::DRAM_READ)) {
                result.counters.dram_read_gbps = (mv.at(metric::DRAM_READ) / (result.op_ms / 1000.0)) / 1e9;
            }
            if (mv.count(metric::DRAM_WRITE)) {
                result.counters.dram_write_gbps = (mv.at(metric::DRAM_WRITE) / (result.op_ms / 1000.0)) / 1e9;
            }
            if (mv.count(metric::IPC)) {
                result.counters.ipc = mv.at(metric::IPC);
            }

            log->info("  profiling: regs={} shmem={}B occupancy={:.1f}% DRAM(R={:.2f} W={:.2f} GB/s) IPC={:.2f}",
                result.counters.regs_per_thread, result.counters.shared_mem_bytes,
                result.counters.occupancy * 100.0,
                result.counters.dram_read_gbps, result.counters.dram_write_gbps, result.counters.ipc);
            result.counters.available = true;
            nvtxRangePop();
        }

        result.verified = desc.verify(ctx_);
        if (result.verified) {
            log->info("  verify: passed");
        } else {
            log->warn("  verify: FAILED");
        }
        desc.cleanup(ctx_);
        result.success = true;

    } catch (const std::exception& e) {
        log->error("{}: {}", result.kernel_name, e.what());
        result.success = false;
        result.error = e.what();
        try { desc.cleanup(ctx_); } catch (...) {}

        // reset context in case of error
        try {
            CUresult ctx_status = cuCtxSynchronize();
            if (ctx_status != CUDA_SUCCESS) {
                log->warn("{}: CUDA context is in error state, resetting ...", result.kernel_name);
                loader_.unload_all();
                ctx_.reset();
            }
        } catch (...) {
            log->error("{}: failed to recover CUDA context, subsequent kernels may fail", result.kernel_name);
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

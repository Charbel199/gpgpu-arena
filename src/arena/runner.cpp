#include "arena/runner.hpp"
#include "arena/device/utils.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <nvtx3/nvToolsExt.h>

namespace arena {

Runner::Runner(Context& ctx, KernelLoader& loader, KernelCompiler& compiler,
               Profiler& profiler, PowerMonitor& power)
    : ctx_(ctx), loader_(loader), compiler_(compiler),
      profiler_(profiler), power_(power) {}

RunResult Runner::run(KernelDescriptor& desc, const RunConfig& config) {
    RunResult result;
    result.kernel_name = desc.name();
    result.category = desc.category();
    result.description = desc.description();

    auto log = spdlog::get("runner");
    log->info("-------- {} --------", result.kernel_name);

    try {
        desc.set_problem_size(config.params);
        desc.set_input_spec(config.input_distribution, config.input_seed);
        desc.set_block_size(config.block_size);
        desc.set_compile_options(config.compile_options);

        // runtime compilation for DSL kernels
        if (desc.needs_compilation()) {
            auto cr = compiler_.compile(desc.source_path(), desc.compile_options());
            result.cache_hit  = cr.cache_hit;
            result.compile_ms = cr.compile_ms;
            result.import_ms  = cr.import_ms;
            result.invoke_ms  = cr.invoke_ms;
            result.compile_options = desc.compile_options();
            desc.set_compile_result(cr);
        }

        // --- Tier 2: module load ---
        CUmodule module = nullptr;
        CUfunction func = nullptr;
        if (desc.uses_module()) {
            auto load_t0 = std::chrono::steady_clock::now();
            module = loader_.load_module(desc.module_path());
            func = loader_.get_function(module, desc.function_name());
            result.module_load_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - load_t0).count();
        }

        // --- allocation, with memory accounting ---
        ctx_.reset_peak();
        desc.allocate(ctx_);
        desc.initialize(ctx_);
        result.peak_device_bytes = ctx_.peak_bytes();

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
                const auto lr = loader_.launch(func, launch_config, args.data());
                // A rejected launch used to be logged and then ignored, which
                // made the kernel look extremely fast while producing nothing.
                // Both the Warp and cuTile breakages presented that way.
                if (lr.result != CUDA_SUCCESS) {
                    const char* msg = nullptr;
                    cuGetErrorString(lr.result, &msg);
                    throw std::runtime_error(
                        std::string("kernel launch failed: ") + (msg ? msg : "unknown"));
                }
            } else {
                desc.execute(ctx_);
            }
        };

        auto reset_fn = [&]() { desc.initialize(ctx_); };

        result.sm_clock_start_mhz = power_.sm_clock_mhz();

        // --- Tier 2: warmup / cold start ---
        auto warm = measure::warmup(launch_kernel, reset_fn, config);
        result.warmup_iterations = warm.iterations;
        result.warmup_converged  = warm.converged;
        result.first_launch_ms   = warm.first_launch_ms;
        log->info("  warmup: {} iterations, {}", warm.iterations,
            warm.converged ? "converged" : "DID NOT CONVERGE");

        // --- Tier 1: operation timing ---
        log->info("  benchmark: {} runs ...", config.number_of_runs);
        nvtxRangePushA(("BENCHMARK: " + result.kernel_name).c_str());
        auto bench_result = measure::time_operation(launch_kernel, reset_fn,
                                                    config.number_of_runs);
        nvtxRangePop();

        result.op_ms = bench_result.median_ms;
        result.all_times_ms = bench_result.all_times_ms;

        // --- Tier 1: SASS breakdown ---
        reset_fn();
        auto activity = profiler_.collect_activity(launch_kernel);
        result.gpu_ms = activity.kernel_time_ms;
        result.sub_kernels = activity.sub_kernels;
        result.launch_count = static_cast<int>(activity.sub_kernels.size());
        result.overhead_ms = std::max(0.0f, result.op_ms - result.gpu_ms);
        result.counters.regs_per_thread = activity.registers_per_thread;
        result.counters.shared_mem_bytes = activity.shared_memory_per_block;
        result.uses_module = desc.uses_module();

        // --- throughput: op_ms denominator ---
        result.gflops = measure::rate_per_sec(
            desc.calculate_flops(), result.op_ms) / 1e9;
        result.bandwidth_gbps = measure::rate_per_sec(
            desc.calculate_bytes_accessed(), result.op_ms) / 1e9;

        log->info("  result: op={:.3f} ms  gpu={:.3f} ms  overhead={:.3f} ms  "
                  "{:.2f} GFLOPS  {:.2f} GB/s",
            result.op_ms, result.gpu_ms, result.overhead_ms,
            result.gflops, result.bandwidth_gbps);

        // profile
        if (config.collect_metrics) {
            log->info("  profiling: collecting hardware counters ...");
            nvtxRangePushA(("PROFILER: " + result.kernel_name).c_str());
            auto mv = profiler_.collect_counters(launch_kernel, reset_fn);
            nvtxRangePop();

            // Counters are collected for a single launch replay, so the
            // matching interval is single-launch SASS time, not the whole
            // operation. Dividing by op_ms would mix a single-launch
            // numerator with a multi-launch denominator.
            if (mv.count(metric::OCCUPANCY)) {
                result.counters.occupancy = mv.at(metric::OCCUPANCY) / 100.0;
            }
            if (mv.count(metric::DRAM_READ)) {
                result.counters.dram_read_gbps = measure::rate_per_sec(
                    mv.at(metric::DRAM_READ), result.gpu_ms) / 1e9;
            }
            if (mv.count(metric::DRAM_WRITE)) {
                result.counters.dram_write_gbps = measure::rate_per_sec(
                    mv.at(metric::DRAM_WRITE), result.gpu_ms) / 1e9;
            }
            if (mv.count(metric::IPC)) {
                result.counters.ipc = mv.at(metric::IPC);
            }
            result.counters.available = true;

            log->info("  profiling: regs={} shmem={}B occupancy={:.1f}% DRAM(R={:.2f} W={:.2f} GB/s) IPC={:.2f}",
                result.counters.regs_per_thread, result.counters.shared_mem_bytes,
                result.counters.occupancy * 100.0,
                result.counters.dram_read_gbps, result.counters.dram_write_gbps, result.counters.ipc);
        }

        // --- verification gets its own clean run ---
        // Independent of which profiling passes ran, so a multi-pass
        // UserReplay counter collection cannot cause a spurious failure.
        reset_fn();
        launch_kernel();
        check_cuda(cuCtxSynchronize(), "verify sync");
        const auto v = desc.verify(ctx_);
        result.verified = v.passed;
        result.input_dtype  = dtype_name(desc.input_dtype());
        result.output_dtype = dtype_name(desc.output_dtype());
        result.compute_mode = compute_mode_name(desc.compute_mode());
        result.accuracy.checked = v.elements_checked > 0;
        result.accuracy.max_rel_error = v.max_rel_error;
        result.accuracy.mean_rel_error = v.mean_rel_error;
        result.accuracy.max_total_error = v.max_total_error;
        result.accuracy.mean_total_error = v.mean_total_error;
        result.accuracy.elements_checked = v.elements_checked;
        result.accuracy.tolerance = v.tolerance;

        if (v.passed) {
            log->info("  verify: passed (arithmetic {:.3e}, total {:.3e}, tolerance {:.3e})",
                v.max_rel_error, v.max_total_error, v.tolerance);
        } else {
            log->warn("  verify: FAILED (max rel err {:.3e} exceeds tolerance {:.3e})",
                v.max_rel_error, v.tolerance);
        }

        // --- energy: LAST, because it dirties the output buffer ---
        if (config.collect_energy && power_.available()) {
            log->info("  energy: sustained load for {:.0f} ms ...",
                config.energy_window_ms);
            auto e = measure::measure_energy(launch_kernel, power_,
                                             config.energy_window_ms);
            result.energy.available  = e.available;
            result.energy.mj_per_op  = e.mj_per_op;
            result.energy.avg_watts  = e.avg_watts;
            result.energy.iterations = e.iterations;

            if (e.available) {
                // Rough device-busy estimate: isolated gpu_ms times the
                // iteration count, over the actual window. It is only an
                // estimate -- gpu_ms comes from a single cold activity run, so
                // values above 100% mean sustained throughput beat that
                // sample. Values well below 100% are the actionable case: the
                // host could not issue launches fast enough, so the GPU idled
                // and mj_per_op is charged that idle draw.
                const double busy_pct = (e.total_ms > 0.0f)
                    ? (result.gpu_ms * e.iterations) / e.total_ms * 100.0 : 0.0;
                log->info("  energy: {:.4f} mJ/op  {:.1f} W  ({} iterations, "
                          "~{:.0f}% device-busy)",
                    e.mj_per_op, e.avg_watts, e.iterations, busy_pct);
                if (busy_pct < 80.0) {
                    log->warn("  energy: launch-bound (~{:.0f}% device-busy) - "
                              "mj_per_op includes idle draw, treat as an upper bound",
                              busy_pct);
                }
            } else {
                log->warn("  energy: counter did not advance, try a longer window");
            }
        }

        result.sm_clock_end_mhz = power_.sm_clock_mhz();
        log->debug("  clocks: {} -> {} MHz",
            result.sm_clock_start_mhz, result.sm_clock_end_mhz);

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

#include "arena/result_io.hpp"

#include "arena/measurement/launch_floor.hpp"

#include <cuda.h>
#include <spdlog/spdlog.h>
#include <sstream>

namespace arena::result_io {

using nlohmann::json;

std::string detect_dsl(const KernelDescriptor* d) {
    if (!d) return "unknown";
    const std::string src = d->source_path();

    if (src.find(".triton.") != std::string::npos) return "triton";
    if (src.find(".cutile.") != std::string::npos) return "cutile";
    if (src.find(".warp.")   != std::string::npos) return "warp";
    if (src.size() >= 3 && src.compare(src.size() - 3, 3, ".cu") == 0) return "cuda";

    // No runtime-compiled source: nvcc compiled it into the binary at build
    // time. That is how the CUB/Thrust descriptors are wired up.
    if (src.empty()) return "cub";

    return "cuda";
}

json to_json(const RunResult& r) {
    json j;
    j["kernel"]      = r.kernel_name;
    j["category"]    = r.category;
    j["description"] = r.description;
    j["source"]      = r.source;
    j["uses_module"] = r.uses_module;
    j["reference"]   = r.reference;

    j["input"] = {
        {"params",       r.params},
        {"distribution", r.distribution},
        {"seed",         r.seed},
        {"verify_seed",  r.verify_seed},
        {"cold_l2",      r.cold_l2},
    };

    j["launch"] = {
        {"block", {r.block_x, r.block_y, r.block_z}},
        {"grid",  {r.grid_x,  r.grid_y,  r.grid_z}},
        {"shared_mem_bytes", r.shared_mem_bytes},
        {"scratch_policy", r.scratch_policy},
    };

    // Tier 1: per-invocation. These are the only figures comparable as ms/op.
    j["timing"] = {
        {"op_ms",        r.op_ms},
        {"op_ms_trimmed", r.op_ms_trimmed},
        {"gpu_ms",       r.gpu_ms},
        {"overhead_ms",  r.overhead_ms},
        {"launch_count", r.launch_count},
        {"timing_mode",  r.timing_mode},
        {"all_times_ms", r.all_times_ms},
    };

    // Tier 2: per-process cold start.
    j["cold_start"] = {
        {"module_load_ms",  r.module_load_ms},
        {"first_launch_ms", r.first_launch_ms},
    };

    // Tier 3: one-time build. Zeroed on a cache hit, which cache_hit signals.
    j["build"] = {
        {"cache_hit",  r.cache_hit},
        {"compile_ms", r.compile_ms},
        {"import_ms",  r.import_ms},
        {"invoke_ms",  r.invoke_ms},
        {"options",    r.compile_options},
    };

    j["throughput"] = {
        {"gflops",         r.gflops},
        {"bandwidth_gbps", r.bandwidth_gbps},
    };
    // Per op, the split of the bytes bandwidth_gbps counts. null when the
    // descriptor does not say.
    j["throughput"]["ideal_read_bytes"] =
        r.ideal_bytes_known ? json(r.ideal_read_bytes) : json(nullptr);
    j["throughput"]["ideal_write_bytes"] =
        r.ideal_bytes_known ? json(r.ideal_write_bytes) : json(nullptr);

    j["counters"] = {
        {"available",       r.counters.available},
        {"regs_per_thread", r.counters.regs_per_thread},
        {"shared_mem_bytes", r.counters.shared_mem_bytes},
        {"occupancy",       r.counters.occupancy},
        {"dram_read_bytes", r.counters.dram_read_bytes},
        {"dram_write_bytes", r.counters.dram_write_bytes},
        {"dram_read_gbps",  r.counters.dram_read_gbps},
        {"dram_write_gbps", r.counters.dram_write_gbps},
        {"ipc",             r.counters.ipc},
        {"counters_level",  r.counters.level},
    };
    // --counters full. Each key is present only if the chip measured it.
    for (const auto& [key, value] : r.counters.extra) j["counters"][key] = value;
    if (!r.counters.pipes.empty())  j["counters"]["pipes"]  = r.counters.pipes;
    if (!r.counters.stalls.empty()) j["counters"]["stalls"] = r.counters.stalls;

    j["energy"] = {
        {"available",  r.energy.available},
        {"mj_per_op",  r.energy.mj_per_op},
        {"avg_watts",  r.energy.avg_watts},
        {"iterations", r.energy.iterations},
    };

    j["memory"] = { {"peak_device_bytes", r.peak_device_bytes} };

    j["numerics"] = {
        {"input_dtype",  r.input_dtype},
        {"output_dtype", r.output_dtype},
        {"compute_mode", r.compute_mode},
        {"checked",           r.accuracy.checked},
        {"max_rel_error",     r.accuracy.max_rel_error},
        {"mean_rel_error",    r.accuracy.mean_rel_error},
        {"max_total_error",   r.accuracy.max_total_error},
        {"mean_total_error",  r.accuracy.mean_total_error},
        {"elements_checked",  r.accuracy.elements_checked},
        {"tolerance",         r.accuracy.tolerance},
    };

    j["quality"] = {
        {"warmup_iterations",  r.warmup_iterations},
        {"warmup_converged",   r.warmup_converged},
        {"warmup_stop",        r.warmup_stop},
        {"sm_clock_start_mhz", r.sm_clock_start_mhz},
        {"sm_clock_end_mhz",   r.sm_clock_end_mhz},
        // Sampled across the timed runs only.
        {"sm_clock_mean_mhz",  r.sm_clock_mean_mhz},
        {"sm_clock_min_mhz",   r.sm_clock_min_mhz},
        {"sm_clock_max_mhz",   r.sm_clock_max_mhz},
        {"power_mean_w",       r.power_mean_w},
        {"power_max_w",        r.power_max_w},
        {"clock_samples",      r.clock_samples},
    };

    j["sub_kernels"] = json::array();
    for (const auto& sk : r.sub_kernels) {
        j["sub_kernels"].push_back({
            {"name",        sk.name},
            {"duration_ms", sk.duration_ms},
            {"start_offset_ms", sk.start_offset_ms},
            {"registers",   sk.registers},
            {"shared_mem",  sk.shared_memory},
        });
    }

    // The same launches grouped by kernel, so a library call shows which
    // kernels it dispatched and where the time went.
    j["launched_kernels"] = json::array();
    for (const auto& lk : r.launched_kernels) {
        j["launched_kernels"].push_back({
            {"name",         lk.name},
            {"count_per_op", lk.count_per_op},
            {"gpu_ms",       lk.gpu_ms},
        });
    }

    j["verified"] = r.verified;
    j["success"]  = r.success;
    j["error"]    = r.error;
    return j;
}

json environment_json(const Context& ctx, const PowerMonitor& power) {
    int driver_version = 0;
    cuDriverGetVersion(&driver_version);

    json j;
    j["device_name"]         = ctx.device_name();
    j["compute_capability"]  = std::to_string(ctx.compute_capability_major()) + "." +
                               std::to_string(ctx.compute_capability_minor());
    j["sm_count"]            = ctx.sm_count();
    j["total_memory_bytes"]  = ctx.total_memory();
    j["sm_clock_khz"]        = ctx.clock_rate_khz();
    j["memory_clock_khz"]    = ctx.memory_clock_khz();
    j["memory_bus_width"]    = ctx.memory_bus_width();
    j["l2_cache_bytes"]      = ctx.l2_cache_bytes();
    j["arch"]                = ctx.arch();
    j["repo_root"]           = ARENA_SOURCE_DIR;
    j["peak_dram_gbps"]      = ctx.peak_dram_gbps();
    j["peak_fp32_gflops"]    = ctx.peak_fp32_gflops();
    // The clock peak_fp32_gflops assumes, next to the highest the board can
    // reach, so a run at a locked or boosted clock can be rescaled.
    j["peak_fp32_gflops_at_mhz"] = ctx.clock_rate_khz() / 1000;
    j["max_sm_clock_mhz"]    = power.max_sm_clock_mhz();
    // Measured once per process, on first use.
    static const float launch_floor = measure_launch_floor_ms();
    static const float roundtrip_floor = measure_roundtrip_floor_ms();
    j["launch_floor_ms"]     = launch_floor;
    j["roundtrip_floor_ms"]  = roundtrip_floor;
    j["driver_version"]      = driver_version;
    j["cuda_toolkit_version"] = CUDA_VERSION;
    j["nvml_available"]      = power.available();
    j["sm_clock_now_mhz"]    = power.sm_clock_mhz();
    const auto& lock = power.clock_lock();
    if (lock.requested) {
        // achieved_mhz is the mean SM clock over every timed window of the
        // process. A lock is a ceiling, so it can sit well below
        // requested_mhz while applied is true: on an RTX PRO 6000 Blackwell
        // a 3090 MHz lock ran at about 2817 MHz with no throttle reason set,
        // the boost limit at that voltage and temperature.
        const auto achieved = power.achieved_sm_clock_mhz();
        j["clock_lock"] = {
            {"requested_mhz", lock.requested_mhz},
            {"applied",       lock.applied},
            {"error",         lock.error},
            {"achieved_mhz",  achieved ? json(*achieved) : json(nullptr)},
        };
        static bool warned = false;
        if (!warned && lock.applied && achieved &&
            measure::clock_below_request(*achieved, lock.requested_mhz)) {
            warned = true;
            spdlog::get("power")->warn(
                "--lock-clocks asked for {} MHz but the timed runs averaged {:.0f} MHz "
                "({:.1f}% lower). A lock is a ceiling: the board's boost limit, power or "
                "thermal limits kept it below that, so the clock was not fixed. Lock at "
                "or below {:.0f} MHz for a steady clock.",
                lock.requested_mhz, *achieved,
                100.0 * (1.0 - *achieved / lock.requested_mhz), *achieved);
        }
    }
    return j;
}

json config_json(const RunConfig& cfg) {
    json j;
    j["warmup_mode"]      = (cfg.warmup_mode == RunConfig::WarmupMode::Auto) ? "auto" : "fixed";
    j["warmup_runs"]      = cfg.warmup_runs;
    j["warmup_min"]       = cfg.warmup_min;
    j["warmup_max"]       = cfg.warmup_max;
    j["warmup_max_ms"]    = cfg.warmup_max_ms;
    j["drift_window"]     = cfg.drift_window;
    j["drift_threshold"]  = cfg.drift_threshold;
    j["number_of_runs"]   = cfg.number_of_runs;
    j["collect_metrics"]  = cfg.collect_metrics;
    j["cold_l2"]          = cfg.cold_l2;
    j["collect_energy"]   = cfg.collect_energy;
    j["energy_window_ms"] = cfg.energy_window_ms;
    j["sweep_min"]        = cfg.sweep_min;
    j["sweep_max"]        = cfg.sweep_max;
    j["sweep_factor"]     = cfg.sweep_factor;
    j["block_size"]       = cfg.block_size;
    j["input_distribution"] = distribution_name(cfg.input_distribution);
    j["input_seed"]         = cfg.input_seed;
    j["params"]           = cfg.params;
    return j;
}

std::string csv_header() {
    return "kernel,category,dsl,block,grid,op_ms,gpu_ms,overhead_ms,launch_count,"
           "gflops,bandwidth_gbps,status,counters_available,regs,shmem_bytes,"
           "occupancy_pct,ipc,dram_read_gbps,dram_write_gbps,"
           "peak_device_bytes,energy_available,mj_per_op,avg_watts,"
           "module_load_ms,first_launch_ms,cache_hit,compile_ms,import_ms,invoke_ms,"
           "warmup_iterations,warmup_converged,sm_clock_start_mhz,sm_clock_end_mhz,"
           "input_dtype,output_dtype,compute_mode,max_rel_error,mean_rel_error,"
           "max_total_error,mean_total_error,error_tolerance,elements_checked";
}

std::string csv_row(const RunResult& r, const std::string& dsl) {
    std::ostringstream f;
    f << r.kernel_name << "," << r.category << "," << dsl << ","
      << r.block_x << "x" << r.block_y << "x" << r.block_z << ","
      << r.grid_x << "x" << r.grid_y << "x" << r.grid_z << ","
      << r.op_ms << "," << r.gpu_ms << "," << r.overhead_ms << ","
      << r.launch_count << ","
      << r.gflops << "," << r.bandwidth_gbps << ","
      << (r.success ? (r.verified ? "OK" : "WARN") : "FAIL") << ","
      << (r.counters.available ? "true" : "false") << ","
      << r.counters.regs_per_thread << "," << r.counters.shared_mem_bytes << ","
      << r.counters.occupancy * 100.0 << "," << r.counters.ipc << ","
      << r.counters.dram_read_gbps << "," << r.counters.dram_write_gbps << ","
      << r.peak_device_bytes << ","
      << (r.energy.available ? "true" : "false") << ","
      << r.energy.mj_per_op << "," << r.energy.avg_watts << ","
      << r.module_load_ms << "," << r.first_launch_ms << ","
      << (r.cache_hit ? "true" : "false") << ","
      << r.compile_ms << "," << r.import_ms << "," << r.invoke_ms << ","
      << r.warmup_iterations << ","
      << (r.warmup_converged ? "true" : "false") << ","
      << r.sm_clock_start_mhz << "," << r.sm_clock_end_mhz << ","
      << r.input_dtype << "," << r.output_dtype << "," << r.compute_mode << ","
      << r.accuracy.max_rel_error << "," << r.accuracy.mean_rel_error << ","
      << r.accuracy.max_total_error << "," << r.accuracy.mean_total_error << ","
      << r.accuracy.tolerance << "," << r.accuracy.elements_checked;
    return f.str();
}

}

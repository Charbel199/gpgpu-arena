#include "arena/result_io.hpp"

#include <cuda.h>
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
    j["uses_module"] = r.uses_module;

    j["launch"] = {
        {"block", {r.block_x, r.block_y, r.block_z}},
        {"grid",  {r.grid_x,  r.grid_y,  r.grid_z}},
        {"shared_mem_bytes", r.shared_mem_bytes},
    };

    // Tier 1: per-invocation. These are the only figures comparable as ms/op.
    j["timing"] = {
        {"op_ms",        r.op_ms},
        {"gpu_ms",       r.gpu_ms},
        {"overhead_ms",  r.overhead_ms},
        {"launch_count", r.launch_count},
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
    };

    j["throughput"] = {
        {"gflops",         r.gflops},
        {"bandwidth_gbps", r.bandwidth_gbps},
    };

    j["counters"] = {
        {"available",       r.counters.available},
        {"regs_per_thread", r.counters.regs_per_thread},
        {"shared_mem_bytes", r.counters.shared_mem_bytes},
        {"occupancy",       r.counters.occupancy},
        {"dram_read_gbps",  r.counters.dram_read_gbps},
        {"dram_write_gbps", r.counters.dram_write_gbps},
        {"ipc",             r.counters.ipc},
    };

    j["energy"] = {
        {"available",  r.energy.available},
        {"mj_per_op",  r.energy.mj_per_op},
        {"avg_watts",  r.energy.avg_watts},
        {"iterations", r.energy.iterations},
    };

    j["memory"] = { {"peak_device_bytes", r.peak_device_bytes} };

    j["quality"] = {
        {"warmup_iterations",  r.warmup_iterations},
        {"warmup_converged",   r.warmup_converged},
        {"sm_clock_start_mhz", r.sm_clock_start_mhz},
        {"sm_clock_end_mhz",   r.sm_clock_end_mhz},
    };

    j["sub_kernels"] = json::array();
    for (const auto& sk : r.sub_kernels) {
        j["sub_kernels"].push_back({
            {"name",        sk.name},
            {"duration_ms", sk.duration_ms},
            {"registers",   sk.registers},
            {"shared_mem",  sk.shared_memory},
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
    j["driver_version"]      = driver_version;
    j["cuda_toolkit_version"] = CUDA_VERSION;
    j["nvml_available"]      = power.available();
    j["sm_clock_now_mhz"]    = power.sm_clock_mhz();
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
    j["collect_energy"]   = cfg.collect_energy;
    j["energy_window_ms"] = cfg.energy_window_ms;
    j["params"]           = cfg.params;
    return j;
}

std::string csv_header() {
    return "kernel,category,dsl,block,grid,op_ms,gpu_ms,overhead_ms,launch_count,"
           "gflops,bandwidth_gbps,status,counters_available,regs,shmem_bytes,"
           "occupancy_pct,ipc,dram_read_gbps,dram_write_gbps,"
           "peak_device_bytes,energy_available,mj_per_op,avg_watts,"
           "module_load_ms,first_launch_ms,cache_hit,compile_ms,import_ms,invoke_ms,"
           "warmup_iterations,warmup_converged,sm_clock_start_mhz,sm_clock_end_mhz";
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
      << r.sm_clock_start_mhz << "," << r.sm_clock_end_mhz;
    return f.str();
}

}

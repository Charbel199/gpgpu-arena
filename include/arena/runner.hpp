#pragma once

#include "arena/device/context.hpp"
#include "arena/device/kernel_loader.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/runner_config.hpp"
#include "arena/measurement/measure.hpp"
#include "arena/measurement/profiler.hpp"
#include "arena/measurement/power.hpp"
#include "arena/kernel_descriptor.hpp"
#include <string>
#include <vector>
#include <map>

namespace arena {

struct RunResult {
    std::string kernel_name;
    std::string category;
    std::string description;

    // launch type: true = cubin via driver API, false = compiled (e.g. CUB)
    bool uses_module = false;

    // launch config
    unsigned int grid_x = 0, grid_y = 0, grid_z = 0;
    unsigned int block_x = 0, block_y = 0, block_z = 0;
    unsigned int shared_mem_bytes = 0;

    // --- Tier 1: per-invocation ---
    float op_ms        = 0.0f;   // event-bracketed whole operation, median
    float gpu_ms       = 0.0f;   // sum of SASS durations (CUPTI Activity)
    float overhead_ms  = 0.0f;   // max(0, op_ms - gpu_ms)
    int   launch_count = 0;
    std::vector<float> all_times_ms;
    std::vector<Profiler::SubKernelInfo> sub_kernels;

    // --- Tier 2: per-process cold start ---
    float module_load_ms  = 0.0f;
    float first_launch_ms = 0.0f;

    // --- Tier 3: build (cacheable to disk) ---
    float compile_ms = 0.0f;   // the DSL's own compile time
    float import_ms  = 0.0f;   // Python import cost, DSL only
    float invoke_ms  = 0.0f;   // full subprocess wall time
    bool  cache_hit  = false;

    // --- throughput: denominator is op_ms ---
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;

    // --- hardware counters: denominator is gpu_ms ---
    struct Counters {
        bool   available = false;
        int    regs_per_thread = 0;
        int    shared_mem_bytes = 0;
        double occupancy = 0.0;        // fraction, 0..1
        double dram_read_gbps = 0.0;
        double dram_write_gbps = 0.0;
        double ipc = 0.0;
    } counters;

    struct Energy {
        bool   available = false;
        double mj_per_op = 0.0;
        double avg_watts = 0.0;
        int    iterations = 0;
    } energy;

    size_t peak_device_bytes = 0;

    // --- diagnostics ---
    int  warmup_iterations = 0;
    bool warmup_converged  = false;
    unsigned sm_clock_start_mhz = 0;
    unsigned sm_clock_end_mhz = 0;

    bool verified = false;
    bool success = false;
    std::string error;
};

class Runner {
public:
    Runner(Context& ctx, KernelLoader& loader, KernelCompiler& compiler,
           Profiler& profiler, PowerMonitor& power);

    RunResult run(KernelDescriptor& descriptor, const RunConfig& config);

    std::vector<RunResult> run_category(const std::string& category, const RunConfig& config);


    std::vector<std::string> get_categories() const;
    std::vector<KernelDescriptor*> get_kernels_by_category(const std::string& category) const;
    std::vector<KernelDescriptor*> get_all_kernels() const;

    const Context& context() const { return ctx_; }
    Context& mutable_context() { return ctx_; }
    KernelCompiler& compiler() { return compiler_; }
    const PowerMonitor& power() const { return power_; }

private:
    Context& ctx_;
    KernelLoader& loader_;
    KernelCompiler& compiler_;
    Profiler& profiler_;
    PowerMonitor& power_;
};

}

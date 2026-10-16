#pragma once

#include "arena/device/context.hpp"
#include "arena/device/kernel_loader.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/runner_config.hpp"
#include "arena/measurement/measure.hpp"
#include "arena/measurement/profiler.hpp"
#include "arena/measurement/accuracy.hpp"
#include "arena/dtype.hpp"
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
    std::string source;   // the file actually compiled, override included

    // What it ran on. A CLI run can cover several sizes, distributions and
    // seeds, so each result says which one it is.
    std::map<std::string, int> params;
    std::string distribution;
    uint64_t seed = 0;
    uint64_t verify_seed = 0;   // inputs are regenerated with this for verification
    bool cold_l2 = false;

    // launch type: true = cubin via driver API, false = compiled (e.g. CUB)
    bool uses_module = false;

    // a vendor library baseline such as cuBLAS (KernelDescriptor::reference)
    bool reference = false;

    // launch config
    unsigned int grid_x = 0, grid_y = 0, grid_z = 0;
    unsigned int block_x = 0, block_y = 0, block_z = 0;
    unsigned int shared_mem_bytes = 0;
    std::string scratch_policy = "legacy";   // see ScratchPolicy

    // --- Tier 1: per-invocation ---
    float op_ms        = 0.0f;   // event-bracketed whole operation, median
    float op_ms_trimmed = 0.0f;  // same, interquartile mean: steadier for short ops
    float gpu_ms       = 0.0f;   // sum of SASS durations (CUPTI Activity)
    float overhead_ms  = 0.0f;   // max(0, op_ms - gpu_ms)
    int   launch_count = 0;
    std::vector<float> all_times_ms;
    std::vector<Profiler::SubKernelInfo> sub_kernels;
    std::vector<LaunchedKernel> launched_kernels;   // sub_kernels grouped by name

    // --- Tier 2: per-process cold start ---
    float module_load_ms  = 0.0f;
    float first_launch_ms = 0.0f;

    // --- Tier 3: build (cacheable to disk) ---
    float compile_ms = 0.0f;   // the DSL's own compile time
    float import_ms  = 0.0f;   // Python import cost, DSL only
    float invoke_ms  = 0.0f;   // full subprocess wall time
    bool  cache_hit  = false;
    std::map<std::string, int> compile_options;   // knobs this cubin was built with

    // --- throughput: denominator is op_ms ---
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;

    // --- hardware counters: denominator is gpu_ms ---
    struct Counters {
        bool   available = false;
        int    regs_per_thread = 0;
        int    shared_mem_bytes = 0;
        double occupancy = 0.0;        // fraction, 0..1
        // Bytes are the number the hardware actually reports and they match
        // Nsight Compute exactly. The GB/s below divides them by the duration
        // of the same profiled launch, which is instrumented and therefore
        // slower than the unprofiled kernel, so treat the rate as a profiled
        // figure rather than the steady-state one.
        double dram_read_bytes = 0.0;
        double dram_write_bytes = 0.0;
        double dram_read_gbps = 0.0;
        double dram_write_gbps = 0.0;
        double ipc = 0.0;

        // --counters full. Only what the chip supports is present.
        std::string level = "basic";
        std::map<std::string, double> extra;    // issue_active, eligible_warps, ...
        std::map<std::string, double> pipes;    // fraction 0..1 of peak
        std::map<std::string, double> stalls;   // warps stalled per issued instruction
    } counters;

    struct Energy {
        bool   available = false;
        double mj_per_op = 0.0;
        double avg_watts = 0.0;
        int    iterations = 0;
    } energy;

    size_t peak_device_bytes = 0;

    // --- numerics ---
    std::string input_dtype = "fp32";
    std::string output_dtype = "fp32";
    std::string compute_mode = "default";
    struct Accuracy {
        bool   checked = false;
        double max_rel_error = 0.0;     // vs the inputs the kernel received
        double mean_rel_error = 0.0;
        double max_total_error = 0.0;   // vs the original fp32 data
        double mean_total_error = 0.0;
        int    elements_checked = 0;
        double tolerance = 0.0;
    } accuracy;

    // --- diagnostics ---
    int  warmup_iterations = 0;
    bool warmup_converged  = false;
    std::string warmup_stop;   // why it stopped, for when it did not converge
    unsigned sm_clock_start_mhz = 0;
    unsigned sm_clock_end_mhz = 0;
    // Sampled by NVML across the timed runs only. Zero samples when NVML is
    // unavailable.
    double   sm_clock_mean_mhz = 0.0;
    unsigned sm_clock_min_mhz = 0;
    unsigned sm_clock_max_mhz = 0;
    double   power_mean_w = 0.0;
    double   power_max_w = 0.0;
    int      clock_samples = 0;

    bool verified = false;
    bool success = false;
    std::string error;
};

class Runner {
public:
    Runner(Context& ctx, KernelLoader& loader, KernelCompiler& compiler,
           Profiler& profiler, PowerMonitor& power);
    ~Runner();
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

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
    // Overwrites a buffer twice the size of L2, which evicts whatever the
    // previous launch left there. Enqueued on the null stream, so it finishes
    // before the next timed launch starts and is not part of its interval.
    void scrub_l2();

    Context& ctx_;
    KernelLoader& loader_;
    KernelCompiler& compiler_;
    Profiler& profiler_;
    PowerMonitor& power_;

    CUdeviceptr scrub_buf_ = 0;
    size_t      scrub_bytes_ = 0;
    CUcontext   scrub_ctx_ = nullptr;   // a context reset invalidates the buffer
    unsigned    scrub_value_ = 0;
};

}

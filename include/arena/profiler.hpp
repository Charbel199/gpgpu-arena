#pragma once

#include <cuda.h>
#include <cupti.h>
#include <string>
#include <vector>
#include <functional>

namespace arena {

class Profiler {
public:
    Profiler();
    ~Profiler();


    struct ProfilerConfig {
        int number_of_runs = 10;
    };


    // Metrics we can collect
    struct KernelMetrics {
        // Timing
        float elapsed_ms = 0.0f;
        
        // Memory throughput
        double dram_read_throughput_gbps = 0.0;
        double dram_write_throughput_gbps = 0.0;
        
        // Compute efficiency
        double achieved_occupancy = 0.0;      // 0.0 - 1.0
        double warp_execution_efficiency = 0.0;
        double sm_efficiency = 0.0;
        
        // Instruction stats
        uint64_t instructions_executed = 0;
        double ipc = 0.0;  // Instructions per cycle
        
        // Register usage
        int registers_per_thread = 0;
        int shared_memory_per_block = 0;
    };

    // Profile a kernel launch (wrap the launch in this)
    using KernelLaunchFn = std::function<void()>;
    KernelMetrics profile(KernelLaunchFn launch_fn, ProfilerConfig config);

    // Get available metrics for the current device
    std::vector<std::string> available_metrics() const;

    // Enable/disable specific metrics (some have overhead)
    void enable_metric(const std::string& metric_name);
    void disable_metric(const std::string& metric_name);

private:
    void init_cupti();
    void cleanup_cupti();

    bool initialized_ = false;
};

}

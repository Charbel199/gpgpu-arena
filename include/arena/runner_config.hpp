#pragma once

#include "arena/distribution.hpp"

#include <cstdint>
#include <map>
#include <string>

namespace arena {

struct RunConfig {
    enum class WarmupMode { Fixed, Auto };

    WarmupMode warmup_mode   = WarmupMode::Auto;
    int   warmup_runs        = 10;      // used when warmup_mode == Fixed
    int   warmup_min         = 3;       // Auto: minimum iterations
    int   warmup_max         = 100;     // Auto: iteration cap
    float warmup_max_ms      = 500.0f;  // Auto: wall-clock cap
    int   drift_window       = 5;
    float drift_threshold    = 0.02f;

    // Note: in Auto mode the effective iteration floor is
    // max(warmup_min, 2 * drift_window), because check_drift cannot report
    // convergence until it has two full windows to compare. Setting
    // warmup_min below 2 * drift_window has no effect.

    int  number_of_runs   = 10;
    bool collect_metrics  = false;

    bool  collect_energy   = false;
    float energy_window_ms = 500.0f;

    // Block size to launch at, or 0 for each kernel's own default. Only
    // meaningful for kernels that report tunable_block_sizes().
    int block_size = 0;

    // Compile-time defines for DSL kernels, empty for the source's own
    // defaults. The DSL equivalent of block_size: it needs a recompile, so
    // each distinct set gets its own cached cubin.
    std::map<std::string, int> compile_options;

    // Sweep range. Zero means "use the category's own default ladder", so
    // nothing changes until you actually set a range.
    int    sweep_min = 0;
    int    sweep_max = 0;
    double sweep_factor = 4.0;   // each step multiplies the size by this

    // Input data. The seed makes a run reproducible and lets the CPU
    // reference be regenerated from the same numbers the GPU saw.
    Distribution input_distribution = Distribution::Uniform;
    uint64_t     input_seed = 42;

    std::map<std::string, int> params;
};

}

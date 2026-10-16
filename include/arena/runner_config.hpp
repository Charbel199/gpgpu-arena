#pragma once

#include "arena/distribution.hpp"

#include <cstdint>
#include <map>
#include <string>

namespace arena {

// How a manifest's scratch buffers are prepared before each launch.
//
// Legacy zeroes every one outside the timed op, so a kernel that needs a
// zeroed workspace gets it for free. Declared charges for it instead: a
// buffer marked "zero": true is cleared by a memset inside the timed op, and
// every other buffer is filled with 0xFF bytes outside it, so a kernel that
// relies on zeros it did not ask for fails verification.
enum class ScratchPolicy { Legacy, Declared };

inline const char* scratch_policy_name(ScratchPolicy p) {
    return p == ScratchPolicy::Declared ? "declared" : "legacy";
}

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

    // What collect_metrics gathers. Basic is DRAM bytes, occupancy and IPC.
    // Full adds pipe utilisation, stall reasons, shared memory and L2
    // traffic, at the cost of many more replay passes.
    enum class CounterLevel { Basic, Full };
    CounterLevel counter_level = CounterLevel::Basic;

    // Evict L2 before every timed launch. Off, a working set that fits in L2
    // (16 MB at n=4M on a 128 MB part) is served from cache on every run after
    // the first, and the timing describes L2 rather than DRAM. On, every
    // launch starts cold, which is what a kernel sees when its input was
    // produced by something else. The scrub runs outside the timed interval.
    bool cold_l2 = false;

    bool  collect_energy   = false;
    float energy_window_ms = 500.0f;

    // Block size to launch at, or 0 for each kernel's own default. Only
    // meaningful for kernels that report tunable_block_sizes().
    int block_size = 0;

    // Compile-time defines for DSL kernels, empty for the source's own
    // defaults. The DSL equivalent of block_size: it needs a recompile, so
    // each distinct set gets its own cached cubin.
    std::map<std::string, int> compile_options;

    ScratchPolicy scratch_policy = ScratchPolicy::Legacy;

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

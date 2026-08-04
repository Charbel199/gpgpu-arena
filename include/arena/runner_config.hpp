#pragma once

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

    std::map<std::string, int> params;
};

}

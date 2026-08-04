#pragma once

#include <nvml.h>
#include <cstdint>

namespace arena {

// NVML wrapper for energy and clock sampling. Degrades to available()==false
// rather than throwing, exactly like Profiler does when CUPTI is unavailable.
class PowerMonitor {
public:
    explicit PowerMonitor(int device_id = 0);
    ~PowerMonitor();

    PowerMonitor(const PowerMonitor&) = delete;
    PowerMonitor& operator=(const PowerMonitor&) = delete;

    // false when NVML is missing, the device handle failed, or the device
    // does not expose a total-energy counter.
    bool available() const { return available_; }

    // Cumulative device energy in millijoules since driver load.
    // Returns 0 when unavailable.
    uint64_t energy_mj() const;

    // Current SM clock in MHz. Returns 0 when unavailable.
    unsigned sm_clock_mhz() const;

private:
    bool         available_ = false;
    bool         nvml_initialized_ = false;
    nvmlDevice_t device_{};
};

}

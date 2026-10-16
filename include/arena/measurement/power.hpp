#pragma once

#include "arena/measurement/measure_policy.hpp"

#include <nvml.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace arena {

// --lock-clocks outcome, for the environment record.
struct ClockLockStatus {
    bool        requested = false;
    unsigned    requested_mhz = 0;
    bool        applied = false;
    std::string error;
};

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

    // Current SM clock in MHz. Needs only a device handle, not the energy
    // counter. Returns 0 when unavailable.
    unsigned sm_clock_mhz() const;

    // Highest SM clock the board allows. Returns 0 when unavailable.
    unsigned max_sm_clock_mhz() const;

    // Samples SM clock and power on a background thread, every interval,
    // until stop_sampling. Meant to bracket the timed runs only.
    //
    // NVML refreshes the SM clock reading only about every 500 ms (measured
    // on a 595 driver), so a shorter window repeats one value however often
    // it is polled. Power comes from the driver's own sample buffer, one
    // reading per 20 ms, for the samples that fall inside the window. A
    // window too short to contain one falls back to the polled
    // instantaneous reading. The plain nvmlDeviceGetPowerUsage is a 1 s
    // average on Ampere and later, so it is only the last resort.
    void start_sampling(std::chrono::microseconds interval = std::chrono::microseconds(2000));
    measure::ClockPowerStats stop_sampling();

    // Locks the SM clock to mhz, or for 0 to the clock pick_auto_lock_clock
    // chooses. The lock is a ceiling: a clock above what the board sustains
    // is not reached, the boost limit and power and thermal limits hold it
    // down as they would unlocked. achieved_sm_clock_mhz() says what it really ran at. Needs root, and inside a container CAP_SYS_ADMIN as well
    // (root alone gets NVML_ERROR_NO_PERMISSION). On failure it logs a
    // warning and the run goes on unlocked. The lock is released by the
    // destructor, at exit, on SIGINT/SIGTERM and on std::terminate.
    const ClockLockStatus& lock_clocks(unsigned mhz);
    const ClockLockStatus& clock_lock() const { return lock_; }

    // Mean SM clock over every window sampled so far in this process, which
    // is every timed benchmark. nullopt when nothing was sampled.
    std::optional<double> achieved_sm_clock_mhz() const { return tally_.mean(); }

private:
    void sample_once();

    bool         available_ = false;
    bool         device_ok_ = false;
    bool         nvml_initialized_ = false;
    bool         instant_power_ = false;
    nvmlDevice_t device_{};

    ClockLockStatus lock_;
    measure::ClockTally tally_;

    std::thread             sampler_;
    std::mutex              sampler_mu_;
    std::condition_variable sampler_cv_;
    bool                    sampler_stop_ = false;
    std::vector<unsigned>   clock_samples_;
    std::vector<double>     power_samples_;
    unsigned long long      window_start_us_ = 0;   // NVML timestamps are wall-clock us
};

}

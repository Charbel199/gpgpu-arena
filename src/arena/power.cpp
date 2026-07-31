#include "arena/power.hpp"
#include <spdlog/spdlog.h>

namespace arena {

PowerMonitor::PowerMonitor(int device_id) {
    auto log = spdlog::get("power");

    nvmlReturn_t rc = nvmlInit();
    if (rc != NVML_SUCCESS) {
        log->warn("NVML init failed ({}), energy measurement disabled",
            nvmlErrorString(rc));
        return;
    }
    nvml_initialized_ = true;

    rc = nvmlDeviceGetHandleByIndex(static_cast<unsigned>(device_id), &device_);
    if (rc != NVML_SUCCESS) {
        log->warn("NVML device {} handle failed ({}), energy measurement disabled",
            device_id, nvmlErrorString(rc));
        return;
    }

    // Probe the energy counter: it is the one capability we actually require.
    unsigned long long probe = 0;
    rc = nvmlDeviceGetTotalEnergyConsumption(device_, &probe);
    if (rc != NVML_SUCCESS) {
        log->warn("Energy counter unsupported on this device ({}), "
                  "energy measurement disabled", nvmlErrorString(rc));
        return;
    }

    available_ = true;
    log->info("NVML initialized (device {}, energy counter available)", device_id);
}

PowerMonitor::~PowerMonitor() {
    if (nvml_initialized_) nvmlShutdown();
}

uint64_t PowerMonitor::energy_mj() const {
    if (!available_) return 0;
    unsigned long long mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(device_, &mj) != NVML_SUCCESS) return 0;
    return static_cast<uint64_t>(mj);
}

unsigned PowerMonitor::sm_clock_mhz() const {
    if (!available_) return 0;
    unsigned mhz = 0;
    if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &mhz) != NVML_SUCCESS) return 0;
    return mhz;
}

}

#include "arena/measurement/power.hpp"
#include <spdlog/spdlog.h>

#include <atomic>
#include <csignal>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <cstdlib>
#include <exception>

namespace arena {

namespace {

// The device whose clocks are locked, for the exit paths that cannot reach
// the PowerMonitor. The flag is cleared by whichever path resets first.
nvmlDevice_t      g_locked_device{};
std::atomic<bool> g_clocks_locked{false};
std::terminate_handler g_prev_terminate = nullptr;

void reset_locked_clocks() {
    if (g_clocks_locked.exchange(false)) {
        nvmlDeviceResetGpuLockedClocks(g_locked_device);
    }
}

// Signals that end the process and would otherwise leave the clock locked.
// SIGKILL cannot be caught, so a killed run still leaves it locked.
constexpr int kResetSignals[] = {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGABRT, SIGSEGV, SIGBUS};

// NVML is not async-signal-safe, so the reset can in principle hang if the
// signal lands inside another NVML call. SA_RESETHAND and SA_NODEFER keep
// that recoverable: the default action is already back and the signal is not
// blocked, so a second Ctrl-C ends the process. The raise ends it once the
// reset returns. As PID 1 in a container the default action of SIGINT and
// SIGTERM is dropped, and the run would go on unlocked while its results
// still say locked, so it exits here instead.
void on_fatal_signal(int sig) {
    reset_locked_clocks();
    std::raise(sig);
    _exit(128 + sig);
}

void on_terminate() {
    reset_locked_clocks();
    if (g_prev_terminate) g_prev_terminate();
    std::abort();
}

void install_reset_handlers() {
    static bool installed = false;
    if (installed) return;
    installed = true;
    std::atexit(reset_locked_clocks);
    for (int sig : kResetSignals) {
        // A signal the parent set to ignored (nohup, a background job) stays
        // ignored rather than starting to kill the process.
        struct sigaction old{};
        if (sigaction(sig, nullptr, &old) == 0 && old.sa_handler == SIG_IGN) continue;
        struct sigaction sa{};
        sa.sa_handler = on_fatal_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESETHAND | SA_NODEFER;
        sigaction(sig, &sa, nullptr);
    }
    g_prev_terminate = std::set_terminate(on_terminate);
}

}

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
    device_ok_ = true;

    nvmlFieldValue_t fv{};
    fv.fieldId = NVML_FI_DEV_POWER_INSTANT;
    instant_power_ = nvmlDeviceGetFieldValues(device_, 1, &fv) == NVML_SUCCESS &&
                     fv.nvmlReturn == NVML_SUCCESS;

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
    if (sampler_.joinable()) stop_sampling();
    // The global flag, not lock_.applied: a second lock_clocks call that
    // fails clears lock_ while the first lock is still in place, and the
    // atexit reset would come after nvmlShutdown and do nothing.
    if (g_clocks_locked && g_locked_device == device_) {
        reset_locked_clocks();
        spdlog::get("power")->info("GPU clock lock released");
    }
    if (nvml_initialized_) nvmlShutdown();
}

uint64_t PowerMonitor::energy_mj() const {
    if (!available_) return 0;
    unsigned long long mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(device_, &mj) != NVML_SUCCESS) return 0;
    return static_cast<uint64_t>(mj);
}

unsigned PowerMonitor::sm_clock_mhz() const {
    if (!device_ok_) return 0;
    unsigned mhz = 0;
    if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &mhz) != NVML_SUCCESS) return 0;
    return mhz;
}

unsigned PowerMonitor::max_sm_clock_mhz() const {
    if (!device_ok_) return 0;
    unsigned mhz = 0;
    if (nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_SM, &mhz) != NVML_SUCCESS) return 0;
    return mhz;
}

void PowerMonitor::sample_once() {
    unsigned mhz = 0;
    if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) {
        clock_samples_.push_back(mhz);
    }

    if (instant_power_) {
        nvmlFieldValue_t fv{};
        fv.fieldId = NVML_FI_DEV_POWER_INSTANT;
        if (nvmlDeviceGetFieldValues(device_, 1, &fv) == NVML_SUCCESS &&
            fv.nvmlReturn == NVML_SUCCESS) {
            power_samples_.push_back(fv.value.uiVal / 1000.0);   // mW
        }
    } else {
        unsigned mw = 0;
        if (nvmlDeviceGetPowerUsage(device_, &mw) == NVML_SUCCESS) {
            power_samples_.push_back(mw / 1000.0);
        }
    }
}

namespace {
unsigned long long wall_clock_us() {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}
}

void PowerMonitor::start_sampling(std::chrono::microseconds interval) {
    if (!device_ok_ || sampler_.joinable()) return;
    clock_samples_.clear();
    power_samples_.clear();
    sampler_stop_ = false;
    window_start_us_ = wall_clock_us();

    sampler_ = std::thread([this, interval] {
        // Keep the reset handler off this thread, which spends its time
        // inside NVML calls the handler would interrupt.
        sigset_t block;
        sigemptyset(&block);
        for (int sig : kResetSignals) sigaddset(&block, sig);
        pthread_sigmask(SIG_BLOCK, &block, nullptr);

        std::unique_lock<std::mutex> lk(sampler_mu_);
        for (;;) {
            sample_once();
            if (sampler_cv_.wait_for(lk, interval, [this] { return sampler_stop_; })) break;
        }
    });
}

measure::ClockPowerStats PowerMonitor::stop_sampling() {
    if (!sampler_.joinable()) return {};
    {
        std::lock_guard<std::mutex> lk(sampler_mu_);
        sampler_stop_ = true;
    }
    sampler_cv_.notify_one();
    sampler_.join();
    const unsigned long long window_end_us = wall_clock_us();
    tally_.add(clock_samples_);

    // The driver's power buffer, if it covers the window.
    std::vector<double> buffered;
    nvmlValueType_t type;
    unsigned count = 0;
    if (nvmlDeviceGetSamples(device_, NVML_TOTAL_POWER_SAMPLES, window_start_us_,
                             &type, &count, nullptr) == NVML_SUCCESS && count > 0) {
        std::vector<nvmlSample_t> raw(count);
        if (nvmlDeviceGetSamples(device_, NVML_TOTAL_POWER_SAMPLES, window_start_us_,
                                 &type, &count, raw.data()) == NVML_SUCCESS) {
            for (unsigned i = 0; i < count && i < raw.size(); i++) {
                if (raw[i].timeStamp < window_start_us_ || raw[i].timeStamp > window_end_us) continue;
                double mw = 0.0;
                switch (type) {
                    case NVML_VALUE_TYPE_UNSIGNED_INT:  mw = raw[i].sampleValue.uiVal; break;
                    case NVML_VALUE_TYPE_UNSIGNED_LONG: mw = raw[i].sampleValue.ulVal; break;
                    case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: mw = raw[i].sampleValue.ullVal; break;
                    case NVML_VALUE_TYPE_DOUBLE:        mw = raw[i].sampleValue.dVal; break;
                    default: continue;
                }
                buffered.push_back(mw / 1000.0);
            }
        }
    }
    return measure::summarize_clock_power(clock_samples_,
                                          buffered.empty() ? power_samples_ : buffered);
}

const ClockLockStatus& PowerMonitor::lock_clocks(unsigned mhz) {
    auto log = spdlog::get("power");
    lock_ = {};
    lock_.requested = true;
    lock_.requested_mhz = mhz;

    if (!device_ok_) {
        lock_.error = "NVML unavailable";
        log->warn("--lock-clocks: {}, running unlocked", lock_.error);
        return lock_;
    }

    if (mhz == 0) {
        unsigned app_default = 0;
        nvmlDeviceGetClock(device_, NVML_CLOCK_SM, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, &app_default);

        std::vector<unsigned> supported;
        unsigned mem_mhz = 0;
        if (nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_MEM, &mem_mhz) == NVML_SUCCESS) {
            supported.resize(1024);
            unsigned count = static_cast<unsigned>(supported.size());
            if (nvmlDeviceGetSupportedGraphicsClocks(device_, mem_mhz, &count,
                    supported.data()) == NVML_SUCCESS) {
                supported.resize(count);
            } else {
                supported.clear();
            }
        }
        lock_.requested_mhz = measure::pick_auto_lock_clock(
            max_sm_clock_mhz(), app_default, supported);
        log->info("--lock-clocks auto: {} MHz (max {} MHz, default application clock {} MHz)",
            lock_.requested_mhz, max_sm_clock_mhz(), app_default);
        if (lock_.requested_mhz == 0) {
            lock_.error = "could not read the max SM clock";
            log->warn("--lock-clocks: {}, running unlocked", lock_.error);
            return lock_;
        }
    }

    // Armed before the call, so a signal that arrives just after the lock
    // takes effect still releases it.
    g_locked_device = device_;
    install_reset_handlers();
    g_clocks_locked = true;
    const nvmlReturn_t rc = nvmlDeviceSetGpuLockedClocks(
        device_, lock_.requested_mhz, lock_.requested_mhz);
    if (rc != NVML_SUCCESS) {
        g_clocks_locked = false;
        lock_.error = nvmlErrorString(rc);
        if (rc == NVML_ERROR_NO_PERMISSION) lock_.error += " (needs root)";
        log->warn("--lock-clocks {} MHz failed: {}; running unlocked",
            lock_.requested_mhz, lock_.error);
        return lock_;
    }

    lock_.applied = true;
    log->info("GPU clock locked to {} MHz (a ceiling: the boost limit, power and thermal "
              "limits can still hold it lower, see environment.clock_lock.achieved_mhz)",
              lock_.requested_mhz);
    return lock_;
}

}

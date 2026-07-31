#include "arena/measure.hpp"

#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <string>

namespace arena::measure {

namespace {

// RAII pair of CUDA events, created once and reused across iterations.
class EventTimer {
public:
    EventTimer() {
        cuEventCreate(&start_, CU_EVENT_DEFAULT);
        cuEventCreate(&stop_,  CU_EVENT_DEFAULT);
    }
    ~EventTimer() {
        cuEventDestroy(start_);
        cuEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    float time(const LaunchFn& launch_fn) {
        cuEventRecord(start_, nullptr);
        launch_fn();
        cuEventRecord(stop_, nullptr);
        cuEventSynchronize(stop_);
        float ms = 0.0f;
        cuEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    CUevent start_{}, stop_{};
};

}

WarmupResult warmup(const LaunchFn& launch_fn, const LaunchFn& reset_fn,
                    const RunConfig& config) {
    WarmupResult out;
    EventTimer timer;
    auto log = spdlog::get("benchmark");

    nvtxRangePushA("WARMUP");

    if (config.warmup_mode == RunConfig::WarmupMode::Fixed) {
        for (int i = 0; i < config.warmup_runs; i++) {
            if (reset_fn) reset_fn();
            float ms = timer.time(launch_fn);
            if (i == 0) out.first_launch_ms = ms;
            out.iterations++;
        }
        out.converged = true;   // fixed mode makes no claim; treat as satisfied
        nvtxRangePop();
        return out;
    }

    std::vector<float> history;
    history.reserve(config.warmup_max);
    const auto wall_start = std::chrono::steady_clock::now();

    for (int i = 0; i < config.warmup_max; i++) {
        if (reset_fn) reset_fn();
        float ms = timer.time(launch_fn);
        if (i == 0) out.first_launch_ms = ms;
        history.push_back(ms);
        out.iterations++;

        if (out.iterations < config.warmup_min) continue;

        auto drift = check_drift(history, config.drift_window, config.drift_threshold);
        if (drift.converged) {
            out.converged = true;
            log->debug("warmup converged after {} iterations (drift {:.3f}%)",
                out.iterations, drift.rel_change * 100.0f);
            break;
        }

        const float wall_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - wall_start).count();
        if (wall_ms >= config.warmup_max_ms) {
            log->debug("warmup hit wall-clock cap ({:.0f} ms) after {} iterations "
                       "without converging", config.warmup_max_ms, out.iterations);
            break;
        }
    }

    if (!out.converged && out.iterations >= config.warmup_max) {
        log->debug("warmup hit iteration cap ({}) without converging", config.warmup_max);
    }

    nvtxRangePop();
    return out;
}

TimingResult time_operation(const LaunchFn& launch_fn, const LaunchFn& reset_fn,
                            int runs) {
    TimingResult out;
    if (runs <= 0) return out;

    out.all_times_ms.reserve(runs);
    EventTimer timer;

    for (int i = 0; i < runs; i++) {
        if (reset_fn) reset_fn();
        nvtxRangePushA(("Run " + std::to_string(i)).c_str());
        out.all_times_ms.push_back(timer.time(launch_fn));
        nvtxRangePop();
    }

    out.median_ms = median(out.all_times_ms);
    return out;
}

}

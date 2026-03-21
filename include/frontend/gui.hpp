#pragma once

#include "arena/runner.hpp"
#include "arena/kernel_descriptor.hpp"
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>

namespace frontend {

struct KernelState {
    arena::KernelDescriptor* descriptor = nullptr;
    arena::RunResult result;
    bool selected = true;
    bool has_run = false;
};

struct LogEntry {
    enum Level { INFO, WARN, ERR };
    Level level;
    std::string message;
};

struct SizedResult {
    int problem_size = 0;
    arena::RunResult result;
};

class Gui {
public:
    Gui(arena::Runner& runner);
    ~Gui();

    void run();

private:
    void init_window();
    void shutdown();
    void render_frame();
    void apply_scale();

    void render_device_info();
    void render_category_selector();
    void render_kernel_list();
    void render_problem_config();
    void render_results_table();
    void render_performance_chart();
    void render_profiling_chart();
    void render_scaling_chart();
    void render_controls();
    void render_log();

    void run_selected_kernels();
    void reset_results();
    void refresh_kernels();
    void select_category(const std::string& category);
    void log(LogEntry::Level level, const std::string& msg);

    void benchmark_thread_func(std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
                               arena::RunConfig config);
    void drain_pending_results();

    bool is_matmul() const { return current_category_ == "matmul"; }
    std::vector<KernelState>* current_kernels();

    arena::Runner& runner_;

    GLFWwindow* window_ = nullptr;
    bool running_ = false;

    std::vector<std::string> categories_;
    std::string current_category_;
    std::map<std::string, std::vector<KernelState>> kernels_by_category_;

    arena::RunConfig config_;
    bool lock_square_ = true;
    float ui_scale_ = 1.0f;
    bool scale_changed_ = false;

    std::deque<LogEntry> log_entries_;
    static constexpr size_t MAX_LOG_ENTRIES = 200;

    float results_height_ = 0;
    float performance_height_ = 0;
    float profiling_height_ = 0;
    float scaling_height_ = 0;
    float log_height_ = 0;

    std::thread benchmark_thread_;
    std::mutex mutex_;
    std::atomic<bool> benchmark_running_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<int> benchmark_current_{0};
    std::atomic<int> benchmark_total_{0};
    std::string benchmark_current_name_;  // guarded by mutex_

    struct PendingResult {
        std::string category;
        std::string kernel_name;
        arena::RunResult result;
        std::vector<LogEntry> logs;
    };
    std::vector<PendingResult> pending_results_;  // guarded by mutex_

    enum class ScalingMetric { Performance, WallTime, GpuTime };
    ScalingMetric scaling_metric_ = ScalingMetric::Performance;

    std::map<std::string, std::map<std::string, std::vector<SizedResult>>> scaling_history_;
};

int run_gui(arena::Runner& runner);

}

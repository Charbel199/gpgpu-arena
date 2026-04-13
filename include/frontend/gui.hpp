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
#include <array>

namespace frontend {

enum class DSLType { CUDA, Triton, CuTile, Warp, CUB };
enum class LogFilter { All, ErrorsOnly, CurrentKernelOnly };

struct LogEntry {
    enum Level { INFO, WARN, ERR, COMPILE, BENCHMARK, PROFILE };
    Level level;
    std::string message;
};

struct KernelState {
    arena::KernelDescriptor* descriptor = nullptr;
    arena::RunResult result;
    bool selected = true;
    bool has_run = false;
};

struct SizedResult {
    int problem_size = 0;
    arena::RunResult result;
};

// Circular buffer for per-kernel timing history across multiple benchmark runs
template<typename T, size_t Cap>
struct RingBuffer {
    std::array<T, Cap> data{};
    size_t head = 0;
    size_t count = 0;

    void push(T v) {
        data[head] = v;
        head = (head + 1) % Cap;
        if (count < Cap) count++;
    }
    void clear() { head = 0; count = 0; }
    size_t size() const { return count; }
    T operator[](size_t i) const { return data[(head + Cap - count + i) % Cap]; }
};

struct UIState {
    std::string selected_kernel_name;
    std::string selected_category;
    LogFilter log_filter = LogFilter::All;
    bool autoscroll = true;
    bool log_collapsed = false;
};

class Gui {
public:
    Gui(arena::Runner& runner);
    ~Gui();
    void run();

private:
    // Lifecycle
    void init_window();
    void shutdown();
    void render_frame();
    void apply_scale();

    // Dashboard panels
    void render_header_bar();
    void render_kernel_sidebar();
    void render_benchmark_panel();
    void render_results_table();
    void render_profile_sidebar();
    void render_log_panel();
    void render_problem_config();
    void render_run_controls();

    // Helpers
    void render_kpi_card(int id, const char* label, const char* value_str, const char* unit,
                         float pct_of_peak, bool available, const char* tooltip = nullptr);
    void render_dsl_badge(DSLType type);
    DSLType detect_dsl_type(const arena::KernelDescriptor* desc) const;
    static void format_time(float ms, char* buf, size_t buf_size);
    const KernelState* selected_kernel() const;
    const char* dsl_type_name(DSLType type) const;
    void export_results_csv();
    std::string read_kernel_source(const std::string& rel_path);

    // Actions
    void run_selected_kernels();
    void run_sweep();
    void reset_results();
    void refresh_kernels();
    void select_category(const std::string& category);
    void log(LogEntry::Level level, const std::string& msg);

    // Threading
    void benchmark_thread_func(std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
                               arena::RunConfig config);
    void sweep_thread_func(std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
                           std::vector<std::map<std::string, int>> sweep_configs,
                           arena::RunConfig config);
    void drain_pending_results();
    bool is_matmul() const { return current_category_ == "matmul"; }
    std::vector<KernelState>* current_kernels();

    // Core state
    arena::Runner& runner_;
    GLFWwindow* window_ = nullptr;
    bool running_ = false;

    // Category and kernel management
    std::vector<std::string> categories_;
    std::string current_category_;
    std::map<std::string, std::vector<KernelState>> kernels_by_category_;

    // Configuration
    arena::RunConfig config_;
    bool lock_square_ = true;
    float ui_scale_ = 1.0f;
    bool scale_changed_ = false;

    // UI state
    UIState ui_state_;

    // GPU theoretical peaks (computed once at startup)
    float peak_fp32_gflops_ = 0;
    float peak_mem_bw_gbs_ = 0;

    // Source file cache for viewer
    std::map<std::string, std::string> source_cache_;

    // Logging
    std::deque<LogEntry> log_entries_;
    static constexpr size_t MAX_LOG_ENTRIES = 500;

    // Per-kernel timing ring buffers
    std::map<std::string, RingBuffer<float, 512>> timing_history_;

    // Threading
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
        std::map<std::string, int> params;
    };
    std::vector<PendingResult> pending_results_;  // guarded by mutex_

    // Scaling
    enum class ScalingMetric { Performance, WallTime, GpuTime };
    ScalingMetric scaling_metric_ = ScalingMetric::Performance;
    std::map<std::string, std::map<std::string, std::vector<SizedResult>>> scaling_history_;
};

int run_gui(arena::Runner& runner);

}

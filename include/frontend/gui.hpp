#pragma once

#include "arena/cli_args.hpp"
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

    // Which config produced `result`, so a row in the table can say what it
    // was measured at rather than leaving it to be inferred.
    std::string result_config = "default";

    // Config every later run of this kernel uses, once tuning has found one.
    // Kernels tune to different block sizes, so this has to be per kernel
    // rather than a single setting on RunConfig.
    arena::cli::TuningVariant pinned;
    bool has_pinned = false;
};

struct SizedResult {
    int problem_size = 0;
    int run_id = 0;        // which run produced it, so deleting one prunes it
    arena::RunResult result;
};

// One point on a kernel's tuning axis. For a hand-written CUDA kernel that is
// a block size and the run is a relaunch; for a DSL kernel it is a set of
// compile-time defines and the run needed a recompile. The label is what the
// UI shows, since the two axes have nothing else in common.
struct TunedResult {
    std::string label;
    arena::cli::TuningVariant variant;   // kept so the winner can be pinned
    arena::RunResult result;
};

// A completed benchmark run, kept so later runs can be measured against it.
// This is what makes tuning answerable: snapshot the untuned run, tune, run
// again, and compare. It generalises to any pair of runs, not just that one.
struct RunSnapshot {
    int         id = 0;
    std::string name;      // user-editable; defaults to the run's own summary
    std::string category;
    std::string summary;   // size, distribution and run count at capture time
    std::string taken_at;  // wall clock, so two runs of the same setup differ
    std::map<std::string, int> params;   // groups a sweep into one run per size
    bool expanded = false; // every run is on screen at once, so most start shut

    // Keyed by kernel name. A kernel that was not selected for the run simply
    // is not here, and rows without a counterpart show no comparison.
    std::map<std::string, arena::RunResult> results;
    std::map<std::string, std::string>      configs;
};

// One row of the results table. Mirrors the fields KernelState exposes so the
// table can render a recorded run exactly as it renders the live one; `live`
// is null for a recorded run, which is what disables actions that would write
// back into something already measured.
struct TableRow {
    arena::KernelDescriptor* descriptor = nullptr;
    arena::RunResult result;
    std::string result_config = "default";
    bool has_pinned = false;
    arena::cli::TuningVariant pinned;
    struct KernelState* live = nullptr;
};

// One kernel to benchmark, at one config. The config travels with the work
// item so the benchmark thread never reads kernel state the UI thread owns.
struct BenchWork {
    std::string category;
    arena::KernelDescriptor* descriptor = nullptr;
    arena::cli::TuningVariant variant;
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

    // Kernel every other row is measured against, per category, so switching
    // categories does not leave a baseline pointing at a kernel that is not
    // in the table. Empty means absolute numbers only.
    std::map<std::string, std::string> baseline_kernel;
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
    void render_results_table(const std::vector<TableRow>& rows,
                              const char* table_id, const RunSnapshot* prev);
    void render_profile_sidebar();
    void render_log_panel();
    void render_problem_config();
    void render_run_controls();
    void render_run_buttons();

    // Helpers
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
    void run_tuning();
    void apply_best_configs();
    void clear_pinned_configs();
    void commit_snapshot();
    void forget_run(int run_id);
    void sync_live_from_newest();
    void rebuild_timing_history();
    void render_runs_tab();
    // Returns the run id to delete, or 0. The caller deletes after its loop:
    // deleting here would reallocate run_history_ and dangle every pointer
    // into it, including the one the comparison column reads from.
    int  render_run_header(RunSnapshot* snap, std::vector<TableRow>& rows);
    std::vector<TableRow> rows_for(const RunSnapshot* snap);
    const RunSnapshot* compare_snapshot() const;
    void reset_results();
    void refresh_kernels();
    void select_category(const std::string& category);
    void log(LogEntry::Level level, const std::string& msg);

    // Threading
    void benchmark_thread_func(std::vector<BenchWork> work, arena::RunConfig config);
    void sweep_thread_func(std::vector<BenchWork> work,
                           std::vector<std::map<std::string, int>> sweep_configs,
                           arena::RunConfig config);
    void tuning_thread_func(std::vector<BenchWork> work, arena::RunConfig config);
    void render_tuning_section();
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
        std::string config_label = "default";
        bool is_default_config = true;
        // Set only by a tuning run, and it is what routes the result to the
        // tuning table instead of overwriting the kernel's headline result.
        std::string tuning_label;
        arena::cli::TuningVariant tuning_variant;
    };
    std::vector<PendingResult> pending_results_;  // guarded by mutex_

    // Scaling
    enum class ScalingMetric { Performance, OpTime, GpuTime };
    ScalingMetric scaling_metric_ = ScalingMetric::Performance;
    std::map<std::string, std::map<std::string, std::vector<SizedResult>>> scaling_history_;

    // Tuning: category -> kernel -> one entry per config tried.
    std::map<std::string, std::map<std::string, std::vector<TunedResult>>> tuning_history_;

    // Run history. Results accumulate into pending_ during a run and are
    // committed as one snapshot when it finishes, so a snapshot is exactly
    // the set of kernels that run measured.
    std::vector<RunSnapshot> run_history_;
    RunSnapshot pending_snapshot_;
    int  next_snapshot_id_    = 1;
    int  compare_snapshot_id_ = 0;   // 0 = compare against nothing
    static constexpr size_t MAX_SNAPSHOTS = 50;
};

int run_gui(arena::Runner& runner);

}

#pragma once

#include "arena/runner.hpp"
#include "arena/kernel_descriptor.hpp"
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <termios.h>

namespace frontend {

enum class TuiDSL { CUDA, Triton, CuTile, Warp, CUB };

struct TuiLogEntry {
    enum Level { INFO, WARN, ERR, COMPILE, BENCHMARK, PROFILE };
    Level level;
    std::string message;
};

struct TuiKernelState {
    arena::KernelDescriptor* descriptor = nullptr;
    arena::RunResult result;
    bool selected = true;
    bool has_run = false;
};

struct TuiSizedResult {
    int problem_size = 0;
    arena::RunResult result;
};

enum class TuiPanel {
    WallVsGpu,       // wall time vs GPU time comparison bars
    Throughput,      // GFLOPS / GB/s bars vs theoretical peak
    Speedup,         // side-by-side median time with speedup labels
    ProfilingCompare,// occupancy + IPC side-by-side
    Roofline,        // arithmetic-intensity vs performance
    SubKernelTimeline, // gantt of sub-kernels for selected kernel
    TimingDist,      // per-run scatter + median line for selected kernel
    Scaling,         // scaling across problem sizes
    COUNT
};

enum class TuiSortCol {
    Kernel, Wall, GPU, Perf, Regs, SHMem, Occupancy, IPC, COUNT
};

enum class TuiFocus { KernelList, Table, Panel, Log };

struct TuiPendingResult {
    std::string category;
    std::string kernel_name;
    arena::RunResult result;
    std::vector<TuiLogEntry> logs;
    std::map<std::string, int> params;
};

struct TuiCell {
    uint32_t ch = ' ';
    uint32_t fg = 0;  // 0xRRGGBB, 0 = terminal default
    uint32_t bg = 0;
    uint8_t  attrs = 0;
    bool operator==(const TuiCell& o) const {
        return ch == o.ch && fg == o.fg && bg == o.bg && attrs == o.attrs;
    }
};

class Tui {
public:
    Tui(arena::Runner& runner);
    ~Tui();
    void run();

private:
    void enter_raw_mode();
    void leave_raw_mode();
    void enter_alt_screen();
    void leave_alt_screen();
    void refresh_size();
    void refresh_kernels();
    void compute_peaks();

    void render();
    void flush_diff();
    void clear_frame();

    void render_header(int y);
    void render_kernel_list(int x, int y, int w, int h);
    void render_settings(int x, int y, int w, int h);
    void render_center(int x, int y, int w, int h);
    void render_results_table(int x, int y, int w, int& row);
    void render_kpi_strip(int x, int y, int w, int& row);
    void render_panel(int x, int y, int w, int h);
    void render_detail(int x, int y, int w, int h);
    void render_log(int x, int y, int w, int h);
    void render_footer(int y);
    void render_help_overlay();

    void panel_wall_vs_gpu(int x, int y, int w, int h);
    void panel_throughput(int x, int y, int w, int h);
    void panel_speedup(int x, int y, int w, int h);
    void panel_profiling_compare(int x, int y, int w, int h);
    void panel_roofline(int x, int y, int w, int h);
    void panel_subkernel_timeline(int x, int y, int w, int h);
    void panel_timing_dist(int x, int y, int w, int h);
    void panel_scaling(int x, int y, int w, int h);

    void put_cell(int x, int y, uint32_t ch, uint32_t fg = 0, uint32_t bg = 0, uint8_t attrs = 0);
    void put_text(int x, int y, const std::string& text, uint32_t fg = 0, uint32_t bg = 0, uint8_t attrs = 0);
    int  put_text_clipped(int x, int y, int max_w, const std::string& text, uint32_t fg = 0, uint32_t bg = 0, uint8_t attrs = 0);
    void draw_box(int x, int y, int w, int h, const std::string& title = "", uint32_t border_fg = 0, bool accent = false);
    void draw_hline(int x, int y, int w, uint32_t fg = 0);
    void fill_rect(int x, int y, int w, int h, uint32_t ch, uint32_t fg = 0, uint32_t bg = 0);
    void draw_hbar(int x, int y, int w, float pct, uint32_t filled_fg, uint32_t track_fg);
    void draw_checkbox(int x, int y, bool checked, uint32_t fg = 0);
    void draw_badge(int x, int y, TuiDSL dsl);
    int  badge_width(TuiDSL dsl) const;
    void draw_status_dot(int x, int y, bool ok, bool running, bool has_run);

    std::string fmt_time(float ms) const;
    std::string fmt_bytes(int b) const;
    std::string fmt_count(int n) const;
    TuiDSL detect_dsl(const arena::KernelDescriptor* d) const;
    uint32_t  dsl_color(TuiDSL d) const;
    const char* dsl_tag(TuiDSL d) const;
    const char* panel_name(TuiPanel p) const;
    const char* sort_name(TuiSortCol c) const;

    enum Key {
        K_NONE = 0, K_CHAR,
        K_UP, K_DOWN, K_LEFT, K_RIGHT,
        K_PGUP, K_PGDN, K_HOME, K_END,
        K_ENTER, K_ESC, K_TAB, K_STAB, K_SPACE, K_BSPC, K_DEL
    };
    struct KeyEvent { Key key = K_NONE; int ch = 0; };
    KeyEvent read_key(int timeout_ms);
    void handle_key(const KeyEvent& ev);

    void move_selection(int delta);
    void toggle_kernel_selected();
    void toggle_kernel_focus();
    void select_all(bool all);
    void run_selected();
    void run_sweep();
    void cancel_benchmark();
    void reset_results();
    void export_csv();
    void cycle_panel(int dir);
    void cycle_sort(int dir);
    void adjust_problem_size(int dir, bool big_step);
    void adjust_warmup(int dir);
    void adjust_runs(int dir);
    void toggle_profile();
    void toggle_log_collapsed();
    void toggle_help();
    void cycle_category(int dir);
    void jump_category(int idx);
    void clear_cache();
    void log(TuiLogEntry::Level lv, const std::string& msg);

    std::vector<TuiKernelState>* current_kernels();
    const TuiKernelState* selected_kernel() const;
    std::vector<int> sorted_table_indices() const;
    bool has_any_profile() const;
    bool is_matmul() const { return current_category_ == "matmul"; }

    void benchmark_thread_func(std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
                               arena::RunConfig cfg);
    void sweep_thread_func(std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
                           std::vector<std::map<std::string, int>> sweep_cfgs,
                           arena::RunConfig cfg);
    void drain_pending_results();

    arena::Runner& runner_;

    struct termios orig_termios_{};
    bool termios_saved_ = false;
    bool alt_screen_ = false;
    int term_w_ = 80;
    int term_h_ = 24;

    std::vector<TuiCell> front_;
    std::vector<TuiCell> back_;
    std::string out_buf_;
    bool force_full_redraw_ = true;

    std::vector<std::string> categories_;
    std::string current_category_;
    std::map<std::string, std::vector<TuiKernelState>> kernels_by_category_;

    arena::RunConfig config_;
    bool lock_square_ = true;

    int  list_cursor_ = 0;
    int  list_scroll_ = 0;
    std::string selected_kernel_name_;
    TuiPanel panel_ = TuiPanel::WallVsGpu;
    TuiSortCol sort_col_ = TuiSortCol::Perf;
    bool sort_desc_ = true;
    bool log_collapsed_ = false;
    bool show_help_ = false;
    int  log_scroll_ = 0;
    TuiFocus focus_ = TuiFocus::KernelList;
    bool running_ = false;

    float peak_fp32_gflops_ = 0;
    float peak_mem_bw_gbs_  = 0;

    std::deque<TuiLogEntry> log_entries_;
    static constexpr size_t MAX_LOG_ENTRIES = 500;

    std::map<std::string, std::map<std::string, std::vector<TuiSizedResult>>> scaling_history_;

    std::thread benchmark_thread_;
    std::mutex mutex_;
    std::atomic<bool> benchmark_running_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<int>  benchmark_current_{0};
    std::atomic<int>  benchmark_total_{0};
    std::string benchmark_current_name_;  // guarded by mutex_
    std::vector<TuiPendingResult> pending_results_;  // guarded by mutex_
};

int run_tui(arena::Runner& runner);

}

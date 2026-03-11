#pragma once

#include "arena/runner.hpp"
#include "arena/kernel_descriptor.hpp"
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <map>
#include <deque>

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
    void render_controls();
    void render_log();

    void run_selected_kernels();
    void reset_results();
    void refresh_kernels();
    void select_category(const std::string& category);
    void log(LogEntry::Level level, const std::string& msg);

    arena::Runner& runner_;

    GLFWwindow* window_ = nullptr;
    bool running_ = false;

    std::vector<std::string> categories_;
    std::string current_category_;
    std::map<std::string, std::vector<KernelState>> kernels_by_category_;

    arena::RunConfig config_;
    float ui_scale_ = 1.0f;
    bool scale_changed_ = false;

    std::deque<LogEntry> log_entries_;
    static constexpr size_t MAX_LOG_ENTRIES = 200;

    std::vector<int> sorted_indices_;
};

int run_gui(arena::Runner& runner);

}

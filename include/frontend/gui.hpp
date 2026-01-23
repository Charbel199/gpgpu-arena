#pragma once

#include "arena/benchmark.hpp"
#include "arena/kernel_descriptor.hpp"
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <map>

namespace frontend {

/**
 * State for each kernel in the GUI.
 */
struct KernelState {
    arena::KernelDescriptor* descriptor = nullptr;
    arena::BenchmarkResult result;
    bool selected = true;
    bool has_run = false;
};

/**
 * ImGui-based GUI for GPGPU Arena.
 * Supports category-based kernel comparison.
 */
class Gui {
public:
    Gui(arena::Benchmark& benchmark);
    ~Gui();

    void run();

private:
    void init_window();
    void shutdown();
    void render_frame();
    void apply_scale();

    // UI panels
    void render_device_info();
    void render_category_selector();
    void render_kernel_list();
    void render_problem_config();
    void render_results_table();
    void render_performance_chart();
    void render_controls();

    // Actions
    void run_selected_kernels();
    void reset_results();
    void refresh_kernels();
    void select_category(const std::string& category);

    // Backend
    arena::Benchmark& benchmark_;

    // Window
    GLFWwindow* window_ = nullptr;
    bool running_ = false;

    // Categories and kernels
    std::vector<std::string> categories_;
    std::string current_category_;
    std::map<std::string, std::vector<KernelState>> kernels_by_category_;

    // Settings
    arena::BenchmarkConfig config_;
    float ui_scale_ = 1.0f;
    bool scale_changed_ = false;
};

int run_gui(arena::Benchmark& benchmark);

}

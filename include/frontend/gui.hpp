#pragma once

#include "arena/benchmark.hpp"
#include <GLFW/glfw3.h>
#include <vector>

namespace frontend {

struct KernelState {
    arena::KernelInfo info;
    arena::BenchmarkResult result;
    bool selected = true;
    bool has_run = false;
};

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
    void render_kernel_list();
    void render_results_table();
    void render_performance_chart();
    void render_controls();

    // Actions
    void run_selected_kernels();
    void refresh_kernels();

    // Backend
    arena::Benchmark& benchmark_;

    // Window
    GLFWwindow* window_ = nullptr;
    bool running_ = false;

    // Kernel state
    std::vector<KernelState> kernels_;

    // Settings
    arena::BenchmarkConfig config_;
    float ui_scale_ = 1.0f;
    bool scale_changed_ = false;
};

int run_gui(arena::Benchmark& benchmark);

}

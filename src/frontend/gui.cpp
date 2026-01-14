#include "frontend/gui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <stdexcept>
#include <algorithm>

namespace frontend {

Gui::Gui(arena::Benchmark& benchmark)
    : benchmark_(benchmark) {
    // Default config for matmul
    config_.params["M"] = 1024;
    config_.params["K"] = 1024;
    config_.params["N"] = 1024;
    config_.params["n"] = 1000000;
    config_.warmup_runs = 10;
    config_.number_of_runs = 10;
    
    refresh_kernels();
}

Gui::~Gui() {
    shutdown();
}

void Gui::init_window() {
    glfwSetErrorCallback([](int error, const char* desc) {
        fprintf(stderr, "GLFW Error %d: %s\n", error, desc);
    });

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(1400, 800, "GPGPU Arena", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    // Get monitor DPI scale
    float xscale, yscale;
    glfwGetWindowContentScale(window_, &xscale, &yscale);
    ui_scale_ = xscale > yscale ? xscale : yscale;
    if (ui_scale_ < 1.0f) ui_scale_ = 1.0f;
    if (ui_scale_ >= 3.0f) ui_scale_ = 4.0f;
    else if (ui_scale_ >= 1.5f) ui_scale_ = 2.0f;
    else ui_scale_ = 1.0f;

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    apply_scale();
}

void Gui::apply_scale() {
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = ui_scale_;

    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    style.ScaleAllSizes(ui_scale_);
    ImGui::StyleColorsDark();

    ImPlotStyle& plot_style = ImPlot::GetStyle();
    plot_style = ImPlotStyle();
    plot_style.PlotPadding = ImVec2(10 * ui_scale_, 10 * ui_scale_);
    plot_style.LabelPadding = ImVec2(5 * ui_scale_, 5 * ui_scale_);
    plot_style.LegendPadding = ImVec2(10 * ui_scale_, 10 * ui_scale_);
    plot_style.PlotMinSize = ImVec2(200 * ui_scale_, 150 * ui_scale_);

    scale_changed_ = false;
}

void Gui::shutdown() {
    if (window_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

void Gui::run() {
    init_window();
    running_ = true;

    while (running_ && !glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        render_frame();
    }
}

void Gui::render_frame() {
    if (scale_changed_) {
        apply_scale();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("GPGPU Arena", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    render_device_info();
    ImGui::Separator();

    // Left panel: category selector, kernel list, controls
    float panel_width = ImGui::GetContentRegionAvail().x * 0.30f;

    ImGui::BeginChild("LeftPanel", ImVec2(panel_width, 0), true);
    render_category_selector();
    ImGui::Separator();
    render_kernel_list();
    ImGui::Separator();
    render_problem_config();
    ImGui::Separator();
    render_controls();
    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel: results and chart
    ImGui::BeginChild("RightPanel", ImVec2(0, 0), true);
    render_results_table();
    ImGui::Spacing();
    render_performance_chart();
    ImGui::EndChild();

    ImGui::End();

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

void Gui::render_device_info() {
    const auto& ctx = benchmark_.context();
    ImGui::Text("GPU: %s", ctx.device_name().c_str());
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 200 * ui_scale_);
    ImGui::Text("SM %d.%d | %zu MB",
        ctx.compute_capability_major(),
        ctx.compute_capability_minor(),
        ctx.total_memory() / (1024 * 1024));
}

void Gui::render_category_selector() {
    ImGui::Text("Operation Category");
    ImGui::Spacing();

    for (const auto& cat : categories_) {
        bool is_selected = (cat == current_category_);
        std::string label = cat;
        
        // Capitalize first letter
        if (!label.empty()) {
            label[0] = std::toupper(label[0]);
        }
        
        // Show kernel count
        auto it = kernels_by_category_.find(cat);
        int count = (it != kernels_by_category_.end()) ? it->second.size() : 0;
        label += " (" + std::to_string(count) + ")";
        
        if (ImGui::Selectable(label.c_str(), is_selected)) {
            select_category(cat);
        }
    }
    
    if (categories_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No kernels registered");
    }
}

void Gui::render_kernel_list() {
    ImGui::Text("Kernels");
    ImGui::Spacing();

    if (current_category_.empty() || 
        kernels_by_category_.find(current_category_) == kernels_by_category_.end()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Select a category");
        return;
    }

    auto& kernels = kernels_by_category_[current_category_];
    
    // Select all / none buttons
    if (ImGui::SmallButton("All")) {
        for (auto& k : kernels) k.selected = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("None")) {
        for (auto& k : kernels) k.selected = false;
    }
    ImGui::Spacing();

    for (auto& k : kernels) {
        ImGui::Checkbox(k.descriptor->name().c_str(), &k.selected);
        
        // Show description on hover
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(300.0f * ui_scale_);
            ImGui::TextUnformatted(k.descriptor->description().c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
}

void Gui::render_problem_config() {
    ImGui::Text("Problem Size");
    ImGui::Spacing();

    if (current_category_ == "matmul") {
        int size = config_.params["M"];
        if (ImGui::SliderInt("Matrix Size", &size, 256, 4096)) {
            config_.params["M"] = size;
            config_.params["K"] = size;
            config_.params["N"] = size;
        }
        ImGui::Text("(%d x %d) x (%d x %d)", size, size, size, size);
    } else if (current_category_ == "reduce") {
        int n = config_.params["n"];
        if (ImGui::SliderInt("Elements (M)", &n, 100000, 100000000, "%d", ImGuiSliderFlags_Logarithmic)) {
            config_.params["n"] = n;
        }
        ImGui::Text("%d elements (%.1f MB)", n, (n * sizeof(float)) / (1024.0f * 1024.0f));
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Select a category");
    }
}

void Gui::render_controls() {
    ImGui::Text("Settings");
    ImGui::Spacing();

    ImGui::SliderInt("Warmup Runs", &config_.warmup_runs, 0, 50);
    ImGui::SliderInt("Benchmark Runs", &config_.number_of_runs, 1, 100);

    ImGui::Spacing();
    ImGui::Text("UI Scale:");
    ImGui::SameLine();
    if (ImGui::RadioButton("1x", ui_scale_ == 1.0f)) { ui_scale_ = 1.0f; scale_changed_ = true; }
    ImGui::SameLine();
    if (ImGui::RadioButton("2x", ui_scale_ == 2.0f)) { ui_scale_ = 2.0f; scale_changed_ = true; }
    ImGui::SameLine();
    if (ImGui::RadioButton("4x", ui_scale_ == 4.0f)) { ui_scale_ = 4.0f; scale_changed_ = true; }

    ImGui::Spacing();
    ImGui::Spacing();

    // Count selected kernels
    int selected_count = 0;
    if (!current_category_.empty() && 
        kernels_by_category_.find(current_category_) != kernels_by_category_.end()) {
        for (const auto& k : kernels_by_category_[current_category_]) {
            if (k.selected) selected_count++;
        }
    }

    ImGui::BeginDisabled(selected_count == 0);
    std::string btn_label = "Run Selected (" + std::to_string(selected_count) + ")";
    if (ImGui::Button(btn_label.c_str(), ImVec2(-1, 35 * ui_scale_))) {
        run_selected_kernels();
    }
    ImGui::EndDisabled();
}

void Gui::render_results_table() {
    ImGui::Text("Results");
    ImGui::Spacing();

    if (current_category_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Select a category and run benchmarks");
        return;
    }

    // Determine which columns to show based on category
    bool show_gflops = (current_category_ == "matmul");
    bool show_bandwidth = (current_category_ == "reduce");

    int num_cols = 7;  // Base columns
    
    if (ImGui::BeginTable("results", num_cols,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | 
        ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
        ImVec2(0, 200 * ui_scale_))) {

        ImGui::TableSetupColumn("Kernel", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Block", ImGuiTableColumnFlags_WidthFixed, 80 * ui_scale_);
        ImGui::TableSetupColumn("Grid", ImGuiTableColumnFlags_WidthFixed, 90 * ui_scale_);
        ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 70 * ui_scale_);
        ImGui::TableSetupColumn(show_gflops ? "GFLOPS" : "GB/s", ImGuiTableColumnFlags_WidthFixed, 70 * ui_scale_);
        ImGui::TableSetupColumn("Regs", ImGuiTableColumnFlags_WidthFixed, 50 * ui_scale_);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 60 * ui_scale_);
        ImGui::TableHeadersRow();

        auto it = kernels_by_category_.find(current_category_);
        if (it != kernels_by_category_.end()) {
            for (const auto& k : it->second) {
                if (!k.has_run) continue;

                ImGui::TableNextRow();
                
                // Kernel name
                ImGui::TableNextColumn();
                ImGui::Text("%s", k.result.kernel_name.c_str());
                if (ImGui::IsItemHovered() && !k.result.description.empty()) {
                    ImGui::SetTooltip("%s", k.result.description.c_str());
                }
                
                // Block size
                ImGui::TableNextColumn();
                ImGui::Text("%dx%d", k.result.block_x, k.result.block_y);
                
                // Grid size
                ImGui::TableNextColumn();
                ImGui::Text("%dx%d", k.result.grid_x, k.result.grid_y);
                
                // Time
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", k.result.elapsed_ms);
                
                // Performance (GFLOPS or GB/s)
                ImGui::TableNextColumn();
                if (show_gflops) {
                    ImGui::Text("%.1f", k.result.gflops);
                } else {
                    ImGui::Text("%.1f", k.result.bandwidth_gbps);
                }
                
                // Registers
                ImGui::TableNextColumn();
                ImGui::Text("%d", k.result.registers_per_thread);
                
                // Status
                ImGui::TableNextColumn();
                if (k.result.success) {
                    if (k.result.verified) {
                        ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "OK");
                    } else {
                        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.2f, 1.0f), "WARN");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Verification failed or not implemented");
                        }
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), "FAIL");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", k.result.error.c_str());
                    }
                }
            }
        }

        ImGui::EndTable();
    }
}

void Gui::render_performance_chart() {
    ImGui::Text("Performance Comparison");

    std::vector<const char*> labels;
    std::vector<double> values;

    auto it = kernels_by_category_.find(current_category_);
    if (it != kernels_by_category_.end()) {
        for (const auto& k : it->second) {
            if (k.has_run && k.result.success) {
                labels.push_back(k.result.kernel_name.c_str());
                // Use GFLOPS for compute, GB/s for memory-bound
                if (current_category_ == "matmul") {
                    values.push_back(k.result.gflops);
                } else {
                    values.push_back(k.result.bandwidth_gbps);
                }
            }
        }
    }

    if (values.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Run kernels to see comparison");
        return;
    }

    const char* y_label = (current_category_ == "matmul") ? "GFLOPS" : "GB/s";

    float plot_height = ImGui::GetContentRegionAvail().y - 10;
    if (plot_height < 150 * ui_scale_) plot_height = 150 * ui_scale_;
    
    if (ImPlot::BeginPlot("##Performance", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("", y_label, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, (double)(labels.size() - 1), (int)labels.size(), labels.data());

        std::vector<double> positions(values.size());
        for (size_t i = 0; i < positions.size(); i++) {
            positions[i] = (double)i;
        }

        ImPlot::PlotBars("Performance", positions.data(), values.data(), (int)values.size(), 0.6);
        ImPlot::EndPlot();
    }
}

void Gui::run_selected_kernels() {
    if (current_category_.empty()) return;
    
    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return;

    for (auto& k : it->second) {
        if (k.selected && k.descriptor) {
            k.result = benchmark_.run(*k.descriptor, config_);
            k.has_run = true;
        }
    }
}

void Gui::refresh_kernels() {
    categories_ = benchmark_.get_categories();
    kernels_by_category_.clear();

    for (const auto& cat : categories_) {
        auto descriptors = benchmark_.get_kernels_by_category(cat);
        std::vector<KernelState> states;
        
        for (auto* desc : descriptors) {
            KernelState state;
            state.descriptor = desc;
            state.selected = true;
            state.has_run = false;
            states.push_back(state);
        }
        
        kernels_by_category_[cat] = std::move(states);
    }

    // Select first category by default
    if (!categories_.empty() && current_category_.empty()) {
        current_category_ = categories_[0];
    }
}

void Gui::select_category(const std::string& category) {
    current_category_ = category;
}

int run_gui(arena::Benchmark& benchmark) {
    Gui gui(benchmark);
    gui.run();
    return 0;
}

}

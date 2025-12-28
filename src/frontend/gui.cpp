#include "frontend/gui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <stdexcept>

namespace frontend {

Gui::Gui(arena::Benchmark& benchmark)
    : benchmark_(benchmark) {
    config_.matrix_size = 1024;
    config_.warmup_runs = 10;
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

    window_ = glfwCreateWindow(1280, 720, "GPGPU Arena", nullptr, nullptr);
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
    // Round to nearest: 1, 2, or 4
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

    float panel_width = ImGui::GetContentRegionAvail().x * 0.35f;

    ImGui::BeginChild("LeftPanel", ImVec2(panel_width, 0), true);
    render_kernel_list();
    ImGui::Separator();
    render_controls();
    ImGui::EndChild();

    ImGui::SameLine();

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

void Gui::render_kernel_list() {
    ImGui::Text("Kernels");
    ImGui::Spacing();

    for (auto& k : kernels_) {
        ImGui::Checkbox(k.info.name.c_str(), &k.selected);
    }

    if (kernels_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No kernels found");
    }
}

void Gui::render_controls() {
    ImGui::Text("Settings");
    ImGui::Spacing();

    ImGui::SliderInt("Matrix Size", &config_.matrix_size, 256, 4096);
    ImGui::SliderInt("Warmup Runs", &config_.warmup_runs, 0, 50);

    ImGui::Text("UI Scale:");
    ImGui::SameLine();
    if (ImGui::RadioButton("1x", ui_scale_ == 1.0f)) { ui_scale_ = 1.0f; scale_changed_ = true; }
    ImGui::SameLine();
    if (ImGui::RadioButton("2x", ui_scale_ == 2.0f)) { ui_scale_ = 2.0f; scale_changed_ = true; }
    ImGui::SameLine();
    if (ImGui::RadioButton("4x", ui_scale_ == 4.0f)) { ui_scale_ = 4.0f; scale_changed_ = true; }

    ImGui::Spacing();

    if (ImGui::Button("Run Selected", ImVec2(-1, 30 * ui_scale_))) {
        run_selected_kernels();
    }

    if (ImGui::Button("Refresh Kernels", ImVec2(-1, 0))) {
        refresh_kernels();
    }
}

void Gui::render_results_table() {
    ImGui::Text("Results");
    ImGui::Spacing();

    if (ImGui::BeginTable("results", 5,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {

        ImGui::TableSetupColumn("Kernel");
        ImGui::TableSetupColumn("Time (ms)");
        ImGui::TableSetupColumn("GFLOPS");
        ImGui::TableSetupColumn("Regs");
        ImGui::TableSetupColumn("ShMem");
        ImGui::TableHeadersRow();

        for (const auto& k : kernels_) {
            if (!k.has_run) continue;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", k.info.name.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", k.result.elapsed_ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", k.result.gflops);
            ImGui::TableNextColumn();
            ImGui::Text("%d", k.result.registers_per_thread);
            ImGui::TableNextColumn();
            if (k.result.shared_memory_bytes >= 1024) {
                ImGui::Text("%.1fKB", k.result.shared_memory_bytes / 1024.0);
            } else {
                ImGui::Text("%dB", k.result.shared_memory_bytes);
            }
        }

        ImGui::EndTable();
    }
}

void Gui::render_performance_chart() {
    ImGui::Text("Performance Comparison");

    std::vector<const char*> labels;
    std::vector<double> values;

    for (const auto& k : kernels_) {
        if (k.has_run && k.result.success) {
            labels.push_back(k.info.name.c_str());
            values.push_back(k.result.gflops);
        }
    }

    if (values.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Run kernels to see results");
        return;
    }

    float plot_height = 200 * ui_scale_;
    if (ImPlot::BeginPlot("##GFLOPS", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("", "GFLOPS", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, (double)(labels.size() - 1), (int)labels.size(), labels.data());

        std::vector<double> positions(values.size());
        for (size_t i = 0; i < positions.size(); i++) positions[i] = (double)i;

        ImPlot::PlotBars("GFLOPS", positions.data(), values.data(), (int)values.size(), 0.6);
        ImPlot::EndPlot();
    }
}

void Gui::run_selected_kernels() {
    for (auto& k : kernels_) {
        if (k.selected) {
            k.result = benchmark_.run(k.info, config_);
            k.has_run = true;
        }
    }
}

void Gui::refresh_kernels() {
    kernels_.clear();
    auto infos = benchmark_.scan_kernels();
    for (const auto& info : infos) {
        KernelState state;
        state.info = info;
        state.selected = true;
        state.has_run = false;
        kernels_.push_back(state);
    }
}

int run_gui(arena::Benchmark& benchmark) {
    Gui gui(benchmark);
    gui.run();
    return 0;
}

}


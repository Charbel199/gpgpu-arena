#include "frontend/gui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <cuda.h>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

namespace frontend {

Gui::Gui(arena::Runner& runner)
    : runner_(runner) {
    config_.params["M"] = 1024;
    config_.params["K"] = 1024;
    config_.params["N"] = 1024;
    config_.params["n"] = 1000000;
    config_.warmup_runs = 10;
    config_.number_of_runs = 10;

    refresh_kernels();
}

Gui::~Gui() {
    cancel_requested_ = true;
    if (benchmark_thread_.joinable()) {
        benchmark_thread_.join();
    }
    shutdown();
}

void Gui::log(LogEntry::Level level, const std::string& msg) {
    log_entries_.push_back({level, msg});
    if (log_entries_.size() > MAX_LOG_ENTRIES) {
        log_entries_.pop_front();
    }
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

    window_ = glfwCreateWindow(1400, 900, "GPGPU Arena", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    float xscale, yscale;
    glfwGetWindowContentScale(window_, &xscale, &yscale);
    ui_scale_ = xscale > yscale ? xscale : yscale;
    if (ui_scale_ < 1.0f) ui_scale_ = 1.0f;
    if (ui_scale_ >= 3.0f) ui_scale_ = 4.0f;
    else if (ui_scale_ >= 1.5f) ui_scale_ = 2.0f;
    else ui_scale_ = 1.0f;

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
        drain_pending_results();
        render_frame();
    }
}

void Gui::benchmark_thread_func(
    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
    arena::RunConfig config) {

    // push CUDA context onto this thread (driver API contexts are thread-local)
    CUcontext cuda_ctx = runner_.context().handle();
    cuCtxPushCurrent(cuda_ctx);

    for (int i = 0; i < (int)work.size(); i++) {
        if (cancel_requested_) break;

        auto& [cat, descriptor] = work[i];
        benchmark_current_ = i;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            benchmark_current_name_ = descriptor->name();
        }

        PendingResult pr;
        pr.category = cat;
        pr.kernel_name = descriptor->name();
        pr.logs.push_back({LogEntry::INFO, "[BENCHMARK] Running " + descriptor->name() + "..."});

        pr.result = runner_.run(*descriptor, config);

        if (pr.result.success) {
            char buf[256];
            bool is_matmul = (cat == "matmul");
            snprintf(buf, sizeof(buf), "[BENCHMARK] %s - wall=%.3f ms gpu=%.3f ms | %.2f %s",
                pr.result.kernel_name.c_str(),
                pr.result.elapsed_ms,
                pr.result.kernel_ms,
                is_matmul ? pr.result.gflops : pr.result.bandwidth_gbps,
                is_matmul ? "GFLOPS" : "GB/s");
            pr.logs.push_back({LogEntry::INFO, buf});

            if (config.collect_metrics && pr.result.achieved_occupancy > 0) {
                snprintf(buf, sizeof(buf),
                    "[PROFILER]  %s - %d regs | %d B shmem | occupancy=%.1f%% | IPC=%.2f",
                    pr.result.kernel_name.c_str(),
                    pr.result.registers_per_thread,
                    pr.result.shared_memory_bytes,
                    pr.result.achieved_occupancy * 100.0,
                    pr.result.ipc);
                pr.logs.push_back({LogEntry::INFO, buf});
            }

            if (!pr.result.verified) {
                pr.logs.push_back({LogEntry::WARN, pr.result.kernel_name + " failed verification"});
            }
        } else {
            pr.logs.push_back({LogEntry::ERR, pr.result.kernel_name + " failed: " + pr.result.error});
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_results_.push_back(std::move(pr));
        }
    }

    benchmark_current_ = (int)work.size();

    CUcontext popped;
    cuCtxPopCurrent(&popped);

    benchmark_running_ = false;
}

void Gui::drain_pending_results() {
    std::vector<PendingResult> results;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        results.swap(pending_results_);
    }

    for (auto& pr : results) {
        for (auto& entry : pr.logs) {
            log(entry.level, entry.message);
        }

        auto cat_it = kernels_by_category_.find(pr.category);
        if (cat_it != kernels_by_category_.end()) {
            for (auto& k : cat_it->second) {
                if (k.descriptor && k.descriptor->name() == pr.kernel_name) {
                    k.result = pr.result;
                    k.has_run = true;
                    break;
                }
            }
        }

        if (pr.result.success) {
            int problem_size = (pr.category == "matmul") ?
                config_.params["M"] : config_.params["n"];

            auto& hist = scaling_history_[pr.category][pr.kernel_name];
            // replace existing entry for same problem size
            bool found = false;
            for (auto& entry : hist) {
                if (entry.problem_size == problem_size) {
                    entry.result = pr.result;
                    found = true;
                    break;
                }
            }
            if (!found) {
                hist.push_back({problem_size, pr.result});
                std::sort(hist.begin(), hist.end(),
                    [](const SizedResult& a, const SizedResult& b) {
                        return a.problem_size < b.problem_size;
                    });
            }
        }
    }

    if (!benchmark_running_ && benchmark_thread_.joinable()) {
        benchmark_thread_.join();
        if (benchmark_current_ >= benchmark_total_) {
            log(LogEntry::INFO, "--- Done ---");
        } else {
            log(LogEntry::WARN, "--- Cancelled ---");
        }
    }
}

std::vector<KernelState>* Gui::current_kernels() {
    auto it = kernels_by_category_.find(current_category_);
    return (it != kernels_by_category_.end()) ? &it->second : nullptr;
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

    float panel_width = ImGui::GetContentRegionAvail().x * 0.25f;

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

    ImGui::BeginChild("RightPanel", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);

    float min_h = 100 * ui_scale_;
    if (results_height_ < min_h) results_height_ = 250 * ui_scale_;
    if (performance_height_ < min_h) performance_height_ = 300 * ui_scale_;
    if (profiling_height_ < min_h) profiling_height_ = 300 * ui_scale_;
    if (scaling_height_ < min_h) scaling_height_ = 300 * ui_scale_;
    if (log_height_ < min_h) log_height_ = 200 * ui_scale_;

    auto resize_handle = [&](const char* id, float& height) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        ImGui::Button(id, ImVec2(-1, 4 * ui_scale_));
        if (ImGui::IsItemActive()) {
            height += ImGui::GetIO().MouseDelta.y;
            if (height < min_h) height = min_h;
        }
        if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        ImGui::PopStyleColor(3);
    };

    if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginChild("ResultsContent", ImVec2(0, results_height_), false);
        render_results_table();
        ImGui::EndChild();
        resize_handle("##resize_results", results_height_);
    }

    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginChild("PerformanceContent", ImVec2(0, performance_height_), false);
        render_performance_chart();
        ImGui::EndChild();
        resize_handle("##resize_performance", performance_height_);
    }

    if (ImGui::CollapsingHeader("Profiling")) {
        ImGui::BeginChild("ProfilingContent", ImVec2(0, profiling_height_), false);
        render_profiling_chart();
        ImGui::EndChild();
        resize_handle("##resize_profiling", profiling_height_);
    }

    if (ImGui::CollapsingHeader("Scaling")) {
        ImGui::BeginChild("ScalingContent", ImVec2(0, scaling_height_), false);
        render_scaling_chart();
        ImGui::EndChild();
        resize_handle("##resize_scaling", scaling_height_);
    }

    if (ImGui::CollapsingHeader("Log")) {
        ImGui::BeginChild("LogContent", ImVec2(0, log_height_), false);
        render_log();
        ImGui::EndChild();
        resize_handle("##resize_log", log_height_);
    }

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
    const auto& ctx = runner_.context();
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

        if (!label.empty()) {
            label[0] = std::toupper(label[0]);
        }

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

    auto* kernels_ptr = current_kernels();
    if (!kernels_ptr) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Select a category");
        return;
    }

    auto& kernels = *kernels_ptr;

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
        ImGui::Checkbox("Lock Square", &lock_square_);

        int m = config_.params["M"];
        int k = config_.params["K"];
        int n = config_.params["N"];

        if (lock_square_) {
            if (ImGui::SliderInt("Matrix Size", &m, 256, 4096)) {
                config_.params["M"] = m;
                config_.params["K"] = m;
                config_.params["N"] = m;
            }
            ImGui::Text("(%d x %d) x (%d x %d)", m, m, m, m);
        } else {
            bool changed = false;
            changed |= ImGui::SliderInt("M (rows A)", &m, 256, 4096);
            changed |= ImGui::SliderInt("K (shared)", &k, 256, 4096);
            changed |= ImGui::SliderInt("N (cols B)", &n, 256, 4096);
            if (changed) {
                config_.params["M"] = m;
                config_.params["K"] = k;
                config_.params["N"] = n;
            }
            ImGui::Text("(%d x %d) x (%d x %d)", m, k, k, n);
        }
    } else if (current_category_ == "reduce" || current_category_ == "scan") {
        int n = config_.params["n"];
        if (ImGui::SliderInt("Elements", &n, 100000, 100000000, "%d", ImGuiSliderFlags_Logarithmic)) {
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
    ImGui::Checkbox("Collect Profiling Data", &config_.collect_metrics);

    if (config_.collect_metrics) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f), "  (slower: kernel replay)");
    }

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

    // progress bar when running
    if (benchmark_running_) {
        int current = benchmark_current_.load();
        int total = benchmark_total_.load();
        float fraction = total > 0 ? (float)current / (float)total : 0.0f;

        std::string overlay;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            overlay = benchmark_current_name_ + " (" + std::to_string(current + 1) + "/" + std::to_string(total) + ")";
        }
        ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay.c_str());

        if (ImGui::Button("Cancel", ImVec2(-1, 30 * ui_scale_))) {
            cancel_requested_ = true;
        }
    } else {
        int selected_count = 0;
        if (auto* kernels = current_kernels()) {
            for (const auto& k : *kernels)
                if (k.selected) selected_count++;
        }

        ImGui::BeginDisabled(selected_count == 0);
        char btn_label[64];
        snprintf(btn_label, sizeof(btn_label), "Run Selected (%d)", selected_count);
        if (ImGui::Button(btn_label, ImVec2(-1, 35 * ui_scale_))) {
            run_selected_kernels();
        }
        ImGui::EndDisabled();
    }

    ImGui::Spacing();

    bool has_results = false;
    if (auto* kernels = current_kernels()) {
        for (const auto& k : *kernels)
            if (k.has_run) { has_results = true; break; }
    }

    ImGui::BeginDisabled(!has_results || benchmark_running_);
    if (ImGui::Button("Reset Results", ImVec2(-1, 30 * ui_scale_))) {
        reset_results();
    }
    ImGui::EndDisabled();
}

void Gui::render_results_table() {
    auto* kernels = current_kernels();
    if (!kernels) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Select a category and run benchmarks");
        return;
    }

    bool show_gflops = is_matmul();
    bool has_profiling = false;
    for (const auto& k : *kernels) {
        if (k.has_run && k.result.registers_per_thread > 0) {
            has_profiling = true;
            break;
        }
    }

    enum ColumnID { Col_Kernel = 0, Col_Block, Col_Grid, Col_Wall, Col_GPU, Col_Perf, Col_Status,
                    Col_Regs, Col_SHMem, Col_Occup, Col_IPC };

    int num_cols = has_profiling ? 11 : 8;

    if (ImGui::BeginTable("results", num_cols,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate,
        ImVec2(0, ImGui::GetContentRegionAvail().y))) {

        ImGui::TableSetupColumn("Kernel", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_DefaultSort, 0, Col_Kernel);
        ImGui::TableSetupColumn("Block", ImGuiTableColumnFlags_WidthFixed, 70 * ui_scale_, Col_Block);
        ImGui::TableSetupColumn("Grid", ImGuiTableColumnFlags_WidthFixed, 80 * ui_scale_, Col_Grid);
        ImGui::TableSetupColumn("Wall (ms)", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 65 * ui_scale_, Col_Wall);
        ImGui::TableSetupColumn("GPU (ms)", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 60 * ui_scale_, Col_GPU);
        ImGui::TableSetupColumn(show_gflops ? "GFLOPS" : "GB/s", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 60 * ui_scale_, Col_Perf);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, 50 * ui_scale_, Col_Status);

        if (has_profiling) {
            ImGui::TableSetupColumn("Regs", ImGuiTableColumnFlags_WidthFixed, 40 * ui_scale_, Col_Regs);
            ImGui::TableSetupColumn("SHMem", ImGuiTableColumnFlags_WidthFixed, 55 * ui_scale_, Col_SHMem);
            ImGui::TableSetupColumn("Occup%", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 55 * ui_scale_, Col_Occup);
            ImGui::TableSetupColumn("IPC", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 45 * ui_scale_, Col_IPC);
        } else {
            ImGui::TableSetupColumn("Regs", ImGuiTableColumnFlags_WidthFixed, 45 * ui_scale_, Col_Regs);
        }

        ImGui::TableHeadersRow();

        if (cat_it != kernels_by_category_.end()) {
            sorted_indices_.clear();
            for (int i = 0; i < (int)cat_it->second.size(); i++) {
                if (cat_it->second[i].has_run)
                    sorted_indices_.push_back(i);
            }

            if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
                if (sort_specs->SpecsCount > 0) {
                    const auto& spec = sort_specs->Specs[0];
                    const auto& kernels = cat_it->second;
                    bool ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);

                    std::sort(sorted_indices_.begin(), sorted_indices_.end(),
                        [&](int a, int b) {
                            const auto& ra = kernels[a].result;
                            const auto& rb = kernels[b].result;
                            int cmp = 0;
                            switch (spec.ColumnUserID) {
                                case Col_Kernel:   cmp = ra.kernel_name.compare(rb.kernel_name); break;
                                case Col_Block:    cmp = (ra.block_x * ra.block_y) - (rb.block_x * rb.block_y); break;
                                case Col_Grid:     cmp = (ra.grid_x * ra.grid_y) - (rb.grid_x * rb.grid_y); break;
                                case Col_Wall:     cmp = (ra.elapsed_ms < rb.elapsed_ms) ? -1 : (ra.elapsed_ms > rb.elapsed_ms) ? 1 : 0; break;
                                case Col_GPU:      cmp = (ra.kernel_ms < rb.kernel_ms) ? -1 : (ra.kernel_ms > rb.kernel_ms) ? 1 : 0; break;
                                case Col_Perf: {
                                    double va = show_gflops ? ra.gflops : ra.bandwidth_gbps;
                                    double vb = show_gflops ? rb.gflops : rb.bandwidth_gbps;
                                    cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
                                    break;
                                }
                                case Col_Regs:  cmp = ra.registers_per_thread - rb.registers_per_thread; break;
                                case Col_SHMem: cmp = ra.shared_memory_bytes - rb.shared_memory_bytes; break;
                                case Col_Occup: cmp = (ra.achieved_occupancy < rb.achieved_occupancy) ? -1 : (ra.achieved_occupancy > rb.achieved_occupancy) ? 1 : 0; break;
                                case Col_IPC:   cmp = (ra.ipc < rb.ipc) ? -1 : (ra.ipc > rb.ipc) ? 1 : 0; break;
                                default: break;
                            }
                            return ascending ? (cmp < 0) : (cmp > 0);
                        });
                }
                sort_specs->SpecsDirty = false;
            }

            for (int idx : sorted_indices_) {
                const auto& k = cat_it->second[idx];

                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("%s %s", k.result.uses_ptx ? "[PTX]" : "[RT]", k.result.kernel_name.c_str());
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("%s", k.result.description.c_str());
                    ImGui::Separator();
                    ImGui::Text("Type: %s", k.result.uses_ptx ? "PTX (Driver API)" : "Compiled (Runtime API)");
                    if (!k.result.sub_kernels.empty()) {
                        ImGui::Text("GPU Kernels (%zu):", k.result.sub_kernels.size());
                        for (const auto& sk : k.result.sub_kernels) {
                            ImGui::Text("  %.3f ms | %d regs | %d B shmem | %s",
                                sk.duration_ms, sk.registers, sk.shared_memory, sk.name.c_str());
                        }
                    }
                    ImGui::EndTooltip();
                }

                ImGui::TableNextColumn();
                ImGui::Text("%dx%d", k.result.block_x, k.result.block_y);

                ImGui::TableNextColumn();
                ImGui::Text("%dx%d", k.result.grid_x, k.result.grid_y);

                ImGui::TableNextColumn();
                ImGui::Text("%.3f", k.result.elapsed_ms);

                ImGui::TableNextColumn();
                ImGui::Text("%.3f", k.result.kernel_ms);
                if (ImGui::IsItemHovered() && k.result.kernel_ms > 0 && k.result.elapsed_ms > k.result.kernel_ms * 1.05f) {
                    float overhead_pct = ((k.result.elapsed_ms - k.result.kernel_ms) / k.result.elapsed_ms) * 100.0f;
                    ImGui::SetTooltip("Host overhead: %.1f%% (%.3f ms)", overhead_pct, k.result.elapsed_ms - k.result.kernel_ms);
                }

                ImGui::TableNextColumn();
                if (show_gflops) {
                    ImGui::Text("%.1f", k.result.gflops);
                } else {
                    ImGui::Text("%.1f", k.result.bandwidth_gbps);
                }

                ImGui::TableNextColumn();
                if (k.result.success) {
                    if (k.result.verified) {
                        ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "OK");
                    } else {
                        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.2f, 1.0f), "WARN");
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), "FAIL");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", k.result.error.c_str());
                    }
                }

                if (has_profiling) {
                    ImGui::TableNextColumn();
                    if (k.result.registers_per_thread > 0)
                        ImGui::Text("%d", k.result.registers_per_thread);
                    else
                        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");

                    ImGui::TableNextColumn();
                    if (k.result.shared_memory_bytes > 0)
                        ImGui::Text("%d", k.result.shared_memory_bytes);
                    else
                        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");

                    ImGui::TableNextColumn();
                    if (k.result.achieved_occupancy > 0)
                        ImGui::Text("%.1f", k.result.achieved_occupancy * 100.0);
                    else
                        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");

                    ImGui::TableNextColumn();
                    if (k.result.ipc > 0)
                        ImGui::Text("%.2f", k.result.ipc);
                    else
                        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
                } else {
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", k.result.registers_per_thread);
                }
            }
        }

        ImGui::EndTable();
    }
}

void Gui::render_performance_chart() {
    std::vector<std::string> label_strings;
    std::vector<double> values;

    auto it = kernels_by_category_.find(current_category_);
    if (it != kernels_by_category_.end()) {
        for (const auto& k : it->second) {
            if (k.has_run && k.result.success) {
                label_strings.push_back(k.result.kernel_name);
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

    // sort by value descending
    std::vector<int> order(values.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return values[a] > values[b]; });

    std::vector<const char*> sorted_labels(values.size());
    std::vector<double> sorted_values(values.size());
    for (size_t i = 0; i < order.size(); i++) {
        sorted_labels[i] = label_strings[order[i]].c_str();
        sorted_values[i] = values[order[i]];
    }

    const char* y_label = (current_category_ == "matmul") ? "GFLOPS" : "GB/s";

    if (ImPlot::BeginPlot("##Performance", ImVec2(-1, ImGui::GetContentRegionAvail().y))) {
        ImPlot::SetupAxes("", y_label, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, (double)(sorted_labels.size() - 1), (int)sorted_labels.size(), sorted_labels.data());

        std::vector<double> positions(sorted_values.size());
        for (size_t i = 0; i < positions.size(); i++) {
            positions[i] = (double)i;
        }

        ImPlot::PlotBars("Performance", positions.data(), sorted_values.data(), (int)sorted_values.size(), 0.6);
        ImPlot::EndPlot();
    }
}

void Gui::render_profiling_chart() {
    std::vector<const char*> labels;
    std::vector<double> occupancy;
    std::vector<double> ipc;

    auto it = kernels_by_category_.find(current_category_);
    if (it != kernels_by_category_.end()) {
        for (const auto& k : it->second) {
            if (k.has_run && k.result.success && k.result.achieved_occupancy > 0) {
                labels.push_back(k.result.kernel_name.c_str());
                occupancy.push_back(k.result.achieved_occupancy * 100.0);
                ipc.push_back(k.result.ipc);
            }
        }
    }

    if (labels.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Enable 'Collect Profiling Data' and run kernels to see profiling charts");
        return;
    }

    int n = (int)labels.size();
    float avail = ImGui::GetContentRegionAvail().y;
    float plot_height = (avail - 10 * ui_scale_) * 0.5f;
    if (plot_height < 100 * ui_scale_) plot_height = 100 * ui_scale_;

    std::vector<double> positions(n);
    for (int i = 0; i < n; i++) positions[i] = (double)i;

    if (ImPlot::BeginPlot("Occupancy & IPC", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("", "Occupancy %", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y2, "IPC", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, (double)(n - 1), n, labels.data());

        double bar_w = 0.3;
        std::vector<double> pos_left(n), pos_right(n);
        for (int i = 0; i < n; i++) {
            pos_left[i] = positions[i] - bar_w * 0.5;
            pos_right[i] = positions[i] + bar_w * 0.5;
        }

        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
        ImPlot::PlotBars("Occupancy %", pos_left.data(), occupancy.data(), n, bar_w);

        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
        ImPlot::PlotBars("IPC", pos_right.data(), ipc.data(), n, bar_w);

        ImPlot::EndPlot();
    }

    ImGui::Spacing();

    // scatter: occupancy vs IPC
    if (ImPlot::BeginPlot("Occupancy vs IPC", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("Occupancy %", "IPC", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 100, ImPlotCond_Always);

        for (int i = 0; i < n; i++) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8 * ui_scale_);
            ImPlot::PlotScatter(labels[i], &occupancy[i], &ipc[i], 1);
        }

        ImPlot::EndPlot();
    }
}

void Gui::render_scaling_chart() {
    auto cat_it = scaling_history_.find(current_category_);
    if (cat_it == scaling_history_.end() || cat_it->second.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Run benchmarks at different problem sizes to see scaling comparison");
        return;
    }

    // check if any kernel has >1 data point
    bool has_multi = false;
    for (const auto& [name, hist] : cat_it->second) {
        if (hist.size() > 1) { has_multi = true; break; }
    }

    if (!has_multi) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Run at multiple problem sizes to see scaling (currently 1 size per kernel)");
        return;
    }

    bool is_matmul = (current_category_ == "matmul");

    // metric selector
    const char* metric_names[] = { "Performance", "Wall Time", "GPU Time" };
    int metric_idx = (int)scaling_metric_;
    ImGui::SetNextItemWidth(160 * ui_scale_);
    if (ImGui::Combo("Metric##scaling", &metric_idx, metric_names, 3)) {
        scaling_metric_ = (ScalingMetric)metric_idx;
    }

    const char* x_label = is_matmul ? "Matrix Size" : "Elements";
    const char* y_label;
    switch (scaling_metric_) {
        case ScalingMetric::Performance: y_label = is_matmul ? "GFLOPS" : "GB/s"; break;
        case ScalingMetric::WallTime:    y_label = "Wall Time (ms)"; break;
        case ScalingMetric::GpuTime:     y_label = "GPU Time (ms)"; break;
    }

    if (ImPlot::BeginPlot("##Scaling", ImVec2(-1, ImGui::GetContentRegionAvail().y))) {
        ImPlot::SetupAxes(x_label, y_label, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        for (const auto& [name, hist] : cat_it->second) {
            if (hist.size() < 2) continue;

            std::vector<double> xs, ys;
            for (const auto& entry : hist) {
                xs.push_back((double)entry.problem_size);
                switch (scaling_metric_) {
                    case ScalingMetric::Performance:
                        ys.push_back(is_matmul ? entry.result.gflops : entry.result.bandwidth_gbps);
                        break;
                    case ScalingMetric::WallTime:
                        ys.push_back((double)entry.result.elapsed_ms);
                        break;
                    case ScalingMetric::GpuTime:
                        ys.push_back((double)entry.result.kernel_ms);
                        break;
                }
            }

            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4 * ui_scale_);
            ImPlot::PlotLine(name.c_str(), xs.data(), ys.data(), (int)xs.size());
        }

        ImPlot::EndPlot();
    }
}

void Gui::render_log() {
    ImGui::BeginChild("LogScroll", ImVec2(0, ImGui::GetContentRegionAvail().y), false);
    for (const auto& entry : log_entries_) {
        ImVec4 color;
        const char* prefix;
        switch (entry.level) {
            case LogEntry::INFO: color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f); prefix = "[INFO] "; break;
            case LogEntry::WARN: color = ImVec4(0.9f, 0.9f, 0.2f, 1.0f); prefix = "[WARN] "; break;
            case LogEntry::ERR:  color = ImVec4(0.9f, 0.2f, 0.2f, 1.0f); prefix = "[ERR]  "; break;
        }
        ImGui::TextColored(color, "%s%s", prefix, entry.message.c_str());
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
}

void Gui::run_selected_kernels() {
    if (current_category_.empty()) return;
    if (benchmark_running_) return;

    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return;

    // build work list
    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work;
    for (auto& k : it->second) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor});
        }
    }

    if (work.empty()) return;

    // join previous thread if any
    if (benchmark_thread_.joinable()) {
        benchmark_thread_.join();
    }

    cancel_requested_ = false;
    benchmark_running_ = true;
    benchmark_current_ = 0;
    benchmark_total_ = (int)work.size();

    benchmark_thread_ = std::thread(&Gui::benchmark_thread_func, this, std::move(work), config_);
}

void Gui::reset_results() {
    if (current_category_.empty()) return;

    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return;

    for (auto& k : it->second) {
        k.has_run = false;
        k.result = arena::RunResult{};
    }
    log(LogEntry::INFO, "Results reset");
}

void Gui::refresh_kernels() {
    categories_ = runner_.get_categories();
    kernels_by_category_.clear();

    for (const auto& cat : categories_) {
        auto descriptors = runner_.get_kernels_by_category(cat);
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

    if (!categories_.empty() && current_category_.empty()) {
        current_category_ = categories_[0];
    }
}

void Gui::select_category(const std::string& category) {
    current_category_ = category;
}

int run_gui(arena::Runner& runner) {
    Gui gui(runner);
    gui.run();
    return 0;
}

}

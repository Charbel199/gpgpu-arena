#include "frontend/gui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <cuda.h>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cmath>

namespace frontend {

// ============================================================================
// Theme colors  dark GPU/compute aesthetic
// ============================================================================
namespace UITheme {
    constexpr ImVec4 ACCENT        = {0.0f,  0.831f, 0.667f, 1.0f};   // #00D4AA teal-green
    constexpr ImVec4 ACCENT_DIM    = {0.0f,  0.35f,  0.28f,  1.0f};

    constexpr ImVec4 CUDA_BADGE    = {0.463f, 0.725f, 0.0f,   1.0f};  // #76B900 NVIDIA green
    constexpr ImVec4 TRITON_BADGE  = {1.0f,   0.42f,  0.169f, 1.0f};  // #FF6B2B orange
    constexpr ImVec4 CUTILE_BADGE  = {0.357f, 0.608f, 0.835f, 1.0f};  // #5B9BD5 blue
    constexpr ImVec4 WARP_BADGE    = {0.6f,   0.4f,   0.8f,   1.0f};  // purple
    constexpr ImVec4 CUB_BADGE     = {0.55f,  0.55f,  0.55f,  1.0f};  // gray

    constexpr ImVec4 ERROR_RED     = {1.0f,   0.267f, 0.267f, 1.0f};  // #FF4444
    constexpr ImVec4 WARN_YELLOW   = {1.0f,   0.702f, 0.0f,   1.0f};  // #FFB300
    constexpr ImVec4 SUCCESS_GREEN = {0.0f,   0.784f, 0.325f, 1.0f};  // #00C853

    constexpr ImVec4 HEADER_TEXT   = {0.9f,   0.9f,   0.9f,   1.0f};
    constexpr ImVec4 BODY_TEXT     = {0.78f,  0.78f,  0.78f,  1.0f};
    constexpr ImVec4 TEXT_DIM      = {0.5f,   0.5f,   0.5f,   1.0f};

    constexpr ImVec4 LOG_INFO      = {0.6f,   0.6f,   0.6f,   1.0f};
    constexpr ImVec4 LOG_COMPILE   = {0.0f,   0.8f,   0.8f,   1.0f};  // cyan
    constexpr ImVec4 LOG_BENCHMARK = {0.9f,   0.9f,   0.9f,   1.0f};  // white
    constexpr ImVec4 LOG_PROFILE   = {0.6f,   0.4f,   0.8f,   1.0f};  // purple
    constexpr ImVec4 LOG_WARN      = {1.0f,   0.702f, 0.0f,   1.0f};  // yellow
    constexpr ImVec4 LOG_ERROR     = {1.0f,   0.267f, 0.267f, 1.0f};  // red
}

// ============================================================================
// Layout constants (before ui_scale_ multiplication)
// ============================================================================
namespace Layout {
    constexpr float HEADER_HEIGHT       = 44.0f;
    constexpr float SIDEBAR_LEFT_WIDTH  = 260.0f;
    constexpr float SIDEBAR_RIGHT_WIDTH = 290.0f;
    constexpr float LOG_HEIGHT          = 160.0f;
    constexpr float LOG_COLLAPSED_HEIGHT = 28.0f;
    constexpr float KPI_CARD_WIDTH      = 145.0f;
    constexpr float KPI_CARD_HEIGHT     = 90.0f;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================
Gui::Gui(arena::Runner& runner)
    : runner_(runner) {
    config_.params["M"] = 1024;
    config_.params["K"] = 1024;
    config_.params["N"] = 1024;
    config_.params["n"] = 1000000;
    config_.params["rows"] = 1024;
    config_.params["cols"] = 1024;
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

// ============================================================================
// Logging
// ============================================================================
void Gui::log(LogEntry::Level level, const std::string& msg) {
    log_entries_.push_back({level, msg});
    if (log_entries_.size() > MAX_LOG_ENTRIES) {
        log_entries_.pop_front();
    }
}

// ============================================================================
// Window init / shutdown / scale
// ============================================================================
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

    window_ = glfwCreateWindow(1700, 1000, "GPGPU Arena", nullptr, nullptr);
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

void Gui::apply_scale() {
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = ui_scale_;

    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    style.ScaleAllSizes(ui_scale_);

    // FrameRounding = 6.0f throughout
    style.FrameRounding    = 6.0f * ui_scale_;
    style.GrabRounding     = 4.0f * ui_scale_;
    style.ChildRounding    = 6.0f * ui_scale_;
    style.PopupRounding    = 6.0f * ui_scale_;
    style.ScrollbarRounding = 4.0f * ui_scale_;
    style.WindowRounding   = 0.0f;

    ImGui::StyleColorsDark();

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg]             = {0.067f, 0.067f, 0.067f, 1.0f};  // #111111
    c[ImGuiCol_ChildBg]              = {0.051f, 0.051f, 0.051f, 1.0f};  // #0D0D0D
    c[ImGuiCol_Border]               = {0.15f,  0.15f,  0.15f,  1.0f};
    c[ImGuiCol_FrameBg]              = {0.10f,  0.10f,  0.10f,  1.0f};
    c[ImGuiCol_FrameBgHovered]       = {0.15f,  0.15f,  0.15f,  1.0f};
    c[ImGuiCol_FrameBgActive]        = {0.20f,  0.20f,  0.20f,  1.0f};
    c[ImGuiCol_TitleBg]              = {0.051f, 0.051f, 0.051f, 1.0f};
    c[ImGuiCol_TitleBgActive]        = {0.051f, 0.051f, 0.051f, 1.0f};
    c[ImGuiCol_Header]               = {0.10f,  0.10f,  0.10f,  1.0f};
    c[ImGuiCol_HeaderHovered]        = {0.15f,  0.15f,  0.15f,  1.0f};
    c[ImGuiCol_HeaderActive]         = {0.20f,  0.20f,  0.20f,  1.0f};
    c[ImGuiCol_Button]               = {0.0f,   0.40f,  0.32f,  1.0f};
    c[ImGuiCol_ButtonHovered]        = {0.0f,   0.55f,  0.44f,  1.0f};
    c[ImGuiCol_ButtonActive]         = UITheme::ACCENT;
    c[ImGuiCol_CheckMark]            = UITheme::ACCENT;
    c[ImGuiCol_SliderGrab]           = UITheme::ACCENT;
    c[ImGuiCol_SliderGrabActive]     = {0.0f,   1.0f,   0.8f,   1.0f};
    c[ImGuiCol_ScrollbarBg]          = {0.05f,  0.05f,  0.05f,  1.0f};
    c[ImGuiCol_ScrollbarGrab]        = {0.20f,  0.20f,  0.20f,  1.0f};
    c[ImGuiCol_ScrollbarGrabHovered] = {0.30f,  0.30f,  0.30f,  1.0f};
    c[ImGuiCol_ScrollbarGrabActive]  = {0.40f,  0.40f,  0.40f,  1.0f};
    c[ImGuiCol_Separator]            = {0.15f,  0.15f,  0.15f,  1.0f};
    c[ImGuiCol_Text]                 = UITheme::BODY_TEXT;
    c[ImGuiCol_PlotHistogram]        = UITheme::ACCENT;
    c[ImGuiCol_TableHeaderBg]        = {0.08f,  0.08f,  0.08f,  1.0f};
    c[ImGuiCol_TableBorderStrong]    = {0.15f,  0.15f,  0.15f,  1.0f};
    c[ImGuiCol_TableBorderLight]     = {0.10f,  0.10f,  0.10f,  1.0f};
    c[ImGuiCol_TableRowBgAlt]        = {0.04f,  0.04f,  0.04f,  1.0f};

    // ImPlot
    ImPlotStyle& ps = ImPlot::GetStyle();
    ps = ImPlotStyle();
    ps.PlotPadding   = ImVec2(10 * ui_scale_, 10 * ui_scale_);
    ps.LabelPadding  = ImVec2(5 * ui_scale_,  5 * ui_scale_);
    ps.LegendPadding = ImVec2(10 * ui_scale_, 10 * ui_scale_);
    ps.PlotMinSize   = ImVec2(200 * ui_scale_, 150 * ui_scale_);
    ImPlot::StyleColorsDark();

    scale_changed_ = false;
}

// ============================================================================
// Main loop
// ============================================================================
void Gui::run() {
    init_window();
    running_ = true;
    while (running_ && !glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        drain_pending_results();
        render_frame();
    }
}

// ============================================================================
// Drain pending results from benchmark thread
// ============================================================================
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

        // Feed individual times into the per-kernel ring buffer
        if (pr.result.success && !pr.result.all_times_ms.empty()) {
            auto& ring = timing_history_[pr.kernel_name];
            for (float t : pr.result.all_times_ms) {
                ring.push(t);
            }
        }

        if (pr.result.success) {
            int problem_size;
            if (pr.category == "matmul")
                problem_size = pr.params.count("M") ? pr.params.at("M") : 0;
            else if (pr.category == "softmax")
                problem_size = pr.params.count("rows") ? pr.params.at("rows") : 0;
            else
                problem_size = pr.params.count("n") ? pr.params.at("n") : 0;

            auto& hist = scaling_history_[pr.category][pr.kernel_name];
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

// ============================================================================
// Helpers
// ============================================================================
std::vector<KernelState>* Gui::current_kernels() {
    auto it = kernels_by_category_.find(current_category_);
    return (it != kernels_by_category_.end()) ? &it->second : nullptr;
}

const KernelState* Gui::selected_kernel() const {
    if (ui_state_.selected_kernel_name.empty()) return nullptr;
    for (const auto& [cat, states] : kernels_by_category_) {
        for (const auto& ks : states) {
            if (ks.descriptor && ks.descriptor->name() == ui_state_.selected_kernel_name) {
                return &ks;
            }
        }
    }
    return nullptr;
}

DSLType Gui::detect_dsl_type(const arena::KernelDescriptor* desc) const {
    if (!desc->uses_module()) return DSLType::CUB;
    if (!desc->needs_compilation()) return DSLType::CUDA;
    std::string src = desc->source_path();
    if (src.find(".triton.") != std::string::npos) return DSLType::Triton;
    if (src.find(".cutile.") != std::string::npos) return DSLType::CuTile;
    if (src.find(".warp.")   != std::string::npos) return DSLType::Warp;
    return DSLType::CUDA;
}

void Gui::format_time(float ms, char* buf, size_t buf_size) {
    if (ms <= 0.0f)     { snprintf(buf, buf_size, "--"); return; }
    if (ms < 0.001f)      snprintf(buf, buf_size, "%.0f ns", ms * 1e6f);
    else if (ms < 1.0f)   snprintf(buf, buf_size, "%.1f us", ms * 1000.0f);
    else if (ms < 1000.0f) snprintf(buf, buf_size, "%.2f ms", ms);
    else                   snprintf(buf, buf_size, "%.2f s",  ms / 1000.0f);
}

// ============================================================================
// DSL Badge  small colored rounded rect with DSL name
// ============================================================================
void Gui::render_dsl_badge(DSLType type) {
    const char* text;
    ImVec4 color;
    switch (type) {
        case DSLType::CUDA:   text = "CUDA";   color = UITheme::CUDA_BADGE;   break;
        case DSLType::Triton: text = "Triton";  color = UITheme::TRITON_BADGE; break;
        case DSLType::CuTile: text = "cuTile";  color = UITheme::CUTILE_BADGE; break;
        case DSLType::Warp:   text = "Warp";    color = UITheme::WARP_BADGE;   break;
        case DSLType::CUB:    text = "CUB";     color = UITheme::CUB_BADGE;    break;
    }

    ImVec2 tsz = ImGui::CalcTextSize(text);
    float px = 5.0f * ui_scale_;
    float py = 2.0f * ui_scale_;
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float w = tsz.x + px * 2;
    float h = tsz.y + py * 2;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, {pos.x + w, pos.y + h},
        ImGui::ColorConvertFloat4ToU32({color.x, color.y, color.z, 0.2f}),
        3.0f * ui_scale_);
    dl->AddText({pos.x + px, pos.y + py},
        ImGui::ColorConvertFloat4ToU32(color), text);

    ImGui::Dummy({w, h});
}

// ============================================================================
// KPI Card  framed child region with metric display
// ============================================================================
void Gui::render_kpi_card(int id, const char* label, const char* value_str,
                          const char* unit, float pct_of_peak, bool available,
                          const char* tooltip) {
    float s = ui_scale_;
    float w = Layout::KPI_CARD_WIDTH * s;
    float h = Layout::KPI_CARD_HEIGHT * s;

    ImGui::PushID(id);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f * s);
    ImGui::PushStyleColor(ImGuiCol_ChildBg,
        available ? ImVec4(0.07f, 0.07f, 0.07f, 1.0f) : ImVec4(0.04f, 0.04f, 0.04f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border,
        available ? ImVec4(0.16f, 0.16f, 0.16f, 1.0f) : ImVec4(0.10f, 0.10f, 0.10f, 1.0f));

    ImGui::BeginChild("kpi", {w, h}, true, ImGuiWindowFlags_NoScrollbar);

    // Label
    ImGui::TextColored(available ? UITheme::BODY_TEXT : UITheme::TEXT_DIM, "%s", label);

    // Value (large)
    if (available) {
        ImFont* font = ImGui::GetFont();
        float saved = font->Scale;
        font->Scale *= 1.4f;
        ImGui::PushFont(font);
        ImGui::TextColored(UITheme::ACCENT, "%s", value_str);
        font->Scale = saved;
        ImGui::PopFont();
        ImGui::SameLine(0, 2 * s);
        ImGui::TextColored(UITheme::TEXT_DIM, "%s", unit);
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "--");
        ImGui::TextColored({0.3f, 0.3f, 0.3f, 0.6f}, "no data");
    }

    // Peak % bar at bottom
    if (pct_of_peak >= 0.0f && available) {
        ImVec2 ws = ImGui::GetWindowSize();
        float pad = ImGui::GetStyle().WindowPadding.x;
        ImGui::SetCursorPosY(ws.y - 12 * s);
        ImVec2 bar_pos = ImGui::GetCursorScreenPos();
        float bar_w = ws.x - pad * 2;
        float bar_h = 4 * s;
        ImDrawList* dl = ImGui::GetWindowDrawList();

        dl->AddRectFilled(bar_pos, {bar_pos.x + bar_w, bar_pos.y + bar_h},
            IM_COL32(40, 40, 40, 255), 2.0f);

        float clamped = std::min(std::max(pct_of_peak, 0.0f), 1.0f);
        ImVec4 bar_color = (clamped < 0.33f) ? UITheme::ERROR_RED :
                           (clamped < 0.66f) ? UITheme::WARN_YELLOW : UITheme::SUCCESS_GREEN;
        dl->AddRectFilled(bar_pos, {bar_pos.x + bar_w * clamped, bar_pos.y + bar_h},
            ImGui::ColorConvertFloat4ToU32(bar_color), 2.0f);
    }

    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
    ImGui::PopID();

    if (tooltip && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", tooltip);
    }
}

// ============================================================================
// render_frame  main layout orchestrator
// ============================================================================
void Gui::render_frame() {
    if (scale_changed_) apply_scale();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    float s = ui_scale_;

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("##Main", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    float header_h = Layout::HEADER_HEIGHT * s;
    float log_h    = ui_state_.log_collapsed ? Layout::LOG_COLLAPSED_HEIGHT * s
                                             : Layout::LOG_HEIGHT * s;
    float left_w   = Layout::SIDEBAR_LEFT_WIDTH * s;
    float right_w  = Layout::SIDEBAR_RIGHT_WIDTH * s;

    // ---- Header Bar ----
    ImGui::BeginChild("##Header", {0, header_h}, true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    render_header_bar();
    ImGui::EndChild();

    // ---- Middle 3-column area ----
    float middle_h = ImGui::GetContentRegionAvail().y - log_h;

    ImGui::BeginChild("##LeftSidebar", {left_w, middle_h}, true);
    render_kernel_sidebar();
    ImGui::EndChild();

    ImGui::SameLine();

    float center_w = ImGui::GetContentRegionAvail().x - right_w;
    ImGui::BeginChild("##Center", {center_w, middle_h}, true,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);
    render_benchmark_panel();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("##RightSidebar", {0, middle_h}, true,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);
    render_profile_sidebar();
    ImGui::EndChild();

    // ---- Log Panel ----
    ImGui::BeginChild("##LogPanel", {0, 0}, true);
    render_log_panel();
    ImGui::EndChild();

    ImGui::End();

    // GL render
    ImGui::Render();
    int dw, dh;
    glfwGetFramebufferSize(window_, &dw, &dh);
    glViewport(0, 0, dw, dh);
    glClearColor(0.067f, 0.067f, 0.067f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

// ============================================================================
// Header Bar
// ============================================================================
void Gui::render_header_bar() {
    float s = ui_scale_;
    const auto& ctx = runner_.context();

    // Branding
    ImFont* font = ImGui::GetFont();
    float saved = font->Scale;
    font->Scale *= 1.2f;
    ImGui::PushFont(font);
    ImGui::TextColored(UITheme::ACCENT, "GPGPU Arena");
    font->Scale = saved;
    ImGui::PopFont();

    // GPU info
    ImGui::SameLine(180 * s);
    size_t mem_mb = ctx.total_memory() / (1024 * 1024);
    char vram_buf[32];
    if (mem_mb >= 1024)
        snprintf(vram_buf, sizeof(vram_buf), "%.1f GB", mem_mb / 1024.0f);
    else
        snprintf(vram_buf, sizeof(vram_buf), "%zu MB", mem_mb);
    ImGui::TextColored(UITheme::TEXT_DIM, "%s | sm_%d%d | %d SMs | %s",
        ctx.device_name().c_str(),
        ctx.compute_capability_major(), ctx.compute_capability_minor(),
        ctx.sm_count(), vram_buf);

    // Right-aligned items
    float right_edge = ImGui::GetContentRegionMax().x;

    // Status indicator
    ImGui::SameLine(right_edge - 340 * s);
    if (benchmark_running_) {
        if (config_.collect_metrics)
            ImGui::TextColored(UITheme::LOG_PROFILE, "PROFILING");
        else
            ImGui::TextColored(UITheme::ACCENT, "RUNNING");

        ImGui::SameLine();
        int cur = benchmark_current_.load();
        int tot = benchmark_total_.load();
        float frac = tot > 0 ? (float)cur / (float)tot : 0.0f;
        ImGui::ProgressBar(frac, {100 * s, 0});
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "IDLE");
    }

    // Kernel count
    ImGui::SameLine(right_edge - 170 * s);
    int total_k = 0;
    for (const auto& [c, ks] : kernels_by_category_) total_k += (int)ks.size();
    ImGui::Text("%d kernels", total_k);

    // Run All / Stop buttons
    ImGui::SameLine(right_edge - 85 * s);
    if (benchmark_running_) {
        ImGui::PushStyleColor(ImGuiCol_Button, {0.5f, 0.1f, 0.1f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.65f, 0.15f, 0.15f, 1.0f});
        if (ImGui::Button("Stop", {80 * s, 0})) {
            cancel_requested_ = true;
        }
        ImGui::PopStyleColor(2);
    } else {
        if (ImGui::Button("Run All", {80 * s, 0})) {
            run_selected_kernels();
        }
    }
}

// ============================================================================
// Left Sidebar  Kernel List + Config + Controls
// ============================================================================
void Gui::render_kernel_sidebar() {
    float s = ui_scale_;

    // ---- Category tabs ----
    for (size_t ci = 0; ci < categories_.size(); ci++) {
        const auto& cat = categories_[ci];
        bool active = (cat == current_category_);

        if (active) ImGui::PushStyleColor(ImGuiCol_Button, UITheme::ACCENT_DIM);

        std::string label = cat;
        if (!label.empty()) label[0] = (char)std::toupper(label[0]);

        if (ImGui::SmallButton(label.c_str())) {
            select_category(cat);
        }
        if (active) ImGui::PopStyleColor();

        if (ci < categories_.size() - 1) ImGui::SameLine();
    }

    ImGui::Separator();
    ImGui::Spacing();

    auto* kernels = current_kernels();
    if (!kernels) {
        ImGui::TextColored(UITheme::TEXT_DIM, "No category selected");
        return;
    }

    // ---- Select All / None ----
    if (ImGui::SmallButton("All")) { for (auto& k : *kernels) k.selected = true; }
    ImGui::SameLine();
    if (ImGui::SmallButton("None")) { for (auto& k : *kernels) k.selected = false; }
    ImGui::Separator();

    // Read running kernel name once (avoid repeated locking)
    std::string running_name;
    if (benchmark_running_) {
        std::lock_guard<std::mutex> lock(mutex_);
        running_name = benchmark_current_name_;
    }

    // ---- Kernel list (scrollable middle) ----
    float bottom_reserve = 260 * s;
    float list_h = ImGui::GetContentRegionAvail().y - bottom_reserve;
    if (list_h < 80 * s) list_h = 80 * s;
    ImGui::BeginChild("##KernelList", {0, list_h}, false);

    for (size_t i = 0; i < kernels->size(); i++) {
        auto& k = (*kernels)[i];
        bool is_display_sel = (k.descriptor->name() == ui_state_.selected_kernel_name);
        bool is_running = benchmark_running_ && (k.descriptor->name() == running_name);

        ImGui::PushID((int)i);

        // Checkbox for benchmark selection
        ImGui::Checkbox("##chk", &k.selected);
        ImGui::SameLine();

        // Compact DSL badge (colored 2-letter code)
        DSLType dsl = detect_dsl_type(k.descriptor);
        ImVec4 badge_color;
        const char* badge_text;
        switch (dsl) {
            case DSLType::CUDA:   badge_color = UITheme::CUDA_BADGE;   badge_text = "CU"; break;
            case DSLType::Triton: badge_color = UITheme::TRITON_BADGE; badge_text = "TR"; break;
            case DSLType::CuTile: badge_color = UITheme::CUTILE_BADGE; badge_text = "CT"; break;
            case DSLType::Warp:   badge_color = UITheme::WARP_BADGE;   badge_text = "WP"; break;
            case DSLType::CUB:    badge_color = UITheme::CUB_BADGE;    badge_text = "CB"; break;
        }
        ImVec2 bpos = ImGui::GetCursorScreenPos();
        ImVec2 tsz = ImGui::CalcTextSize(badge_text);
        float bpad = 2 * s;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(bpos, {bpos.x + tsz.x + bpad * 2, bpos.y + tsz.y + bpad},
            ImGui::ColorConvertFloat4ToU32({badge_color.x, badge_color.y, badge_color.z, 0.2f}),
            3.0f);
        dl->AddText({bpos.x + bpad, bpos.y + bpad * 0.5f},
            ImGui::ColorConvertFloat4ToU32(badge_color), badge_text);
        ImGui::Dummy({tsz.x + bpad * 2 + 2 * s, tsz.y + bpad});
        ImGui::SameLine();

        // Selectable kernel name
        float name_w = ImGui::GetContentRegionAvail().x - 60 * s;
        if (name_w < 30 * s) name_w = 30 * s;
        if (ImGui::Selectable(k.descriptor->name().c_str(), is_display_sel,
                ImGuiSelectableFlags_None, {name_w, 0})) {
            if (is_display_sel) {
                ui_state_.selected_kernel_name.clear();
                ui_state_.selected_category.clear();
            } else {
                ui_state_.selected_kernel_name = k.descriptor->name();
                ui_state_.selected_category = k.descriptor->category();
                current_category_ = k.descriptor->category();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(300.0f * s);
            ImGui::TextUnformatted(k.descriptor->description().c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }

        // Status + time (right-aligned)
        ImGui::SameLine(ImGui::GetContentRegionMax().x - 55 * s);
        if (is_running) {
            ImGui::TextColored(UITheme::ACCENT, "...");
        } else if (!k.has_run) {
            ImGui::TextColored(UITheme::TEXT_DIM, "--");
        } else if (k.result.success) {
            char tbuf[32];
            format_time(k.result.elapsed_ms, tbuf, sizeof(tbuf));
            ImGui::TextColored(k.result.verified ? UITheme::SUCCESS_GREEN : UITheme::WARN_YELLOW,
                "%s", tbuf);
        } else {
            ImGui::TextColored(UITheme::ERROR_RED, "ERR");
        }

        ImGui::PopID();
    }

    ImGui::EndChild();

    // ---- Bottom sections ----
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Problem Size", ImGuiTreeNodeFlags_DefaultOpen)) {
        render_problem_config();
    }

    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        render_run_controls();
    }
}

// ============================================================================
// Problem Config (kept from original, per-category sliders)
// ============================================================================
void Gui::render_problem_config() {
    if (current_category_ == "matmul") {
        ImGui::Checkbox("Lock Square", &lock_square_);
        int m = config_.params["M"];
        int k = config_.params["K"];
        int n = config_.params["N"];

        if (lock_square_) {
            if (ImGui::SliderInt("Size", &m, 256, 4096)) {
                config_.params["M"] = m; config_.params["K"] = m; config_.params["N"] = m;
            }
            ImGui::Text("(%d x %d) x (%d x %d)", m, m, m, m);
        } else {
            bool changed = false;
            changed |= ImGui::SliderInt("M", &m, 256, 4096);
            changed |= ImGui::SliderInt("K", &k, 256, 4096);
            changed |= ImGui::SliderInt("N", &n, 256, 4096);
            if (changed) {
                config_.params["M"] = m; config_.params["K"] = k; config_.params["N"] = n;
            }
            ImGui::Text("(%d x %d) x (%d x %d)", m, k, k, n);
        }
    } else if (current_category_ == "softmax") {
        int rows = config_.params["rows"];
        int cols = config_.params["cols"];
        bool changed = false;
        changed |= ImGui::SliderInt("Rows", &rows, 64, 8192);
        changed |= ImGui::SliderInt("Cols", &cols, 64, 8192);
        if (changed) { config_.params["rows"] = rows; config_.params["cols"] = cols; }
        float mb = (2.0f * rows * cols * sizeof(float)) / (1024.0f * 1024.0f);
        ImGui::Text("%d x %d (%.1f MB)", rows, cols, mb);
    } else if (current_category_ == "reduce" || current_category_ == "scan") {
        int n = config_.params["n"];
        if (ImGui::SliderInt("Elements", &n, 100000, 100000000, "%d",
                ImGuiSliderFlags_Logarithmic)) {
            config_.params["n"] = n;
        }
        ImGui::Text("%d elements (%.1f MB)", n, (n * sizeof(float)) / (1024.0f * 1024.0f));
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Select a category");
    }
}

// ============================================================================
// Run Controls (settings + buttons + progress)
// ============================================================================
void Gui::render_run_controls() {
    float s = ui_scale_;

    ImGui::SliderInt("Warmup", &config_.warmup_runs, 0, 50);
    ImGui::SliderInt("Runs", &config_.number_of_runs, 1, 100);
    ImGui::Checkbox("Profile", &config_.collect_metrics);
    if (config_.collect_metrics) {
        ImGui::SameLine();
        ImGui::TextColored(UITheme::WARN_YELLOW, "(slower)");
    }

    ImGui::Spacing();
    ImGui::Text("Scale:");
    ImGui::SameLine();
    if (ImGui::RadioButton("1x", ui_scale_ == 1.0f)) { ui_scale_ = 1.0f; scale_changed_ = true; }
    ImGui::SameLine();
    if (ImGui::RadioButton("2x", ui_scale_ == 2.0f)) { ui_scale_ = 2.0f; scale_changed_ = true; }
    ImGui::SameLine();
    if (ImGui::RadioButton("4x", ui_scale_ == 4.0f)) { ui_scale_ = 4.0f; scale_changed_ = true; }

    ImGui::Spacing();

    if (benchmark_running_) {
        int cur = benchmark_current_.load();
        int tot = benchmark_total_.load();
        float frac = tot > 0 ? (float)cur / (float)tot : 0.0f;
        std::string overlay;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            overlay = benchmark_current_name_ + " (" +
                      std::to_string(cur + 1) + "/" + std::to_string(tot) + ")";
        }
        ImGui::ProgressBar(frac, {-1, 0}, overlay.c_str());

        if (ImGui::Button("Cancel", {-1, 28 * s})) {
            cancel_requested_ = true;
        }
    } else {
        int selected_count = 0;
        if (auto* ks = current_kernels()) {
            for (const auto& k : *ks) if (k.selected) selected_count++;
        }

        ImGui::BeginDisabled(selected_count == 0);
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "Run Selected (%d)", selected_count);
        if (ImGui::Button(lbl, {-1, 30 * s})) {
            run_selected_kernels();
        }

        snprintf(lbl, sizeof(lbl), "Run Sweep (%d)", selected_count);
        if (ImGui::Button(lbl, {-1, 30 * s})) {
            run_sweep();
        }
        if (ImGui::IsItemHovered()) {
            if (auto* ks = current_kernels()) {
                for (const auto& k : *ks) {
                    if (k.selected && k.descriptor) {
                        auto configs = k.descriptor->get_sweep_configs();
                        if (!configs.empty()) {
                            ImGui::BeginTooltip();
                            ImGui::Text("Run at %zu sizes:", configs.size());
                            for (const auto& cfg : configs) {
                                std::string line;
                                for (const auto& [key, val] : cfg) {
                                    if (!line.empty()) line += ", ";
                                    line += key + "=" + std::to_string(val);
                                }
                                ImGui::BulletText("%s", line.c_str());
                            }
                            ImGui::EndTooltip();
                        }
                        break;
                    }
                }
            }
        }
        ImGui::EndDisabled();

        bool has_results = false;
        if (auto* ks = current_kernels()) {
            for (const auto& k : *ks) if (k.has_run) { has_results = true; break; }
        }

        ImGui::BeginDisabled(!has_results || benchmark_running_);
        if (ImGui::Button("Reset Results", {-1, 24 * s})) {
            reset_results();
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(benchmark_running_);
        if (ImGui::Button("Clear Cache", {-1, 24 * s})) {
            runner_.compiler().clear_cache();
            log(LogEntry::INFO, "Kernel cache cleared");
        }
        ImGui::EndDisabled();
    }
}

// ============================================================================
// Center  Benchmark Results Panel
// ============================================================================
void Gui::render_benchmark_panel() {
    float s = ui_scale_;
    const auto* sel = selected_kernel();
    auto* kernels = current_kernels();

    if (!kernels || kernels->empty()) {
        ImGui::TextColored(UITheme::TEXT_DIM, "Select a category to get started");
        return;
    }

    // ================================================================
    // Results Table (sortable overview of all kernels)
    // ================================================================
    if (ImGui::CollapsingHeader("Results Table", ImGuiTreeNodeFlags_DefaultOpen)) {
        render_results_table();
    }

    ImGui::Spacing();

    // ================================================================
    // KPI Cards Row (selected kernel)
    // ================================================================
    {
        bool has_data    = sel && sel->has_run && sel->result.success;
        bool has_profile = has_data && sel->result.achieved_occupancy > 0;

        float card_w  = Layout::KPI_CARD_WIDTH * s;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float avail_w = ImGui::GetContentRegionAvail().x;
        int cols = std::max(1, (int)(avail_w / (card_w + spacing)));

        char vbuf[64];
        const char* no_profiler = "Enable GPU perf counters - see README profiling section";

        // Card 0: Median Time
        if (has_data) format_time(sel->result.elapsed_ms, vbuf, sizeof(vbuf));
        else          snprintf(vbuf, sizeof(vbuf), "--");
        render_kpi_card(0, "Median Time", has_data ? vbuf : "--", "", -1.0f, has_data);

        // Card 1: Memory BW
        if (1 % cols != 0) ImGui::SameLine();
        char bw_buf[64]; snprintf(bw_buf, sizeof(bw_buf), "%.1f", has_data ? sel->result.bandwidth_gbps : 0.0);
        render_kpi_card(1, "Memory BW", has_data ? bw_buf : "--", "GB/s", -1.0f, has_data,
            has_data ? nullptr : "// TODO: expose bytes_transferred from kernel descriptor");

        // Card 2: Compute
        if (2 % cols != 0) ImGui::SameLine();
        char flops_buf[64]; const char* flops_unit = "GFLOPS";
        if (has_data) {
            if (sel->result.gflops >= 1000.0) {
                snprintf(flops_buf, sizeof(flops_buf), "%.2f", sel->result.gflops / 1000.0);
                flops_unit = "TFLOPS";
            } else {
                snprintf(flops_buf, sizeof(flops_buf), "%.1f", sel->result.gflops);
            }
        }
        render_kpi_card(2, "Compute", has_data ? flops_buf : "--", flops_unit, -1.0f, has_data,
            has_data ? nullptr : "// TODO: expose flop_count from kernel descriptor");

        // Card 3: Occupancy
        if (3 % cols != 0) ImGui::SameLine();
        char occ_buf[64];
        if (has_profile) snprintf(occ_buf, sizeof(occ_buf), "%.1f", sel->result.achieved_occupancy * 100.0);
        render_kpi_card(3, "Occupancy", has_profile ? occ_buf : "--", "%",
            has_profile ? (float)sel->result.achieved_occupancy : -1.0f,
            has_profile, has_profile ? nullptr : no_profiler);

        // Card 4: IPC
        if (4 % cols != 0) ImGui::SameLine();
        char ipc_buf[64];
        bool has_ipc = has_profile && sel->result.ipc > 0;
        if (has_ipc) snprintf(ipc_buf, sizeof(ipc_buf), "%.2f", sel->result.ipc);
        render_kpi_card(4, "IPC", has_ipc ? ipc_buf : "--", "",
            -1.0f, has_ipc, has_ipc ? nullptr : no_profiler);

        // Card 5: DRAM BW
        if (5 % cols != 0) ImGui::SameLine();
        char dram_buf[64];
        double total_dram = has_profile ? sel->result.dram_read_gbps + sel->result.dram_write_gbps : 0.0;
        bool has_dram = total_dram > 0;
        if (has_dram) snprintf(dram_buf, sizeof(dram_buf), "%.1f", total_dram);
        render_kpi_card(5, "DRAM BW", has_dram ? dram_buf : "--", "GB/s",
            -1.0f, has_dram, has_dram ? nullptr : no_profiler);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ================================================================
    // Timing Distribution Graph (selected kernel)
    // ================================================================
    if (sel && sel->has_run && sel->result.success && !sel->result.all_times_ms.empty()) {
        const auto& times = sel->result.all_times_ms;
        int n = (int)times.size();

        char td_header[128];
        snprintf(td_header, sizeof(td_header), "%s -- Timing Distribution (%d runs)###TimingDist",
            sel->result.kernel_name.c_str(), n);
        if (ImGui::CollapsingHeader(td_header)) {
        // Build plot data in microseconds
        std::vector<double> xs(n), ys(n);
        double min_t = 1e9, max_t = 0;
        for (int i = 0; i < n; i++) {
            xs[i] = (double)(i + 1);
            ys[i] = (double)times[i] * 1000.0;  // ms -> us
            if (ys[i] < min_t) min_t = ys[i];
            if (ys[i] > max_t) max_t = ys[i];
        }
        double median_us = (double)sel->result.elapsed_ms * 1000.0;

        float plot_h = 200 * s;
        if (ImPlot::BeginPlot("##TimingDist", {-1, plot_h})) {
            ImPlot::SetupAxes("Run Index", "Time (us)",
                ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

            // Min/Max shaded band
            std::vector<double> min_band(n, min_t), max_band(n, max_t);
            ImPlot::SetNextFillStyle({1, 1, 1, 0.06f});
            ImPlot::PlotShaded("Min/Max", xs.data(), min_band.data(), max_band.data(), n);

            // Individual runs as scatter
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4 * s,
                UITheme::ACCENT, 1.0f);
            ImPlot::PlotScatter("Runs", xs.data(), ys.data(), n);

            // Median line (dashed via bright color)
            double med_xs[2] = {0.5, (double)n + 0.5};
            double med_ys[2] = {median_us, median_us};
            ImPlot::SetNextLineStyle({1.0f, 0.9f, 0.0f, 0.8f}, 2.0f);
            ImPlot::PlotLine("Median", med_xs, med_ys, 2);

            ImPlot::EndPlot();
        }
        } // end CollapsingHeader
    }

    // ================================================================
    // Multi-Kernel Comparison Bar Chart
    // ================================================================
    {
        struct KernelBar {
            std::string name;
            double median_ms;
            DSLType dsl;
        };
        std::vector<KernelBar> bars;

        for (const auto& k : *kernels) {
            if (k.has_run && k.result.success) {
                bars.push_back({k.result.kernel_name, (double)k.result.elapsed_ms,
                                detect_dsl_type(k.descriptor)});
            }
        }

        if (bars.size() >= 2) {
            ImGui::TextColored(UITheme::HEADER_TEXT,
                "Side-by-Side Median Time Comparison");

            // Sort ascending by median
            std::sort(bars.begin(), bars.end(),
                [](const KernelBar& a, const KernelBar& b) {
                    return a.median_ms < b.median_ms;
                });

            double slowest = bars.back().median_ms;

            // Group by DSL type for colored bars
            struct DSLGroup {
                std::vector<double> positions;
                std::vector<double> values;
            };
            std::map<DSLType, DSLGroup> groups;

            std::vector<std::string> label_strings(bars.size());
            std::vector<const char*> tick_labels(bars.size());
            std::vector<double> tick_positions(bars.size());

            for (size_t i = 0; i < bars.size(); i++) {
                tick_positions[i] = (double)i;
                label_strings[i] = bars[i].name;
                tick_labels[i] = label_strings[i].c_str();
                groups[bars[i].dsl].positions.push_back((double)i);
                groups[bars[i].dsl].values.push_back(bars[i].median_ms);
            }

            float plot_h = 250 * s;
            if (ImPlot::BeginPlot("##Comparison", {-1, plot_h})) {
                ImPlot::SetupAxes("", "Median Time (ms)",
                    ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisTicks(ImAxis_X1, tick_positions.data(),
                    (int)tick_positions.size(), tick_labels.data());

                auto plot_dsl = [&](DSLType type, const char* name, ImVec4 color) {
                    auto it = groups.find(type);
                    if (it == groups.end()) return;
                    ImPlot::SetNextFillStyle(color);
                    ImPlot::PlotBars(name, it->second.positions.data(),
                        it->second.values.data(), (int)it->second.positions.size(), 0.6);
                };

                plot_dsl(DSLType::CUDA,   "CUDA",   UITheme::CUDA_BADGE);
                plot_dsl(DSLType::Triton, "Triton", UITheme::TRITON_BADGE);
                plot_dsl(DSLType::CuTile, "cuTile", UITheme::CUTILE_BADGE);
                plot_dsl(DSLType::Warp,   "Warp",   UITheme::WARP_BADGE);
                plot_dsl(DSLType::CUB,    "CUB",    UITheme::CUB_BADGE);

                // Speedup labels above bars
                for (size_t i = 0; i < bars.size(); i++) {
                    double speedup = slowest / bars[i].median_ms;
                    if (speedup > 1.01) {
                        char txt[32];
                        snprintf(txt, sizeof(txt), "%.1fx", speedup);
                        ImPlot::PlotText(txt, (double)i, bars[i].median_ms, {0, -10});
                    }
                }

                ImPlot::EndPlot();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
    }

    // ================================================================
    // Wall Time vs GPU Time Comparison
    // ================================================================
    {
        struct TimeEntry {
            std::string name;
            double wall_ms;
            double gpu_ms;
        };
        std::vector<TimeEntry> entries;
        for (const auto& k : *kernels) {
            if (k.has_run && k.result.success) {
                entries.push_back({k.result.kernel_name,
                    (double)k.result.elapsed_ms, (double)k.result.kernel_ms});
            }
        }

        if (!entries.empty()) {
            if (ImGui::CollapsingHeader("Wall vs GPU Time", ImGuiTreeNodeFlags_DefaultOpen)) {
                static int time_sort = 0;
                ImGui::SetNextItemWidth(150 * s);
                const char* sort_opts[] = {"Sort: Wall Time", "Sort: GPU Time", "Sort: Overhead"};
                ImGui::Combo("##timesort", &time_sort, sort_opts, 3);

                std::sort(entries.begin(), entries.end(),
                    [&](const TimeEntry& a, const TimeEntry& b) {
                        switch (time_sort) {
                            case 1:  return a.gpu_ms < b.gpu_ms;
                            case 2:  return (a.wall_ms - a.gpu_ms) > (b.wall_ms - b.gpu_ms);
                            default: return a.wall_ms < b.wall_ms;
                        }
                    });

                int n = (int)entries.size();
                std::vector<std::string> label_store(n);
                std::vector<const char*> labels(n);
                std::vector<double> positions(n), wall_vals(n), gpu_vals(n);
                for (int i = 0; i < n; i++) {
                    positions[i] = (double)i;
                    label_store[i] = entries[i].name;
                    labels[i] = label_store[i].c_str();
                    wall_vals[i] = entries[i].wall_ms;
                    gpu_vals[i] = entries[i].gpu_ms;
                }

                double bw = 0.3;
                std::vector<double> pos_wall(n), pos_gpu(n);
                for (int i = 0; i < n; i++) {
                    pos_wall[i] = positions[i] - bw * 0.55;
                    pos_gpu[i]  = positions[i] + bw * 0.55;
                }

                float plot_h = 220 * s;
                if (ImPlot::BeginPlot("##WallGPU", {-1, plot_h})) {
                    ImPlot::SetupAxes("", "Time (ms)",
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, labels.data());

                    ImPlot::SetNextFillStyle(UITheme::ACCENT);
                    ImPlot::PlotBars("Wall Time", pos_wall.data(), wall_vals.data(), n, bw);

                    ImPlot::SetNextFillStyle({0.35f, 0.60f, 0.85f, 1.0f});
                    ImPlot::PlotBars("GPU Time", pos_gpu.data(), gpu_vals.data(), n, bw);

                    // Overhead % annotation above wall bar
                    for (int i = 0; i < n; i++) {
                        double overhead = wall_vals[i] > 0
                            ? ((wall_vals[i] - gpu_vals[i]) / wall_vals[i]) * 100.0 : 0;
                        if (overhead > 1.0) {
                            char txt[32];
                            snprintf(txt, sizeof(txt), "+%.0f%%", overhead);
                            ImPlot::PlotText(txt, positions[i], wall_vals[i], {0, -10});
                        }
                    }

                    ImPlot::EndPlot();
                }
            }

            ImGui::Spacing();
        }
    }

    // ================================================================
    // Performance Chart (GFLOPS / GB/s bars)
    // ================================================================
    {
        std::vector<std::string> label_strings;
        std::vector<double> values;

        for (const auto& k : *kernels) {
            if (k.has_run && k.result.success) {
                label_strings.push_back(k.result.kernel_name);
                values.push_back(is_matmul() ? k.result.gflops : k.result.bandwidth_gbps);
            }
        }

        if (!values.empty()) {
            const char* y_label = is_matmul() ? "GFLOPS" : "GB/s";
            ImGui::TextColored(UITheme::HEADER_TEXT, "Throughput (%s)", y_label);

            // Sort descending
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

            float plot_h = 200 * s;
            if (ImPlot::BeginPlot("##Performance", {-1, plot_h})) {
                ImPlot::SetupAxes("", y_label,
                    ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisTicks(ImAxis_X1, 0,
                    (double)(sorted_labels.size() - 1),
                    (int)sorted_labels.size(), sorted_labels.data());

                std::vector<double> positions(sorted_values.size());
                for (size_t i = 0; i < positions.size(); i++) positions[i] = (double)i;

                ImPlot::PlotBars("Performance", positions.data(),
                    sorted_values.data(), (int)sorted_values.size(), 0.6);
                ImPlot::EndPlot();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
    }

    // ================================================================
    // Profiling Comparison (all kernels side-by-side)
    // ================================================================
    {
        std::vector<const char*> prof_labels;
        std::vector<double> occupancy, ipc_vals;

        for (const auto& k : *kernels) {
            if (k.has_run && k.result.success && k.result.achieved_occupancy > 0) {
                prof_labels.push_back(k.result.kernel_name.c_str());
                occupancy.push_back(k.result.achieved_occupancy * 100.0);
                ipc_vals.push_back(k.result.ipc);
            }
        }

        if (!prof_labels.empty()) {
            if (ImGui::CollapsingHeader("Profiling Comparison")) {
                int pn = (int)prof_labels.size();
                std::vector<double> positions(pn);
                for (int i = 0; i < pn; i++) positions[i] = (double)i;

                float plot_h = 200 * s;
                if (ImPlot::BeginPlot("##ProfComp", {-1, plot_h})) {
                    ImPlot::SetupAxes("", "Occupancy %",
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxis(ImAxis_Y2, "IPC",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_AuxDefault);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImPlotCond_Always);
                    ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), pn, prof_labels.data());

                    double bw = 0.3;
                    std::vector<double> pos_left(pn), pos_right(pn);
                    for (int i = 0; i < pn; i++) {
                        pos_left[i]  = positions[i] - bw * 0.55;
                        pos_right[i] = positions[i] + bw * 0.55;
                    }

                    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                    ImPlot::SetNextFillStyle(UITheme::ACCENT);
                    ImPlot::PlotBars("Occupancy %", pos_left.data(), occupancy.data(), pn, bw);

                    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                    ImPlot::SetNextFillStyle(UITheme::WARN_YELLOW);
                    ImPlot::PlotBars("IPC", pos_right.data(), ipc_vals.data(), pn, bw);

                    ImPlot::EndPlot();
                }
            }

            ImGui::Spacing();
        }
    }

    // ================================================================
    // Scaling Chart (multi-size history)
    // ================================================================
    {
        auto cat_it = scaling_history_.find(current_category_);
        if (cat_it != scaling_history_.end() && !cat_it->second.empty()) {
            bool has_multi = false;
            for (const auto& [name, hist] : cat_it->second) {
                if (hist.size() > 1) { has_multi = true; break; }
            }

            if (has_multi) {
                ImGui::TextColored(UITheme::HEADER_TEXT, "Scaling");

                const char* metric_names[] = {"Performance", "Wall Time", "GPU Time"};
                int metric_idx = (int)scaling_metric_;
                ImGui::SetNextItemWidth(160 * s);
                if (ImGui::Combo("Metric##scaling", &metric_idx, metric_names, 3)) {
                    scaling_metric_ = (ScalingMetric)metric_idx;
                }

                const char* x_label = is_matmul() ? "Matrix Size" :
                    (current_category_ == "softmax") ? "Rows" : "Elements";
                const char* y_label;
                switch (scaling_metric_) {
                    case ScalingMetric::Performance:
                        y_label = is_matmul() ? "GFLOPS" : "GB/s"; break;
                    case ScalingMetric::WallTime:
                        y_label = "Wall Time (ms)"; break;
                    case ScalingMetric::GpuTime:
                        y_label = "GPU Time (ms)"; break;
                }

                float plot_h = 250 * s;
                if (ImPlot::BeginPlot("##Scaling", {-1, plot_h})) {
                    ImPlot::SetupAxes(x_label, y_label,
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    std::vector<double> xs, ys;
                    for (const auto& [name, hist] : cat_it->second) {
                        if (hist.size() < 2) continue;
                        xs.clear(); ys.clear();
                        for (const auto& entry : hist) {
                            xs.push_back((double)entry.problem_size);
                            switch (scaling_metric_) {
                                case ScalingMetric::Performance:
                                    ys.push_back(is_matmul() ? entry.result.gflops
                                                             : entry.result.bandwidth_gbps);
                                    break;
                                case ScalingMetric::WallTime:
                                    ys.push_back((double)entry.result.elapsed_ms);
                                    break;
                                case ScalingMetric::GpuTime:
                                    ys.push_back((double)entry.result.kernel_ms);
                                    break;
                            }
                        }
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4 * s);
                        ImPlot::PlotLine(name.c_str(), xs.data(), ys.data(), (int)xs.size());
                    }

                    ImPlot::EndPlot();
                }
            }
        }
    }
}

// ============================================================================
// Results Table  sortable overview of all kernels in current category
// ============================================================================
void Gui::render_results_table() {
    float s = ui_scale_;
    auto* kernels = current_kernels();
    if (!kernels) return;

    bool show_gflops = is_matmul();
    bool has_profiling = false;
    for (const auto& k : *kernels) {
        if (k.has_run && k.result.registers_per_thread > 0) {
            has_profiling = true; break;
        }
    }

    enum ColumnID { Col_Kernel = 0, Col_Block, Col_Grid, Col_Wall, Col_GPU,
                    Col_Perf, Col_Status, Col_Regs, Col_SHMem, Col_Occup, Col_IPC };

    int num_cols = has_profiling ? 11 : 8;

    float table_h = std::min(ImGui::GetContentRegionAvail().y, 300 * s);
    if (table_h < 100 * s) table_h = 100 * s;

    if (ImGui::BeginTable("results", num_cols,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
            ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate,
            {0, table_h})) {

        ImGui::TableSetupColumn("Kernel",
            ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_DefaultSort, 0, Col_Kernel);
        ImGui::TableSetupColumn("Block",  ImGuiTableColumnFlags_WidthFixed, 70 * s, Col_Block);
        ImGui::TableSetupColumn("Grid",   ImGuiTableColumnFlags_WidthFixed, 80 * s, Col_Grid);
        ImGui::TableSetupColumn("Wall (ms)",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 65 * s, Col_Wall);
        ImGui::TableSetupColumn("GPU (ms)",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 60 * s, Col_GPU);
        ImGui::TableSetupColumn(show_gflops ? "GFLOPS" : "GB/s",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 60 * s, Col_Perf);
        ImGui::TableSetupColumn("Status",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, 50 * s, Col_Status);

        if (has_profiling) {
            ImGui::TableSetupColumn("Regs",   ImGuiTableColumnFlags_WidthFixed, 40 * s, Col_Regs);
            ImGui::TableSetupColumn("SHMem",  ImGuiTableColumnFlags_WidthFixed, 55 * s, Col_SHMem);
            ImGui::TableSetupColumn("Occup%",
                ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 55 * s, Col_Occup);
            ImGui::TableSetupColumn("IPC",
                ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 45 * s, Col_IPC);
        } else {
            ImGui::TableSetupColumn("Regs", ImGuiTableColumnFlags_WidthFixed, 45 * s, Col_Regs);
        }

        ImGui::TableHeadersRow();

        // Sort
        std::vector<int> sorted_indices;
        for (int i = 0; i < (int)kernels->size(); i++) {
            if ((*kernels)[i].has_run) sorted_indices.push_back(i);
        }

        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty) sort_specs->SpecsDirty = false;
            if (sort_specs->SpecsCount > 0) {
                const auto& spec = sort_specs->Specs[0];
                bool asc = (spec.SortDirection == ImGuiSortDirection_Ascending);
                std::sort(sorted_indices.begin(), sorted_indices.end(),
                    [&](int a, int b) {
                        const auto& ra = (*kernels)[a].result;
                        const auto& rb = (*kernels)[b].result;
                        int cmp = 0;
                        switch (spec.ColumnUserID) {
                            case Col_Kernel: cmp = ra.kernel_name.compare(rb.kernel_name); break;
                            case Col_Block:  cmp = (int)(ra.block_x * ra.block_y) - (int)(rb.block_x * rb.block_y); break;
                            case Col_Grid:   cmp = (int)(ra.grid_x * ra.grid_y) - (int)(rb.grid_x * rb.grid_y); break;
                            case Col_Wall:   cmp = (ra.elapsed_ms < rb.elapsed_ms) ? -1 : (ra.elapsed_ms > rb.elapsed_ms) ? 1 : 0; break;
                            case Col_GPU:    cmp = (ra.kernel_ms < rb.kernel_ms) ? -1 : (ra.kernel_ms > rb.kernel_ms) ? 1 : 0; break;
                            case Col_Perf: {
                                double va = show_gflops ? ra.gflops : ra.bandwidth_gbps;
                                double vb = show_gflops ? rb.gflops : rb.bandwidth_gbps;
                                cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0; break;
                            }
                            case Col_Regs:  cmp = ra.registers_per_thread - rb.registers_per_thread; break;
                            case Col_SHMem: cmp = ra.shared_memory_bytes - rb.shared_memory_bytes; break;
                            case Col_Occup: cmp = (ra.achieved_occupancy < rb.achieved_occupancy) ? -1 : 1; break;
                            case Col_IPC:   cmp = (ra.ipc < rb.ipc) ? -1 : (ra.ipc > rb.ipc) ? 1 : 0; break;
                            default: break;
                        }
                        return asc ? (cmp < 0) : (cmp > 0);
                    });
            }
        }

        for (int idx : sorted_indices) {
            const auto& k = (*kernels)[idx];
            ImGui::TableNextRow();

            // Kernel name with DSL color tag + click-to-select
            ImGui::TableNextColumn();
            DSLType dsl = detect_dsl_type(k.descriptor);
            ImVec4 badge_col;
            const char* badge_tag;
            switch (dsl) {
                case DSLType::CUDA:   badge_col = UITheme::CUDA_BADGE;   badge_tag = "[CU]"; break;
                case DSLType::Triton: badge_col = UITheme::TRITON_BADGE; badge_tag = "[TR]"; break;
                case DSLType::CuTile: badge_col = UITheme::CUTILE_BADGE; badge_tag = "[CT]"; break;
                case DSLType::Warp:   badge_col = UITheme::WARP_BADGE;   badge_tag = "[WP]"; break;
                case DSLType::CUB:    badge_col = UITheme::CUB_BADGE;    badge_tag = "[CB]"; break;
            }
            ImGui::TextColored(badge_col, "%s", badge_tag);
            ImGui::SameLine();
            bool is_sel = (k.descriptor->name() == ui_state_.selected_kernel_name);
            if (ImGui::Selectable(k.result.kernel_name.c_str(), is_sel,
                    ImGuiSelectableFlags_SpanAllColumns)) {
                if (is_sel) {
                    ui_state_.selected_kernel_name.clear();
                    ui_state_.selected_category.clear();
                } else {
                    ui_state_.selected_kernel_name = k.descriptor->name();
                    ui_state_.selected_category = k.descriptor->category();
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("%s", k.result.description.c_str());
                if (!k.result.sub_kernels.empty()) {
                    ImGui::Separator();
                    for (const auto& sk : k.result.sub_kernels)
                        ImGui::Text("  %.3f ms | %d regs | %s",
                            sk.duration_ms, sk.registers, sk.name.c_str());
                }
                ImGui::EndTooltip();
            }

            ImGui::TableNextColumn(); ImGui::Text("%ux%u", k.result.block_x, k.result.block_y);
            ImGui::TableNextColumn(); ImGui::Text("%ux%u", k.result.grid_x, k.result.grid_y);
            ImGui::TableNextColumn(); ImGui::Text("%.3f", k.result.elapsed_ms);

            // GPU time  highlight overhead
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", k.result.kernel_ms);
            if (ImGui::IsItemHovered() && k.result.kernel_ms > 0 &&
                k.result.elapsed_ms > k.result.kernel_ms * 1.05f) {
                float overhead_pct = ((k.result.elapsed_ms - k.result.kernel_ms) /
                                       k.result.elapsed_ms) * 100.0f;
                ImGui::SetTooltip("Host overhead: %.1f%% (%.3f ms)",
                    overhead_pct, k.result.elapsed_ms - k.result.kernel_ms);
            }

            ImGui::TableNextColumn();
            if (show_gflops) ImGui::Text("%.1f", k.result.gflops);
            else             ImGui::Text("%.1f", k.result.bandwidth_gbps);

            ImGui::TableNextColumn();
            if (k.result.success) {
                if (k.result.verified) ImGui::TextColored(UITheme::SUCCESS_GREEN, "OK");
                else                   ImGui::TextColored(UITheme::WARN_YELLOW, "WARN");
            } else {
                ImGui::TextColored(UITheme::ERROR_RED, "FAIL");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", k.result.error.c_str());
            }

            if (has_profiling) {
                ImGui::TableNextColumn();
                if (k.result.registers_per_thread > 0)
                    ImGui::Text("%d", k.result.registers_per_thread);
                else ImGui::TextColored(UITheme::TEXT_DIM, "-");

                ImGui::TableNextColumn();
                if (k.result.shared_memory_bytes > 0)
                    ImGui::Text("%d", k.result.shared_memory_bytes);
                else ImGui::TextColored(UITheme::TEXT_DIM, "-");

                ImGui::TableNextColumn();
                if (k.result.achieved_occupancy > 0)
                    ImGui::Text("%.1f", k.result.achieved_occupancy * 100.0);
                else ImGui::TextColored(UITheme::TEXT_DIM, "-");

                ImGui::TableNextColumn();
                if (k.result.ipc > 0) ImGui::Text("%.2f", k.result.ipc);
                else ImGui::TextColored(UITheme::TEXT_DIM, "-");
            } else {
                ImGui::TableNextColumn();
                ImGui::Text("%d", k.result.registers_per_thread);
            }
        }

        ImGui::EndTable();
    }
}

// ============================================================================
// Right Sidebar  Profile & Detail Panel
// ============================================================================
void Gui::render_profile_sidebar() {
    float s = ui_scale_;
    const auto* sel = selected_kernel();

    if (!sel) {
        ImGui::TextColored(UITheme::TEXT_DIM, "Select a kernel to see details");
        return;
    }

    auto* desc = sel->descriptor;
    const auto& r = sel->result;
    bool has_data     = sel->has_run;
    bool has_counters = has_data && r.achieved_occupancy > 0;

    // ================================================================
    // Compilation Info
    // ================================================================
    ImGui::TextColored(UITheme::HEADER_TEXT, "Compilation");
    ImGui::Separator();
    ImGui::Spacing();

    render_dsl_badge(detect_dsl_type(desc));
    ImGui::SameLine();
    ImGui::Text("%s", desc->name().c_str());

    if (desc->needs_compilation()) {
        std::string src = desc->source_path();
        if (src.length() > 28) {
            ImGui::TextColored(UITheme::TEXT_DIM, "...%s", src.c_str() + src.length() - 28);
        } else {
            ImGui::TextColored(UITheme::TEXT_DIM, "%s", src.c_str());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Source: %s", src.c_str());
        }
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Source: built-in (%s)",
            r.uses_module ? "cubin" : "runtime");
    }

    // TODO: wire to backend  expose cache hit/miss from KernelCompiler
    ImGui::TextColored(UITheme::TEXT_DIM, "Cache: N/A");
    // TODO: wire to backend  track compilation time
    ImGui::TextColored(UITheme::TEXT_DIM, "Compile time: N/A");

    // Error output
    if (has_data && !r.success && !r.error.empty()) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.25f, 0.05f, 0.05f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Text, UITheme::ERROR_RED);
        std::string err_copy = r.error;
        ImGui::InputTextMultiline("##err", err_copy.data(), err_copy.size() + 1,
            {-1, 60 * s}, ImGuiInputTextFlags_ReadOnly);
        ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // ================================================================
    // Hardware Counters
    // ================================================================
    ImGui::TextColored(UITheme::HEADER_TEXT, "Hardware Counters");
    ImGui::Separator();
    ImGui::Spacing();

    if (!has_counters && has_data) {
        ImGui::TextColored(UITheme::WARN_YELLOW, "Profiler unavailable");
        ImGui::TextColored(UITheme::TEXT_DIM, "Enable 'Profile' and see");
        ImGui::TextColored(UITheme::TEXT_DIM, "README profiling section");
        ImGui::Spacing();
    }

    // Registers (from Activity API  usually available)
    if (has_data && r.registers_per_thread > 0) {
        ImGui::Text("Registers/thread: %d", r.registers_per_thread);
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Registers/thread: --");
    }

    // Shared memory
    if (has_data && r.shared_memory_bytes > 0) {
        if (r.shared_memory_bytes >= 1024)
            ImGui::Text("Shared mem: %.1f KB", r.shared_memory_bytes / 1024.0f);
        else
            ImGui::Text("Shared mem: %d B", r.shared_memory_bytes);
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Shared mem: --");
    }

    ImGui::Spacing();

    // Color-coded progress bars for profiler metrics
    auto colored_bar = [&](const char* label, float value, float max_val, const char* fmt) {
        ImGui::Text("%s", label);
        float pct = max_val > 0 ? value / max_val : 0;
        ImVec4 bar_color = (pct < 0.33f) ? UITheme::ERROR_RED :
                           (pct < 0.66f) ? UITheme::WARN_YELLOW : UITheme::SUCCESS_GREEN;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
        char overlay[64];
        snprintf(overlay, sizeof(overlay), fmt, value);
        ImGui::ProgressBar(pct, {-1, 0}, overlay);
        ImGui::PopStyleColor();
    };

    if (has_counters) {
        colored_bar("Achieved Occupancy",
            (float)(r.achieved_occupancy * 100.0), 100.0f, "%.1f%%");

        if (r.ipc > 0) {
            colored_bar("IPC", (float)r.ipc, 4.0f, "%.2f");
        }

        double total_dram = r.dram_read_gbps + r.dram_write_gbps;
        if (total_dram > 0) {
            // TODO: wire to backend for theoretical peak DRAM BW
            colored_bar("DRAM Throughput", (float)total_dram, 1000.0f, "%.1f GB/s");
            ImGui::TextColored(UITheme::TEXT_DIM, "  R: %.1f  W: %.1f GB/s",
                r.dram_read_gbps, r.dram_write_gbps);
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, UITheme::TEXT_DIM);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, {0.15f, 0.15f, 0.15f, 1.0f});
        ImGui::Text("Occupancy");
        ImGui::ProgressBar(0, {-1, 0}, "--");
        ImGui::Text("IPC");
        ImGui::ProgressBar(0, {-1, 0}, "--");
        ImGui::Text("DRAM Throughput");
        ImGui::ProgressBar(0, {-1, 0}, "--");
        ImGui::PopStyleColor(2);

        if (has_data && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enable GPU perf counters - see README profiling section");
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // ================================================================
    // Verification Result
    // ================================================================
    ImGui::TextColored(UITheme::HEADER_TEXT, "Verification");
    ImGui::Separator();
    ImGui::Spacing();

    if (has_data && r.success) {
        ImFont* font = ImGui::GetFont();
        float saved = font->Scale;
        font->Scale *= 1.3f;
        ImGui::PushFont(font);
        if (r.verified) {
            ImGui::TextColored(UITheme::SUCCESS_GREEN, "PASS");
        } else {
            ImGui::TextColored(UITheme::ERROR_RED, "FAIL");
        }
        font->Scale = saved;
        ImGui::PopFont();
    } else if (has_data) {
        ImGui::TextColored(UITheme::ERROR_RED, "Execution Failed");
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Not yet run");
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // ================================================================
    // Launch Configuration
    // ================================================================
    ImGui::TextColored(UITheme::HEADER_TEXT, "Launch Configuration");
    ImGui::Separator();
    ImGui::Spacing();

    if (has_data && r.success) {
        ImGui::Text("Grid:  (%u, %u, %u)", r.grid_x, r.grid_y, r.grid_z);
        ImGui::Text("Block: (%u, %u, %u)", r.block_x, r.block_y, r.block_z);
        unsigned long long total_threads =
            (unsigned long long)r.grid_x * r.grid_y * r.grid_z *
            (unsigned long long)r.block_x * r.block_y * r.block_z;
        ImGui::Text("Total threads: %llu", total_threads);
        if (r.shared_mem_bytes > 0) {
            ImGui::Text("Shared mem: %u B", r.shared_mem_bytes);
        }
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Run a benchmark first");
    }

    // ================================================================
    // Sub-Kernel Breakdown
    // ================================================================
    if (has_data && r.success && !r.sub_kernels.empty()) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::TextColored(UITheme::HEADER_TEXT, "Sub-Kernels (%zu)", r.sub_kernels.size());
        ImGui::Separator();
        ImGui::Spacing();

        for (const auto& sk : r.sub_kernels) {
            ImGui::BulletText("%.3f ms | %d regs | %d B shmem",
                sk.duration_ms, sk.registers, sk.shared_memory);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", sk.name.c_str());
            }
        }
    }
}

// ============================================================================
// Bottom  Log / Event Feed
// ============================================================================
void Gui::render_log_panel() {
    float s = ui_scale_;

    // Collapse toggle + title
    bool collapsed = ui_state_.log_collapsed;
    if (ImGui::ArrowButton("##logcol", collapsed ? ImGuiDir_Right : ImGuiDir_Down)) {
        ui_state_.log_collapsed = !ui_state_.log_collapsed;
    }
    ImGui::SameLine();
    ImGui::TextColored(UITheme::HEADER_TEXT, "Event Log");

    if (ui_state_.log_collapsed) return;

    // Controls
    float ctrl_x = ImGui::GetContentRegionMax().x - 280 * s;
    if (ctrl_x > ImGui::GetCursorPosX() + 50 * s) {
        ImGui::SameLine(ctrl_x);
    }

    const char* filter_names[] = {"All", "Errors Only", "Current Kernel"};
    int filter_idx = (int)ui_state_.log_filter;
    ImGui::SetNextItemWidth(120 * s);
    if (ImGui::Combo("##logfilt", &filter_idx, filter_names, 3)) {
        ui_state_.log_filter = (LogFilter)filter_idx;
    }

    ImGui::SameLine();
    ImGui::Checkbox("Auto", &ui_state_.autoscroll);

    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        log_entries_.clear();
    }

    // Scrollable log content
    ImGui::BeginChild("##LogScroll", {0, ImGui::GetContentRegionAvail().y}, false);

    for (const auto& entry : log_entries_) {
        // Apply filter
        if (ui_state_.log_filter == LogFilter::ErrorsOnly &&
            entry.level != LogEntry::ERR && entry.level != LogEntry::WARN)
            continue;
        if (ui_state_.log_filter == LogFilter::CurrentKernelOnly &&
            !ui_state_.selected_kernel_name.empty() &&
            entry.message.find(ui_state_.selected_kernel_name) == std::string::npos)
            continue;

        ImVec4 color;
        const char* prefix;
        switch (entry.level) {
            case LogEntry::INFO:      color = UITheme::LOG_INFO;      prefix = "[INFO]      "; break;
            case LogEntry::WARN:      color = UITheme::LOG_WARN;      prefix = "[WARN]      "; break;
            case LogEntry::ERR:       color = UITheme::LOG_ERROR;     prefix = "[ERROR]     "; break;
            case LogEntry::COMPILE:   color = UITheme::LOG_COMPILE;   prefix = "[COMPILE]   "; break;
            case LogEntry::BENCHMARK: color = UITheme::LOG_BENCHMARK; prefix = "[BENCHMARK] "; break;
            case LogEntry::PROFILE:   color = UITheme::LOG_PROFILE;   prefix = "[PROFILE]   "; break;
        }
        ImGui::TextColored(color, "%s%s", prefix, entry.message.c_str());
    }

    if (ui_state_.autoscroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20) {
        ImGui::SetScrollHereY(1.0f);
    }

    ImGui::EndChild();
}

// ============================================================================
// Benchmark thread (unchanged logic, preserved from original)
// ============================================================================
void Gui::benchmark_thread_func(
    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
    arena::RunConfig config) {

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
        pr.params = config.params;
        pr.logs.push_back({LogEntry::INFO, "Running " + descriptor->name() + " ..."});

        pr.result = runner_.run(*descriptor, config);

        if (pr.result.success) {
            char buf[256];
            bool matmul = (cat == "matmul");
            snprintf(buf, sizeof(buf), "%s: wall=%.3f ms  kernel=%.3f ms  %.2f %s",
                pr.result.kernel_name.c_str(),
                pr.result.elapsed_ms, pr.result.kernel_ms,
                matmul ? pr.result.gflops : pr.result.bandwidth_gbps,
                matmul ? "GFLOPS" : "GB/s");
            pr.logs.push_back({LogEntry::INFO, buf});

            if (config.collect_metrics && pr.result.achieved_occupancy > 0) {
                snprintf(buf, sizeof(buf),
                    "%s: regs=%d  shmem=%dB  occupancy=%.1f%%  IPC=%.2f",
                    pr.result.kernel_name.c_str(),
                    pr.result.registers_per_thread,
                    pr.result.shared_memory_bytes,
                    pr.result.achieved_occupancy * 100.0,
                    pr.result.ipc);
                pr.logs.push_back({LogEntry::INFO, buf});
            }

            if (!pr.result.verified) {
                pr.logs.push_back({LogEntry::WARN,
                    pr.result.kernel_name + ": verification FAILED"});
            }
        } else {
            pr.logs.push_back({LogEntry::ERR,
                pr.result.kernel_name + ": " + pr.result.error});
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

void Gui::sweep_thread_func(
    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
    std::vector<std::map<std::string, int>> sweep_configs,
    arena::RunConfig config) {

    CUcontext ctx = runner_.context().handle();
    cuCtxPushCurrent(ctx);

    int i = 0;
    for (const auto& params : sweep_configs) {
        if (cancel_requested_) break;
        config.params = params;

        for (auto& [cat, descriptor] : work) {
            if (cancel_requested_) break;

            benchmark_current_ = i++;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                benchmark_current_name_ = descriptor->name();
            }

            PendingResult pr;
            pr.category = cat;
            pr.kernel_name = descriptor->name();
            pr.params = params;

            std::string size_str;
            for (auto& [k, v] : params) {
                if (!size_str.empty()) size_str += ",";
                size_str += k + "=" + std::to_string(v);
            }
            pr.logs.push_back({LogEntry::INFO,
                "Sweep " + descriptor->name() + " [" + size_str + "] ..."});

            pr.result = runner_.run(*descriptor, config);

            if (pr.result.success) {
                char buf[256];
                bool matmul = (cat == "matmul");
                snprintf(buf, sizeof(buf), "%s [%s]: wall=%.3f ms  %.2f %s",
                    pr.result.kernel_name.c_str(), size_str.c_str(),
                    pr.result.elapsed_ms,
                    matmul ? pr.result.gflops : pr.result.bandwidth_gbps,
                    matmul ? "GFLOPS" : "GB/s");
                pr.logs.push_back({LogEntry::INFO, buf});
            } else {
                pr.logs.push_back({LogEntry::ERR,
                    pr.result.kernel_name + ": " + pr.result.error});
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending_results_.push_back(std::move(pr));
            }
        }
    }

    benchmark_current_ = (int)(work.size() * sweep_configs.size());

    CUcontext popped;
    cuCtxPopCurrent(&popped);

    benchmark_running_ = false;
}

// ============================================================================
// Run commands
// ============================================================================
void Gui::run_selected_kernels() {
    if (current_category_.empty()) return;
    if (benchmark_running_) return;

    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return;

    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work;
    for (auto& k : it->second) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor});
        }
    }
    if (work.empty()) return;

    if (benchmark_thread_.joinable()) benchmark_thread_.join();

    cancel_requested_ = false;
    benchmark_running_ = true;
    benchmark_current_ = 0;
    benchmark_total_ = (int)work.size();

    benchmark_thread_ = std::thread(&Gui::benchmark_thread_func, this,
        std::move(work), config_);
}

void Gui::run_sweep() {
    if (current_category_.empty()) return;
    if (benchmark_running_) return;

    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return;

    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work;
    for (auto& k : it->second) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor});
        }
    }
    if (work.empty()) return;

    auto sweep_configs = work[0].second->get_sweep_configs();
    if (sweep_configs.empty()) {
        log(LogEntry::WARN, "No sweep configs defined for this category");
        return;
    }

    if (benchmark_thread_.joinable()) benchmark_thread_.join();

    cancel_requested_ = false;
    benchmark_running_ = true;
    benchmark_current_ = 0;
    benchmark_total_ = (int)(work.size() * sweep_configs.size());

    benchmark_thread_ = std::thread(&Gui::sweep_thread_func, this,
        std::move(work), std::move(sweep_configs), config_);
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

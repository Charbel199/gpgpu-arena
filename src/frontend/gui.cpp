#include "frontend/gui.hpp"
#include "arena/cli_args.hpp"
#include "arena/result_io.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <cuda.h>
#include <stdexcept>
#include <algorithm>
#include <implot_internal.h>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <fstream>
#include <numeric>
#include <unistd.h>

namespace frontend {

namespace {

// The configs a kernel will be tried at, whichever axis it happens to have.
std::vector<arena::cli::TuningVariant> tuning_variants(
    const arena::KernelDescriptor& d) {
    return arena::cli::tuning_variants_for(
        true, 0, {}, d.tunable_block_sizes(), d.tunable_compile_options());
}

// What the table shows for one config. A CUDA kernel varies one number, so
// "block=256" says everything; a DSL kernel varies named knobs. A kernel with
// no axis at all reads "default": a block size of 0 means the descriptor
// chooses, not that anything launched with zero threads.
std::string tuning_label_for(const arena::cli::TuningVariant& v) {
    if (!v.defines.empty()) {
        std::string out;
        for (const auto& [key, val] : v.defines) {
            if (!out.empty()) out += ", ";
            out += key + "=" + std::to_string(val);
        }
        return out;
    }
    if (v.block_size <= 0) return "default";
    return "block=" + std::to_string(v.block_size);
}

}   // namespace

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

    // Storage type. Deliberately cool-to-warm as precision drops, so a table
    // mixing dtypes reads at a glance without checking the Type column.
    // Keyed on the input>output pair, not just the input: fp16>fp32 and
    // fp16>fp16 keep very different amounts of precision and should not look
    // alike. Cooler is more precise.
    constexpr ImVec4 DTYPE_F32_F32 = {0.36f,  0.66f,  0.94f,  1.0f};  // blue
    constexpr ImVec4 DTYPE_F16_F32 = {0.96f,  0.70f,  0.24f,  1.0f};  // amber
    constexpr ImVec4 DTYPE_F16_F16 = {0.93f,  0.39f,  0.24f,  1.0f};  // burnt orange
    constexpr ImVec4 DTYPE_B16_F32 = {0.72f,  0.55f,  0.90f,  1.0f};  // violet
    constexpr ImVec4 DTYPE_B16_B16 = {0.90f,  0.42f,  0.62f,  1.0f};  // rose
    constexpr ImVec4 DTYPE_F8_F32  = {0.95f,  0.30f,  0.45f,  1.0f};  // crimson
    constexpr ImVec4 DTYPE_F8_F8   = {0.80f,  0.20f,  0.35f,  1.0f};  // deep crimson
    constexpr ImVec4 DTYPE_F4      = {0.65f,  0.15f,  0.45f,  1.0f};  // magenta
    constexpr ImVec4 DTYPE_OTHER   = {0.60f,  0.60f,  0.60f,  1.0f};
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
    constexpr float HEADER_HEIGHT       = 68.0f;
    constexpr float SIDEBAR_LEFT_WIDTH  = 260.0f;
    constexpr float SIDEBAR_RIGHT_WIDTH = 290.0f;
    constexpr float LOG_HEIGHT          = 160.0f;
    constexpr float LOG_COLLAPSED_HEIGHT = 28.0f;
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

    // Compute GPU theoretical peaks
    const auto& ctx = runner_.context();
    int sms = ctx.sm_count();
    int clock_khz = ctx.clock_rate_khz();
    int mem_clock_khz = ctx.memory_clock_khz();
    int bus_width = ctx.memory_bus_width();
    int cc = ctx.compute_capability_major();
    int cc_minor = ctx.compute_capability_minor();

    // FP32 cores per SM by compute capability
    int fp32_per_sm = 128;
    if (cc == 7) fp32_per_sm = 64;
    else if (cc == 8 && cc_minor == 0) fp32_per_sm = 64;  // A100

    peak_fp32_gflops_ = (float)sms * fp32_per_sm * clock_khz * 2.0f / 1e6f;
    peak_mem_bw_gbs_  = (float)mem_clock_khz * 2.0f * (bus_width / 8.0f) / 1e6f;

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

        // A tuning run measures configs the kernel is not normally launched
        // at, so it goes to its own table rather than replacing the headline
        // result the rest of the UI reads.
        if (!pr.tuning_label.empty()) {
            if (pr.result.success) {
                tuning_history_[pr.category][pr.kernel_name].push_back(
                    {pr.tuning_label, pr.tuning_variant, pr.result});
            }
            continue;
        }

        auto cat_it = kernels_by_category_.find(pr.category);
        if (cat_it != kernels_by_category_.end()) {
            for (auto& k : cat_it->second) {
                if (k.descriptor && k.descriptor->name() == pr.kernel_name) {
                    k.result = pr.result;
                    k.result_config = pr.config_label;
                    k.has_run = true;
                    break;
                }
            }
        }

        // Everything this run measured goes into the snapshot it will be
        // committed as, so history records the run rather than whatever
        // happens to be on screen when it ends.
        if (pr.result.success && pr.tuning_label.empty()) {
            // A sweep walks several problem sizes in one press. Each size is
            // its own run rather than the last one overwriting the rest, so a
            // sweep produces a table per size instead of a single table whose
            // other sizes vanished.
            if (!pending_snapshot_.results.empty() &&
                pending_snapshot_.params != pr.params) {
                commit_snapshot();
            }
            pending_snapshot_.results[pr.kernel_name] = pr.result;
            pending_snapshot_.configs[pr.kernel_name] = pr.config_label;
            pending_snapshot_.category = pr.category;
            pending_snapshot_.params   = pr.params;
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

            // next_snapshot_id_ is the id this run will be committed under, so
            // tagging now is what lets a later delete find these points again.
            auto& hist = scaling_history_[pr.category][pr.kernel_name];
            bool found = false;
            for (auto& entry : hist) {
                if (entry.problem_size == problem_size) {
                    entry.result = pr.result;
                    entry.run_id = next_snapshot_id_;
                    found = true;
                    break;
                }
            }
            if (!found) {
                hist.push_back({problem_size, next_snapshot_id_, pr.result});
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
        // A cancelled run still measured whatever it got through, and that is
        // worth keeping rather than discarding.
        commit_snapshot();
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

// Always "input>output", even when they match. Collapsing fp32>fp32 to "fp32"
// leaves the reader guessing whether it meant the input, the output, or both.
static std::string dtype_label(const arena::KernelDescriptor* d) {
    return std::string(arena::dtype_name(d->input_dtype())) + ">"
         + arena::dtype_name(d->output_dtype());
}

static std::string dtype_label(const std::string& in, const std::string& out) {
    return in + ">" + out;
}

// Colour for an "input>output" label.
static ImVec4 dtype_color(const std::string& label) {
    if (label == "fp32>fp32") return UITheme::DTYPE_F32_F32;
    if (label == "fp16>fp32") return UITheme::DTYPE_F16_F32;
    if (label == "fp16>fp16") return UITheme::DTYPE_F16_F16;
    if (label == "bf16>fp32") return UITheme::DTYPE_B16_F32;
    if (label == "bf16>bf16") return UITheme::DTYPE_B16_B16;
    // Narrower still. Matched on prefix so both fp8 variants share a family.
    if (label.rfind("fp4", 0) == 0)  return UITheme::DTYPE_F4;
    if (label.rfind("fp8", 0) == 0)
        return label.find(">fp32") != std::string::npos ? UITheme::DTYPE_F8_F32
                                                        : UITheme::DTYPE_F8_F8;
    return UITheme::DTYPE_OTHER;
}

DSLType Gui::detect_dsl_type(const arena::KernelDescriptor* desc) const {
    // One source of truth, shared with the CLI. Previously this tested
    // uses_module() first, which mislabelled reduce_two_stage as CUB: that
    // flag describes how a kernel is launched, not what it is written in.
    const std::string dsl = arena::result_io::detect_dsl(desc);
    if (dsl == "triton") return DSLType::Triton;
    if (dsl == "cutile") return DSLType::CuTile;
    if (dsl == "warp")   return DSLType::Warp;
    if (dsl == "cub")    return DSLType::CUB;
    return DSLType::CUDA;
}

void Gui::format_time(float ms, char* buf, size_t buf_size) {
    if (ms <= 0.0f)     { snprintf(buf, buf_size, "--"); return; }
    if (ms < 0.001f)      snprintf(buf, buf_size, "%.0f ns", ms * 1e6f);
    else if (ms < 1.0f)   snprintf(buf, buf_size, "%.1f us", ms * 1000.0f);
    else if (ms < 1000.0f) snprintf(buf, buf_size, "%.2f ms", ms);
    else                   snprintf(buf, buf_size, "%.2f s",  ms / 1000.0f);
}

const char* Gui::dsl_type_name(DSLType type) const {
    switch (type) {
        case DSLType::CUDA:   return "CUDA";
        case DSLType::Triton: return "Triton";
        case DSLType::CuTile: return "cuTile";
        case DSLType::Warp:   return "Warp";
        case DSLType::CUB:    return "CUB";
    }
    return "Unknown";
}

void Gui::export_results_csv() {
    // Use absolute path so the user can find the file
    char cwd[1024];
    std::string path = "gpgpu_arena_results.csv";
    if (getcwd(cwd, sizeof(cwd))) {
        path = std::string(cwd) + "/gpgpu_arena_results.csv";
    }
    std::ofstream f(path);
    if (!f.is_open()) { log(LogEntry::ERR, "Failed to open " + path); return; }

    f << arena::result_io::csv_header() << "\n";

    for (const auto& [cat, states] : kernels_by_category_) {
        for (const auto& k : states) {
            if (!k.has_run) continue;
            f << arena::result_io::csv_row(
                     k.result, dsl_type_name(detect_dsl_type(k.descriptor)))
              << "\n";
        }
    }
    log(LogEntry::INFO, "Exported to " + path);
}

std::string Gui::read_kernel_source(const std::string& rel_path) {
    auto it = source_cache_.find(rel_path);
    if (it != source_cache_.end()) return it->second;

    std::string full_path = std::string(ARENA_KERNEL_DIR) + "/" + rel_path;
    std::ifstream f(full_path);
    if (!f.is_open()) return "// Could not read " + full_path;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    source_cache_[rel_path] = content;
    return content;
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
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Peak FP32: %.0f GFLOPS | Peak Mem BW: %.0f GB/s\n"
            "Clock: %d MHz | Mem Clock: %d MHz | Bus: %d-bit",
            peak_fp32_gflops_, peak_mem_bw_gbs_,
            ctx.clock_rate_khz() / 1000, ctx.memory_clock_khz() / 1000,
            ctx.memory_bus_width());
    }

    // Right section fixed positions so nothing shifts between idle/running
    float right_edge = ImGui::GetContentRegionMax().x;
    float btn_w = 52 * s;
    float gap = ImGui::GetStyle().ItemSpacing.x;

    // 3 buttons pinned to right edge (always same position)
    float btns_start = right_edge - btn_w * 3 - gap * 2;
    ImGui::SameLine(btns_start);

    ImGui::BeginDisabled(benchmark_running_);
    if (ImGui::Button("Export", {btn_w, 0})) {
        export_results_csv();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(benchmark_running_);
    if (ImGui::Button("Reset", {btn_w, 0})) {
        runner_.mutable_context().reset();
        log(LogEntry::WARN, "GPU context reset");
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (benchmark_running_) {
        ImGui::PushStyleColor(ImGuiCol_Button, {0.5f, 0.1f, 0.1f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.65f, 0.15f, 0.15f, 1.0f});
        if (ImGui::Button("Stop", {btn_w, 0})) {
            cancel_requested_ = true;
        }
        ImGui::PopStyleColor(2);
    } else {
        if (ImGui::Button("Run", {btn_w, 0})) {
            run_selected_kernels();
        }
    }

    // Status text between GPU info and buttons (fixed slot)
    float status_x = btns_start - 180 * s;
    if (status_x > 400 * s) {
        ImGui::SameLine(status_x);
        if (benchmark_running_) {
            if (config_.collect_metrics)
                ImGui::TextColored(UITheme::LOG_PROFILE, "PROFILING");
            else
                ImGui::TextColored(UITheme::ACCENT, "RUNNING");
            ImGui::SameLine();
            int cur = benchmark_current_.load();
            int tot = benchmark_total_.load();
            float frac = tot > 0 ? (float)cur / (float)tot : 0.0f;
            ImGui::ProgressBar(frac, {80 * s, 0});
        } else {
            int total_k = 0;
            for (const auto& [c, ks] : kernels_by_category_) total_k += (int)ks.size();
            ImGui::TextColored(UITheme::TEXT_DIM, "IDLE | %d kernels", total_k);
        }
    }

    // Second line: what the next run will actually measure. Problem size,
    // input data and any pinned configs all change the numbers, and all of
    // them used to be several clicks away from the results they explain.
    ImGui::Separator();

    std::string size_str;
    for (const auto& name : {"n", "M", "K", "N", "rows", "cols"}) {
        auto it = config_.params.find(name);
        if (it == config_.params.end()) continue;
        if (current_category_ == "matmul" && std::string(name) == "n") continue;
        if (current_category_ != "matmul" &&
            (std::string(name) == "M" || std::string(name) == "K" ||
             std::string(name) == "N")) continue;
        if (current_category_ != "softmax" &&
            (std::string(name) == "rows" || std::string(name) == "cols")) continue;
        if (!size_str.empty()) size_str += " ";
        size_str += std::string(name) + "=" + std::to_string(it->second);
    }

    int sel_n = 0, pinned_n = 0;
    if (auto* ks = current_kernels()) {
        for (const auto& k : *ks) {
            if (k.selected) sel_n++;
            if (k.has_pinned) pinned_n++;
        }
    }

    // A lost context is the one condition where nothing on screen means
    // anything any more, so it takes over the status line.
    if (runner_.context().device_lost()) {
        ImGui::TextColored(UITheme::ERROR_RED,
            "CUDA CONTEXT LOST - restart required. Cause: %s",
            runner_.context().lost_reason().c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "A kernel made an illegal memory access. CUDA cannot clear that\n"
                "inside a running process, so every later kernel fails with\n"
                "\"invalid device context\" regardless of whether it is correct.\n"
                "Restart the app; the named kernel is the one to look at.");
        }
        return;   // inside the header child, so nothing to close here
    }

    ImGui::TextColored(UITheme::TEXT_DIM, "%s | %s | %s seed=%llu | %d selected",
        current_category_.empty() ? "no category" : current_category_.c_str(),
        size_str.empty() ? "default size" : size_str.c_str(),
        arena::distribution_name(config_.input_distribution),
        (unsigned long long)config_.input_seed, sel_n);

    if (pinned_n > 0) {
        ImGui::SameLine();
        ImGui::TextColored(UITheme::ACCENT, "| %d tuned", pinned_n);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%d kernel(s) pinned to a non-default config", pinned_n);
    }

    ImGui::SameLine();
    ImGui::TextColored(UITheme::TEXT_DIM, "| runs=%d%s%s",
        config_.number_of_runs,
        config_.collect_metrics ? " +profile" : "",
        config_.collect_energy  ? " +energy"  : "");
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

    // Type legend, doubling as a filter: clicking one selects exactly the
    // kernels of that type. Only the types actually present are listed, so a
    // category of plain fp32 kernels costs one short line.
    {
        std::vector<std::string> seen;
        for (const auto& k : *kernels) {
            const std::string lbl = dtype_label(k.descriptor);
            if (std::find(seen.begin(), seen.end(), lbl) == seen.end()) seen.push_back(lbl);
        }
        for (size_t li = 0; li < seen.size(); li++) {
            if (li) ImGui::SameLine(0, 6 * s);
            const ImVec4 col = dtype_color(seen[li]);

            ImGui::PushID((int)li + 9000);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            ImGui::PushStyleColor(ImGuiCol_Button, {col.x, col.y, col.z, 0.14f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {col.x, col.y, col.z, 0.30f});
            if (ImGui::SmallButton(seen[li].c_str())) {
                for (auto& k : *kernels) k.selected = (dtype_label(k.descriptor) == seen[li]);
            }
            ImGui::PopStyleColor(3);
            ImGui::PopID();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Select only %s kernels", seen[li].c_str());
        }
        ImGui::Separator();
    }

    // Read running kernel name once (avoid repeated locking)
    std::string running_name;
    if (benchmark_running_) {
        std::lock_guard<std::mutex> lock(mutex_);
        running_name = benchmark_current_name_;
    }

    // ---- Kernel list (scrollable middle) ----
    float bottom_reserve = 315 * s;
    float list_h = ImGui::GetContentRegionAvail().y - bottom_reserve;
    if (list_h < 80 * s) list_h = 80 * s;
    ImGui::BeginChild("##KernelList", {0, list_h}, false);

    for (size_t i = 0; i < kernels->size(); i++) {
        auto& k = (*kernels)[i];
        bool is_display_sel = (k.descriptor->name() == ui_state_.selected_kernel_name);
        bool is_running = benchmark_running_ && (k.descriptor->name() == running_name);

        ImGui::PushID((int)i);

        // Dtype stripe pinned to the panel's left edge, so a mixed-precision
        // list reads without checking the type text on every row. Taken from
        // the descriptor, not the result: this list exists before anything runs.
        {
            const float  x0 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x;
            const ImVec2 rp = ImGui::GetCursorScreenPos();
            const float  rh = ImGui::GetFrameHeight();
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(x0, rp.y), ImVec2(x0 + 3.0f * s, rp.y + rh),
                ImGui::GetColorU32(dtype_color(dtype_label(k.descriptor))),
                1.5f * s);
        }
        ImGui::Indent(8.0f * s);

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
        float name_w = ImGui::GetContentRegionAvail().x - 62 * s;
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
            ImGui::Separator();
            ImGui::TextDisabled("in %s, out %s",
                arena::dtype_name(k.descriptor->input_dtype()),
                arena::dtype_name(k.descriptor->output_dtype()));
            ImGui::Separator();
            ImGui::TextDisabled("config: %s%s",
                k.has_pinned ? tuning_label_for(k.pinned).c_str() : "default",
                k.has_pinned ? " (pinned)" : "");
            if (const auto* snap = compare_snapshot()) {
                auto it = snap->results.find(k.descriptor->name());
                if (k.has_run && it != snap->results.end() && k.result.op_ms > 0.0f) {
                    ImGui::TextDisabled("%s: %.4f ms, now %.4f ms",
                        snap->name.c_str(), it->second.op_ms, k.result.op_ms);
                }
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }

        // Marks a kernel that is no longer running at its own default, so a
        // tuned list does not look like an untuned one.
        if (k.has_pinned) {
            ImGui::SameLine(ImGui::GetContentRegionMax().x - 66 * s);
            ImGui::TextColored(UITheme::ACCENT, "*");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Pinned to %s", tuning_label_for(k.pinned).c_str());
        }

        // Status + time (right-aligned)
        ImGui::SameLine(ImGui::GetContentRegionMax().x - 55 * s);
        if (is_running) {
            ImGui::TextColored(UITheme::ACCENT, "...");
        } else if (!k.has_run) {
            ImGui::TextColored(UITheme::TEXT_DIM, "--");
        } else if (k.result.success) {
            char tbuf[32];
            format_time(k.result.op_ms, tbuf, sizeof(tbuf));
            ImGui::TextColored(k.result.verified ? UITheme::SUCCESS_GREEN : UITheme::WARN_YELLOW,
                "%s", tbuf);
        } else {
            ImGui::TextColored(UITheme::ERROR_RED, "ERR");
        }

        ImGui::Unindent(8.0f * s);
        ImGui::PopID();
    }

    ImGui::EndChild();

    // ---- Bottom: size and settings share the space, buttons never do ----
    ImGui::Separator();

    if (ImGui::BeginTabBar("##SidebarTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Size")) {
            ImGui::Spacing();
            render_problem_config();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Settings")) {
            ImGui::Spacing();
            render_run_controls();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    render_run_buttons();
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

    // No warmup slider: warmup runs until the timings stop drifting, so a
    // fixed count is not a thing to set. Auto is the mode; the fixed count
    // stays reachable from the CLI with --warmup for reproducing a run.
    bool fixed_warmup = (config_.warmup_mode == arena::RunConfig::WarmupMode::Fixed);
    if (ImGui::Checkbox("Fixed warmup", &fixed_warmup)) {
        config_.warmup_mode = fixed_warmup ? arena::RunConfig::WarmupMode::Fixed
                                           : arena::RunConfig::WarmupMode::Auto;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Off: warm up until the median stops drifting (up to %d runs or %.0f ms).\n"
            "On: always run exactly %d, for reproducing a specific run.",
            config_.warmup_max, config_.warmup_max_ms, config_.warmup_runs);
    }
    if (fixed_warmup) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90 * s);
        ImGui::SliderInt("##warmupn", &config_.warmup_runs, 1, 50);
    }
    ImGui::SliderInt("Runs", &config_.number_of_runs, 1, 100);
    ImGui::Checkbox("Profile", &config_.collect_metrics);
    ImGui::SameLine();
    ImGui::Checkbox("Energy", &config_.collect_energy);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Sustained-load energy pass (adds ~%.0f ms per kernel).\n"
            "Measures pipelined throughput, not isolated launches.\n"
            "Whole-board energy: treat mJ/op as an upper bound.",
            config_.energy_window_ms);
    }
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
}

// The run buttons live outside the settings so they are always reachable:
// starting a run is the one thing this panel exists for.
void Gui::render_run_buttons() {
    float s = ui_scale_;

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

        int pinned_count = 0;
        if (auto* ks = current_kernels()) {
            for (const auto& k : *ks) if (k.selected && k.has_pinned) pinned_count++;
        }
        if (pinned_count > 0) {
            ImGui::TextColored(UITheme::ACCENT, "%d kernel(s) at a tuned config",
                               pinned_count);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                if (auto* ks = current_kernels()) {
                    for (const auto& k : *ks) {
                        if (!k.selected || !k.has_pinned || !k.descriptor) continue;
                        ImGui::BulletText("%s: %s", k.descriptor->name().c_str(),
                                          tuning_label_for(k.pinned).c_str());
                    }
                }
                ImGui::EndTooltip();
            }
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
                        auto configs = k.descriptor->get_sweep_configs(config_);
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

        // Tuning is a separate axis from problem size: it holds n fixed and
        // varies how the kernel is launched or built.
        int tuning_runs = 0, tunable_kernels = 0;
        if (auto* ks = current_kernels()) {
            for (const auto& k : *ks) {
                if (!k.selected || !k.descriptor) continue;
                const auto n = tuning_variants(*k.descriptor).size();
                if (n > 1) { tunable_kernels++; tuning_runs += (int)n; }
            }
        }

        ImGui::BeginDisabled(tunable_kernels == 0);
        snprintf(lbl, sizeof(lbl), "Run Tuning (%d)", tuning_runs);
        if (ImGui::Button(lbl, {-1, 30 * s})) {
            run_tuning();
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            if (tunable_kernels == 0) {
                ImGui::Text("No selected kernel has a tuning axis.");
            } else {
                ImGui::Text("Sweep block size (CUDA) or compile config (DSL)");
                ImGui::Text("at the current problem size.");
                ImGui::Separator();
                if (auto* ks = current_kernels()) {
                    for (const auto& k : *ks) {
                        if (!k.selected || !k.descriptor) continue;
                        const auto v = tuning_variants(*k.descriptor);
                        if (v.size() < 2) continue;
                        ImGui::BulletText("%s: %zu configs", k.descriptor->name().c_str(), v.size());
                    }
                }
                ImGui::TextColored(UITheme::WARN_YELLOW,
                    "DSL configs recompile on first use.");
            }
            ImGui::EndTooltip();
        }

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

    // Plots size against the panel now that each tab holds only a few of
    // them, rather than the fixed heights they had when eight sections shared
    // one scroll. Taken once, before anything draws, so the second plot in a
    // tab is not smaller than the first.
    const float panel_h = ImGui::GetContentRegionAvail().y;
    auto plot_height = [&](float frac, float min_px) {
        const float h = panel_h * frac;
        return h < min_px * s ? min_px * s : h;
    };

    // Sections used to stack as collapsing headers in one long scroll, which
    // meant hunting for the panel you wanted. They are tabs now: each answers
    // a different question, and only one is ever competing for the space.
    if (!ImGui::BeginTabBar("##CenterTabs", ImGuiTabBarFlags_None)) return;

    if (ImGui::BeginTabItem("Runs")) {
        ImGui::Spacing();
        render_runs_tab();
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Compare")) {
        ImGui::Spacing();
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
                    bars.push_back({k.result.kernel_name, (double)k.result.op_ms,
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

                float plot_h = plot_height(0.34f, 200);
                if (ImPlot::BeginPlot("##Comparison", {-1, plot_h})) {
                    ImPlot::SetupAxes("", "Median Time (ms)",
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxisTicks(ImAxis_X1, tick_positions.data(),
                        (int)tick_positions.size(), tick_labels.data());

                    // Track which series the legend is currently showing, so the
                    // speedup labels can be suppressed alongside their bars.
                    std::vector<const char*> shown;
                    auto plot_dsl = [&](DSLType type, const char* name, ImVec4 color) {
                        auto it = groups.find(type);
                        if (it == groups.end()) return;
                        ImPlot::SetNextFillStyle(color);
                        ImPlot::PlotBars(name, it->second.positions.data(),
                            it->second.values.data(), (int)it->second.positions.size(), 0.6);
                        const ImPlotItem* item = ImPlot::GetItem(name);
                        if (!item || item->Show) shown.push_back(name);
                    };

                    auto dsl_shown = [&](DSLType type) {
                        const char* n = nullptr;
                        switch (type) {
                            case DSLType::CUDA:   n = "CUDA";   break;
                            case DSLType::Triton: n = "Triton"; break;
                            case DSLType::CuTile: n = "cuTile"; break;
                            case DSLType::Warp:   n = "Warp";   break;
                            case DSLType::CUB:    n = "CUB";    break;
                        }
                        return std::find(shown.begin(), shown.end(), n) != shown.end();
                    };

                    plot_dsl(DSLType::CUDA,   "CUDA",   UITheme::CUDA_BADGE);
                    plot_dsl(DSLType::Triton, "Triton", UITheme::TRITON_BADGE);
                    plot_dsl(DSLType::CuTile, "cuTile", UITheme::CUTILE_BADGE);
                    plot_dsl(DSLType::Warp,   "Warp",   UITheme::WARP_BADGE);
                    plot_dsl(DSLType::CUB,    "CUB",    UITheme::CUB_BADGE);

                    // Speedup labels above bars. PlotText is not a plot item, so it
                    // does not hide itself when its series is toggled off in the
                    // legend; the visibility check has to be explicit.
                    for (size_t i = 0; i < bars.size(); i++) {
                        if (!dsl_shown(bars[i].dsl)) continue;
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
        // Op Time vs GPU Time Comparison
        // ================================================================
        {
            struct TimeEntry {
                std::string name;
                double bar_op_ms;
                double gpu_ms;
            };
            std::vector<TimeEntry> entries;
            for (const auto& k : *kernels) {
                if (k.has_run && k.result.success) {
                    entries.push_back({k.result.kernel_name,
                        (double)k.result.op_ms, (double)k.result.gpu_ms});
                }
            }

            if (!entries.empty()) {
                if (ImGui::CollapsingHeader("Op vs GPU Time", ImGuiTreeNodeFlags_DefaultOpen)) {
                    static int time_sort = 0;
                    ImGui::SetNextItemWidth(150 * s);
                    const char* sort_opts[] = {"Sort: Op Time", "Sort: GPU Time", "Sort: Overhead"};
                    ImGui::Combo("##timesort", &time_sort, sort_opts, 3);

                    std::sort(entries.begin(), entries.end(),
                        [&](const TimeEntry& a, const TimeEntry& b) {
                            switch (time_sort) {
                                case 1:  return a.gpu_ms < b.gpu_ms;
                                case 2:  return (a.bar_op_ms - a.gpu_ms) > (b.bar_op_ms - b.gpu_ms);
                                default: return a.bar_op_ms < b.bar_op_ms;
                            }
                        });

                    int n = (int)entries.size();
                    std::vector<std::string> label_store(n);
                    std::vector<const char*> labels(n);
                    std::vector<double> positions(n), op_vals(n), gpu_vals(n);
                    for (int i = 0; i < n; i++) {
                        positions[i] = (double)i;
                        label_store[i] = entries[i].name;
                        labels[i] = label_store[i].c_str();
                        op_vals[i] = entries[i].bar_op_ms;
                        gpu_vals[i] = entries[i].gpu_ms;
                    }

                    double bw = 0.3;
                    std::vector<double> pos_op(n), pos_gpu(n);
                    for (int i = 0; i < n; i++) {
                        pos_op[i] = positions[i] - bw * 0.55;
                        pos_gpu[i]  = positions[i] + bw * 0.55;
                    }

                    float plot_h = plot_height(0.34f, 200);
                    if (ImPlot::BeginPlot("##OpGPU", {-1, plot_h})) {
                        ImPlot::SetupAxes("", "Time (ms)",
                            ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, labels.data());

                        ImPlot::SetNextFillStyle(UITheme::ACCENT);
                        ImPlot::PlotBars("Op Time", pos_op.data(), op_vals.data(), n, bw);

                        ImPlot::SetNextFillStyle({0.35f, 0.60f, 0.85f, 1.0f});
                        ImPlot::PlotBars("GPU Time", pos_gpu.data(), gpu_vals.data(), n, bw);

                        // Overhead % annotation above the op bar
                        for (int i = 0; i < n; i++) {
                            double overhead = op_vals[i] > 0
                                ? ((op_vals[i] - gpu_vals[i]) / op_vals[i]) * 100.0 : 0;
                            if (overhead > 1.0) {
                                char txt[32];
                                snprintf(txt, sizeof(txt), "+%.0f%%", overhead);
                                ImPlot::PlotText(txt, positions[i], op_vals[i], {0, -10});
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

                double peak = is_matmul() ? (double)peak_fp32_gflops_ : (double)peak_mem_bw_gbs_;

                float plot_h = plot_height(0.34f, 200);
                if (ImPlot::BeginPlot("##Performance", {-1, plot_h})) {
                    ImPlot::SetupAxes("", y_label,
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    // Force Y-axis to include peak so the red line is always visible
                    if (peak > 0) {
                        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, peak * 1.08, ImPlotCond_Always);
                    }
                    ImPlot::SetupAxisTicks(ImAxis_X1, 0,
                        (double)(sorted_labels.size() - 1),
                        (int)sorted_labels.size(), sorted_labels.data());

                    std::vector<double> positions(sorted_values.size());
                    for (size_t i = 0; i < positions.size(); i++) positions[i] = (double)i;

                    ImPlot::PlotBars("Performance", positions.data(),
                        sorted_values.data(), (int)sorted_values.size(), 0.6);

                    // Theoretical peak line
                    if (peak > 0) {
                        double pk_xs[2] = {-0.5, (double)sorted_values.size() - 0.5};
                        double pk_ys[2] = {peak, peak};
                        ImPlot::SetNextLineStyle({1.0f, 0.3f, 0.3f, 0.9f}, 2.0f);
                        ImPlot::PlotLine("Theoretical Peak", pk_xs, pk_ys, 2);

                        // Show % of peak above each bar
                        for (size_t i = 0; i < sorted_values.size(); i++) {
                            char pct[16];
                            snprintf(pct, sizeof(pct), "%.0f%%", sorted_values[i] / peak * 100.0);
                            ImPlot::PlotText(pct, positions[i], sorted_values[i], {0, -8});
                        }
                    }

                    ImPlot::EndPlot();
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
            }
        }
        ImGui::EndTabItem();
    }

    // Only worth a tab once something has been tuned; an empty tab invites a
    // click that leads nowhere.
    {
        auto th = tuning_history_.find(current_category_);
        const bool has_tuning = th != tuning_history_.end() && !th->second.empty();
        if (has_tuning && ImGui::BeginTabItem("Tuning")) {
            ImGui::Spacing();
            render_tuning_section();
            ImGui::EndTabItem();
        }
    }

    if (ImGui::BeginTabItem("Scaling")) {
        ImGui::Spacing();
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

                    const char* metric_names[] = {"Performance", "Op Time", "GPU Time"};
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
                        case ScalingMetric::OpTime:
                            y_label = "Op Time (ms)"; break;
                        case ScalingMetric::GpuTime:
                            y_label = "GPU Time (ms)"; break;
                    }

                    float plot_h = plot_height(0.86f, 380);
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
                                    case ScalingMetric::OpTime:
                                        ys.push_back((double)entry.result.op_ms);
                                        break;
                                    case ScalingMetric::GpuTime:
                                        ys.push_back((double)entry.result.gpu_ms);
                                        break;
                                }
                            }
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5 * s);
                            ImPlot::PlotLine(name.c_str(), xs.data(), ys.data(), (int)xs.size());
                        }

                        ImPlot::EndPlot();
                    }
                }
            }
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Profiling")) {
        ImGui::Spacing();
        // ================================================================
        // Profiling Comparison (all kernels side-by-side)
        // ================================================================
        {
            std::vector<const char*> prof_labels;
            std::vector<double> occupancy, ipc_vals;

            for (const auto& k : *kernels) {
                if (k.has_run && k.result.success && k.result.counters.occupancy > 0) {
                    prof_labels.push_back(k.result.kernel_name.c_str());
                    occupancy.push_back(k.result.counters.occupancy * 100.0);
                    ipc_vals.push_back(k.result.counters.ipc);
                }
            }

            if (!prof_labels.empty()) {
                if (ImGui::CollapsingHeader("Profiling Comparison")) {
                    int pn = (int)prof_labels.size();
                    std::vector<double> positions(pn);
                    for (int i = 0; i < pn; i++) positions[i] = (double)i;

                    float plot_h = plot_height(0.62f, 340);
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
        // Roofline Plot (Feature 3)
        // ================================================================
        {
            struct RoofPoint { double ai; double gflops; std::string name; DSLType dsl; };
            std::vector<RoofPoint> points;
            double worst_over_roof = 0.0;
            for (const auto& k : *kernels) {
                if (k.has_run && k.result.success && k.result.gflops > 0 && k.result.bandwidth_gbps > 0) {
                    // Only use measured DRAM bandwidth (requires profiling)
                    double actual_dram = k.result.counters.dram_read_gbps + k.result.counters.dram_write_gbps;
                    if (actual_dram <= 0) continue;  // skip kernels without profiling data
                    if (k.result.gpu_ms <= 0.0f) continue;

                    // Both axes have to come off the same clock. gflops is per
                    // wall-clock op_ms and the counters are per gpu_ms, so
                    // dividing one by the other folded the host overhead into
                    // the arithmetic intensity, which is a property of the
                    // algorithm and should not move with launch latency.
                    const double flops_total =
                        k.result.gflops * 1e9 * (k.result.op_ms / 1000.0);
                    const double bytes_total =
                        actual_dram * 1e9 * (k.result.gpu_ms / 1000.0);
                    if (bytes_total <= 0.0) continue;

                    const double ai = flops_total / bytes_total;
                    const double gflops_gpu =
                        flops_total / (k.result.gpu_ms / 1000.0) / 1e9;

                    const double roof = std::min((double)peak_fp32_gflops_,
                                                 ai * (double)peak_mem_bw_gbs_);
                    if (roof > 0.0) worst_over_roof =
                        std::max(worst_over_roof, gflops_gpu / roof);

                    points.push_back({ai, gflops_gpu, k.result.kernel_name,
                                      detect_dsl_type(k.descriptor)});
                }
            }

            if (points.empty() && peak_fp32_gflops_ > 0) {
                if (ImGui::CollapsingHeader("Roofline Model")) {
                    ImGui::TextColored(UITheme::TEXT_DIM,
                        "Enable 'Profile' and re-run to see the roofline (requires DRAM counters).");
                }
            }
            if (!points.empty() && peak_fp32_gflops_ > 0 && peak_mem_bw_gbs_ > 0) {
                if (ImGui::CollapsingHeader("Roofline Model")) {
                    ImGui::TextColored(UITheme::TEXT_DIM,
                        "X = FLOP/Byte (higher = more compute-intensive). "
                        "Y = GFLOP/s over GPU time. Gray lines = GPU limits.");

                    // A point above the roof is not a faster-than-physics
                    // result: it means DRAM was not the limit, because the
                    // working set fit in L2 and the reads never reached DRAM.
                    if (worst_over_roof > 1.05) {
                        const size_t l2 = runner_.context().l2_cache_bytes();
                        size_t working = 0;
                        for (const auto& k : *kernels) {
                            if (k.has_run && k.result.success)
                                working = std::max(working, k.result.peak_device_bytes);
                        }
                        ImGui::TextColored(UITheme::WARN_YELLOW,
                            "Points sit up to %.1fx above the memory roof.", worst_over_roof);
                        if (l2 > 0 && working > 0 && working <= l2) {
                            ImGui::TextColored(UITheme::TEXT_DIM,
                                "The working set (%.0f MB) fits in L2 (%.0f MB), so reads are "
                                "served from cache and DRAM is not the binding limit. Raise the "
                                "problem size past L2 for a roofline that bounds these kernels.",
                                working / 1e6, l2 / 1e6);
                        } else {
                            ImGui::TextColored(UITheme::TEXT_DIM,
                                "Working set %.0f MB against %.0f MB of L2. Partial cache "
                                "residency puts the effective roof above the DRAM one.",
                                working / 1e6, l2 / 1e6);
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Points below the memory roof are memory-bound.\n"
                            "Points below the compute roof are compute-bound.\n"
                            "If all points share the same X, enable 'Profile' to\n"
                            "use measured DRAM bandwidth (spreads points).");
                    }
                    float plot_h = plot_height(0.72f, 380);
                    if (ImPlot::BeginPlot("##Roofline", {-1, plot_h})) {
                        ImPlot::SetupAxes("Arithmetic Intensity (FLOP/Byte)", "Performance (GFLOP/s)");
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                        ImPlot::SetupAxesLimits(0.1, 200, 1, peak_fp32_gflops_ * 1.2, ImPlotCond_Once);

                        // Memory roof line
                        double ridge = peak_fp32_gflops_ / peak_mem_bw_gbs_;
                        double mem_xs[] = {0.1, ridge};
                        double mem_ys[] = {peak_mem_bw_gbs_ * 0.1, peak_fp32_gflops_};
                        ImPlot::SetNextLineStyle({0.6f, 0.6f, 0.6f, 0.7f}, 2.0f);
                        ImPlot::PlotLine("Mem BW Roof", mem_xs, mem_ys, 2);

                        // Compute roof line
                        double comp_xs[] = {ridge, 200.0};
                        double comp_ys[] = {peak_fp32_gflops_, peak_fp32_gflops_};
                        ImPlot::SetNextLineStyle({0.6f, 0.6f, 0.6f, 0.7f}, 2.0f);
                        ImPlot::PlotLine("Compute Roof", comp_xs, comp_ys, 2);

                        // Plot each kernel as a labeled point colored by DSL
                        for (const auto& p : points) {
                            ImVec4 col;
                            switch (p.dsl) {
                                case DSLType::CUDA:   col = UITheme::CUDA_BADGE; break;
                                case DSLType::Triton: col = UITheme::TRITON_BADGE; break;
                                case DSLType::CuTile: col = UITheme::CUTILE_BADGE; break;
                                case DSLType::Warp:   col = UITheme::WARP_BADGE; break;
                                case DSLType::CUB:    col = UITheme::CUB_BADGE; break;
                            }
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7 * s, col, 1.5f, col);
                            ImPlot::PlotScatter(p.name.c_str(), &p.ai, &p.gflops, 1);
                        }

                        ImPlot::EndPlot();
                    }
                }
                ImGui::Spacing();
            }
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Per-Kernel")) {
        ImGui::Spacing();
        if (!sel) {
            ImGui::TextColored(UITheme::TEXT_DIM,
                "Select a kernel in the list to see its run detail.");
        }
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
            double median_us = (double)sel->result.op_ms * 1000.0;

            float plot_h = plot_height(0.60f, 320);
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
        // Sub-Kernel Timeline (Feature 4)
        // ================================================================
        if (sel && sel->has_run && sel->result.success && !sel->result.sub_kernels.empty()) {
            const auto& sks = sel->result.sub_kernels;
            int n = (int)sks.size();

            char tl_header[128];
            snprintf(tl_header, sizeof(tl_header),
                "Sub-Kernel Timeline  (%d kernels, %.3f ms total)###SubKTL",
                n, sel->result.gpu_ms);
            if (ImGui::CollapsingHeader(tl_header, ImGuiTreeNodeFlags_DefaultOpen)) {

                // Summary line
                ImGui::TextColored(UITheme::TEXT_DIM,
                    "GPU kernel breakdown for %s  (Activity API)",
                    sel->result.kernel_name.c_str());
                ImGui::Spacing();

                // Compute total for percentage bars
                double total_ms = 0;
                for (const auto& sk : sks) total_ms += sk.duration_ms;

                // Gantt-style rows drawn manually for clarity
                float row_h = 32 * s;
                float bar_pad = 6 * s;
                float label_w = 220 * s;
                float avail_w = ImGui::GetContentRegionAvail().x - label_w - 100 * s;
                if (avail_w < 80 * s) avail_w = 80 * s;

                for (int i = 0; i < n; i++) {
                    const auto& sk = sks[i];
                    float pct = total_ms > 0 ? (float)(sk.duration_ms / total_ms) : 0;
                    float bar_w = avail_w * pct;
                    if (bar_w < 2 * s) bar_w = 2 * s;

                    ImGui::PushID(i);

                    // Row background
                    ImVec2 row_pos = ImGui::GetCursorScreenPos();
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    ImU32 row_bg = (i % 2 == 0) ? IM_COL32(18, 18, 18, 255)
                                                 : IM_COL32(24, 24, 24, 255);
                    dl->AddRectFilled(row_pos,
                        {row_pos.x + ImGui::GetContentRegionAvail().x, row_pos.y + row_h},
                        row_bg, 4.0f * s);

                    // Kernel name (truncated, full name on hover)
                    ImGui::SetCursorScreenPos({row_pos.x + 4 * s, row_pos.y + bar_pad});
                    std::string display_name = sk.name;
                    if (display_name.length() > 30)
                        display_name = "..." + display_name.substr(display_name.length() - 27);
                    ImGui::TextColored(UITheme::BODY_TEXT, "%s", display_name.c_str());
                    if (ImGui::IsItemHovered() && sk.name.length() > 30) {
                        ImGui::SetTooltip("%s", sk.name.c_str());
                    }

                    // Horizontal bar
                    float bar_x = row_pos.x + label_w;
                    float bar_y = row_pos.y + bar_pad;
                    float bar_h = row_h - bar_pad * 2;

                    // Bar background track
                    dl->AddRectFilled({bar_x, bar_y}, {bar_x + avail_w, bar_y + bar_h},
                        IM_COL32(40, 40, 40, 255), 4.0f * s);

                    // Filled bar
                    ImU32 bar_col = ImGui::ColorConvertFloat4ToU32(UITheme::ACCENT);
                    dl->AddRectFilled({bar_x, bar_y}, {bar_x + bar_w, bar_y + bar_h},
                        bar_col, 4.0f * s);

                    // Duration + percentage text to the right of the bar
                    char info[64];
                    snprintf(info, sizeof(info), "%.3f ms  (%.0f%%)", sk.duration_ms, pct * 100);
                    float info_x = bar_x + avail_w + 8 * s;
                    dl->AddText({info_x, bar_y + (bar_h - ImGui::GetTextLineHeight()) * 0.5f},
                        ImGui::ColorConvertFloat4ToU32(UITheme::BODY_TEXT), info);

                    // Detail on hover
                    ImGui::SetCursorScreenPos(row_pos);
                    ImGui::InvisibleButton("##row", {ImGui::GetContentRegionAvail().x, row_h});
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s\n  Duration: %.3f ms (%.1f%%)\n  Registers: %d\n  Shared mem: %d B",
                            sk.name.c_str(), sk.duration_ms, pct * 100, sk.registers, sk.shared_memory);
                    }

                    ImGui::PopID();
                }

                // Total line
                ImGui::Spacing();
                ImGui::TextColored(UITheme::HEADER_TEXT, "Total GPU time: %.3f ms across %d kernel(s)",
                    total_ms, n);
            }
            ImGui::Spacing();
        }
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
}


// ============================================================================
// Tuning results  one row per kernel, expandable to every config tried
// ============================================================================
void Gui::render_tuning_section() {
    float s = ui_scale_;

    auto cat_it = tuning_history_.find(current_category_);
    if (cat_it == tuning_history_.end() || cat_it->second.empty()) return;


    ImGui::TextColored(UITheme::TEXT_DIM,
        "Best and worst config per kernel at the current problem size.");

    // Tuning is only half useful if the answer cannot be used, so applying it
    // lives right next to the numbers that justify it.
    ImGui::BeginDisabled(benchmark_running_);
    if (ImGui::Button("Apply Best Configs")) {
        apply_best_configs();
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Pin each kernel's fastest config.\n"
            "Run Selected and Run Sweep use it from then on.");
    }

    bool any_pinned = false;
    if (auto* ks = current_kernels()) {
        for (const auto& k : *ks) if (k.has_pinned) { any_pinned = true; break; }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!any_pinned || benchmark_running_);
    if (ImGui::Button("Clear Pinned")) {
        clear_pinned_configs();
    }
    ImGui::EndDisabled();
    if (any_pinned) {
        ImGui::SameLine();
        ImGui::TextColored(UITheme::ACCENT, "pinned configs active");
    }
    ImGui::Spacing();

    const ImGuiTableFlags flags = ImGuiTableFlags_Borders |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;

    if (ImGui::BeginTable("##tuning", 5, flags)) {
        ImGui::TableSetupColumn("Kernel", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableSetupColumn("Best config", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("Best", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("Worst", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("Spread", ImGuiTableColumnFlags_WidthStretch, 0.7f);
        ImGui::TableHeadersRow();

        for (const auto& [name, entries] : cat_it->second) {
            if (entries.empty()) continue;

            const TunedResult* best = &entries[0];
            const TunedResult* worst = &entries[0];
            for (const auto& e : entries) {
                if (e.result.op_ms < best->result.op_ms)  best = &e;
                if (e.result.op_ms > worst->result.op_ms) worst = &e;
            }
            const double spread = best->result.op_ms > 0.0
                ? worst->result.op_ms / best->result.op_ms : 1.0;

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);

            // The tree opens the full list of configs for this kernel; the
            // summary stays on the same row so the table still scans.
            const bool open = ImGui::TreeNodeEx(name.c_str(),
                ImGuiTreeNodeFlags_SpanAvailWidth);

            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(UITheme::ACCENT, "%s", best->label.c_str());
            bool pinned_here = false;
            if (auto* ks = current_kernels()) {
                for (const auto& k : *ks) {
                    if (k.descriptor && k.descriptor->name() == name && k.has_pinned) {
                        pinned_here = true;
                        break;
                    }
                }
            }
            if (pinned_here) {
                ImGui::SameLine();
                ImGui::TextColored(UITheme::CUDA_BADGE, "[pinned]");
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.4f ms", best->result.op_ms);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.4f ms", worst->result.op_ms);
            ImGui::TableSetColumnIndex(4);
            // A flat kernel is the useful negative result: it says the axis is
            // not what limits this one.
            ImGui::TextColored(spread >= 1.5 ? UITheme::WARN_YELLOW : UITheme::TEXT_DIM,
                "%.2fx", spread);

            if (open) {
                auto sorted = entries;
                std::sort(sorted.begin(), sorted.end(),
                    [](const TunedResult& a, const TunedResult& b) {
                        return a.result.op_ms < b.result.op_ms;
                    });
                const float top = sorted.front().result.op_ms;
                for (const auto& e : sorted) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextColored(&e == &sorted.front() ? UITheme::ACCENT
                                                             : UITheme::TEXT_DIM,
                        "%s", e.label.c_str());
                    if (!e.result.verified) {
                        ImGui::SameLine(0, 6);
                        ImGui::TextColored(UITheme::ERROR_RED, "unverified");
                    }
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.4f ms", e.result.op_ms);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::TextColored(UITheme::TEXT_DIM, "%.1f %s",
                        is_matmul() ? e.result.gflops : e.result.bandwidth_gbps,
                        is_matmul() ? "GFLOPS" : "GB/s");
                    // What this config costs against the best one, which is
                    // the number that says whether the choice matters.
                    ImGui::TableSetColumnIndex(4);
                    if (top > 0.0f) {
                        const double rel = e.result.op_ms / top;
                        ImGui::TextColored(rel <= 1.02 ? UITheme::ACCENT : UITheme::TEXT_DIM,
                            "%.2fx", rel);
                    }
                }
                ImGui::TreePop();
            }
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
}


// ============================================================================
// Run rows and headers
// ============================================================================

// Rows for one run. Every run on screen is a recorded one, so the source is
// always a snapshot: what a run measured cannot change after the fact.
std::vector<TableRow> Gui::rows_for(const RunSnapshot* snap) {
    std::vector<TableRow> rows;
    auto* kernels = current_kernels();
    if (!kernels || !snap) return rows;

    for (const auto& [name, res] : snap->results) {
        TableRow r;
        for (auto& k : *kernels) {
            if (k.descriptor && k.descriptor->name() == name) {
                r.descriptor = k.descriptor;
                r.live = &k;          // for actions on the kernel, not the run
                r.pinned = k.pinned;
                break;
            }
        }
        if (!r.descriptor) continue;   // kernel no longer registered
        r.result = res;
        auto cit = snap->configs.find(name);
        r.result_config = cit != snap->configs.end() ? cit->second : "default";
        // Whether this run used a non-default config, which is a fact about
        // the run. The kernel's current pin is separate and may have moved on.
        r.has_pinned = (r.result_config != "default");
        rows.push_back(std::move(r));
    }
    return rows;
}

// One run's header strip: expand arrow, name, what it measured, and the
// controls. Everything that acts on a run sits on the run itself rather than
// in a selector somewhere else, so renaming is a click into the field.
int Gui::render_run_header(RunSnapshot* snap, std::vector<TableRow>& rows) {
    float s = ui_scale_;

    if (!snap) return 0;

    bool& open = snap->expanded;
    const int id = snap->id;
    const bool is_cmp = (snap->id == compare_snapshot_id_);

    ImGui::PushID(id);

    if (ImGui::ArrowButton("##toggle", open ? ImGuiDir_Down : ImGuiDir_Right)) {
        open = !open;
    }
    ImGui::SameLine();

    ImGui::SetNextItemWidth(210 * s);
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", snap->name.c_str());
    if (ImGui::InputText("##name", buf, sizeof(buf))) snap->name = buf;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click to rename this run");

    ImGui::SameLine();
    ImGui::TextColored(UITheme::TEXT_DIM, "%zu kernels | %s | %s",
                       rows.size(), snap->summary.c_str(), snap->taken_at.c_str());

    // Right-aligned controls, so they line up down the column of runs.
    const float ctl_w = 190 * s;
    const float right = ImGui::GetContentRegionMax().x - ctl_w;
    if (right > 320 * s) ImGui::SameLine(right);
    else                 ImGui::SameLine();

    bool cmp = is_cmp;
    if (ImGui::Checkbox("compare", &cmp)) {
        compare_snapshot_id_ = cmp ? snap->id : 0;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Every other run's table gets a column measured "
                          "against this one");
    }
    ImGui::SameLine();
    int to_delete = 0;
    if (ImGui::SmallButton("Delete")) to_delete = snap->id;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Deletes this run everywhere, including its points "
                          "on the Scaling chart");
    }

    ImGui::PopID();
    return to_delete;
}

// ============================================================================
// Runs  the live results and every recorded run, stacked
// ============================================================================
void Gui::render_runs_tab() {
    float s = ui_scale_;
    auto* kernels = current_kernels();
    if (!kernels) return;

    // Kernel to measure every row against, chosen once and applied to every
    // table below, so the runs stay directly comparable to each other.
    {
        auto live_rows = rows_for(nullptr);
        std::vector<std::string> owned{"(none)"};
        for (const auto& r : live_rows) {
            if (r.result.success) owned.push_back(r.descriptor->name());
        }
        for (const auto& snap : run_history_) {
            if (snap.category != current_category_) continue;
            for (const auto& [name, res] : snap.results) {
                if (std::find(owned.begin(), owned.end(), name) == owned.end())
                    owned.push_back(name);
            }
        }
        std::vector<const char*> names;
        names.reserve(owned.size());
        for (const auto& n : owned) names.push_back(n.c_str());

        std::string& baseline = ui_state_.baseline_kernel[current_category_];
        int cur = 0;
        for (int i = 1; i < (int)owned.size(); i++) {
            if (baseline == owned[i]) { cur = i; break; }
        }
        if (cur == 0) baseline.clear();

        ImGui::TextColored(UITheme::TEXT_DIM, "Compare kernels against:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(220 * s);
        if (ImGui::Combo("##baseline", &cur, names.data(), (int)names.size())) {
            baseline = (cur == 0) ? std::string() : owned[cur];
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fills the \"vs base\" column in every table below.\n"
                              "Right-click a row to set it.");
        }

        if (!run_history_.empty()) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Expand all")) {
                for (auto& x : run_history_) x.expanded = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Collapse all")) {
                for (auto& x : run_history_) x.expanded = false;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear history")) {
                // Through the same path, so nothing derived is left behind.
                std::vector<int> ids;
                for (const auto& x : run_history_) ids.push_back(x.id);
                for (int id : ids) forget_run(id);
                log(LogEntry::INFO, "Run history cleared");
            }
        }
    }

    ImGui::Separator();
    ImGui::Spacing();

    // Newest first: the latest run is the one you just produced, and it opens
    // expanded, so it reads as "the current results" without being a special
    // case in the code.
    bool any = false;
    int  to_delete = 0;
    for (int i = (int)run_history_.size() - 1; i >= 0; i--) {
        auto& snap = run_history_[i];
        if (snap.category != current_category_) continue;

        any = true;
        const int id = snap.id;
        auto rows = rows_for(&snap);
        if (int d = render_run_header(&snap, rows)) to_delete = d;

        if (snap.expanded) {
            char tid[32];
            snprintf(tid, sizeof(tid), "run%d", id);
            // Resolved per run: nothing here may cache a pointer into
            // run_history_ across a frame in which it can be deleted.
            const RunSnapshot* prev = compare_snapshot();
            // A run is never compared against itself.
            render_results_table(rows, tid, (prev && prev->id == id) ? nullptr : prev);
        }
        ImGui::Spacing();
    }

    // Deleted only once the loop is done with the vector.
    if (to_delete) forget_run(to_delete);

    if (!any) {
        ImGui::TextColored(UITheme::TEXT_DIM,
            "No runs yet. Select kernels on the left and press Run Selected; "
            "every run is recorded here.");
    }
}

// ============================================================================
// Results Table  sortable overview of all kernels in current category
// ============================================================================
void Gui::render_results_table(const std::vector<TableRow>& rows,
                              const char* table_id, const RunSnapshot* prev) {
    float s = ui_scale_;
    if (rows.empty()) {
        ImGui::TextColored(UITheme::TEXT_DIM, "  (no results in this run)");
        return;
    }

    // The comparison target is chosen once for the whole tab, so every run's
    // table is measured against the same thing.
    std::string& baseline = ui_state_.baseline_kernel[current_category_];

    // The baseline's own time, which every ratio is formed against.
    float baseline_ms = 0.0f;
    for (const auto& r : rows) {
        if (r.result.success && r.descriptor->name() == baseline) {
            baseline_ms = r.result.op_ms;
            break;
        }
    }

    bool show_gflops = is_matmul();
    bool has_profiling = false;
    for (const auto& r : rows) {
        if (r.result.counters.regs_per_thread > 0) { has_profiling = true; break; }
    }

    enum ColumnID { Col_Kernel = 0, Col_Config, Col_VsRun, Col_VsBaseline,
                    Col_Block, Col_Grid,
                    Col_Op, Col_GPU,
                    Col_Overhead, Col_Launches, Col_Perf, Col_PeakMem, Col_Energy,
                    Col_Dtype, Col_Error, Col_Status,
                    Col_Regs, Col_SHMem, Col_Occup, Col_IPC };

    int num_cols = has_profiling ? 20 : 17;

    float table_h = std::min(ImGui::GetContentRegionAvail().y, 300 * s);
    if (table_h < 100 * s) table_h = 100 * s;

    ImGui::PushID(table_id);
    if (ImGui::BeginTable("results", num_cols,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
            ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate |
            ImGuiTableFlags_Hideable | ImGuiTableFlags_Reorderable,
            {0, table_h})) {

        // Kernel is fixed rather than stretched. As the only stretch column it
        // absorbed all the slack, leaving the numeric columns cramped enough to
        // ellipsize their headers.
        ImGui::TableSetupColumn("Kernel",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort, 200 * s, Col_Kernel);
        ImGui::TableSetupColumn("Config",
            ImGuiTableColumnFlags_WidthFixed, 150 * s, Col_Config);
        // The name goes in the header so a stack of tables says what each is
        // measured against without hovering. The ### keeps the column's own id
        // stable, or changing the label would reset its width and sort.
        char vsrun_hdr[128];
        snprintf(vsrun_hdr, sizeof(vsrun_hdr), "vs %s###vsrun",
                 prev ? prev->name.c_str() : "run");
        ImGui::TableSetupColumn(vsrun_hdr,
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            prev ? 130 * s : 78 * s, Col_VsRun);
        char vsbase_hdr[128];
        snprintf(vsbase_hdr, sizeof(vsbase_hdr), "vs %s###vsbase",
                 baseline.empty() ? "base" : baseline.c_str());
        ImGui::TableSetupColumn(vsbase_hdr,
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending,
            baseline.empty() ? 92 * s : 130 * s, Col_VsBaseline);
        ImGui::TableSetupColumn("Block",  ImGuiTableColumnFlags_WidthFixed, 70 * s, Col_Block);
        ImGui::TableSetupColumn("Grid",   ImGuiTableColumnFlags_WidthFixed, 80 * s, Col_Grid);
        ImGui::TableSetupColumn("Op (ms)",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 65 * s, Col_Op);
        ImGui::TableSetupColumn("GPU (ms)",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 72 * s, Col_GPU);
        ImGui::TableSetupColumn("Overhead",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 76 * s, Col_Overhead);
        ImGui::TableSetupColumn("Lch",
            ImGuiTableColumnFlags_WidthFixed, 38 * s, Col_Launches);
        ImGui::TableSetupColumn(show_gflops ? "GFLOPS" : "GB/s",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 70 * s, Col_Perf);
        ImGui::TableSetupColumn("Peak Mem",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 80 * s, Col_PeakMem);
        ImGui::TableSetupColumn("mJ/op",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 65 * s, Col_Energy);
        ImGui::TableSetupColumn("Type",
            ImGuiTableColumnFlags_WidthFixed, 82 * s, Col_Dtype);
        ImGui::TableSetupColumn("Rel err",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 72 * s, Col_Error);
        ImGui::TableSetupColumn("Status",
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, 58 * s, Col_Status);

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
        for (int i = 0; i < (int)rows.size(); i++) sorted_indices.push_back(i);

        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty) sort_specs->SpecsDirty = false;
            if (sort_specs->SpecsCount > 0) {
                const auto& spec = sort_specs->Specs[0];
                bool asc = (spec.SortDirection == ImGuiSortDirection_Ascending);
                std::sort(sorted_indices.begin(), sorted_indices.end(),
                    [&](int a, int b) {
                        const auto& ra = rows[a].result;
                        const auto& rb = rows[b].result;
                        int cmp = 0;
                        switch (spec.ColumnUserID) {
                            case Col_Kernel: cmp = ra.kernel_name.compare(rb.kernel_name); break;
                            case Col_Config:
                                cmp = rows[a].result_config.compare(rows[b].result_config);
                                break;
                            case Col_VsBaseline: {
                                // Ratio against the baseline, so sorting this
                                // column ranks by relative speed.
                                auto rel = [&](const arena::RunResult& r) {
                                    if (baseline_ms <= 0.0f || r.op_ms <= 0.0f) return 1.0;
                                    return (double)baseline_ms / r.op_ms;
                                };
                                const double va = rel(ra), vb = rel(rb);
                                cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0; break;
                            }
                            case Col_VsRun: {
                                auto gain = [&](int i) {
                                    const auto& ks = rows[i];
                                    if (!prev || ks.result.op_ms <= 0.0f) return 1.0;
                                    auto it = prev->results.find(ks.descriptor->name());
                                    if (it == prev->results.end()) return 1.0;
                                    return (double)it->second.op_ms / ks.result.op_ms;
                                };
                                const double ga = gain(a), gb = gain(b);
                                cmp = (ga < gb) ? -1 : (ga > gb) ? 1 : 0; break;
                            }
                            case Col_Block:  cmp = (int)(ra.block_x * ra.block_y) - (int)(rb.block_x * rb.block_y); break;
                            case Col_Grid:   cmp = (int)(ra.grid_x * ra.grid_y) - (int)(rb.grid_x * rb.grid_y); break;
                            case Col_Op:   cmp = (ra.op_ms < rb.op_ms) ? -1 : (ra.op_ms > rb.op_ms) ? 1 : 0; break;
                            case Col_GPU:    cmp = (ra.gpu_ms < rb.gpu_ms) ? -1 : (ra.gpu_ms > rb.gpu_ms) ? 1 : 0; break;
                            case Col_Perf: {
                                double va = show_gflops ? ra.gflops : ra.bandwidth_gbps;
                                double vb = show_gflops ? rb.gflops : rb.bandwidth_gbps;
                                cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0; break;
                            }
                            case Col_Overhead: cmp = (ra.overhead_ms < rb.overhead_ms) ? -1 : (ra.overhead_ms > rb.overhead_ms) ? 1 : 0; break;
                            case Col_Launches: cmp = ra.launch_count - rb.launch_count; break;
                            case Col_PeakMem:  cmp = (ra.peak_device_bytes < rb.peak_device_bytes) ? -1 : (ra.peak_device_bytes > rb.peak_device_bytes) ? 1 : 0; break;
                            case Col_Energy:   cmp = (ra.energy.mj_per_op < rb.energy.mj_per_op) ? -1 : (ra.energy.mj_per_op > rb.energy.mj_per_op) ? 1 : 0; break;
                            case Col_Dtype: cmp = ra.input_dtype.compare(rb.input_dtype); break;
                            case Col_Error: cmp = (ra.accuracy.max_total_error < rb.accuracy.max_total_error) ? -1 : (ra.accuracy.max_total_error > rb.accuracy.max_total_error) ? 1 : 0; break;
                            case Col_Regs:  cmp = ra.counters.regs_per_thread - rb.counters.regs_per_thread; break;
                            case Col_SHMem: cmp = ra.counters.shared_mem_bytes - rb.counters.shared_mem_bytes; break;
                            case Col_Occup: cmp = (ra.counters.occupancy < rb.counters.occupancy) ? -1 : 1; break;
                            case Col_IPC:   cmp = (ra.counters.ipc < rb.counters.ipc) ? -1 : (ra.counters.ipc > rb.counters.ipc) ? 1 : 0; break;
                            default: break;
                        }
                        return asc ? (cmp < 0) : (cmp > 0);
                    });
            }
        }

        for (int idx : sorted_indices) {
            auto& k = rows[idx];
            ImGui::TableNextRow();

            // Tint the whole row when something is off, so a bad result is
            // visible without reading the Status column. Kept low-alpha: it
            // has to sit under the text without fighting it.
            if (!k.result.success) {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                    ImGui::GetColorU32(ImVec4(UITheme::ERROR_RED.x, UITheme::ERROR_RED.y,
                                              UITheme::ERROR_RED.z, 0.16f)));
            } else if (!k.result.verified) {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                    ImGui::GetColorU32(ImVec4(UITheme::WARN_YELLOW.x, UITheme::WARN_YELLOW.y,
                                              UITheme::WARN_YELLOW.z, 0.14f)));
            }

            // Kernel name with DSL color tag + click-to-select
            ImGui::TableNextColumn();

            // Thin dtype stripe on the left edge of the row. Cheaper to scan
            // than reading the Type column on every line.
            {
                const ImVec2 cp = ImGui::GetCursorScreenPos();
                const float  lh = ImGui::GetTextLineHeight();
                ImGui::GetWindowDrawList()->AddRectFilled(
                    ImVec2(cp.x - 5.0f * s, cp.y),
                    ImVec2(cp.x - 2.0f * s, cp.y + lh),
                    ImGui::GetColorU32(dtype_color(
                        k.result.input_dtype + ">" + k.result.output_dtype)),
                    1.0f * s);
            }
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
            // Right-click is the fast path: comparing usually starts from a
            // row you are already looking at, not from the combo above.
            if (ImGui::BeginPopupContextItem("##rowctx")) {
                ImGui::TextDisabled("%s", k.descriptor->name().c_str());
                ImGui::Separator();
                const bool is_base = (k.descriptor->name() == baseline);
                if (ImGui::MenuItem("Set as comparison baseline", nullptr, is_base)) {
                    baseline = is_base ? std::string() : k.descriptor->name();
                }
                if (k.live && k.has_pinned && ImGui::MenuItem("Unpin tuned config")) {
                    k.live->has_pinned = false;
                    k.live->pinned = {};
                }
                ImGui::EndPopup();
            }

            if (k.result.success && !k.result.warmup_converged) {
                ImGui::SameLine(0, 4);
                ImGui::TextColored(UITheme::WARN_YELLOW, "~");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Warmup did not converge; clock was still drifting");
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

            // Config this row was measured at, and how it compares to the
            // kernel's own default. Without these two the table cannot say
            // whether a number came from tuning or from the source file.
            ImGui::TableNextColumn();
            {
                bool pinned = k.has_pinned;
                ImGui::TextColored(pinned ? UITheme::ACCENT : UITheme::TEXT_DIM,
                    "%s", k.result_config.c_str());
                if (pinned) {
                    ImGui::SameLine(0, 4);
                    ImGui::TextColored(UITheme::CUDA_BADGE, "*");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Measured at: %s", k.result_config.c_str());
                    if (pinned)
                        ImGui::Text("Pinned by tuning; * marks a non-default config.");
                    else
                        ImGui::Text("The kernel's own default.");
                    ImGui::EndTooltip();
                }
            }

            ImGui::TableNextColumn();
            const arena::RunResult* prev_r = nullptr;
            if (prev) {
                auto it = prev->results.find(k.descriptor->name());
                if (it != prev->results.end()) prev_r = &it->second;
            }
            if (prev_r && k.result.op_ms > 0.0f && prev_r->op_ms > 0.0f) {
                const double gain = (double)prev_r->op_ms / k.result.op_ms;
                // Within a couple of percent is noise, not a result.
                const ImVec4 col = gain >= 1.02 ? UITheme::SUCCESS_GREEN
                                 : gain <= 0.98 ? UITheme::ERROR_RED
                                                : UITheme::TEXT_DIM;
                ImGui::TextColored(col, "%.2fx", gain);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s: %.4f ms (%s)\nnow: %.4f ms (%s)",
                        prev->name.c_str(), prev_r->op_ms,
                        prev->configs.count(k.descriptor->name())
                            ? prev->configs.at(k.descriptor->name()).c_str() : "?",
                        k.result.op_ms, k.result_config.c_str());
                }
            } else {
                ImGui::TextColored(UITheme::TEXT_DIM, "-");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(prev
                        ? "Not measured in the run being compared against."
                        : "Pick a run to compare against, above the table.");
                }
            }

            ImGui::TableNextColumn();
            if (baseline.empty()) {
                ImGui::TextColored(UITheme::TEXT_DIM, "-");
            } else if (k.descriptor->name() == baseline) {
                ImGui::TextColored(UITheme::ACCENT, "baseline");
            } else if (baseline_ms > 0.0f && k.result.op_ms > 0.0f) {
                const double r = (double)baseline_ms / k.result.op_ms;
                if (r >= 1.02) {
                    ImGui::TextColored(UITheme::SUCCESS_GREEN, "%.2fx faster", r);
                } else if (r <= 0.98) {
                    // Shown as how much slower rather than a fraction: 3.2x
                    // slower is easier to read than 0.31x.
                    ImGui::TextColored(UITheme::ERROR_RED, "%.2fx slower", 1.0 / r);
                } else {
                    ImGui::TextColored(UITheme::TEXT_DIM, "same");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s: %.4f ms\n%s: %.4f ms",
                        baseline.c_str(), baseline_ms,
                        k.result.kernel_name.c_str(), k.result.op_ms);
                }
            } else {
                ImGui::TextColored(UITheme::TEXT_DIM, "-");
            }

            ImGui::TableNextColumn(); ImGui::Text("%ux%u", k.result.block_x, k.result.block_y);
            ImGui::TableNextColumn(); ImGui::Text("%ux%u", k.result.grid_x, k.result.grid_y);

            // Op time with min/max/stddev tooltip
            ImGui::TableNextColumn(); ImGui::Text("%.3f", k.result.op_ms);
            if (ImGui::IsItemHovered() && !k.result.all_times_ms.empty()) {
                const auto& t = k.result.all_times_ms;
                float tmin = *std::min_element(t.begin(), t.end());
                float tmax = *std::max_element(t.begin(), t.end());
                float mean = std::accumulate(t.begin(), t.end(), 0.0f) / t.size();
                float var = 0;
                for (float v : t) var += (v - mean) * (v - mean);
                float stddev = std::sqrt(var / t.size());
                ImGui::SetTooltip("Min: %.3f ms\nMax: %.3f ms\nStdDev: %.3f ms\nRuns: %zu",
                    tmin, tmax, stddev, t.size());
            }

            // GPU time  highlight overhead
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", k.result.gpu_ms);
            if (ImGui::IsItemHovered() && k.result.gpu_ms > 0 &&
                k.result.op_ms > k.result.gpu_ms * 1.05f) {
                float overhead_pct = ((k.result.op_ms - k.result.gpu_ms) /
                                       k.result.op_ms) * 100.0f;
                ImGui::SetTooltip("Host overhead: %.1f%% (%.3f ms)",
                    overhead_pct, k.result.op_ms - k.result.gpu_ms);
            }

            // Overhead: launch latency and inter-kernel gaps
            ImGui::TableNextColumn();
            if (k.result.gpu_ms > 0.0f) {
                float pct = (k.result.op_ms > 0.0f)
                    ? (k.result.overhead_ms / k.result.op_ms) * 100.0f : 0.0f;
                if (pct >= 50.0f) ImGui::TextColored(UITheme::WARN_YELLOW, "%.3f", k.result.overhead_ms);
                else              ImGui::Text("%.3f", k.result.overhead_ms);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "%.1f%% of operation time.\n"
                        "Launch latency and inter-kernel gaps: the cost of\n"
                        "getting work onto the GPU, not the work itself.", pct);
                }
            } else {
                ImGui::TextColored(UITheme::TEXT_DIM, "--");
            }

            ImGui::TableNextColumn();
            if (k.result.launch_count > 0) ImGui::Text("%d", k.result.launch_count);
            else                           ImGui::TextColored(UITheme::TEXT_DIM, "--");

            ImGui::TableNextColumn();
            if (show_gflops) ImGui::Text("%.1f", k.result.gflops);
            else             ImGui::Text("%.1f", k.result.bandwidth_gbps);

            ImGui::TableNextColumn();
            if (k.result.peak_device_bytes > 0) {
                ImGui::Text("%.1f MB", k.result.peak_device_bytes / (1024.0 * 1024.0));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%zu bytes, high-water mark across this run",
                        k.result.peak_device_bytes);
            } else {
                ImGui::TextColored(UITheme::TEXT_DIM, "--");
            }

            ImGui::TableNextColumn();
            if (k.result.energy.available) {
                ImGui::Text("%.3f", k.result.energy.mj_per_op);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "%.1f W average over %d iterations.\n"
                        "Whole-board energy, so this is an upper bound on the\n"
                        "kernel's marginal cost.",
                        k.result.energy.avg_watts, k.result.energy.iterations);
                }
            } else {
                ImGui::TextColored(UITheme::TEXT_DIM, "--");
            }

            ImGui::TableNextColumn();
            const std::string tl = dtype_label(k.result.input_dtype, k.result.output_dtype);
            ImGui::TextColored(dtype_color(tl), "%s", tl.c_str());
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("in %s, out %s, compute %s\naccuracy is judged against the output type",
                    k.result.input_dtype.c_str(), k.result.output_dtype.c_str(),
                    k.result.compute_mode.c_str());
            }

            // Relative error against a double-precision CPU reference. Coloured
            // by how close it sits to the tolerance, so an imprecise kernel
            // reads differently from a wrong one.
            // Total error is shown because the table mixes dtypes and this is
            // the number comparable across them. The colour tracks that same
            // number, so a bigger figure never reads as safer than a smaller
            // one. Pass/fail is already carried by the row tint and Status.
            ImGui::TableNextColumn();
            if (k.result.accuracy.checked) {
                const auto& a = k.result.accuracy;
                if (a.max_total_error > a.tolerance)
                    ImGui::TextColored(UITheme::ERROR_RED, "%.2e", a.max_total_error);
                else if (a.max_total_error > a.tolerance * 0.1)
                    ImGui::TextColored(UITheme::WARN_YELLOW, "%.2e", a.max_total_error);
                else
                    ImGui::Text("%.2e", a.max_total_error);

                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("total      max %.3e  mean %.3e", a.max_total_error, a.mean_total_error);
                    ImGui::TextDisabled("vs the original fp32 data, comparable across dtypes");
                    ImGui::Separator();
                    ImGui::Text("arithmetic max %.3e  mean %.3e", a.max_rel_error, a.mean_rel_error);
                    ImGui::TextDisabled("vs the inputs this kernel received, judged against %.3e", a.tolerance);
                    ImGui::Separator();
                    ImGui::Text("%d elements checked", a.elements_checked);
                    ImGui::EndTooltip();
                }
            } else {
                ImGui::TextColored(UITheme::TEXT_DIM, "--");
            }

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
                if (k.result.counters.regs_per_thread > 0)
                    ImGui::Text("%d", k.result.counters.regs_per_thread);
                else ImGui::TextColored(UITheme::TEXT_DIM, "-");

                ImGui::TableNextColumn();
                if (k.result.counters.shared_mem_bytes > 0)
                    ImGui::Text("%d", k.result.counters.shared_mem_bytes);
                else ImGui::TextColored(UITheme::TEXT_DIM, "-");

                // "--" means the counter pass did not run, which is a
                // different statement from a measured zero.
                ImGui::TableNextColumn();
                if (k.result.counters.available)
                    ImGui::Text("%.1f", k.result.counters.occupancy * 100.0);
                else ImGui::TextColored(UITheme::TEXT_DIM, "--");

                ImGui::TableNextColumn();
                if (k.result.counters.available) ImGui::Text("%.2f", k.result.counters.ipc);
                else ImGui::TextColored(UITheme::TEXT_DIM, "--");
            } else {
                ImGui::TableNextColumn();
                ImGui::Text("%d", k.result.counters.regs_per_thread);
            }
        }

        ImGui::EndTable();
    }
    ImGui::PopID();
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
    bool has_counters = has_data && r.counters.available;

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

    // Compilation cache and timing (Feature 5)
    if (has_data && desc->needs_compilation()) {
        if (r.cache_hit)
            ImGui::TextColored(UITheme::SUCCESS_GREEN, "Cache: Hit (skipped recompile)");
        else
            ImGui::TextColored(UITheme::WARN_YELLOW, "Cache: Miss (freshly compiled)");

        if (r.compile_ms > 0)
            ImGui::Text("Compile time: %.0f ms", r.compile_ms);
        else
            ImGui::TextColored(UITheme::TEXT_DIM, "Compile time: <1 ms (cached)");
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Cache: N/A (native kernel)");
        ImGui::TextColored(UITheme::TEXT_DIM, "Compile time: N/A");
    }

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
    if (has_data && r.counters.regs_per_thread > 0) {
        ImGui::Text("Registers/thread: %d", r.counters.regs_per_thread);
    } else {
        ImGui::TextColored(UITheme::TEXT_DIM, "Registers/thread: --");
    }

    // Shared memory
    if (has_data && r.counters.shared_mem_bytes > 0) {
        if (r.counters.shared_mem_bytes >= 1024)
            ImGui::Text("Shared mem: %.1f KB", r.counters.shared_mem_bytes / 1024.0f);
        else
            ImGui::Text("Shared mem: %d B", r.counters.shared_mem_bytes);
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
            (float)(r.counters.occupancy * 100.0), 100.0f, "%.1f%%");

        if (r.counters.ipc > 0) {
            colored_bar("IPC", (float)r.counters.ipc, 4.0f, "%.2f");
        }

        double total_dram = r.counters.dram_read_gbps + r.counters.dram_write_gbps;
        if (total_dram > 0) {
            colored_bar("DRAM Throughput", (float)total_dram,
                peak_mem_bw_gbs_ > 0 ? peak_mem_bw_gbs_ : 1000.0f, "%.1f GB/s");
            ImGui::TextColored(UITheme::TEXT_DIM, "  R: %.1f  W: %.1f GB/s",
                r.counters.dram_read_gbps, r.counters.dram_write_gbps);
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

    // ================================================================
    // Cold start / build / measurement quality
    //
    // These are one-time or per-process costs, deliberately kept out of the
    // main table: their units (ms) are not comparable with the per-invocation
    // tier (ms/op), and nothing should sum across the two.
    // ================================================================
    if (has_data && r.success) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(UITheme::TEXT_DIM, "Cold start (per process)");
        ImGui::Text("Module load:  %.2f ms", r.module_load_ms);
        ImGui::Text("First launch: %.3f ms", r.first_launch_ms);
        if (r.op_ms > 0.0f && r.first_launch_ms > r.op_ms) {
            ImGui::SameLine();
            ImGui::TextColored(UITheme::TEXT_DIM, "(%.1fx steady state)",
                r.first_launch_ms / r.op_ms);
        }

        ImGui::Spacing();
        ImGui::TextColored(UITheme::TEXT_DIM, "Build (one time, cached)");
        if (r.cache_hit) {
            ImGui::TextColored(UITheme::TEXT_DIM, "cached");
        } else if (r.invoke_ms > 0.0f) {
            ImGui::Text("Compile: %.0f ms", r.compile_ms);
            if (r.import_ms > 0.0f) {
                ImGui::Text("Import:  %.0f ms", r.import_ms);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Python interpreter startup and module imports.\n"
                        "Not part of the DSL's compile cost.");
                }
            }
            ImGui::Text("Invoke:  %.0f ms", r.invoke_ms);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Full subprocess wall time, including spawn.");
            }
        } else {
            ImGui::TextColored(UITheme::TEXT_DIM, "n/a (no runtime compilation)");
        }

        ImGui::Spacing();
        ImGui::TextColored(UITheme::TEXT_DIM, "Measurement quality");
        ImGui::Text("Warmup: %d iterations", r.warmup_iterations);
        if (!r.warmup_converged) {
            ImGui::SameLine();
            ImGui::TextColored(UITheme::WARN_YELLOW, "(no convergence)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Timings were still drifting when warmup hit its cap.\n"
                    "This result was measured on an unsettled clock.");
            }
        }
        if (r.sm_clock_start_mhz > 0) {
            ImGui::Text("SM clock: %u -> %u MHz",
                r.sm_clock_start_mhz, r.sm_clock_end_mhz);
        }
    }

    // ================================================================
    // Source Viewer (Feature 9)
    // ================================================================
    if (desc->needs_compilation() && !desc->source_path().empty()) {
        ImGui::Spacing();
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Source Code")) {
            std::string src = read_kernel_source(desc->source_path());
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.04f, 0.04f, 0.04f, 1.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {4 * s, 4 * s});
            float viewer_h = std::min(300 * s, ImGui::GetContentRegionAvail().y - 10 * s);
            if (viewer_h < 80 * s) viewer_h = 80 * s;
            ImGui::InputTextMultiline("##src", src.data(), src.size() + 1,
                {-1, viewer_h}, ImGuiInputTextFlags_ReadOnly);
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
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
    std::vector<BenchWork> work, arena::RunConfig config) {

    CUcontext cuda_ctx = runner_.context().handle();
    cuCtxPushCurrent(cuda_ctx);

    for (int i = 0; i < (int)work.size(); i++) {
        if (cancel_requested_) break;

        auto& [cat, descriptor, variant] = work[i];
        benchmark_current_ = i;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            benchmark_current_name_ = descriptor->name();
        }

        PendingResult pr;
        pr.category = cat;
        pr.kernel_name = descriptor->name();
        pr.params = config.params;
        pr.config_label = tuning_label_for(variant);
        pr.is_default_config = (variant.block_size <= 0 && variant.defines.empty());
        pr.logs.push_back({LogEntry::INFO, "Running " + descriptor->name() +
                           " [" + pr.config_label + "] ..."});

        auto run_config = config;
        run_config.block_size      = variant.block_size;
        run_config.compile_options = variant.defines;
        pr.result = runner_.run(*descriptor, run_config);

        if (pr.result.success) {
            char buf[256];
            bool matmul = (cat == "matmul");
            snprintf(buf, sizeof(buf), "%s: op=%.3f ms  gpu=%.3f ms  %.2f %s",
                pr.result.kernel_name.c_str(),
                pr.result.op_ms, pr.result.gpu_ms,
                matmul ? pr.result.gflops : pr.result.bandwidth_gbps,
                matmul ? "GFLOPS" : "GB/s");
            pr.logs.push_back({LogEntry::INFO, buf});

            if (config.collect_metrics && pr.result.counters.occupancy > 0) {
                snprintf(buf, sizeof(buf),
                    "%s: regs=%d  shmem=%dB  occupancy=%.1f%%  IPC=%.2f",
                    pr.result.kernel_name.c_str(),
                    pr.result.counters.regs_per_thread,
                    pr.result.counters.shared_mem_bytes,
                    pr.result.counters.occupancy * 100.0,
                    pr.result.counters.ipc);
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
    std::vector<BenchWork> work,
    std::vector<std::map<std::string, int>> sweep_configs,
    arena::RunConfig config) {

    CUcontext ctx = runner_.context().handle();
    cuCtxPushCurrent(ctx);

    int i = 0;
    for (const auto& params : sweep_configs) {
        if (cancel_requested_) break;
        config.params = params;

        for (auto& [cat, descriptor, variant] : work) {
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
            pr.config_label = tuning_label_for(variant);
            pr.is_default_config = (variant.block_size <= 0 && variant.defines.empty());

            std::string size_str;
            for (auto& [k, v] : params) {
                if (!size_str.empty()) size_str += ",";
                size_str += k + "=" + std::to_string(v);
            }
            pr.logs.push_back({LogEntry::INFO,
                "Sweep " + descriptor->name() + " [" + size_str + "] ..."});

            auto run_config = config;
            run_config.block_size      = variant.block_size;
            run_config.compile_options = variant.defines;
            pr.result = runner_.run(*descriptor, run_config);

            if (pr.result.success) {
                char buf[256];
                bool matmul = (cat == "matmul");
                snprintf(buf, sizeof(buf), "%s [%s]: op=%.3f ms  %.2f %s",
                    pr.result.kernel_name.c_str(), size_str.c_str(),
                    pr.result.op_ms,
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


// A tuning run walks each kernel's own axis instead of the problem size:
// block sizes for a hand-written CUDA kernel, compile-time configs for a DSL
// one. Both come back as labelled points so the table can show them together.
void Gui::tuning_thread_func(
    std::vector<BenchWork> work, arena::RunConfig config) {

    CUcontext ctx = runner_.context().handle();
    cuCtxPushCurrent(ctx);

    int i = 0;
    for (auto& [cat, descriptor, ignored] : work) {
        if (cancel_requested_) break;

        const auto variants = arena::cli::tuning_variants_for(
            true, 0, {}, descriptor->tunable_block_sizes(),
            descriptor->tunable_compile_options());

        for (const auto& variant : variants) {
            if (cancel_requested_) break;

            benchmark_current_ = i++;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                benchmark_current_name_ = descriptor->name();
            }

            auto run_config = config;
            run_config.block_size      = variant.block_size;
            run_config.compile_options = variant.defines;

            PendingResult pr;
            pr.category = cat;
            pr.kernel_name = descriptor->name();
            pr.params = config.params;
            pr.tuning_label = tuning_label_for(variant);
            pr.tuning_variant = variant;

            pr.result = runner_.run(*descriptor, run_config);

            char buf[256];
            if (pr.result.success) {
                snprintf(buf, sizeof(buf), "%s [%s]: op=%.4f ms",
                    descriptor->name().c_str(), pr.tuning_label.c_str(),
                    pr.result.op_ms);
                pr.logs.push_back({LogEntry::INFO, buf});
            } else {
                pr.logs.push_back({LogEntry::ERR,
                    descriptor->name() + " [" + pr.tuning_label + "]: " + pr.result.error});
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending_results_.push_back(std::move(pr));
            }
        }
    }

    benchmark_current_ = benchmark_total_.load();

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

    // A pinned config, if tuning found one, is what this kernel now runs at.
    std::vector<BenchWork> work;
    for (auto& k : it->second) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor,
                            k.has_pinned ? k.pinned : arena::cli::TuningVariant{}});
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

    // A pinned config, if tuning found one, is what this kernel now runs at.
    std::vector<BenchWork> work;
    for (auto& k : it->second) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor,
                            k.has_pinned ? k.pinned : arena::cli::TuningVariant{}});
        }
    }
    if (work.empty()) return;

    auto sweep_configs = work[0].descriptor->get_sweep_configs(config_);
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

void Gui::run_tuning() {
    if (current_category_.empty()) return;
    if (benchmark_running_) return;

    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return;

    std::vector<BenchWork> work;
    int total = 0;
    for (auto& k : it->second) {
        if (!k.selected || !k.descriptor) continue;
        work.push_back({current_category_, k.descriptor, {}});
        total += (int)tuning_variants(*k.descriptor).size();
    }
    if (work.empty()) return;

    if (total == (int)work.size()) {
        log(LogEntry::WARN,
            "None of the selected kernels has a tuning axis; nothing to sweep");
        return;
    }

    if (benchmark_thread_.joinable()) benchmark_thread_.join();

    tuning_history_[current_category_].clear();

    cancel_requested_ = false;
    benchmark_running_ = true;
    benchmark_current_ = 0;
    benchmark_total_ = total;

    benchmark_thread_ = std::thread(&Gui::tuning_thread_func, this,
        std::move(work), config_);
}

// Pins each kernel's fastest measured config, so Run Selected and Run Sweep
// use it from here on. Only kernels that actually have an axis are touched:
// pinning "default" on the rest would just be noise in the UI.
void Gui::apply_best_configs() {
    auto cat_it = tuning_history_.find(current_category_);
    if (cat_it == tuning_history_.end()) return;

    auto* kernels = current_kernels();
    if (!kernels) return;

    int applied = 0;
    for (auto& k : *kernels) {
        if (!k.descriptor) continue;
        auto hist_it = cat_it->second.find(k.descriptor->name());
        if (hist_it == cat_it->second.end() || hist_it->second.size() < 2) continue;

        const TunedResult* best = &hist_it->second[0];
        for (const auto& e : hist_it->second) {
            if (e.result.op_ms < best->result.op_ms) best = &e;
        }
        k.pinned = best->variant;
        k.has_pinned = true;
        applied++;
        log(LogEntry::INFO, k.descriptor->name() + ": pinned " + best->label);
    }

    if (applied == 0) {
        log(LogEntry::WARN, "No tuned kernel to pin a config for");
    } else {
        log(LogEntry::INFO, "Pinned best config for " + std::to_string(applied) +
                            " kernel(s); later runs will use them");
    }
}

void Gui::clear_pinned_configs() {
    auto* kernels = current_kernels();
    if (!kernels) return;
    for (auto& k : *kernels) { k.has_pinned = false; k.pinned = {}; }
    log(LogEntry::INFO, "Cleared pinned configs; back to kernel defaults");
}

// Turns whatever the finished run measured into a named snapshot. Tuning runs
// are excluded: they deliberately measure configs the kernel is not normally
// launched at, so folding them into history would compare against numbers no
// ordinary run would produce.
void Gui::commit_snapshot() {
    if (pending_snapshot_.results.empty()) {
        pending_snapshot_ = RunSnapshot{};
        return;
    }

    RunSnapshot snap = std::move(pending_snapshot_);
    pending_snapshot_ = RunSnapshot{};

    snap.id = next_snapshot_id_++;

    std::string size_str;
    for (const auto& [key, val] : snap.params) {
        // Only the parameters this category actually uses; the map carries
        // defaults for all of them.
        const bool relevant =
            (snap.category == "matmul"  && (key == "M" || key == "K" || key == "N")) ||
            (snap.category == "softmax" && (key == "rows" || key == "cols")) ||
            (snap.category != "matmul" && snap.category != "softmax" && key == "n");
        if (!relevant) continue;
        if (!size_str.empty()) size_str += " ";
        size_str += key + "=" + std::to_string(val);
    }

    int tuned = 0;
    for (const auto& [name, cfg] : snap.configs) {
        if (cfg != "default") tuned++;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%s | %s | %s | %d runs%s",
        snap.category.c_str(),
        size_str.empty() ? "default size" : size_str.c_str(),
        arena::distribution_name(config_.input_distribution),
        config_.number_of_runs,
        tuned > 0 ? " | tuned" : "");
    snap.summary = buf;

    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    char stamp[16];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &tm_buf);
    snap.taken_at = stamp;

    snprintf(buf, sizeof(buf), "#%d %s%s", snap.id, stamp,
             tuned > 0 ? " (tuned)" : "");
    snap.name = buf;

    // Expanded, and the previous newest collapses: the latest run is the one
    // you want in front of you, without burying it under every earlier one.
    for (auto& x : run_history_) x.expanded = false;
    snap.expanded = true;
    run_history_.push_back(std::move(snap));

    // Bounded so a long session cannot grow without limit. The oldest goes
    // first, and the comparison selection is cleared if it was the casualty.
    if (run_history_.size() > MAX_SNAPSHOTS) {
        forget_run(run_history_.front().id);
    }

    log(LogEntry::INFO, "Recorded run " + run_history_.back().name + " (" +
        std::to_string(run_history_.back().results.size()) + " kernels)");
}

// Everything a deleted run leaves behind. A run is the unit the user deletes,
// so nothing derived from it should outlive it: the scaling chart would keep
// plotting its points and the other tabs would keep showing its numbers.
void Gui::forget_run(int run_id) {
    if (compare_snapshot_id_ == run_id) compare_snapshot_id_ = 0;

    run_history_.erase(
        std::remove_if(run_history_.begin(), run_history_.end(),
            [&](const RunSnapshot& x) { return x.id == run_id; }),
        run_history_.end());

    // Scaling points carry the run that produced them.
    for (auto& [cat, per_kernel] : scaling_history_) {
        for (auto& [name, hist] : per_kernel) {
            hist.erase(std::remove_if(hist.begin(), hist.end(),
                [&](const SizedResult& e) { return e.run_id == run_id; }), hist.end());
        }
        for (auto it = per_kernel.begin(); it != per_kernel.end(); ) {
            it = it->second.empty() ? per_kernel.erase(it) : std::next(it);
        }
    }

    rebuild_timing_history();
    sync_live_from_newest();
}

// The tabs other than Runs read the live kernel state, so it has to track the
// newest surviving run rather than whatever was measured last. Without this,
// deleting the newest run leaves its numbers on the Compare and Profiling
// tabs with nothing on screen still claiming to own them.
void Gui::sync_live_from_newest() {
    for (auto& [cat, kernels] : kernels_by_category_) {
        const RunSnapshot* newest = nullptr;
        for (const auto& snap : run_history_) {
            if (snap.category != cat) continue;
            if (!newest || snap.id > newest->id) newest = &snap;
        }

        for (auto& k : kernels) {
            if (!k.descriptor) continue;
            const auto* res = newest ? [&]() -> const arena::RunResult* {
                auto it = newest->results.find(k.descriptor->name());
                return it == newest->results.end() ? nullptr : &it->second;
            }() : nullptr;

            if (res) {
                k.result = *res;
                k.has_run = true;
                auto cit = newest->configs.find(k.descriptor->name());
                k.result_config = cit != newest->configs.end() ? cit->second : "default";
            } else {
                k.result = arena::RunResult{};
                k.has_run = false;
                k.result_config = "default";
            }
        }
    }
}

// Ring buffers cannot drop one run's samples in place, so they are rebuilt
// from the runs that remain, oldest first to keep the order they arrived in.
void Gui::rebuild_timing_history() {
    timing_history_.clear();
    for (const auto& snap : run_history_) {
        for (const auto& [name, res] : snap.results) {
            if (!res.success) continue;
            auto& ring = timing_history_[name];
            for (float t : res.all_times_ms) ring.push(t);
        }
    }
}


const RunSnapshot* Gui::compare_snapshot() const {
    if (compare_snapshot_id_ == 0) return nullptr;
    for (const auto& s : run_history_) {
        if (s.id == compare_snapshot_id_) return &s;
    }
    return nullptr;
}


void Gui::reset_results() {
    if (current_category_.empty()) return;

    // What is on screen is a view of the recorded runs, so clearing the view
    // means dropping the runs behind it. Leaving them would put the results
    // straight back on the next redraw.
    std::vector<int> ids;
    for (const auto& x : run_history_) {
        if (x.category == current_category_) ids.push_back(x.id);
    }
    for (int id : ids) forget_run(id);

    log(LogEntry::INFO, "Results reset for " + current_category_ +
        " (" + std::to_string(ids.size()) + " run(s) dropped)");
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

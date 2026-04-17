#include "frontend/tui.hpp"
#include "arena/logger.hpp"

#include <cuda.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <signal.h>
#include <fcntl.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <fstream>
#include <chrono>
#include <cctype>
#include <set>

namespace frontend {

namespace C {
    // 0 = default (terminal default). Any non-zero = true-color 0xRRGGBB.
    constexpr uint32_t ACCENT     = 0x00D4AA;  // teal-green
    constexpr uint32_t ACCENT_DIM = 0x005844;
    constexpr uint32_t CUDA_CLR   = 0x76B900;  // NVIDIA green
    constexpr uint32_t TRITON_CLR = 0xFF6B2B;
    constexpr uint32_t CUTILE_CLR = 0x5B9BD5;
    constexpr uint32_t WARP_CLR   = 0x9966CC;
    constexpr uint32_t CUB_CLR    = 0x8C8C8C;
    constexpr uint32_t RED        = 0xFF4444;
    constexpr uint32_t YELLOW     = 0xFFB300;
    constexpr uint32_t GREEN      = 0x00C853;
    constexpr uint32_t HEADER     = 0xE6E6E6;
    constexpr uint32_t BODY       = 0xC8C8C8;
    constexpr uint32_t DIM        = 0x808080;
    constexpr uint32_t BORDER     = 0x505050;
    constexpr uint32_t BORDER_DIM = 0x353535;
    constexpr uint32_t BG_ALT     = 0x1A1A1A;
    constexpr uint32_t BG_HDR     = 0x0D0D0D;
}

enum Attr : uint8_t {
    ATTR_BOLD      = 1 << 0,
    ATTR_DIM       = 1 << 1,
    ATTR_UNDERLINE = 1 << 2,
    ATTR_REVERSE   = 1 << 3
};

static Tui* g_tui_instance = nullptr;
static struct sigaction g_old_sigint{}, g_old_sigterm{}, g_old_winch{};

static void restore_terminal_atexit() {
    // safety net if we die before the destructor runs
    const char restore[] = "\x1b[0m\x1b[?25h\x1b[?1049l";
    (void)!write(STDOUT_FILENO, restore, sizeof(restore) - 1);
}

static volatile sig_atomic_t g_resize_pending = 0;
static volatile sig_atomic_t g_exit_pending   = 0;

static void tui_sig_winch(int) { g_resize_pending = 1; }
static void tui_sig_exit(int)  { g_exit_pending   = 1; }

static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

Tui::Tui(arena::Runner& runner) : runner_(runner) {
    config_.params["M"] = 1024;
    config_.params["K"] = 1024;
    config_.params["N"] = 1024;
    config_.params["n"] = 1000000;
    config_.params["rows"] = 1024;
    config_.params["cols"] = 1024;
    config_.warmup_runs = 10;
    config_.number_of_runs = 10;

    compute_peaks();
    refresh_kernels();
}

Tui::~Tui() {
    cancel_requested_ = true;
    if (benchmark_thread_.joinable()) {
        benchmark_thread_.join();
    }
    leave_alt_screen();
    leave_raw_mode();
    g_tui_instance = nullptr;
}

void Tui::compute_peaks() {
    const auto& ctx = runner_.context();
    int sms = ctx.sm_count();
    int clock_khz = ctx.clock_rate_khz();
    int mem_clock_khz = ctx.memory_clock_khz();
    int bus_width = ctx.memory_bus_width();
    int cc = ctx.compute_capability_major();
    int cc_minor = ctx.compute_capability_minor();

    int fp32_per_sm = 128;
    if (cc == 7) fp32_per_sm = 64;
    else if (cc == 8 && cc_minor == 0) fp32_per_sm = 64;

    peak_fp32_gflops_ = (float)sms * fp32_per_sm * clock_khz * 2.0f / 1e6f;
    peak_mem_bw_gbs_  = (float)mem_clock_khz * 2.0f * (bus_width / 8.0f) / 1e6f;
}

void Tui::refresh_kernels() {
    categories_ = runner_.get_categories();
    kernels_by_category_.clear();
    for (const auto& cat : categories_) {
        auto descs = runner_.get_kernels_by_category(cat);
        std::vector<TuiKernelState> states;
        for (auto* d : descs) {
            TuiKernelState s;
            s.descriptor = d;
            s.selected = true;
            states.push_back(s);
        }
        kernels_by_category_[cat] = std::move(states);
    }
    if (!categories_.empty() && current_category_.empty()) {
        current_category_ = categories_[0];
    }
}

void Tui::enter_raw_mode() {
    if (termios_saved_) return;
    if (tcgetattr(STDIN_FILENO, &orig_termios_) != 0) return;
    termios_saved_ = true;

    struct termios raw = orig_termios_;
    raw.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    raw.c_oflag &= ~OPOST;
    raw.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    raw.c_cflag &= ~(CSIZE | PARENB);
    raw.c_cflag |= CS8;
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void Tui::leave_raw_mode() {
    if (!termios_saved_) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios_);
    termios_saved_ = false;
}

void Tui::enter_alt_screen() {
    if (alt_screen_) return;
    // stdout logs would tear the frame; file sink keeps recording
    arena::set_console_logging(false);
    const char seq[] = "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H\x1b[0m";
    (void)!write(STDOUT_FILENO, seq, sizeof(seq) - 1);
    alt_screen_ = true;
}

void Tui::leave_alt_screen() {
    if (!alt_screen_) return;
    const char seq[] = "\x1b[0m\x1b[?25h\x1b[?1049l";
    (void)!write(STDOUT_FILENO, seq, sizeof(seq) - 1);
    arena::set_console_logging(true);
    alt_screen_ = false;
}

void Tui::refresh_size() {
    struct winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        int new_w = ws.ws_col;
        int new_h = ws.ws_row;
        if (new_w != term_w_ || new_h != term_h_) {
            term_w_ = new_w;
            term_h_ = new_h;
            front_.assign((size_t)term_w_ * term_h_, TuiCell{});
            back_.assign((size_t)term_w_ * term_h_, TuiCell{});
            force_full_redraw_ = true;
        }
    }
    if ((int)back_.size() != term_w_ * term_h_) {
        back_.assign((size_t)term_w_ * term_h_, TuiCell{});
    }
    if ((int)front_.size() != term_w_ * term_h_) {
        front_.assign((size_t)term_w_ * term_h_, TuiCell{});
    }
}

std::string Tui::fmt_time(float ms) const {
    char buf[32];
    if (ms <= 0.0f)      snprintf(buf, sizeof(buf), "--");
    else if (ms < 0.001f) snprintf(buf, sizeof(buf), "%.0f ns", ms * 1e6f);
    else if (ms < 1.0f)   snprintf(buf, sizeof(buf), "%.2f us", ms * 1000.0f);
    else if (ms < 1000.0f) snprintf(buf, sizeof(buf), "%.2f ms", ms);
    else                   snprintf(buf, sizeof(buf), "%.2f s",  ms / 1000.0f);
    return buf;
}

std::string Tui::fmt_bytes(int b) const {
    char buf[32];
    if (b >= 1024 * 1024) snprintf(buf, sizeof(buf), "%.1f MB", b / (1024.0 * 1024.0));
    else if (b >= 1024)   snprintf(buf, sizeof(buf), "%.1f KB", b / 1024.0);
    else                  snprintf(buf, sizeof(buf), "%d B", b);
    return buf;
}

std::string Tui::fmt_count(int n) const {
    char buf[32];
    if (n >= 1000000000) snprintf(buf, sizeof(buf), "%.1fB", n / 1e9);
    else if (n >= 1000000) snprintf(buf, sizeof(buf), "%.1fM", n / 1e6);
    else if (n >= 1000)    snprintf(buf, sizeof(buf), "%.1fK", n / 1e3);
    else                   snprintf(buf, sizeof(buf), "%d", n);
    return buf;
}

TuiDSL Tui::detect_dsl(const arena::KernelDescriptor* d) const {
    if (!d) return TuiDSL::CUDA;
    if (!d->uses_module()) return TuiDSL::CUB;
    if (!d->needs_compilation()) return TuiDSL::CUDA;
    const std::string src = d->source_path();
    if (src.find(".triton.") != std::string::npos) return TuiDSL::Triton;
    if (src.find(".cutile.") != std::string::npos) return TuiDSL::CuTile;
    if (src.find(".warp.")   != std::string::npos) return TuiDSL::Warp;
    return TuiDSL::CUDA;
}

uint32_t Tui::dsl_color(TuiDSL d) const {
    switch (d) {
        case TuiDSL::CUDA:   return C::CUDA_CLR;
        case TuiDSL::Triton: return C::TRITON_CLR;
        case TuiDSL::CuTile: return C::CUTILE_CLR;
        case TuiDSL::Warp:   return C::WARP_CLR;
        case TuiDSL::CUB:    return C::CUB_CLR;
    }
    return C::DIM;
}

const char* Tui::dsl_tag(TuiDSL d) const {
    switch (d) {
        case TuiDSL::CUDA:   return "CU";
        case TuiDSL::Triton: return "TR";
        case TuiDSL::CuTile: return "CT";
        case TuiDSL::Warp:   return "WP";
        case TuiDSL::CUB:    return "CB";
    }
    return "??";
}

const char* Tui::panel_name(TuiPanel p) const {
    switch (p) {
        case TuiPanel::WallVsGpu:         return "Wall vs GPU Time";
        case TuiPanel::Throughput:        return "Throughput vs Peak";
        case TuiPanel::Speedup:           return "Median Time + Speedup";
        case TuiPanel::ProfilingCompare:  return "Profiling Comparison";
        case TuiPanel::Roofline:          return "Roofline Model";
        case TuiPanel::SubKernelTimeline: return "Sub-Kernel Timeline";
        case TuiPanel::TimingDist:        return "Timing Distribution";
        case TuiPanel::Scaling:           return "Scaling";
        default: return "?";
    }
}

const char* Tui::sort_name(TuiSortCol c) const {
    switch (c) {
        case TuiSortCol::Kernel:     return "Name";
        case TuiSortCol::Wall:       return "Wall";
        case TuiSortCol::GPU:        return "GPU";
        case TuiSortCol::Perf:       return "Perf";
        case TuiSortCol::Regs:       return "Regs";
        case TuiSortCol::SHMem:      return "SHMem";
        case TuiSortCol::Occupancy:  return "Occup";
        case TuiSortCol::IPC:        return "IPC";
        default: return "?";
    }
}

void Tui::put_cell(int x, int y, uint32_t ch, uint32_t fg, uint32_t bg, uint8_t attrs) {
    if (x < 0 || y < 0 || x >= term_w_ || y >= term_h_) return;
    TuiCell& c = back_[(size_t)y * term_w_ + x];
    c.ch = ch;
    c.fg = fg;
    c.bg = bg;
    c.attrs = attrs;
}

void Tui::put_text(int x, int y, const std::string& text, uint32_t fg, uint32_t bg, uint8_t attrs) {
    put_text_clipped(x, y, term_w_ - x, text, fg, bg, attrs);
}

int Tui::put_text_clipped(int x, int y, int max_w, const std::string& text,
                           uint32_t fg, uint32_t bg, uint8_t attrs) {
    if (y < 0 || y >= term_h_ || max_w <= 0) return 0;
    int col = 0;
    size_t i = 0;
    while (i < text.size() && col < max_w) {
        unsigned char b = (unsigned char)text[i];
        uint32_t cp;
        int len;
        if (b < 0x80)       { cp = b; len = 1; }
        else if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; len = 2; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; len = 3; }
        else if ((b & 0xF8) == 0xF0) { cp = b & 0x07; len = 4; }
        else                         { i++; continue; }
        if (i + len > text.size()) break;
        for (int k = 1; k < len; k++) {
            cp = (cp << 6) | ((unsigned char)text[i + k] & 0x3F);
        }
        i += len;
        put_cell(x + col, y, cp, fg, bg, attrs);
        col++;
    }
    return col;
}

void Tui::draw_hline(int x, int y, int w, uint32_t fg) {
    for (int i = 0; i < w; i++) put_cell(x + i, y, 0x2500, fg);
}

void Tui::fill_rect(int x, int y, int w, int h, uint32_t ch, uint32_t fg, uint32_t bg) {
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) put_cell(x + i, y + j, ch, fg, bg);
    }
}

void Tui::draw_box(int x, int y, int w, int h, const std::string& title,
                   uint32_t border_fg, bool accent) {
    if (w < 2 || h < 2) return;
    uint32_t bfg = border_fg ? border_fg : (accent ? C::ACCENT : C::BORDER);
    put_cell(x,             y,         0x256D, bfg);  // ╭
    put_cell(x + w - 1,     y,         0x256E, bfg);  // ╮
    put_cell(x,             y + h - 1, 0x2570, bfg);  // ╰
    put_cell(x + w - 1,     y + h - 1, 0x256F, bfg);  // ╯
    for (int i = 1; i < w - 1; i++) {
        put_cell(x + i,     y,         0x2500, bfg);
        put_cell(x + i,     y + h - 1, 0x2500, bfg);
    }
    for (int j = 1; j < h - 1; j++) {
        put_cell(x,         y + j, 0x2502, bfg);
        put_cell(x + w - 1, y + j, 0x2502, bfg);
    }
    if (!title.empty()) {
        // Compact padding: one space either side if it fits, otherwise none.
        int max_inside = w - 2;
        if (max_inside <= 0) return;
        std::string t;
        if ((int)title.size() + 2 <= max_inside) t = " " + title + " ";
        else if ((int)title.size() <= max_inside) t = title;
        else { t = title; t.resize(max_inside); }
        int tx = x + 1;
        uint32_t tfg = accent ? C::ACCENT : C::HEADER;
        put_text_clipped(tx, y, max_inside, t, tfg, 0, ATTR_BOLD);
    }
}

void Tui::draw_hbar(int x, int y, int w, float pct, uint32_t filled_fg, uint32_t track_fg) {
    if (w <= 0) return;
    if (pct < 0) pct = 0;
    if (pct > 1) pct = 1;
    int full_cells = (int)(pct * w);
    int partial = (int)(pct * w * 8) % 8;
    static const uint32_t partials[] = {0x0020, 0x258F, 0x258E, 0x258D, 0x258C, 0x258B, 0x258A, 0x2589};
    for (int i = 0; i < w; i++) {
        if (i < full_cells) {
            put_cell(x + i, y, 0x2588, filled_fg);  // █
        } else if (i == full_cells && partial > 0) {
            put_cell(x + i, y, partials[partial], filled_fg);
        } else {
            put_cell(x + i, y, 0x2591, track_fg);   // ░ light shade
        }
    }
}

void Tui::draw_checkbox(int x, int y, bool checked, uint32_t fg) {
    if (checked) {
        put_cell(x, y, 0x2714, fg ? fg : C::ACCENT);  // ✔
    } else {
        put_cell(x, y, 0x00B7, C::DIM);               // ·
    }
}

int Tui::badge_width(TuiDSL dsl) const {
    (void)dsl;
    return 4;  // "[XX]"
}

void Tui::draw_badge(int x, int y, TuiDSL dsl) {
    uint32_t col = dsl_color(dsl);
    const char* tag = dsl_tag(dsl);
    put_cell(x,     y, '[', col);
    put_cell(x + 1, y, (uint32_t)tag[0], col, 0, ATTR_BOLD);
    put_cell(x + 2, y, (uint32_t)tag[1], col, 0, ATTR_BOLD);
    put_cell(x + 3, y, ']', col);
}

void Tui::draw_status_dot(int x, int y, bool ok, bool running, bool has_run) {
    if (running) {
        put_cell(x, y, 0x25CB, C::ACCENT);         // ○ running
    } else if (!has_run) {
        put_cell(x, y, 0x00B7, C::DIM);            // ·
    } else if (ok) {
        put_cell(x, y, 0x25CF, C::GREEN);          // ● ok
    } else {
        put_cell(x, y, 0x25CF, C::RED);            // ● fail
    }
}

void Tui::clear_frame() {
    for (auto& c : back_) c = TuiCell{' ', 0, 0, 0};
}

void Tui::flush_diff() {
    out_buf_.clear();
    uint32_t cur_fg = 0xFFFFFFFF, cur_bg = 0xFFFFFFFF;
    uint8_t  cur_attrs = 0xFF;
    int      cur_x = -1, cur_y = -1;

    auto emit_sgr = [&](uint32_t fg, uint32_t bg, uint8_t attrs) {
        char buf[64];
        if (cur_attrs != attrs || cur_fg != fg || cur_bg != bg) {
            out_buf_ += "\x1b[0";
            if (attrs & ATTR_BOLD)      out_buf_ += ";1";
            if (attrs & ATTR_DIM)       out_buf_ += ";2";
            if (attrs & ATTR_UNDERLINE) out_buf_ += ";4";
            if (attrs & ATTR_REVERSE)   out_buf_ += ";7";
            if (fg) {
                snprintf(buf, sizeof(buf), ";38;2;%u;%u;%u",
                    (fg >> 16) & 0xFF, (fg >> 8) & 0xFF, fg & 0xFF);
                out_buf_ += buf;
            }
            if (bg) {
                snprintf(buf, sizeof(buf), ";48;2;%u;%u;%u",
                    (bg >> 16) & 0xFF, (bg >> 8) & 0xFF, bg & 0xFF);
                out_buf_ += buf;
            }
            out_buf_ += "m";
            cur_fg = fg; cur_bg = bg; cur_attrs = attrs;
        }
    };

    auto move_to = [&](int x, int y) {
        char buf[32];
        snprintf(buf, sizeof(buf), "\x1b[%d;%dH", y + 1, x + 1);
        out_buf_ += buf;
        cur_x = x; cur_y = y;
    };

    for (int y = 0; y < term_h_; y++) {
        for (int x = 0; x < term_w_; x++) {
            size_t idx = (size_t)y * term_w_ + x;
            const TuiCell& bc = back_[idx];
            const TuiCell& fc = front_[idx];
            if (!force_full_redraw_ && bc == fc) continue;
            if (cur_y != y || cur_x != x) move_to(x, y);
            emit_sgr(bc.fg, bc.bg, bc.attrs);
            append_utf8(out_buf_, bc.ch == 0 ? ' ' : bc.ch);
            cur_x++;  // single-column only (we don't use wide chars)
        }
    }
    out_buf_ += "\x1b[0m";
    if (!out_buf_.empty()) {
        (void)!write(STDOUT_FILENO, out_buf_.data(), out_buf_.size());
    }
    front_ = back_;
    force_full_redraw_ = false;
}

Tui::KeyEvent Tui::read_key(int timeout_ms) {
    KeyEvent ev;

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int rc = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);
    if (rc <= 0) return ev;

    unsigned char buf[16];
    int n = (int)read(STDIN_FILENO, buf, sizeof(buf));
    if (n <= 0) return ev;

    if (buf[0] == 0x1B) {
        if (n == 1) { ev.key = K_ESC; return ev; }
        if (buf[1] == '[' || buf[1] == 'O') {
            if (n >= 3) {
                switch (buf[2]) {
                    case 'A': ev.key = K_UP;    return ev;
                    case 'B': ev.key = K_DOWN;  return ev;
                    case 'C': ev.key = K_RIGHT; return ev;
                    case 'D': ev.key = K_LEFT;  return ev;
                    case 'H': ev.key = K_HOME;  return ev;
                    case 'F': ev.key = K_END;   return ev;
                    case 'Z': ev.key = K_STAB;  return ev;
                    case '5':
                        if (n >= 4 && buf[3] == '~') { ev.key = K_PGUP; return ev; }
                        break;
                    case '6':
                        if (n >= 4 && buf[3] == '~') { ev.key = K_PGDN; return ev; }
                        break;
                    case '3':
                        if (n >= 4 && buf[3] == '~') { ev.key = K_DEL; return ev; }
                        break;
                }
            }
        }
        // unknown escape - consume and ignore
        return ev;
    }

    if (buf[0] == '\r' || buf[0] == '\n') { ev.key = K_ENTER; return ev; }
    if (buf[0] == '\t')                    { ev.key = K_TAB; return ev; }
    if (buf[0] == 127 || buf[0] == 8)      { ev.key = K_BSPC; return ev; }
    if (buf[0] == ' ')                     { ev.key = K_SPACE; ev.ch = ' '; return ev; }

    ev.key = K_CHAR;
    ev.ch  = buf[0];
    return ev;
}

void Tui::run() {
    g_tui_instance = this;

    struct sigaction sa{};
    sa.sa_handler = tui_sig_winch;
    sigaction(SIGWINCH, &sa, &g_old_winch);
    sa.sa_handler = tui_sig_exit;
    sigaction(SIGINT,  &sa, &g_old_sigint);
    sigaction(SIGTERM, &sa, &g_old_sigterm);
    atexit(restore_terminal_atexit);

    enter_raw_mode();
    enter_alt_screen();
    refresh_size();

    running_ = true;
    log(TuiLogEntry::INFO, "GPGPU Arena TUI ready. Press ? for help.");

    auto last_render = std::chrono::steady_clock::now();
    const auto frame_budget = std::chrono::milliseconds(50);  // ~20 fps idle cap

    render();

    while (running_) {
        if (g_exit_pending) { running_ = false; break; }
        if (g_resize_pending) {
            g_resize_pending = 0;
            refresh_size();
            force_full_redraw_ = true;
        }

        int timeout_ms = benchmark_running_.load() ? 50 : 200;
        KeyEvent ev = read_key(timeout_ms);

        bool had_work = false;
        if (ev.key != K_NONE) { handle_key(ev); had_work = true; }

        drain_pending_results();
        auto now = std::chrono::steady_clock::now();
        if (benchmark_running_.load() || had_work || (now - last_render) > frame_budget) {
            render();
            last_render = now;
        }
    }

    leave_alt_screen();
    leave_raw_mode();
}

void Tui::handle_key(const KeyEvent& ev) {
    if (show_help_) {
        if (ev.key == K_ESC || (ev.key == K_CHAR && (ev.ch == '?' || ev.ch == 'q' || ev.ch == 'h'))) {
            show_help_ = false;
        }
        return;
    }

    switch (ev.key) {
        case K_UP:    move_selection(-1); return;
        case K_DOWN:  move_selection(+1); return;
        case K_PGUP:  move_selection(-10); return;
        case K_PGDN:  move_selection(+10); return;
        case K_HOME:  list_cursor_ = 0; list_scroll_ = 0; return;
        case K_END: {
            auto* ks = current_kernels();
            if (ks) list_cursor_ = (int)ks->size() - 1;
            return;
        }
        case K_SPACE: toggle_kernel_selected(); return;
        case K_ENTER: toggle_kernel_focus(); return;
        case K_TAB:   cycle_category(+1); return;
        case K_STAB:  cycle_category(-1); return;
        case K_LEFT:  cycle_panel(-1); return;
        case K_RIGHT: cycle_panel(+1); return;
        case K_ESC:   selected_kernel_name_.clear(); return;
        default: break;
    }

    if (ev.key != K_CHAR) return;
    int c = ev.ch;

    switch (c) {
        case 'q': case 'Q': running_ = false; return;
        case '?': case 'h': show_help_ = true; return;
        case 'j': move_selection(+1); return;
        case 'k': move_selection(-1); return;
        case 'J': move_selection(+10); return;
        case 'K': move_selection(-10); return;
        case 'g': list_cursor_ = 0; list_scroll_ = 0; return;
        case 'G': {
            auto* ks = current_kernels();
            if (ks) list_cursor_ = (int)ks->size() - 1;
            return;
        }
        case 'a': select_all(true); return;
        case 'A': select_all(false); return;
        case 'r': run_selected(); return;
        case 's': run_sweep(); return;
        case 'c': cancel_benchmark(); return;
        case 'R': reset_results(); return;
        case 'p': toggle_profile(); return;
        case 'v': cycle_panel(+1); return;
        case 'V': cycle_panel(-1); return;
        case 'o': cycle_sort(+1); return;
        case 'O': sort_desc_ = !sort_desc_; return;
        case 'l': case 'L': toggle_log_collapsed(); return;
        case 'e': export_csv(); return;
        case 'C': clear_cache(); return;
        case '[': adjust_problem_size(-1, false); return;
        case ']': adjust_problem_size(+1, false); return;
        case '{': adjust_problem_size(-1, true); return;
        case '}': adjust_problem_size(+1, true); return;
        case '-': case '_': adjust_warmup(-1); return;
        case '+': case '=': adjust_warmup(+1); return;
        case ',': case '<': adjust_runs(-1); return;
        case '.': case '>': adjust_runs(+1); return;
        case 'x': lock_square_ = !lock_square_; return;
    }
    if (c >= '1' && c <= '9') {
        jump_category(c - '1');
    }
}

void Tui::cycle_category(int dir) {
    if (categories_.empty()) return;
    int idx = 0;
    for (int i = 0; i < (int)categories_.size(); i++) {
        if (categories_[i] == current_category_) { idx = i; break; }
    }
    idx = (idx + dir + (int)categories_.size()) % (int)categories_.size();
    current_category_ = categories_[idx];
    list_cursor_ = 0;
    list_scroll_ = 0;
}

void Tui::jump_category(int idx) {
    if (idx < 0 || idx >= (int)categories_.size()) return;
    current_category_ = categories_[idx];
    list_cursor_ = 0;
    list_scroll_ = 0;
}

void Tui::cycle_panel(int dir) {
    int n = (int)TuiPanel::COUNT;
    int cur = (int)panel_;
    cur = (cur + dir + n) % n;
    panel_ = (TuiPanel)cur;
}

void Tui::cycle_sort(int dir) {
    int n = (int)TuiSortCol::COUNT;
    int cur = (int)sort_col_;
    cur = (cur + dir + n) % n;
    sort_col_ = (TuiSortCol)cur;
}

void Tui::move_selection(int delta) {
    auto* ks = current_kernels();
    if (!ks || ks->empty()) return;
    int n = (int)ks->size();
    list_cursor_ = std::max(0, std::min(n - 1, list_cursor_ + delta));
}

void Tui::toggle_kernel_selected() {
    auto* ks = current_kernels();
    if (!ks || ks->empty() || list_cursor_ < 0 || list_cursor_ >= (int)ks->size()) return;
    (*ks)[list_cursor_].selected = !(*ks)[list_cursor_].selected;
}

void Tui::toggle_kernel_focus() {
    auto* ks = current_kernels();
    if (!ks || ks->empty() || list_cursor_ < 0 || list_cursor_ >= (int)ks->size()) return;
    const std::string& name = (*ks)[list_cursor_].descriptor->name();
    if (selected_kernel_name_ == name) selected_kernel_name_.clear();
    else                                selected_kernel_name_ = name;
}

void Tui::select_all(bool all) {
    auto* ks = current_kernels();
    if (!ks) return;
    for (auto& k : *ks) k.selected = all;
}

void Tui::adjust_problem_size(int dir, bool big_step) {
    float factor = big_step ? 2.0f : 1.25f;
    auto clamp = [](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };
    if (current_category_ == "matmul") {
        int m = config_.params["M"];
        int k = config_.params["K"];
        int n = config_.params["N"];
        if (lock_square_) {
            int v = (int)(m * (dir > 0 ? factor : 1.0f / factor));
            v = clamp(v, 64, 8192);
            v = (v / 64) * 64;
            config_.params["M"] = v;
            config_.params["K"] = v;
            config_.params["N"] = v;
        } else {
            int v = (int)(m * (dir > 0 ? factor : 1.0f / factor));
            v = clamp(v, 64, 8192);
            config_.params["M"] = (v / 64) * 64;
        }
        (void)k; (void)n;
    } else if (current_category_ == "softmax") {
        int rows = config_.params["rows"];
        int cols = config_.params["cols"];
        int v = (int)(rows * (dir > 0 ? factor : 1.0f / factor));
        v = clamp(v, 64, 8192);
        v = (v / 64) * 64;
        config_.params["rows"] = v;
        config_.params["cols"] = v;
        (void)cols;
    } else {
        int n = config_.params["n"];
        int v = (int)(n * (dir > 0 ? factor : 1.0f / factor));
        v = clamp(v, 1000, 100000000);
        config_.params["n"] = v;
    }
}

void Tui::adjust_warmup(int dir) {
    int v = std::max(0, std::min(100, config_.warmup_runs + dir));
    config_.warmup_runs = v;
}

void Tui::adjust_runs(int dir) {
    int v = std::max(1, std::min(1000, config_.number_of_runs + dir));
    config_.number_of_runs = v;
}

void Tui::toggle_profile() {
    config_.collect_metrics = !config_.collect_metrics;
    log(TuiLogEntry::INFO, config_.collect_metrics
        ? "Profiling enabled (slower)" : "Profiling disabled");
}

void Tui::toggle_log_collapsed() {
    log_collapsed_ = !log_collapsed_;
}

std::vector<TuiKernelState>* Tui::current_kernels() {
    auto it = kernels_by_category_.find(current_category_);
    return it == kernels_by_category_.end() ? nullptr : &it->second;
}

const TuiKernelState* Tui::selected_kernel() const {
    if (selected_kernel_name_.empty()) {
        // default to cursor
        auto it = kernels_by_category_.find(current_category_);
        if (it == kernels_by_category_.end() || it->second.empty()) return nullptr;
        int c = std::max(0, std::min(list_cursor_, (int)it->second.size() - 1));
        return &it->second[c];
    }
    for (const auto& [cat, states] : kernels_by_category_) {
        for (const auto& s : states) {
            if (s.descriptor && s.descriptor->name() == selected_kernel_name_) return &s;
        }
    }
    return nullptr;
}

bool Tui::has_any_profile() const {
    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return false;
    for (const auto& k : it->second) {
        if (k.has_run && k.result.achieved_occupancy > 0) return true;
    }
    return false;
}

std::vector<int> Tui::sorted_table_indices() const {
    std::vector<int> out;
    auto it = kernels_by_category_.find(current_category_);
    if (it == kernels_by_category_.end()) return out;
    const auto& ks = it->second;
    for (int i = 0; i < (int)ks.size(); i++) if (ks[i].has_run) out.push_back(i);

    bool matmul = (current_category_ == "matmul");
    TuiSortCol col = sort_col_;
    bool desc = sort_desc_;

    std::sort(out.begin(), out.end(), [&](int a, int b) {
        const auto& ra = ks[a].result;
        const auto& rb = ks[b].result;
        int cmp = 0;
        switch (col) {
            case TuiSortCol::Kernel: cmp = ra.kernel_name.compare(rb.kernel_name); break;
            case TuiSortCol::Wall:   cmp = (ra.elapsed_ms < rb.elapsed_ms) ? -1 : (ra.elapsed_ms > rb.elapsed_ms); break;
            case TuiSortCol::GPU:    cmp = (ra.kernel_ms < rb.kernel_ms) ? -1 : (ra.kernel_ms > rb.kernel_ms); break;
            case TuiSortCol::Perf: {
                double va = matmul ? ra.gflops : ra.bandwidth_gbps;
                double vb = matmul ? rb.gflops : rb.bandwidth_gbps;
                cmp = (va < vb) ? -1 : (va > vb); break;
            }
            case TuiSortCol::Regs:  cmp = ra.registers_per_thread - rb.registers_per_thread; break;
            case TuiSortCol::SHMem: cmp = ra.shared_memory_bytes  - rb.shared_memory_bytes; break;
            case TuiSortCol::Occupancy:
                cmp = (ra.achieved_occupancy < rb.achieved_occupancy) ? -1 : 1; break;
            case TuiSortCol::IPC:
                cmp = (ra.ipc < rb.ipc) ? -1 : (ra.ipc > rb.ipc); break;
            default: break;
        }
        return desc ? (cmp > 0) : (cmp < 0);
    });
    return out;
}

void Tui::log(TuiLogEntry::Level lv, const std::string& msg) {
    log_entries_.push_back({lv, msg});
    if (log_entries_.size() > MAX_LOG_ENTRIES) log_entries_.pop_front();
}

void Tui::run_selected() {
    if (benchmark_running_) return;
    auto* ks = current_kernels();
    if (!ks) return;

    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work;
    for (auto& k : *ks) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor});
        }
    }
    if (work.empty()) {
        log(TuiLogEntry::WARN, "Nothing selected - press Space to toggle kernels");
        return;
    }

    if (benchmark_thread_.joinable()) benchmark_thread_.join();
    cancel_requested_   = false;
    benchmark_running_  = true;
    benchmark_current_  = 0;
    benchmark_total_    = (int)work.size();
    benchmark_thread_ = std::thread(&Tui::benchmark_thread_func, this,
        std::move(work), config_);
}

void Tui::run_sweep() {
    if (benchmark_running_) return;
    auto* ks = current_kernels();
    if (!ks) return;

    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work;
    for (auto& k : *ks) {
        if (k.selected && k.descriptor) {
            work.push_back({current_category_, k.descriptor});
        }
    }
    if (work.empty()) { log(TuiLogEntry::WARN, "Nothing selected for sweep"); return; }

    auto sweep_cfgs = work[0].second->get_sweep_configs();
    if (sweep_cfgs.empty()) {
        log(TuiLogEntry::WARN, "No sweep configs defined for this category");
        return;
    }

    if (benchmark_thread_.joinable()) benchmark_thread_.join();
    cancel_requested_   = false;
    benchmark_running_  = true;
    benchmark_current_  = 0;
    benchmark_total_    = (int)(work.size() * sweep_cfgs.size());
    benchmark_thread_ = std::thread(&Tui::sweep_thread_func, this,
        std::move(work), std::move(sweep_cfgs), config_);
}

void Tui::cancel_benchmark() {
    if (!benchmark_running_) return;
    cancel_requested_ = true;
    log(TuiLogEntry::WARN, "Cancel requested");
}

void Tui::reset_results() {
    if (benchmark_running_) return;
    auto* ks = current_kernels();
    if (!ks) return;
    for (auto& k : *ks) { k.has_run = false; k.result = arena::RunResult{}; }
    log(TuiLogEntry::INFO, "Results reset");
}

void Tui::clear_cache() {
    if (benchmark_running_) return;
    runner_.compiler().clear_cache();
    log(TuiLogEntry::INFO, "Kernel cache cleared");
}

void Tui::export_csv() {
    char cwd[1024];
    std::string path = "gpgpu_arena_results.csv";
    if (getcwd(cwd, sizeof(cwd))) {
        path = std::string(cwd) + "/gpgpu_arena_results.csv";
    }
    std::ofstream f(path);
    if (!f.is_open()) { log(TuiLogEntry::ERR, "Failed to open " + path); return; }

    f << "kernel,category,dsl,block,grid,wall_ms,gpu_ms,gflops,bandwidth_gbps,"
         "status,regs,shmem_bytes,occupancy_pct,ipc,dram_read_gbps,dram_write_gbps,"
         "cache_hit,compile_time_ms\n";

    for (const auto& [cat, states] : kernels_by_category_) {
        for (const auto& k : states) {
            if (!k.has_run) continue;
            const auto& r = k.result;
            f << r.kernel_name << "," << r.category << ","
              << dsl_tag(detect_dsl(k.descriptor)) << ","
              << r.block_x << "x" << r.block_y << "x" << r.block_z << ","
              << r.grid_x << "x" << r.grid_y << "x" << r.grid_z << ","
              << r.elapsed_ms << "," << r.kernel_ms << ","
              << r.gflops << "," << r.bandwidth_gbps << ","
              << (r.success ? (r.verified ? "OK" : "WARN") : "FAIL") << ","
              << r.registers_per_thread << "," << r.shared_memory_bytes << ","
              << r.achieved_occupancy * 100.0 << "," << r.ipc << ","
              << r.dram_read_gbps << "," << r.dram_write_gbps << ","
              << (r.cache_hit ? "true" : "false") << "," << r.compile_time_ms << "\n";
        }
    }
    log(TuiLogEntry::INFO, "Exported to " + path);
}

void Tui::benchmark_thread_func(
    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
    arena::RunConfig cfg) {

    CUcontext ctx = runner_.context().handle();
    cuCtxPushCurrent(ctx);

    for (int i = 0; i < (int)work.size(); i++) {
        if (cancel_requested_) break;
        auto& [cat, desc] = work[i];
        benchmark_current_ = i;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            benchmark_current_name_ = desc->name();
        }

        TuiPendingResult pr;
        pr.category = cat;
        pr.kernel_name = desc->name();
        pr.params = cfg.params;
        pr.logs.push_back({TuiLogEntry::INFO, "Running " + desc->name() + " ..."});

        pr.result = runner_.run(*desc, cfg);

        if (pr.result.success) {
            char buf[256];
            bool matmul = (cat == "matmul");
            snprintf(buf, sizeof(buf), "%s: wall=%s  gpu=%s  %.2f %s",
                pr.result.kernel_name.c_str(),
                fmt_time(pr.result.elapsed_ms).c_str(),
                fmt_time(pr.result.kernel_ms).c_str(),
                matmul ? pr.result.gflops : pr.result.bandwidth_gbps,
                matmul ? "GFLOPS" : "GB/s");
            pr.logs.push_back({TuiLogEntry::BENCHMARK, buf});
            if (cfg.collect_metrics && pr.result.achieved_occupancy > 0) {
                snprintf(buf, sizeof(buf),
                    "%s: regs=%d  shmem=%dB  occ=%.1f%%  IPC=%.2f",
                    pr.result.kernel_name.c_str(),
                    pr.result.registers_per_thread,
                    pr.result.shared_memory_bytes,
                    pr.result.achieved_occupancy * 100.0,
                    pr.result.ipc);
                pr.logs.push_back({TuiLogEntry::PROFILE, buf});
            }
            if (!pr.result.verified) {
                pr.logs.push_back({TuiLogEntry::WARN,
                    pr.result.kernel_name + ": verification FAILED"});
            }
        } else {
            pr.logs.push_back({TuiLogEntry::ERR,
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

void Tui::sweep_thread_func(
    std::vector<std::pair<std::string, arena::KernelDescriptor*>> work,
    std::vector<std::map<std::string, int>> sweep_cfgs,
    arena::RunConfig cfg) {

    CUcontext ctx = runner_.context().handle();
    cuCtxPushCurrent(ctx);

    int i = 0;
    for (const auto& params : sweep_cfgs) {
        if (cancel_requested_) break;
        cfg.params = params;
        for (auto& [cat, desc] : work) {
            if (cancel_requested_) break;
            benchmark_current_ = i++;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                benchmark_current_name_ = desc->name();
            }

            TuiPendingResult pr;
            pr.category = cat;
            pr.kernel_name = desc->name();
            pr.params = params;

            std::string sz_str;
            for (auto& [k, v] : params) {
                if (!sz_str.empty()) sz_str += ",";
                sz_str += k + "=" + std::to_string(v);
            }
            pr.logs.push_back({TuiLogEntry::INFO,
                "Sweep " + desc->name() + " [" + sz_str + "] ..."});

            pr.result = runner_.run(*desc, cfg);

            if (pr.result.success) {
                char buf[256];
                bool matmul = (cat == "matmul");
                snprintf(buf, sizeof(buf), "%s [%s]: wall=%s  %.2f %s",
                    pr.result.kernel_name.c_str(), sz_str.c_str(),
                    fmt_time(pr.result.elapsed_ms).c_str(),
                    matmul ? pr.result.gflops : pr.result.bandwidth_gbps,
                    matmul ? "GFLOPS" : "GB/s");
                pr.logs.push_back({TuiLogEntry::BENCHMARK, buf});
            } else {
                pr.logs.push_back({TuiLogEntry::ERR,
                    pr.result.kernel_name + ": " + pr.result.error});
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending_results_.push_back(std::move(pr));
            }
        }
    }

    benchmark_current_ = (int)(work.size() * sweep_cfgs.size());
    CUcontext popped;
    cuCtxPopCurrent(&popped);
    benchmark_running_ = false;
}

void Tui::drain_pending_results() {
    std::vector<TuiPendingResult> results;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        results.swap(pending_results_);
    }
    for (auto& pr : results) {
        for (auto& entry : pr.logs) log(entry.level, entry.message);

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
                    [](const TuiSizedResult& a, const TuiSizedResult& b) {
                        return a.problem_size < b.problem_size;
                    });
            }
        }
    }

    if (!benchmark_running_ && benchmark_thread_.joinable()) {
        benchmark_thread_.join();
        if (benchmark_current_ >= benchmark_total_) log(TuiLogEntry::INFO, "--- Done ---");
        else                                         log(TuiLogEntry::WARN, "--- Cancelled ---");
    }
}

void Tui::render() {
    if (term_w_ < 80 || term_h_ < 20) {
        clear_frame();
        put_text(0, 0, "Terminal too small. Please resize to at least 80x20.", C::YELLOW);
        flush_diff();
        return;
    }

    clear_frame();

    render_header(0);

    int log_h;
    if (log_collapsed_) log_h = 2;
    else                log_h = std::max(5, term_h_ / 5);
    if (log_h > term_h_ - 10) log_h = term_h_ - 10;

    int footer_h = 1;
    int middle_y = 2;
    int middle_h = term_h_ - 2 - log_h - footer_h;

    int left_w, right_w;
    if (term_w_ >= 160) { left_w = 32; right_w = 36; }
    else if (term_w_ >= 130) { left_w = 28; right_w = 32; }
    else if (term_w_ >= 100) { left_w = 24; right_w = 28; }
    else                     { left_w = 20; right_w = 24; }
    int center_w = term_w_ - left_w - right_w;
    if (center_w < 36) {
        // Shrink or drop the right sidebar to give center enough room
        right_w = std::max(0, term_w_ - left_w - 36);
        center_w = term_w_ - left_w - right_w;
    }

    int settings_h = 14;
    if (settings_h > middle_h / 2) settings_h = middle_h / 2;
    int list_h = middle_h - settings_h;
    render_kernel_list(0, middle_y, left_w, list_h);
    render_settings(0, middle_y + list_h, left_w, settings_h);

    render_center(left_w, middle_y, center_w, middle_h);

    if (right_w > 0) {
        render_detail(left_w + center_w, middle_y, right_w, middle_h);
    }

    render_log(0, middle_y + middle_h, term_w_, log_h);

    render_footer(term_h_ - 1);

    if (show_help_) render_help_overlay();

    flush_diff();
}

void Tui::render_header(int y) {
    fill_rect(0, y, term_w_, 2, ' ', 0, C::BG_HDR);

    put_text(1, y,   "GPGPU Arena", C::ACCENT, C::BG_HDR, ATTR_BOLD);

    const auto& ctx = runner_.context();
    char info[256];
    size_t mem_mb = ctx.total_memory() / (1024 * 1024);
    char vram[32];
    if (mem_mb >= 1024) snprintf(vram, sizeof(vram), "%.1f GB", mem_mb / 1024.0);
    else                snprintf(vram, sizeof(vram), "%zu MB", mem_mb);
    snprintf(info, sizeof(info), "%s  sm_%d%d  %d SMs  %s",
        ctx.device_name().c_str(),
        ctx.compute_capability_major(), ctx.compute_capability_minor(),
        ctx.sm_count(), vram);
    put_text_clipped(14, y, term_w_ - 40, info, C::BODY, C::BG_HDR);

    std::string status;
    uint32_t status_col;
    if (benchmark_running_) {
        int cur = benchmark_current_.load();
        int tot = benchmark_total_.load();
        char b[128];
        if (config_.collect_metrics)
            snprintf(b, sizeof(b), "PROFILING  %d/%d", cur + 1, tot);
        else
            snprintf(b, sizeof(b), "RUNNING    %d/%d", cur + 1, tot);
        status = b;
        status_col = config_.collect_metrics ? C::WARP_CLR : C::ACCENT;
    } else {
        int total_k = 0;
        for (const auto& [c, ks] : kernels_by_category_) total_k += (int)ks.size();
        char b[64];
        snprintf(b, sizeof(b), "IDLE  %d kernels", total_k);
        status = b;
        status_col = C::DIM;
    }
    int sx = term_w_ - (int)status.size() - 2;
    if (sx < 0) sx = 0;
    put_text(sx, y, status, status_col, C::BG_HDR, ATTR_BOLD);

    int x = 1;
    put_text(x, y + 1, "Categories:", C::DIM, C::BG_HDR);
    x += 12;
    for (size_t i = 0; i < categories_.size() && x < term_w_ - 4; i++) {
        const auto& cat = categories_[i];
        bool active = (cat == current_category_);
        std::string tab = " ";
        tab += (char)('1' + i);
        tab += " ";
        if (!cat.empty()) {
            std::string name = cat;
            name[0] = (char)std::toupper(name[0]);
            tab += name;
        }
        tab += " ";
        uint32_t fg = active ? 0x0B0B0B : C::BODY;
        uint32_t bg = active ? C::ACCENT : C::BG_HDR;
        uint8_t  at = active ? ATTR_BOLD : 0;
        x += put_text_clipped(x, y + 1, term_w_ - 2 - x, tab, fg, bg, at);
        x++;
    }

    std::string pbadge = std::string("  Profile: ") + (config_.collect_metrics ? "ON " : "OFF");
    int px = term_w_ - (int)pbadge.size() - 1;
    if (px > x + 4) {
        put_text(px, y + 1, pbadge,
            config_.collect_metrics ? C::WARP_CLR : C::DIM, C::BG_HDR,
            config_.collect_metrics ? ATTR_BOLD : 0);
    }
}

void Tui::render_kernel_list(int x, int y, int w, int h) {
    std::string title = "Kernels";
    auto* ks = current_kernels();
    if (ks) {
        int sel = 0;
        for (const auto& k : *ks) if (k.selected) sel++;
        char buf[64];
        snprintf(buf, sizeof(buf), "Kernels [%d/%zu]", sel, ks->size());
        title = buf;
    }
    bool is_focus = (focus_ == TuiFocus::KernelList);
    draw_box(x, y, w, h, title, 0, is_focus);

    if (!ks || ks->empty()) {
        put_text(x + 2, y + 2, "(none)", C::DIM);
        return;
    }

    int inner_h = h - 2;
    int inner_w = w - 2;
    int cur = list_cursor_;
    int n = (int)ks->size();
    if (cur < list_scroll_) list_scroll_ = cur;
    if (cur >= list_scroll_ + inner_h) list_scroll_ = cur - inner_h + 1;
    list_scroll_ = std::max(0, std::min(list_scroll_, std::max(0, n - inner_h)));

    std::string running_name;
    if (benchmark_running_) {
        std::lock_guard<std::mutex> lock(mutex_);
        running_name = benchmark_current_name_;
    }

    for (int i = 0; i < inner_h && (list_scroll_ + i) < n; i++) {
        int idx = list_scroll_ + i;
        const auto& k = (*ks)[idx];
        int rx = x + 1;
        int ry = y + 1 + i;
        bool is_cursor = (idx == cur);
        bool is_focus_sel = (selected_kernel_name_ == k.descriptor->name());

        uint32_t row_bg = 0;
        if (is_cursor)     row_bg = 0x262626;
        if (is_focus_sel)  row_bg = 0x1F3530;

        for (int j = 0; j < inner_w; j++) put_cell(rx + j, ry, ' ', 0, row_bg);

        draw_checkbox(rx, ry, k.selected);

        TuiDSL dsl = detect_dsl(k.descriptor);
        draw_badge(rx + 2, ry, dsl);

        const std::string& nm = k.descriptor->name();
        int name_x = rx + 7;
        int reserve_right = 10;  // time column
        int name_w = inner_w - (name_x - rx) - reserve_right;
        if (name_w < 5) name_w = 5;
        uint32_t nfg = C::BODY;
        if (is_cursor) nfg = C::HEADER;
        if (is_focus_sel) nfg = C::ACCENT;
        put_text_clipped(name_x, ry, name_w, nm, nfg, row_bg, is_focus_sel ? ATTR_BOLD : 0);

        int tx_x = x + w - 1 - reserve_right;
        bool is_running = benchmark_running_ && running_name == nm;
        if (is_running) {
            put_text(tx_x, ry, " ... ", C::ACCENT, row_bg, ATTR_BOLD);
        } else if (!k.has_run) {
            put_text(tx_x, ry, "  --  ", C::DIM, row_bg);
        } else if (!k.result.success) {
            put_text(tx_x, ry, " ERR  ", C::RED, row_bg, ATTR_BOLD);
        } else {
            std::string t = fmt_time(k.result.elapsed_ms);
            int tw = (int)t.size();
            int cx = x + w - 1 - tw - 1;
            put_text(cx, ry, t, k.result.verified ? C::GREEN : C::YELLOW, row_bg);
        }
    }

    if (n > inner_h) {
        int bar_x = x + w - 1;
        for (int i = 0; i < inner_h; i++) put_cell(bar_x, y + 1 + i, 0x2502, C::BORDER_DIM);
        int knob_h = std::max(1, inner_h * inner_h / n);
        int knob_y = inner_h - knob_h;
        if (n > inner_h) knob_y = list_scroll_ * (inner_h - knob_h) / (n - inner_h);
        for (int i = 0; i < knob_h; i++) put_cell(bar_x, y + 1 + knob_y + i, 0x2588, C::BORDER);
    }
}

void Tui::render_settings(int x, int y, int w, int h) {
    draw_box(x, y, w, h, "Settings");
    int ix = x + 2;
    int iy = y + 1;
    int iw = w - 4;
    if (iw < 10) return;

    put_text(ix, iy, "Problem Size", C::HEADER, 0, ATTR_BOLD);
    iy++;
    if (current_category_ == "matmul") {
        int m = config_.params["M"];
        int k = config_.params["K"];
        int n = config_.params["N"];
        char b[64];
        if (lock_square_) {
            snprintf(b, sizeof(b), "  M=K=N= %d (x)", m);
        } else {
            snprintf(b, sizeof(b), "  M=%d K=%d N=%d", m, k, n);
        }
        put_text_clipped(ix, iy++, iw, b, C::BODY);
        put_text_clipped(ix, iy++, iw, std::string("  square: ") + (lock_square_ ? "locked" : "free"), C::DIM);
    } else if (current_category_ == "softmax") {
        int rows = config_.params["rows"];
        int cols = config_.params["cols"];
        char b[64];
        snprintf(b, sizeof(b), "  %d x %d", rows, cols);
        put_text_clipped(ix, iy++, iw, b, C::BODY);
        float mb = (2.0f * rows * cols * sizeof(float)) / (1024.0f * 1024.0f);
        char mb_s[32]; snprintf(mb_s, sizeof(mb_s), "  %.1f MB", mb);
        put_text_clipped(ix, iy++, iw, mb_s, C::DIM);
    } else {
        int n = config_.params["n"];
        char b[64];
        snprintf(b, sizeof(b), "  n = %s elems", fmt_count(n).c_str());
        put_text_clipped(ix, iy++, iw, b, C::BODY);
        float mb = (n * sizeof(float)) / (1024.0f * 1024.0f);
        char mb_s[32]; snprintf(mb_s, sizeof(mb_s), "  %.1f MB", mb);
        put_text_clipped(ix, iy++, iw, mb_s, C::DIM);
    }
    put_text_clipped(ix, iy++, iw, "  [ ] size-   [ ] size+", C::DIM);
    iy++;

    if (iy < y + h - 1) {
        put_text(ix, iy, "Timing", C::HEADER, 0, ATTR_BOLD);
        iy++;
        char wb[64], rb[64];
        snprintf(wb, sizeof(wb), "  Warmup: %d  (- +)", config_.warmup_runs);
        snprintf(rb, sizeof(rb), "  Runs:   %d  (< >)", config_.number_of_runs);
        put_text_clipped(ix, iy++, iw, wb, C::BODY);
        put_text_clipped(ix, iy++, iw, rb, C::BODY);
    }
    iy++;

    if (iy < y + h - 1) {
        std::string p = std::string("Profile [p]: ") +
            (config_.collect_metrics ? "ON" : "OFF");
        put_text_clipped(ix, iy++, iw, p,
            config_.collect_metrics ? C::WARP_CLR : C::DIM,
            0, config_.collect_metrics ? ATTR_BOLD : 0);
        if (config_.collect_metrics) {
            put_text_clipped(ix, iy++, iw, "  (slower: kernel replay)", C::YELLOW);
        }
    }
}

void Tui::render_center(int x, int y, int w, int h) {
    draw_box(x, y, w, h, "Benchmarks");
    int ix = x + 1;
    int iy = y + 1;
    int iw = w - 2;

    int row = iy;
    render_results_table(ix, row, iw, row);
    if (row < y + h - 2) {
        row++;
        render_kpi_strip(ix, row, iw, row);
    }
    if (row < y + h - 3) {
        row++;
        draw_hline(ix, row, iw, C::BORDER_DIM);
        row++;
        render_panel(ix, row, iw, (y + h - 1) - row);
    }
}

void Tui::render_results_table(int x, int y, int w, int& row_out) {
    auto* ks = current_kernels();
    if (!ks || ks->empty()) {
        put_text(x, y, "(no kernels)", C::DIM);
        row_out = y;
        return;
    }
    bool matmul = is_matmul();
    bool has_prof = has_any_profile();

    int cx = x;
    put_text(cx, y, " ", 0); cx++;
    const char* col_kernel = "Kernel";
    put_text(cx, y, col_kernel,
        sort_col_ == TuiSortCol::Kernel ? C::ACCENT : C::HEADER,
        0, ATTR_BOLD);
    int name_w = std::min(28, w - 55);
    if (name_w < 10) name_w = 10;
    cx = x + name_w;

    struct Col { const char* name; int width; TuiSortCol id; };
    std::vector<Col> cols = {
        {"Block",    7, TuiSortCol::COUNT},
        {"Grid",     8, TuiSortCol::COUNT},
        {"Wall",     9, TuiSortCol::Wall},
        {"GPU",      9, TuiSortCol::GPU},
        {matmul ? "GFLOPS" : "GB/s", 8, TuiSortCol::Perf},
        {"St",       3, TuiSortCol::COUNT},
    };
    if (has_prof) {
        cols.push_back({"Regs",  5, TuiSortCol::Regs});
        cols.push_back({"SHMem", 6, TuiSortCol::SHMem});
        cols.push_back({"Occ%",  6, TuiSortCol::Occupancy});
        cols.push_back({"IPC",   5, TuiSortCol::IPC});
    }

    for (auto& c : cols) {
        if (cx + c.width >= x + w) break;
        bool sel = (c.id == sort_col_ && c.id != TuiSortCol::COUNT);
        std::string hdr = c.name;
        if (sel) hdr = (sort_desc_ ? std::string("↓") : std::string("↑")) + hdr;
        int pad = c.width - (int)hdr.size();
        if (pad < 0) pad = 0;
        for (int i = 0; i < pad; i++) put_cell(cx + i, y, ' ');
        put_text_clipped(cx + pad, y, c.width - pad, hdr,
            sel ? C::ACCENT : C::HEADER, 0, ATTR_BOLD);
        cx += c.width + 1;
    }
    int hdr_y = y;
    int cur_y = y + 1;
    for (int i = 0; i < w; i++) put_cell(x + i, cur_y, 0x2500, C::BORDER_DIM);
    cur_y++;

    auto indices = sorted_table_indices();
    std::vector<int> unrun;
    for (int i = 0; i < (int)ks->size(); i++) {
        if (!(*ks)[i].has_run) unrun.push_back(i);
    }
    std::vector<int> all_idx = indices;
    for (int u : unrun) all_idx.push_back(u);

    int max_rows = std::max(3, std::min(10, (int)all_idx.size()));
    for (int r = 0; r < (int)all_idx.size() && r < max_rows; r++) {
        int ki = all_idx[r];
        const auto& k = (*ks)[ki];
        bool is_cursor_row = (ki == list_cursor_);
        bool is_focus_row = (k.descriptor && selected_kernel_name_ == k.descriptor->name());

        uint32_t row_bg = (r & 1) ? 0x101010 : 0;
        if (is_cursor_row)     row_bg = 0x1F1F1F;
        if (is_focus_row)      row_bg = 0x1F3530;
        for (int i = 0; i < w; i++) put_cell(x + i, cur_y, ' ', 0, row_bg);

        TuiDSL dsl = detect_dsl(k.descriptor);
        draw_badge(x + 1, cur_y, dsl);
        put_text_clipped(x + 1 + 4 + 1, cur_y, name_w - 6,
            k.descriptor->name(),
            is_focus_row ? C::ACCENT : C::BODY, row_bg,
            is_focus_row ? ATTR_BOLD : 0);

        int col_x = x + name_w;
        auto cell_put_right = [&](int width, const std::string& s, uint32_t fg) {
            int sw = (int)s.size();
            if (sw > width) sw = width;
            int pad = width - sw;
            put_text_clipped(col_x + pad, cur_y, sw, s, fg, row_bg);
            col_x += width + 1;
        };

        char buf[64];
        if (k.has_run) {
            snprintf(buf, sizeof(buf), "%ux%u", k.result.block_x, k.result.block_y);
            cell_put_right(cols[0].width, buf, C::BODY);
            snprintf(buf, sizeof(buf), "%ux%u", k.result.grid_x, k.result.grid_y);
            cell_put_right(cols[1].width, buf, C::BODY);
        } else {
            cell_put_right(cols[0].width, "--", C::DIM);
            cell_put_right(cols[1].width, "--", C::DIM);
        }

        if (k.has_run && k.result.success) {
            cell_put_right(cols[2].width, fmt_time(k.result.elapsed_ms), C::BODY);
            cell_put_right(cols[3].width, fmt_time(k.result.kernel_ms), C::BODY);
            snprintf(buf, sizeof(buf), "%.1f",
                matmul ? k.result.gflops : k.result.bandwidth_gbps);
            cell_put_right(cols[4].width, buf, C::ACCENT);
            cell_put_right(cols[5].width,
                k.result.verified ? "OK" : "WN",
                k.result.verified ? C::GREEN : C::YELLOW);
            if (has_prof) {
                if (k.result.registers_per_thread > 0) {
                    snprintf(buf, sizeof(buf), "%d", k.result.registers_per_thread);
                    cell_put_right(cols[6].width, buf, C::BODY);
                } else cell_put_right(cols[6].width, "-", C::DIM);

                if (k.result.shared_memory_bytes > 0) {
                    snprintf(buf, sizeof(buf), "%d", k.result.shared_memory_bytes);
                    cell_put_right(cols[7].width, buf, C::BODY);
                } else cell_put_right(cols[7].width, "-", C::DIM);

                if (k.result.achieved_occupancy > 0) {
                    snprintf(buf, sizeof(buf), "%.1f", k.result.achieved_occupancy * 100.0);
                    cell_put_right(cols[8].width, buf, C::BODY);
                } else cell_put_right(cols[8].width, "-", C::DIM);

                if (k.result.ipc > 0) {
                    snprintf(buf, sizeof(buf), "%.2f", k.result.ipc);
                    cell_put_right(cols[9].width, buf, C::BODY);
                } else cell_put_right(cols[9].width, "-", C::DIM);
            }
        } else if (k.has_run) {
            cell_put_right(cols[2].width, "--", C::DIM);
            cell_put_right(cols[3].width, "--", C::DIM);
            cell_put_right(cols[4].width, "--", C::DIM);
            cell_put_right(cols[5].width, "FAIL", C::RED);
        } else {
            cell_put_right(cols[2].width, "--", C::DIM);
            cell_put_right(cols[3].width, "--", C::DIM);
            cell_put_right(cols[4].width, "--", C::DIM);
            cell_put_right(cols[5].width, "--", C::DIM);
        }
        cur_y++;
    }
    (void)hdr_y;
    row_out = cur_y;
}

void Tui::render_kpi_strip(int x, int y, int w, int& row_out) {
    const auto* sel = selected_kernel();
    const bool has_data    = sel && sel->has_run && sel->result.success;
    const bool has_profile = has_data && sel->result.achieved_occupancy > 0;

    // Layout: 6 KPI cards in a row. Each needs ~17 wide, 3 tall.
    int cards = 6;
    int card_w = (w - (cards - 1)) / cards;
    if (card_w < 12) {
        cards = 3;
        card_w = (w - (cards - 1)) / cards;
    }
    if (card_w < 10) { row_out = y; return; }
    int card_h = 3;

    struct KPI {
        const char* label;
        std::string value;
        const char* unit;
        bool ok;
        float pct;  // -1 to disable bar
        uint32_t color;
    };
    std::vector<KPI> kpis;

    kpis.push_back({"Median", has_data ? fmt_time(sel->result.elapsed_ms) : "--",
                    "", has_data, -1.0f, C::ACCENT});

    char mb_s[32];
    if (has_data) snprintf(mb_s, sizeof(mb_s), "%.1f", sel->result.bandwidth_gbps);
    else          snprintf(mb_s, sizeof(mb_s), "--");
    kpis.push_back({"Mem BW", mb_s, "GB/s", has_data,
                    has_data && peak_mem_bw_gbs_ > 0
                        ? (float)(sel->result.bandwidth_gbps / peak_mem_bw_gbs_) : -1.0f,
                    C::ACCENT});

    const char* flops_unit = "GFLOP";
    char fl_s[32];
    if (has_data) {
        if (sel->result.gflops >= 1000.0) {
            snprintf(fl_s, sizeof(fl_s), "%.2f", sel->result.gflops / 1000.0);
            flops_unit = "TFLOP";
        } else {
            snprintf(fl_s, sizeof(fl_s), "%.1f", sel->result.gflops);
        }
    } else snprintf(fl_s, sizeof(fl_s), "--");
    kpis.push_back({"Compute", fl_s, flops_unit, has_data,
                    has_data && peak_fp32_gflops_ > 0
                        ? (float)(sel->result.gflops / peak_fp32_gflops_) : -1.0f,
                    C::ACCENT});

    char oc_s[32];
    if (has_profile) snprintf(oc_s, sizeof(oc_s), "%.1f", sel->result.achieved_occupancy * 100.0);
    else             snprintf(oc_s, sizeof(oc_s), "--");
    kpis.push_back({"Occup", oc_s, "%", has_profile,
                    has_profile ? (float)sel->result.achieved_occupancy : -1.0f,
                    C::ACCENT});

    char ipc_s[32];
    bool has_ipc = has_profile && sel->result.ipc > 0;
    if (has_ipc) snprintf(ipc_s, sizeof(ipc_s), "%.2f", sel->result.ipc);
    else         snprintf(ipc_s, sizeof(ipc_s), "--");
    kpis.push_back({"IPC", ipc_s, "", has_ipc,
                    has_ipc ? (float)(sel->result.ipc / 4.0) : -1.0f, C::ACCENT});

    char dr_s[32];
    double total_dram = has_profile ? sel->result.dram_read_gbps + sel->result.dram_write_gbps : 0;
    bool has_dram = total_dram > 0;
    if (has_dram) snprintf(dr_s, sizeof(dr_s), "%.1f", total_dram);
    else          snprintf(dr_s, sizeof(dr_s), "--");
    kpis.push_back({"DRAM", dr_s, "GB/s", has_dram,
                    has_dram && peak_mem_bw_gbs_ > 0
                        ? (float)(total_dram / peak_mem_bw_gbs_) : -1.0f, C::ACCENT});

    for (int i = 0; i < cards && i < (int)kpis.size(); i++) {
        const KPI& k = kpis[i];
        int cx = x + i * (card_w + 1);
        draw_box(cx, y, card_w, card_h, "", C::BORDER_DIM);
        uint32_t lfg = k.ok ? C::BODY : C::DIM;
        put_text_clipped(cx + 1, y, card_w - 2, k.label, lfg, 0, ATTR_BOLD);
        std::string vs = k.value;
        if (k.unit && *k.unit) { vs += " "; vs += k.unit; }
        uint32_t vfg = k.ok ? C::ACCENT : C::DIM;
        put_text_clipped(cx + 1, y + 1, card_w - 2, vs, vfg, 0, ATTR_BOLD);
        if (k.pct >= 0) {
            uint32_t bar = (k.pct < 0.33f) ? C::RED :
                           (k.pct < 0.66f) ? C::YELLOW : C::GREEN;
            draw_hbar(cx + 1, y + 2, card_w - 2, k.pct, bar, C::BORDER_DIM);
        } else if (k.ok) {
            put_text_clipped(cx + 1, y + 2, card_w - 2, " ", C::DIM);
        } else {
            put_text_clipped(cx + 1, y + 2, card_w - 2, "no data", C::DIM);
        }
    }
    row_out = y + card_h;
}

void Tui::render_panel(int x, int y, int w, int h) {
    if (h < 3) return;
    char hdr[128];
    snprintf(hdr, sizeof(hdr), "View: %s   [←/→ change]  [o/O sort: %s %s]",
        panel_name(panel_), sort_name(sort_col_), sort_desc_ ? "↓" : "↑");
    put_text_clipped(x, y, w, hdr, C::HEADER, 0, ATTR_BOLD);
    int py = y + 1;
    int ph = h - 1;
    if (ph < 2) return;

    switch (panel_) {
        case TuiPanel::WallVsGpu:         panel_wall_vs_gpu(x, py, w, ph); break;
        case TuiPanel::Throughput:        panel_throughput(x, py, w, ph);  break;
        case TuiPanel::Speedup:           panel_speedup(x, py, w, ph);     break;
        case TuiPanel::ProfilingCompare:  panel_profiling_compare(x, py, w, ph); break;
        case TuiPanel::Roofline:          panel_roofline(x, py, w, ph);    break;
        case TuiPanel::SubKernelTimeline: panel_subkernel_timeline(x, py, w, ph); break;
        case TuiPanel::TimingDist:        panel_timing_dist(x, py, w, ph); break;
        case TuiPanel::Scaling:           panel_scaling(x, py, w, ph);     break;
        default: break;
    }
}

struct BarEntry {
    std::string name;
    double wall_ms;
    double gpu_ms;
    double perf;
    double occupancy;
    double ipc;
    double dram_total;
    TuiDSL dsl;
};

static TuiDSL inline_detect_dsl(const arena::KernelDescriptor* d) {
    if (!d) return TuiDSL::CUDA;
    if (!d->uses_module()) return TuiDSL::CUB;
    if (!d->needs_compilation()) return TuiDSL::CUDA;
    const std::string src = d->source_path();
    if (src.find(".triton.") != std::string::npos) return TuiDSL::Triton;
    if (src.find(".cutile.") != std::string::npos) return TuiDSL::CuTile;
    if (src.find(".warp.")   != std::string::npos) return TuiDSL::Warp;
    return TuiDSL::CUDA;
}

static std::vector<BarEntry> gather_entries(const std::vector<TuiKernelState>& ks,
                                             bool matmul) {
    std::vector<BarEntry> out;
    for (const auto& k : ks) {
        if (!(k.has_run && k.result.success)) continue;
        BarEntry e;
        e.name = k.result.kernel_name;
        e.wall_ms = k.result.elapsed_ms;
        e.gpu_ms  = k.result.kernel_ms;
        e.perf    = matmul ? k.result.gflops : k.result.bandwidth_gbps;
        e.occupancy = k.result.achieved_occupancy * 100.0;
        e.ipc       = k.result.ipc;
        e.dram_total = k.result.dram_read_gbps + k.result.dram_write_gbps;
        e.dsl = inline_detect_dsl(k.descriptor);
        out.push_back(std::move(e));
    }
    return out;
}

void Tui::panel_wall_vs_gpu(int x, int y, int w, int h) {
    auto* ks = current_kernels();
    if (!ks) return;
    auto entries = gather_entries(*ks, is_matmul());
    if (entries.empty()) {
        put_text(x, y, "No data. Press r to run selected kernels.", C::DIM);
        return;
    }
    std::sort(entries.begin(), entries.end(),
        [](const BarEntry& a, const BarEntry& b) { return a.wall_ms < b.wall_ms; });

    double max_wall = 0;
    for (const auto& e : entries) max_wall = std::max(max_wall, e.wall_ms);

    int name_w = 0;
    for (const auto& e : entries) name_w = std::max(name_w, (int)e.name.size());
    name_w = std::min(name_w + 1, std::max(10, w / 3));
    int val_w = 18;
    int bar_w = w - name_w - val_w - 2;
    if (bar_w < 5) return;

    int n = std::min((int)entries.size(), h - 1);
    for (int i = 0; i < n; i++) {
        const auto& e = entries[i];
        int row = y + i;
        uint32_t name_c = dsl_color(e.dsl);
        put_text_clipped(x, row, name_w, e.name, name_c);
        float pct_wall = max_wall > 0 ? (float)(e.wall_ms / max_wall) : 0;
        float pct_gpu  = max_wall > 0 ? (float)(e.gpu_ms  / max_wall) : 0;
        for (int b = 0; b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2591, C::BORDER_DIM);
        int w_cells = (int)(pct_wall * bar_w);
        int g_cells = (int)(pct_gpu  * bar_w);
        for (int b = 0; b < w_cells && b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2588, C::ACCENT);
        for (int b = 0; b < g_cells && b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2588, C::CUTILE_CLR);

        char vb[64];
        double overhead = e.wall_ms > 0 ? (e.wall_ms - e.gpu_ms) / e.wall_ms * 100.0 : 0;
        snprintf(vb, sizeof(vb), " %s / %s  +%.0f%%",
            fmt_time((float)e.wall_ms).c_str(), fmt_time((float)e.gpu_ms).c_str(), overhead);
        put_text_clipped(x + name_w + 1 + bar_w, row, val_w + 1, vb, C::DIM);
    }
    if (n < (int)entries.size()) {
        char more[32]; snprintf(more, sizeof(more), "+ %d more ...", (int)entries.size() - n);
        put_text(x, y + n, more, C::DIM);
    }
    if (h > n + 1) {
        int ly = y + n + 1;
        put_cell(x, ly, 0x2588, C::ACCENT);
        put_text(x + 1, ly, " wall  ", C::DIM);
        put_cell(x + 8, ly, 0x2588, C::CUTILE_CLR);
        put_text(x + 9, ly, " gpu", C::DIM);
    }
}

void Tui::panel_throughput(int x, int y, int w, int h) {
    auto* ks = current_kernels();
    if (!ks) return;
    bool matmul = is_matmul();
    auto entries = gather_entries(*ks, matmul);
    if (entries.empty()) {
        put_text(x, y, "No data. Press r to run selected kernels.", C::DIM);
        return;
    }
    std::sort(entries.begin(), entries.end(),
        [](const BarEntry& a, const BarEntry& b) { return a.perf > b.perf; });

    double peak = matmul ? peak_fp32_gflops_ : peak_mem_bw_gbs_;
    double scale_max = peak;
    if (peak <= 0 || entries[0].perf > peak) scale_max = entries[0].perf * 1.1;

    int name_w = 0;
    for (const auto& e : entries) name_w = std::max(name_w, (int)e.name.size());
    name_w = std::min(name_w + 1, std::max(10, w / 3));
    int val_w = 18;
    int bar_w = w - name_w - val_w - 2;
    if (bar_w < 5) return;

    int n = std::min((int)entries.size(), h - 2);
    for (int i = 0; i < n; i++) {
        const auto& e = entries[i];
        int row = y + i;
        uint32_t name_c = dsl_color(e.dsl);
        put_text_clipped(x, row, name_w, e.name, name_c);
        for (int b = 0; b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2591, C::BORDER_DIM);
        float pct = scale_max > 0 ? (float)(e.perf / scale_max) : 0;
        int cells = (int)(pct * bar_w);
        uint32_t bar = (pct < 0.33f) ? C::RED : (pct < 0.66f) ? C::YELLOW : C::GREEN;
        for (int b = 0; b < cells && b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2588, bar);

        if (peak > 0 && peak < scale_max * 1.001) {
            int peak_col = (int)((peak / scale_max) * bar_w);
            if (peak_col < bar_w) put_cell(x + name_w + 1 + peak_col, row, 0x2503, C::RED);
        }

        char vb[64];
        double pct_peak = peak > 0 ? e.perf / peak * 100.0 : 0;
        snprintf(vb, sizeof(vb), " %.1f %s (%.0f%%)",
            e.perf, matmul ? "GFLOPS" : "GB/s", pct_peak);
        put_text_clipped(x + name_w + 1 + bar_w, row, val_w + 1, vb, C::DIM);
    }
    if (h > n + 1 && peak > 0) {
        char pk[128];
        snprintf(pk, sizeof(pk), "red bar = theoretical peak: %.0f %s",
            peak, matmul ? "GFLOPS" : "GB/s");
        put_text_clipped(x, y + n + 1, w, pk, C::DIM);
    }
}

void Tui::panel_speedup(int x, int y, int w, int h) {
    auto* ks = current_kernels();
    if (!ks) return;
    auto entries = gather_entries(*ks, is_matmul());
    if (entries.size() < 2) {
        put_text(x, y, "Need >=2 kernels with results.", C::DIM);
        return;
    }
    std::sort(entries.begin(), entries.end(),
        [](const BarEntry& a, const BarEntry& b) { return a.wall_ms < b.wall_ms; });
    double slowest = entries.back().wall_ms;

    int name_w = 0;
    for (const auto& e : entries) name_w = std::max(name_w, (int)e.name.size());
    name_w = std::min(name_w + 1, std::max(10, w / 3));
    int val_w = 18;
    int bar_w = w - name_w - val_w - 2;
    if (bar_w < 5) return;

    int n = std::min((int)entries.size(), h);
    for (int i = 0; i < n; i++) {
        const auto& e = entries[i];
        int row = y + i;
        uint32_t name_c = dsl_color(e.dsl);
        put_text_clipped(x, row, name_w, e.name, name_c);

        float pct = slowest > 0 ? (float)(e.wall_ms / slowest) : 0;
        for (int b = 0; b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2591, C::BORDER_DIM);
        int cells = (int)(pct * bar_w);
        uint32_t bar_col = (i == 0) ? C::GREEN : (pct < 0.5f ? C::ACCENT : C::YELLOW);
        for (int b = 0; b < cells && b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2588, bar_col);

        char vb[64];
        double sp = e.wall_ms > 0 ? slowest / e.wall_ms : 0;
        snprintf(vb, sizeof(vb), " %s  %.1fx", fmt_time((float)e.wall_ms).c_str(), sp);
        put_text_clipped(x + name_w + 1 + bar_w, row, val_w + 1, vb,
            (i == 0) ? C::GREEN : C::DIM, 0, (i == 0) ? ATTR_BOLD : 0);
    }
}

void Tui::panel_profiling_compare(int x, int y, int w, int h) {
    auto* ks = current_kernels();
    if (!ks) return;
    auto entries = gather_entries(*ks, is_matmul());
    entries.erase(std::remove_if(entries.begin(), entries.end(),
        [](const BarEntry& e) { return e.occupancy <= 0; }), entries.end());
    if (entries.empty()) {
        put_text(x, y, "No profiling data. Enable Profile [p] and re-run.", C::DIM);
        return;
    }
    std::sort(entries.begin(), entries.end(),
        [](const BarEntry& a, const BarEntry& b) { return a.occupancy > b.occupancy; });

    int name_w = 0;
    for (const auto& e : entries) name_w = std::max(name_w, (int)e.name.size());
    name_w = std::min(name_w + 1, std::max(10, w / 3));
    int metric_w = (w - name_w - 2) / 2;
    if (metric_w < 10) return;

    put_text_clipped(x + name_w + 2, y, metric_w, "Occupancy %", C::HEADER, 0, ATTR_BOLD);
    put_text_clipped(x + name_w + 2 + metric_w, y, metric_w, "IPC (of ~4)", C::HEADER, 0, ATTR_BOLD);

    int n = std::min((int)entries.size(), h - 1);
    for (int i = 0; i < n; i++) {
        const auto& e = entries[i];
        int row = y + 1 + i;
        put_text_clipped(x, row, name_w, e.name, dsl_color(e.dsl));

        int occ_bar_w = metric_w - 8;
        float occ_pct = std::min(1.0f, (float)e.occupancy / 100.0f);
        for (int b = 0; b < occ_bar_w; b++) put_cell(x + name_w + 2 + b, row, 0x2591, C::BORDER_DIM);
        int occ_cells = (int)(occ_pct * occ_bar_w);
        for (int b = 0; b < occ_cells; b++) put_cell(x + name_w + 2 + b, row, 0x2588, C::ACCENT);
        char ob[16]; snprintf(ob, sizeof(ob), " %5.1f", e.occupancy);
        put_text_clipped(x + name_w + 2 + occ_bar_w, row, 8, ob, C::BODY);

        int ipc_bar_w = metric_w - 8;
        float ipc_pct = std::min(1.0f, (float)(e.ipc / 4.0));
        int ix_x = x + name_w + 2 + metric_w;
        for (int b = 0; b < ipc_bar_w; b++) put_cell(ix_x + b, row, 0x2591, C::BORDER_DIM);
        int ipc_cells = (int)(ipc_pct * ipc_bar_w);
        for (int b = 0; b < ipc_cells; b++) put_cell(ix_x + b, row, 0x2588, C::YELLOW);
        char ib[16]; snprintf(ib, sizeof(ib), " %5.2f", e.ipc);
        put_text_clipped(ix_x + ipc_bar_w, row, 8, ib, C::BODY);
    }
}

void Tui::panel_roofline(int x, int y, int w, int h) {
    auto* ks = current_kernels();
    if (!ks) return;
    if (peak_fp32_gflops_ <= 0 || peak_mem_bw_gbs_ <= 0) {
        put_text(x, y, "Peak info unavailable.", C::DIM);
        return;
    }

    struct Pt { double ai; double gflops; TuiDSL dsl; std::string name; };
    std::vector<Pt> pts;
    for (const auto& k : *ks) {
        if (!(k.has_run && k.result.success)) continue;
        double dram = k.result.dram_read_gbps + k.result.dram_write_gbps;
        if (dram <= 0) continue;
        pts.push_back({k.result.gflops / dram, k.result.gflops,
            detect_dsl(k.descriptor), k.result.kernel_name});
    }
    if (pts.empty()) {
        put_text(x, y, "No roofline data. Enable Profile [p] and re-run.", C::DIM);
        return;
    }

    double ridge = peak_fp32_gflops_ / peak_mem_bw_gbs_;

    // Axis ranges snapped to decades, padded around data and the ridge.
    double ai_lo = pts[0].ai, ai_hi = pts[0].ai;
    double gf_lo = pts[0].gflops, gf_hi = pts[0].gflops;
    for (const auto& p : pts) {
        ai_lo = std::min(ai_lo, p.ai);
        ai_hi = std::max(ai_hi, p.ai);
        gf_lo = std::min(gf_lo, p.gflops);
        gf_hi = std::max(gf_hi, p.gflops);
    }
    double x_min = std::pow(10.0, std::floor(std::log10(std::min({ai_lo * 0.5, 0.1, ridge * 0.1}))));
    double x_max = std::pow(10.0, std::ceil (std::log10(std::max({ai_hi * 2.0, 100.0, ridge * 10.0}))));
    double y_min = std::pow(10.0, std::floor(std::log10(std::max(1.0, gf_lo * 0.5))));
    double y_max = std::pow(10.0, std::ceil (std::log10(std::max((double)peak_fp32_gflops_ * 1.2, gf_hi * 2.0))));

    int y_label_w = 6;
    int plot_x = x + y_label_w + 1;
    int plot_y = y;
    int plot_w = w - y_label_w - 1;
    int plot_h = h - 3;
    if (plot_w < 24 || plot_h < 6) {
        put_text(x, y, "Plot area too small - expand the terminal.", C::DIM);
        return;
    }

    auto xmap = [&](double v) {
        double t = (std::log10(std::max(v, x_min)) - std::log10(x_min))
                 / (std::log10(x_max) - std::log10(x_min));
        return plot_x + (int)std::round(t * (plot_w - 1));
    };
    auto ymap = [&](double v) {
        double t = (std::log10(std::max(v, y_min)) - std::log10(y_min))
                 / (std::log10(y_max) - std::log10(y_min));
        return plot_y + plot_h - 1 - (int)std::round(t * (plot_h - 1));
    };

    // Decade grid (very dim).
    for (double d = x_min; d <= x_max * 1.0001; d *= 10) {
        int col = xmap(d);
        if (col < plot_x || col >= plot_x + plot_w) continue;
        for (int j = 0; j < plot_h; j++) put_cell(col, plot_y + j, 0x2502, 0x1C1C1C);
    }
    for (double d = y_min; d <= y_max * 1.0001; d *= 10) {
        int row = ymap(d);
        if (row < plot_y || row >= plot_y + plot_h) continue;
        for (int i = 0; i < plot_w; i++) put_cell(plot_x + i, row, 0x2500, 0x1C1C1C);
    }

    // Axis frame.
    for (int j = 0; j < plot_h; j++) put_cell(plot_x - 1, plot_y + j, 0x2502, C::BORDER);
    for (int i = 0; i < plot_w; i++) put_cell(plot_x + i, plot_y + plot_h, 0x2500, C::BORDER);
    put_cell(plot_x - 1, plot_y + plot_h, 0x2514, C::BORDER);

    // Y decade ticks + labels.
    for (double d = y_min; d <= y_max * 1.0001; d *= 10) {
        int row = ymap(d);
        if (row < plot_y || row >= plot_y + plot_h) continue;
        put_cell(plot_x - 1, row, 0x2524, C::BORDER);
        char lb[16];
        if      (d >= 1e6)  snprintf(lb, sizeof(lb), "%.0fM", d / 1e6);
        else if (d >= 1000) snprintf(lb, sizeof(lb), "%.0fK", d / 1e3);
        else                snprintf(lb, sizeof(lb), "%g",    d);
        int lw = (int)strlen(lb);
        int lx = plot_x - 2 - lw;
        if (lx >= x) put_text(lx, row, lb, C::DIM);
    }

    // X decade ticks + labels.
    int x_tick_row = plot_y + plot_h;
    int x_label_row = plot_y + plot_h + 1;
    for (double d = x_min; d <= x_max * 1.0001; d *= 10) {
        int col = xmap(d);
        if (col < plot_x || col >= plot_x + plot_w) continue;
        put_cell(col, x_tick_row, 0x252C, C::BORDER);
        char lb[16];
        if      (d >= 1000) snprintf(lb, sizeof(lb), "%.0fK", d / 1e3);
        else if (d >= 1)    snprintf(lb, sizeof(lb), "%.0f",  d);
        else                snprintf(lb, sizeof(lb), "%g",    d);
        int lw = (int)strlen(lb);
        int lx = col - lw / 2;
        if (lx < plot_x) lx = plot_x;
        if (lx + lw > plot_x + plot_w) lx = plot_x + plot_w - lw;
        put_text(lx, x_label_row, lb, C::DIM);
    }

    // Roof: memory-bound on the left (diagonal on log-log), compute-bound on the right (flat).
    // Walk column-by-column so the line stays continuous on the cell grid.
    int prev_ry = -1;
    for (int i = 0; i < plot_w; i++) {
        double t = (double)i / (plot_w - 1);
        double ai = std::pow(10.0, std::log10(x_min) + t * (std::log10(x_max) - std::log10(x_min)));
        double roof = std::min((double)peak_fp32_gflops_, ai * peak_mem_bw_gbs_);
        int rx = plot_x + i;
        int ry = ymap(roof);
        if (ry < plot_y || ry >= plot_y + plot_h) { prev_ry = ry; continue; }
        uint32_t ch = (ai < ridge) ? 0x2571 /* ╱ */ : 0x2501 /* ━ */;
        put_cell(rx, ry, ch, C::YELLOW);
        if (prev_ry >= plot_y && prev_ry < plot_y + plot_h && std::abs(prev_ry - ry) > 1) {
            int lo = std::min(prev_ry, ry) + 1;
            int hi = std::max(prev_ry, ry);
            for (int r = lo; r < hi; r++) put_cell(rx, r, 0x2502, C::YELLOW);
        }
        prev_ry = ry;
    }

    // Ridge marker.
    int ridge_x = xmap(ridge);
    int ridge_y = ymap((double)peak_fp32_gflops_);
    if (ridge_x >= plot_x && ridge_x < plot_x + plot_w &&
        ridge_y >= plot_y && ridge_y < plot_y + plot_h) {
        put_cell(ridge_x, ridge_y, 0x25C6, C::YELLOW);  // ◆
    }

    // Points, then labels where space allows (prefer right side, flip if clipped).
    for (const auto& p : pts) {
        int px = xmap(p.ai);
        int py = ymap(p.gflops);
        if (px < plot_x || px >= plot_x + plot_w) continue;
        if (py < plot_y || py >= plot_y + plot_h) continue;
        put_cell(px, py, 0x25CF, dsl_color(p.dsl));

        std::string nm = p.name.size() > 12 ? p.name.substr(0, 12) : p.name;
        int right_x = px + 2;
        int left_x  = px - 1 - (int)nm.size();
        if (right_x + (int)nm.size() <= plot_x + plot_w) {
            put_text(right_x, py, nm, dsl_color(p.dsl));
        } else if (left_x >= plot_x) {
            put_text(left_x, py, nm, dsl_color(p.dsl));
        }
    }

    // X-axis title.
    int atx_row = plot_y + plot_h + 2;
    if (atx_row < y + h) {
        const char* axt = "Arithmetic Intensity (FLOP/Byte)";
        int lw = (int)strlen(axt);
        int lx = plot_x + (plot_w - lw) / 2;
        if (lx < plot_x) lx = plot_x;
        put_text(lx, atx_row, axt, C::DIM);
    }

    // Legend, top-right of plot.
    char pk[32], mb[32], rg[32];
    snprintf(pk, sizeof(pk), "peak  %.0f GF/s", peak_fp32_gflops_);
    snprintf(mb, sizeof(mb), "mem   %.0f GB/s", peak_mem_bw_gbs_);
    snprintf(rg, sizeof(rg), "ridge %.1f F/B",  ridge);
    int legend_w = std::max({(int)strlen(pk), (int)strlen(mb), (int)strlen(rg)});
    int lx = plot_x + plot_w - legend_w - 1;
    if (lx > plot_x + plot_w / 2) {
        put_text(lx, plot_y,     pk, C::YELLOW);
        put_text(lx, plot_y + 1, mb, C::YELLOW);
        put_text(lx, plot_y + 2, rg, C::DIM);
    }
}

void Tui::panel_subkernel_timeline(int x, int y, int w, int h) {
    const auto* sel = selected_kernel();
    if (!sel || !sel->has_run || !sel->result.success || sel->result.sub_kernels.empty()) {
        put_text(x, y, "Select a kernel with sub-kernel data (Enter on a row).", C::DIM);
        return;
    }
    const auto& sks = sel->result.sub_kernels;
    double total = 0;
    for (const auto& sk : sks) total += sk.duration_ms;

    char hdr[128];
    snprintf(hdr, sizeof(hdr), "%s  -  %zu kernels  total %s",
        sel->result.kernel_name.c_str(), sks.size(), fmt_time((float)total).c_str());
    put_text_clipped(x, y, w, hdr, C::HEADER, 0, ATTR_BOLD);

    int name_w = 22;
    int val_w  = 20;
    int bar_w  = w - name_w - val_w - 2;
    if (bar_w < 8) return;

    int n = std::min((int)sks.size(), h - 1);
    for (int i = 0; i < n; i++) {
        const auto& sk = sks[i];
        int row = y + 1 + i;
        std::string name = sk.name;
        if ((int)name.size() > name_w - 1)
            name = "..." + name.substr(name.size() - (name_w - 4));
        put_text_clipped(x, row, name_w, name, C::BODY);
        float pct = total > 0 ? (float)(sk.duration_ms / total) : 0;
        for (int b = 0; b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2591, C::BORDER_DIM);
        int cells = (int)(pct * bar_w);
        for (int b = 0; b < cells && b < bar_w; b++) put_cell(x + name_w + 1 + b, row, 0x2588, C::ACCENT);
        char vb[64];
        snprintf(vb, sizeof(vb), " %s  (%.0f%%)  %dr",
            fmt_time((float)sk.duration_ms).c_str(), pct * 100, sk.registers);
        put_text_clipped(x + name_w + 1 + bar_w, row, val_w + 1, vb, C::DIM);
    }
}

void Tui::panel_timing_dist(int x, int y, int w, int h) {
    const auto* sel = selected_kernel();
    if (!sel || !sel->has_run || !sel->result.success || sel->result.all_times_ms.empty()) {
        put_text(x, y, "Select a kernel with per-run timings.", C::DIM);
        return;
    }
    const auto& ts = sel->result.all_times_ms;
    double mn = *std::min_element(ts.begin(), ts.end());
    double mx = *std::max_element(ts.begin(), ts.end());
    double med = sel->result.elapsed_ms;
    double sum = 0;
    for (float v : ts) sum += v;
    double mean = sum / ts.size();
    double var = 0;
    for (float v : ts) var += (v - mean) * (v - mean);
    double sd = std::sqrt(var / ts.size());

    char hdr[128];
    snprintf(hdr, sizeof(hdr), "%s  -  %zu runs  min %s  max %s  med %s  stddev %s",
        sel->result.kernel_name.c_str(), ts.size(),
        fmt_time((float)mn).c_str(), fmt_time((float)mx).c_str(),
        fmt_time((float)med).c_str(), fmt_time((float)sd).c_str());
    put_text_clipped(x, y, w, hdr, C::HEADER, 0, ATTR_BOLD);

    int bar_w = w - 2;
    int bar_h = std::max(3, h - 3);
    if (bar_w < 10 || bar_h < 3) return;

    int bx = x + 1;
    int by = y + 1;

    for (int j = 0; j < bar_h; j++) put_cell(bx - 1, by + j, 0x2502, C::BORDER_DIM);
    for (int i = 0; i < bar_w; i++) put_cell(bx + i, by + bar_h, 0x2500, C::BORDER_DIM);
    put_cell(bx - 1, by + bar_h, 0x2514, C::BORDER_DIM);

    double span = std::max(1e-9, mx - mn);
    for (size_t i = 0; i < ts.size(); i++) {
        int px = bx + (int)((double)i / (ts.size() - (ts.size() > 1 ? 1 : 0)) * (bar_w - 1));
        double norm = (ts[i] - mn) / span;
        int h_cells = (int)(norm * (bar_h - 1));
        int py = by + (bar_h - 1) - h_cells;
        put_cell(px, py, 0x25CF, C::ACCENT);
    }
    double nmed = (med - mn) / span;
    int ny = by + (bar_h - 1) - (int)(nmed * (bar_h - 1));
    for (int i = 0; i < bar_w; i += 2) put_cell(bx + i, ny, 0x2508, C::YELLOW);

    put_text(x, by, fmt_time((float)mx).c_str(), C::DIM);
    put_text(x, by + bar_h - 1, fmt_time((float)mn).c_str(), C::DIM);
}

void Tui::panel_scaling(int x, int y, int w, int h) {
    auto cat_it = scaling_history_.find(current_category_);
    if (cat_it == scaling_history_.end() || cat_it->second.empty()) {
        put_text(x, y, "No scaling data. Run 'sweep' [s] across multiple sizes.", C::DIM);
        return;
    }
    bool has_multi = false;
    for (const auto& [name, hist] : cat_it->second) {
        if (hist.size() > 1) { has_multi = true; break; }
    }
    if (!has_multi) {
        put_text(x, y, "Need >=2 sizes per kernel. Press [s] for sweep or adjust [ ] size.", C::DIM);
        return;
    }

    bool matmul = is_matmul();
    std::set<int> sizes;
    for (const auto& [name, hist] : cat_it->second)
        for (const auto& e : hist) sizes.insert(e.problem_size);
    std::vector<int> szv(sizes.begin(), sizes.end());

    int name_w = 0;
    for (const auto& [name, hist] : cat_it->second) name_w = std::max(name_w, (int)name.size());
    name_w = std::min(name_w + 1, std::max(12, w / 4));

    int col_w = 10;
    int max_cols = (w - name_w - 1) / col_w;
    int n_cols = std::min(max_cols, (int)szv.size());
    if (n_cols < 1) return;

    put_text_clipped(x, y, name_w, "Kernel", C::HEADER, 0, ATTR_BOLD);
    for (int c = 0; c < n_cols; c++) {
        char hdr[16]; snprintf(hdr, sizeof(hdr), "%d", szv[c]);
        int cx = x + name_w + c * col_w;
        int pad = col_w - (int)strlen(hdr);
        put_text_clipped(cx + pad, y, col_w - pad, hdr, C::HEADER);
    }

    int row = y + 1;
    for (const auto& [name, hist] : cat_it->second) {
        if (row >= y + h) break;
        put_text_clipped(x, row, name_w, name, C::BODY);
        for (int c = 0; c < n_cols; c++) {
            int sz = szv[c];
            bool found = false;
            double val = 0;
            for (const auto& e : hist) {
                if (e.problem_size == sz && e.result.success) {
                    val = matmul ? e.result.gflops : e.result.bandwidth_gbps;
                    found = true;
                    break;
                }
            }
            char vb[16];
            if (found) snprintf(vb, sizeof(vb), "%.1f", val);
            else       snprintf(vb, sizeof(vb), "-");
            int cx = x + name_w + c * col_w;
            int pad = col_w - (int)strlen(vb);
            put_text_clipped(cx + pad, row, col_w - pad, vb,
                found ? C::BODY : C::DIM);
        }
        row++;
    }

    if (row < y + h)
        put_text_clipped(x, row, w, matmul ? "(GFLOPS)" : "(GB/s)", C::DIM);
}

void Tui::render_detail(int x, int y, int w, int h) {
    const auto* sel = selected_kernel();
    std::string title = sel && sel->descriptor ? sel->descriptor->name() : "Detail";
    if (title.size() > (size_t)w - 4) title = title.substr(0, w - 7) + "...";
    draw_box(x, y, w, h, title);

    int ix = x + 2;
    int iy = y + 1;
    int iw = w - 4;
    if (!sel || !sel->descriptor) {
        put_text(ix, iy, "Press ↓/↑ then Enter", C::DIM);
        return;
    }

    auto* desc = sel->descriptor;
    const auto& r = sel->result;
    bool has_data = sel->has_run;
    bool has_counters = has_data && r.achieved_occupancy > 0;

    put_text(ix, iy, "Compilation", C::ACCENT, 0, ATTR_BOLD);
    iy++;
    TuiDSL dsl = detect_dsl(desc);
    draw_badge(ix, iy, dsl);
    put_text(ix + 5, iy, desc->category(), C::BODY);
    iy++;
    if (desc->needs_compilation()) {
        std::string src = desc->source_path();
        if ((int)src.size() > iw - 3)
            src = "..." + src.substr(src.size() - (iw - 6));
        put_text_clipped(ix, iy++, iw, src, C::DIM);
    } else {
        put_text_clipped(ix, iy++, iw, "built-in", C::DIM);
    }
    if (has_data && desc->needs_compilation()) {
        if (r.cache_hit) put_text(ix, iy++, "cache: HIT", C::GREEN);
        else             put_text(ix, iy++, "cache: MISS", C::YELLOW);
        char tb[32];
        snprintf(tb, sizeof(tb), "compile: %s",
            r.compile_time_ms > 0 ? fmt_time(r.compile_time_ms).c_str() : "<1 ms");
        put_text_clipped(ix, iy++, iw, tb, C::DIM);
    } else {
        put_text(ix, iy++, "cache: n/a (native)", C::DIM);
    }
    iy++;
    if (iy >= y + h - 2) return;

    put_text(ix, iy, "Hardware Counters", C::ACCENT, 0, ATTR_BOLD);
    iy++;
    if (has_data && r.registers_per_thread > 0) {
        char b[32]; snprintf(b, sizeof(b), "Regs/thr: %d", r.registers_per_thread);
        put_text_clipped(ix, iy++, iw, b, C::BODY);
    } else put_text(ix, iy++, "Regs/thr: --", C::DIM);
    if (has_data && r.shared_memory_bytes > 0) {
        std::string b = "SHMem:   " + fmt_bytes(r.shared_memory_bytes);
        put_text_clipped(ix, iy++, iw, b, C::BODY);
    } else put_text(ix, iy++, "SHMem:   --", C::DIM);

    if (iy >= y + h - 2) return;
    if (has_counters) {
        put_text(ix, iy, "Occupancy", C::BODY);
        int bw = std::max(5, iw - 8);
        if (bw > 0 && iy + 1 < y + h) {
            float pct = (float)r.achieved_occupancy;
            uint32_t col = (pct < 0.33f) ? C::RED : (pct < 0.66f) ? C::YELLOW : C::GREEN;
            draw_hbar(ix, iy + 1, bw, pct, col, C::BORDER_DIM);
            char b[16]; snprintf(b, sizeof(b), "%5.1f%%", pct * 100.0);
            put_text_clipped(ix + bw + 1, iy + 1, 8, b, C::BODY);
        }
        iy += 2;
        if (r.ipc > 0 && iy + 1 < y + h) {
            put_text(ix, iy, "IPC", C::BODY);
            float pct = std::min(1.0f, (float)(r.ipc / 4.0));
            uint32_t col = (pct < 0.33f) ? C::RED : (pct < 0.66f) ? C::YELLOW : C::GREEN;
            int bw = std::max(5, iw - 8);
            draw_hbar(ix, iy + 1, bw, pct, col, C::BORDER_DIM);
            char b[16]; snprintf(b, sizeof(b), "%5.2f", r.ipc);
            put_text_clipped(ix + bw + 1, iy + 1, 8, b, C::BODY);
            iy += 2;
        }
        double dram = r.dram_read_gbps + r.dram_write_gbps;
        if (dram > 0 && iy + 1 < y + h) {
            put_text(ix, iy, "DRAM BW", C::BODY);
            float pct = peak_mem_bw_gbs_ > 0
                ? std::min(1.0f, (float)(dram / peak_mem_bw_gbs_)) : 0;
            uint32_t col = (pct < 0.33f) ? C::RED : (pct < 0.66f) ? C::YELLOW : C::GREEN;
            int bw = std::max(5, iw - 12);
            draw_hbar(ix, iy + 1, bw, pct, col, C::BORDER_DIM);
            char b[16]; snprintf(b, sizeof(b), "%5.1f GB/s", dram);
            put_text_clipped(ix + bw + 1, iy + 1, 12, b, C::BODY);
            iy += 2;
            char rw[64];
            snprintf(rw, sizeof(rw), "R: %.1f  W: %.1f GB/s",
                r.dram_read_gbps, r.dram_write_gbps);
            put_text_clipped(ix, iy++, iw, rw, C::DIM);
        }
    } else {
        put_text(ix, iy++, "(enable Profile [p])", C::DIM);
    }

    iy++;
    if (iy >= y + h - 2) return;

    put_text(ix, iy, "Verification", C::ACCENT, 0, ATTR_BOLD);
    iy++;
    if (has_data && r.success) {
        if (r.verified) put_text(ix, iy++, "PASS", C::GREEN, 0, ATTR_BOLD);
        else            put_text(ix, iy++, "FAIL (numeric)", C::RED, 0, ATTR_BOLD);
    } else if (has_data) put_text(ix, iy++, "EXECUTION FAILED", C::RED, 0, ATTR_BOLD);
    else                 put_text(ix, iy++, "not run", C::DIM);

    iy++;
    if (iy >= y + h - 2) return;

    put_text(ix, iy, "Launch Config", C::ACCENT, 0, ATTR_BOLD);
    iy++;
    if (has_data && r.success) {
        char b[96];
        snprintf(b, sizeof(b), "Grid:  (%u,%u,%u)", r.grid_x, r.grid_y, r.grid_z);
        put_text_clipped(ix, iy++, iw, b, C::BODY);
        snprintf(b, sizeof(b), "Block: (%u,%u,%u)", r.block_x, r.block_y, r.block_z);
        put_text_clipped(ix, iy++, iw, b, C::BODY);
        unsigned long long tt = (unsigned long long)r.grid_x * r.grid_y * r.grid_z *
                                (unsigned long long)r.block_x * r.block_y * r.block_z;
        snprintf(b, sizeof(b), "Threads: %llu", tt);
        put_text_clipped(ix, iy++, iw, b, C::BODY);
        if (r.shared_mem_bytes > 0) {
            snprintf(b, sizeof(b), "SHMem dyn: %u B", r.shared_mem_bytes);
            put_text_clipped(ix, iy++, iw, b, C::BODY);
        }
    } else put_text(ix, iy++, "(not run)", C::DIM);

    if (has_data && r.success && !r.sub_kernels.empty() && iy < y + h - 2) {
        iy++;
        char b[32]; snprintf(b, sizeof(b), "Sub-Kernels (%zu)", r.sub_kernels.size());
        put_text(ix, iy++, b, C::ACCENT, 0, ATTR_BOLD);
        for (const auto& sk : r.sub_kernels) {
            if (iy >= y + h - 2) break;
            char line[128];
            snprintf(line, sizeof(line), "%s  %dr  %dB",
                fmt_time((float)sk.duration_ms).c_str(),
                sk.registers, sk.shared_memory);
            put_text_clipped(ix, iy++, iw, line, C::BODY);
        }
    }
    if (has_data && !r.success && !r.error.empty() && iy < y + h - 2) {
        iy++;
        put_text(ix, iy++, "Error:", C::RED, 0, ATTR_BOLD);
        std::string err = r.error;
        while (iy < y + h - 1 && !err.empty()) {
            int chunk = std::min((int)err.size(), iw);
            put_text_clipped(ix, iy++, iw, err.substr(0, chunk), C::RED);
            err.erase(0, chunk);
        }
    }
}

void Tui::render_log(int x, int y, int w, int h) {
    std::string title = log_collapsed_ ? "Event Log (collapsed, [l] to expand)" : "Event Log";
    draw_box(x, y, w, h, title);
    if (log_collapsed_) return;

    int ix = x + 1;
    int iy = y + 1;
    int iw = w - 2;
    int ih = h - 2;
    if (ih < 1) return;

    int n = (int)log_entries_.size();
    int start = std::max(0, n - ih);
    for (int i = 0; i < ih && start + i < n; i++) {
        const auto& e = log_entries_[start + i];
        const char* prefix; uint32_t color;
        switch (e.level) {
            case TuiLogEntry::INFO:      prefix = "[INFO]   "; color = C::DIM; break;
            case TuiLogEntry::WARN:      prefix = "[WARN]   "; color = C::YELLOW; break;
            case TuiLogEntry::ERR:       prefix = "[ERROR]  "; color = C::RED; break;
            case TuiLogEntry::COMPILE:   prefix = "[COMPILE]"; color = C::CUTILE_CLR; break;
            case TuiLogEntry::BENCHMARK: prefix = "[BENCH]  "; color = C::ACCENT; break;
            case TuiLogEntry::PROFILE:   prefix = "[PROFILE]"; color = C::WARP_CLR; break;
            default: prefix = "[???]    "; color = C::BODY; break;
        }
        int len = put_text_clipped(ix, iy + i, 10, prefix, color);
        put_text_clipped(ix + len + 1, iy + i, iw - len - 1, e.message, C::BODY);
    }
}

void Tui::render_footer(int y) {
    fill_rect(0, y, term_w_, 1, ' ', 0, C::BG_HDR);
    const char* hints =
        " [r] run  [s] sweep  [c] cancel  [Space] toggle  [Tab] cat  [v] view  "
        "[o] sort  [p] profile  [e] export  [?] help  [q] quit";
    put_text_clipped(0, y, term_w_, hints, C::DIM, C::BG_HDR);
}

void Tui::render_help_overlay() {
    int w = std::min(70, term_w_ - 4);
    int h = std::min(30, term_h_ - 4);
    int x = (term_w_ - w) / 2;
    int y = (term_h_ - h) / 2;
    fill_rect(x, y, w, h, ' ', C::BODY, 0x0D0D0D);
    draw_box(x, y, w, h, "Keybindings (press ? or Esc to close)", 0, true);

    struct K { const char* key; const char* desc; };
    std::vector<K> keys = {
        {"Navigation", ""},
        {"j/k  ↑/↓",        "move cursor in kernel list"},
        {"g / G",            "jump to top / bottom"},
        {"PgUp / PgDn",     "page up / down (10)"},
        {"1-9",              "jump to category by index"},
        {"Tab  Shift-Tab",   "next / prev category"},
        {"",""},
        {"Selection & runs", ""},
        {"Space",            "toggle kernel selected (checkbox)"},
        {"Enter",            "focus kernel for detail panel (toggle)"},
        {"Esc",              "clear detail focus"},
        {"a / A",             "select all / deselect all"},
        {"r",                "run selected kernels"},
        {"s",                "sweep (run across multiple sizes)"},
        {"c",                "cancel running benchmark"},
        {"R",                "reset results (clear data)"},
        {"C",                "clear compile cache"},
        {"e",                "export results to CSV"},
        {"",""},
        {"View & sort", ""},
        {"v / V  or ←/→",   "cycle visualization panel"},
        {"o",                "cycle sort column"},
        {"O",                "reverse sort direction"},
        {"l",                "toggle log panel"},
        {"",""},
        {"Problem size & timing", ""},
        {"[  ]",              "decrease / increase size (1.25x)"},
        {"{  }",              "decrease / increase size (2x)"},
        {"-  +",              "warmup runs -/+"},
        {",  .",              "benchmark runs -/+"},
        {"p",                "toggle hardware profile (occupancy/IPC)"},
        {"x",                "toggle lock-square (matmul)"},
    };

    int iy = y + 2;
    for (const auto& k : keys) {
        if (iy >= y + h - 1) break;
        if (k.desc[0] == '\0') {
            if (k.key[0]) put_text(x + 2, iy, k.key, C::ACCENT, 0x0D0D0D, ATTR_BOLD);
            iy++;
            continue;
        }
        put_text_clipped(x + 3, iy, 18, k.key, C::YELLOW, 0x0D0D0D, ATTR_BOLD);
        put_text_clipped(x + 22, iy, w - 24, k.desc, C::BODY, 0x0D0D0D);
        iy++;
    }
}

int run_tui(arena::Runner& runner) {
    Tui tui(runner);
    tui.run();
    return 0;
}

}  // namespace frontend

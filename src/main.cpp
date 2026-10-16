#include <iostream>
#include <cstring>

#include "arena/paths.hpp"
#include "arena/device/context.hpp"
#include "arena/device/kernel_loader.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/cuda_backend.hpp"
#include "arena/compilers/triton_backend.hpp"
#include "arena/compilers/cutile_backend.hpp"
#include "arena/compilers/warp_backend.hpp"
#include "arena/measurement/profiler.hpp"
#include "arena/measurement/power.hpp"
#include "arena/runner.hpp"
#include "arena/cli_args.hpp"
#include "arena/logger.hpp"
#include "frontend/cli.hpp"
#include "arena/manifest/manifest_kernel.hpp"

#include <filesystem>
#include <optional>
#include <vector>

#ifdef ARENA_GUI_ENABLED
#include "frontend/gui.hpp"
#endif

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n"
#ifdef ARENA_GUI_ENABLED
              << "  (no options)              Launch the graphical interface\n"
              << "  --gui                     Launch the graphical interface\n"
              << "  --gui --open <dir>        ... with every kernel manifest under dir loaded\n"
#endif
              << "  --help, -h                Show this help message\n\n";
    frontend::print_cli_usage(program);
    std::cout <<
        "\nMeasurement:\n"
        "  --timing-mode <v1|v2>     How each timed launch is bracketed, for the whole\n"
        "                            process. v1 (default) queues the start event right\n"
        "                            behind the untimed reset and L2 scrub; such an event\n"
        "                            lands on a 2.048 us grid on some GPUs, so small ops\n"
        "                            read as 4.096, 6.144, 8.192 us. v2 waits for the\n"
        "                            reset first (untimed), so the start event keeps its\n"
        "                            full resolution and op_ms is measured the way\n"
        "                            launch_floor_ms is. Small-shape op_ms differs between\n"
        "                            the two: compare runs of the same mode only\n"
        "  --lock-clocks note        A lock is a ceiling, not a floor. Locking above the\n"
        "                            clock the board sustains does not raise it: the\n"
        "                            boost limit, power and thermal limits still hold\n"
        "                            it down.\n"
        "                            environment.clock_lock.achieved_mhz reports the mean\n"
        "                            clock of the timed runs, and a warning is logged\n"
        "                            when it is more than 3% below the request\n";
}

int main(int argc, char** argv) {
    // Process-wide, so handled here like --lock-clocks, and taken out of argv
    // before the CLI parser sees it.
    std::optional<arena::TimingMode> timing_mode;
    if (auto tm = arena::cli::take_timing_mode(argc, argv)) {
        timing_mode = arena::cli::parse_timing_mode(*tm);
        if (!timing_mode) {
            std::cerr << "Error: unknown --timing-mode '" << *tm << "' (expected v1 or v2)\n";
            return static_cast<int>(frontend::CliExit::UsageError);
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // --open <dir>, repeatable: folders of kernel manifests to load next to
    // the built-in kernels. Relative paths are from the current directory, or
    // else the repo root.
    std::vector<std::string> open_dirs;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--open") == 0) {
            open_dirs.push_back(arena::resolve_user_path(argv[++i], ARENA_SOURCE_DIR));
        }
    }

    const bool cli_mode = frontend::wants_cli(argc, argv);
    if (cli_mode && !frontend::validate_cli_args(argc, argv)) {
        return static_cast<int>(frontend::CliExit::UsageError);
    }

#ifndef ARENA_GUI_ENABLED
    if (!cli_mode) {
        std::cerr << "Error: this binary was built with BUILD_GUI=OFF.\n"
                  << "Use headless mode (--list / --run), or rebuild with "
                     "-DBUILD_GUI=ON for the GUI.\n\n";
        frontend::print_cli_usage(argv[0]);
        return 3;
    }
#endif

    // When the payload goes to stdout, stdout must carry the payload alone.
    // The file sink keeps the full log either way.
    const bool quiet = frontend::cli_writes_to_stdout(argc, argv);
    arena::init_logging(!quiet);

    try {
        spdlog::info("Initializing CUDA context ...");
        arena::Context ctx(0);

        spdlog::info("Setting up kernel compiler (kernel dir: {}) ...", ARENA_KERNEL_DIR);
        arena::KernelLoader loader;
        arena::KernelCompiler compiler("kernels");
        compiler.set_target_arch(ctx.arch());
        compiler.register_compiler(".cu",
            std::make_unique<arena::CudaBackend>(ARENA_KERNEL_DIR));
        compiler.register_compiler(".triton.py",
            std::make_unique<arena::TritonBackend>(ARENA_KERNEL_DIR));
        compiler.register_compiler(".cutile.py",
            std::make_unique<arena::CuTileBackend>(ARENA_KERNEL_DIR));
        compiler.register_compiler(".warp.py",
            std::make_unique<arena::WarpBackend>(ARENA_KERNEL_DIR));

        arena::Profiler profiler;
        arena::PowerMonitor power(0);

        // Locked before anything runs; PowerMonitor releases it on every
        // way out, including Ctrl-C.
        for (int i = 1; i + 1 < argc; i++) {
            if (strcmp(argv[i], "--lock-clocks") == 0) {
                if (auto mhz = arena::cli::parse_lock_clocks(argv[i + 1])) power.lock_clocks(*mhz);
            }
        }
        arena::Runner runner(ctx, loader, compiler, profiler, power);
        if (timing_mode) runner.set_timing_mode(timing_mode);

        if (cli_mode) {
            // The GUI loads these itself so it can log them in its panel.
            for (const auto& dir : open_dirs) {
                std::vector<std::string> taken;
                for (auto* k : runner.get_all_kernels()) taken.push_back(k->name());
                auto scan = arena::scan_manifests(dir, taken);
                for (const auto& e : scan.errors) {
                    spdlog::warn("--open: {}", e);
                    std::cerr << "--open: " << e << "\n";
                }
                if (scan.kernels.empty() && !std::filesystem::is_directory(dir)) return 2;
                for (auto& k : scan.kernels) {
                    const std::string file = k->descriptor_file();
                    arena::KernelRegistry::instance().register_kernel(std::move(k), file.c_str());
                }
                spdlog::info("--open {}: {} kernel(s)", dir, scan.kernels.size());
            }
        }

        auto categories = runner.get_categories();
        auto all_kernels = runner.get_all_kernels();
        spdlog::info("Registered {} kernels across {} categories",
            all_kernels.size(), categories.size());
        for (const auto& cat : categories) {
            auto kernels = runner.get_kernels_by_category(cat);
            spdlog::info("  {} - {} kernels", cat, kernels.size());
        }

        if (cli_mode) {
            return frontend::run_cli(runner, argc, argv);
        }

#ifdef ARENA_GUI_ENABLED
        spdlog::info("Starting GUI mode");
        return frontend::run_gui(runner, open_dirs);
#else
        return 3;
#endif

    } catch (const std::exception& e) {
        spdlog::error("Fatal: {}", e.what());
        return 2;
    }
}

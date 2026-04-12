#include <iostream>
#include <cstring>

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/cuda_compiler.hpp"
#include "arena/compilers/triton_compiler.hpp"
#include "arena/compilers/cutile_compiler.hpp"
#include "arena/compilers/warp_compiler.hpp"
#include "arena/benchmark.hpp"
#include "arena/profiler.hpp"
#include "arena/runner.hpp"
#include "arena/logger.hpp"
#include "frontend/cli.hpp"

#ifdef ARENA_GUI_ENABLED
#include "frontend/gui.hpp"
#endif

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --cli       Run in command-line mode\n"
#ifdef ARENA_GUI_ENABLED
              << "  --gui       Run with graphical interface (default)\n"
#endif
              << "  --help      Show this help message\n";
}

int main(int argc, char** argv) {
    bool use_gui = false;
    bool use_cli = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--cli") == 0) {
            use_cli = true;
        } else if (strcmp(argv[i], "--gui") == 0) {
            use_gui = true;
        }
    }

#ifdef ARENA_GUI_ENABLED
    if (!use_cli && !use_gui) {
        use_gui = true;
    }
#else
    use_cli = true;
#endif

    if (use_gui && use_cli) {
        std::cerr << "Error: Cannot use both --cli and --gui\n";
        return 1;
    }

    arena::init_logging();

    try {
        spdlog::info("Initializing CUDA context ...");
        arena::Context ctx(0);

        spdlog::info("Setting up kernel compiler (kernel dir: {}) ...", ARENA_KERNEL_DIR);
        arena::KernelLoader loader;
        arena::KernelCompiler compiler("kernels");
        compiler.register_compiler(".cu",
            std::make_unique<arena::CudaCompiler>(ARENA_KERNEL_DIR));
        compiler.register_compiler(".triton.py",
            std::make_unique<arena::TritonCompiler>(ARENA_KERNEL_DIR));
        compiler.register_compiler(".cutile.py",
            std::make_unique<arena::CuTileCompiler>(ARENA_KERNEL_DIR));
        compiler.register_compiler(".warp.py",
            std::make_unique<arena::WarpCompiler>(ARENA_KERNEL_DIR));

        arena::Benchmark benchmark;
        arena::Profiler profiler;
        arena::Runner runner(ctx, loader, compiler, benchmark, profiler);

        auto categories = runner.get_categories();
        auto all_kernels = runner.get_all_kernels();
        spdlog::info("Registered {} kernels across {} categories",
            all_kernels.size(), categories.size());
        for (const auto& cat : categories) {
            auto kernels = runner.get_kernels_by_category(cat);
            spdlog::info("  {} - {} kernels", cat, kernels.size());
        }

        spdlog::info("Starting {} mode", use_gui ? "GUI" : "CLI");

#ifdef ARENA_GUI_ENABLED
        if (use_gui) {
            return frontend::run_gui(runner);
        }
#endif
        return frontend::run_cli(runner);

    } catch (const std::exception& e) {
        spdlog::error("Fatal: {}", e.what());
        return 1;
    }
}

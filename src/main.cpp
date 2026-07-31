#include <iostream>
#include <cstring>

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/cuda_compiler.hpp"
#include "arena/compilers/triton_compiler.hpp"
#include "arena/compilers/cutile_compiler.hpp"
#include "arena/compilers/warp_compiler.hpp"
#include "arena/profiler.hpp"
#include "arena/power.hpp"
#include "arena/runner.hpp"
#include "arena/logger.hpp"

#ifdef ARENA_GUI_ENABLED
#include "frontend/gui.hpp"
#endif

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --gui       Run with graphical interface (default)\n"
              << "  --help      Show this help message\n";
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

#ifndef ARENA_GUI_ENABLED
    std::cerr << "Error: this binary was built with BUILD_GUI=OFF and has no frontend.\n"
              << "Rebuild with -DBUILD_GUI=ON.\n";
    return 1;
#else
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

        arena::Profiler profiler;
        arena::PowerMonitor power(0);
        arena::Runner runner(ctx, loader, compiler, profiler, power);

        auto categories = runner.get_categories();
        auto all_kernels = runner.get_all_kernels();
        spdlog::info("Registered {} kernels across {} categories",
            all_kernels.size(), categories.size());
        for (const auto& cat : categories) {
            auto kernels = runner.get_kernels_by_category(cat);
            spdlog::info("  {} - {} kernels", cat, kernels.size());
        }

        spdlog::info("Starting GUI mode");
        return frontend::run_gui(runner);

    } catch (const std::exception& e) {
        spdlog::error("Fatal: {}", e.what());
        return 1;
    }
#endif
}

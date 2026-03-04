#include <iostream>
#include <cstring>

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include "arena/profiler.hpp"
#include "arena/benchmark.hpp"
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
        arena::Context ctx(0);
        arena::KernelLoader loader;
        arena::Profiler profiler;
        arena::Benchmark benchmark(ctx, loader, profiler);

#ifdef ARENA_GUI_ENABLED
        if (use_gui) {
            return frontend::run_gui(benchmark);
        }
#endif
        return frontend::run_cli(benchmark);

    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }
}

#include "frontend/cli.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace frontend {

Cli::Cli(arena::Benchmark& benchmark)
    : benchmark_(benchmark) {
    config_.matrix_size = 1024;
    config_.warmup_runs = 10;
    refresh_kernels();
}

void Cli::run() {
    print_banner();
    print_help();

    std::string line;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, line)) break;

        line = trim(line);
        if (line.empty()) continue;

        if (!execute(line)) break;
    }

    std::cout << "Goodbye!\n";
}

void Cli::print_banner() {
    std::cout << R"(
   ╔═══════════════════════════════════════════════════════════╗
   ║       GPGPU ARENA - Kernel Benchmarking Platform          ║
   ╚═══════════════════════════════════════════════════════════╝
)" << std::endl;
    const auto& ctx = benchmark_.context();
    std::cout << "GPU: " << ctx.device_name() << "\n";
    std::cout << "Compute: SM " << ctx.compute_capability_major()
              << "." << ctx.compute_capability_minor()
              << " | Memory: " << (ctx.total_memory() / (1024 * 1024)) << " MB\n";
}

void Cli::print_help() {
    std::cout << "\nCommands:\n";
    std::cout << "  list              List available kernels\n";
    std::cout << "  run <name|all>    Run benchmark for kernel or all kernels\n";
    std::cout << "  results           Show benchmark results\n";
    std::cout << "  set size <n>      Set matrix size (default: 1024)\n";
    std::cout << "  set warmup <n>    Set warmup runs (default: 10)\n";
    std::cout << "  refresh           Rescan kernels directory\n";
    std::cout << "  help              Show this help\n";
    std::cout << "  quit              Exit\n";
}

bool Cli::execute(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

    if (cmd == "quit" || cmd == "exit" || cmd == "q") {
        return false;
    } else if (cmd == "help" || cmd == "h" || cmd == "?") {
        print_help();
    } else if (cmd == "list" || cmd == "ls") {
        cmd_list();
    } else if (cmd == "run") {
        std::string arg;
        iss >> arg;
        cmd_run(arg);
    } else if (cmd == "results" || cmd == "res") {
        cmd_results();
    } else if (cmd == "set") {
        std::string what;
        int value;
        iss >> what >> value;
        cmd_set(what, value);
    } else if (cmd == "refresh") {
        refresh_kernels();
        std::cout << "Found " << kernels_.size() << " kernel(s).\n";
    } else {
        std::cout << "Unknown command: " << cmd << ". Type 'help' for commands.\n";
    }

    return true;
}

void Cli::cmd_list() {
    if (kernels_.empty()) {
        std::cout << "No kernels found. Run 'refresh' to rescan.\n";
        return;
    }

    std::cout << "\nAvailable kernels:\n";
    for (size_t i = 0; i < kernels_.size(); i++) {
        std::cout << "  [" << i << "] " << kernels_[i].name;
        if (results_.count(kernels_[i].name)) {
            auto& r = results_[kernels_[i].name];
            if (r.success) {
                std::cout << "  (" << std::fixed << std::setprecision(1)
                          << r.gflops << " GFLOPS)";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\nSettings: matrix=" << config_.matrix_size
              << "x" << config_.matrix_size
              << ", warmup=" << config_.warmup_runs << "\n";
}

void Cli::cmd_run(const std::string& arg) {
    if (arg.empty()) {
        std::cout << "Usage: run <kernel_name|index|all>\n";
        return;
    }

    if (arg == "all") {
        for (const auto& k : kernels_) {
            run_kernel(k);
        }
        return;
    }

    // Try as index
    try {
        size_t idx = std::stoul(arg);
        if (idx < kernels_.size()) {
            run_kernel(kernels_[idx]);
            return;
        }
    } catch (...) {}

    // Try as name
    for (const auto& k : kernels_) {
        if (k.name == arg || k.name.find(arg) != std::string::npos) {
            run_kernel(k);
            return;
        }
    }

    std::cout << "Kernel not found: " << arg << "\n";
}

void Cli::run_kernel(const arena::KernelInfo& kernel) {
    std::cout << "Running " << kernel.name << "... " << std::flush;
    auto result = benchmark_.run(kernel, config_);
    results_[kernel.name] = result;

    if (result.success) {
        std::cout << std::fixed << std::setprecision(3)
                  << result.elapsed_ms << " ms, "
                  << std::setprecision(1) << result.gflops << " GFLOPS\n";
    } else {
        std::cout << "FAILED: " << result.error << "\n";
    }
}

void Cli::cmd_results() {
    if (results_.empty()) {
        std::cout << "No results yet. Run some benchmarks first.\n";
        return;
    }

    std::cout << "\n";
    std::cout << std::left << std::setw(25) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOPS" << "\n";
    std::cout << std::string(49, '-') << "\n";

    for (const auto& [name, r] : results_) {
        std::cout << std::left << std::setw(25) << name;
        if (r.success) {
            std::cout << std::right << std::fixed
                      << std::setw(12) << std::setprecision(3) << r.elapsed_ms
                      << std::setw(12) << std::setprecision(1) << r.gflops;
        } else {
            std::cout << std::right << std::setw(24) << "FAILED";
        }
        std::cout << "\n";
    }
}

void Cli::cmd_set(const std::string& what, int value) {
    if (what == "size" || what == "matrix") {
        if (value >= 64 && value <= 8192) {
            config_.matrix_size = value;
            std::cout << "Matrix size set to " << value << "x" << value << "\n";
        } else {
            std::cout << "Size must be between 64 and 8192.\n";
        }
    } else if (what == "warmup") {
        if (value >= 0 && value <= 100) {
            config_.warmup_runs = value;
            std::cout << "Warmup runs set to " << value << "\n";
        } else {
            std::cout << "Warmup must be between 0 and 100.\n";
        }
    } else {
        std::cout << "Unknown setting: " << what << "\n";
        std::cout << "Available: size, warmup\n";
    }
}

void Cli::refresh_kernels() {
    kernels_ = benchmark_.scan_kernels();
}

std::string Cli::trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

int run_cli(arena::Benchmark& benchmark) {
    Cli cli(benchmark);
    cli.run();
    return 0;
}

}

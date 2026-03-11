#include "frontend/cli.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace frontend {

Cli::Cli(arena::Runner& runner)
    : runner_(runner) {
    config_.params["M"] = 1024;
    config_.params["K"] = 1024;
    config_.params["N"] = 1024;
    config_.params["n"] = 1000000;
    config_.warmup_runs = 10;
    config_.number_of_runs = 10;

    auto categories = runner_.get_categories();
    if (!categories.empty()) {
        current_category_ = categories[0];
        current_kernels_ = runner_.get_kernels_by_category(current_category_);
    }
}

void Cli::run() {
    print_banner();
    print_help();

    std::string line;
    while (true) {
        std::cout << "\n[" << (current_category_.empty() ? "none" : current_category_) << "] > ";
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
    const auto& ctx = runner_.context();
    std::cout << "GPU: " << ctx.device_name() << "\n";
    std::cout << "Compute: SM " << ctx.compute_capability_major()
              << "." << ctx.compute_capability_minor()
              << " | Memory: " << (ctx.total_memory() / (1024 * 1024)) << " MB\n";
}

void Cli::print_help() {
    std::cout << "\nCommands:\n";
    std::cout << "  categories        List available kernel categories\n";
    std::cout << "  select <cat>      Select a category (matmul, reduce, etc.)\n";
    std::cout << "  list              List kernels in current category\n";
    std::cout << "  run <name|all>    Run benchmark for kernel or all in category\n";
    std::cout << "  results           Show benchmark results\n";
    std::cout << "  set size <n>      Set problem size (matrix size or element count)\n";
    std::cout << "  set warmup <n>    Set warmup runs (default: 10)\n";
    std::cout << "  set runs <n>      Set benchmark runs (default: 10)\n";
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
    } else if (cmd == "categories" || cmd == "cats") {
        cmd_categories();
    } else if (cmd == "select" || cmd == "sel") {
        std::string arg;
        iss >> arg;
        cmd_select(arg);
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
    } else {
        std::cout << "Unknown command: " << cmd << ". Type 'help' for commands.\n";
    }

    return true;
}

void Cli::cmd_categories() {
    auto categories = runner_.get_categories();

    if (categories.empty()) {
        std::cout << "No kernel categories available.\n";
        return;
    }

    std::cout << "\nAvailable categories:\n";
    for (const auto& cat : categories) {
        auto kernels = runner_.get_kernels_by_category(cat);
        std::cout << "  " << std::left << std::setw(15) << cat
                  << " (" << kernels.size() << " kernels)";
        if (cat == current_category_) {
            std::cout << " [selected]";
        }
        std::cout << "\n";
    }
}

void Cli::cmd_select(const std::string& category) {
    if (category.empty()) {
        std::cout << "Usage: select <category>\n";
        cmd_categories();
        return;
    }

    auto categories = runner_.get_categories();
    auto it = std::find(categories.begin(), categories.end(), category);

    if (it == categories.end()) {
        for (const auto& cat : categories) {
            if (cat.find(category) != std::string::npos) {
                current_category_ = cat;
                current_kernels_ = runner_.get_kernels_by_category(cat);
                std::cout << "Selected category: " << cat << "\n";
                return;
            }
        }
        std::cout << "Category not found: " << category << "\n";
        cmd_categories();
        return;
    }

    current_category_ = category;
    current_kernels_ = runner_.get_kernels_by_category(category);
    std::cout << "Selected category: " << category << " (" << current_kernels_.size() << " kernels)\n";
}

void Cli::cmd_list() {
    if (current_category_.empty()) {
        std::cout << "No category selected. Use 'select <category>' first.\n";
        cmd_categories();
        return;
    }

    if (current_kernels_.empty()) {
        std::cout << "No kernels in category '" << current_category_ << "'.\n";
        return;
    }

    std::cout << "\nKernels in '" << current_category_ << "':\n";
    for (size_t i = 0; i < current_kernels_.size(); i++) {
        auto* k = current_kernels_[i];
        std::cout << "  [" << i << "] " << std::left << std::setw(25) << k->name();

        if (results_.count(k->name())) {
            auto& r = results_[k->name()];
            if (r.success) {
                if (current_category_ == "matmul") {
                    std::cout << std::fixed << std::setprecision(1) << r.gflops << " GFLOPS";
                } else {
                    std::cout << std::fixed << std::setprecision(1) << r.bandwidth_gbps << " GB/s";
                }
            }
        }
        std::cout << "\n";
        std::cout << "      " << k->description() << "\n";
    }

    std::cout << "\nSettings: ";
    if (current_category_ == "matmul") {
        std::cout << "matrix=" << config_.params["M"] << "x" << config_.params["M"];
    } else if (current_category_ == "reduce") {
        std::cout << "n=" << config_.params["n"];
    }
    std::cout << ", warmup=" << config_.warmup_runs
              << ", runs=" << config_.number_of_runs << "\n";
}

void Cli::cmd_run(const std::string& arg) {
    if (current_category_.empty()) {
        std::cout << "No category selected. Use 'select <category>' first.\n";
        return;
    }

    if (arg.empty()) {
        std::cout << "Usage: run <kernel_name|index|all>\n";
        return;
    }

    if (arg == "all") {
        for (auto* k : current_kernels_) {
            run_kernel(k);
        }
        return;
    }

    try {
        size_t idx = std::stoul(arg);
        if (idx < current_kernels_.size()) {
            run_kernel(current_kernels_[idx]);
            return;
        }
    } catch (...) {}

    for (auto* k : current_kernels_) {
        if (k->name() == arg || k->name().find(arg) != std::string::npos) {
            run_kernel(k);
            return;
        }
    }

    std::cout << "Kernel not found: " << arg << "\n";
}

void Cli::run_kernel(arena::KernelDescriptor* kernel) {
    std::cout << "Running " << kernel->name() << "... " << std::flush;
    auto result = runner_.run(*kernel, config_);
    results_[kernel->name()] = result;

    if (result.success) {
        std::cout << std::fixed << std::setprecision(3)
                  << result.elapsed_ms << " ms";

        if (current_category_ == "matmul") {
            std::cout << ", " << std::setprecision(1) << result.gflops << " GFLOPS";
        } else {
            std::cout << ", " << std::setprecision(1) << result.bandwidth_gbps << " GB/s";
        }

        std::cout << " | grid=" << result.grid_x << "x" << result.grid_y
                  << ", block=" << result.block_x << "x" << result.block_y;

        if (result.verified) {
            std::cout << " [OK]";
        } else {
            std::cout << " [WARN]";
        }
        std::cout << "\n";
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
              << std::right << std::setw(10) << "Time(ms)"
              << std::setw(10) << "Perf"
              << std::setw(12) << "Grid"
              << std::setw(12) << "Block"
              << std::setw(8) << "Status" << "\n";
    std::cout << std::string(77, '-') << "\n";

    for (const auto& [name, r] : results_) {
        std::cout << std::left << std::setw(25) << name;
        if (r.success) {
            std::cout << std::right << std::fixed
                      << std::setw(10) << std::setprecision(3) << r.elapsed_ms;

            if (r.category == "matmul") {
                std::cout << std::setw(10) << std::setprecision(1) << r.gflops;
            } else {
                std::cout << std::setw(10) << std::setprecision(1) << r.bandwidth_gbps;
            }

            std::string grid = std::to_string(r.grid_x) + "x" + std::to_string(r.grid_y);
            std::string block = std::to_string(r.block_x) + "x" + std::to_string(r.block_y);

            std::cout << std::setw(12) << grid
                      << std::setw(12) << block
                      << std::setw(8) << (r.verified ? "OK" : "WARN");
        } else {
            std::cout << std::right << std::setw(52) << "FAILED";
        }
        std::cout << "\n";
    }
}

void Cli::cmd_set(const std::string& what, int value) {
    if (what == "size" || what == "matrix" || what == "n") {
        if (current_category_ == "matmul") {
            if (value >= 64 && value <= 8192) {
                config_.params["M"] = value;
                config_.params["K"] = value;
                config_.params["N"] = value;
                std::cout << "Matrix size set to " << value << "x" << value << "\n";
            } else {
                std::cout << "Matrix size must be between 64 and 8192.\n";
            }
        } else if (current_category_ == "reduce") {
            if (value >= 1000 && value <= 100000000) {
                config_.params["n"] = value;
                std::cout << "Element count set to " << value << "\n";
            } else {
                std::cout << "Element count must be between 1000 and 100000000.\n";
            }
        } else {
            std::cout << "Select a category first.\n";
        }
    } else if (what == "warmup") {
        if (value >= 0 && value <= 100) {
            config_.warmup_runs = value;
            std::cout << "Warmup runs set to " << value << "\n";
        } else {
            std::cout << "Warmup must be between 0 and 100.\n";
        }
    } else if (what == "runs") {
        if (value >= 1 && value <= 1000) {
            config_.number_of_runs = value;
            std::cout << "Benchmark runs set to " << value << "\n";
        } else {
            std::cout << "Runs must be between 1 and 1000.\n";
        }
    } else {
        std::cout << "Unknown setting: " << what << "\n";
        std::cout << "Available: size, warmup, runs\n";
    }
}

std::string Cli::trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

int run_cli(arena::Runner& runner) {
    Cli cli(runner);
    cli.run();
    return 0;
}

}

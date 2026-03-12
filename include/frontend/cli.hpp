#pragma once

#include "arena/runner.hpp"
#include "arena/kernel_descriptor.hpp"
#include <vector>
#include <map>
#include <string>

namespace frontend {

// result snapshot at a particular problem size
struct CliSizedResult {
    int problem_size = 0;
    arena::RunResult result;
};

class Cli {
public:
    Cli(arena::Runner& runner);

    void run();

private:
    void print_banner();
    void print_help();
    bool execute(const std::string& line);

    void cmd_list();
    void cmd_categories();
    void cmd_select(const std::string& category);
    void cmd_run(const std::string& arg);
    void cmd_results();
    void cmd_set(const std::string& what, const std::string& value_str);
    void cmd_compare();

    void run_kernel(arena::KernelDescriptor* kernel);

    static std::string trim(const std::string& s);

    arena::Runner& runner_;

    std::string current_category_;
    std::vector<arena::KernelDescriptor*> current_kernels_;
    std::map<std::string, arena::RunResult> results_;
    arena::RunConfig config_;

    // comparison across problem sizes: kernel_name -> results at different sizes
    std::map<std::string, std::vector<CliSizedResult>> scaling_history_;
};

int run_cli(arena::Runner& runner);

}

#include "frontend/cli.hpp"

#include "arena/cli_args.hpp"
#include "arena/distribution.hpp"
#include "arena/result_io.hpp"
#include "arena/logger.hpp"

#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace frontend {

namespace {

using nlohmann::json;

struct CliOptions {
    bool list = false;
    std::string selector;             // kernel name, category, or "all"
    std::string output = "-";         // "-" = stdout
    std::string format = "json";      // json | csv
    arena::RunConfig config;
    bool ok = true;
    std::string error;
};

bool arg_is(const char* a, const char* name) { return std::strcmp(a, name) == 0; }

// Reads the value that follows a flag, or records a usage error.
bool take_value(int argc, char** argv, int& i, const char* flag,
                std::string& out, CliOptions& o) {
    if (i + 1 >= argc) {
        o.ok = false;
        o.error = std::string(flag) + " requires a value";
        return false;
    }
    out = argv[++i];
    return true;
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions o;
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        std::string v;

        if (arg_is(a, "--list")) {
            o.list = true;
        } else if (arg_is(a, "--run")) {
            if (!take_value(argc, argv, i, "--run", v, o)) return o;
            o.selector = v;
        } else if (arg_is(a, "--param") || arg_is(a, "-p")) {
            if (!take_value(argc, argv, i, "--param", v, o)) return o;
            if (!arena::cli::apply_params(v, o.config.params)) {
                o.ok = false;
                o.error = "malformed --param '" + v + "' (expected key=integer)";
                return o;
            }
        } else if (arg_is(a, "--json")) {
            if (!take_value(argc, argv, i, "--json", v, o)) return o;
            o.output = v;
            o.format = "json";
        } else if (arg_is(a, "--output") || arg_is(a, "-o")) {
            if (!take_value(argc, argv, i, "--output", v, o)) return o;
            o.output = v;
        } else if (arg_is(a, "--format")) {
            if (!take_value(argc, argv, i, "--format", v, o)) return o;
            if (v != "json" && v != "csv") {
                o.ok = false;
                o.error = "unknown --format '" + v + "' (expected json or csv)";
                return o;
            }
            o.format = v;
        } else if (arg_is(a, "--dist")) {
            if (!take_value(argc, argv, i, "--dist", v, o)) return o;
            bool ok = false;
            o.config.input_distribution = arena::distribution_from_string(v, &ok);
            if (!ok) {
                o.ok = false;
                o.error = "unknown --dist '" + v + "' (ones, uniform, normal, adversarial)";
                return o;
            }
        } else if (arg_is(a, "--seed")) {
            if (!take_value(argc, argv, i, "--seed", v, o)) return o;
            o.config.input_seed = std::strtoull(v.c_str(), nullptr, 10);
        } else if (arg_is(a, "--sweep-min")) {
            if (!take_value(argc, argv, i, "--sweep-min", v, o)) return o;
            o.config.sweep_min = std::atoi(v.c_str());
        } else if (arg_is(a, "--sweep-max")) {
            if (!take_value(argc, argv, i, "--sweep-max", v, o)) return o;
            o.config.sweep_max = std::atoi(v.c_str());
        } else if (arg_is(a, "--sweep-factor")) {
            if (!take_value(argc, argv, i, "--sweep-factor", v, o)) return o;
            o.config.sweep_factor = std::atof(v.c_str());
        } else if (arg_is(a, "--profile")) {
            o.config.collect_metrics = true;
        } else if (arg_is(a, "--energy")) {
            o.config.collect_energy = true;
        } else if (arg_is(a, "--runs")) {
            if (!take_value(argc, argv, i, "--runs", v, o)) return o;
            o.config.number_of_runs = std::max(1, std::atoi(v.c_str()));
        } else if (arg_is(a, "--warmup")) {
            if (!take_value(argc, argv, i, "--warmup", v, o)) return o;
            o.config.warmup_mode = arena::RunConfig::WarmupMode::Fixed;
            o.config.warmup_runs = std::max(0, std::atoi(v.c_str()));
        } else if (arg_is(a, "--energy-window")) {
            if (!take_value(argc, argv, i, "--energy-window", v, o)) return o;
            o.config.energy_window_ms = static_cast<float>(std::atof(v.c_str()));
        } else if (arg_is(a, "--gui") || arg_is(a, "--help") || arg_is(a, "-h")) {
            // handled by main
        } else {
            o.ok = false;
            o.error = std::string("unknown option '") + a + "'";
            return o;
        }
    }
    return o;
}

std::vector<arena::KernelDescriptor*> select_kernels(
    arena::Runner& runner, const std::string& selector, std::string& error) {

    if (selector == "all") return runner.get_all_kernels();

    auto by_cat = runner.get_kernels_by_category(selector);
    if (!by_cat.empty()) return by_cat;

    for (auto* k : runner.get_all_kernels()) {
        if (k->name() == selector) return {k};
    }

    error = "no kernel or category named '" + selector + "'";
    return {};
}

}   // namespace

bool wants_cli(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (arg_is(argv[i], "--list") || arg_is(argv[i], "--run")) return true;
    }
    return false;
}

bool cli_writes_to_stdout(int argc, char** argv) {
    if (!wants_cli(argc, argv)) return false;
    // stdout is the default; only an explicit non-"-" destination redirects it
    for (int i = 1; i < argc - 1; i++) {
        if (arg_is(argv[i], "--json") || arg_is(argv[i], "--output") ||
            arg_is(argv[i], "-o")) {
            if (std::strcmp(argv[i + 1], "-") != 0) return false;
        }
    }
    return true;
}

void print_cli_usage(const char* program) {
    std::cout <<
        "Headless mode:\n"
        "  " << program << " --list\n"
        "  " << program << " --run <kernel|category|all> [options]\n\n"
        "Options:\n"
        "  --list                    List kernels and categories as JSON, then exit\n"
        "  --run <selector>          Kernel name, category name, or 'all'\n"
        "  --param k=v, -p k=v       Problem size, repeatable (e.g. -p n=1000000)\n"
        "  --runs <n>                Timed runs per kernel (default 10)\n"
        "  --dist <name>             Input data: ones, uniform, normal, adversarial\n"
        "  --seed <n>                Input seed, for reproducible runs (default 42)\n"
        "  --sweep-min/-max <n>      Sweep range; 0 uses the category default\n"
        "  --sweep-factor <x>        Step multiplier between sweep sizes (default 4)\n"
        "  --warmup <n>              Fixed warmup count (default: auto steady-state)\n"
        "  --profile                 Collect hardware counters (needs perf access)\n"
        "  --energy                  Collect NVML energy (adds a sustained pass)\n"
        "  --energy-window <ms>      Energy window (default 500)\n"
        "  --format <json|csv>       Output format (default json)\n"
        "  --json <file|->           Write JSON to file, or - for stdout\n"
        "  --output <file|->, -o     Same, honouring --format\n\n"
        "Exit codes:\n"
        "  0  all kernels ran and verified\n"
        "  1  ran, but at least one kernel failed verification\n"
        "  2  at least one kernel errored\n"
        "  3  usage error\n";
}

int run_cli(arena::Runner& runner, int argc, char** argv) {
    CliOptions o = parse_args(argc, argv);
    if (!o.ok) {
        std::cerr << "Error: " << o.error << "\n\n";
        print_cli_usage(argv[0]);
        return static_cast<int>(CliExit::UsageError);
    }

    // Console logging was already muted in main() when stdout is the
    // destination -- it has to happen before any startup logging, not here.
    const bool to_stdout = (o.output == "-");

    if (o.list) {
        json j;
        j["categories"] = runner.get_categories();
        j["kernels"] = json::array();
        for (auto* k : runner.get_all_kernels()) {
            j["kernels"].push_back({
                {"name",        k->name()},
                {"category",    k->category()},
                {"description", k->description()},
                {"dsl",         arena::result_io::detect_dsl(k)},
                {"parameters",  k->get_parameter_names()},
                {"source",      k->source_path()},
            });
        }
        std::ostream* out = &std::cout;
        std::ofstream file;
        if (!to_stdout) {
            file.open(o.output);
            if (!file) {
                std::cerr << "Error: cannot write " << o.output << "\n";
                return static_cast<int>(CliExit::UsageError);
            }
            out = &file;
        }
        *out << j.dump(2) << std::endl;
        return static_cast<int>(CliExit::Ok);
    }

    if (o.selector.empty()) {
        std::cerr << "Error: --run requires a selector\n\n";
        print_cli_usage(argv[0]);
        return static_cast<int>(CliExit::UsageError);
    }

    std::string sel_error;
    auto kernels = select_kernels(runner, o.selector, sel_error);
    if (kernels.empty()) {
        std::cerr << "Error: " << sel_error << "\n";
        return static_cast<int>(CliExit::UsageError);
    }

    std::vector<arena::RunResult> results;
    results.reserve(kernels.size());
    int errored = 0, unverified = 0;

    for (auto* k : kernels) {
        auto r = runner.run(*k, o.config);
        if (!r.success)      errored++;
        else if (!r.verified) unverified++;
        results.push_back(std::move(r));
    }

    // Emit
    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!to_stdout) {
        file.open(o.output);
        if (!file) {
            std::cerr << "Error: cannot write " << o.output << "\n";
            return static_cast<int>(CliExit::UsageError);
        }
        out = &file;
    }

    if (o.format == "csv") {
        *out << arena::result_io::csv_header() << "\n";
        for (size_t i = 0; i < results.size(); i++) {
            *out << arena::result_io::csv_row(results[i], arena::result_io::detect_dsl(kernels[i])) << "\n";
        }
    } else {
        json j;
        j["environment"] = arena::result_io::environment_json(
            runner.context(), runner.power());
        j["config"]  = arena::result_io::config_json(o.config);
        j["results"] = json::array();
        for (size_t i = 0; i < results.size(); i++) {
            auto rj = arena::result_io::to_json(results[i]);
            rj["dsl"] = arena::result_io::detect_dsl(kernels[i]);
            j["results"].push_back(std::move(rj));
        }
        j["summary"] = {
            {"total",      (int)results.size()},
            {"errored",    errored},
            {"unverified", unverified},
        };
        *out << j.dump(2) << std::endl;
    }

    if (!to_stdout) {
        std::cerr << "Wrote " << results.size() << " result(s) to " << o.output << "\n";
    }

    if (errored > 0)    return static_cast<int>(CliExit::RunError);
    if (unverified > 0) return static_cast<int>(CliExit::VerificationFailed);
    return static_cast<int>(CliExit::Ok);
}

}

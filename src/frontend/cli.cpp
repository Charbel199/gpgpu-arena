#include "frontend/cli.hpp"

#include "arena/cli_args.hpp"
#include "arena/distribution.hpp"
#include "arena/result_io.hpp"
#include "arena/logger.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/sass.hpp"

#include <nlohmann/json.hpp>
#include <cstring>
#include <filesystem>
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

    // Inspection modes: compile one kernel and describe it instead of running.
    enum class Inspect { None, Summary, Sass, Ptx };
    Inspect inspect = Inspect::None;

    // Every combination of these runs, so one invocation can cover several
    // sizes, input distributions and seeds. A single value is the usual case.
    std::map<std::string, std::vector<int>> param_lists;
    std::vector<arena::Distribution> dists = {arena::Distribution::Uniform};
    std::vector<uint64_t> seeds = {42};

    std::string source_override;      // as given; resolved in run_cli
    std::string output = "-";         // "-" = stdout
    std::string format = "json";      // json | csv
    arena::RunConfig config;
    bool sweep_block = false;
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
        } else if (arg_is(a, "--inspect") || arg_is(a, "--sass") || arg_is(a, "--ptx")) {
            const char* flag = a;
            if (!take_value(argc, argv, i, flag, v, o)) return o;
            o.selector = v;
            o.inspect = arg_is(flag, "--inspect") ? CliOptions::Inspect::Summary
                      : arg_is(flag, "--sass")    ? CliOptions::Inspect::Sass
                                                  : CliOptions::Inspect::Ptx;
        } else if (arg_is(a, "--source-override")) {
            if (!take_value(argc, argv, i, "--source-override", v, o)) return o;
            o.source_override = v;
        } else if (arg_is(a, "--l2")) {
            if (!take_value(argc, argv, i, "--l2", v, o)) return o;
            if (v != "cold" && v != "warm") {
                o.ok = false;
                o.error = "unknown --l2 '" + v + "' (expected cold or warm)";
                return o;
            }
            o.config.cold_l2 = (v == "cold");
        } else if (arg_is(a, "--param") || arg_is(a, "-p")) {
            if (!take_value(argc, argv, i, "--param", v, o)) return o;
            if (!arena::cli::apply_param_list(v, o.param_lists)) {
                o.ok = false;
                o.error = "malformed --param '" + v +
                          "' (expected key=integer or key=int,int,...)";
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
            o.dists.clear();
            const auto names = arena::cli::split_csv(v);
            for (const auto& name : names) {
                bool ok = false;
                o.dists.push_back(arena::distribution_from_string(name, &ok));
                if (!ok) {
                    o.ok = false;
                    o.error = "unknown --dist '" + name + "' (ones, uniform, normal, adversarial)";
                    return o;
                }
            }
            if (o.dists.empty()) {
                o.ok = false;
                o.error = "--dist requires at least one distribution";
                return o;
            }
        } else if (arg_is(a, "--seed")) {
            if (!take_value(argc, argv, i, "--seed", v, o)) return o;
            auto seeds = arena::cli::parse_u64_list(v);
            if (!seeds) {
                o.ok = false;
                o.error = "malformed --seed '" + v + "' (expected n or n,n,...)";
                return o;
            }
            o.seeds = *seeds;
        } else if (arg_is(a, "--block")) {
            if (!take_value(argc, argv, i, "--block", v, o)) return o;
            o.config.block_size = std::atoi(v.c_str());
        } else if (arg_is(a, "--define")) {
            if (!take_value(argc, argv, i, "--define", v, o)) return o;
            if (!arena::cli::apply_params(v, o.config.compile_options)) {
                o.ok = false;
                o.error = "malformed --define '" + v + "' (expected key=integer)";
                return o;
            }
        } else if (arg_is(a, "--sweep-block")) {
            o.sweep_block = true;
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


// Resolves the override path and attaches it to the single selected kernel.
bool apply_source_override(const std::vector<arena::KernelDescriptor*>& kernels,
                           const std::string& given, std::string& err) {
    namespace fs = std::filesystem;
    if (kernels.size() != 1) {
        err = "--source-override needs exactly one kernel, got " +
              std::to_string(kernels.size()) + " from the selector";
        return false;
    }
    auto* k = kernels.front();
    if (!k->needs_compilation() || k->source_path().empty()) {
        err = k->name() + " is built into the binary and has no source slot to override";
        return false;
    }

    fs::path p(given);
    if (p.is_relative()) p = fs::path(ARENA_SOURCE_DIR) / p;
    std::error_code ec;
    p = fs::weakly_canonical(p, ec);
    if (!fs::exists(p)) {
        err = "--source-override file not found: " + p.string();
        return false;
    }

    // The backend is chosen by extension and the descriptor's ABI assumes
    // its own DSL, so a Triton slot can only take a Triton source.
    const auto want = arena::KernelCompiler::get_extension(k->source_path());
    const auto got  = arena::KernelCompiler::get_extension(p.filename().string());
    if (want != got) {
        err = "--source-override for " + k->name() + " must be a " + want +
              " file (got " + (got.empty() ? "no extension" : got) + ")";
        return false;
    }

    k->set_source_override(p.string());
    return true;
}

std::string abs_kernel_source(const arena::KernelDescriptor& k) {
    const auto src = k.effective_source_path();
    if (src.empty()) return "";
    return arena::resolve_source(ARENA_KERNEL_DIR, src);
}

// --inspect / --sass / --ptx. Compiles exactly as a run would (same defines,
// same override) and describes the result without launching anything.
int run_inspect(arena::Runner& runner, arena::KernelDescriptor& k,
                const CliOptions& o, bool to_stdout) {
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

    const std::string dsl = arena::result_io::detect_dsl(&k);
    if (!k.needs_compilation() || k.source_path().empty()) {
        if (o.inspect != CliOptions::Inspect::Summary) {
            std::cerr << "Error: " << k.name() << " is compiled into the binary (" << dsl
                      << "); there is no cubin to disassemble\n";
            return static_cast<int>(CliExit::UsageError);
        }
    }

    try {
        arena::CompileResult cr;
        const bool compiled = k.needs_compilation() && !k.source_path().empty();
        if (compiled) cr = runner.compiler().compile(k.effective_source_path(), o.config.compile_options);

        // Entry point as the runner would look it up. DSL descriptors read it
        // from the compile result, so it has to be set first.
        k.set_compile_result(cr);
        const std::string entry = compiled ? k.function_name() : "";

        if (o.inspect == CliOptions::Inspect::Ptx) {
            const auto ptx = runner.compiler().emit_ptx(k.effective_source_path(),
                                                        o.config.compile_options);
            if (ptx.empty()) {
                std::cerr << "Error: " << dsl << " does not expose PTX for " << k.name() << "\n";
                return static_cast<int>(CliExit::RunError);
            }
            *out << arena::read_file(ptx);
            return static_cast<int>(CliExit::Ok);
        }

        std::string sass_dump, res_dump;
        if (compiled) {
            const auto cubin = arena::shell_quote(cr.module_path);
            sass_dump = arena::run_command("cuobjdump -sass " + cubin + " 2>&1", "cuobjdump -sass");
            res_dump  = arena::run_command("cuobjdump -res-usage " + cubin + " 2>&1", "cuobjdump -res-usage");
        }
        const std::string fn_sass = arena::sass::function_text(sass_dump, entry);

        if (o.inspect == CliOptions::Inspect::Sass) {
            *out << fn_sass;
            return static_cast<int>(CliExit::Ok);
        }

        json j;
        j["kernel"]          = k.name();
        // Paths below are absolute in the binary's filesystem, which inside
        // Docker is not the host's. Tools relativize them against this.
        j["repo_root"]       = ARENA_SOURCE_DIR;
        j["category"]        = k.category();
        j["description"]     = k.description();
        j["dsl"]             = dsl;
        j["source"]          = abs_kernel_source(k);
        j["slot_source"]     = k.source_path().empty() ? "" : arena::resolve_source(ARENA_KERNEL_DIR, k.source_path());
        j["descriptor_file"] = k.descriptor_file();
        j["source_override"] = k.source_override();
        j["parameters"]      = k.get_parameter_names();
        j["tunable_block_sizes"]     = k.tunable_block_sizes();
        j["tunable_compile_options"] = k.tunable_compile_options();
        j["input_dtype"]     = arena::dtype_name(k.input_dtype());
        j["output_dtype"]    = arena::dtype_name(k.output_dtype());

        if (compiled) {
            const auto hist = arena::sass::opcode_histogram(fn_sass);
            const auto res  = arena::sass::parse_resource_usage(res_dump, entry);
            int total = 0;
            for (const auto& [op, n] : hist) total += n;

            j["entry_point"] = entry;
            j["cubin"] = cr.module_path;
            j["build"] = {
                {"cache_hit", cr.cache_hit},
                {"options",   o.config.compile_options},
                {"constants", cr.constants},
                {"num_warps", cr.num_warps},
                {"block_dim", cr.block_dim},
                {"num_params", cr.num_params},
            };
            j["resources"] = {
                {"registers", res.registers},
                {"stack",     res.stack},
                {"shared",    res.shared},
                {"local",     res.local},
                {"spills_or_local_arrays", res.local > 0 || res.stack > 0},
            };
            j["sass"] = {
                {"instructions",  total},
                {"classes",       arena::sass::classify(hist)},
                {"mean_global_load_bytes", arena::sass::mean_global_load_bytes(hist)},
                {"opcodes",       hist},
            };
        }
        *out << j.dump(2) << std::endl;
        return static_cast<int>(CliExit::Ok);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return static_cast<int>(CliExit::RunError);
    }
}

}   // namespace

bool wants_cli(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (arg_is(argv[i], "--list") || arg_is(argv[i], "--run") ||
            arg_is(argv[i], "--inspect") || arg_is(argv[i], "--sass") ||
            arg_is(argv[i], "--ptx")) return true;
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
        "  " << program << " --run <kernel|category|all> [options]\n"
        "  " << program << " --inspect|--sass|--ptx <kernel> [options]\n\n"
        "Options:\n"
        "  --list                    List kernels and categories as JSON, then exit\n"
        "  --run <selector>          Kernel name, category name, or 'all'\n"
        "  --inspect <kernel>        Compile and describe one kernel as JSON: source,\n"
        "                            descriptor, registers, spills, SASS instruction mix\n"
        "  --sass <kernel>           Compile and print the kernel's SASS\n"
        "  --ptx <kernel>            Compile and print PTX (CUDA and Triton only)\n"
        "  --source-override <file>  Use this source in the kernel's slot instead of its\n"
        "                            own; relative paths are from the repo root. Needs a\n"
        "                            single kernel and the same file type as the slot\n"
        "  --param k=v, -p k=v       Problem size, repeatable (e.g. -p n=1000000);\n"
        "                            k=v1,v2 runs each value\n"
        "  --runs <n>                Timed runs per kernel (default 10)\n"
        "  --l2 <cold|warm>          Evict L2 before every timed launch, or not\n"
        "                            (default warm)\n"
        "  --dist <name[,name]>      Input data: ones, uniform, normal, adversarial\n"
        "  --seed <n[,n]>            Input seed, for reproducible runs (default 42)\n"
        "  --sweep-min/-max <n>      Sweep range; 0 uses the category default\n"
        "  --sweep-factor <x>        Step multiplier between sweep sizes (default 4)\n"
        "  --block <n>               Launch at this block size where the kernel allows it\n"
        "  --sweep-block             Sweep the tuning axis: block size for CUDA kernels,\n"
        "                            compile-time config for DSL kernels\n"
        "  --define <KEY=n>          Compile-time knob for DSL kernels (e.g. BLOCK_SIZE=512)\n"
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

    if (o.inspect != CliOptions::Inspect::None && kernels.size() != 1) {
        std::cerr << "Error: --inspect, --sass and --ptx take one kernel name, not a category\n";
        return static_cast<int>(CliExit::UsageError);
    }

    if (!o.source_override.empty()) {
        std::string err;
        if (!apply_source_override(kernels, o.source_override, err)) {
            std::cerr << "Error: " << err << "\n";
            return static_cast<int>(CliExit::UsageError);
        }
    }

    if (o.inspect != CliOptions::Inspect::None) {
        return run_inspect(runner, *kernels.front(), o, to_stdout);
    }

    const auto param_sets = arena::cli::expand_grid(o.param_lists);

    std::vector<arena::RunResult> results;
    std::vector<arena::KernelDescriptor*> ran;   // parallel to results, for the dsl column
    int errored = 0, unverified = 0;

    for (auto* k : kernels) {
        const auto variants = arena::cli::tuning_variants_for(
            o.sweep_block, o.config.block_size, o.config.compile_options,
            k->tunable_block_sizes(), k->tunable_compile_options());

        for (const auto& params : param_sets)
        for (auto dist : o.dists)
        for (auto seed : o.seeds)
        for (const auto& v : variants) {
            auto cfg = o.config;
            cfg.params             = params;
            cfg.input_distribution = dist;
            cfg.input_seed         = seed;
            cfg.block_size         = v.block_size;
            cfg.compile_options    = v.defines;
            auto r = runner.run(*k, cfg);
            if (!r.success)      errored++;
            else if (!r.verified) unverified++;
            results.push_back(std::move(r));
            ran.push_back(k);
        }
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
            *out << arena::result_io::csv_row(results[i], arena::result_io::detect_dsl(ran[i])) << "\n";
        }
    } else {
        json j;
        j["environment"] = arena::result_io::environment_json(
            runner.context(), runner.power());
        j["config"]  = arena::result_io::config_json(o.config);
        // The per-run values live on each result's "input"; these are the
        // lists the invocation asked for.
        j["config"]["param_lists"] = o.param_lists;
        j["config"]["seeds"] = o.seeds;
        j["config"]["distributions"] = json::array();
        for (auto d : o.dists) j["config"]["distributions"].push_back(arena::distribution_name(d));
        if (!o.source_override.empty())
            j["config"]["source_override"] = kernels.front()->source_override();
        j["results"] = json::array();
        for (size_t i = 0; i < results.size(); i++) {
            auto rj = arena::result_io::to_json(results[i]);
            rj["dsl"] = arena::result_io::detect_dsl(ran[i]);
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

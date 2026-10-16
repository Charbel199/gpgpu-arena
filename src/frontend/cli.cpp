#include "frontend/cli.hpp"

#include "arena/cli_args.hpp"
#include "arena/paths.hpp"
#include "arena/distribution.hpp"
#include "arena/result_io.hpp"
#include "arena/logger.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/sass.hpp"
#include "arena/manifest/manifest_kernel.hpp"

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
    bool run = false;
    bool plan = false;                // resolve a manifest's launches, no GPU work

    // Every combination of these runs, so one invocation can cover several
    // sizes, input distributions and seeds. A single value is the usual case.
    std::map<std::string, std::vector<int>> param_lists;
    std::vector<std::map<std::string, int>> shapes;   // explicit --shape list
    std::vector<arena::Distribution> dists = {arena::Distribution::Uniform};
    std::vector<uint64_t> seeds = {42};

    std::string source_override;      // as given; resolved in run_cli
    std::string manifest;             // a kernel described by JSON instead of C++
    std::string artifacts_dir;        // --inspect copies compiler sidecars here
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
            o.run = true;
        } else if (arg_is(a, "--inspect") || arg_is(a, "--sass") || arg_is(a, "--ptx")) {
            const char* flag = a;
            // The kernel name is optional here, for use with --manifest.
            if (i + 1 < argc && argv[i + 1][0] != '-') o.selector = argv[++i];
            o.inspect = arg_is(flag, "--inspect") ? CliOptions::Inspect::Summary
                      : arg_is(flag, "--sass")    ? CliOptions::Inspect::Sass
                                                  : CliOptions::Inspect::Ptx;
        } else if (arg_is(a, "--plan")) {
            o.plan = true;
        } else if (arg_is(a, "--scratch-policy")) {
            if (!take_value(argc, argv, i, "--scratch-policy", v, o)) return o;
            auto p = arena::cli::parse_scratch_policy(v);
            if (!p) {
                o.ok = false;
                o.error = "unknown --scratch-policy '" + v + "' (expected legacy or declared)";
                return o;
            }
            o.config.scratch_policy = *p;
        } else if (arg_is(a, "--shape")) {
            if (!take_value(argc, argv, i, "--shape", v, o)) return o;
            auto shape = arena::cli::parse_shape(v);
            if (!shape) {
                o.ok = false;
                o.error = "malformed --shape '" + v + "' (expected key=n,key=n)";
                return o;
            }
            o.shapes.push_back(*shape);
        } else if (arg_is(a, "--open")) {
            // handled by main: registers the folder's manifests as kernels
            if (!take_value(argc, argv, i, "--open", v, o)) return o;
        } else if (arg_is(a, "--artifacts-dir")) {
            if (!take_value(argc, argv, i, "--artifacts-dir", v, o)) return o;
            o.artifacts_dir = v;
        } else if (arg_is(a, "--manifest")) {
            if (!take_value(argc, argv, i, "--manifest", v, o)) return o;
            o.manifest = v;
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
        } else if (arg_is(a, "--counters")) {
            if (!take_value(argc, argv, i, "--counters", v, o)) return o;
            if (v != "basic" && v != "full") {
                o.ok = false;
                o.error = "unknown --counters '" + v + "' (expected basic or full)";
                return o;
            }
            o.config.counter_level = (v == "full")
                ? arena::RunConfig::CounterLevel::Full
                : arena::RunConfig::CounterLevel::Basic;
            if (v == "full") o.config.collect_metrics = true;
        } else if (arg_is(a, "--lock-clocks")) {
            // applied by main, right after NVML comes up
            if (!take_value(argc, argv, i, "--lock-clocks", v, o)) return o;
            if (!arena::cli::parse_lock_clocks(v)) {
                o.ok = false;
                o.error = "malformed --lock-clocks '" + v + "' (expected MHz or auto)";
                return o;
            }
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
    if (o.plan) {
        o.error = arena::cli::plan_args_error(!o.manifest.empty(), o.run, o.list,
                                              o.inspect != CliOptions::Inspect::None);
        o.ok = o.error.empty();
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

    fs::path p(arena::resolve_user_path(given, ARENA_SOURCE_DIR));
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
        std::map<std::string, arena::CompileResult> extra_cr;
        for (const auto& src : k.extra_sources()) extra_cr[src] = runner.compiler().compile(src);
        if (!extra_cr.empty()) k.set_extra_compile_results(extra_cr);

        // Entry point as the runner would look it up. DSL descriptors read it
        // from the compile result, so it has to be set first.
        k.set_compile_result(cr);
        const std::string entry = compiled ? k.function_name() : "";

        // A manifest can launch several entry points from one cubin; each is
        // described, and the instruction mix covers all of them.
        std::vector<std::string> entries = {entry};
        const auto* minfo = dynamic_cast<const arena::ManifestInfo*>(&k);
        if (minfo && compiled) entries = minfo->entry_points();

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

        // Dumps per source: a multi-source manifest keeps its entry points in
        // several cubins, and two files may use the same kernel name.
        std::map<std::string, std::pair<std::string, std::string>> dumps;   // source -> (sass, res)
        std::string sass_dump, res_dump;
        if (compiled) {
            std::map<std::string, std::string> cubins = {{k.effective_source_path(), cr.module_path}};
            for (const auto& [src, ecr] : extra_cr) cubins[src] = ecr.module_path;
            for (const auto& [src, c] : cubins) {
                const auto q = arena::shell_quote(c);
                auto s = arena::run_command("cuobjdump -sass " + q + " 2>&1", "cuobjdump -sass");
                auto r = arena::run_command("cuobjdump -res-usage " + q + " 2>&1", "cuobjdump -res-usage");
                sass_dump += s;
                res_dump += r;
                dumps[src] = {std::move(s), std::move(r)};
            }
        }
        std::vector<std::string> entry_srcs(entries.size(), k.effective_source_path());
        if (minfo && compiled) entry_srcs = minfo->entry_sources();
        auto dump_for = [&](size_t i) -> const std::pair<std::string, std::string>& {
            auto it = dumps.find(entry_srcs[i]);
            return it != dumps.end() ? it->second : dumps.begin()->second;
        };
        std::string fn_sass;
        for (size_t i = 0; i < entries.size() && !dumps.empty(); i++) {
            const auto& d = dump_for(i).first;
            const auto t = arena::sass::function_text(d, entries[i]);
            if (fn_sass.find(t) == std::string::npos) fn_sass += t;
        }

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
        j["reference"]       = k.reference();
        j["source"]        = abs_kernel_source(k);
        j["slot_source"]     = k.source_path().empty() ? "" : arena::resolve_source(ARENA_KERNEL_DIR, k.source_path());
        j["descriptor_file"] = k.descriptor_file();
        j["source_override"] = k.source_override();
        if (minfo) j["manifest"] = minfo->manifest_path();
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
            j["cubin"] = std::filesystem::absolute(cr.module_path).string();
            auto artifacts = runner.compiler().sidecars(k.effective_source_path(),
                                                        o.config.compile_options);
            if (!o.artifacts_dir.empty()) {
                namespace fs = std::filesystem;
                fs::path dir(arena::resolve_user_path(o.artifacts_dir, ARENA_SOURCE_DIR));
                fs::create_directories(dir);
                // CUDA has no sidecar; its PTX is a separate nvcc pass.
                if (dsl == "cuda") {
                    const auto ptx = runner.compiler().emit_ptx(k.effective_source_path(),
                                                                o.config.compile_options);
                    if (!ptx.empty()) artifacts.push_back(fs::absolute(ptx).string());
                }
                std::vector<std::string> copied;
                for (const auto& a : artifacts) {
                    const std::string name = fs::path(a).filename().string();
                    const auto dot = name.find('.');
                    const fs::path dest = dir / ("kernel" + (dot == std::string::npos ? "" : name.substr(dot)));
                    std::error_code ec;
                    fs::copy_file(a, dest, fs::copy_options::overwrite_existing, ec);
                    if (!ec) copied.push_back(dest.string());
                }
                j["artifacts_copied"] = copied;
            }
            j["artifacts"] = artifacts;
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
            if (entries.size() > 1) {
                j["entries"] = json::array();
                for (size_t i = 0; i < entries.size(); i++) {
                    const auto& e = entries[i];
                    const auto& d = dump_for(i);
                    const auto h = arena::sass::opcode_histogram(arena::sass::function_text(d.first, e));
                    const auto r = arena::sass::parse_resource_usage(d.second, e);
                    int n = 0;
                    for (const auto& [op, c] : h) n += c;
                    j["entries"].push_back({{"entry", e}, {"registers", r.registers},
                                            {"shared", r.shared}, {"local", r.local},
                                            {"instructions", n}, {"classes", arena::sass::classify(h)}});
                }
            }
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

// --plan: compiles the manifest's sources, then for each problem shape
// resolves every launch the way a run would, without allocating or
// launching. Exit 0 if every shape resolved, 1 if any has an error, 2 if a
// source does not compile.
int run_plan(arena::Runner& runner, arena::KernelDescriptor& k, const CliOptions& o,
             const std::vector<std::map<std::string, int>>& param_sets, bool to_stdout) {
    auto* minfo = dynamic_cast<arena::ManifestInfo*>(&k);
    if (!minfo) {
        std::cerr << "Error: --plan needs a manifest kernel\n";
        return static_cast<int>(CliExit::UsageError);
    }
    try {
        k.set_compile_options(o.config.compile_options);
        k.set_compile_result(runner.compiler().compile(k.effective_source_path(), k.compile_options()));
        std::map<std::string, arena::CompileResult> extra;
        for (const auto& src : k.extra_sources()) extra[src] = runner.compiler().compile(src);
        if (!extra.empty()) k.set_extra_compile_results(extra);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return static_cast<int>(CliExit::RunError);
    }

    json j = {{"plans", json::array()}, {"ok", true}};
    for (const auto& params : param_sets) {
        k.set_problem_size(params);
        auto p = minfo->plan(runner.mutable_context());
        if (!p["error"].get<std::string>().empty()) j["ok"] = false;
        j["plans"].push_back(std::move(p));
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
    return j["ok"].get<bool>() ? 0 : 1;
}

}   // namespace

bool wants_cli(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (arg_is(argv[i], "--list") || arg_is(argv[i], "--run") ||
            arg_is(argv[i], "--inspect") || arg_is(argv[i], "--sass") ||
            arg_is(argv[i], "--ptx") || arg_is(argv[i], "--manifest") ||
            arg_is(argv[i], "--plan")) return true;
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
        "  " << program << " --inspect|--sass|--ptx <kernel> [options]\n"
        "  " << program << " --plan --manifest <file.json> [--shape k=v,...]\n\n"
        "Options:\n"
        "  --list                    List kernels and categories as JSON, then exit\n"
        "  --run <selector>          Kernel name, category name, or 'all'\n"
        "  --inspect <kernel>        Compile and describe one kernel as JSON: source,\n"
        "                            descriptor, registers, spills, SASS instruction mix\n"
        "  --sass <kernel>           Compile and print the kernel's SASS\n"
        "  --ptx <kernel>            Compile and print PTX (CUDA and Triton only)\n"
        "  --open <dir>              Load every kernel manifest under dir (recursively) as\n"
        "                            a kernel, for --list, --run and the GUI; repeatable.\n"
        "                            Relative paths here and below are from the current\n"
        "                            directory, or else the repo root\n"
        "  --manifest <file.json>    Run, or with --inspect/--sass/--ptx describe, a kernel\n"
        "                            defined by a JSON manifest instead of a C++\n"
        "                            descriptor\n"
        "  --plan                    With --manifest: compile, then print as JSON what\n"
        "                            each --shape would launch (scratch sizes, grid,\n"
        "                            block, shared memory, arguments) without allocating\n"
        "                            or launching. Exit 0 if every shape resolves, 1 if\n"
        "                            any has an error\n"
        "  --scratch-policy <p>      legacy (default): manifest scratch is zeroed before\n"
        "                            every launch, untimed. declared: scratch marked\n"
        "                            \"zero\": true is zeroed inside the timed op, the\n"
        "                            rest is filled with 0xFF bytes\n"
        "  --artifacts-dir <dir>     With --inspect: copy PTX, Triton IR and Warp's\n"
        "                            generated CUDA into dir, since the compile cache may\n"
        "                            not be visible outside Docker\n"
        "  --source-override <file>  Use this source in the kernel's slot instead of its\n"
        "                            own. Needs a single kernel and the same file type\n"
        "                            as the slot\n"
        "  --param k=v, -p k=v       Problem size, repeatable (e.g. -p n=1000000);\n"
        "                            k=v1,v2 runs each value\n"
        "  --shape k=v,k=v           One explicit problem shape, repeatable (e.g.\n"
        "                            --shape rows=64,cols=1024 --shape rows=16384,cols=4096);\n"
        "                            an alternative to -p for sizes that are not a grid\n"
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
        "  --counters <basic|full>   Counter set for --profile (default basic: DRAM bytes,\n"
        "                            occupancy, IPC). full adds pipe utilisation, stall\n"
        "                            reasons, shared memory and L2 traffic, in many more\n"
        "                            replay passes; it implies --profile\n"
        "  --lock-clocks <MHz|auto>  Lock the SM clock for the whole process and release\n"
        "                            it on exit. Needs root, and in Docker also\n"
        "                            --cap-add SYS_ADMIN; without them the run goes on\n"
        "                            unlocked. auto picks the lower of\n"
        "                            the board's default application clock and 85% of the\n"
        "                            max SM clock\n"
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

bool validate_cli_args(int argc, char** argv) {
    CliOptions o = parse_args(argc, argv);
    if (o.ok) return true;
    std::cerr << "Error: " << o.error << "\n\n";
    print_cli_usage(argv[0]);
    return false;
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
                {"reference",   k->reference()},
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

    std::vector<arena::KernelDescriptor*> kernels;
    std::unique_ptr<arena::KernelDescriptor> manifest_kernel;
    if (!o.manifest.empty()) {
        if (!o.selector.empty() || !o.source_override.empty()) {
            std::cerr << "Error: --manifest names its own kernel; drop the kernel name and "
                         "--source-override\n";
            return static_cast<int>(CliExit::UsageError);
        }
        std::filesystem::path mp(arena::resolve_user_path(o.manifest, ARENA_SOURCE_DIR));
        try {
            manifest_kernel = arena::load_manifest(mp.string());
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return static_cast<int>(CliExit::UsageError);
        }
        kernels = {manifest_kernel.get()};
    } else {
        if (o.selector.empty()) {
            std::cerr << "Error: --run, --inspect, --sass and --ptx need a kernel name or --manifest\n\n";
            print_cli_usage(argv[0]);
            return static_cast<int>(CliExit::UsageError);
        }
        std::string sel_error;
        kernels = select_kernels(runner, o.selector, sel_error);
        if (kernels.empty()) {
            std::cerr << "Error: " << sel_error << "\n";
            return static_cast<int>(CliExit::UsageError);
        }
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

    if (!o.shapes.empty() && !o.param_lists.empty()) {
        std::cerr << "Error: use either --shape or -p, not both\n";
        return static_cast<int>(CliExit::UsageError);
    }
    const auto param_sets = o.shapes.empty() ? arena::cli::expand_grid(o.param_lists) : o.shapes;

    if (o.plan) return run_plan(runner, *kernels.front(), o, param_sets, to_stdout);

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

#include "arena/manifest/manifest_kernel.hpp"
#include "arena/manifest/args.hpp"
#include "arena/manifest/expr.hpp"
#include "arena/manifest/limits.hpp"
#include "arena/compilers/kernel_compiler.hpp"
#include "arena/paths.hpp"
#include "arena/categories/reduce_base.hpp"
#include "arena/categories/scan_base.hpp"
#include "arena/categories/softmax_base.hpp"
#include "arena/categories/layernorm_base.hpp"
#include "arena/categories/matmul_base.hpp"
#include "arena/categories/attention_base.hpp"
#include "arena/device/utils.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace arena {

namespace fs = std::filesystem;
using nlohmann::json;

std::unique_ptr<KernelDescriptor> load_manifest_spec(const nlohmann::json& spec, const std::string& path);

namespace {

// Categories with a float constant in the problem (layernorm's eps) expose
// it as kEps; manifests can pass it as "eps".
template <typename B, typename = void>
struct HasEps : std::false_type {};
template <typename B>
struct HasEps<B, std::void_t<decltype(B::kEps)>> : std::true_type {};

manifest::DeviceLimits query_device_limits() {
    CUdevice dev = 0;
    check_cuda(cuCtxGetDevice(&dev), "cuCtxGetDevice");
    auto attr = [&](CUdevice_attribute a) {
        int v = 0;
        check_cuda(cuDeviceGetAttribute(&v, a, dev), "cuDeviceGetAttribute");
        return static_cast<int64_t>(v);
    };
    manifest::DeviceLimits d;
    d.max_threads_per_block = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    d.max_block = {attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X), attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                   attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)};
    d.max_grid = {attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X), attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
                  attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)};
    d.max_shared_per_block = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    d.max_shared_per_block_optin = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    d.max_regs_per_block = attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    d.max_regs_per_sm = attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    return d;
}

manifest::FunctionLimits query_function_limits(CUfunction fn) {
    auto attr = [&](CUfunction_attribute a) {
        int v = 0;
        check_cuda(cuFuncGetAttribute(&v, a, fn), "cuFuncGetAttribute");
        return static_cast<int64_t>(v);
    };
    manifest::FunctionLimits f;
    f.registers = attr(CU_FUNC_ATTRIBUTE_NUM_REGS);
    f.static_shared_bytes = attr(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    f.max_threads_per_block = attr(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    return f;
}

json device_limits_json(const manifest::DeviceLimits& d) {
    return {{"max_threads_per_block", d.max_threads_per_block},
            {"max_block", d.max_block},
            {"max_grid", d.max_grid},
            {"max_shared_per_block", d.max_shared_per_block},
            {"max_shared_per_block_optin", d.max_shared_per_block_optin},
            {"max_registers_per_block", d.max_regs_per_block},
            {"max_registers_per_sm", d.max_regs_per_sm}};
}

struct ScratchSpec {
    std::string name;
    json bytes;          // expression
    bool zero = false;   // under ScratchPolicy::Declared, cleared inside the timed op
    CUdeviceptr ptr = 0;
    size_t size = 0;
};

template <typename Base>
class ManifestKernel : public Base, public ManifestInfo {
public:
    ManifestKernel(json spec, std::string path, std::string source, std::string name)
        : spec_(std::move(spec)), path_(std::move(path)),
          source_(std::move(source)), name_(std::move(name)) {
        for (const auto& s : spec_.value("scratch", json::array())) {
            scratch_.push_back({s.at("name").get<std::string>(), s.at("bytes"), s.value("zero", false)});
        }
    }

    std::string name() const override { return name_; }
    std::string description() const override {
        return spec_.value("description", "manifest kernel from " + fs::path(path_).filename().string());
    }
    std::string module_path() const override { return this->compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return source_; }
    bool uses_module() const override { return false; }

    std::string function_name() const override {
        const auto e = entry_points();
        return e.empty() ? std::string() : e.front();
    }

    const std::string& manifest_path() const override { return path_; }

    // Entry points in launch order. A launch without "entry" uses the name
    // the compiler reported, which is how DSL kernels are found.
    std::vector<std::string> entry_points() const override {
        std::vector<std::string> out;
        for (const auto& l : spec_.at("launches")) {
            out.push_back(l.contains("entry") ? l.at("entry").get<std::string>()
                                              : result_for(l).kernel_name);
        }
        return out;
    }

    std::vector<std::string> entry_sources() const override {
        std::vector<std::string> out;
        for (const auto& l : spec_.at("launches")) out.push_back(launch_key(l));
        return out;
    }

    json device_limits(Context&) override { return device_limits_json(query_device_limits()); }

    // Launches may name their own source, relative to the manifest; the
    // runner compiles those too. This is how a candidate combines kernels
    // from several Triton, cuTile or Warp files (one kernel per file), or
    // dispatches different sizes to different DSLs. A launch with "defines"
    // is a variant of its source (KernelCompiler::variant_path), compiled once
    // per distinct set of defines.
    std::vector<std::string> extra_sources() const override {
        std::vector<std::string> out;
        for (const auto& l : spec_.at("launches")) {
            const auto s = launch_key(l);
            if (s != source_ && std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
        }
        return out;
    }
    void set_extra_compile_results(const std::map<std::string, CompileResult>& r) override {
        extra_results_ = r;
    }

    void set_scratch_policy(ScratchPolicy p) override { policy_ = p; }

    void allocate(Context& ctx) override {
        Base::allocate(ctx);
        std::vector<int64_t> sizes;
        resolve_scratch(sizes);
        for (size_t i = 0; i < scratch_.size(); i++) {
            scratch_[i].size = static_cast<size_t>(sizes[i]);
            scratch_[i].ptr = ctx.allocate(scratch_[i].size);
        }

        auto buffers = this->named_buffers();
        for (const auto& s : scratch_) buffers[s.name] = s.ptr;
        std::vector<Resolved> resolved;
        resolve_launches(buffers, sizes, load_modules(), resolved);
        for (auto& r : resolved) {
            if (!r.applies) continue;
            // The same check --plan reports, so a launch the device cannot
            // run fails here with the reason instead of "invalid argument".
            if (!r.problems.empty()) throw std::runtime_error(problem_message(r));
            launches_.push_back({r.fn, r.cfg, std::move(r.args), r.index, r.entry});
        }
        if (launches_.empty()) throw std::runtime_error(kNoLaunchApplies);
    }

    json plan(Context& ctx) override {
        this->capture_device_props(ctx);   // sm_count, which Base::allocate would set
        json out = {{"params", this->problem_values()}, {"scratch", json::array()},
                    {"launches", json::array()}, {"error", ""}};
        std::vector<int64_t> sizes;
        std::vector<Resolved> resolved;
        try {
            resolve_scratch(sizes);
            auto buffers = this->named_buffers();   // null until allocate(), reported by name
            for (const auto& s : scratch_) buffers[s.name] = 0;
            resolve_launches(buffers, sizes, load_modules(), resolved);
            if (std::none_of(resolved.begin(), resolved.end(), [](const Resolved& r) { return r.applies; }))
                throw std::runtime_error(kNoLaunchApplies);
            for (const auto& r : resolved) {
                if (r.applies && !r.problems.empty()) throw std::runtime_error(problem_message(r));
            }
        } catch (const std::exception& e) {
            out["error"] = e.what();
        }
        loader_.unload_all();
        module_ = nullptr;

        for (size_t i = 0; i < sizes.size(); i++)
            out["scratch"].push_back({{"name", scratch_[i].name}, {"bytes", sizes[i]}});
        // Canonical like the sources, so a symlinked folder still gives "v1.cu"
        const fs::path dir = fs::weakly_canonical(fs::path(path_).parent_path());
        for (const auto& r : resolved) {
            json l = {{"index", r.index}, {"entry", r.entry},
                      {"source", fs::path(r.source).lexically_relative(dir).generic_string()},
                      {"applies", r.applies}};
            if (!r.defines.empty()) l["defines"] = r.defines;
            if (r.applies) {
                l["grid"]  = r.shape.grid;
                l["block"] = r.shape.block;
                l["shared_mem_bytes"] = r.shape.dynamic_shared_bytes;
                l["args"] = r.args_described;
                if (r.fn) {
                    l["resources"] = {{"registers", r.fn_limits.registers},
                                      {"static_shared_bytes", r.fn_limits.static_shared_bytes},
                                      {"max_threads_per_block", r.fn_limits.max_threads_per_block}};
                }
                l["problems"] = r.problems;
            }
            out["launches"].push_back(std::move(l));
        }
        return out;
    }

    // Legacy: every scratch buffer starts at zero on every launch, outside
    // the timed op, as the base resets its output. Declared: buffers marked
    // "zero" are cleared by execute(), inside the timed op, and the rest are
    // filled with 0xFF here so a kernel cannot rely on zeros it did not
    // declare.
    void initialize(Context& ctx) override {
        Base::initialize(ctx);
        prepare_scratch(ctx);
    }

    void reset(Context& ctx) override {
        Base::reset(ctx);
        prepare_scratch(ctx);
    }

    void cleanup(Context& ctx) override {
        for (auto& s : scratch_) {
            if (s.ptr) ctx.free(s.ptr);
            s.ptr = 0;
        }
        launches_.clear();
        loader_.unload_all();
        module_ = nullptr;
        Base::cleanup(ctx);
    }

    void execute(Context&) override {
        // On the stream the launches and the timing events use, so the
        // memset is part of the op's time. It is not a kernel, so it is not
        // in launch_count or gpu_ms.
        if (policy_ == ScratchPolicy::Declared) {
            for (const auto& s : scratch_) {
                if (s.zero && s.ptr)
                    check_cuda(cuMemsetD8Async(s.ptr, 0, s.size, nullptr), "cuMemsetD8Async (scratch)");
            }
        }
        for (auto& l : launches_) {
            const auto r = loader_.launch(l.fn, l.cfg, l.args.pointers().data());
            if (r.result != CUDA_SUCCESS) {
                const char* msg = nullptr;
                cuGetErrorString(r.result, &msg);
                std::string why = std::string("manifest launch ") + std::to_string(l.index) + " (" +
                                  l.entry + "): kernel launch failed: " + (msg ? msg : "unknown");
                // allocate() already refused what launch_problems() knows
                // about; say so, since "invalid argument" alone gives no lead.
                if (r.result == CUDA_ERROR_INVALID_VALUE)
                    why += " (the launch passed the grid, block and shared memory checks; "
                           "check the argument layout and the function's shared memory attribute)";
                throw std::runtime_error(why);
            }
        }
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        return launches_.empty() ? KernelLoader::LaunchConfig{} : launches_.front().cfg;
    }

    std::vector<void*> get_kernel_args() override { return {}; }

private:
    struct Launch {
        CUfunction fn = nullptr;
        KernelLoader::LaunchConfig cfg;
        manifest::PackedArgs args;
        size_t index = 0;
        std::string entry;
    };

    // One manifest launch worked out at the current problem size.
    struct Resolved {
        size_t index = 0;
        std::string entry, source;
        bool applies = true;
        CUfunction fn = nullptr;
        KernelLoader::LaunchConfig cfg;
        manifest::LaunchShape shape;
        manifest::FunctionLimits fn_limits;
        std::vector<std::string> problems;   // what the device or function cannot run
        std::map<std::string, int> defines;  // the launch's own "defines"
        manifest::PackedArgs args;
        json args_described;
    };

    static std::string problem_message(const Resolved& r) {
        std::string msg = "manifest launch " + std::to_string(r.index) + " (" + r.entry + "): ";
        for (size_t i = 0; i < r.problems.size(); i++) msg += (i ? "; " : "") + r.problems[i];
        return msg;
    }

    static constexpr const char* kNoLaunchApplies =
        "no manifest launch applies at this size (every \"when\" is 0)";

    // Scratch sizes in bytes, in declaration order; each may use the sizes
    // declared before it. Appends as it goes, so a caller that catches still
    // has the ones that resolved.
    void resolve_scratch(std::vector<int64_t>& sizes) const {
        auto vars = variables();
        for (const auto& s : scratch_) {
            int64_t b = 0;
            try {
                b = manifest::eval_json(s.bytes, vars);
            } catch (const std::exception& e) {
                throw std::runtime_error("scratch '" + s.name + "': " + e.what());
            }
            if (b <= 0) throw std::runtime_error("scratch '" + s.name + "' has size " + std::to_string(b));
            vars[s.name + "_bytes"] = b;
            sizes.push_back(b);
        }
    }

    // Keyed like extra_sources(): a variant is its own module.
    std::map<std::string, CUmodule> load_modules() {
        module_ = loader_.load_module(module_path());
        std::map<std::string, CUmodule> modules = {{source_, module_}};
        for (const auto& [src, cr] : extra_results_) modules[src] = loader_.load_module(cr.module_path);
        return modules;
    }

    // Works out every launch: whether its "when" applies and, if it does,
    // its function, launch config and packed arguments. Shared by allocate()
    // and plan() so the two cannot disagree. Appends to `out` as it goes and
    // throws on the first error, naming the launch.
    void resolve_launches(const std::map<std::string, CUdeviceptr>& buffers,
                          const std::vector<int64_t>& scratch_bytes,
                          const std::map<std::string, CUmodule>& modules,
                          std::vector<Resolved>& out) {
        const auto entries = entry_points();
        const auto& launches = spec_.at("launches");
        for (size_t i = 0; i < launches.size(); i++) {
            const auto& l = launches[i];
            Resolved r;
            r.index = i;
            r.entry = entries[i];
            r.source = launch_source(l);
            r.defines = launch_defines(l);
            // "when": launch only if the expression is non-zero for this
            // problem size, so one manifest can dispatch small and large
            // inputs to different kernels.
            if (l.contains("when")) {
                int64_t on = 0;
                try {
                    on = manifest::eval_json(l.at("when"), variables_for(l));
                } catch (const std::exception& e) {
                    throw std::runtime_error("manifest launch " + std::to_string(i) + " \"when\": " + e.what());
                }
                if (!on) {
                    r.applies = false;
                    out.push_back(std::move(r));
                    continue;
                }
            }
            r.fn = loader_.get_function(modules.at(launch_key(l)), r.entry);
            // Expressions see the constants of the source this launch uses.
            auto lvars = variables_for(l);
            for (size_t s = 0; s < scratch_.size(); s++) lvars[scratch_[s].name + "_bytes"] = scratch_bytes[s];

            manifest::ArgContext actx;
            for (const auto& [k, v] : buffers) actx.buffers[k] = v;
            actx.vars = lvars;
            actx.floats = float_values();
            actx.param_sizes = param_sizes(r.fn);

            try {
                r.shape = launch_shape(l, lvars);
                r.fn_limits = query_function_limits(r.fn);
                if (!device_limits_) device_limits_ = std::make_unique<manifest::DeviceLimits>(query_device_limits());
                r.problems = manifest::launch_problems(*device_limits_, r.fn_limits, r.shape);
                r.cfg = to_config(r.shape);
                r.args = manifest::pack_args(l.at("args"), actx, &r.args_described);
            } catch (const std::exception& e) {
                throw std::runtime_error("manifest launch " + std::to_string(i) + " (" +
                                         r.entry + "): " + e.what());
            }
            out.push_back(std::move(r));
        }
    }

    // Names an expression can use: problem sizes, the SM count, and what the
    // compiler reported for the launch's source, constants included.
    // A launch's own "defines" are laid over the reported constants, since a
    // CUDA build reports only its "// arena:" lines, not the -D values.
    manifest::Vars variables_for(const CompileResult& cr, const std::map<std::string, int>& defines) const {
        manifest::Vars v;
        for (const auto& [k, x] : this->problem_values()) v[k] = x;
        for (const auto& [k, x] : cr.constants) v[k] = x;
        for (const auto& [k, x] : defines) v[k] = x;
        v["sm_count"] = this->sm_count();
        v["num_warps"] = cr.num_warps;
        v["block_dim"] = cr.block_dim;
        v["shared_memory"] = cr.shared_memory;
        return v;
    }
    manifest::Vars variables() const { return variables_for(this->compile_result_, this->compile_options()); }
    manifest::Vars variables_for(const json& l) const {
        return variables_for(result_for(l), key_defines(l));
    }

    std::map<std::string, double> float_values() const {
        if constexpr (HasEps<Base>::value) return {{"eps", static_cast<double>(Base::kEps)}};
        else return {};
    }

    static std::map<std::string, int> launch_defines(const json& l) {
        return l.contains("defines") ? manifest::parse_defines(l.at("defines")) : std::map<std::string, int>{};
    }

    // Defines a launch's build uses: for the manifest's own source the run's
    // --define values, then the launch's "defines" over them.
    std::map<std::string, int> key_defines(const json& l) const {
        const auto own = launch_defines(l);
        if (own.empty()) return launch_source(l) == source_ ? this->compile_options() : own;
        std::map<std::string, int> d = launch_source(l) == source_ ? this->compile_options()
                                                                   : std::map<std::string, int>{};
        for (const auto& [k, v] : own) d[k] = v;
        return d;
    }

    // The source a launch's module comes from: its file, or a variant of it
    // when the launch has "defines". The manifest's own source without
    // launch defines is source_ itself, built by the runner.
    std::string launch_key(const json& l) const {
        const auto s = launch_source(l);
        if (!l.contains("defines") || launch_defines(l).empty()) return s;
        return KernelCompiler::variant_path(s, key_defines(l));
    }

    std::string launch_source(const json& l) const {
        if (!l.contains("source")) return source_;
        fs::path p = l.at("source").get<std::string>();
        if (p.is_relative()) p = fs::path(path_).parent_path() / p;
        return fs::weakly_canonical(p).string();
    }

    const CompileResult& result_for(const json& l) const {
        const auto s = launch_key(l);
        if (s == source_) return this->compile_result_;
        auto it = extra_results_.find(s);
        if (it == extra_results_.end()) throw std::runtime_error("launch source not compiled: " + s);
        return it->second;
    }

    // Grid, block and shared memory as evaluated. Sizes the device cannot
    // run are left for manifest::launch_problems() to report.
    static manifest::LaunchShape launch_shape(const json& l, const manifest::Vars& vars) {
        auto dims = [&](const char* key) {
            std::array<int64_t, 3> d{1, 1, 1};
            if (!l.contains(key)) throw std::runtime_error(std::string("missing \"") + key + "\"");
            const auto& j = l.at(key);
            if (!j.is_array() || j.empty() || j.size() > 3)
                throw std::runtime_error(std::string("\"") + key + "\" must be a list of 1 to 3 sizes");
            for (size_t i = 0; i < j.size(); i++) d[i] = manifest::eval_json(j[i], vars);
            return d;
        };
        manifest::LaunchShape s;
        s.grid = dims("grid");
        s.block = dims("block");
        s.dynamic_shared_bytes = l.contains("shared_mem") ? manifest::eval_json(l.at("shared_mem"), vars) : 0;
        return s;
    }

    // Only called on a shape without problems for a launch, so every value
    // fits; clamped anyway so --plan never prints a wrapped number.
    static KernelLoader::LaunchConfig to_config(const manifest::LaunchShape& s) {
        auto u = [](int64_t x) { return static_cast<unsigned>(std::clamp<int64_t>(x, 0, 0xFFFFFFFFLL)); };
        KernelLoader::LaunchConfig c;
        c.grid_x = u(s.grid[0]); c.grid_y = u(s.grid[1]); c.grid_z = u(s.grid[2]);
        c.block_x = u(s.block[0]); c.block_y = u(s.block[1]); c.block_z = u(s.block[2]);
        c.shared_mem_bytes = u(s.dynamic_shared_bytes);
        return c;
    }

    // Parameter sizes as the driver sees them, so arguments can be checked
    // against the compiled function rather than against what the manifest
    // assumes.
    static std::vector<size_t> param_sizes(CUfunction fn) {
        std::vector<size_t> out;
        for (size_t i = 0;; i++) {
            size_t offset = 0, size = 0;
            if (cuFuncGetParamInfo(fn, i, &offset, &size) != CUDA_SUCCESS) break;
            out.push_back(size);
        }
        return out;
    }

    void prepare_scratch(Context& ctx) {
        for (auto& s : scratch_) {
            if (!s.ptr) continue;
            if (policy_ == ScratchPolicy::Legacy) ctx.zero_device(s.ptr, s.size);
            else if (!s.zero) check_cuda(cuMemsetD8(s.ptr, 0xFF, s.size), "cuMemsetD8 (scratch fill)");
        }
    }

    json spec_;
    std::string path_, source_, name_;
    std::vector<ScratchSpec> scratch_;
    std::map<std::string, CompileResult> extra_results_;
    KernelLoader loader_;
    CUmodule module_ = nullptr;
    std::vector<Launch> launches_;
    std::unique_ptr<manifest::DeviceLimits> device_limits_;   // queried once
    ScratchPolicy policy_ = ScratchPolicy::Legacy;
};

}

bool looks_like_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    try {
        json j;
        f >> j;
        return j.is_object() && j.contains("category") && j.contains("source") && j.contains("launches");
    } catch (...) {
        return false;
    }
}

ManifestScan scan_manifests(const std::string& dir, const std::vector<std::string>& taken) {
    ManifestScan out;
    std::set<std::string> names(taken.begin(), taken.end());
    const std::string d = trim_path(dir);

    std::error_code ec;
    const fs::path root = fs::absolute(d, ec);
    if (!fs::exists(root, ec)) {
        out.errors.push_back("\"" + d + "\": no such folder");
        return out;
    }
    if (!fs::is_directory(root, ec)) {
        out.errors.push_back("\"" + d + "\": not a folder");
        return out;
    }

    std::vector<fs::path> found;
    auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
    for (; !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        const auto& p = it->path();
        const std::string name = p.filename().string();
        if (it->is_directory(ec)) {
            if (name == "inspections" || name == "obs" || name == "work" || name == "__pycache__" ||
                name.rfind("build", 0) == 0 || (!name.empty() && name[0] == '.')) {
                it.disable_recursion_pending();
            }
            continue;
        }
        if (p.extension() == ".json" && looks_like_manifest(p.string())) found.push_back(p);
    }
    std::sort(found.begin(), found.end());

    for (const auto& p : found) {
        try {
            auto d = load_manifest(p.string());
            json spec;
            std::ifstream(p) >> spec;
            std::string name = spec.contains("name") ? spec["name"].get<std::string>()
                             : p.parent_path().filename().string() + "/" + p.stem().string();
            std::string unique = name;
            for (int n = 2; names.count(unique); n++) unique = name + " #" + std::to_string(n);
            names.insert(unique);
            if (unique != d->name()) {
                // the descriptor's name comes from the spec; rebuild with the unique one
                spec["name"] = unique;
                const fs::path tmp = p;
                d = load_manifest_spec(spec, tmp.string());
            }
            out.kernels.push_back(std::move(d));
        } catch (const std::exception& e) {
            out.errors.push_back(p.string() + ": " + e.what());
        }
    }
    return out;
}

std::unique_ptr<KernelDescriptor> load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read manifest " + path);
    json spec;
    try {
        f >> spec;
    } catch (const json::exception& e) {
        throw std::runtime_error("manifest " + path + " is not valid JSON: " + e.what());
    }
    return load_manifest_spec(spec, path);
}

std::unique_ptr<KernelDescriptor> load_manifest_spec(const json& spec, const std::string& path) {
    auto require = [&](const char* key) {
        if (!spec.contains(key)) throw std::runtime_error("manifest " + path + " has no \"" + key + "\"");
    };
    require("category");
    require("source");
    require("launches");
    if (!spec.at("launches").is_array() || spec.at("launches").empty())
        throw std::runtime_error("manifest " + path + ": \"launches\" must be a non-empty list");

    const fs::path mpath = fs::absolute(path);
    fs::path src = spec.at("source").get<std::string>();
    if (src.is_relative()) src = mpath.parent_path() / src;
    src = fs::weakly_canonical(src);
    if (!fs::exists(src)) throw std::runtime_error("manifest source not found: " + src.string());

    for (const auto& s : spec.value("scratch", json::array())) {
        if (!s.is_object() || !s.contains("name") || !s.at("name").is_string() || !s.contains("bytes"))
            throw std::runtime_error("manifest " + path + ": each scratch entry needs \"name\" and \"bytes\"");
        if (s.contains("zero") && !s.at("zero").is_boolean())
            throw std::runtime_error("manifest " + path + ": scratch \"zero\" must be true or false");
    }
    const auto& launches = spec.at("launches");
    for (size_t i = 0; i < launches.size(); i++) {
        const auto& l = launches[i];
        fs::path ls = src;
        if (l.contains("source")) {
            ls = l.at("source").get<std::string>();
            if (ls.is_relative()) ls = mpath.parent_path() / ls;
            if (!fs::exists(ls)) throw std::runtime_error("launch source not found: " + ls.string());
        }
        if (!l.contains("defines")) continue;
        try {
            manifest::parse_defines(l.at("defines"));
        } catch (const std::exception& e) {
            throw std::runtime_error("manifest " + path + " launch " + std::to_string(i) + ": " + e.what());
        }
        // warp_base.py accepts --define but ignores it, so a Warp variant
        // would silently be the source's own build.
        if (KernelCompiler::get_extension(ls.string()) == ".warp.py" && !l.at("defines").empty())
            throw std::runtime_error("manifest " + path + " launch " + std::to_string(i) +
                                     ": \"defines\" do not reach Warp kernels; Warp fixes its constants in the source");
    }
    const std::string name = spec.value("name", mpath.stem().string());
    const std::string category = spec.at("category").get<std::string>();

    std::unique_ptr<KernelDescriptor> d;
    if (category == "reduce")       d = std::make_unique<ManifestKernel<ReduceDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "scan")    d = std::make_unique<ManifestKernel<ScanDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "softmax") d = std::make_unique<ManifestKernel<SoftmaxDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "layernorm") d = std::make_unique<ManifestKernel<LayerNormDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "matmul")  d = std::make_unique<ManifestKernel<MatmulDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "attention") d = std::make_unique<ManifestKernel<AttentionDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else throw std::runtime_error("manifest category '" + category + "' is not one of reduce, scan, softmax, layernorm, matmul, attention");
    d->set_descriptor_file(mpath.string());
    return d;
}

}

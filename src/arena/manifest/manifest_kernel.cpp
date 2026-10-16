#include "arena/manifest/manifest_kernel.hpp"
#include "arena/manifest/args.hpp"
#include "arena/manifest/expr.hpp"
#include "arena/paths.hpp"
#include "arena/categories/reduce_base.hpp"
#include "arena/categories/scan_base.hpp"
#include "arena/categories/softmax_base.hpp"
#include "arena/categories/matmul_base.hpp"
#include "arena/categories/attention_base.hpp"
#include "arena/device/utils.hpp"

#include <nlohmann/json.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace arena {

namespace fs = std::filesystem;
using nlohmann::json;

std::unique_ptr<KernelDescriptor> load_manifest_spec(const nlohmann::json& spec, const std::string& path);

namespace {

struct ScratchSpec {
    std::string name;
    json bytes;          // expression
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
            scratch_.push_back({s.at("name").get<std::string>(), s.at("bytes")});
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
        for (const auto& l : spec_.at("launches")) out.push_back(launch_source(l));
        return out;
    }

    // Launches may name their own source, relative to the manifest; the
    // runner compiles those too. This is how a candidate combines kernels
    // from several Triton, cuTile or Warp files (one kernel per file), or
    // dispatches different sizes to different DSLs.
    std::vector<std::string> extra_sources() const override {
        std::vector<std::string> out;
        for (const auto& l : spec_.at("launches")) {
            const auto s = launch_source(l);
            if (s != source_ && std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
        }
        return out;
    }
    void set_extra_compile_results(const std::map<std::string, CompileResult>& r) override {
        extra_results_ = r;
    }

    void allocate(Context& ctx) override {
        Base::allocate(ctx);
        auto vars = variables();
        for (auto& s : scratch_) {
            const int64_t b = manifest::eval_json(s.bytes, vars);
            if (b <= 0) throw std::runtime_error("scratch '" + s.name + "' has size " + std::to_string(b));
            s.size = static_cast<size_t>(b);
            s.ptr = ctx.allocate(s.size);
            vars[s.name + "_bytes"] = b;
        }

        module_ = loader_.load_module(module_path());
        std::map<std::string, CUmodule> modules = {{source_, module_}};
        for (const auto& [src, cr] : extra_results_) modules[src] = loader_.load_module(cr.module_path);
        const auto entries = entry_points();
        const auto& launches = spec_.at("launches");
        for (size_t i = 0; i < launches.size(); i++) {
            const auto& l = launches[i];
            // "when": launch only if the expression is non-zero for this
            // problem size, so one manifest can dispatch small and large
            // inputs to different kernels.
            if (l.contains("when")) {
                int64_t on = 0;
                try {
                    on = manifest::eval_json(l.at("when"), variables_for(result_for(l)));
                } catch (const std::exception& e) {
                    throw std::runtime_error("manifest launch " + std::to_string(i) + " \"when\": " + e.what());
                }
                if (!on) continue;
            }
            Launch launch;
            const std::string src = launch_source(l);
            launch.fn = loader_.get_function(modules.at(src), entries[i]);
            // Expressions see the constants of the source this launch uses.
            auto lvars = variables_for(result_for(l));
            for (const auto& s : scratch_) lvars[s.name + "_bytes"] = (int64_t)s.size;

            manifest::ArgContext actx;
            for (const auto& [k, v] : this->named_buffers()) actx.buffers[k] = v;
            for (const auto& s : scratch_) actx.buffers[s.name] = s.ptr;
            actx.vars = lvars;
            actx.param_sizes = param_sizes(launch.fn);

            try {
                launch.cfg = launch_config(l, lvars);
                launch.args = manifest::pack_args(l.at("args"), actx);
            } catch (const std::exception& e) {
                throw std::runtime_error("manifest launch " + std::to_string(i) + " (" +
                                         entries[i] + "): " + e.what());
            }
            launches_.push_back(std::move(launch));
        }
        if (launches_.empty()) {
            throw std::runtime_error("no manifest launch applies at this size (every \"when\" is 0)");
        }
    }

    void initialize(Context& ctx) override {
        Base::initialize(ctx);
        zero_scratch(ctx);
    }

    // Scratch starts at zero on every launch, as the base resets its output.
    void reset(Context& ctx) override {
        Base::reset(ctx);
        zero_scratch(ctx);
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
        for (auto& l : launches_) {
            const auto r = loader_.launch(l.fn, l.cfg, l.args.pointers().data());
            if (r.result != CUDA_SUCCESS) {
                const char* msg = nullptr;
                cuGetErrorString(r.result, &msg);
                throw std::runtime_error(std::string("kernel launch failed: ") + (msg ? msg : "unknown"));
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
    };

    // Names an expression can use: problem sizes, the SM count, and what the
    // compiler reported for the launch's source, constants included.
    manifest::Vars variables_for(const CompileResult& cr) const {
        manifest::Vars v;
        for (const auto& [k, x] : this->problem_values()) v[k] = x;
        for (const auto& [k, x] : cr.constants) v[k] = x;
        v["sm_count"] = this->sm_count();
        v["num_warps"] = cr.num_warps;
        v["block_dim"] = cr.block_dim;
        v["shared_memory"] = cr.shared_memory;
        return v;
    }
    manifest::Vars variables() const { return variables_for(this->compile_result_); }

    std::string launch_source(const json& l) const {
        if (!l.contains("source")) return source_;
        fs::path p = l.at("source").get<std::string>();
        if (p.is_relative()) p = fs::path(path_).parent_path() / p;
        return fs::weakly_canonical(p).string();
    }

    const CompileResult& result_for(const json& l) const {
        const auto s = launch_source(l);
        if (s == source_) return this->compile_result_;
        auto it = extra_results_.find(s);
        if (it == extra_results_.end()) throw std::runtime_error("launch source not compiled: " + s);
        return it->second;
    }

    static KernelLoader::LaunchConfig launch_config(const json& l, const manifest::Vars& vars) {
        auto dims = [&](const char* key) {
            std::array<int64_t, 3> d{1, 1, 1};
            if (!l.contains(key)) throw std::runtime_error(std::string("missing \"") + key + "\"");
            const auto& j = l.at(key);
            if (!j.is_array() || j.empty() || j.size() > 3)
                throw std::runtime_error(std::string("\"") + key + "\" must be a list of 1 to 3 sizes");
            for (size_t i = 0; i < j.size(); i++) d[i] = manifest::eval_json(j[i], vars);
            for (auto x : d)
                if (x <= 0) throw std::runtime_error(std::string("\"") + key + "\" has a size <= 0");
            return d;
        };
        const auto g = dims("grid");
        const auto b = dims("block");
        if (b[0] * b[1] * b[2] > 1024) throw std::runtime_error("block has more than 1024 threads");
        const int64_t shmem = l.contains("shared_mem") ? manifest::eval_json(l.at("shared_mem"), vars) : 0;
        if (shmem < 0) throw std::runtime_error("shared_mem is negative");

        KernelLoader::LaunchConfig c;
        c.grid_x = (unsigned)g[0]; c.grid_y = (unsigned)g[1]; c.grid_z = (unsigned)g[2];
        c.block_x = (unsigned)b[0]; c.block_y = (unsigned)b[1]; c.block_z = (unsigned)b[2];
        c.shared_mem_bytes = (unsigned)shmem;
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

    void zero_scratch(Context& ctx) {
        for (auto& s : scratch_) if (s.ptr) ctx.zero_device(s.ptr, s.size);
    }

    json spec_;
    std::string path_, source_, name_;
    std::vector<ScratchSpec> scratch_;
    std::map<std::string, CompileResult> extra_results_;
    KernelLoader loader_;
    CUmodule module_ = nullptr;
    std::vector<Launch> launches_;
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

    for (const auto& l : spec.at("launches")) {
        if (!l.contains("source")) continue;
        fs::path ls = l.at("source").get<std::string>();
        if (ls.is_relative()) ls = mpath.parent_path() / ls;
        if (!fs::exists(ls)) throw std::runtime_error("launch source not found: " + ls.string());
    }
    const std::string name = spec.value("name", mpath.stem().string());
    const std::string category = spec.at("category").get<std::string>();

    std::unique_ptr<KernelDescriptor> d;
    if (category == "reduce")       d = std::make_unique<ManifestKernel<ReduceDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "scan")    d = std::make_unique<ManifestKernel<ScanDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "softmax") d = std::make_unique<ManifestKernel<SoftmaxDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "matmul")  d = std::make_unique<ManifestKernel<MatmulDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else if (category == "attention") d = std::make_unique<ManifestKernel<AttentionDescriptorBase>>(spec, mpath.string(), src.string(), name);
    else throw std::runtime_error("manifest category '" + category + "' is not one of reduce, scan, softmax, matmul, attention");
    d->set_descriptor_file(mpath.string());
    return d;
}

}

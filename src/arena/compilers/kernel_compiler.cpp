#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include "arena/compilers/cache_key.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <stdexcept>

namespace arena {

namespace fs = std::filesystem;

namespace {
const char* kSidecarExtensions[] = {".ptx", ".ttgir", ".warp_gen.cu"};
}

std::vector<std::string> KernelCompiler::sidecars(const std::string& source_path,
                                                  const CompileDefines& defines) const {
    std::vector<std::string> out;
    const auto base = cache_dir_ + "/" + derive_output_name(source_path) + defines_suffix(defines);
    for (const char* ext : kSidecarExtensions) {
        const fs::path p = base + ext;
        if (fs::exists(p)) out.push_back(fs::absolute(p).string());
    }
    return out;
}

std::string defines_suffix(const CompileDefines& defines) {
    // Sorted, because std::map is, so the same set of knobs always names the
    // same cubin no matter what order they were set in.
    std::string out;
    for (const auto& [key, val] : defines) {
        out += "_" + key + std::to_string(val);
    }
    return out;
}

KernelCompiler::KernelCompiler(const std::string& cache_dir)
    : cache_dir_(cache_dir) {}

void KernelCompiler::register_compiler(const std::string& extension,
                                        std::unique_ptr<CompilerBackend> compiler) {
    compilers_[extension] = std::move(compiler);
}

std::string KernelCompiler::missing_toolchain(const std::string& source_path) {
    try {
        const std::string id = backend_for(source_path).toolchain_id();
        if (id.find(" missing ") != std::string::npos) return id.substr(0, id.find(' '));
    } catch (...) {
    }
    return "";
}

CompilerBackend& KernelCompiler::backend_for(const std::string& source_path) {
    auto ext = get_extension(source_path);
    auto it = compilers_.find(ext);
    if (it == compilers_.end()) {
        throw std::runtime_error("No compiler registered for extension: " + ext);
    }
    return *it->second;
}

std::string KernelCompiler::fingerprint(const std::string& source_path,
                                        const CompileDefines& defines) {
    auto& backend = backend_for(source_path);
    BuildInputs in;
    in.source_text = read_file(resolve_source(ARENA_KERNEL_DIR, source_path));
    if (in.source_text.empty()) {
        throw std::runtime_error("Cannot read kernel source: " +
                                 resolve_source(ARENA_KERNEL_DIR, source_path));
    }
    for (const auto& dep : backend.dependencies()) {
        in.dependency_texts.push_back(read_file(dep));
    }
    in.toolchain = backend.toolchain_id();
    in.arch = arch_;
    in.defines = defines;
    return build_fingerprint(in);
}

CompileResult KernelCompiler::compile(const std::string& source_path,
                                      const CompileDefines& defines) {
    auto log = spdlog::get("compiler");

    // The fingerprint covers the source text, so an edited file never hits a
    // stale in-memory entry either. Hashing a kernel source costs microseconds.
    const std::string fp = fingerprint(source_path, defines);
    const std::string key = source_path + defines_suffix(defines) + "#" + fp;

    auto mem_it = cache_.find(key);
    if (mem_it != cache_.end()) {
        log->debug("{}: in-memory cache hit", source_path);
        auto hit = mem_it->second;
        hit.cache_hit = true;
        hit.compile_ms = 0.0f;
        hit.import_ms  = 0.0f;
        hit.invoke_ms  = 0.0f;
        return hit;
    }

    auto& backend = backend_for(source_path);
    fs::create_directories(cache_dir_);
    auto output_name = derive_output_name(source_path) + defines_suffix(defines);

    CompileResult result;
    if (try_disk_cache(output_name, fp, result)) {
        log->info("{}: using cached {} (kernel={})",
            source_path, result.module_path, result.kernel_name);
        result.cache_hit = true;
        result.compile_ms = 0.0f;
        result.import_ms  = 0.0f;
        result.invoke_ms  = 0.0f;
        cache_[key] = result;
        return result;
    }

    // Sidecars from an earlier build would otherwise survive next to a fresh
    // cubin and describe a different program.
    std::error_code ec;
    for (const char* ext : kSidecarExtensions) fs::remove(cache_dir_ + "/" + output_name + ext, ec);

    log->info("{}: compiling ({} compiler) ...", source_path, get_extension(source_path));
    auto t0 = std::chrono::high_resolution_clock::now();
    result = backend.compile(source_path, output_name, cache_dir_, defines);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.cache_hit = false;
    result.invoke_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    log->info("{}: compiled -> {} (kernel={})",
        source_path, result.module_path, result.kernel_name);

    save_disk_cache(source_path, output_name, fp, result);
    cache_[key] = result;
    return result;
}

std::string KernelCompiler::emit_ptx(const std::string& source_path,
                                     const CompileDefines& defines) {
    compile(source_path, defines);
    auto output_name = derive_output_name(source_path) + defines_suffix(defines);
    return backend_for(source_path).emit_ptx(source_path, output_name, cache_dir_, defines);
}

void KernelCompiler::clear_cache() {
    auto log = spdlog::get("compiler");
    cache_.clear();

    if (fs::exists(cache_dir_)) {
        int removed = 0;
        for (auto& entry : fs::directory_iterator(cache_dir_)) {
            auto ext = entry.path().extension().string();
            if (ext == ".cubin" || ext == ".json" || ext == ".ptx" || ext == ".ttgir" || ext == ".cu") {
                fs::remove(entry.path());
                removed++;
            }
        }
        log->info("Cleared compile cache: {} files removed from {}/", removed, cache_dir_);
    } else {
        log->info("Compile cache is empty (nothing to clear)");
    }
}

bool KernelCompiler::try_disk_cache(const std::string& output_name,
                                     const std::string& fingerprint,
                                     CompileResult& result) {
    auto meta_path = cache_dir_ + "/" + output_name + ".json";
    auto cubin_path = cache_dir_ + "/" + output_name + ".cubin";

    if (!fs::exists(meta_path) || !fs::exists(cubin_path)) return false;

    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) return false;

    nlohmann::json j;
    try { meta_file >> j; } catch (...) { return false; }

    // Entries written before fingerprints existed carry none and rebuild once.
    if (j.value("fingerprint", "") != fingerprint) return false;

    result = result_from_json(j, cubin_path);
    return true;
}

void KernelCompiler::save_disk_cache(const std::string& source_path,
                                      const std::string& output_name,
                                      const std::string& fingerprint,
                                      const CompileResult& result) {
    auto meta_path = cache_dir_ + "/" + output_name + ".json";
    nlohmann::json j;
    j["kernel_name"]   = result.kernel_name;
    j["num_warps"]     = result.num_warps;
    j["shared_memory"] = result.shared_memory;
    j["num_params"]    = result.num_params;
    j["block_dim"]     = result.block_dim;
    j["constants"]     = result.constants;
    j["compile_ms"]    = result.compile_ms;
    j["import_ms"]     = result.import_ms;
    j["fingerprint"]   = fingerprint;
    j["arch"]          = arch_;
    j["toolchain"]     = backend_for(source_path).toolchain_id();
    j["source_full_path"] = resolve_source(ARENA_KERNEL_DIR, source_path);

    std::ofstream meta_file(meta_path);
    meta_file << j.dump(2) << std::endl;
}

std::string KernelCompiler::derive_output_name(const std::string& source_path) {
    // "reduce/fp32/baseline.cu"        -> "reduce_fp32_baseline"
    // "reduce/fp32/reduce.triton.py"   -> "reduce_fp32_triton_reduce"
    // "/abs/path/cand_3.cu"            -> "ext_<hash of path>_cand_3"
    //
    // Every directory component is folded into the name, not just the parent,
    // so the dtype folder keeps fp32 and fp16 builds of the same kernel from
    // overwriting each other's cubin. A source outside the kernel dir would
    // make that an unreadable path, so it gets a short hash of the path.
    fs::path p(source_path);

    std::string prefix;
    if (p.is_absolute()) {
        prefix = "ext_" + hex64(fnv1a64(p.parent_path().string())).substr(0, 8);
    } else {
        for (const auto& part : p.parent_path()) {
            if (part.empty()) continue;
            if (!prefix.empty()) prefix += "_";
            prefix += part.string();
        }
    }

    std::string filename = p.filename().string();

    auto dot1 = filename.find('.');
    std::string name = filename.substr(0, dot1);

    std::string dsl;
    if (dot1 != std::string::npos) {
        auto dot2 = filename.find('.', dot1 + 1);
        if (dot2 != std::string::npos) {
            dsl = filename.substr(dot1 + 1, dot2 - dot1 - 1);
        }
    }

    if (dsl.empty()) return prefix + "_" + name;
    return prefix + "_" + dsl + "_" + name;
}

std::string KernelCompiler::get_extension(const std::string& source_path) {
    // "reduce/reduce.triton.py" -> ".triton.py"
    // "reduce/baseline.cu"      -> ".cu"
    auto filename = fs::path(source_path).filename().string();
    auto dot = filename.find('.');
    if (dot == std::string::npos) return "";
    return filename.substr(dot);
}

}

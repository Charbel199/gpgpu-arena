#include "arena/compilers/kernel_compiler.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <stdexcept>

namespace arena {

namespace fs = std::filesystem;

KernelCompiler::KernelCompiler(const std::string& cache_dir)
    : cache_dir_(cache_dir) {}

void KernelCompiler::register_compiler(const std::string& extension,
                                        std::unique_ptr<Compiler> compiler) {
    compilers_[extension] = std::move(compiler);
}

CompileResult KernelCompiler::compile(const std::string& source_path) {
    // check in-memory cache
    auto mem_it = cache_.find(source_path);
    if (mem_it != cache_.end()) {
        spdlog::get("compiler")->debug("{}: in-memory cache hit", source_path);
        auto hit = mem_it->second;
        hit.cache_hit = true;
        hit.compile_time_ms = 0.0f;
        return hit;
    }

    auto log = spdlog::get("compiler");
    auto ext = get_extension(source_path);
    auto comp_it = compilers_.find(ext);
    if (comp_it == compilers_.end()) {
        throw std::runtime_error("No compiler registered for extension: " + ext);
    }

    fs::create_directories(cache_dir_);
    auto output_name = derive_output_name(source_path);

    // fall back disk cache
    CompileResult result;
    if (try_disk_cache(source_path, output_name, result)) {
        log->info("{}: using cached {} (kernel={})",
            source_path, result.module_path, result.kernel_name);
        result.cache_hit = true;
        result.compile_time_ms = 0.0f;
        cache_[source_path] = result;
        return result;
    }

    // if nothing in cache -> compile
    log->info("{}: compiling ({} compiler) ...", source_path, ext);
    auto t0 = std::chrono::high_resolution_clock::now();
    result = comp_it->second->compile(source_path, output_name, cache_dir_);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.cache_hit = false;
    result.compile_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    log->info("{}: compiled -> {} (kernel={})",
        source_path, result.module_path, result.kernel_name);

    // save to both caches
    save_disk_cache(source_path, output_name, result);
    cache_[source_path] = result;
    return result;
}

void KernelCompiler::clear_cache() {
    auto log = spdlog::get("compiler");
    cache_.clear();

    if (fs::exists(cache_dir_)) {
        int removed = 0;
        for (auto& entry : fs::directory_iterator(cache_dir_)) {
            auto ext = entry.path().extension().string();
            if (ext == ".cubin" || ext == ".json") {
                fs::remove(entry.path());
                removed++;
            }
        }
        log->info("Cleared compile cache: {} files removed from {}/", removed, cache_dir_);
    } else {
        log->info("Compile cache is empty (nothing to clear)");
    }
}

bool KernelCompiler::try_disk_cache(const std::string& source_path,
                                     const std::string& output_name,
                                     CompileResult& result) {
    auto meta_path = cache_dir_ + "/" + output_name + ".json";
    auto cubin_path = cache_dir_ + "/" + output_name + ".cubin";

    if (!fs::exists(meta_path) || !fs::exists(cubin_path)) return false;

    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) return false;

    nlohmann::json j;
    try { meta_file >> j; } catch (...) { return false; }

    // Check source mtime against cached mtime
    auto source_full = j.value("source_full_path", "");
    auto cached_mtime = j.value("source_mtime", 0L);
    if (source_full.empty() || !fs::exists(source_full)) return false;

    auto current_mtime = fs::last_write_time(source_full).time_since_epoch().count();
    if (current_mtime != cached_mtime) return false;

    result = result_from_json(j, cubin_path);
    return true;
}

void KernelCompiler::save_disk_cache(const std::string& source_path,
                                      const std::string& output_name,
                                      const CompileResult& result) {
    auto source_full = std::string(ARENA_KERNEL_DIR) + "/" + source_path;

    auto meta_path = cache_dir_ + "/" + output_name + ".json";
    nlohmann::json j;
    j["kernel_name"]   = result.kernel_name;
    j["num_warps"]     = result.num_warps;
    j["shared_memory"] = result.shared_memory;
    j["num_params"]    = result.num_params;
    j["block_dim"]     = result.block_dim;
    j["constants"]     = result.constants;

    if (fs::exists(source_full)) {
        j["source_full_path"] = source_full;
        j["source_mtime"] = fs::last_write_time(source_full).time_since_epoch().count();
    }

    std::ofstream meta_file(meta_path);
    meta_file << j.dump(2) << std::endl;
}

std::string KernelCompiler::derive_output_name(const std::string& source_path) {
    // "reduce/reduce.triton.py" -> "reduce_triton_reduce"
    // "reduce/baseline.cu"      -> "reduce_baseline"
    fs::path p(source_path);

    std::string category = p.parent_path().filename().string();
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

    if (dsl.empty()) return category + "_" + name;
    return category + "_" + dsl + "_" + name;
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

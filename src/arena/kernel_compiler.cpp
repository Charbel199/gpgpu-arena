#include "arena/kernel_compiler.hpp"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <array>

namespace arena {

// --- TritonCompiler ---

TritonCompiler::TritonCompiler(const std::string& kernel_dir)
    : kernel_dir_(kernel_dir) {}

CompileResult TritonCompiler::compile(const std::string& source_path,
                                       const std::string& output_name,
                                       const std::string& cache_dir) {
    auto full_source = kernel_dir_ + "/" + source_path;

    std::string cmd =
        "PYTHONPATH=" + kernel_dir_ +
        " " ARENA_PYTHON " " + full_source +
        " --output-dir " + cache_dir +
        " --output-name " + output_name +
        " 2>&1";

    // Run compiler and capture output
    std::array<char, 4096> buffer;
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("Failed to run Triton compiler for " + source_path);

    while (fgets(buffer.data(), buffer.size(), pipe)) {
        output += buffer.data();
    }

    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Triton compilation failed for " + source_path + ": " + output);
    }

    // Find JSON line in output (last line starting with '{')
    std::string json_line;
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line[0] == '{') json_line = line;
    }

    if (json_line.empty()) {
        throw std::runtime_error("No JSON metadata in Triton compiler output for " + source_path);
    }

    // Minimal JSON parsing for flat object with known keys
    auto json_str = [&](const std::string& key) -> std::string {
        auto pos = json_line.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = json_line.find(':', pos);
        auto start = json_line.find('"', pos + 1);
        auto end = json_line.find('"', start + 1);
        return json_line.substr(start + 1, end - start - 1);
    };

    auto json_int = [&](const std::string& key) -> int {
        auto pos = json_line.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = json_line.find(':', pos);
        pos++;
        while (pos < json_line.size() && json_line[pos] == ' ') pos++;
        return std::stoi(json_line.substr(pos));
    };

    CompileResult result;
    result.ptx_path      = cache_dir + "/" + output_name + ".ptx";
    result.kernel_name   = json_str("kernel_name");
    result.num_warps     = json_int("num_warps");
    result.shared_memory = json_int("shared_memory");
    result.num_params    = json_int("num_params");
    return result;
}

// --- KernelCompiler (dispatcher) ---

KernelCompiler::KernelCompiler(const std::string& cache_dir)
    : cache_dir_(cache_dir) {}

void KernelCompiler::register_compiler(const std::string& extension,
                                        std::unique_ptr<Compiler> compiler) {
    compilers_[extension] = std::move(compiler);
}

CompileResult KernelCompiler::compile(const std::string& source_path) {
    auto it = cache_.find(source_path);
    if (it != cache_.end()) return it->second;

    auto log = spdlog::get("compiler");
    auto ext = get_extension(source_path);
    auto comp_it = compilers_.find(ext);
    if (comp_it == compilers_.end()) {
        throw std::runtime_error("No compiler registered for extension: " + ext);
    }

    std::filesystem::create_directories(cache_dir_);
    auto output_name = derive_output_name(source_path);

    log->info("Compiling {} ...", source_path);
    auto result = comp_it->second->compile(source_path, output_name, cache_dir_);

    log->info("Compiled {} -> {} (kernel={}, warps={}, shmem={}, params={})",
        source_path, result.ptx_path, result.kernel_name,
        result.num_warps, result.shared_memory, result.num_params);

    cache_[source_path] = result;
    return result;
}

std::string KernelCompiler::derive_output_name(const std::string& source_path) {
    // "reduce/reduce.triton.py" -> "reduce_triton_reduce"
    namespace fs = std::filesystem;
    fs::path p(source_path);

    std::string category = p.parent_path().filename().string();
    std::string filename = p.filename().string();

    auto dot1 = filename.find('.');
    std::string name = filename.substr(0, dot1);

    std::string dsl;
    if (dot1 != std::string::npos) {
        auto dot2 = filename.find('.', dot1 + 1);
        if (dot2 != std::string::npos) {
            dsl = filename.substr(dot1 + 1, dot2 - dot1 - 1); // TODO: clean ugly code
        }
    }

    if (dsl.empty()) return category + "_" + name;
    return category + "_" + dsl + "_" + name;
}

std::string KernelCompiler::get_extension(const std::string& source_path) {
    // "reduce/reduce.triton.py" -> ".triton.py"
    auto filename = std::filesystem::path(source_path).filename().string();
    auto dot = filename.find('.');
    if (dot == std::string::npos) return "";
    return filename.substr(dot);
}

}

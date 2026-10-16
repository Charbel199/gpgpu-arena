#include "arena/compilers/compiler_utils.hpp"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <array>
#include <fstream>
#include <regex>
#include <sys/wait.h>

namespace arena {

std::string run_command(const std::string& cmd, const std::string& context) {
    auto log = spdlog::get("compiler");
    log->debug("exec: {}", cmd);

    std::array<char, 4096> buffer;
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("Failed to run " + context);

    while (fgets(buffer.data(), buffer.size(), pipe)) {
        output += buffer.data();
    }

    int status = pclose(pipe);
    if (status != 0) {
        log->error("{}: exit code {}", context, WEXITSTATUS(status));
        throw std::runtime_error(context + " failed:\n" + output);
    }
    return output;
}

nlohmann::json parse_json_output(const std::string& output, const std::string& context) {
    std::string json_line;
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line[0] == '{') json_line = line;
    }
    if (json_line.empty()) {
        throw std::runtime_error("No JSON metadata in " + context + " output");
    }
    return nlohmann::json::parse(json_line);
}

CompileResult result_from_json(const nlohmann::json& j, const std::string& module_path) {
    CompileResult result;
    result.module_path   = module_path;
    result.kernel_name   = j.value("kernel_name", "");
    result.num_warps     = j.value("num_warps", 4);
    result.shared_memory = j.value("shared_memory", 0);
    result.num_params    = j.value("num_params", 3);
    result.block_dim     = j.value("block_dim", 0);
    result.compile_ms    = j.value("compile_ms", 0.0f);
    result.import_ms     = j.value("import_ms", 0.0f);

    if (j.contains("constants") && j["constants"].is_object()) {
        for (auto& [key, val] : j["constants"].items()) {
            result.constants[key] = val.get<int>();
        }
    }
    return result;
}


std::string defines_args(const CompileDefines& defines) {
    std::string out;
    for (const auto& [key, val] : defines) {
        out += " --define " + key + "=" + std::to_string(val);
    }
    return out;
}

std::map<std::string, int> parse_arena_directives(const std::string& source) {
    static const std::regex line_re(R"(arena:([^\n]*))");
    static const std::regex kv_re(R"(\b([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\b)");
    std::map<std::string, int> out;
    for (auto it = std::sregex_iterator(source.begin(), source.end(), line_re);
         it != std::sregex_iterator(); ++it) {
        const std::string rest = (*it)[1].str();
        for (auto kv = std::sregex_iterator(rest.begin(), rest.end(), kv_re);
             kv != std::sregex_iterator(); ++kv) {
            out[(*kv)[1].str()] = std::stoi((*kv)[2].str());
        }
    }
    return out;
}

std::string resolve_source(const std::string& kernel_dir, const std::string& source_path) {
    if (!source_path.empty() && source_path[0] == '/') return source_path;
    return kernel_dir + "/" + source_path;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out + "'";
}

}

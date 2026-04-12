#include "arena/compilers/compiler_utils.hpp"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <array>
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

    if (j.contains("constants") && j["constants"].is_object()) {
        for (auto& [key, val] : j["constants"].items()) {
            result.constants[key] = val.get<int>();
        }
    }
    return result;
}

}

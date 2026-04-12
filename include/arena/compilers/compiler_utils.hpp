#pragma once

#include "arena/compilers/compiler.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace arena {

// run a shell command, return stdout. Throws on non-zero exit.
std::string run_command(const std::string& cmd, const std::string& context);

// find last JSON line in output (line starting with '{')
nlohmann::json parse_json_output(const std::string& output, const std::string& context);

// build CompileResult from parsed JSON + module path
CompileResult result_from_json(const nlohmann::json& j, const std::string& module_path);

}

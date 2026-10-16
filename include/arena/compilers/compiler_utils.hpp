#pragma once

#include "arena/compilers/backend.hpp"
#include <nlohmann/json.hpp>
#include <map>
#include <string>

namespace arena {

// run a shell command, return stdout. Throws on non-zero exit.
std::string run_command(const std::string& cmd, const std::string& context);

// find last JSON line in output (line starting with '{')
nlohmann::json parse_json_output(const std::string& output, const std::string& context);

// build CompileResult from parsed JSON + module path
CompileResult result_from_json(const nlohmann::json& j, const std::string& module_path);

// " --define BLOCK_SIZE=512 --define num_warps=8", for the Python compilers.
std::string defines_args(const CompileDefines& defines);

// Absolute paths pass through; anything else is relative to the kernel dir.
std::string resolve_source(const std::string& kernel_dir, const std::string& source_path);

// Whole file as a string, or "" if it cannot be read.
std::string read_file(const std::string& path);

// Launch constants declared in a CUDA source, e.g.
//   // arena: BLOCK_M=128 BLOCK_N=64 THREADS=256
// Keys are upper case, values non-negative integers, several per line
// allowed. They become CompileResult::constants, which is how a CUDA
// descriptor can let the source choose its launch shape the way a Triton
// one reads its constexprs. Lower-case keys are left alone: `block=` is read
// by tools/pgaio.py, not here.
std::map<std::string, int> parse_arena_directives(const std::string& source);

// Single-quoted for /bin/sh, so paths with spaces survive popen().
std::string shell_quote(const std::string& s);

}

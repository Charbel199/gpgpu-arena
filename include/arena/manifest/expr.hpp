#pragma once

#include <nlohmann/json.hpp>
#include <cstdint>
#include <map>
#include <string>

// Integer expressions for kernel manifests: grid and block sizes, scratch
// sizes, scalar arguments. Small on purpose: integers, + - * / %, unary minus,
// parentheses, variables, and cdiv(a, b), min(a, ...), max(a, ...).
//
//   "cdiv(rows, ROWS_PER_PROGRAM)"    "num_warps * 32"    "64 + 16 * cdiv(n, 32)"
//
// A manifest is data the agent writes, never code, so nothing it says can
// run outside these operations. No CUDA here; unit-tested without a GPU.

namespace arena::manifest {

using Vars = std::map<std::string, int64_t>;

// Throws std::runtime_error naming the expression on a syntax error, an
// unknown variable or function, or division by zero.
int64_t eval(const std::string& expr, const Vars& vars);

// A JSON number is taken as is; a string is evaluated.
int64_t eval_json(const nlohmann::json& v, const Vars& vars);

}

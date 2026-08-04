#pragma once

#include <map>
#include <optional>
#include <string>

// Pure argument-parsing helpers: no CUDA, no I/O, unit-testable without a GPU.

namespace arena::cli {

// "n=1000000" -> {"n", 1000000}. Returns nullopt on a malformed spec:
// missing '=', empty key, empty value, non-numeric value, or trailing junk.
std::optional<std::pair<std::string, int>> parse_param(const std::string& spec);

// Applies a repeated --param spec onto an existing map. Returns false and
// leaves the map untouched if any spec is malformed.
bool apply_params(const std::string& spec, std::map<std::string, int>& out);

}

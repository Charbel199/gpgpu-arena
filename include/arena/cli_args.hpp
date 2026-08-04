#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

// Pure argument-parsing helpers: no CUDA, no I/O, unit-testable without a GPU.

namespace arena::cli {

// The block sizes to run one kernel at.
//
// Without --sweep-block that is a single entry, whatever --block asked for,
// with 0 meaning the descriptor's own default. With it, every size the kernel
// reports as tunable. Kernels whose cubin pins the block size report none, so
// they still get exactly one run rather than being dropped from the sweep.
std::vector<int> block_sizes_for(bool sweep, int requested,
                                 const std::vector<int>& tunable);

// "n=1000000" -> {"n", 1000000}. Returns nullopt on a malformed spec:
// missing '=', empty key, empty value, non-numeric value, or trailing junk.
std::optional<std::pair<std::string, int>> parse_param(const std::string& spec);

// Applies a repeated --param spec onto an existing map. Returns false and
// leaves the map untouched if any spec is malformed.
bool apply_params(const std::string& spec, std::map<std::string, int>& out);

}

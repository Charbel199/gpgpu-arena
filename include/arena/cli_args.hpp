#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

// Pure argument-parsing helpers: no CUDA, no I/O, unit-testable without a GPU.

namespace arena::cli {

// One run of a kernel at one point on the tuning axis. A hand-written CUDA
// kernel varies block_size and is relaunched; a DSL kernel varies defines and
// is recompiled. No kernel does both, but the caller does not have to care
// which kind it is holding.
struct TuningVariant {
    int block_size = 0;                    // 0 = descriptor default
    std::map<std::string, int> defines;    // empty = source defaults
};

// The variants to run one kernel at.
//
// Without a sweep that is a single entry carrying whatever --block and
// --define asked for. With one, it is every block size the kernel declares
// tunable, or failing that every compile-time config it declares. A kernel
// that declares neither still gets exactly one run rather than dropping out
// of the sweep.
std::vector<TuningVariant> tuning_variants_for(
    bool sweep,
    int requested_block,
    const std::map<std::string, int>& requested_defines,
    const std::vector<int>& tunable_blocks,
    const std::vector<std::map<std::string, int>>& tunable_defines);

// "n=1000000" -> {"n", 1000000}. Returns nullopt on a malformed spec:
// missing '=', empty key, empty value, non-numeric value, or trailing junk.
std::optional<std::pair<std::string, int>> parse_param(const std::string& spec);

// Applies a repeated --param spec onto an existing map. Returns false and
// leaves the map untouched if any spec is malformed.
bool apply_params(const std::string& spec, std::map<std::string, int>& out);

}

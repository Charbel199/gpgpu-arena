#pragma once

#include "arena/runner_config.hpp"

#include <cstdint>
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

// "n=1000,4000000" -> {"n", {1000, 4000000}}. A single value is a one-element
// list. Same rules as parse_param for each value; an empty item is malformed.
std::optional<std::pair<std::string, std::vector<int>>> parse_param_list(const std::string& spec);

// Applies a list spec onto a map of lists. Returns false on a malformed spec.
bool apply_param_list(const std::string& spec, std::map<std::string, std::vector<int>>& out);

// "rows=64,cols=1024" -> {{"rows", 64}, {"cols", 1024}}: one explicit problem
// shape, for size ladders that are not a cartesian grid. nullopt if any item
// is malformed or a key repeats.
std::optional<std::map<std::string, int>> parse_shape(const std::string& spec);

// "a,b,,c" -> {"a", "b", "c"}. Empty items are dropped.
std::vector<std::string> split_csv(const std::string& s);

// Comma-separated non-negative integers, e.g. a --seed list. nullopt if any
// item is not a number.
std::optional<std::vector<uint64_t>> parse_u64_list(const std::string& s);

// --lock-clocks value: a clock in MHz, or "auto" which returns 0. nullopt if
// it is neither a positive integer nor "auto".
std::optional<unsigned> parse_lock_clocks(const std::string& s);

// Cartesian product of per-parameter value lists, in key order, the last key
// varying fastest. {"M":{1,2},"N":{3}} -> [{M:1,N:3},{M:2,N:3}]. An empty map
// gives one empty combination, so a run with no -p still runs once.
std::vector<std::map<std::string, int>> expand_grid(
    const std::map<std::string, std::vector<int>>& lists);

// --timing-mode value: "v1" or "v2". nullopt for anything else.
std::optional<TimingMode> parse_timing_mode(const std::string& s);

// Removes every "--timing-mode <value>" pair from argv, in place, and
// returns the last value, or nullopt when there is none. main handles the flag
// itself, like --lock-clocks, because it applies to the whole process. A
// trailing --timing-mode with no value is left for the CLI to reject.
std::optional<std::string> take_timing_mode(int& argc, char** argv);

// "legacy" or "declared". nullopt for anything else.
std::optional<ScratchPolicy> parse_scratch_policy(const std::string& s);

// --plan resolves a manifest's launches without running anything, so it
// needs --manifest and does not mix with --run, --list or --inspect/--sass/
// --ptx. Returns the usage error, or "" when the combination is fine.
std::string plan_args_error(bool manifest, bool run, bool list, bool inspect);

}

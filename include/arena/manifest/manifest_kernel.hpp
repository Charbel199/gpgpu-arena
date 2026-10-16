#pragma once

#include "arena/kernel_descriptor.hpp"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <vector>

// A kernel described by a JSON manifest instead of a compiled C++ descriptor.
//
// The manifest names a category, a source file in any DSL, optional scratch
// buffers ({"name", "bytes", "zero"}, see ScratchPolicy), and one or more
// launches, each with an entry point, grid, block,
// shared memory and argument list (see manifest/args.hpp and expr.hpp):
//
//   {
//     "category": "softmax",
//     "source": "softmax_v3.triton.py",
//     "launches": [{
//       "grid":  ["cdiv(rows, ROWS_PER_PROGRAM)"],
//       "block": ["num_warps * 32"],
//       "shared_mem": "shared_memory",
//       "args":  ["input", "output", "rows", "cols", "..."]
//     }]
//   }
//
// A launch may also name its own "source" and carry "defines": {"BLOCK": 256}
// so one source compiles into several variants.
//
// Everything that decides the score stays with the category: input data,
// the CPU reference, verification, tolerance, and the byte and FLOP counts
// SOL is computed from. A manifest is data, so it can choose how the kernel
// is launched but cannot run code of its own, which is what lets an agent
// write one per candidate without being able to game the measurement.

namespace arena {

// Implemented by manifest kernels, for tools that want more than one entry.
class ManifestInfo {
public:
    virtual ~ManifestInfo() = default;
    virtual const std::string& manifest_path() const = 0;
    virtual std::vector<std::string> entry_points() const = 0;
    // Source of each entry point, in the same order, as extra_sources() names
    // it: a launch with "defines" gives KernelCompiler::variant_path().
    virtual std::vector<std::string> entry_sources() const = 0;

    // The device limits launches are checked against, for --plan:
    // {"max_threads_per_block", "max_block", "max_grid",
    //  "max_shared_per_block", "max_shared_per_block_optin",
    //  "max_registers_per_block", "max_registers_per_sm"}.
    virtual nlohmann::json device_limits(Context& ctx) = 0;

    // What a run would launch at the current problem size, without
    // allocating or launching anything: scratch sizes, and for each launch
    // whether it applies, its grid, block, shared memory and arguments
    // (buffers by name). Needs the compile results set and the modules
    // loadable, since entry points and parameter sizes come from the cubin.
    // Never throws for a manifest error: the first one goes in "error".
    // Each applying launch is checked against the device and the compiled
    // function (manifest/limits.hpp); a launch with "problems" makes them the
    // shape's error, naming the launch.
    //
    //   {"params": {...}, "scratch": [{"name", "bytes"}],
    //    "launches": [{"index", "entry", "source", "defines", "applies",
    //                  "grid", "block", "shared_mem_bytes", "args",
    //                  "resources": {"registers", "static_shared_bytes",
    //                                "max_threads_per_block"},
    //                  "problems": ["..."]}],
    //    "error": ""}
    virtual nlohmann::json plan(Context& ctx) = 0;
};

// Reads and validates a manifest. Relative "source" paths resolve against
// the manifest's directory. Throws std::runtime_error with the reason.
std::unique_ptr<KernelDescriptor> load_manifest(const std::string& path);

// True if the file parses as JSON with "category", "source" and "launches",
// which is how a folder scan tells manifests from other JSON.
bool looks_like_manifest(const std::string& path);

struct ManifestScan {
    std::vector<std::unique_ptr<KernelDescriptor>> kernels;
    std::vector<std::string> errors;   // "path: reason" for manifests that failed to load
};

// Loads every manifest under dir, recursively. Skips folders that only hold
// copies or compiler output (inspections, obs, work, __pycache__, build*,
// hidden folders), so opening a PGAIO run lists each measured candidate
// once. A manifest without "name" (PGAIO snapshots have one) is named
// "<parent folder>/<file stem>", made unique against `taken` and against the
// others in the scan.
ManifestScan scan_manifests(const std::string& dir, const std::vector<std::string>& taken);

}

#pragma once

#include "arena/kernel_descriptor.hpp"
#include <memory>
#include <string>
#include <vector>

// A kernel described by a JSON manifest instead of a compiled C++ descriptor.
//
// The manifest names a category, a source file in any DSL, optional scratch
// buffers, and one or more launches, each with an entry point, grid, block,
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
};

// Reads and validates a manifest. Relative "source" paths resolve against
// the manifest's directory. Throws std::runtime_error with the reason.
std::unique_ptr<KernelDescriptor> load_manifest(const std::string& path);

}

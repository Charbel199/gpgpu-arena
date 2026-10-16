#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Checks a manifest launch against what the device and the compiled function
// allow, so a launch that cannot run is reported with the reason instead of
// failing with the driver's "invalid argument". Pure, so it is unit-tested
// without a GPU; the caller fills the limits from cuDeviceGetAttribute and
// cuFuncGetAttribute.

namespace arena::manifest {

struct DeviceLimits {
    int64_t max_threads_per_block = 1024;
    std::array<int64_t, 3> max_block{1024, 1024, 64};
    std::array<int64_t, 3> max_grid{2147483647, 65535, 65535};
    int64_t max_shared_per_block = 48 * 1024;          // without opting in
    int64_t max_shared_per_block_optin = 48 * 1024;    // static + dynamic, after opting in
    int64_t max_regs_per_block = 65536;
    int64_t max_regs_per_sm = 65536;
};

struct FunctionLimits {
    int64_t registers = 0;               // per thread
    int64_t static_shared_bytes = 0;
    int64_t max_threads_per_block = 1024;   // lowered by register use
};

struct LaunchShape {
    std::array<int64_t, 3> grid{1, 1, 1};
    std::array<int64_t, 3> block{1, 1, 1};
    int64_t dynamic_shared_bytes = 0;
};

// One readable sentence per problem; empty when the launch fits.
inline std::vector<std::string> launch_problems(const DeviceLimits& dev, const FunctionLimits& fn,
                                                const LaunchShape& l) {
    std::vector<std::string> out;
    const char* axis[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; i++) {
        if (l.grid[i] <= 0)
            out.push_back("grid." + std::string(axis[i]) + " is " + std::to_string(l.grid[i]) +
                          ", every grid size must be at least 1");
        else if (l.grid[i] > dev.max_grid[i])
            out.push_back("grid." + std::string(axis[i]) + " is " + std::to_string(l.grid[i]) +
                          ", the device allows " + std::to_string(dev.max_grid[i]));
    }
    for (int i = 0; i < 3; i++) {
        if (l.block[i] <= 0)
            out.push_back("block." + std::string(axis[i]) + " is " + std::to_string(l.block[i]) +
                          ", every block size must be at least 1");
        else if (l.block[i] > dev.max_block[i])
            out.push_back("block." + std::string(axis[i]) + " is " + std::to_string(l.block[i]) +
                          ", the device allows " + std::to_string(dev.max_block[i]));
    }
    const bool block_ok = l.block[0] > 0 && l.block[1] > 0 && l.block[2] > 0;
    const int64_t threads = block_ok ? l.block[0] * l.block[1] * l.block[2] : 0;
    if (threads > dev.max_threads_per_block) {
        out.push_back("block has " + std::to_string(threads) + " threads, the device allows " +
                      std::to_string(dev.max_threads_per_block));
    } else if (threads > fn.max_threads_per_block) {
        out.push_back("block has " + std::to_string(threads) + " threads, but at " +
                      std::to_string(fn.registers) + " registers per thread the function can run at most " +
                      std::to_string(fn.max_threads_per_block) + " per block");
    }
    if (l.dynamic_shared_bytes < 0) {
        out.push_back("shared_mem is " + std::to_string(l.dynamic_shared_bytes) + ", it cannot be negative");
    } else {
        const int64_t total = l.dynamic_shared_bytes + fn.static_shared_bytes;
        if (total > dev.max_shared_per_block_optin)
            out.push_back("needs " + std::to_string(total) + " B of shared memory (" +
                          std::to_string(l.dynamic_shared_bytes) + " dynamic + " +
                          std::to_string(fn.static_shared_bytes) + " static), the device allows " +
                          std::to_string(dev.max_shared_per_block_optin) + " per block");
    }
    return out;
}

}

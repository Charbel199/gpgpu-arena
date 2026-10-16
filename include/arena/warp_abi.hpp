#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

// Warp ABI structs, mirroring warp/native/builtin.h and array.h, shared by
// every Warp descriptor.
//
// Verified against warp 1.16 with offsetof: launch_bounds_t<1> is 24 bytes
// (shape@0, size@8, coord_mult@16) and array_t<float> is 56 (data@0, grad@8,
// shape@16, strides@32, ndim@48).
//
// This layout changed in warp 1.13: shape went from a fixed int[4] plus an
// ndim field to int[N], and coord_mult was added. A kernel built against the
// old layout still launches, reads a garbage thread count, exits immediately
// and reports an impossible bandwidth, so check_warp_version() refuses to run
// against a version whose layout these structs do not match.

namespace arena {

struct WarpLaunchBounds {
    int32_t  shape[1];    // launch extent for a 1D kernel
    int32_t  _pad;
    uint64_t size;        // total thread count
    uint64_t coord_mult;  // threads per coord tuple; 1 for a plain 1D launch
};
static_assert(sizeof(WarpLaunchBounds) == 24, "warp launch_bounds_t<1> layout changed");

struct WarpArrayF32 {
    uint64_t data;      // CUdeviceptr
    uint64_t grad;      // 0 (no gradient)
    int32_t shape[4];
    int32_t strides[4];
    int32_t ndim;
    int32_t _pad;
};
static_assert(sizeof(WarpArrayF32) == 56, "warp array_t<float> layout changed");

// warp_base reports its version in the compile constants.
inline void check_warp_version(const std::map<std::string, int>& constants) {
    if (constants.count("warp_major") && constants.count("warp_minor")) {
        const int major = constants.at("warp_major");
        const int minor = constants.at("warp_minor");
        if (major != 1 || minor < 13) {
            throw std::runtime_error(
                "warp " + std::to_string(major) + "." + std::to_string(minor) +
                ": launch_bounds_t has a different layout than this descriptor "
                "expects (needs 1.13 or later)");
        }
    }
}

inline WarpLaunchBounds warp_bounds_1d(int32_t n) {
    WarpLaunchBounds b{};
    b.shape[0] = n;
    b.size = static_cast<uint64_t>(n);
    b.coord_mult = 1;
    return b;
}

inline WarpArrayF32 warp_array_1d(uint64_t ptr, int32_t n) {
    WarpArrayF32 a{};
    a.data = ptr;
    a.shape[0] = n;
    a.strides[0] = sizeof(float);
    a.ndim = 1;
    return a;
}

}

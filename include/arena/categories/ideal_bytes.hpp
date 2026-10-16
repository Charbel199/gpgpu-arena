#pragma once

#include "arena/dtype.hpp"

#include <cstddef>

// The least DRAM traffic one op of each category can get away with, split
// into bytes read and bytes written. Each total matches the category's
// calculate_bytes_accessed(), which bandwidth_gbps divides by op_ms.
//
// The split is there because writes can be deferred: a store with an
// evict_last policy can leave the output in L2 when the timed window closes,
// and the write-back is paid by whatever runs next. Knowing how much of the
// ideal traffic is writes lets a scorer tell that apart from a real speedup.
//
// Pure arithmetic, no CUDA, so the tests can check every category.

namespace arena {

struct IdealBytes {
    double read  = 0.0;   // per op
    double write = 0.0;   // per op
    // false for a descriptor that does not say, which is reported as unknown
    // rather than as zero traffic
    bool known() const { return read > 0.0 || write > 0.0; }
    double total() const { return read + write; }
};

namespace ideal {

// sum of n values: the input read once, one value written
inline IdealBytes reduce(size_t n, DType in, DType out) {
    return {static_cast<double>(dtype_buffer_bytes(n, in)),
            static_cast<double>(dtype_buffer_bytes(1, out))};
}

// exclusive scan, fp32: n values read, n written
inline IdealBytes scan(size_t n) {
    const double b = static_cast<double>(n) * sizeof(float);
    return {b, b};
}

// row softmax, fp32: the input read once, the output written once
inline IdealBytes softmax(size_t rows, size_t cols) {
    const double b = static_cast<double>(rows) * cols * sizeof(float);
    return {b, b};
}

// layernorm, fp32: x read once, y written once, gamma and beta read once in
// total since every row shares them
inline IdealBytes layernorm(size_t rows, size_t cols) {
    const double x = static_cast<double>(rows) * cols * sizeof(float);
    return {x + 2.0 * cols * sizeof(float), x};
}

// C = A B, fp32: A and B read once, C written once
inline IdealBytes matmul(size_t M, size_t K, size_t N) {
    return {(static_cast<double>(M) * K + static_cast<double>(K) * N) * sizeof(float),
            static_cast<double>(M) * N * sizeof(float)};
}

// fused attention, fp32: Q, K and V read once, O written once
inline IdealBytes attention(size_t BH, size_t S, size_t D) {
    const double t = static_cast<double>(BH) * S * D * sizeof(float);
    return {3.0 * t, t};
}

}

}

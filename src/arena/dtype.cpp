#include "arena/dtype.hpp"

namespace arena {

size_t dtype_size(DType d) {
    switch (d) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
    }
    return 4;
}

const char* dtype_name(DType d) {
    switch (d) {
        case DType::FP32: return "fp32";
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
    }
    return "fp32";
}

const char* compute_mode_name(ComputeMode m) {
    return m == ComputeMode::TF32 ? "tf32" : "default";
}

int dtype_mantissa_bits(DType d) {
    switch (d) {
        case DType::FP32: return 24;   // 23 stored + implicit leading 1
        case DType::FP16: return 11;   // 10 stored + implicit
        case DType::BF16: return 8;    //  7 stored + implicit
    }
    return 24;
}

double default_tolerance(DType d, ComputeMode m) {
    // Per-element floor: one unit in the last place with a factor of 8 of
    // slack. Callers scale this by the accumulation length, because error
    // grows with how many values were summed, not just with the type. See
    // KernelDescriptor::tolerance().
    const int bits = (m == ComputeMode::TF32)
        ? 11                            // tf32 keeps 10 mantissa bits + implicit
        : dtype_mantissa_bits(d);
    const double ulp = 1.0 / static_cast<double>(1ULL << (bits - 1));
    return ulp * 8.0;
}

}

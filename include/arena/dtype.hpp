#pragma once

#include <cstddef>
#include <string>

namespace arena {

// Storage type a kernel operates on. This is part of a kernel's identity, not
// a run-time axis: reduce_baseline and reduce_baseline_fp16 are separate
// descriptors that happen to share a category.
//
// tf32 is deliberately absent. It is a compute mode for fp32 storage on tensor
// cores, not a storage type, so it lives on ComputeMode instead.
enum class DType { FP32, FP16, BF16 };

// How tensor-core work is carried out for a given storage type. FP32 storage
// with TF32 compute is what cuTile's ct.mma does today, and it is why matmul
// needed a loosened tolerance before accuracy became a reported number.
enum class ComputeMode { Default, TF32 };

size_t      dtype_size(DType d);
const char* dtype_name(DType d);
const char* compute_mode_name(ComputeMode m);

// Mantissa bits including the implicit leading one. Drives the default
// tolerance and explains why an fp16 accumulator stops counting at 2048.
int dtype_mantissa_bits(DType d);

// Error budget a kernel of this type is expected to stay within. Derived from
// the mantissa rather than hand-picked, with headroom for accumulation order.
double default_tolerance(DType d, ComputeMode m = ComputeMode::Default);

}

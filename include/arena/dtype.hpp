#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

// Per-element error budget for this type, derived from the mantissa rather
// than hand-picked. Scale it by the accumulation length to get a kernel's
// tolerance: summing more values legitimately costs more precision.
double default_tolerance(DType d, ComputeMode m = ComputeMode::Default);

// IEEE 754 binary16 conversion, round to nearest even. Implemented in plain
// C++ rather than via cuda_fp16.h because descriptors are compiled by the host
// compiler, which does not see CUDA headers.
uint16_t float_to_half(float f);
float    half_to_float(uint16_t h);

// bfloat16: the top 16 bits of an fp32, so conversion is a shift plus rounding.
uint16_t float_to_bf16(float f);
float    bf16_to_float(uint16_t h);

// Round-trips a buffer through the storage type, giving the values the GPU
// will actually see. The CPU reference is built from these so a kernel is
// judged on its arithmetic, not on the input rounding it had no say in.
void quantize_in_place(std::vector<float>& v, DType d);

}

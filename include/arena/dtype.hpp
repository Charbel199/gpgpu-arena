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
enum class DType {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,   // OCP fp8: 4 exponent, 3 mantissa bits. No infinity, max 448.
    FP8_E5M2,   // OCP fp8: 5 exponent, 2 mantissa bits. IEEE-like, has inf.
    FP4_E2M1,   // 4 bits: 2 exponent, 1 mantissa. Magnitudes 0..6.
};

// How tensor-core work is carried out for a given storage type. FP32 storage
// with TF32 compute is what cuTile's ct.mma does today, and it is why matmul
// needed a loosened tolerance before accuracy became a reported number.
enum class ComputeMode { Default, TF32 };

// Bits per element. Not bytes: fp4 is half a byte, so anything sizing a
// buffer has to work in bits and round up.
int dtype_bits(DType d);

// Bytes needed to hold count elements, rounded up to a whole byte.
size_t dtype_buffer_bytes(size_t count, DType d);

// True for formats that carry a separate per-block scale tensor. NVFP4 is
// FP4_E2M1 plus an e4m3 scale per 16 elements, so a descriptor using it needs
// a second buffer the current base classes do not allocate.
bool dtype_is_block_scaled(DType d);
int  dtype_scale_block(DType d);
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

// fp8, both OCP variants. e4m3 has no infinity and saturates at 448.
uint8_t float_to_fp8_e4m3(float f);
float   fp8_e4m3_to_float(uint8_t b);
uint8_t float_to_fp8_e5m2(float f);
float   fp8_e5m2_to_float(uint8_t b);

// fp4 e2m1. Only 16 representable values, so this is a table lookup.
uint8_t float_to_fp4_e2m1(float f);
float   fp4_e2m1_to_float(uint8_t nibble);

// Pack floats into the storage type, two fp4 values per byte. Unpack reads
// one element back out by index.
std::vector<uint8_t> pack_values(const std::vector<float>& v, DType d);
float unpack_value(const void* buffer, size_t index, DType d);

// Round-trips a buffer through the storage type, giving the values the GPU
// will actually see. The CPU reference is built from these so a kernel is
// judged on its arithmetic, not on the input rounding it had no say in.
void quantize_in_place(std::vector<float>& v, DType d);

}

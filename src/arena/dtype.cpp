#include "arena/dtype.hpp"

#include <cstring>

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


namespace {
uint32_t bits_of(float f)   { uint32_t b; std::memcpy(&b, &f, 4); return b; }
float    float_of(uint32_t b) { float f;   std::memcpy(&f, &b, 4); return f; }
}

uint16_t float_to_half(float f) {
    const uint32_t x = bits_of(f);
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man  = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {           // inf or NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);   // overflow to inf
    if (exp <= 0) {                                // subnormal or zero
        if (exp < -10) return sign;                // too small even for subnormal
        const uint32_t sub = (man | 0x800000u) >> (1 - exp + 13);
        const uint32_t rem = (man | 0x800000u) >> (1 - exp + 12);
        return static_cast<uint16_t>(sign | (sub + (rem & 1u)));
    }

    uint16_t out = static_cast<uint16_t>(sign | (exp << 10) | (man >> 13));
    // round to nearest even on the 13 discarded bits
    const uint32_t discarded = man & 0x1FFFu;
    if (discarded > 0x1000u || (discarded == 0x1000u && ((man >> 13) & 1u))) out++;
    return out;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man = static_cast<uint32_t>(h & 0x3FFu);

    if (exp == 0) {
        if (man == 0) return float_of(sign);       // signed zero
        int e = -1;                                 // renormalise the subnormal
        do { man <<= 1; e++; } while ((man & 0x400u) == 0);
        man &= 0x3FFu;
        return float_of(sign | ((127 - 15 - e) << 23) | (man << 13));
    }
    if (exp == 0x1F) return float_of(sign | 0x7F800000u | (man << 13));
    return float_of(sign | ((exp - 15 + 127) << 23) | (man << 13));
}

uint16_t float_to_bf16(float f) {
    const uint32_t x = bits_of(f);
    if (((x >> 23) & 0xFFu) == 0xFFu && (x & 0x7FFFFFu)) {
        return static_cast<uint16_t>((x >> 16) | 0x40u);   // keep NaN a NaN
    }
    const uint32_t rounded = x + 0x7FFFu + ((x >> 16) & 1u);   // nearest even
    return static_cast<uint16_t>(rounded >> 16);
}

float bf16_to_float(uint16_t h) {
    return float_of(static_cast<uint32_t>(h) << 16);
}

void quantize_in_place(std::vector<float>& v, DType d) {
    switch (d) {
        case DType::FP32: return;
        case DType::FP16: for (float& x : v) x = half_to_float(float_to_half(x)); return;
        case DType::BF16: for (float& x : v) x = bf16_to_float(float_to_bf16(x)); return;
    }
}

}

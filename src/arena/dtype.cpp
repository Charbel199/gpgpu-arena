#include "arena/dtype.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

namespace arena {

int dtype_bits(DType d) {
    switch (d) {
        case DType::FP32:     return 32;
        case DType::FP16:     return 16;
        case DType::BF16:     return 16;
        case DType::FP8_E4M3: return 8;
        case DType::FP8_E5M2: return 8;
        case DType::FP4_E2M1: return 4;
    }
    return 32;
}

size_t dtype_buffer_bytes(size_t count, DType d) {
    const size_t bits = count * static_cast<size_t>(dtype_bits(d));
    return (bits + 7) / 8;
}

size_t dtype_size(DType d) {
    const int bits = dtype_bits(d);
    assert(bits % 8 == 0 && "sub-byte type: use dtype_buffer_bytes");
    return static_cast<size_t>(bits / 8);
}

bool dtype_is_block_scaled(DType d) {
    return d == DType::FP4_E2M1;
}

int dtype_scale_block(DType d) {
    // NVFP4 shares one e4m3 scale across 16 consecutive values.
    return d == DType::FP4_E2M1 ? 16 : 0;
}

const char* dtype_name(DType d) {
    switch (d) {
        case DType::FP32:     return "fp32";
        case DType::FP16:     return "fp16";
        case DType::BF16:     return "bf16";
        case DType::FP8_E4M3: return "fp8e4m3";
        case DType::FP8_E5M2: return "fp8e5m2";
        case DType::FP4_E2M1: return "fp4";
    }
    return "fp32";
}

const char* compute_mode_name(ComputeMode m) {
    return m == ComputeMode::TF32 ? "tf32" : "default";
}

int dtype_mantissa_bits(DType d) {
    switch (d) {
        case DType::FP32:     return 24;   // 23 stored + implicit leading 1
        case DType::FP16:     return 11;   // 10 stored + implicit
        case DType::BF16:     return 8;    //  7 stored + implicit
        case DType::FP8_E4M3: return 4;    //  3 stored + implicit
        case DType::FP8_E5M2: return 3;    //  2 stored + implicit
        case DType::FP4_E2M1: return 2;    //  1 stored + implicit
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
        case DType::FP8_E4M3: for (float& x : v) x = fp8_e4m3_to_float(float_to_fp8_e4m3(x)); return;
        case DType::FP8_E5M2: for (float& x : v) x = fp8_e5m2_to_float(float_to_fp8_e5m2(x)); return;
        case DType::FP4_E2M1: for (float& x : v) x = fp4_e2m1_to_float(float_to_fp4_e2m1(x)); return;
    }
}

// ---- fp8 ----------------------------------------------------------------
// Both OCP variants, done by routing through half: fp16 has more exponent
// range and mantissa than either, so the only rounding is the intended one.

namespace {
uint8_t narrow(float f, int exp_bits, int man_bits, int bias, bool has_inf, uint8_t max_finite) {
    const uint32_t x = bits_of(f);
    const uint8_t  sign = static_cast<uint8_t>((x >> 24) & 0x80u);
    const int32_t  e    = static_cast<int32_t>((x >> 23) & 0xFFu) - 127;
    const uint32_t man  = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {                 // inf or NaN in
        if (man) return static_cast<uint8_t>(sign | max_finite | 1u);   // NaN
        return has_inf ? static_cast<uint8_t>(sign | (((1u << exp_bits) - 1u) << man_bits))
                       : static_cast<uint8_t>(sign | max_finite);
    }

    int32_t be = e + bias;
    if (be >= (has_inf ? (1 << exp_bits) - 1 : (1 << exp_bits))) {
        return static_cast<uint8_t>(sign | max_finite);  // saturate
    }
    if (be <= 0) {                                       // subnormal or zero
        const int shift = 23 - man_bits + 1 - be;
        if (shift > 31) return sign;
        const uint32_t full = man | 0x800000u;
        uint32_t q = full >> shift;
        if ((full >> (shift - 1)) & 1u) q++;             // round to nearest
        return static_cast<uint8_t>(sign | q);
    }

    uint32_t q = man >> (23 - man_bits);
    const uint32_t rem = man & ((1u << (23 - man_bits)) - 1u);
    const uint32_t half = 1u << (22 - man_bits);
    if (rem > half || (rem == half && (q & 1u))) {       // round to nearest even
        q++;
        if (q >> man_bits) { q = 0; be++; }
        if (be >= (has_inf ? (1 << exp_bits) - 1 : (1 << exp_bits)))
            return static_cast<uint8_t>(sign | max_finite);
    }
    return static_cast<uint8_t>(sign | (be << man_bits) | q);
}

float widen(uint8_t b, int exp_bits, int man_bits, int bias, bool has_inf) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t emax = (1u << exp_bits) - 1u;
    uint32_t e = (b >> man_bits) & emax;
    uint32_t m = b & ((1u << man_bits) - 1u);

    if (has_inf && e == emax) {
        return float_of(sign | 0x7F800000u | (m << (23 - man_bits)));
    }
    if (!has_inf && e == emax && m == ((1u << man_bits) - 1u)) {
        return float_of(sign | 0x7FC00000u);             // e4m3 NaN
    }
    if (e == 0) {
        if (m == 0) return float_of(sign);
        int shift = 0;
        while (!(m & (1u << man_bits))) { m <<= 1; shift++; }
        m &= (1u << man_bits) - 1u;
        return float_of(sign | ((127 - bias - shift + 1) << 23) | (m << (23 - man_bits)));
    }
    return float_of(sign | ((e - bias + 127) << 23) | (m << (23 - man_bits)));
}
}

uint8_t float_to_fp8_e4m3(float f) { return narrow(f, 4, 3, 7, false, 0x7E); }
float   fp8_e4m3_to_float(uint8_t b) { return widen(b, 4, 3, 7, false); }
uint8_t float_to_fp8_e5m2(float f) { return narrow(f, 5, 2, 15, true, 0x7B); }
float   fp8_e5m2_to_float(uint8_t b) { return widen(b, 5, 2, 15, true); }

// ---- fp4 e2m1 -----------------------------------------------------------
// Only eight magnitudes, so a table is exact and cheaper than bit twiddling.

namespace { const float kFp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f}; }

uint8_t float_to_fp4_e2m1(float f) {
    const uint8_t sign = std::signbit(f) ? 0x8u : 0x0u;
    const float a = std::fabs(f);
    if (std::isnan(a)) return sign | 0x7u;              // no NaN encoding: saturate
    int best = 0;
    float bestd = std::fabs(a - kFp4[0]);
    for (int i = 1; i < 8; i++) {
        const float d = std::fabs(a - kFp4[i]);
        if (d < bestd) { bestd = d; best = i; }
    }
    return static_cast<uint8_t>(sign | best);
}

float fp4_e2m1_to_float(uint8_t nibble) {
    const float v = kFp4[nibble & 0x7u];
    return (nibble & 0x8u) ? -v : v;
}

// ---- packing ------------------------------------------------------------

std::vector<uint8_t> pack_values(const std::vector<float>& v, DType d) {
    std::vector<uint8_t> out(dtype_buffer_bytes(v.size(), d), 0);
    for (size_t i = 0; i < v.size(); i++) {
        switch (d) {
            case DType::FP32: {
                const uint32_t b = bits_of(v[i]);
                std::memcpy(out.data() + i * 4, &b, 4);
                break;
            }
            case DType::FP16: case DType::BF16: {
                const uint16_t b = (d == DType::FP16) ? float_to_half(v[i]) : float_to_bf16(v[i]);
                std::memcpy(out.data() + i * 2, &b, 2);
                break;
            }
            case DType::FP8_E4M3: out[i] = float_to_fp8_e4m3(v[i]); break;
            case DType::FP8_E5M2: out[i] = float_to_fp8_e5m2(v[i]); break;
            case DType::FP4_E2M1: {
                // Two per byte, low nibble first.
                const uint8_t n = float_to_fp4_e2m1(v[i]) & 0xFu;
                if (i % 2 == 0) out[i / 2] = n;
                else            out[i / 2] |= static_cast<uint8_t>(n << 4);
                break;
            }
        }
    }
    return out;
}

float unpack_value(const void* buffer, size_t index, DType d) {
    const uint8_t* p = static_cast<const uint8_t*>(buffer);
    switch (d) {
        case DType::FP32: { uint32_t b; std::memcpy(&b, p + index * 4, 4); return float_of(b); }
        case DType::FP16: { uint16_t b; std::memcpy(&b, p + index * 2, 2); return half_to_float(b); }
        case DType::BF16: { uint16_t b; std::memcpy(&b, p + index * 2, 2); return bf16_to_float(b); }
        case DType::FP8_E4M3: return fp8_e4m3_to_float(p[index]);
        case DType::FP8_E5M2: return fp8_e5m2_to_float(p[index]);
        case DType::FP4_E2M1: {
            const uint8_t byte = p[index / 2];
            return fp4_e2m1_to_float((index % 2 == 0) ? (byte & 0xFu) : (byte >> 4));
        }
    }
    return 0.0f;
}

}

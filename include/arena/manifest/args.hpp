#pragma once

#include "arena/manifest/expr.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <map>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Builds a kernel's launch parameters from a manifest's "args" list.
//
// Every entry draws from a fixed vocabulary, so a manifest can say what to
// pass but never compute or copy anything itself:
//
//   "input"                       a named device buffer of the category, or a
//                                 declared scratch buffer: one pointer
//   "rows"                        a problem size: one int32
//   "null"                        a null pointer
//   "..."                         null pointers until the cubin's parameter
//                                 count is reached (Triton appends scratch
//                                 parameters after the declared ones)
//   {"i32": expr} {"u32": expr} {"i64": expr}   a scalar of that width
//   {"cutile_array": "input", "shape": [...], "strides": [...]}
//                                 cuTile's expansion of an array: pointer,
//                                 then each shape, then each stride, as u32;
//                                 strides in elements, row-major by default
//   {"warp_array": "input", "shape": [...]}
//                                 Warp's array_t<float> struct, one parameter;
//                                 strides in bytes, contiguous by default
//   {"warp_bounds": expr}         Warp's 1D launch_bounds_t, one parameter
//
// No CUDA here; unit-tested without a GPU.

namespace arena::manifest {

struct ArgContext {
    std::map<std::string, uint64_t> buffers;   // device pointers by name
    Vars vars;                                 // sizes, constants, sm_count

    // Byte size of each parameter the compiled function takes, from the
    // driver. When known, every argument must match its parameter's size and
    // the count must match exactly, so a wrong layout is an error rather
    // than a launch that reads garbage. Empty means unknown (tests).
    std::vector<size_t> param_sizes;
};

// Owns the bytes the parameter pointers point at. Move-only, so the pointers
// stay valid for as long as the object lives.
class PackedArgs {
public:
    std::vector<void*>& pointers() { return ptrs_; }
    size_t size() const { return ptrs_.size(); }

    // Raw bytes of parameter i, for tests.
    const std::vector<uint8_t>& bytes(size_t i) const { return *store_.at(i); }

    size_t size_of(size_t i) const { return store_.at(i)->size(); }

    template <typename T>
    void push(const T& value) {
        auto b = std::make_unique<std::vector<uint8_t>>(sizeof(T));
        std::memcpy(b->data(), &value, sizeof(T));
        ptrs_.push_back(b->data());
        store_.push_back(std::move(b));
    }

private:
    std::vector<std::unique_ptr<std::vector<uint8_t>>> store_;
    std::vector<void*> ptrs_;
};

// Throws std::runtime_error naming the offending entry.
//
// If `described` is given, it receives one object per manifest entry saying
// what was passed, for --plan: {"buffer": name} for a pointer,
// {"name", "type", "value"} for a named size, {"type", "expr", "value"} for
// an explicit scalar, {"null": true}, {"padding": count} for "...", and the
// resolved shape and strides for cutile_array and warp_array.
PackedArgs pack_args(const nlohmann::json& args, const ArgContext& ctx,
                     nlohmann::json* described = nullptr);

}

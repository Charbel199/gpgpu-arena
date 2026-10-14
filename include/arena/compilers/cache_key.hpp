#pragma once

#include "arena/compilers/backend.hpp"
#include <cstdint>
#include <string>
#include <vector>

// Pure helpers for the compile cache key. No CUDA and no I/O, so they can be
// unit-tested without a GPU.

namespace arena {

uint64_t fnv1a64(const std::string& bytes, uint64_t seed = 0xcbf29ce484222325ULL);

// 16 lowercase hex digits.
std::string hex64(uint64_t v);

// Everything that decides what a cubin contains, folded into one string.
//
// The cache used to be keyed on the source mtime alone, so a cubin outlived
// the toolchain and the GPU that built it and was silently reused. Editing a
// shared base module (triton_base.py) did not invalidate anything either.
// Now any change to the source text, a dependency, the compiler, its flags,
// the target arch or the defines produces a different fingerprint.
struct BuildInputs {
    std::string source_text;
    std::vector<std::string> dependency_texts;   // e.g. the DSL's *_base.py
    std::string toolchain;                       // compiler + version + flags
    std::string arch;                            // e.g. "sm_120"
    CompileDefines defines;
};

std::string build_fingerprint(const BuildInputs& in);

}

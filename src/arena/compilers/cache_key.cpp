#include "arena/compilers/cache_key.hpp"

#include <cstdio>

namespace arena {

uint64_t fnv1a64(const std::string& bytes, uint64_t seed) {
    uint64_t h = seed;
    for (unsigned char c : bytes) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

std::string build_fingerprint(const BuildInputs& in) {
    // Each field is hashed with a length prefix so that moving bytes from one
    // field to the next cannot produce the same key.
    auto mix = [](uint64_t h, const std::string& s) {
        h = fnv1a64(std::to_string(s.size()) + ":", h);
        return fnv1a64(s, h);
    };

    uint64_t h = 0xcbf29ce484222325ULL;
    h = mix(h, in.source_text);
    for (const auto& d : in.dependency_texts) h = mix(h, d);
    h = mix(h, in.toolchain);
    h = mix(h, in.arch);
    for (const auto& [k, v] : in.defines) h = mix(h, k + "=" + std::to_string(v));
    return hex64(h);
}

}

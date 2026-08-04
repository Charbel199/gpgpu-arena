#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Input data generation. No CUDA, so the generators are unit-testable.

namespace arena {

enum class Distribution {
    Ones,          // every element 1.0. Fast sanity check, but hides ordering
                   // bugs and makes tree reductions look exact.
    Uniform,       // uniform in [-1, 1]. The sensible default.
    Normal,        // standard normal. Occasional large magnitudes.
    Adversarial,   // mixed magnitudes plus denormals and near-overflow values,
                   // to expose catastrophic cancellation and saturation.
};

Distribution  distribution_from_string(const std::string& s, bool* ok = nullptr);
const char*   distribution_name(Distribution d);

// Fills out with n values. Deterministic in seed, so a run is reproducible
// and the CPU reference can be regenerated exactly.
void generate(std::vector<float>& out, size_t n, Distribution d, uint64_t seed);

}

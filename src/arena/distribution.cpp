#include "arena/distribution.hpp"

#include <cmath>
#include <limits>
#include <random>

namespace arena {

Distribution distribution_from_string(const std::string& s, bool* ok) {
    if (ok) *ok = true;
    if (s == "ones")        return Distribution::Ones;
    if (s == "uniform")     return Distribution::Uniform;
    if (s == "normal")      return Distribution::Normal;
    if (s == "adversarial") return Distribution::Adversarial;
    if (ok) *ok = false;
    return Distribution::Uniform;
}

const char* distribution_name(Distribution d) {
    switch (d) {
        case Distribution::Ones:        return "ones";
        case Distribution::Uniform:     return "uniform";
        case Distribution::Normal:      return "normal";
        case Distribution::Adversarial: return "adversarial";
    }
    return "uniform";
}

void generate(std::vector<float>& out, size_t n, Distribution d, uint64_t seed) {
    out.resize(n);
    std::mt19937_64 rng(seed);

    switch (d) {
        case Distribution::Ones:
            std::fill(out.begin(), out.end(), 1.0f);
            return;

        case Distribution::Uniform: {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : out) v = dist(rng);
            return;
        }

        case Distribution::Normal: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& v : out) v = dist(rng);
            return;
        }

        case Distribution::Adversarial: {
            // Deliberately nasty but still summable in double: most values are
            // small, a few are large enough to swamp them in fp32, and a
            // scattering of denormals checks that flush-to-zero behaviour is
            // at least visible in the error rather than silent.
            //
            // No NaN or infinity: those make every downstream number NaN and
            // tell you nothing beyond "it propagated", which is not worth
            // destroying the whole measurement for.
            std::uniform_real_distribution<float> small(-1.0f, 1.0f);
            std::uniform_int_distribution<int> pick(0, 999);
            for (auto& v : out) {
                const int p = pick(rng);
                if (p < 5)        v = small(rng) * 1e7f;                    // swamps the sum
                else if (p < 10)  v = std::numeric_limits<float>::denorm_min() * (float)(p + 1);
                else if (p < 15)  v = 0.0f;
                else              v = small(rng);
            }
            return;
        }
    }
}

}

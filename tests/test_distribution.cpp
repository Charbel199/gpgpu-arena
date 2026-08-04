#include <doctest/doctest.h>
#include "arena/distribution.hpp"

#include <algorithm>
#include <cmath>

using namespace arena;

TEST_CASE("distribution_from_string") {
    SUBCASE("known names") {
        bool ok = false;
        CHECK(distribution_from_string("ones", &ok) == Distribution::Ones);
        CHECK(ok);
        CHECK(distribution_from_string("adversarial", &ok) == Distribution::Adversarial);
        CHECK(ok);
    }
    SUBCASE("unknown name reports failure and falls back to uniform") {
        bool ok = true;
        CHECK(distribution_from_string("gaussian", &ok) == Distribution::Uniform);
        CHECK_FALSE(ok);
    }
    SUBCASE("names round-trip") {
        for (auto d : {Distribution::Ones, Distribution::Uniform,
                       Distribution::Normal, Distribution::Adversarial}) {
            CHECK(distribution_from_string(distribution_name(d)) == d);
        }
    }
}

TEST_CASE("generate") {
    std::vector<float> v;

    SUBCASE("produces the requested count") {
        generate(v, 1000, Distribution::Uniform, 42);
        CHECK(v.size() == 1000);
    }

    SUBCASE("same seed gives identical data") {
        // The CPU reference is regenerated from the seed, so this has to hold
        // or verification compares against the wrong numbers.
        std::vector<float> a, b;
        generate(a, 500, Distribution::Normal, 7);
        generate(b, 500, Distribution::Normal, 7);
        CHECK(a == b);
    }

    SUBCASE("different seeds give different data") {
        std::vector<float> a, b;
        generate(a, 500, Distribution::Uniform, 1);
        generate(b, 500, Distribution::Uniform, 2);
        CHECK(a != b);
    }

    SUBCASE("ones is exactly ones") {
        generate(v, 100, Distribution::Ones, 42);
        CHECK(std::all_of(v.begin(), v.end(), [](float x) { return x == 1.0f; }));
    }

    SUBCASE("uniform stays in range and is not constant") {
        generate(v, 5000, Distribution::Uniform, 42);
        CHECK(std::all_of(v.begin(), v.end(), [](float x) { return x >= -1.0f && x <= 1.0f; }));
        CHECK(*std::min_element(v.begin(), v.end()) < -0.9f);
        CHECK(*std::max_element(v.begin(), v.end()) > 0.9f);
    }

    SUBCASE("adversarial contains large values that swamp an fp32 sum") {
        generate(v, 20000, Distribution::Adversarial, 42);
        CHECK(std::any_of(v.begin(), v.end(), [](float x) { return std::abs(x) > 1e6f; }));
    }

    SUBCASE("adversarial contains exact zeros and denormals") {
        generate(v, 20000, Distribution::Adversarial, 42);
        CHECK(std::any_of(v.begin(), v.end(), [](float x) { return x == 0.0f; }));
        CHECK(std::any_of(v.begin(), v.end(),
            [](float x) { return x != 0.0f && std::abs(x) < 1e-37f; }));
    }

    SUBCASE("no distribution emits NaN or infinity") {
        // A NaN input makes every downstream error NaN, which destroys the
        // measurement rather than informing it.
        for (auto d : {Distribution::Ones, Distribution::Uniform,
                       Distribution::Normal, Distribution::Adversarial}) {
            generate(v, 20000, d, 3);
            CHECK(std::none_of(v.begin(), v.end(),
                [](float x) { return std::isnan(x) || std::isinf(x); }));
        }
    }
}

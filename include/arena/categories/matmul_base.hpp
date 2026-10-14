#pragma once

#include "arena/kernel_descriptor.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <spdlog/spdlog.h>

namespace arena {

class MatmulDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "matmul"; }
    std::string function_name() const override { return "matmul"; }

    std::vector<std::string> get_parameter_names() const override {
        return {"M", "K", "N"};
    }

    int sweep_default_min() const override { return 64; }
    int sweep_default_max() const override { return 8192; }

    int accumulation_length() const override { return K_; }

    void set_problem_size(const std::map<std::string, int>& params) override {
        M_ = params.count("M") ? params.at("M") : 1024;
        K_ = params.count("K") ? params.at("K") : M_;
        N_ = params.count("N") ? params.at("N") : M_;
    }

    void allocate(Context& ctx) override {
        capture_device_props(ctx);
        size_a_ = M_ * K_ * sizeof(float);
        size_b_ = K_ * N_ * sizeof(float);
        size_c_ = M_ * N_ * sizeof(float);

        d_a_ = ctx.allocate(size_a_);
        d_b_ = ctx.allocate(size_b_);
        d_c_ = ctx.allocate(size_c_);
    }

    void initialize(Context& ctx) override {
        generate(h_a_, (size_t)M_ * K_, distribution_, input_seed_);
        generate(h_b_, (size_t)K_ * N_, distribution_, input_seed_ + 1);

        ctx.copy_to_device(d_a_, h_a_.data(), size_a_);
        ctx.copy_to_device(d_b_, h_b_.data(), size_b_);
    }

    // A and B are read-only and C is written outright rather than accumulated
    // into, which is why initialize() never zeroed it. Nothing to restore.
    void reset(Context&) override {}

    void cleanup(Context& ctx) override {
        ctx.free(d_a_);
        ctx.free(d_b_);
        ctx.free(d_c_);
        d_a_ = d_b_ = d_c_ = 0;
        h_a_.clear();
        h_b_.clear();
    }

    std::vector<void*> get_kernel_args() override {
        return { &d_a_, &d_b_, &d_c_, &M_, &K_, &N_ };
    }

    double calculate_flops() const override {
        return 2.0 * M_ * N_ * K_;
    }

    double calculate_bytes_accessed() const override {
        return static_cast<double>(size_a_ + size_b_ + size_c_);
    }

    VerifyResult verify(Context& ctx) override {
        std::vector<float> h_c((size_t)M_ * N_);
        ctx.copy_to_host(h_c.data(), d_c_, size_c_);

        // Checking every element costs O(M*N*K) on the host, so the output is
        // covered instead: C is cut into tiles and one element is checked in
        // each, plus the whole last row and last column, where partial-tile
        // bugs live. Positions follow the input seed, so they cannot be
        // learned. The old 64 fixed positions let a wrong tile go unseen.
        //
        // Tiles are 16x16 or larger, sized to keep this near 65k dot
        // products: about a quarter of a second at 4096^3.
        //
        // Each element is judged against the sum of |a*b| over its dot
        // product, as scan is: with thousands of checks some dot products
        // of mixed-sign data land near zero, where the plain ratio would fail
        // a correct kernel.
        // B is transposed once so each dot product walks memory in order.
        std::vector<float> bt((size_t)K_ * N_);
        for (int k = 0; k < K_; k++)
            for (int n = 0; n < N_; n++) bt[(size_t)n * K_ + k] = h_b_[(size_t)k * N_ + n];

        ErrorAccumulator acc;
        auto check = [&](int row, int col) {
            const float* a = h_a_.data() + (size_t)row * K_;
            const float* b = bt.data() + (size_t)col * K_;
            double ref = 0.0, magnitude = 0.0;
            for (int k = 0; k < K_; k++) {
                const double p = (double)a[k] * (double)b[k];
                ref += p;
                magnitude += std::abs(p);
            }
            acc.add_scaled(h_c[(size_t)row * N_ + col], ref, magnitude);
        };

        int tile = 16;
        while ((size_t)((M_ + tile - 1) / tile) * ((N_ + tile - 1) / tile) > 65536) tile *= 2;

        std::mt19937 rng(static_cast<uint32_t>(input_seed_ * 2654435761u + 1337));
        for (int tr = 0; tr < M_; tr += tile) {
            for (int tc = 0; tc < N_; tc += tile) {
                check(tr + (int)(rng() % (uint32_t)std::min(tile, M_ - tr)),
                      tc + (int)(rng() % (uint32_t)std::min(tile, N_ - tc)));
            }
        }
        for (int col = 0; col < N_; col++) check(M_ - 1, col);
        for (int row = 0; row < M_ - 1; row++) check(row, N_ - 1);

        auto r = acc.finish(tolerance());
        spdlog::get("verify")->debug("matmul: max rel err {:.3e} over {} checked elements",
            r.max_rel_error, r.elements_checked);
        return r;
    }

protected:
    int M_ = 1024, K_ = 1024, N_ = 1024;
    CUdeviceptr d_a_ = 0, d_b_ = 0, d_c_ = 0;
    size_t size_a_ = 0, size_b_ = 0, size_c_ = 0;
    std::vector<float> h_a_, h_b_;
};

}

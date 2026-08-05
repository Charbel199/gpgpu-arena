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
        std::vector<float> h_c(M_ * N_);
        ctx.copy_to_host(h_c.data(), d_c_, size_c_);

        // Spot-check random output elements against a double-accumulated CPU
        // dot product. Checking all of them would cost O(M*N*K) on the host.
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> row_dist(0, M_ - 1);
        std::uniform_int_distribution<int> col_dist(0, N_ - 1);

        ErrorAccumulator acc;
        const int checks = 64;
        for (int t = 0; t < checks; t++) {
            const int row = row_dist(rng);
            const int col = col_dist(rng);

            double ref = 0.0;
            for (int k = 0; k < K_; k++)
                ref += (double)h_a_[row * K_ + k] * (double)h_b_[k * N_ + col];

            acc.add(h_c[row * N_ + col], ref);
        }

        auto r = acc.finish(tolerance());
        spdlog::get("verify")->debug("matmul: max rel err {:.3e} over {} checks",
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

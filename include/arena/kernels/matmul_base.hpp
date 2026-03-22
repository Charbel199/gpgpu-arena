#pragma once

#include "arena/kernel_descriptor.hpp"
#include <vector>
#include <cmath>
#include <random>

namespace arena {

class MatmulDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "matmul"; }
    std::string function_name() const override { return "matmul"; }

    std::vector<std::string> get_parameter_names() const override {
        return {"M", "K", "N"};
    }

    void set_problem_size(const std::map<std::string, int>& params) override {
        M_ = params.count("M") ? params.at("M") : 1024;
        K_ = params.count("K") ? params.at("K") : M_;
        N_ = params.count("N") ? params.at("N") : M_;
    }

    void allocate(Context& ctx) override {
        size_a_ = M_ * K_ * sizeof(float);
        size_b_ = K_ * N_ * sizeof(float);
        size_c_ = M_ * N_ * sizeof(float);

        d_a_ = ctx.allocate(size_a_);
        d_b_ = ctx.allocate(size_b_);
        d_c_ = ctx.allocate(size_c_);
    }

    void initialize(Context& ctx) override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        h_a_.resize(M_ * K_);
        h_b_.resize(K_ * N_);
        for (float& v : h_a_) v = dist(rng);
        for (float& v : h_b_) v = dist(rng);

        ctx.copy_to_device(d_a_, h_a_.data(), size_a_);
        ctx.copy_to_device(d_b_, h_b_.data(), size_b_);
    }

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

    bool verify(Context& ctx) override {
        std::vector<float> h_c(M_ * N_);
        ctx.copy_to_host(h_c.data(), d_c_, size_c_);

        // spot-check 64 random output elements via CPU dot product
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> row_dist(0, M_ - 1);
        std::uniform_int_distribution<int> col_dist(0, N_ - 1);

        const int checks = 64;
        for (int t = 0; t < checks; t++) {
            int row = row_dist(rng);
            int col = col_dist(rng);

            double ref = 0.0;
            for (int k = 0; k < K_; k++)
                ref += (double)h_a_[row * K_ + k] * (double)h_b_[k * N_ + col];

            float got = h_c[row * N_ + col];
            float rel_err = std::abs((float)ref - got) / (std::abs((float)ref) + 1e-6f);
            if (rel_err > 1e-3f) return false;
        }
        return true;
    }

protected:
    int M_ = 1024, K_ = 1024, N_ = 1024;
    CUdeviceptr d_a_ = 0, d_b_ = 0, d_c_ = 0;
    size_t size_a_ = 0, size_b_ = 0, size_c_ = 0;
    std::vector<float> h_a_, h_b_;
};

}

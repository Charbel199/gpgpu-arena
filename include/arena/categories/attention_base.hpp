#pragma once

#include "arena/kernel_descriptor.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace arena {

// Scaled dot-product attention, non-causal, fp32:
//   O = softmax(Q K^T / sqrt(D)) V
// Q, K, V and O are [BH, S, D], contiguous, where BH is batch times heads.
// Every query attends to all S keys of its own head.
class AttentionDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "attention"; }
    std::string function_name() const override { return "attention"; }

    std::vector<std::string> get_parameter_names() const override { return {"BH", "S", "D"}; }

    int sweep_default_min() const override { return 128; }
    int sweep_default_max() const override { return 4096; }

    int accumulation_length() const override { return S_; }

    void set_problem_size(const std::map<std::string, int>& params) override {
        BH_ = params.count("BH") ? params.at("BH") : 16;
        S_  = params.count("S")  ? params.at("S")  : 1024;
        D_  = params.count("D")  ? params.at("D")  : 64;
    }

    // The sweep moves the sequence length; heads and head size stay put.
    std::vector<std::map<std::string, int>> get_sweep_configs(const RunConfig& cfg) const override {
        auto out = KernelDescriptor::get_sweep_configs(cfg);
        for (auto& p : out) { p["BH"] = BH_; p["D"] = D_; }
        return out;
    }

    void allocate(Context& ctx) override {
        capture_device_props(ctx);
        size_ = (size_t)BH_ * S_ * D_ * sizeof(float);
        d_q_ = ctx.allocate(size_);
        d_k_ = ctx.allocate(size_);
        d_v_ = ctx.allocate(size_);
        d_o_ = ctx.allocate(size_);
    }

    void initialize(Context& ctx) override {
        const size_t n = (size_t)BH_ * S_ * D_;
        auto make = [&](std::vector<float>& h, uint64_t salt) {
            generate(h, n, distribution_, input_seed_ + salt);
            // Rescaled rather than clamped, as in softmax: the distribution's
            // shape (mixed magnitudes, zeros, denormals) is what matters, and
            // raw adversarial values would overflow the Q.K dot products.
            float m = 0.0f;
            for (float& v : h) { if (!std::isfinite(v)) v = 0.0f; m = std::max(m, std::abs(v)); }
            if (m > 0.0f) for (float& v : h) v = v / m * 2.0f;
        };
        make(h_q_, 0);
        make(h_k_, 1);
        make(h_v_, 2);
        ctx.copy_to_device(d_q_, h_q_.data(), size_);
        ctx.copy_to_device(d_k_, h_k_.data(), size_);
        ctx.copy_to_device(d_v_, h_v_.data(), size_);
        ctx.zero_device(d_o_, size_);
    }

    void reset(Context& ctx) override { ctx.zero_device(d_o_, size_); }

    void cleanup(Context& ctx) override {
        ctx.free(d_q_); ctx.free(d_k_); ctx.free(d_v_); ctx.free(d_o_);
        d_q_ = d_k_ = d_v_ = d_o_ = 0;
        h_q_.clear(); h_k_.clear(); h_v_.clear();
    }

    std::vector<void*> get_kernel_args() override {
        return { &d_q_, &d_k_, &d_v_, &d_o_, &BH_, &S_, &D_ };
    }

    std::map<std::string, CUdeviceptr> named_buffers() const override {
        return {{"q", d_q_}, {"k", d_k_}, {"v", d_v_}, {"o", d_o_}};
    }
    std::map<std::string, int> problem_values() const override {
        return {{"BH", BH_}, {"S", S_}, {"D", D_}};
    }

    // Q K^T and P V: two multiply-adds per (query, key, d).
    double calculate_flops() const override { return 4.0 * BH_ * (double)S_ * S_ * D_; }
    // Q, K and V read once, O written once: what a fused kernel must move.
    double calculate_bytes_accessed() const override { return 4.0 * (double)size_; }

    VerifyResult verify(Context& ctx) override {
        std::vector<float> h_o((size_t)BH_ * S_ * D_);
        ctx.copy_to_host(h_o.data(), d_o_, size_);

        // Every head, every 64-row tile of queries: one row chosen by the seed,
        // plus each head's last row. A full check is O(BH * S^2 * D) on the
        // host. Errors are judged against sum_j p_j |v_jd|, the scale this
        // weighted sum's error grows with, since an output can be near zero.
        std::mt19937 rng(static_cast<uint32_t>(input_seed_ * 2654435761u + 7));
        ErrorAccumulator acc;
        std::vector<double> p(S_);
        const double scale = 1.0 / std::sqrt((double)D_);
        auto check_row = [&](int bh, int row) {
            const float* q = h_q_.data() + ((size_t)bh * S_ + row) * D_;
            const float* k = h_k_.data() + (size_t)bh * S_ * D_;
            const float* v = h_v_.data() + (size_t)bh * S_ * D_;
            double m = -1e300;
            for (int j = 0; j < S_; j++) {
                double s = 0.0;
                for (int d = 0; d < D_; d++) s += (double)q[d] * k[(size_t)j * D_ + d];
                p[j] = s * scale;
                m = std::max(m, p[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < S_; j++) { p[j] = std::exp(p[j] - m); sum += p[j]; }
            for (int d = 0; d < D_; d++) {
                double o = 0.0, mag = 0.0;
                for (int j = 0; j < S_; j++) {
                    o += p[j] * v[(size_t)j * D_ + d];
                    mag += p[j] * std::abs((double)v[(size_t)j * D_ + d]);
                }
                acc.add_scaled(h_o[((size_t)bh * S_ + row) * D_ + d], o / sum, mag / sum);
            }
        };
        for (int bh = 0; bh < BH_; bh++) {
            for (int t = 0; t < S_; t += 64) check_row(bh, t + (int)(rng() % (uint32_t)std::min(64, S_ - t)));
            check_row(bh, S_ - 1);
        }

        auto r = acc.finish(tolerance());
        spdlog::get("verify")->debug("attention: max rel err {:.3e} over {} elements",
            r.max_rel_error, r.elements_checked);
        return r;
    }

protected:
    int BH_ = 16, S_ = 1024, D_ = 64;
    size_t size_ = 0;
    CUdeviceptr d_q_ = 0, d_k_ = 0, d_v_ = 0, d_o_ = 0;
    std::vector<float> h_q_, h_k_, h_v_;
};

}

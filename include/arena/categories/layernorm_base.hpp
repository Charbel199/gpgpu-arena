#pragma once

#include "arena/kernel_descriptor.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace arena {

// Row-wise layer normalization, forward, fp32:
//   y[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
// x and y are [rows, cols], contiguous. var is the biased (population)
// variance, as in torch.nn.LayerNorm. eps is fixed at 1e-5 and is not a
// kernel argument: manifests have no float scalars, so every kernel bakes it
// in, the same way the problem definition does.
class LayerNormDescriptorBase : public KernelDescriptor {
public:
    static constexpr double kEps = 1e-5;

    std::string category() const override { return "layernorm"; }
    std::string function_name() const override { return "layernorm"; }

    std::vector<std::string> get_parameter_names() const override {
        return {"rows", "cols"};
    }

    int sweep_default_min() const override { return 64; }
    int sweep_default_max() const override { return 8192; }

    int accumulation_length() const override { return cols_; }

    void set_problem_size(const std::map<std::string, int>& params) override {
        rows_ = params.count("rows") ? params.at("rows") : 1024;
        cols_ = params.count("cols") ? params.at("cols") : 1024;
    }

    void allocate(Context& ctx) override {
        capture_device_props(ctx);
        size_data_ = (size_t)rows_ * cols_ * sizeof(float);
        size_param_ = (size_t)cols_ * sizeof(float);
        d_input_ = ctx.allocate(size_data_);
        d_gamma_ = ctx.allocate(size_param_);
        d_beta_ = ctx.allocate(size_param_);
        d_output_ = ctx.allocate(size_data_);
    }

    void initialize(Context& ctx) override {
        const size_t n = (size_t)rows_ * cols_;
        generate(h_input_, n, distribution_, input_seed_);
        // Each row is shifted by its own offset. The distributions are all
        // centred on zero, so without this a kernel that forgot to subtract
        // the mean would only be off by the sampling noise of the mean. The
        // offset is a few standard deviations at most, as in real
        // activations, so E[x^2] - E[x]^2 stays accurate enough to pass.
        //
        // Two kinds of rows test eps. A quiet row (every 50th from row 7) is
        // uniform in +-1e-3 around zero, a variance of 3e-7: eps dominates
        // it, so a kernel using another eps is off by a large fraction of the
        // output there. A constant row (every 50th from row 32) has zero
        // variance, and a kernel without eps divides zero by zero. Padding
        // rows look like these in practice.
        std::mt19937_64 rng(input_seed_ * 0x9E3779B97F4A7C15ull + 11);
        std::uniform_real_distribution<float> shift(-3.0f, 3.0f);
        std::uniform_real_distribution<float> quiet(-1e-3f, 1e-3f);
        for (int r = 0; r < rows_; r++) {
            const float off = shift(rng);
            float* row = h_input_.data() + (size_t)r * cols_;
            for (int c = 0; c < cols_; c++) {
                if (r % 50 == 7)       row[c] = quiet(rng);
                else if (r % 50 == 32) row[c] = off;
                else row[c] = std::isfinite(row[c]) ? row[c] + off : off;
            }
        }

        // gamma and beta do not follow the distribution: they are learned
        // weights, close to 1 and 0, and an adversarial 1e7 scale would only
        // multiply every error by the same factor.
        std::vector<float> u;
        generate(u, (size_t)cols_, Distribution::Uniform, input_seed_ + 1);
        h_gamma_.resize(cols_);
        for (int c = 0; c < cols_; c++) h_gamma_[c] = 1.0f + 0.5f * u[c];
        generate(u, (size_t)cols_, Distribution::Uniform, input_seed_ + 2);
        h_beta_.resize(cols_);
        for (int c = 0; c < cols_; c++) h_beta_[c] = 0.5f * u[c];

        ctx.copy_to_device(d_input_, h_input_.data(), size_data_);
        ctx.copy_to_device(d_gamma_, h_gamma_.data(), size_param_);
        ctx.copy_to_device(d_beta_, h_beta_.data(), size_param_);
        ctx.zero_device(d_output_, size_data_);
    }

    void reset(Context& ctx) override {
        ctx.zero_device(d_output_, size_data_);
    }

    void cleanup(Context& ctx) override {
        ctx.free(d_input_);
        ctx.free(d_gamma_);
        ctx.free(d_beta_);
        ctx.free(d_output_);
        d_input_ = d_gamma_ = d_beta_ = d_output_ = 0;
        h_input_.clear(); h_gamma_.clear(); h_beta_.clear();
    }

    std::vector<void*> get_kernel_args() override {
        return { &d_input_, &d_gamma_, &d_beta_, &d_output_, &rows_, &cols_ };
    }

    std::map<std::string, CUdeviceptr> named_buffers() const override {
        return {{"input", d_input_}, {"gamma", d_gamma_}, {"beta", d_beta_}, {"output", d_output_}};
    }
    std::map<std::string, int> problem_values() const override {
        return {{"rows", rows_}, {"cols", cols_}};
    }

    double calculate_flops() const override {
        // per element: 1 add for the mean, sub + fma for the variance,
        // sub, mul by rstd and fma with gamma and beta for the output ~ 8
        return 8.0 * rows_ * cols_;
    }

    double calculate_bytes_accessed() const override {
        // x read once and y written once; gamma and beta are read once in
        // total, since every row shares them and they stay in cache
        return (2.0 * rows_ * cols_ + 2.0 * cols_) * sizeof(float);
    }

    VerifyResult verify(Context& ctx) override {
        std::vector<float> h_output((size_t)rows_ * cols_);
        ctx.copy_to_host(h_output.data(), d_output_, size_data_);

        // Every element of every row. The error is judged against
        // |gamma| (|x| + mean|x|) rstd + |beta|, what an fp32 computation of
        // (x - mean) * rstd * gamma + beta can be expected to get right. The
        // subtraction loses precision in proportion to |x| and the error of
        // the fp32 mean, not to the result, which is near zero for x close
        // to the mean. That error grows with mean|x|, not |mean|: a row of
        // large values of both signs sums to a small mean with a large error.
        ErrorAccumulator acc;
        for (int r = 0; r < rows_; r++) {
            const float* in_row  = h_input_.data() + (size_t)r * cols_;
            const float* out_row = h_output.data() + (size_t)r * cols_;

            double mean = 0.0, mean_abs = 0.0;
            for (int c = 0; c < cols_; c++) { mean += in_row[c]; mean_abs += std::abs((double)in_row[c]); }
            mean /= cols_;
            mean_abs /= cols_;
            double var = 0.0;
            for (int c = 0; c < cols_; c++) {
                const double d = in_row[c] - mean;
                var += d * d;
            }
            var /= cols_;
            const double rstd = 1.0 / std::sqrt(var + kEps);

            for (int c = 0; c < cols_; c++) {
                const double g = h_gamma_[c], b = h_beta_[c];
                const double expected = (in_row[c] - mean) * rstd * g + b;
                const double scale = std::abs(g) * (std::abs((double)in_row[c]) + mean_abs) * rstd
                                   + std::abs(b);
                acc.add_scaled(out_row[c], expected, scale);
            }
        }

        auto res = acc.finish(tolerance());
        spdlog::get("verify")->debug("layernorm: max rel err {:.3e} over {} elements",
            res.max_rel_error, res.elements_checked);
        return res;
    }

protected:
    int rows_ = 1024;
    int cols_ = 1024;
    CUdeviceptr d_input_ = 0, d_gamma_ = 0, d_beta_ = 0, d_output_ = 0;
    size_t size_data_ = 0, size_param_ = 0;
    std::vector<float> h_input_, h_gamma_, h_beta_;
};

}

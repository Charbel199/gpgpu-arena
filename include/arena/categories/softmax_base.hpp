#pragma once

#include "arena/kernel_descriptor.hpp"
#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace arena {

class SoftmaxDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "softmax"; }
    std::string function_name() const override { return "softmax"; }

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
        d_input_ = ctx.allocate(size_data_);
        d_output_ = ctx.allocate(size_data_);
    }

    void initialize(Context& ctx) override {
        // Softmax subtracts the row max before exp(), so scale rather than
        // clamp: keeping the shape of the distribution matters more than the
        // absolute range, and huge magnitudes would just saturate every row
        // to a one-hot vector and hide any real difference between kernels.
        std::vector<float> h_input;
        generate(h_input, (size_t)rows_ * cols_, distribution_, input_seed_);
        for (float& v : h_input) {
            v = std::isfinite(v) ? std::max(-20.0f, std::min(20.0f, v * 3.0f)) : 0.0f;
        }
        ctx.copy_to_device(d_input_, h_input.data(), size_data_);

        std::vector<float> h_zeros(rows_ * cols_, 0.0f);
        ctx.copy_to_device(d_output_, h_zeros.data(), size_data_);
    }

    void reset(Context& ctx) override {
        ctx.zero_device(d_output_, size_data_);
    }

    void cleanup(Context& ctx) override {
        ctx.free(d_input_);
        ctx.free(d_output_);
        d_input_ = d_output_ = 0;
    }

    std::vector<void*> get_kernel_args() override {
        return { &d_input_, &d_output_, &rows_, &cols_ };
    }

    double calculate_flops() const override {
        // per row: n comparisons (max) + n exp + n additions (sum) + n divides ~ 4n
        return 4.0 * rows_ * cols_;
    }

    double calculate_bytes_accessed() const override {
        // read input once, write output once
        return 2.0 * rows_ * cols_ * sizeof(float);
    }

    VerifyResult verify(Context& ctx) override {
        std::vector<float> h_input(rows_ * cols_);
        std::vector<float> h_output(rows_ * cols_);
        ctx.copy_to_host(h_input.data(), d_input_, size_data_);
        ctx.copy_to_host(h_output.data(), d_output_, size_data_);

        ErrorAccumulator acc;
        const int rows_checked = std::min(rows_, 8);
        for (int r = 0; r < rows_checked; r++) {
            const float* in_row  = h_input.data()  + r * cols_;
            const float* out_row = h_output.data() + r * cols_;

            const float row_max = *std::max_element(in_row, in_row + cols_);
            double row_sum = 0.0;
            for (int c = 0; c < cols_; c++) row_sum += std::exp((double)in_row[c] - row_max);

            for (int c = 0; c < cols_; c++) {
                acc.add(out_row[c], std::exp((double)in_row[c] - row_max) / row_sum);
            }
        }

        auto res = acc.finish(tolerance());
        spdlog::get("verify")->debug("softmax: max rel err {:.3e} over {} elements",
            res.max_rel_error, res.elements_checked);
        return res;
    }

protected:
    int rows_ = 1024;
    int cols_ = 1024;
    CUdeviceptr d_input_ = 0, d_output_ = 0;
    size_t size_data_ = 0;
};

}

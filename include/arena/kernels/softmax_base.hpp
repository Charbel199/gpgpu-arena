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

    void set_problem_size(const std::map<std::string, int>& params) override {
        rows_ = params.count("rows") ? params.at("rows") : 1024;
        cols_ = params.count("cols") ? params.at("cols") : 1024;
    }

    void allocate(Context& ctx) override {
        size_data_ = (size_t)rows_ * cols_ * sizeof(float);
        d_input_ = ctx.allocate(size_data_);
        d_output_ = ctx.allocate(size_data_);
    }

    void initialize(Context& ctx) override {
        // fill with small values so exp() doesn't overflow
        std::vector<float> h_input(rows_ * cols_);
        for (int i = 0; i < rows_ * cols_; i++) {
            h_input[i] = (float)(i % 7) - 3.0f; // values in [-3, 3]
        }
        ctx.copy_to_device(d_input_, h_input.data(), size_data_);

        std::vector<float> h_zeros(rows_ * cols_, 0.0f);
        ctx.copy_to_device(d_output_, h_zeros.data(), size_data_);
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

    bool verify(Context& ctx) override {
        std::vector<float> h_input(rows_ * cols_);
        std::vector<float> h_output(rows_ * cols_);
        ctx.copy_to_host(h_input.data(), d_input_, size_data_);
        ctx.copy_to_host(h_output.data(), d_output_, size_data_);

        for (int r = 0; r < std::min(rows_, 8); r++) {
            const float* in_row = h_input.data() + r * cols_;
            const float* out_row = h_output.data() + r * cols_;

            // cpu reference: softmax(row)
            float row_max = *std::max_element(in_row, in_row + cols_);
            float row_sum = 0.0f;
            for (int c = 0; c < cols_; c++) {
                row_sum += expf(in_row[c] - row_max);
            }

            for (int c = 0; c < cols_; c++) {
                float expected = expf(in_row[c] - row_max) / row_sum;
                float rel_err = std::abs(out_row[c] - expected) / (expected + 1e-8f);
                if (rel_err > 1e-3f) {
                    spdlog::get("verify")->warn(
                        "row {} col {}: got {}, expected {}", r, c, out_row[c], expected);
                    return false;
                }
            }

            // check row sums to ~1.0
            float sum = 0.0f;
            for (int c = 0; c < cols_; c++) sum += out_row[c];
            if (std::abs(sum - 1.0f) > 1e-3f) {
                spdlog::get("verify")->warn("row {} sum = {}, expected 1.0", r, sum);
                return false;
            }
        }

        spdlog::get("verify")->debug("verify passed");
        return true;
    }

protected:
    int rows_ = 1024;
    int cols_ = 1024;
    CUdeviceptr d_input_ = 0, d_output_ = 0;
    size_t size_data_ = 0;
};

}

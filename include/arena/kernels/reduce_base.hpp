#pragma once

#include "arena/kernel_descriptor.hpp"
#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>

namespace arena {


class ReduceDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "reduce"; }
    std::string function_name() const override { return "reduce_sum"; }
    
    std::vector<std::string> get_parameter_names() const override {
        return {"n"};
    }

    std::vector<std::map<std::string, int>> get_sweep_configs() const override {
        return {
            {{"n", 256}},
            {{"n", 1024}},
            {{"n", 4096}},
            {{"n", 16384}},
            {{"n", 65536}},
            {{"n", 262144}},
            {{"n", 1000000}},
            {{"n", 4000000}},
            {{"n", 16000000}},
            {{"n", 64000000}},
            {{"n", 256000000}},
        };
    }
    
    void set_problem_size(const std::map<std::string, int>& params) override {
        n_ = params.count("n") ? params.at("n") : 1000000;
    }
    
    void allocate(Context& ctx) override {
        size_input_ = n_ * sizeof(float);
        size_output_ = sizeof(float);
        
        d_input_ = ctx.allocate(size_input_);
        d_output_ = ctx.allocate(size_output_);
    }
    
    void initialize(Context& ctx) override {
        std::vector<float> h_input(n_, 1.0f);
        ctx.copy_to_device(d_input_, h_input.data(), size_input_);
        
        float zero = 0.0f;
        ctx.copy_to_device(d_output_, &zero, sizeof(float));
    }
    
    void cleanup(Context& ctx) override {
        ctx.free(d_input_);
        ctx.free(d_output_);
        d_input_ = d_output_ = 0;
    }
    
    std::vector<void*> get_kernel_args() override {
        return { &d_input_, &d_output_, &n_ };
    }
    
    double calculate_flops() const override {
        return static_cast<double>(n_ - 1);
    }
    
    double calculate_bytes_accessed() const override {
        return static_cast<double>(size_input_ + size_output_);
    }
    
    bool verify(Context& ctx) override {
        float result;
        ctx.copy_to_host(&result, d_output_, sizeof(float));

        float expected = static_cast<float>(n_); //TODO: I don't like the idea of not doing a CPU check, what if we initialize RANDOM weights (This is how it should be anw)
        spdlog::get("verify")->debug("reduce: got {}, expected {}", result, expected);
        float rel_error = std::abs(result - expected) / expected;
        return rel_error < 1e-5f;
    }

protected:
    int n_ = 1000000;
    CUdeviceptr d_input_ = 0, d_output_ = 0;
    size_t size_input_ = 0, size_output_ = 0;
};

}

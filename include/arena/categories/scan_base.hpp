#pragma once

#include "arena/kernel_descriptor.hpp"
#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace arena {

class ScanDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "scan"; }
    std::string function_name() const override { return "exclusive_scan"; }
    
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
        capture_device_props(ctx);
        size_data_ = n_ * sizeof(float);
        
        d_input_ = ctx.allocate(size_data_);
        d_output_ = ctx.allocate(size_data_);
    }
    
    void initialize(Context& ctx) override {
        h_input_.assign(n_, 1.0f);
        ctx.copy_to_device(d_input_, h_input_.data(), size_data_);

        std::vector<float> h_output(n_, 0.0f);
        ctx.copy_to_device(d_output_, h_output.data(), size_data_);
    }
    
    void cleanup(Context& ctx) override {
        ctx.free(d_input_);
        ctx.free(d_output_);
        d_input_ = d_output_ = 0;
        h_input_.clear();
    }
    
    std::vector<void*> get_kernel_args() override {
        return { &d_input_, &d_output_, &n_ };
    }
    
    double calculate_flops() const override {
        return static_cast<double>(n_ - 1); // we only need n-1 additions
    }
    
    double calculate_bytes_accessed() const override {
        return static_cast<double>(2 * size_data_); // we read all elements and write all of them back
    }
    
    VerifyResult verify(Context& ctx) override {
        std::vector<float> h_output(n_);
        ctx.copy_to_host(h_output.data(), d_output_, size_data_);

        // Exclusive scan: out[i] is the sum of in[0..i-1]. The running total
        // has to walk every element, but only a strided sample is scored so
        // large n stays cheap.
        ErrorAccumulator acc;
        const int stride = std::max(1, n_ / 1000);
        double running = 0.0;
        for (int i = 0; i < n_; i++) {
            if (i % stride == 0) acc.add(h_output[i], running);
            running += h_input_[i];
        }

        auto r = acc.finish(tolerance());
        spdlog::get("verify")->debug("scan: {} points checked, max rel err {:.3e}",
            r.elements_checked, r.max_rel_error);
        return r;
    }

protected:
    int n_ = 1000000;
    std::vector<float> h_input_;
    CUdeviceptr d_input_ = 0, d_output_ = 0;
    size_t size_data_ = 0;
};

}

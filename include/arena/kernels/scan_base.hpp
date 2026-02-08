#pragma once

#include "arena/kernel_descriptor.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace arena {

class ScanDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "scan"; }
    std::string function_name() const override { return "exclusive_scan"; }
    
    std::vector<std::string> get_parameter_names() const override {
        return {"n"};
    }
    
    void set_problem_size(const std::map<std::string, int>& params) override {
        n_ = params.count("n") ? params.at("n") : 1000000;
    }
    
    void allocate(Context& ctx) override {
        size_data_ = n_ * sizeof(float);
        
        d_input_ = ctx.allocate(size_data_);
        d_output_ = ctx.allocate(size_data_);
    }
    
    void initialize(Context& ctx) override {
        std::vector<float> h_input(n_, 1.0f);
        ctx.copy_to_device(d_input_, h_input.data(), size_data_);
        
        std::vector<float> h_output(n_, 0.0f);
        ctx.copy_to_device(d_output_, h_output.data(), size_data_);
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
        return static_cast<double>(n_ - 1); // we only need n-1 additions
    }
    
    double calculate_bytes_accessed() const override {
        return static_cast<double>(2 * size_data_); // we read all elements and write all of them back
    }
    
    bool verify(Context& ctx) override {
        std::vector<float> h_output(n_);
        ctx.copy_to_host(h_output.data(), d_output_, size_data_);
        
        // output[i] should equal i (0, 1, 2, 3, ...) since we have an array of 1's
        int check_count = std::min(1000, n_);
        for (int i = 0; i < check_count; i++) {
            float expected = static_cast<float>(i);
            if (std::abs(h_output[i] - expected) > 1e-2f) {
                std::cout << "Verification failed at index " << i 
                          << ": got " << h_output[i] 
                          << ", expected " << expected << std::endl;
                return false;
            }
        }
        
        // also check last element: should be n-1
        float last_expected = static_cast<float>(n_ - 1);
        if (std::abs(h_output[n_ - 1] - last_expected) > 1e-2f) {
            std::cout << "Verification failed at last index: got " 
                      << h_output[n_ - 1] << ", expected " << last_expected << std::endl;
            return false;
        }
        
        return true;
    }

protected:
    int n_ = 1000000;
    CUdeviceptr d_input_ = 0, d_output_ = 0;
    size_t size_data_ = 0;
};

}

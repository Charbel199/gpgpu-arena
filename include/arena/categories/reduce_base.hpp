#pragma once

#include "arena/kernel_descriptor.hpp"
#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>
#include <cstdint>

// Kernel folder layout: kernels/<category>/<input dtype>/. The folder is named
// for what a kernel reads, because that is what shapes the source: reading
// __half* rather than float* changes the signature and the load path, while
// the output type is a single store at the end. Variants sharing an input type
// but differing in output live in the same file.

namespace arena {


class ReduceDescriptorBase : public KernelDescriptor {
public:
    std::string category() const override { return "reduce"; }
    std::string function_name() const override { return "reduce_sum"; }
    
    std::vector<std::string> get_parameter_names() const override {
        return {"n"};
    }

    int sweep_default_min() const override { return 256; }
    int sweep_default_max() const override { return 256000000; }
    
    int accumulation_length() const override { return n_; }

    void set_problem_size(const std::map<std::string, int>& params) override {
        n_ = params.count("n") ? params.at("n") : 1000000;
    }
    
    void allocate(Context& ctx) override {
        capture_device_props(ctx);
        size_input_  = dtype_buffer_bytes((size_t)n_, input_dtype());
        size_output_ = dtype_buffer_bytes(1, output_dtype());
        
        d_input_ = ctx.allocate(size_input_);
        d_output_ = ctx.allocate(size_output_);
    }
    
    void initialize(Context& ctx) override {
        std::vector<float> h_input;
        generate(h_input, n_, distribution_, input_seed_);

        // Two references. The exact one comes from the generated fp32 data;
        // the quantized one from the same data after rounding to the kernel's
        // storage type. Comparing against both separates what the kernel got
        // wrong from what its dtype cost it before it ever ran.
        reference_exact_ = 0.0;
        for (float v : h_input) reference_exact_ += v;

        quantize_in_place(h_input, input_dtype());

        reference_ = 0.0;
        for (float v : h_input) reference_ += v;

        if (input_dtype() == DType::FP32) {
            ctx.copy_to_device(d_input_, h_input.data(), size_input_);
        } else {
            const auto packed = pack_values(h_input, input_dtype());
            ctx.copy_to_device(d_input_, packed.data(), size_input_);
        }

        // Zero through the output type, not always as a float: the buffer is
        // only as wide as the output type, so writing a float into a narrower
        // output would run past the end of the allocation.
        const uint64_t zero = 0;
        ctx.copy_to_device(d_output_, &zero, size_output_);
    }

    // The kernels here accumulate into the output, so it has to start at zero
    // every iteration. The input does not: it is read-only and identical each
    // time, so regenerating it was costing a full generate, two summations, a
    // quantize pass and a host-to-device copy per launch.
    void reset(Context& ctx) override {
        ctx.zero_device(d_output_, size_output_);
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
    
    VerifyResult verify(Context& ctx) override {
        // Read back through the output type, whatever its width.
        uint8_t raw[8] = {};
        ctx.copy_to_host(raw, d_output_, size_output_);
        const float result = unpack_value(raw, 0, output_dtype());

        ErrorAccumulator acc;
        acc.add(result, reference_, reference_exact_);
        auto r = acc.finish(tolerance());

        spdlog::get("verify")->debug(
            "reduce: got {}, reference {}, arithmetic err {:.3e}, total err {:.3e}",
            result, reference_, r.max_rel_error, r.max_total_error);
        return r;
    }

protected:
    int n_ = 1000000;
    double reference_ = 0.0;        // sum of the quantized inputs
    double reference_exact_ = 0.0;  // sum of the original fp32 inputs
    CUdeviceptr d_input_ = 0, d_output_ = 0;
    size_t size_input_ = 0, size_output_ = 0;
};

}

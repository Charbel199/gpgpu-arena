#pragma once

#include "arena/device/context.hpp"
#include "arena/device/kernel_loader.hpp"
#include "arena/compilers/backend.hpp"
#include "arena/dtype.hpp"
#include "arena/distribution.hpp"
#include "arena/runner_config.hpp"
#include "arena/measurement/accuracy.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace arena {

// Base class for all kernel descriptors, the goal is to have all kernels inherit from this class 
// and extend it based on their specific type (GEMM, GEMV, SpMV, Reduce, Scan, ...)
class KernelDescriptor {
public:
    virtual ~KernelDescriptor() = default;
    
    // basics
    virtual std::string name() const = 0;
    virtual std::string category() const = 0;
    virtual std::string description() const = 0;
    virtual std::string module_path() const = 0;
    virtual std::string function_name() const = 0;
    
    // problem configuration
    virtual std::vector<std::string> get_parameter_names() const = 0;
    virtual void set_problem_size(const std::map<std::string, int>& params) = 0;
    // Sizes to sweep over. A category supplies its own sensible range and the
    // generator below turns it into a ladder, so the range is one pair of
    // numbers rather than a hand-written list per category.
    virtual int sweep_default_min() const { return 0; }
    virtual int sweep_default_max() const { return 0; }

    virtual std::vector<std::map<std::string, int>> get_sweep_configs(
        const RunConfig& cfg) const {
        const int lo = cfg.sweep_min > 0 ? cfg.sweep_min : sweep_default_min();
        const int hi = cfg.sweep_max > 0 ? cfg.sweep_max : sweep_default_max();
        const double factor = cfg.sweep_factor > 1.0 ? cfg.sweep_factor : 4.0;
        if (lo <= 0 || hi < lo) return {};

        // Every parameter takes the same value at each step. Both multi-param
        // categories here are square (M=K=N, rows=cols), and a per-parameter
        // range would be a lot of UI for a case nothing needs yet.
        std::vector<std::map<std::string, int>> out;
        const auto names = get_parameter_names();
        for (double v = lo; v <= (double)hi * 1.0001; v *= factor) {
            std::map<std::string, int> params;
            for (const auto& n : names) params[n] = (int)(v + 0.5);
            out.push_back(std::move(params));
            if (factor <= 1.0) break;
        }
        return out;
    }
    
    // Input generation settings, pushed in by the runner before allocate().
    void set_input_spec(Distribution d, uint64_t seed) {
        distribution_ = d;
        input_seed_ = seed;
    }

    // memory management (through the context class)
    virtual void allocate(Context& ctx) = 0;
    virtual void initialize(Context& ctx) = 0;
    virtual void cleanup(Context& ctx) = 0;
    
    // Block sizes this kernel can be launched at. Empty means the block size
    // is not a free parameter, which is the case whenever the cubin pins it:
    // cuTile emits .reqntid, and Triton and Warp fix it at compile time from
    // num_warps. Those need a recompile to change, not a different launch.
    virtual std::vector<int> tunable_block_sizes() const { return {}; }

    // Chosen block size, or 0 for the descriptor's own default. Set by the
    // runner before get_launch_config().
    void set_block_size(int n) { block_size_ = n; }

    // Helper for descriptors: the block size to actually launch at.
    int block_size_or(int fallback) const {
        return block_size_ > 0 ? block_size_ : fallback;
    }

    // kernel launch configuration
    virtual KernelLoader::LaunchConfig get_launch_config() const = 0;
    virtual std::vector<void*> get_kernel_args() = 0;

    // calculate speed of light metrics
    virtual double calculate_flops() const = 0;
    virtual double calculate_bytes_accessed() const = 0;
    
    // Element types of the input and output buffers. These are what the
    // framework can actually observe: it allocates those buffers and reads
    // them back. Anything a kernel does internally is its own business, which
    // is why there is no accumulator type here. A kernel can easily have
    // several at different precisions.
    //
    // The two default independently. Declaring narrow input must not quietly
    // relax the accuracy check, which is derived from the output type.
    virtual DType input_dtype() const { return DType::FP32; }
    virtual DType output_dtype() const { return DType::FP32; }

    virtual ComputeMode compute_mode() const { return ComputeMode::Default; }

    // How many values get summed into one output. Reduction error grows with
    // this, so it scales the tolerance: a sum over 4M elements is allowed
    // more drift than a sum over 8.
    virtual int accumulation_length() const { return 1; }

    // How far off the answer may be before the kernel counts as broken.
    //
    // Based on the output type: a kernel cannot produce an answer more precise
    // than the type it writes into, so that is the standard it is held to. A
    // kernel whose internals are coarser than its output will miss this, and
    // that is the intended result rather than a special case to excuse.
    virtual double tolerance() const {
        const double per_element = default_tolerance(output_dtype(), compute_mode());
        const int n = accumulation_length() > 1 ? accumulation_length() : 1;
        const double scaled = per_element * std::sqrt(static_cast<double>(n));

        // Capped, because the sqrt(n) growth models a reduction carried out
        // entirely in the output type. Kernels that accumulate in something
        // wider and only narrow at the end, which is the sensible design, do
        // far better than that, and for a narrow output the unbounded figure
        // reaches 15.6 at n=4M.
        //
        // The cap itself has a floor of half an ulp of the output type: an
        // answer cannot be more accurate than the type it is written into, so
        // demanding 10% from an fp4 output, where a single value can be 25%
        // off, would fail a correct kernel.
        const double half_ulp =
            0.5 / static_cast<double>(1ULL << (dtype_mantissa_bits(output_dtype()) - 1));
        return std::min(scaled, std::max(0.10, half_ulp));
    }

    // Compare against a CPU reference and report how far off it was. Returning
    // a number rather than a boolean means a kernel that is merely imprecise
    // reads differently from one that is wrong.
    virtual VerifyResult verify(Context& ctx) {
        VerifyResult r;
        r.passed = true;   // nothing to check
        return r;
    }


    // run cubin or cpp code
    virtual bool uses_module() const { return true; }
    virtual void execute(Context& ctx) {
        throw std::runtime_error("execute() not implemented for this kernel");
    }

    // runtime compilation (override for DSL kernels like Triton)
    virtual bool needs_compilation() const { return false; }
    virtual std::string source_path() const { return ""; }
    void set_compile_result(const CompileResult& result) { compile_result_ = result; }

    // device properties (captured automatically on first allocate)
    int sm_count() const { return sm_count_; }

protected:
    int block_size_ = 0;   // 0 = descriptor default

    Distribution distribution_ = Distribution::Uniform;
    uint64_t     input_seed_ = 42;

    CompileResult compile_result_;
    int sm_count_ = 0;

    // call from allocate() overrides to capture device properties
    void capture_device_props(Context& ctx) { sm_count_ = ctx.sm_count(); }
};


// registry for kernel descriptors, used to register all kernels in the program
class KernelRegistry {
public:
    static KernelRegistry& instance() { // one way to have a singleton in c++ (only one instance of the class)
        static KernelRegistry registry;
        return registry;
    }
    
    void register_kernel(std::unique_ptr<KernelDescriptor> descriptor) {
        descriptors_.push_back(std::move(descriptor));
    }
    
    // pointers to all kernels in the registry
    std::vector<KernelDescriptor*> get_all() const {
        std::vector<KernelDescriptor*> result;
        for (const auto& d : descriptors_) result.push_back(d.get());
        return result;
    }
    
    std::vector<KernelDescriptor*> get_by_category(const std::string& category) const {
        std::vector<KernelDescriptor*> result;
        for (const auto& d : descriptors_) {
            if (d->category() == category) result.push_back(d.get());
        }
        return result;
    }
    
    KernelDescriptor* get_by_name(const std::string& name) const {
        for (const auto& d : descriptors_) {
            if (d->name() == name) return d.get();
        }
        return nullptr;
    }
    
    std::vector<std::string> get_categories() const {
        std::set<std::string> unique;
        for (const auto& d : descriptors_) {
            unique.insert(d->category());
        }
        return std::vector<std::string>{unique.begin(), unique.end()};
    }

private:
    KernelRegistry() = default;
    std::vector<std::unique_ptr<KernelDescriptor>> descriptors_;
};


// thanks to claude, a simple way to register classes that extend the KernelDescriptor base class
#define REGISTER_KERNEL(DescriptorClass) \
    static bool _registered_##DescriptorClass = []() { \
        arena::KernelRegistry::instance().register_kernel( \
            std::make_unique<DescriptorClass>()); \
        return true; \
    }()

}

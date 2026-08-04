#pragma once

#include "arena/device/context.hpp"
#include "arena/device/kernel_loader.hpp"
#include "arena/compilers/backend.hpp"
#include "arena/dtype.hpp"
#include "arena/distribution.hpp"
#include "arena/measurement/accuracy.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <cmath>

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
    virtual std::vector<std::map<std::string, int>> get_sweep_configs() const { return {}; }
    
    // Input generation settings, pushed in by the runner before allocate().
    void set_input_spec(Distribution d, uint64_t seed) {
        distribution_ = d;
        input_seed_ = seed;
    }

    // memory management (through the context class)
    virtual void allocate(Context& ctx) = 0;
    virtual void initialize(Context& ctx) = 0;
    virtual void cleanup(Context& ctx) = 0;
    
    // kernel launch configuration
    virtual KernelLoader::LaunchConfig get_launch_config() const = 0;
    virtual std::vector<void*> get_kernel_args() = 0;

    // calculate speed of light metrics
    virtual double calculate_flops() const = 0;
    virtual double calculate_bytes_accessed() const = 0;
    
    // Storage type and compute mode. Part of the kernel's identity: an fp16
    // variant is a separate descriptor, not the same one run differently.
    virtual DType dtype() const { return DType::FP32; }
    virtual ComputeMode compute_mode() const { return ComputeMode::Default; }

    // How many values get summed into one output. Reduction error grows with
    // this, so it scales the tolerance: a sum over 4M elements is allowed
    // more drift than a sum over 8.
    virtual int accumulation_length() const { return 1; }

    // Tolerance the measured error is judged against. Per-element budget from
    // the dtype, scaled by sqrt of the accumulation length, which is how
    // rounding error grows when the summands have mixed signs.
    virtual double tolerance() const {
        const double per_element = default_tolerance(dtype(), compute_mode());
        const int n = accumulation_length() > 1 ? accumulation_length() : 1;
        return per_element * std::sqrt(static_cast<double>(n));
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

#pragma once

#include "arena/context.hpp"
#include "arena/kernel_loader.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>

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
    virtual std::string ptx_path() const = 0;
    virtual std::string function_name() const = 0;
    
    // problem configuration
    virtual std::vector<std::string> get_parameter_names() const = 0;
    virtual void set_problem_size(const std::map<std::string, int>& params) = 0;
    
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
    
    // verify if the result of the kernel is correct (TODO: not sure if this will work in all categories)
    virtual bool verify(Context& ctx) { return true; }
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

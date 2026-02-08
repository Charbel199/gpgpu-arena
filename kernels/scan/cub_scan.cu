#include "arena/kernels/scan_base.hpp"
#include <cub/cub.cuh>

namespace arena {

class CubScanDescriptor : public ScanDescriptorBase {
    void* d_temp_storage_ = nullptr;
    size_t temp_storage_bytes_ = 0;
    
public:
    std::string name() const override { return "cub_scan"; }
    std::string description() const override { return "CUB DeviceScan::ExclusiveSum"; }
    std::string ptx_path() const override { return ""; }
    bool uses_ptx() const override { return false; }
    
    void allocate(Context& ctx) override {
        ScanDescriptorBase::allocate(ctx);

        cub::DeviceScan::ExclusiveSum(
            nullptr, temp_storage_bytes_,
            reinterpret_cast<float*>(d_input_),
            reinterpret_cast<float*>(d_output_),
            n_
        );
        cudaMalloc(&d_temp_storage_, temp_storage_bytes_);
    }
    
    void execute(Context& ctx) override {
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage_, temp_storage_bytes_,
            reinterpret_cast<float*>(d_input_),
            reinterpret_cast<float*>(d_output_),
            n_
        );
    }
    
    void cleanup(Context& ctx) override {
        if (d_temp_storage_) cudaFree(d_temp_storage_);
        d_temp_storage_ = nullptr;
        ScanDescriptorBase::cleanup(ctx);
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override { return {}; }
};

REGISTER_KERNEL(CubScanDescriptor);

}

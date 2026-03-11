#include "arena/kernels/reduce_base.hpp"

namespace arena {

class TritonReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "triton_reduce"; }
    std::string description() const override { return "Triton reduce with atomic add"; }
    std::string ptx_path() const override { return "kernels/reduce_triton_reduce.ptx"; }
    std::string function_name() const override { return "reduce_sum"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int BLOCK_SIZE = 128;
        return {
            .grid_x = static_cast<unsigned>((n_ + BLOCK_SIZE - 1) / BLOCK_SIZE),
            .grid_y = 1, .grid_z = 1,
            .block_x = BLOCK_SIZE, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }

    // triton 3.6.0 adds 2 extra metadata pointer params to the PTX signature TODO: clean up
    std::vector<void*> get_kernel_args() override {
        return { &d_input_, &d_output_, &n_, &null_ptr_, &null_ptr_ };
    }

private:
    CUdeviceptr null_ptr_ = 0;
};

REGISTER_KERNEL(TritonReduceDescriptor);

}

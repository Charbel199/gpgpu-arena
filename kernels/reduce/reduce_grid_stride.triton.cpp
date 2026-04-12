#include "arena/kernels/reduce_base.hpp"

namespace arena {

class TritonGridStrideReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "triton_reduce_grid_stride"; }
    std::string description() const override { return "Triton reduce with grid-stride loop"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/reduce_grid_stride.triton.py"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        return {
            .grid_x = static_cast<unsigned>(compile_result_.constants.at("NUM_BLOCKS")),
            .grid_y = 1, .grid_z = 1,
            .block_x = static_cast<unsigned>(compile_result_.num_warps * 32),
            .block_y = 1, .block_z = 1,
            .shared_mem_bytes = static_cast<unsigned>(compile_result_.shared_memory)
        };
    }

    std::vector<void*> get_kernel_args() override {
        std::vector<void*> args = { &d_input_, &d_output_, &n_ };
        for (int i = 3; i < compile_result_.num_params; i++)
            args.push_back(&null_ptr_);
        return args;
    }

private:
    CUdeviceptr null_ptr_ = 0;
};

REGISTER_KERNEL(TritonGridStrideReduceDescriptor);

}

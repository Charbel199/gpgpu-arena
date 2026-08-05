#include "arena/categories/reduce_base.hpp"

namespace arena {

class TritonGridStrideReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "triton_reduce_grid_stride"; }
    std::string description() const override { return "Triton reduce with grid-stride loop"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/fp32/reduce_grid_stride.triton.py"; }

    // A Triton kernel cannot be relaunched at a different block size, so the
    // tuning axis is a recompile. BLOCK_SIZE is the tile each program handles
    // and num_warps is what Triton turns into the thread block, so both have
    // to move to cover the space.
    std::vector<CompileDefines> tunable_compile_options() const override {
        std::vector<CompileDefines> out;
        for (int block : {256, 1024, 4096})
            for (int warps : {2, 4, 8})
                out.push_back({{"BLOCK_SIZE", block}, {"num_warps", warps}});
        return out;
    }

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

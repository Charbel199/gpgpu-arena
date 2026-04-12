#include "arena/kernels/matmul_base.hpp"

namespace arena {

class TritonMatmulDescriptor : public MatmulDescriptorBase {
public:
    std::string name() const override { return "triton_matmul"; }
    std::string description() const override { return "Triton tiled matmul (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "matmul/matmul.triton.py"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        int block_m = compile_result_.constants.at("BLOCK_M");
        int block_n = compile_result_.constants.at("BLOCK_N");
        // 1D grid kernel does swizzled pid_m/pid_n internally
        unsigned num_m = (M_ + block_m - 1) / block_m;
        unsigned num_n = (N_ + block_n - 1) / block_n;
        return {
            .grid_x = num_m * num_n,
            .grid_y = 1, .grid_z = 1,
            .block_x = static_cast<unsigned>(compile_result_.num_warps * 32),
            .block_y = 1, .block_z = 1,
            .shared_mem_bytes = static_cast<unsigned>(compile_result_.shared_memory)
        };
    }

    std::vector<void*> get_kernel_args() override {
        std::vector<void*> args = { &d_a_, &d_b_, &d_c_, &M_, &K_, &N_ };
        for (int i = 6; i < compile_result_.num_params; i++)
            args.push_back(&null_ptr_);
        return args;
    }

private:
    CUdeviceptr null_ptr_ = 0;
};

REGISTER_KERNEL(TritonMatmulDescriptor);

}

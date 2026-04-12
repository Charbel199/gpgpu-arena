#include "arena/kernels/matmul_base.hpp"

namespace arena {

struct MatmulNaive : MatmulDescriptorBase {
    std::string name() const override { return "matmul_naive"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "matmul/naive.cu"; }
    std::string function_name() const override { return "matmul_naive"; }
    std::string description() const override {
        return "Naive global memory matmul";
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override {
        return {
            .grid_x = static_cast<unsigned>((N_ + 15) / 16),
            .grid_y = static_cast<unsigned>((M_ + 15) / 16),
            .grid_z = 1,
            .block_x = 16, .block_y = 16, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }
};

REGISTER_KERNEL(MatmulNaive);

}

#include "arena/kernels/reduce_base.hpp"

namespace arena {

struct ReduceVectorized : ReduceDescriptorBase {
    std::string name() const override { return "reduce_vectorized"; }
    std::string ptx_path() const override { return "kernels/reduce_vectorized.ptx"; }
    std::string function_name() const override { return "reduce_sum_vectorized"; }
    std::string description() const override {
        return "SOL5: vectorized 128-bit loads + grid-stride + warp shuffle";
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int blocksize = 256;
        constexpr int sm_count = 80;
        return {
            .grid_x = sm_count * 32,
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 32 * sizeof(float)
        };
    }
};

REGISTER_KERNEL(ReduceVectorized);

}

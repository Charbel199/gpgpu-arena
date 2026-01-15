#include "arena/kernels/reduce_base.hpp"

namespace arena {

struct ReduceGridStride : ReduceDescriptorBase {
    std::string name() const override { return "reduce_grid_stride"; }
    std::string ptx_path() const override { return "kernels/reduce_grid_stride.ptx"; }
    std::string description() const override {
        return "SOL3: grid-stride loop + warp shuffle";
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override {
        // persistent threads: blocksize=256, gridsize = sm_count * 32
        constexpr int blocksize = 256;
        constexpr int sm_count = 80; //TODO: Should be dynamic
        return {
            .grid_x = sm_count * 32,
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = sizeof(float)
        };
    }
};

REGISTER_KERNEL(ReduceGridStride);

}

#include "arena/kernels/reduce_base.hpp"

namespace arena {

struct ReduceWarpShuffle : ReduceDescriptorBase {
    std::string name() const override { return "reduce_warp_shuffle"; }
    std::string ptx_path() const override { return "kernels/reduce_warp_shuffle.ptx"; }
    std::string description() const override {
        return "SOL2: warp shuffle + block atomic";
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int blocksize = 128;
        return {
            .grid_x = static_cast<unsigned>((n_ + blocksize - 1) / blocksize),
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = sizeof(float)
        };
    }
};

REGISTER_KERNEL(ReduceWarpShuffle);

}

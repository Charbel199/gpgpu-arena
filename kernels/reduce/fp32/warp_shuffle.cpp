#include "arena/categories/reduce_base.hpp"

namespace arena {

struct ReduceWarpShuffle : ReduceDescriptorBase {
    std::string name() const override { return "reduce_warp_shuffle"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/fp32/warp_shuffle.cu"; }
    std::string function_name() const override { return "reduce_sum_warp_shuffle"; }
    std::string description() const override {
        return "SOL2: warp shuffle + block atomic";
    }
    
    // Capped at 1024 by the warp_sums[] arrays, which hold one entry per
    // warp, and restricted to powers of two because the tree reductions
    // halve the block each step.
    std::vector<int> tunable_block_sizes() const override {
        return {64, 128, 256, 512, 1024};
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        const int blocksize = block_size_or(128);
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

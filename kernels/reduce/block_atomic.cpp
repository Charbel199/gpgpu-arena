#include "arena/kernels/reduce_base.hpp"

namespace arena {

struct ReduceBlockAtomic : ReduceDescriptorBase {
    std::string name() const override { return "reduce_block_atomic"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/block_atomic.cu"; }
    std::string function_name() const override { return "reduce_sum_block_atomic"; }
    std::string description() const override {
        return "SOL1: block-level shared memory atomic";
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

REGISTER_KERNEL(ReduceBlockAtomic);

}

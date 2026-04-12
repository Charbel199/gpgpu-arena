#include "arena/kernels/reduce_base.hpp"

namespace arena {

struct ReduceGridStride : ReduceDescriptorBase {
    std::string name() const override { return "reduce_grid_stride"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/grid_stride.cu"; }
    std::string function_name() const override { return "reduce_sum_grid_stride"; }
    std::string description() const override {
        return "SOL3: grid-stride loop + warp shuffle";
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int blocksize = 256;
        return {
            .grid_x = static_cast<unsigned>(sm_count() * 32),
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = sizeof(float)
        };
    }
};

REGISTER_KERNEL(ReduceGridStride);

}

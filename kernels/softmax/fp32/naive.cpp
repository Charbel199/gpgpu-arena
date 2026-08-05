#include "arena/categories/softmax_base.hpp"

namespace arena {

struct SoftmaxNaive : SoftmaxDescriptorBase {
    std::string name() const override { return "softmax_naive"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "softmax/fp32/naive.cu"; }
    std::string function_name() const override { return "softmax_naive"; }
    std::string description() const override {
        return "Naive row-wise softmax: one block per row, 3-pass (max, exp+sum, normalize)";
    }

    // Capped at 1024 by the warp_sums[] arrays, which hold one entry per
    // warp, and restricted to powers of two because the tree reductions
    // halve the block each step.
    std::vector<int> tunable_block_sizes() const override {
        return {64, 128, 256, 512, 1024};
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        const int blocksize = block_size_or(256);
        return {
            .grid_x = static_cast<unsigned>(rows_),
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = static_cast<unsigned>(blocksize * sizeof(float))
        };
    }
};

REGISTER_KERNEL(SoftmaxNaive);

}

#include "arena/kernels/softmax_base.hpp"

namespace arena {

struct SoftmaxNaive : SoftmaxDescriptorBase {
    std::string name() const override { return "softmax_naive"; }
    std::string ptx_path() const override { return "kernels/softmax_naive.ptx"; }
    std::string function_name() const override { return "softmax_naive"; }
    std::string description() const override {
        return "Naive row-wise softmax: one block per row, 3-pass (max, exp+sum, normalize)";
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int blocksize = 256;
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

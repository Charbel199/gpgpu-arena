#include "arena/kernels/reduce_base.hpp"

namespace arena {

struct ReduceBaseline : ReduceDescriptorBase {
    std::string name() const override { return "reduce_baseline"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/baseline.cu"; }
    std::string function_name() const override { return "reduce_sum_baseline"; }
    std::string description() const override {
        return "Baseline: naive atomicAdd per thread";
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int blocksize = 64;
        return {
            .grid_x = static_cast<unsigned>((n_ + blocksize - 1) / blocksize),
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }
};

REGISTER_KERNEL(ReduceBaseline);

}

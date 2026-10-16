#include "arena/categories/attention_base.hpp"

namespace arena {

struct AttentionNaive : AttentionDescriptorBase {
    std::string name() const override { return "attention_naive"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "attention/fp32/naive.cu"; }
    std::string function_name() const override { return "attention_naive"; }
    std::string description() const override {
        return "Naive attention: one block per query row, scores in shared memory";
    }

    // Powers of two, because the reductions halve the block each step.
    std::vector<int> tunable_block_sizes() const override { return {64, 128, 256}; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        const int block = block_size_or(128);
        return {
            .grid_x = static_cast<unsigned>(S_), .grid_y = static_cast<unsigned>(BH_), .grid_z = 1,
            .block_x = static_cast<unsigned>(block), .block_y = 1, .block_z = 1,
            // S scores plus one float of scratch per thread; above 48 KB the
            // loader opts in automatically.
            .shared_mem_bytes = static_cast<unsigned>((S_ + block) * sizeof(float))
        };
    }
};

REGISTER_KERNEL(AttentionNaive);

}

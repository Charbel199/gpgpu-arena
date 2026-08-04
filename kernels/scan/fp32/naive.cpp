#include "arena/categories/scan_base.hpp"

namespace arena {

struct ScanNaive : ScanDescriptorBase {
    std::string name() const override { return "scan_naive"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "scan/fp32/naive.cu"; }
    std::string function_name() const override { return "exclusive_scan_naive"; }
    std::string description() const override {
        return "Naive scan (single block only)";
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
            .grid_x = 1, // 1 block
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = static_cast<unsigned>(blocksize * sizeof(float))
        };
    }
    
};

REGISTER_KERNEL(ScanNaive);

}

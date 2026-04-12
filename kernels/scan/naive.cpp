#include "arena/kernels/scan_base.hpp"

namespace arena {

struct ScanNaive : ScanDescriptorBase {
    std::string name() const override { return "scan_naive"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "scan/naive.cu"; }
    std::string function_name() const override { return "exclusive_scan_naive"; }
    std::string description() const override {
        return "Naive scan (single block only)";
    }
    
    KernelLoader::LaunchConfig get_launch_config() const override {
 
        constexpr int blocksize = 256;
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

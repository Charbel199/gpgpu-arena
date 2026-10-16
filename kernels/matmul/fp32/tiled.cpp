#include "arena/categories/matmul_base.hpp"

namespace arena {

// A CUDA matmul slot whose launch shape comes from the source, so a candidate
// can move from 16x16 tiles to register-blocked 128x128 ones without a new
// descriptor. The source declares, for example,
//   // arena: BLOCK_M=128 BLOCK_N=128 THREADS=256 SHMEM=0
// BLOCK_M x BLOCK_N is the tile of C one block owns. SHMEM is dynamic shared
// memory in bytes; above 48 KB the loader opts in automatically.
struct MatmulTiled : MatmulDescriptorBase {
    std::string name() const override { return "matmul_tiled"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "matmul/fp32/tiled.cu"; }
    std::string function_name() const override { return "matmul_tiled"; }
    std::string description() const override {
        return "Shared-memory tiled matmul slot; tile and block set by the source";
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        const auto& c = compile_result_.constants;
        auto get = [&](const char* k, int d) { return c.count(k) ? c.at(k) : d; };
        const int bm = std::max(1, get("BLOCK_M", 16));
        const int bn = std::max(1, get("BLOCK_N", 16));
        return {
            .grid_x = static_cast<unsigned>((N_ + bn - 1) / bn),
            .grid_y = static_cast<unsigned>((M_ + bm - 1) / bm),
            .grid_z = 1,
            .block_x = static_cast<unsigned>(get("THREADS", 256)), .block_y = 1, .block_z = 1,
            .shared_mem_bytes = static_cast<unsigned>(get("SHMEM", 0))
        };
    }
};

REGISTER_KERNEL(MatmulTiled);

}

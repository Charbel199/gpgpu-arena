#include "arena/categories/scan_base.hpp"

namespace arena {

// A scan slot that does not cap performance: a persistent grid of
// GRID_PER_SM blocks per SM and a zeroed workspace for blocks to hand partial
// sums to each other, which is what a single-pass decoupled look-back scan
// needs. The source picks its launch shape with a directive such as
//   // arena: THREADS=256 GRID_PER_SM=4 SHMEM=0
// so a candidate can change it without a new descriptor.
struct ScanMultiblock : ScanDescriptorBase {
    std::string name() const override { return "scan_multiblock"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "scan/fp32/multiblock.cu"; }
    std::string function_name() const override { return "exclusive_scan_multiblock"; }
    std::string description() const override {
        return "Multi-block single-pass scan slot; launch shape set by the source";
    }

    std::vector<int> tunable_block_sizes() const override {
        return {64, 128, 256, 512, 1024};
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        const auto& c = compile_result_.constants;
        const int threads = block_size_ > 0 ? block_size_
                          : c.count("THREADS") ? c.at("THREADS") : 256;
        const int per_sm = c.count("GRID_PER_SM") ? c.at("GRID_PER_SM") : 4;
        return {
            .grid_x = static_cast<unsigned>(std::max(1, sm_count() * per_sm)),
            .grid_y = 1, .grid_z = 1,
            .block_x = static_cast<unsigned>(threads), .block_y = 1, .block_z = 1,
            .shared_mem_bytes = static_cast<unsigned>(c.count("SHMEM") ? c.at("SHMEM") : 0)
        };
    }

    void allocate(Context& ctx) override {
        ScanDescriptorBase::allocate(ctx);
        // 16 bytes for every 32 elements leaves room for per-tile aggregate,
        // prefix and flag at any tile size from 32 up, plus a 64-byte header
        // for counters.
        workspace_bytes_ = 64 + 16 * (((size_t)n_ + 31) / 32);
        d_workspace_ = ctx.allocate(workspace_bytes_);
    }

    void initialize(Context& ctx) override {
        ScanDescriptorBase::initialize(ctx);
        ctx.zero_device(d_workspace_, workspace_bytes_);
    }

    // The tile counter and flags must start at zero on every launch.
    void reset(Context& ctx) override {
        ScanDescriptorBase::reset(ctx);
        ctx.zero_device(d_workspace_, workspace_bytes_);
    }

    void cleanup(Context& ctx) override {
        ctx.free(d_workspace_);
        d_workspace_ = 0;
        ScanDescriptorBase::cleanup(ctx);
    }

    std::vector<void*> get_kernel_args() override {
        return { &d_input_, &d_output_, &n_, &d_workspace_ };
    }

private:
    CUdeviceptr d_workspace_ = 0;
    size_t workspace_bytes_ = 0;
};

REGISTER_KERNEL(ScanMultiblock);

}

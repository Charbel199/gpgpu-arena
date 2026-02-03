#include "arena/kernels/reduce_base.hpp"
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace arena {

class ReduceTwoStage : public ReduceDescriptorBase {
public:
    std::string name() const override { return "reduce_two_stage"; }
    std::string description() const override {
        return "Two-stage reduction: eliminates global atomic contention";
    }
    std::string ptx_path() const override { return "kernels/reduce_two_stage.ptx"; }
    std::string function_name() const override { return "reduce_sum_blocks"; }

    bool uses_ptx() const override { return false; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        // this is for the first stage //TODO: Not very clean for both stages
        constexpr int blocksize = 256;
        constexpr int sm_count = 80;
        return {
            .grid_x = sm_count * 32,
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 33 * sizeof(float)
        };
    }

    void allocate(Context& ctx) override {
        size_input_ = n_ * sizeof(float);
        size_output_ = sizeof(float);

        // allocate extra buffer for block results
        auto config = get_launch_config();
        num_blocks_ = config.grid_x;
        size_block_results_ = num_blocks_ * sizeof(float);

        d_input_ = ctx.allocate(size_input_);
        d_output_ = ctx.allocate(size_output_);
        d_block_results_ = ctx.allocate(size_block_results_);

        // load module and functions ONCE
        if (!module_) {
            module_ = loader_.load_module(ptx_path());
            func_stage1_ = loader_.get_function(module_, "reduce_sum_blocks");
            func_stage2_ = loader_.get_function(module_, "reduce_sum_final");
        }
    }

    void initialize(Context& ctx) override {
        ReduceDescriptorBase::initialize(ctx);

        // zero the intermediate block_results buffer
        std::vector<float> zeros(num_blocks_, 0.0f);
        ctx.copy_to_device(d_block_results_, zeros.data(), size_block_results_);
    }

    void cleanup(Context& ctx) override {
        ctx.free(d_input_);
        ctx.free(d_output_);
        ctx.free(d_block_results_);
        d_input_ = d_output_ = d_block_results_ = 0;
    }

    void execute(Context& ctx) override {
        // module already loaded in allocate(), just launch kernels
        auto config1 = get_launch_config();

        // Stage 1: Block-level reduction
        void* args1[] = { &d_input_, &d_block_results_, &n_ };
        loader_.launch(func_stage1_, config1, args1);

        // CRITICAL: Stage 1 must complete before stage 2 (bug that I didn't notice for 1 hour)
        cuStreamSynchronize(0);

        // Stage 2: Final reduction (1 block, 256 threads)
        KernelLoader::LaunchConfig config2 = {
            .grid_x = 1,
            .grid_y = 1, .grid_z = 1,
            .block_x = 256, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 33 * sizeof(float)
        };

        void* args2[] = { &d_block_results_, &d_output_, &num_blocks_ };
        loader_.launch(func_stage2_, config2, args2);

        // ensure Stage 2 completes before returning
        cuStreamSynchronize(0);
    }

private:
    CUdeviceptr d_block_results_ = 0;
    size_t size_block_results_ = 0;
    unsigned int num_blocks_ = 0;

    KernelLoader loader_;
    CUmodule module_ = nullptr;
    CUfunction func_stage1_ = nullptr;
    CUfunction func_stage2_ = nullptr;
};

REGISTER_KERNEL(ReduceTwoStage);

}

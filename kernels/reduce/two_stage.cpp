#include "arena/kernels/reduce_base.hpp"
#include <cuda.h>
#include <string>

namespace arena {

class ReduceTwoStage : public ReduceDescriptorBase {
public:
    std::string name() const override { return "reduce_two_stage"; }
    std::string description() const override {
        return "Two-stage reduction: eliminates global atomic contention";
    }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/two_stage.cu"; }
    std::string function_name() const override { return "reduce_sum_blocks"; }

    // two_stage manages its own module + two-kernel launch in execute()
    bool uses_module() const override { return false; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int blocksize = 256;
        return {
            .grid_x = static_cast<unsigned>(sm_count() * 32),
            .grid_y = 1, .grid_z = 1,
            .block_x = blocksize, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 33 * sizeof(float)
        };
    }

    void allocate(Context& ctx) override {
        capture_device_props(ctx);

        size_input_ = n_ * sizeof(float);
        size_output_ = sizeof(float);

        config1_ = get_launch_config();
        num_blocks_ = config1_.grid_x;
        size_block_results_ = num_blocks_ * sizeof(float);

        d_input_ = ctx.allocate(size_input_);
        d_output_ = ctx.allocate(size_output_);
        d_block_results_ = ctx.allocate(size_block_results_);

        // load module and get both stage functions
        if (!module_) {
            module_ = loader_.load_module(module_path());
            func_stage1_ = loader_.get_function(module_, "reduce_sum_blocks");
            func_stage2_ = loader_.get_function(module_, "reduce_sum_final");
        }

        args1_[0] = &d_input_;
        args1_[1] = &d_block_results_;
        args1_[2] = &n_;

        config2_ = {
            .grid_x = 1,
            .grid_y = 1, .grid_z = 1,
            .block_x = 256, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 33 * sizeof(float)
        };

        args2_[0] = &d_block_results_;
        args2_[1] = &d_output_;
        args2_[2] = &num_blocks_;
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
        // Stage 1: Block-level reduction (configs/args prepared in allocate())
        loader_.launch(func_stage1_, config1_, args1_);

        // CRITICAL: Stage 1 must complete before stage 2 (bug that I didn't notice for 1 hour)
        cuStreamSynchronize(0);

        // Stage 2: Final reduction (configs/args prepared in allocate())
        loader_.launch(func_stage2_, config2_, args2_);

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

    // pre-prepared launch configs and args (set once in allocate())
    KernelLoader::LaunchConfig config1_;
    KernelLoader::LaunchConfig config2_;
    void* args1_[3];
    void* args2_[3];
};

REGISTER_KERNEL(ReduceTwoStage);

}

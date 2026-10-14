#include "arena/categories/reduce_base.hpp"
#include "arena/warp_abi.hpp"

namespace arena {

class WarpReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "warp_reduce"; }
    std::string description() const override { return "NVIDIA Warp reduce with atomic add"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/fp32/reduce.warp.py"; }

    // Refuse to run against a Warp whose layout these structs do not match,
    // rather than launching, reading a garbage thread count and reporting an
    // impossible bandwidth. warp_base reports its version in the constants.
    void allocate(Context& ctx) override {
        check_warp_version(compile_result_.constants);
        ReduceDescriptorBase::allocate(ctx);
    }

    KernelLoader::LaunchConfig get_launch_config() const override {
        return {
            .grid_x = static_cast<unsigned>((n_ + 255) / 256),
            .grid_y = 1, .grid_z = 1,
            .block_x = 256, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }

    // warp kernel signature: (launch_bounds_t dim, array_t<f32> input, array_t<f32> output, int32 n)
    std::vector<void*> get_kernel_args() override {
        dim_ = warp_bounds_1d(n_);
        arr_input_ = warp_array_1d(d_input_, n_);
        arr_output_ = warp_array_1d(d_output_, 1);
        arg_n_ = n_;
        return { &dim_, &arr_input_, &arr_output_, &arg_n_ };
    }

private:
    WarpLaunchBounds dim_ = {};
    WarpArrayF32 arr_input_ = {};
    WarpArrayF32 arr_output_ = {};
    int32_t arg_n_ = 0;
};

REGISTER_KERNEL(WarpReduceDescriptor);

}

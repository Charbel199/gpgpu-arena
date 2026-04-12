#include "arena/kernels/reduce_base.hpp"
#include <cstring>

namespace arena {

// Warp ABI structs must match warp/native/array.h and warp/native/builtin.h
struct WarpLaunchBounds {
    int32_t shape[4];   // {total_threads, 0, 0, 0} for 1D
    int32_t ndim;
    int32_t _pad;
    int64_t size;       // total thread count
};

struct WarpArrayF32 {
    uint64_t data;      // CUdeviceptr
    uint64_t grad;      // 0 (no gradient)
    int32_t shape[4];   // {N, 0, 0, 0}
    int32_t strides[4]; // {sizeof(float), 0, 0, 0}
    int32_t ndim;
    int32_t _pad;
};

class WarpReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "warp_reduce"; }
    std::string description() const override { return "NVIDIA Warp reduce with atomic add"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/reduce.warp.py"; }

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
        memset(&dim_, 0, sizeof(dim_));
        dim_.shape[0] = n_;
        dim_.ndim = 1;
        dim_.size = n_;

        memset(&arr_input_, 0, sizeof(arr_input_));
        arr_input_.data = d_input_;
        arr_input_.shape[0] = n_;
        arr_input_.strides[0] = sizeof(float);
        arr_input_.ndim = 1;

        memset(&arr_output_, 0, sizeof(arr_output_));
        arr_output_.data = d_output_;
        arr_output_.shape[0] = 1;
        arr_output_.strides[0] = sizeof(float);
        arr_output_.ndim = 1;

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

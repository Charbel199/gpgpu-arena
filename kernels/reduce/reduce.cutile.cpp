#include "arena/kernels/reduce_base.hpp"

namespace arena {

class CuTileReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "cutile_reduce"; }
    std::string description() const override { return "cuTile reduce with ct.sum"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/reduce.cutile.py"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        int tile_size = compile_result_.constants.at("TILE_SIZE");
        return {
            .grid_x = static_cast<unsigned>((n_ + tile_size - 1) / tile_size),
            .grid_y = 1, .grid_z = 1,
            .block_x = static_cast<unsigned>(compile_result_.block_dim > 0 ? compile_result_.block_dim : 128),
            .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }

    // TODO: cuTile ABI expands each tensor arg into (ptr, shape, stride) at the PTX level.
    //       The layout below was reverse-engineered for 1D, verify for 2D+ tensors
    std::vector<void*> get_kernel_args() override {
        // reduce_sum(input, output, TILE_SIZE: Constant)
        // PTX params: input(ptr,shape[0],stride[0]) output(ptr,shape[0],stride[0]) padding
        arg_input_ptr_     = d_input_;
        arg_input_shape_   = static_cast<uint32_t>(n_);
        arg_input_stride_  = 1u;
        arg_output_ptr_    = d_output_;
        arg_output_shape_  = 1u;
        arg_output_stride_ = 1u;
        arg_const_tile_sz_ = static_cast<uint32_t>(compile_result_.constants.at("TILE_SIZE"));
        return {
            &arg_input_ptr_, &arg_input_shape_, &arg_input_stride_,
            &arg_output_ptr_, &arg_output_shape_, &arg_output_stride_,
            &arg_const_tile_sz_
        };
    }

private:
    uint64_t arg_input_ptr_ = 0, arg_output_ptr_ = 0;
    uint32_t arg_input_shape_ = 0, arg_input_stride_ = 0;
    uint32_t arg_output_shape_ = 0, arg_output_stride_ = 0;
    uint32_t arg_const_tile_sz_ = 0;
};

REGISTER_KERNEL(CuTileReduceDescriptor);

}

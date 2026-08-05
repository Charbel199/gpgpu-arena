#include "arena/categories/reduce_base.hpp"

namespace arena {

class CuTileReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "cutile_reduce"; }
    std::string description() const override { return "cuTile reduce with ct.sum"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/fp32/reduce.cutile.py"; }

    // TILE_SIZE is a ct.Constant, baked into the cubin, so each value here is
    // a separate compile. The thread block itself stays where cuTile puts it:
    // occupancy=2 in the source is what decides that, not the tile.
    std::vector<CompileDefines> tunable_compile_options() const override {
        return {{{"TILE_SIZE", 256}}, {{"TILE_SIZE", 1024}}, {{"TILE_SIZE", 4096}}};
    }

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

    // Each array becomes (ptr, shape[0..ndim), stride[0..ndim)) in the launch
    // signature. Confirmed against the cubin with cuobjdump under cuTile 1.5's
    // cutile_python_v2 convention: six parameters at offsets 0x00, 0x08, 0x0c,
    // 0x10, 0x18, 0x1c. ct.Constant values are baked in and take no slot, so
    // TILE_SIZE is not passed.
    std::vector<void*> get_kernel_args() override {
        arg_input_ptr_     = d_input_;
        arg_input_shape_   = static_cast<uint32_t>(n_);
        arg_input_stride_  = 1u;
        arg_output_ptr_    = d_output_;
        arg_output_shape_  = 1u;
        arg_output_stride_ = 1u;
        return {
            &arg_input_ptr_, &arg_input_shape_, &arg_input_stride_,
            &arg_output_ptr_, &arg_output_shape_, &arg_output_stride_,
        };
    }

private:
    uint64_t arg_input_ptr_ = 0, arg_output_ptr_ = 0;
    uint32_t arg_input_shape_ = 0, arg_input_stride_ = 0;
    uint32_t arg_output_shape_ = 0, arg_output_stride_ = 0;
};

REGISTER_KERNEL(CuTileReduceDescriptor);

}

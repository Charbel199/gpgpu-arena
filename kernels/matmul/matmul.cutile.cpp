#include "arena/kernels/matmul_base.hpp"

namespace arena {

class CuTileMatmulDescriptor : public MatmulDescriptorBase {
public:
    std::string name() const override { return "cutile_matmul"; }
    std::string description() const override { return "cuTile matmul with ct.mma (tf32 tensor cores)"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "matmul/matmul.cutile.py"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        int block_m = compile_result_.constants.at("BLOCK_M");
        int block_n = compile_result_.constants.at("BLOCK_N");
        return {
            .grid_x = static_cast<unsigned>((M_ + block_m - 1) / block_m),
            .grid_y = static_cast<unsigned>((N_ + block_n - 1) / block_n),
            .grid_z = 1,
            .block_x = static_cast<unsigned>(compile_result_.block_dim > 0 ? compile_result_.block_dim : 128),
            .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }

    // TODO: cuTile ABI for 2D tensors reverse-engineered layout, verify with cuobjdump
    //       Each 2D tensor: ptr(u64) + shape[0](u32) + shape[1](u32) + stride[0](u32) + stride[1](u32)
    std::vector<void*> get_kernel_args() override {
        // A (M x K)
        arg_a_ptr_       = d_a_;
        arg_a_shape_[0]  = static_cast<uint32_t>(M_);
        arg_a_shape_[1]  = static_cast<uint32_t>(K_);
        arg_a_stride_[0] = static_cast<uint32_t>(K_);
        arg_a_stride_[1] = 1u;
        // B (K x N)
        arg_b_ptr_       = d_b_;
        arg_b_shape_[0]  = static_cast<uint32_t>(K_);
        arg_b_shape_[1]  = static_cast<uint32_t>(N_);
        arg_b_stride_[0] = static_cast<uint32_t>(N_);
        arg_b_stride_[1] = 1u;
        // C (M x N)
        arg_c_ptr_       = d_c_;
        arg_c_shape_[0]  = static_cast<uint32_t>(M_);
        arg_c_shape_[1]  = static_cast<uint32_t>(N_);
        arg_c_stride_[0] = static_cast<uint32_t>(N_);
        arg_c_stride_[1] = 1u;

        // cuTile ct.Constant params still occupy slots in the cubin (values baked in, slots must exist)
        arg_const_[0] = static_cast<uint32_t>(K_);   // K_DIM
        arg_const_[1] = static_cast<uint32_t>(compile_result_.constants.at("BLOCK_M"));
        arg_const_[2] = static_cast<uint32_t>(compile_result_.constants.at("BLOCK_N"));
        arg_const_[3] = static_cast<uint32_t>(compile_result_.constants.at("BLOCK_K"));

        return {
            &arg_a_ptr_, &arg_a_shape_[0], &arg_a_shape_[1], &arg_a_stride_[0], &arg_a_stride_[1],
            &arg_b_ptr_, &arg_b_shape_[0], &arg_b_shape_[1], &arg_b_stride_[0], &arg_b_stride_[1],
            &arg_c_ptr_, &arg_c_shape_[0], &arg_c_shape_[1], &arg_c_stride_[0], &arg_c_stride_[1],
            &arg_const_[0], &arg_const_[1], &arg_const_[2], &arg_const_[3],
        };
    }

private:
    uint64_t arg_a_ptr_ = 0, arg_b_ptr_ = 0, arg_c_ptr_ = 0;
    uint32_t arg_a_shape_[2] = {}, arg_a_stride_[2] = {};
    uint32_t arg_b_shape_[2] = {}, arg_b_stride_[2] = {};
    uint32_t arg_c_shape_[2] = {}, arg_c_stride_[2] = {};
    uint32_t arg_const_[4] = {};
};

REGISTER_KERNEL(CuTileMatmulDescriptor);

}

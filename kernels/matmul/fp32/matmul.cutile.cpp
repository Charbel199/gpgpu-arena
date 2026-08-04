#include "arena/categories/matmul_base.hpp"

namespace arena {

class CuTileMatmulDescriptor : public MatmulDescriptorBase {
public:
    std::string name() const override { return "cutile_matmul"; }
    std::string description() const override { return "cuTile matmul with ct.mma (tf32 tensor cores)"; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string function_name() const override { return compile_result_.kernel_name; }

    // The kernel casts its fp32 inputs to fp16 before ct.mma, so the multiply
    // keeps 11 mantissa bits. TF32 is the same 11 bits, so it gives the right
    // tolerance; this is what the hardcoded 5e-2 used to stand in for.
    ComputeMode compute_mode() const override { return ComputeMode::TF32; }

    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "matmul/fp32/matmul.cutile.py"; }

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

    // Each 2D array becomes ptr(u64) + shape[0] + shape[1] + stride[0] +
    // stride[1], all u32, then K as a runtime scalar. Confirmed against the
    // cubin with cuobjdump: 16 parameters, the three arrays at offsets 0x00,
    // 0x18 and 0x30. Strides are in elements.
    //
    // BLOCK_M/N/K stay ct.Constant and are baked in, so they take no slot.
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

        arg_k_ = static_cast<int32_t>(K_);

        return {
            &arg_a_ptr_, &arg_a_shape_[0], &arg_a_shape_[1], &arg_a_stride_[0], &arg_a_stride_[1],
            &arg_b_ptr_, &arg_b_shape_[0], &arg_b_shape_[1], &arg_b_stride_[0], &arg_b_stride_[1],
            &arg_c_ptr_, &arg_c_shape_[0], &arg_c_shape_[1], &arg_c_stride_[0], &arg_c_stride_[1],
            &arg_k_,
        };
    }

private:
    uint64_t arg_a_ptr_ = 0, arg_b_ptr_ = 0, arg_c_ptr_ = 0;
    uint32_t arg_a_shape_[2] = {}, arg_a_stride_[2] = {};
    uint32_t arg_b_shape_[2] = {}, arg_b_stride_[2] = {};
    uint32_t arg_c_shape_[2] = {}, arg_c_stride_[2] = {};
    int32_t  arg_k_ = 0;
};

REGISTER_KERNEL(CuTileMatmulDescriptor);

}

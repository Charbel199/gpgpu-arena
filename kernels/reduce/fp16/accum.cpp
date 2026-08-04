#include "arena/categories/reduce_base.hpp"


namespace arena {

// Shared setup for the two accumulator variants. A small grid on purpose: each
// thread then sums roughly n / (sm_count * 4 * 256) values, and it is the
// length of that per-thread run that decides whether a half accumulator
// survives.
class Fp16AccumBase : public ReduceDescriptorBase {
public:
    // fp16 in, fp32 out. Both kernels write a float, they differ only in
    // what they accumulate in, which is exactly the point of the pair.
    DType input_dtype() const override { return DType::FP16; }
    bool needs_compilation() const override { return true; }
    std::string module_path() const override { return compile_result_.module_path; }
    std::string source_path() const override { return "reduce/fp16/accum.cu"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        return {
            .grid_x = static_cast<unsigned>(sm_count() * 4),
            .grid_y = 1, .grid_z = 1,
            .block_x = 256, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 33 * sizeof(float)
        };
    }
};

struct ReduceFp16AccumFp16 : Fp16AccumBase {
    std::string name() const override { return "reduce_fp16_accum_fp16"; }
    std::string function_name() const override { return "reduce_sum_fp16_accum"; }
    std::string description() const override {
        return "fp16 input, half accumulator (stops counting past 2048)";
    }
};

struct ReduceFp16AccumFp32 : Fp16AccumBase {
    std::string name() const override { return "reduce_fp16_accum_fp32"; }
    std::string function_name() const override { return "reduce_sum_fp32_accum"; }
    std::string description() const override {
        return "fp16 input, float accumulator";
    }
};

REGISTER_KERNEL(ReduceFp16AccumFp16);
REGISTER_KERNEL(ReduceFp16AccumFp32);

}

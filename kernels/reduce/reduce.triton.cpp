#include "arena/kernels/reduce_base.hpp"

namespace arena {

class TritonReduceDescriptor : public ReduceDescriptorBase {
public:
    std::string name() const override { return "triton_reduce"; }
    std::string description() const override { return "Triton reduce with atomic add"; }
    std::string ptx_path() const override { return "kernels/reduce_triton_reduce.ptx"; }

    KernelLoader::LaunchConfig get_launch_config() const override {
        constexpr int BLOCK_SIZE = 128; // must match Triton compilation (.maxntid in PTX), TODO: It was supposed to be 256 but Triton generated code with 128, not sure why yet
        return {
            .grid_x = static_cast<unsigned>((n_ + BLOCK_SIZE - 1) / BLOCK_SIZE),
            .grid_y = 1, .grid_z = 1,
            .block_x = BLOCK_SIZE, .block_y = 1, .block_z = 1,
            .shared_mem_bytes = 0
        };
    }
};

REGISTER_KERNEL(TritonReduceDescriptor);

}

---
name: arena-add-kernel
description: Add a new kernel to GPGPU Arena, in CUDA C++, Triton, cuTile or Warp, including where its files go, how its C++ descriptor declares launch config and arguments, and how to promote a PGAIO candidate into a permanent kernel. Use when creating, registering or porting a kernel in the gpgpu-arena repo.
---

# Adding a kernel to GPGPU Arena

A kernel is two files in `kernels/<category>/<input dtype>/`:

1. The GPU source, compiled at run time: `name.cu`, `name.triton.py`, `name.cutile.py` or `name.warp.py`.
2. A C++ descriptor, `name.cpp` (or `name.triton.cpp` and so on), compiled into the binary. It names the kernel and decides its launch config, its arguments and how it is verified.

The folder is named after the input dtype because that is what shapes the code (`float*` against `__half*`). Variants that share an input type but differ in output type go in the same source file. Categories today: `reduce`, `scan`, `softmax`, `matmul`. Their shared logic (allocation, input generation, CPU reference, verification, FLOP and byte counts) lives in `include/arena/categories/<category>_base.hpp`, so a descriptor normally only adds what is specific to it.

Descriptors are found by `file(GLOB_RECURSE kernels/*.cpp)` in CMake, so a new descriptor needs a re-run of CMake. The Docker entrypoint does this on every start; natively, run `cmake -S . -B build && cmake --build build -j`.

## Pick the closest existing kernel and copy it

| You are adding | Start from |
| --- | --- |
| CUDA kernel, one launch | `kernels/reduce/fp32/grid_stride.{cu,cpp}` |
| CUDA kernel, several launches | `kernels/reduce/fp32/two_stage.{cu,cpp}` (`uses_module() = false`, launches in `execute()`) |
| Library call (CUB, cuBLAS) | `kernels/reduce/fp32/cub_reduce.cu` (descriptor and call in one `.cu`, linked at build time) |
| Triton | `kernels/reduce/fp32/reduce_grid_stride.triton.{py,cpp}` |
| cuTile | `kernels/reduce/fp32/reduce.cutile.{py,cpp}` |
| Warp | `kernels/reduce/fp32/reduce.warp.{py,cpp}` (hand-written ABI structs, checked against the Warp version) |
| fp16 input | `kernels/reduce/fp16/accum.{cu,cpp}` |

## Descriptor checklist

```cpp
#include "arena/categories/reduce_base.hpp"
namespace arena {
struct ReduceMyKernel : ReduceDescriptorBase {
    std::string name() const override { return "reduce_my_kernel"; }      // unique
    std::string description() const override { return "one line"; }
    std::string module_path() const override { return compile_result_.module_path; }
    bool needs_compilation() const override { return true; }
    std::string source_path() const override { return "reduce/fp32/my_kernel.cu"; }
    std::string function_name() const override { return "reduce_sum_my_kernel"; } // extern "C" name
    std::vector<int> tunable_block_sizes() const override { return {128, 256, 512}; } // optional
    KernelLoader::LaunchConfig get_launch_config() const override {
        const int block = block_size_or(256);
        return {.grid_x = static_cast<unsigned>(sm_count() * 32), .grid_y = 1, .grid_z = 1,
                .block_x = static_cast<unsigned>(block), .block_y = 1, .block_z = 1,
                .shared_mem_bytes = 0};
    }
};
REGISTER_KERNEL(ReduceMyKernel);
}
```

- CUDA sources export the entry point with `extern "C" __global__`, so the name is not mangled.
- Arguments come from the category base's `get_kernel_args()` (for reduce: input pointer, output pointer, `int n`). Override it only if the kernel needs something else.
- DSL descriptors read their entry point and launch shape from `compile_result_`: `kernel_name`, `num_warps`, `constants`, `block_dim`. Declare `tunable_compile_options()` to put compile-time knobs on the tuning axis.
- `input_dtype()` / `output_dtype()` default to fp32. Override them for narrow types; tolerance follows the output type.
- If the kernel accumulates into its output, `reset()` must restore it before each launch. The base classes already zero the output.

## Check it

```bash
$ARENA --list | grep my_kernel
$ARENA --run reduce_my_kernel -p n=1,1000003,67108864 --dist uniform,adversarial --seed 1,2
$ARENA --run reduce_my_kernel -p n=67108864 --l2 cold --profile
$ARENA --inspect reduce_my_kernel
```

All combinations must verify (exit code 0) before it is worth timing.

## Promoting a PGAIO candidate

A candidate from `pgaio/runs/<run>/candidates/` already fits an existing slot's ABI, so promotion is a copy plus a descriptor:

1. `python3 tools/pgaio.py best <run>` gives the candidate, the slot source and the slot descriptor.
2. Copy the candidate to `kernels/<category>/<dtype>/<new name>.<ext>` and rename its entry point to something unique (CUDA) or keep it (DSLs, whose descriptor reads it from the compile result).
3. Copy the slot's descriptor to `<new name>.cpp`, change `name()`, `description()`, `source_path()` and, for CUDA, `function_name()`.
4. Rebuild, then run the checks above. The new kernel now shows up in the GUI and in category runs next to the original.

Do this after a PGAIO run has finished, never during one: `kernels/` is protected while a run is active, and changing it rejects the iteration.

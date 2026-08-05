# GPGPU Arena

Write the same GPU kernel in CUDA C++, Triton, cuTile or Warp, then run them
side by side on the same inputs with the same measurement code.

![GPGPU Arena UI, matmul comparison across cuTile, Triton and CUDA](docs/ui-matmul.png)

## Quick start

You need Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
xhost +local:docker
docker compose up gui
```

## Headless mode

For scripts and CI. JSON goes to stdout and logging is turned off, so you can
pipe the output straight into other tools.

```bash
./arena --list                                    # kernels and categories
./arena --run reduce -p n=4000000 --runs 20       # one category
./arena --run cub_reduce -p n=1000000 --profile   # one kernel, with counters
./arena --run all --energy -o results.csv --format csv
```

Exit codes let a caller branch without parsing the output:

| Code | Meaning |
| --- | --- |
| 0 | Everything ran and verified |
| 1 | Ran, but something failed verification |
| 2 | Something errored (compile, launch, or device) |
| 3 | Bad arguments |

The JSON also records the device, driver, toolkit and clocks, so a saved
result file says what produced it. `./arena --help` lists every flag.

Build with `-DBUILD_GUI=OFF` if you want a headless binary with no GLFW or
OpenGL dependency.

## Tuning

Two things can be swept: problem size and kernel config.

Problem size is a range. `--sweep-min`, `--sweep-max` and `--sweep-factor` set
it. Leave them at zero and each category uses its own defaults.

Kernel config works differently on the two sides. A hand-written CUDA kernel
can just be relaunched at a different block size. A DSL kernel cannot, because
the block size is baked into the cubin: cuTile emits a `.reqntid` the launch
has to match exactly, and Triton derives the block from `num_warps` at compile
time. Changing those means recompiling. For a DSL, sweeping the block size is
autotuning.

`--sweep-block` handles both. CUDA kernels run at every block size they
declare, DSL kernels run at every compile config they declare, and each config
gets its own cached cubin. Use `--block <n>` or `--define KEY=n` to pin one.

Best against worst config over `reduce` at n=4M:

| kernel | best config | spread |
| --- | --- | --- |
| `reduce_block_atomic` | block=64 | 5.1x |
| `reduce_grid_stride` | block=128 | 3.9x |
| `reduce_warp_shuffle` | block=512 | 3.6x |
| `reduce_warp_shmem` | block=256 | 3.0x |
| `triton_reduce` | BLOCK_SIZE=1024, num_warps=8 | 2.2x |
| `cutile_reduce` | TILE_SIZE=1024 | 1.6x |
| `triton_reduce_grid_stride` | BLOCK_SIZE=4096, num_warps=2 | 1.6x |

The last row is why this matters. Tuned, that kernel runs in 0.0084 ms and
beats `cub_reduce` at 0.0133 and every hand-written CUDA kernel here. At the
value written in its source file it runs in 0.0131. Without this axis you are
comparing how well each kernel was guessed at, not what the language can do.

## Measurement

Timing comes in three tiers. They measure different things, so never add them
together or compare one against another.

| Tier | Fields | What it is |
| --- | --- | --- |
| Per invocation | `op_ms`, `gpu_ms`, `overhead_ms`, `launch_count` | Whole operation against summed SASS time. The gap is launch latency and sync between kernels. |
| Per process | `module_load_ms`, `first_launch_ms` | Cold start, paid once per process. |
| Build | `compile_ms`, `import_ms`, `invoke_ms` | One time, cached to disk. `compile_ms` is the DSL compiler itself, separate from Python startup. |

GFLOPS and GB/s divide by `op_ms`. Hardware counters divide by `gpu_ms`.

Warmup runs until the timings settle instead of a fixed count. It compares the
median of one window against the previous one, because a clock that is still
boosting drifts steadily and a variance test would call that stable. Results
carry `warmup_converged` so you can tell when a number came off an unsettled
clock. Use `--warmup <n>` if you want a fixed count instead.

Energy uses NVML over a sustained back-to-back window, because NVML updates
much more slowly than a single kernel runs. It reads whole-board power, so
`mj_per_op` is an upper bound on what a kernel actually costs. The runner logs
how busy the device was and warns when the loop was launch-bound.

## Numerics

Kernels report measured error, not just pass or fail. Every category checks
against a CPU reference summed in `double` and reports max and mean relative
error next to the tolerance it was judged against.

There are two errors because they answer different questions.

Arithmetic error is measured against the inputs the kernel actually received,
already rounded to its storage type. That part is the kernel's own doing, so
pass and fail are judged on it, and it is comparable within one dtype.

Total error is measured against the original fp32 data, so it includes what a
narrower storage type cost before the kernel ran. That is the number you can
compare across dtypes, and it is what the results table shows.

For an fp32 kernel the two are the same. For `reduce_fp16_accum_fp32` at 64M
they are `7.3e-07` and `6.6e-05`. Its summation is as clean as a good fp32
kernel, and storing the inputs in half still costs two orders of magnitude.

Tolerance comes from the mantissa of the output type, scaled by the square
root of the accumulation length, since summing more values genuinely loses
precision. This separates a kernel that is imprecise from one that is broken.
`reduce_baseline` over 64M ones reports `7.4e-01`, which is an fp32
accumulator saturating at 2^24, not a bug.

Pick the input data with `--dist`:

| Distribution | Use |
| --- | --- |
| `ones` | Quick sanity check. Hides ordering bugs, every reduce scores exactly zero. |
| `uniform` | Default. Mixed signs, so accumulation order starts to matter. |
| `normal` | Occasional large values. |
| `adversarial` | Mixed magnitudes, denormals and exact zeros. |

`--seed` makes a run repeatable. The CPU reference is rebuilt from the same
numbers the GPU saw.

A kernel declares its input and output types separately, because mixed
precision is normal: `reduce_fp16_accum_fp32` reads fp16 and writes fp32.
There is no accumulator type, since a kernel can have several at different
precisions. The framework only models what crosses the buffer boundary.

Tolerance follows the output type. A kernel cannot be more precise than the
type it writes into, so that is what it is held to. A kernel whose internals
are coarser will miss it, which is the point. `reduce_fp16_accum_fp16` writes
fp32 but sums in half, and fails.

Dtype is part of a kernel's identity rather than a switch, so the fp16
variants are separate rows in the same `reduce` table. At n=4M:

| kernel | in > out | error | |
| --- | --- | --- | --- |
| `reduce_fp16_accum_fp16` | fp16 > fp32 | 7.8e-03 | sums in half, fails |
| `reduce_fp16_accum_fp32` | fp16 > fp32 | 1.5e-07 | sums in float |
| `reduce_fp16_out_fp16` | fp16 > fp16 | 5.8e-04 | clean sum, half output |

Supported types are fp32, fp16, bf16, both OCP fp8 variants (e4m3 and e5m2)
and fp4 (e2m1). Buffers are sized in bits, so fp4 packs two values per byte.
Tolerance has a floor of half an ulp of the output type, so a kernel is never
asked to be more accurate than the type it writes into.

NVFP4 is not usable yet. The element type exists, but the format is fp4 values
with a shared e4m3 scale every 16 elements, and the base descriptors do not
allocate a second buffer for those scales. `dtype_is_block_scaled()` flags it
so nothing treats it as a plain narrow type.

Sources live in `kernels/<category>/<input dtype>/`. The folder is named after
what the kernel reads, since that is what shapes the code: reading `__half*`
instead of `float*` changes the signature and the load path, while the output
type is one store at the end. Variants that share an input type but differ in
output go in the same file, which is why all three above are in
`kernels/reduce/fp16/accum.cu`.

## Known issues

Counter values describe a warm launch. The counter pass runs twelve launches
back to back and keeps the smallest DRAM figure. Instrumentation re-cools L2
every few launches, so any single one is warm or cold depending on where it
lands, and since instrumentation can only add DRAM traffic the minimum is the
closest estimate of the untouched timed loop.

This is a different question from the one Nsight Compute answers by default.
For an 86 MB working set on a 128 MB L2, a cold launch moves 86 MB from DRAM
and a warm one moves 0.05 MB. Both are correct measurements of different
things. Expect `dram_read_bytes` to sit far below an `ncu` figure when the data
fits in cache, and to agree when it does not: at a 400 MB working set this
reports 400.1 MB against Nsight's 400.5.

The compile cache is keyed on source mtime and the defines a cubin was built
with, but not on target architecture, compiler version, or the rest of the
compile flags. A cubin can outlive the toolchain that built it and get reused.
This has produced wrong results that verification caught. After a toolchain
change, clear it:

```bash
rm build/kernels/*.cubin build/kernels/*.json
```

## How it works

```
Source (.cu / .triton.py / .cutile.py)
    |
    v
Runtime compiler (nvcc / Triton / cuTile)  ->  .cubin (cached)
    |
    v
cuModuleLoad  ->  cuLaunchKernel  ->  benchmark and profile
```

Everything compiles to a cubin, final SASS, at runtime on first use. Cubins are
cached on disk and later runs skip compilation. Edit a kernel source, run
again, and only that kernel rebuilds.

![Reduce sweep, 12 reduce kernels with speedup, wall against GPU time, and throughput against theoretical peak](docs/reduce-sweep.png)

## Adding a kernel

Each kernel is two files: the GPU source, and a C++ descriptor holding its
metadata.

CUDA C++:

```
kernels/reduce/my_kernel.cu     # extern "C" __global__ kernel
kernels/reduce/my_kernel.cpp    # name, launch config, args, verification
```

Triton:

```
kernels/reduce/my_kernel.triton.py    # @jit kernel plus triton_base.main()
kernels/reduce/my_kernel.triton.cpp   # descriptor using compile_result_
```

cuTile:

```
kernels/reduce/my_kernel.cutile.py    # @ct.kernel plus cutile_base.main()
kernels/reduce/my_kernel.cutile.cpp   # descriptor plus param buffer
```

The descriptor sets `needs_compilation() = true` and `source_path()`. Compiling,
loading, benchmarking and verifying happen automatically.

## Layout

Compiler backends in `src/arena/compilers/`, one per DSL. `CudaBackend` runs
nvcc. `TritonBackend`, `CuTileBackend` and `WarpBackend` run Python scripts
that write a cubin plus JSON metadata.

`KernelCompiler` picks a backend by file extension and owns the two-level
cache (memory, then disk with mtime invalidation), output naming and compile
timing.

The kernel loader in `src/arena/device/` loads cubins with `cuModuleLoad` and
launches with `cuLaunchKernel`.

Measurement code in `src/arena/measurement/` covers warmup, timing, CUPTI
profiling and NVML energy.

`Runner` ties it together: compile, warm up, benchmark, profile, verify.

## Profiling

Occupancy and IPC side by side, plus arithmetic intensity against performance
on a roofline:

![Profiling comparison and roofline model](docs/profiling-roofline.png)

The sub-kernel timeline, from the Activity API, splits multi-launch kernels
into their individual GPU invocations:

![Sub-kernel timeline showing reduce_two_stage split into reduce_sum_blocks and reduce_sum_final](docs/subkernel-timeline.png)

Counters need GPU performance counter access:

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia-profiler.conf
sudo update-initramfs -u
sudo reboot
```

Without it benchmarking still works and only the profiling pass fails.

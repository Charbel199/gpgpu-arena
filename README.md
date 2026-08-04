# GPGPU Arena

A CUDA kernel benchmarking platform. Write GPU kernels in CUDA C++, Triton, cuTile or any DSL benchmark them side by side with identical inputs and measurement infrastructure.

![GPGPU Arena UI - matmul comparison across cuTile, Triton, and CUDA](docs/ui-matmul.png)

## Quick Start

Requires Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
xhost +local:docker
docker compose up gui
```

## Headless Mode

For scripting, CI, and agentic workflows. Writes JSON to stdout by default,
with console logging suppressed so the payload is machine-readable:

```bash
./arena --list                                    # kernels + categories as JSON
./arena --run reduce -p n=4000000 --runs 20       # run a category
./arena --run cub_reduce -p n=1000000 --profile   # one kernel, with counters
./arena --run all --energy -o results.csv --format csv
```

Exit codes are load-bearing, so callers can branch without parsing output:

| Code | Meaning |
| --- | --- |
| 0 | All kernels ran and verified |
| 1 | Ran, but at least one kernel failed verification |
| 2 | At least one kernel errored (compile, launch, or device failure) |
| 3 | Usage error |

JSON output carries environment provenance (device, driver, toolkit, clocks)
alongside the results, so a result file records what produced it. Run
`./arena --help` for the full flag list.

`-DBUILD_GUI=OFF` builds a headless-only binary with no GLFW/OpenGL
dependency.

## Measurement Model

Timing is reported in three tiers, which are **not** comparable with each
other and are never summed:

| Tier | Fields | Meaning |
| --- | --- | --- |
| Per-invocation | `op_ms`, `gpu_ms`, `overhead_ms`, `launch_count` | Whole operation vs. summed SASS. The gap is launch latency and inter-kernel sync. |
| Per-process | `module_load_ms`, `first_launch_ms` | Cold-start cost, paid once per process. |
| Build | `compile_ms`, `import_ms`, `invoke_ms` | One-time, cached to disk. `compile_ms` is the DSL compiler's own time, separate from Python interpreter startup. |

Throughput (GFLOPS, GB/s) divides by `op_ms`. Hardware counters divide by
`gpu_ms`, because the range profiler samples a single launch replay.

Warmup runs until timings stabilise rather than a fixed count, using median
drift between consecutive windows — a sliding-variance test reports a
steadily boosting clock as stable. Results record `warmup_converged` so you
can tell when a number was measured on an unsettled clock.

Energy uses NVML over a sustained back-to-back window, since NVML updates far
more slowly than a single kernel runs. It reports whole-board energy, so
`mj_per_op` is an upper bound on a kernel's marginal cost; the runner logs a
device-busy estimate and warns when the loop is launch-bound.

## How It Works

```
Source (.cu / .triton.py / .cutile.py)
    |
    v
Runtime Compiler (nvcc / Triton / cuTile)  -->  .cubin (cached)
    |
    v
cuModuleLoad  -->  cuLaunchKernel  -->  Benchmark + Profile
```

All kernels compile to **cubin** (final SASS) at runtime on first use. Compiled cubins are cached on disk and subsequent runs skip compilation. Edit a kernel source file, re-run, and only that kernel recompiles.

![Reduce sweep - 12 reduce kernels with speedup, wall vs GPU time, and throughput vs theoretical peak](docs/reduce-sweep.png)

## Adding a Kernel

Each kernel has two files: a **source** (GPU code) and a **descriptor** (C++ metadata).

### CUDA C++

```
kernels/reduce/my_kernel.cu     # GPU kernel with extern "C" __global__
kernels/reduce/my_kernel.cpp    # Descriptor: name, launch config, args, verification
```

### Triton

```
kernels/reduce/my_kernel.triton.py    # Triton @jit kernel + triton_base.main()
kernels/reduce/my_kernel.triton.cpp   # Descriptor using compile_result_
```

### cuTile

```
kernels/reduce/my_kernel.cutile.py    # cuTile @ct.kernel + cutile_base.main()
kernels/reduce/my_kernel.cutile.cpp   # Descriptor using compile_result_ + param buffer
```

The descriptor declares `needs_compilation() = true` and `source_path()`. The arena compiles, loads, benchmarks, and verifies automatically.

## Architecture

- **Runtime Compilers** (`src/arena/compilers/`) - one per DSL. `CudaCompiler` runs nvcc, `TritonCompiler` and `CuTileCompiler` run Python scripts. Disk-cached with mtime invalidation.
- **Kernel Loader** - loads `.cubin` via `cuModuleLoad`, launches via `cuLaunchKernel`.
- **Benchmark** - CUDA events, median over N runs.
- **Profiler** - CUPTI Activity API (kernel time, registers, shared memory) + Range Profiler (occupancy, IPC, DRAM throughput).
- **Runner** - orchestrates: compile -> warmup -> benchmark -> profile -> verify.

## Profiling

Side-by-side occupancy/IPC across kernels plus an arithmetic-intensity vs performance roofline:

![Profiling comparison and roofline model](docs/profiling-roofline.png)

Sub-kernel timeline (Activity API) breaks multi-launch kernels into their individual GPU invocations:

![Sub-kernel timeline showing reduce_two_stage broken into reduce_sum_blocks and reduce_sum_final](docs/subkernel-timeline.png)

Hardware counter collection (occupancy, IPC, DRAM) requires GPU performance counter access:

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia-profiler.conf
sudo update-initramfs -u
sudo reboot
```

Without this, benchmarking still works - only the profiling pass will fail.

# GPGPU Arena

A CUDA kernel benchmarking platform. Write different GPU implementations, run them with the same inputs, and compare their performance.

## Getting Started

### Prerequisites
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU

### Quick Start (Docker)

```bash
# GUI
xhost +
docker compose run --rm run

# Or: interactive shell for development
docker compose run --rm arena
```

In the shell:
```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
./arena
```

### Native Build

Requires: CUDA Toolkit 11.0+, CMake 3.18+, C++17 compiler

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./arena
```

## How It Works

**Context** manages GPU resources. Pre-allocates memory so we're timing compute, not malloc.

**Kernel Loader** uses the CUDA Driver API to load `.ptx` files at runtime:
```
matmul.cu  ->  nvcc -ptx  ->  matmul.ptx  ->  cuModuleLoad()  ->  run
```
Change a kernel, recompile just that `.cu` file, re-run. No full rebuild.

**Benchmark** measures timing only - CUDA events around the kernel, median over N runs.

**Profiler** collects hardware counters via CUPTI (registers, shared memory, occupancy, IPC, DRAM throughput). Uses the Range Profiler API with kernel replay - slower but gives deep insight. _(Note: Will not work if you run the binary with nsys or ncu since CUPTI will be reserved by these external benchmarking/profiling tools)_

**Runner** orchestrates both: warmup -> benchmark -> profile (optional) -> verify.

## Timing: Wall vs GPU

The GUI shows two time columns:

- **Wall (ms)** - CUDA events (`cuEventRecord` start/stop). GPU-side timestamps. Includes everything between the markers: kernel execution, multi-kernel gaps, and for library calls (CUB), the host dispatch overhead. Median over N runs.
- **GPU (ms)** - CUPTI Activity API (`kernel->end - kernel->start`). Pure GPU execution time summed across all sub-kernels. No host overhead, no inter-kernel gaps. Single-run snapshot.

For hand-written PTX kernels, Wall and GPU are nearly identical (the kernel is the only thing between the events). For CUB/library kernels, Wall includes the C++ dispatch overhead - at small N this can be significant (e.g. 0.065 ms wall vs 0.010 ms GPU for CUB reduce at 1M elements).

## Validating with nsys / ncu

The arena is instrumented with NVTX markers. Run with nsys to get a visual timeline:

```bash
cd ~/projs/gpgpu-arena/build
nsys profile --trace=cuda,nvtx --stats=true --force-overwrite true -o /tmp/arena_nsys ./arena --cli
```

Open the report in Nsight Systems GUI:
```bash
nsys-ui /tmp/arena_nsys.nsys-rep
```

Look for:
- **NVTX row** - `BENCHMARK: reduce_grid_stride` -> `Run 0` -> `Kernel` (nested ranges)
- **GPU Kernels row** - actual kernel execution bars
- **CUDA API row** - `cuLaunchKernel`, `cuEventRecord`, `cuEventSynchronize`
- **CCCL row** - CUB/Thrust high-level API calls

The NVTX `Kernel` range spans from `launch_fn()` through `cuEventSynchronize(stop)` - this is the CPU-side view of the measured interval. The actual GPU time is on the kernel bar above.

For hardware counter validation, use ncu:
```bash
ncu --launch-skip 5 --launch-count 1 --kernel-name reduce_sum_grid_stride --set full ./arena --cli
```

### Enabling Hardware Profiling

The CUPTI Range Profiler requires access to GPU performance counters. By default, NVIDIA restricts this to admin users. To enable it for all users:

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia-profiler.conf
sudo update-initramfs -u
sudo reboot
```

Verify after reboot:
```bash
cat /proc/driver/nvidia/params | grep RestrictProfiling
# Should show: RestrictProfilingToAdminUsers: 0
```

Without this, benchmarking still works - only the profiling pass (occupancy, IPC, DRAM counters) will fail.

## Kernel Types

Each kernel in the table shows `[PTX]` or `[RT]`:

- **[PTX]** - loaded from `.ptx` at runtime via CUDA Driver API (`cuModuleLoad` + `cuLaunchKernel`). Minimal host overhead. One kernel per launch.
- **[RT]** - compiled into the executable via CUDA Runtime API (`cudaLaunchKernel`). Used by CUB, Thrust, or any `.cu` file linked directly. May launch multiple internal kernels.

Hover over any kernel name to see:
- The actual GPU kernel names (demangled from CUPTI Activity API)
- Per-kernel duration, register count, shared memory
- For multi-kernel ops: full breakdown of all sub-kernels

Hover over the GPU (ms) cell to see host overhead percentage when Wall > GPU.

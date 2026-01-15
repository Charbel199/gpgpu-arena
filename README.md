# ⚔️ GPGPU Arena

A CUDA kernel benchmarking platform that lets you pit different GPU implementations against each other and visualize their performance characteristics.

## 🚀 Getting Started

### Prerequisites
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU

### Quick Start (Docker)

```bash
# Build and run automatically
docker compose run --rm run

# Or: interactive shell for development
docker compose run --rm shell
```

In the shell:
```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
./arena
```

### Native Build (without Docker)

Requires: CUDA Toolkit 11.0+, CMake 3.18+, C++17 compiler

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./arena
```

### Project Structure

```
gpgpu-arena/
├── CMakeLists.txt           # Build configuration
├── src/
│   ├── main.cpp             # Arena entry point
│   └── arena/
│       ├── context.cpp      # GPU context management
│       ├── kernel_loader.cpp # Dynamic PTX loading
│       └── profiler.cpp     # CUPTI metrics collection
├── include/arena/
│   ├── context.hpp
│   ├── kernel_loader.hpp
│   └── profiler.hpp
└── kernels/
    ├── matmul_naive.cu      # Level I: The Recruit
    ├── matmul_tiled.cu      # Level II: The Tactician
    └── ...                  # Add more gladiators!
```

---

## Why?

I wanted to actually *understand* GPU performance, not just write kernels and hope they're fast. Tools like `nsys` and `ncu` are powerful but dump way too much info when you're starting out. 

This is my attempt at building something simpler: write a kernel, see how fast it is, compare it to other approaches, and understand *why* the optimized version wins.

The idea is to load kernels dynamically (no recompiling the whole thing every time), run them with the same inputs, and eventually get hardware-level metrics that explain performance differences.

---

## How It Works

**Context** manages GPU resources. Pre-allocates memory so we're timing compute, not malloc.

**Kernel Loader** uses the CUDA Driver API to load `.ptx` files at runtime:
```
matmul.cu  →  nvcc -ptx  →  matmul.ptx  →  cuModuleLoad()  →  run
```
Change a kernel, recompile just that `.cu` file, re-run. No full rebuild.

**Profiler** times execution and (eventually) pulls CUPTI metrics like occupancy and memory throughput.

---

## Kernel Progression

The plan is to implement the same algorithm (matrix multiply for now) at different optimization levels:

| Level | Approach | Point |
|-------|----------|-------|
| Naive | Global memory, simple loop | Baseline. See how bad unoptimized code is. |
| Tiled | Shared memory | Learn about the memory hierarchy |
| Register-blocked | + loop unrolling | Squeeze out more ILP |
| Tensor Core | WMMA intrinsics | Use the actual hardware |

Running them back-to-back shows exactly what each optimization buys you.

---

## Status

**Done:**
- Driver API context + memory management
- PTX loading at runtime
- Basic timing
- Docker setup

**Next:**
- CUPTI integration for real metrics (occupancy, throughput, etc.)
- Some kind of visualization
- Side-by-side comparison mode

**TODO:**
- Thrust/CUB support: add `uses_ptx()` and `execute()` to descriptors so host-side libraries can be benchmarked alongside raw CUDA kernels
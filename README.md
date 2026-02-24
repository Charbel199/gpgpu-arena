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
matmul.cu  →  nvcc -ptx  →  matmul.ptx  →  cuModuleLoad()  →  run
```
Change a kernel, recompile just that `.cu` file, re-run. No full rebuild.

**Profiler** collects timing and CUPTI metrics (registers, shared memory usage).

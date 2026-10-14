---
name: arena-cli
description: Run GPGPU Arena kernels headless and read the JSON results. Use when benchmarking, timing, verifying or comparing GPU kernels (CUDA, Triton, cuTile, Warp) in the gpgpu-arena repo, or when a task needs op_ms, gpu_ms, bandwidth, counters, energy or numerical error for a kernel.
---

# Running GPGPU Arena from the command line

Work from the gpgpu-arena repo root. There are two ways to invoke the binary:

| Where | Command | Notes |
| --- | --- | --- |
| Docker (all four DSLs) | `docker compose run --rm -T cli <args>` | Builds first. Add `-e ARENA_SKIP_BUILD=1` after `--rm` once it is built. `-T` keeps stdout clean. |
| Native | `build/arena <args>` (or another build dir) | Triton and cuTile only work if their Python packages are installed on the host. |

Below, `$ARENA` stands for whichever you use. stdout carries only the JSON payload, so it pipes into `jq` or `python3 -c`. Logs go to `arena.log`.

## Choosing what to run

```bash
$ARENA --list                                  # every kernel: name, category, dsl, source
$ARENA --run reduce                            # a whole category
$ARENA --run reduce_vectorized                 # one kernel
$ARENA --run all
```

Problem size, inputs and repetitions:

```bash
$ARENA --run reduce -p n=67108864 --runs 30
$ARENA --run reduce_vectorized -p n=1000,1000003,67108864      # one result per size
$ARENA --run matmul -p M=1024 -p N=1024 -p K=1024
$ARENA --run reduce --dist uniform,normal,adversarial --seed 1,2   # every combination runs
```

Every combination of `-p` lists, `--dist` and `--seed` runs, and each result says which one it was in `results[i].input`.

Measurement options:

| Flag | Effect |
| --- | --- |
| `--l2 cold` | Evict L2 before every timed launch. Use it whenever the working set could fit in L2 (128 MB on Blackwell): warm numbers there measure L2, not DRAM. At n=4M fp32 the gap was 2.2x. |
| `--profile` | Hardware counters: DRAM bytes, occupancy, IPC. Needs perf counter access. |
| `--energy` | NVML energy per op over a sustained window. Upper bound, whole board. |
| `--warmup <n>` | Fixed warmup instead of auto steady state. |
| `--block <n>` | Launch a CUDA kernel at this block size, if its descriptor lists it as tunable. |
| `--define K=v` | Compile-time knob for a DSL kernel (Triton `BLOCK_SIZE`, `num_warps`; cuTile `TILE_SIZE`). |
| `--sweep-block` | Run every block size (CUDA) or compile config (DSL) the kernel declares. |
| `--source-override <file>` | Run another source in this kernel's slot. See the arena-inspect skill. |
| `-o file --format csv` | Write CSV instead of JSON to stdout. |

## Exit codes

`0` all ran and verified, `1` something failed verification, `2` something errored (compile, launch, device), `3` bad arguments. Branch on these before parsing.

## Reading a result

Top level: `environment` (device, arch, `peak_dram_gbps`, `peak_fp32_gflops`, `l2_cache_bytes`, driver, toolkit), `config`, `results[]`, `summary`.

Each result:

```text
kernel, category, dsl, source        what ran (source is the file actually compiled)
input        {params, distribution, seed, cold_l2}
launch       {grid, block, shared_mem_bytes}
timing       op_ms, gpu_ms, overhead_ms, launch_count, all_times_ms
cold_start   module_load_ms, first_launch_ms
build        cache_hit, compile_ms, import_ms, invoke_ms, options
throughput   gflops, bandwidth_gbps          (denominator: op_ms)
counters     dram_read_bytes, dram_write_bytes, occupancy, ipc, regs_per_thread
energy       mj_per_op, avg_watts
numerics     max_rel_error, max_total_error, tolerance, input/output dtype
quality      warmup_converged, warmup_stop, sm clocks
sub_kernels  one entry per launch, in order, with start_offset_ms
verified, success, error
```

Rules that keep comparisons honest:

- Three timing tiers never mix. `op_ms`/`gpu_ms` are per invocation, `module_load_ms`/`first_launch_ms` are once per process, `compile_ms` is once per build. Do not add them.
- `op_ms` is the whole operation including gaps between launches. `gpu_ms` is the sum of kernel durations. A large `overhead_ms` means launch latency, not slow kernels.
- Throughput divides by `op_ms`; counters describe the kernel, so pair them with `gpu_ms`.
- Percentage of speed of light for a memory-bound kernel is `bandwidth_gbps / environment.peak_dram_gbps`. For compute-bound, `gflops / peak_fp32_gflops`. The ridge point is `peak_fp32_gflops / peak_dram_gbps` FLOP/byte.
- `max_total_error` is comparable across dtypes. Pass and fail use `max_rel_error` against `tolerance`.
- If `quality.warmup_converged` is false, the clock was still moving. Rerun or use more runs before drawing conclusions.
- Compare kernels only at the same `input` and `cold_l2` setting.

## Quick extraction

```bash
$ARENA --run reduce -p n=67108864 --l2 cold | python3 -c '
import json, sys
d = json.load(sys.stdin)
peak = d["environment"]["peak_dram_gbps"]
for r in sorted(d["results"], key=lambda r: (not r["success"], r["timing"]["op_ms"])):
    sol = r["throughput"]["bandwidth_gbps"] / peak
    print(r["kernel"].ljust(28), r["dsl"].ljust(7), "%.4f ms" % r["timing"]["op_ms"],
          "%5.1f%% SOL" % (100 * sol), "ok" if r["verified"] else "FAILED")
'
```

## When something looks wrong

- `success: false` with `nvcc ... failed` or `Triton compiler ... failed`: the compiler output is in `error`.
- `named symbol not found` with `--source-override`: the candidate renamed the entry point. It has to keep the slot's.
- `CUDA context lost earlier`: a previous kernel crashed the context. Everything after it in the same process is noise; rerun the rest separately.
- The compile cache lives in `build/kernels/` and is keyed on source contents, toolchain and GPU arch, so stale cubins are not a concern.

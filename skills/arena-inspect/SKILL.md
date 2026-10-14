---
name: arena-inspect
description: Inspect what a GPGPU Arena kernel compiled to (SASS, PTX, registers, spills, instruction mix) and explain its performance with counters and the roofline. Use when asked why a kernel is slow, what Triton/cuTile/Warp generated, how two kernels or candidates differ in SASS, or to try a modified kernel source without rebuilding.
---

# Inspecting kernels in GPGPU Arena

`$ARENA` is `docker compose run --rm -T cli` or a native `build/arena`, as in the arena-cli skill. Run from the repo root.

## What a kernel compiled to

```bash
$ARENA --inspect reduce_vectorized          # JSON summary, no launch
$ARENA --sass reduce_vectorized > v.sass    # SASS of the kernel's entry point
$ARENA --ptx reduce_vectorized > v.ptx      # PTX: CUDA and Triton only
```

`--inspect` fields:

| Field | Meaning |
| --- | --- |
| `source`, `slot_source` | File compiled, and the slot's own file (they differ under `--source-override`). |
| `descriptor_file` | The C++ descriptor: launch config, kernel arguments, verification. Read it to learn the slot's ABI. |
| `entry_point` | Function name the runner looks up in the cubin. |
| `dsl`, `input_dtype`, `output_dtype`, `parameters` | Identity of the slot. |
| `build.constants`, `build.num_warps`, `build.block_dim` | What a DSL compiler chose or was told (`--define`). |
| `resources.registers`, `shared`, `local`, `stack` | Per thread / per block. `local > 0` means spills or a local array. |
| `sass.classes` | Instruction counts by kind: `global_load_32/64/128`, `global_store_*`, `shared_mem`, `global_atomic`, `shared_atomic`, `warp_shuffle`, `warp_reduce`, `sync`, `tensor`, `fp32`, `async_copy`, `local_mem`, `control`, `int_addr_and_misc`, `other`. |
| `sass.mean_global_load_bytes` | 4 means scalar fp32 loads, 16 means 128-bit vector loads. |
| `sass.opcodes` | Every mnemonic with modifiers, counted. |

These are static counts of the instruction stream, not executed counts. A loop body appears once. Use them to see *what kind* of code was generated, and the counters to see what it did.

`$ARENA` paths are absolute in the binary's filesystem. Under Docker that is `/workspace/gpgpu-arena/...`; `repo_root` in the JSON tells you the prefix to strip.

## Explaining performance

Run with cold L2 and counters, at a size larger than L2:

```bash
$ARENA --run reduce_vectorized -p n=67108864 --l2 cold --profile --runs 30
```

Read it in this order:

1. Speed of light. Arithmetic intensity is `gflops / bandwidth_gbps`. Below the ridge (`peak_fp32_gflops / peak_dram_gbps`, about 70 FLOP/B on an RTX PRO 6000) the kernel is memory-bound and SOL is `bandwidth_gbps / peak_dram_gbps`; above, compute-bound against `peak_fp32_gflops`.
2. Launch overhead. `overhead_ms / op_ms` above roughly 20% means time is lost between or around launches. `launch_count` and `sub_kernels` show where.
3. Bytes moved. `counters.dram_read_bytes + dram_write_bytes` against the ideal (`bandwidth_gbps * op_ms`, the bytes the descriptor says the operation must touch). Well above 1x: uncoalesced, redundant or spilled traffic. Well below 1x: the data was served from L2, so use `--l2 cold` or a bigger size.
4. Occupancy. Below about 30% a memory-bound kernel rarely saturates DRAM. Check `resources.registers`, `shared`, and the launch grid against the SM count.
5. Instruction mix. Scalar loads (`mean_global_load_bytes` 4), per-element `global_atomic`, `local_mem`, or missing `tensor` instructions in a matmul are the usual suspects.

These counters are a warm-cache figure by default. The counter pass keeps the operation with the least DRAM traffic across 12 launches, grouped per operation for multi-launch kernels. With `--l2 cold` every launch starts cold.

## Comparing two kernels or two versions

```bash
$ARENA --inspect reduce_grid_stride  > a.json
$ARENA --inspect triton_reduce_grid_stride > b.json
python3 -c '
import json
a, b = (json.load(open(f))["sass"]["classes"] for f in ("a.json", "b.json"))
for k in sorted(set(a) | set(b)):
    if a.get(k, 0) != b.get(k, 0): print(k.ljust(22), a.get(k, 0), "->", b.get(k, 0))'
diff <($ARENA --sass reduce_grid_stride) <($ARENA --sass reduce_vectorized) | head -80
```

If a CUDA documentation MCP tool is connected (Nsight Copilot, CUDA docs), use it for opcodes or PTX instructions you do not recognize rather than guessing their meaning.

Comparing SASS across DSLs is often the quickest explanation: if the Triton version is faster, look for 128-bit loads, `REDUX` or `SHFL` warp reductions, fewer atomics, or a different unroll in its SASS, then write the same thing in the slower DSL.

## Trying a modified source without rebuilding

`--source-override <file>` compiles another file into an existing kernel slot and runs it with that slot's descriptor, buffers, launch config and verification:

```bash
cp kernels/reduce/fp32/grid_stride.cu /tmp/try.cu   # edit /tmp/try.cu
$ARENA --run reduce_grid_stride --source-override /tmp/try.cu -p n=67108864 --l2 cold
$ARENA --sass reduce_grid_stride --source-override /tmp/try.cu
```

Constraints, which together are the slot's ABI:

- Same file type as the slot (`.cu`, `.triton.py`, `.cutile.py`, `.warp.py`).
- Same entry point name (`entry_point` from `--inspect`) and the same parameter list in the same order.
- The descriptor decides the launch grid and block. Read `descriptor_file` before changing how work is split. For example `reduce_baseline` launches one thread per element, while `reduce_grid_stride` launches `32 * SMs` blocks and expects a grid-stride loop. Triton slots take the grid from the source's own constants, so a Triton candidate can change `BLOCK_SIZE`/`NUM_BLOCKS` freely.
- Under Docker the file must be inside the repo, since only the repo is mounted. Relative paths are resolved from the repo root.

If an idea needs a different ABI, it needs a new descriptor: see the arena-add-kernel skill.

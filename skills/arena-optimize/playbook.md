# Optimization playbook

What to try, in roughly the order it pays off, keyed to what limits the kernel at a given shape. `pgaio.py eval` prints which roof each shape is held to (`memory`, `compute` or `launch`); start from the shapes furthest from theirs.

## 1. Latency-bound shapes (small problems)

The roof is the latency floor: an empty launch plus one cold DRAM round trip, about 4 µs. A kernel at 10 µs on a 16 KB input is losing to fixed costs, not bandwidth. Reaching the floor is realistic for simple kernels (a reduce of 4096 elements does).

- **One launch.** A second kernel, or a memset before it, costs another launch floor plus a gap. Fold the work into one kernel, finishing with atomics or a last-block-done counter in scratch.
- **Right-size the grid.** Thousands of blocks for a few thousand elements is mostly scheduling. Use one block, or a few, for small inputs, choosing the grid with an expression on the size.
- **Avoid block-wide synchronization.** Each `__syncthreads` round on a tiny input is pure latency. Prefer warp-level work: one warp per row for short rows, and shuffles (`__shfl_xor_sync`, `tl.reduce` on a small block) over shared-memory trees.
- **Short dependency chains.** Unroll fixed small loops, and keep the critical path (load, reduce, store) short.
- **Dispatch by size, and across DSLs.** A manifest launch can carry `"when": "n < 1048576"` and its own `"source"`, so one candidate can use a CUDA kernel for small inputs and a Triton one for mid sizes. In the scan and softmax runs the mixed candidate beat every single-DSL one. Such candidates count as DSL `mixed`.
- **Expect a ceiling at mid sizes.** A shape around L2 size (a few MB to 16 MB) pays both the latency floor and the transfer, while the roof counts only the larger of the two, so around 60% can already be close to the best achievable.

## 2. Memory-bound shapes (large problems, low arithmetic intensity)

The roof is moving the ideal bytes once at peak DRAM bandwidth.

1. **Touch each byte once.** Multi-pass algorithms (max, then sum, then normalize) re-read their input. Keep the data in registers or shared memory between passes, or use online formulations: running max and rescaled sum for softmax, decoupled look-back for scan.
2. **Wide, coalesced accesses.** In SASS you want `LDG.E.128` / `STG.E.128`. Scalar `LDG.E` on fp32 means 4x the load instructions. Check alignment and contiguity; see the DSL notes below.
3. **Enough bytes in flight.** DRAM needs many outstanding loads per SM. Raise occupancy, issue several independent loads per thread before using them, or prefetch the next tile while computing the current one: persistent blocks, async copies, TMA.
4. **Cache policy.** `evict_first` on streamed input and streaming stores (`st.global.cs`) free L2 for what is reused. An `evict_last` on the output only helps while it fits in L2; at the large shapes it does not, so do not count on it.
5. **Avoid atomics contention.** One global atomic per element serializes. Reduce within the warp and block first; one atomic per block is fine.

## 3. Compute-bound shapes (matmul, attention)

The roof is peak FP32 on CUDA cores. Tensor cores (TF32 or FP16 inputs) can exceed it, and a reading above 100% is allowed and expected then. The output must still pass the fp32 tolerance, so check the verdict.

1. **Data reuse.** Stage tiles of A and B (or K and V) in shared memory, so each global load feeds many FMAs.
2. **Register blocking.** Have each thread compute a small tile of outputs (for example 8x8), so FMAs outnumber shared-memory loads. Look for a high `FFMA` count against `LDS` in SASS.
3. **Tensor cores.** `HMMA`, `UTCHMMA` or `QMMA` in SASS mean they are used. Triton's `tl.dot` and cuTile's `ct.mma` emit them for suitable dtypes; CUDA needs `wmma`/`mma.sync` intrinsics or inline PTX.
4. **Pipelining.** Load the next K-slab while computing the current one: `cp.async` or TMA, double buffering, Triton's `num_stages`.
5. **Attention specifically.** Fuse QKᵀ, softmax and PV with an online softmax, so the S×S score matrix never exists in memory (FlashAttention). Tile queries per block and stream K and V through shared memory.

## Reading SASS

Run `pgaio.py sass <run> <draft> --show LDG,STG,BAR` and check:

| You see | It means |
| --- | --- |
| `LDG.E` (no `.128`) in a streaming loop | scalar loads: an alignment or contiguity fact is missing |
| `LDG.E.128`, `STG.E.128` | 16-byte vector accesses |
| `LDGSTS`, `UBLKCP`, `UTMALDG` | asynchronous copies into shared memory (cp.async, bulk copy, TMA) |
| `BAR.SYNC` between every phase | block-wide synchronization; costly at small sizes |
| `SHFL.BFLY` / `SHFL.DOWN`, `REDUX` | warp-level reductions |
| `ATOM`, `RED` in the hot loop | global atomics; contention if executed per element |
| `LDL` / `STL`, `local > 0` | register spills or a local array |
| `HMMA`, `UTCHMMA`, `QMMA` | tensor cores |
| `MUFU.EX2` | the fast exp2 unit; `expf` lowers to it plus a multiply |

Where instructions sit relative to barriers matters as much as which ones exist. A prefetch the compiler sank below the barrier it was meant to overlap does nothing. `--show` prints them in program order with addresses.

## DSL notes

**CUDA C++.** Full control. nvcc already routes loads through `const T* __restrict__` to the read-only path (`LDG.E.*.CONSTANT`), so adding or removing `__ldg` changes nothing; check with `--show LDG` before spending an iteration on a load-path idea. Never form an exclusive scan as inclusive minus the element's own value (mixed magnitudes cancel); shift the inclusive prefix by one lane instead. Use `float4` loads on aligned data, `__shfl_xor_sync` for reductions, `__ldg` or `ld.global.nc` for read-only data, but note that ptxas may move `.nc` loads across `bar.sync`. Use inline PTX for cache hints (`createpolicy`, `L2::cache_hint`) and for bulk copies. Choose the launch shape in the manifest. Upper-case `// arena: KEY=N` lines become expression variables.

**Triton.** Pointers are marked 16-byte aligned by the helper. Alignment facts do not survive control flow: if `base` and `mask` are assigned in an if/else and the load comes after it, the merged value keeps only the weaker branch's facts and the load goes scalar. Call an inlined `@triton.jit` helper inside each branch instead. A row offset written as `row * ((cols // 4) * 4)` carries divisibility 4. Module-level constants used in a kernel must be `NAME = tl.constexpr(value)`. For other facts use `main(..., divisibility={"cols": 16})`; `tl.multiple_of` alone did not produce vector loads. `tl.range(..., num_stages=k)` pipelines a loop with async copies. `eviction_policy=` and `cache_modifier=` on loads and stores map to L2 hints. Read `kernel.ttgir` from `pgaio sass` to see layouts and vector widths before they become PTX. Register pressure decides occupancy: check `registers` in the inspect output.

**cuTile.** Arrays are declared contiguous and 16-byte aligned by the helper; add facts with `main(..., array_hints={"X": {"stride_divisible_by": [4, 1]}})`. TMA (`UTMALDG`) needs zero padding, so mask by index instead of padding with `NEG_INF`. A `latency=` hint can stop multi-buffering, so try with and without. `occupancy=` in `@ct.kernel` steers blocks per SM. There is no cache-policy control. Look things up with `pgaio.py api cutile <name>`.

**Warp.** SIMT kernels are one thread per `wp.tid()`. Use the tile API (`wp.tile_load`, `wp.tile_max`, `wp.tile_sum`) with a block dimension for cooperative work, and pass `aligned=True` on tile loads to get vector accesses. Tile kernels need dynamic shared memory: use `"shared_mem": "shared_memory"` in the manifest. Variable names cannot change type between branches. Read `kernel.warp_gen.cu` to see what a kernel became. There is no cache-policy control.

## Per-category starting points

- **reduce:** vectorized loads, a warp-shuffle reduction, and one atomic per block on a persistent grid. For small n, a single block with no atomics.
- **scan:** single-pass decoupled look-back; per-tile scans in registers with warp shuffles. For small n, one block.
- **softmax:** one pass with the row in registers, or an online softmax for long rows. For short rows, one warp per row.
- **matmul:** shared-memory tiling, then register blocking, then tensor cores and pipelining.
- **attention:** fused online-softmax attention (FlashAttention style), tiling queries and streaming K and V.

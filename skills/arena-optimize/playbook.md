# Optimization playbook

What to try, in roughly the order it pays off, keyed to what limits the kernel at a given shape. `pgaio.py eval` prints which roof each shape is held to (`memory`, `compute` or `launch`); start from the shapes furthest from theirs.

## 1. Latency-bound shapes (small problems)

The roof is the latency floor: an empty launch plus one cold DRAM round trip, about 3 to 3.7 µs on an RTX PRO 6000. A kernel at 10 µs on a 16 KB input is losing to fixed costs, not bandwidth. Reaching the floor is realistic for simple kernels (a reduce of 4096 elements does).

- **One launch.** A second kernel, or a memset before it, costs another launch floor plus a gap. Fold the work into one kernel, finishing with atomics or a last-block-done counter in scratch.
- **Right-size the grid.** Thousands of blocks for a few thousand elements is mostly scheduling. Use one block, or a few, for small inputs, choosing the grid with an expression on the size.
- **Avoid block-wide synchronization.** Each `__syncthreads` round on a tiny input is pure latency. Prefer warp-level work: one warp per row for short rows, and shuffles (`__shfl_xor_sync`, `tl.reduce` on a small block) over shared-memory trees.
- **Short dependency chains.** Unroll fixed small loops, and keep the critical path (load, reduce, store) short.
- **Dispatch by size, and across DSLs.** A manifest launch can carry `"when": "n < 1048576"` and its own `"source"`, so one candidate can use a CUDA kernel for small inputs and a Triton one for mid sizes. In the scan and softmax runs the mixed candidate beat every single-DSL one. Such candidates count as DSL `mixed`.
- **Mid sizes.** A shape around L2 size (a few MB to 16 MB) pays both the launch and the transfer. Score rule 2 (new runs) counts the launch plus the larger of the transfer and the rest of the round trip, and below L2 it counts only the read bytes, since the output stays in L2 inside the timed window. The best layernorm kernel scores about 70% at mid shapes under rule 2. In a rule 1 run the roof counts only the larger of the two, so around 60% can already be close to the best achievable.

## 2. Memory-bound shapes (large problems, low arithmetic intensity)

The roof is moving the ideal bytes once at peak DRAM bandwidth.

1. **Touch each byte once.** Multi-pass algorithms (max, then sum, then normalize) re-read their input. Keep the data in registers or shared memory between passes, or use online formulations: running max and rescaled sum for softmax, decoupled look-back for scan.
2. **Wide, coalesced accesses.** In SASS you want `LDG.E.128` / `STG.E.128`. Scalar `LDG.E` on fp32 means 4x the load instructions. Check alignment and contiguity; see the DSL notes below.
3. **Enough bytes in flight.** DRAM needs many outstanding loads per SM. Raise occupancy, issue several independent loads per thread before using them, or prefetch the next tile while computing the current one: persistent blocks, async copies, TMA.
4. **Cache policy.** `evict_first` on streamed input and streaming stores (`st.global.cs`) free L2 for what is reused. An `evict_last` output store mostly parks output in L2, to be written back after the timer stops; in the layernorm run that was most of its 4% gain (DRAM writes 237 to 226 MB). Score rule 2 charges parked output at DRAM speed at shapes larger than L2, so do not count on it there.
5. **Avoid atomics contention.** One global atomic per element serializes. Reduce within the warp and block first; one atomic per block is fine.

## 3. Compute-bound shapes (matmul, attention)

The roof is peak FP32 on CUDA cores. Tensor cores (TF32 or FP16 inputs) can exceed it, and a reading above 100% is allowed and expected then. The output must still pass the fp32 tolerance, so check the verdict.

The nominal FP32 roof assumes the boost clock. A power-limited board cannot hold it under a full FFMA load: on the RTX PRO 6000, cuBLAS SGEMM sits at about 50% of nominal, at 2.2 to 2.4 GHz and 520 to 580 W. Judge compute-bound candidates against the library reference (`pgaio.py reference <run> --lib cublas`, then "% of cublas" in every eval) and the "SOL at the measured clock" line, not against 100%.

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
- **layernorm:** read the row once into registers, take the mean, then the centred sum of squares from registers (or Welford), then normalize. Load gamma and beta together with x, not after the reductions. For short rows, one warp per row. See the layernorm section below.
- **matmul:** shared-memory tiling, then register blocking, then pipelining (and tensor cores unless the run forbids them). For FP32 on CUDA cores, see the next section.
- **attention:** fused online-softmax attention (FlashAttention style), tiling queries and streaming K and V.

## Layernorm and other row normalizations

From the layernorm-1 run on an RTX PRO 6000 (fp32). Under score rule 2 its best kernel scores 80.3% (mid shapes about 70%, large about 83%) against 57.9% for the naive baseline.

- Keep the row in registers and load gamma and beta with x, so the row pays one cold DRAM round trip. Every DSL then reached about 82% of DRAM at 16384 x 4096 and 4096 x 16384. Loading gamma and beta first took 64 x 1024 from 8.2 to 5.7 µs.
- At small row counts a 128-thread block per row beat a warp per row by 10% (64 rows, A/B); at 4096 x 768 a warp per row won (17 µs against 19.4). Dispatch on the shape.
- An `evict_last` output store gained about 4% in CUDA and Triton, mostly by leaving output in L2 past the timer (see section 2); rule 2 charges it at shapes larger than L2.
- Not worth trying again: evict-first `.cs` accesses, `L2::256B` prefetch, an evict_first input policy, a 0.5 evict_last fraction, block geometry, a merged Chan reduction for mean and variance, several rows per warp sharing gamma and beta (fewer warps in flight at 1024 rows), and softmax's persistent next-row prefetch (2.0 to 2.7% slower: register double buffering cuts bytes in flight).
- Triton: power-of-two tiles mask 25% of lanes at 768 columns; loading the row as three 256-wide chunks took 4096 x 768 from 18.9 to 17.0 µs.
- cuTile on tileiras 13.1: 255 registers x 256 threads leaves one block per SM, so persistent grids beat non-persistent ones. `allow_tma=False` gives scalar `LDG`.
- Warp allocates every shared tile statically (196 KB failed to launch; `check` now reports it). `tile_map` results store as scalar `STG`; assigning back into the shared x tile gives `STG.E.128`, which only helped at 768 columns.

## Matmul, FP32 on CUDA cores (sm_120)

From the matmul-fp32-vs-public run on an RTX PRO 6000 (best 47.4% of nominal SOL, about 95% of cuBLAS).

cuBLAS with TF32 off dispatches `cutlass_80_simt_sgemm_256x128_8x4` from 2048 to 8192. Its shape is a good first target in CUDA: a 128x256 block tile, BK 8, a 4-stage `cp.async` pipeline, 256 threads with 8x16 outputs each, one block per SM at about 230 registers. Profile the reference once (`profile <run> --reference cublas`) and compare stalls against it.

- Register blocking is most of the gain: 8x8 outputs per thread on a 128x128x8 tile with `LDS.128` on both operands took the naive 4.8% to 41.9% in one step. Count FFMA per `LDS.128` in the SASS: 16 at 8x8, about 21 at 8x16.
- A is stored transposed in shared memory so both operands load as float4. With 4-byte `cp.async`, map 8 lanes to one row (one 32-byte sector) and give k a stride of BM + 4 floats. One row per lane gave 476M shared bank conflicts and 33% instead of 45%. Transpose only the 128-row A tile; B goes in with 16-byte copies.
- Wave quantization: 128 tiles at 2048 and 512 at 4096 leave SMs idle on 188 SMs. Serial split-K fixes it, but order the grid split-major (every tile's split 0, then split 1). Tile-major order makes a tile's splits finish together and queue on the semaphore (2048 went from 0.307 to 0.360 ms); split-major with 4 splits took it to 0.275 ms. The semaphores need zeroed scratch: declare them `"zero": true` (or reset them in the kernel), or the kernel hangs under the declared policy.
- At about 95% of cuBLAS the remaining gap was short_scoreboard, mio and lg_throttle stalls (FFMA waiting on LDS, the 4-byte `LDGSTS`) and the split-K epilogue. The last idea tried, double-buffered register fragments with the barrier at the last k step and a single thread fencing after it, matched the best within noise.
- Triton: `tl.dot(..., input_precision="ieee")` lowers to FFMA only. cuBLAS's configuration (128x256x8, 4 stages, 8 warps) reached 43.7%.
- cuTile: `ct.mma` on f32 tiles lowers to FMUL+FADD, so it caps near half of peak (6%). Write the outer product yourself with `ct.extract` and a broadcast to get FFMA; at 64x128 it reads `LDS.128` and reached 15%. Mark the DSL with `skip-dsl` once the SASS confirms this, rather than spending evals.
- Warp: `tile_matmul` is cuBLASDx, a library, so it is out of bounds. A hand-blocked SIMT kernel reached 32% at 2 blocks per SM. Warp does not unroll `range(32)`, so unroll by hand.
- TMA kernels cannot be written as manifests yet: a tensor map needs host-side setup (`cuTensorMapEncodeTiled`) that a manifest cannot do. cuTile and Triton build their own descriptors, so their TMA paths still work. A possible manifest arg type is sketched in `docs/tma-manifest-args.md`.

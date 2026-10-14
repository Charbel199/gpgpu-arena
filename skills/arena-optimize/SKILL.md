---
name: arena-optimize
description: Profile and benchmark guided AI optimization (PGAIO) of one GPGPU Arena kernel. Runs a measured loop of hypothesis, candidate, evaluation and diagnosis against speed of light, with correctness gates and an iteration budget, in CUDA, Triton, cuTile or Warp. Use when asked to optimize, speed up or tune an arena kernel, run a PGAIO iteration, or measure iterations-to-SOL for a DSL.
---

# PGAIO: optimizing an arena kernel

You are the optimizer. `tools/pgaio.py` is the environment: it compiles and measures each candidate the same way, gates correctness on held-out inputs, scores against the roofline, diffs SASS, and logs every iteration. Read the arena-cli and arena-inspect skills for how the measurements work.

Run everything from the repo root. By default `pgaio.py` drives arena through Docker (`docker compose run --rm -T cli`), which has all four DSLs. For a native build pass `--arena build/arena` (any build dir) to `init` or set `ARENA_CMD`.

## 1. Start a run

Pick a slot: an existing kernel whose descriptor fixes the ABI. The run starts from the slot's own source, or from any file of the same type passed with `--start <file>`. Slots that work well, with their starting SOL on an RTX PRO 6000 at the PGAIO sizes:

| Category | DSL | Slot | Start | Launch shape |
| --- | --- | --- | --- | --- |
| reduce | CUDA | `reduce_grid_stride` | ~75% | `32 * SMs` blocks; `// arena: block=N` picks 64 to 1024 |
| reduce | Triton | `triton_reduce_grid_stride` | ~76% | grid from the source's `NUM_BLOCKS`, tile `BLOCK_SIZE`, threads `num_warps` |
| reduce | cuTile | `cutile_reduce` | | one tile per block, grid `n / TILE_SIZE` |
| reduce | Warp | `warp_reduce` | | one thread per element, block 256; tight, note it in the report |
| scan | CUDA | `scan_multiblock` | ~0.2% | `// arena: THREADS=256 GRID_PER_SM=4 SHMEM=0`; a 4th argument is a zeroed workspace of at least 64 + 16 * ceil(n / 32) bytes |
| softmax | CUDA | `softmax_naive` | ~72% | one block per row; `// arena: block=N` picks 64 to 1024; dynamic shared memory is block * 4 bytes |
| matmul | CUDA | `matmul_tiled` | ~6.5% of FP32 peak | `// arena: BLOCK_M=16 BLOCK_N=16 THREADS=256 SHMEM=0`; grid covers C in BLOCK_M x BLOCK_N tiles |
| matmul | Triton | `triton_matmul` | | grid and tiles from the source's constants |
| matmul | cuTile | `cutile_matmul` | | tiles from `BLOCK_M`/`BLOCK_N`/`BLOCK_K`; computes in fp16 |

Avoid `scan_naive` (one block, capped at a fraction of a percent) and `matmul_naive` (fixed 16x16 thread blocks). Upper-case `// arena: KEY=N` lines in a CUDA source are read into the slot's launch constants, so for slots that use them the candidate chooses its own launch shape. Matmul SOL is measured against the FP32 CUDA-core peak, so a tensor-core kernel can read above 100%.

The slots start at different SOL levels, and the report shows each run's baseline next to its curve, so compare runs from the same starting point.

```bash
python3 tools/pgaio.py init --kernel reduce_grid_stride \
    --budget 20 --target 0.9 --mode isolated --docs off --model <your model id>
```

`init` measures the starting source as iteration 0, prints its observation and any learnings from earlier runs, and creates `pgaio/runs/<run>/` with `work/` for your drafts. Checks the starting source itself fails are dropped from the gate and listed.

Modes:

- `isolated`: study only this slot. Do not read, inspect or time other kernels in the repo, and do not use what you know of them.
- `cross-dsl`: you may run `--inspect`, `--sass` and timings on any other kernel, in any DSL, and use what they show. Say which kernel inspired an idea in its hypothesis.

Keep the mode honest, since the report compares them.

### CUDA documentation tools

Before `init`, check whether a CUDA documentation tool is connected: an MCP tool whose name mentions Nsight Copilot, CUDA docs or similar. Then:

- If one is connected, pass `--docs on` and use it during the run to check API signatures (cuTile and Warp change between versions), architecture limits for the GPU in `environment.arch`, and unfamiliar PTX or SASS instructions. Looking up a fact before submitting is cheaper than a failed compile, which costs an iteration.
- If none is connected, tell the user once that a CUDA documentation MCP server would help (NVIDIA's Nsight Copilot CUDA server, for those with access) and that adding one takes effect in a new session. Do not add it yourself. Then continue with `--docs off`.

Pass `--docs` truthfully: the report compares runs with and without documentation access.

## 2. Learn the slot before changing it

1. Read `descriptor_file` (printed by `init`): launch config, arguments, block size rules.
2. Read the starting source and `pgaio/runs/<run>/obs/000.sass`.
3. Note the entry point and parameter list. A candidate must keep both.

## 3. Iterate

For each iteration:

1. Pick one hypothesis from the last observation: its diagnosis lines, counters and SASS. Prefer the change with the largest expected effect. State it as cause and effect: "scalar 4 B loads and one atomic per thread; a block-level shuffle reduction with float4 loads should cut atomics by 256x and load instructions by 4x".
2. Write the candidate under `pgaio/runs/<run>/work/` as a complete file of the slot's type.
3. Evaluate it:

   ```bash
   python3 tools/pgaio.py eval <run> pgaio/runs/<run>/work/cand.cu --hypothesis "<the hypothesis>"
   ```

4. Read the observation. It has the verdict, correctness, time against baseline and best, SOL, counters, resources, SASS counts and their change against the best so far, and a heuristic diagnosis. The full JSON is in `obs/NNN.json`, the SASS in `obs/NNN.sass`.
5. Decide what the result tells you about the hypothesis before choosing the next one. A slower or equal result is information: say what it rules out.

Verdicts: `ok`, `incorrect` (a gated check failed: size, distribution, seed and error are listed), `compile_error`, `runtime_error`, `rejected_guard`. Every eval spends one iteration, whatever the verdict, so compile-check your reasoning before submitting. `pgaio.py status <run>` shows the history.

Stop when the target is reached, the budget is spent, or three iterations in a row fail to improve while the diagnosis says the kernel is within 15% of the roof.

## 4. Rules

These keep the score meaningful. Breaking one invalidates the run.

- Do not modify anything under `src/`, `include/`, `kernels/`, `tests/`, `tools/`, `CMakeLists.txt` or the Docker files during a run. `eval` hashes them and rejects the iteration if anything changed.
- Compute the result from the inputs, every launch. No special cases for particular sizes, seeds, distributions or values, no caching across launches, no reading the expected answer.
- Keep accuracy within the gate on every distribution, including `adversarial`. Changing the accumulation order is fine; dropping elements or precision to win time is not.
- Use the slot's ABI as it is. If an idea needs a different launch shape or arguments, record it in the final summary as a follow-up kernel (see arena-add-kernel) instead of working around the descriptor.
- One candidate file per iteration, evaluated through `pgaio.py eval`. Do not time candidates with arena directly to pick the best before submitting: every measured attempt counts.

Float atomics make the summation order nondeterministic, so an atomic-heavy kernel can land just inside or just outside tolerance on a given check from one run to the next. If a check fails by a hair on a change that did not touch arithmetic, say so in the next hypothesis.

## 5. Finish

1. `python3 tools/pgaio.py status <run>` and `python3 tools/pgaio.py best <run>`.
2. Record one to three insights that would help a future run, with their evidence:

   ```bash
   python3 tools/pgaio.py learn <run> --insight "Triton reduce: NUM_BLOCKS near 4x SMs with BLOCK_SIZE 4096 beat larger grids" \
       --evidence "iter 6: 81% -> 88% SOL"
   ```

   Insights are about the hardware, the DSL or the problem, not about this run's bookkeeping.
3. `python3 tools/pgaio.py report` prints iterations-to-50/70/80/90% SOL per run and per DSL and mode; `--csv curves.csv` writes the best-so-far curves.
4. Summarize for the user: start and final SOL, the iterations that moved it and why, what did not work, and whether the best candidate should be promoted into a permanent kernel (arena-add-kernel). Promotion changes `kernels/`, so ask first.

---
name: arena-optimize
description: Profile and benchmark guided AI optimization (PGAIO) of a GPGPU Arena problem such as softmax, layernorm, scan, reduce, matmul or attention, for small sizes, large sizes or both. Write as many candidate kernels as needed in CUDA, Triton, cuTile or Warp, compare their SASS for free, measure the promising ones against speed of light with correctness gates, profile them with Nsight Compute, and borrow ideas across DSLs. Runs until the target, a plateau or the user stops it. Use when asked to optimize or speed up an arena kernel or category, run PGAIO, or measure iterations-to-SOL.
---

# PGAIO: optimizing an arena problem

You are the optimizer. `tools/pgaio.py` is the environment: it compiles and measures every candidate the same way, gates correctness on held-out inputs, scores against the roofline at every size, diffs SASS, logs every iteration, and draws the progression. Read the arena-cli and arena-inspect skills for how the measurements work, [manifests.md](manifests.md) for the candidate format, and [playbook.md](playbook.md) for what to try.

## 0. Set the goal

A request only needs to name the problem; everything else has a default. Turn it into run settings before anything else, and say back what you chose:

| The user says | Setting |
| --- | --- |
| the problem ("softmax", "a prefix sum", "matmul") | `--category` (see below) |
| "small inputs", "latency", "decode", "per-token" | `--sizes small` |
| "large", "throughput", "bandwidth", "big batches" | `--sizes large` |
| nothing about size, or "all sizes" | `--sizes all` (default) |
| specific shapes ("4096 x 4096") | `--train rows=4096,cols=4096`, repeatable |
| a language ("in Triton"), or nothing | `--dsls triton`; default every DSL |
| an example to start from | a kernel in arena: `--baseline <name>`; a file: copy it into `work/` as your first draft |
| "do not look at the existing kernels" | `--mode isolated`; default `cross-dsl` |
| a target ("80% of SOL") | `--target 0.8`; default 1.0, as close to the roof as it gets |
| an iteration or time budget | `--budget N`; default none |

Without a budget the run ends at the target, on a plateau (section 2), or when the user stops you. Every eval updates the journal, so stopping at any point leaves a complete record.

If the problem is not one of the categories, say so and stop: a new problem needs a category with its reference, checks and byte counts first (arena-add-kernel), since that is what makes the score mean anything.

A run scores each candidate at every timed shape and combines them with a geometric mean, so a kernel that is fast at large sizes but slow at small ones cannot hide it. At small shapes the roof is the latency floor: an empty launch plus one cold DRAM read and one write (about 4 µs on an RTX PRO 6000), not bandwidth. Read the per-shape table in every observation, and optimize the shapes furthest from their roof first.

Run everything from the repo root. By default `pgaio.py` drives arena through Docker (`docker compose run --rm -T cli`), which has all four DSLs. For a native build pass `--arena build/arena` (any build dir) to `init` or set `ARENA_CMD`.

## 1. Start a run

A run targets a problem: a category, its sizes, and its correctness checks. Candidates can be in any allowed DSL, in any order.

```bash
python3 tools/pgaio.py init --category softmax --sizes all --mode cross-dsl \
    --scratch-policy declared --docs off --model <your model id>
```

Categories: `reduce`, `scan`, `softmax`, `layernorm` (`[rows, cols]` fp32 forward with gamma and beta, biased variance, eps fixed at 1e-5 inside the kernel since manifests have no float arguments), `matmul` and `attention` (`[BH, S, D]` fp32, non-causal, D = 64 in the timed and checked shapes).

`init` prints where the run lives and two files that update after every eval: `journal.md` (each iteration's hypothesis, result and code diff against the previous version in that DSL) and `progress.html` (the progression chart plus every iteration's code). The user follows the run through them, so keep hypotheses clear.

`init` measures the category's plainest kernel as iteration 0 (softmax: `softmax_naive`; layernorm: `layernorm_naive`; reduce: `reduce_baseline`; scan: `scan_multiblock`; matmul: `matmul_naive`; attention: `attention_naive`; override with `--baseline`), prints its observation and any learnings from earlier runs, and creates `pgaio/runs/<run>/` with `work/` for your drafts. Checks the baseline itself fails are dropped from the gate and listed. `--dsls cuda,triton` restricts the languages; `--budget` caps the total and `--budget-per-dsl` each language.

Options worth knowing:

- `--mode isolated`: do not read, inspect or time the existing kernels in `kernels/`, and do not use what you know of them. Your own candidates, in every DSL, are fair game. Reading the harness is fine: the DSL helpers (`kernels/triton_base.py`, `cutile_base.py`, `warp_base.py`), `include/` and `src/`.
- `--mode cross-dsl`: you may also inspect and time any existing kernel and say which one inspired an idea.
- `--docs on|off`: whether a CUDA documentation tool is available (section 5). Record it truthfully.
- `--scratch-policy declared`: use it for new runs. Scratch entries marked `"zero": true` are zeroed inside the timed op, so the candidate pays for the memset; the rest are filled with `0xFF`, so a kernel that quietly depends on zeros fails. The default, `legacy`, zeroes all scratch for free outside timing and stays only so old runs remain comparable. See [manifests.md](manifests.md).
- `--forbid-tensor-cores`: when the user asks for a true FP32 or CUDA-core comparison. A candidate whose SASS has tensor instructions is `rejected_guard`.
- `--lock-clocks <MHz|auto>`: arena locks the SM clock for each measurement, which gives steadier rankings; absolute numbers are lower than at boost. It needs root; through Docker, `init` adds `--cap-add SYS_ADMIN` to the run's arena command for this. If the lock fails, arena warns, runs unlocked and the observation records it.
- `--counters basic|full`: `full` (the default) adds issue active, pipe use, eligible warps, stall reasons, shared wavefronts and bank conflicts, and L2 sectors to every eval. It costs replay passes, so evals are slower; timing is not affected.

## 2. The loop

For matmul, run `python3 tools/pgaio.py reference <run> --lib cublas` once at the start. It times cuBLAS SGEMM (FP32, no TF32) with eval's exact settings and records which cuBLAS kernel ran at each size, so eval, status and report show every candidate as "% of cublas" too. That is the realistic ceiling: on a power-limited board the nominal FP32 roof is out of reach (section 3 of [playbook.md](playbook.md)). References never become the best and do not count as iterations.

For each idea:

1. Write a candidate in `pgaio/runs/<run>/work/`: a kernel source plus a manifest saying how to launch it. The baseline's source is copied to `candidates/000_baseline/` to start from. Copy the matching example from [manifests.md](manifests.md) for a new DSL rather than guessing its argument format.
2. Look at it for free:

   ```bash
   python3 tools/pgaio.py sass <run> pgaio/runs/<run>/work/v3.json
   ```

   This first runs `check` (below), then compiles the draft and prints registers, spills, the SASS instruction mix and its most frequent opcodes, the difference against the best so far, and a `diff` command for the two SASS listings. It saves `kernel.sass` and whatever else the compiler produced into `inspections/NNN_<draft>/`: `kernel.ptx` (CUDA, Triton), `kernel.ttgir` (Triton's IR, where layouts, vector widths and pipelining are decided) and `kernel.warp_gen.cu` (the CUDA C++ Warp generated). It never runs or times the kernel, so it costs nothing. Use it as often as you like: to check that a change compiled the way you meant (128-bit loads, warp shuffles instead of shared memory, no spills), and to compare how the same idea compiles in different DSLs.

   `--show LDG,STG,BAR` lists the SASS lines starting with those mnemonics in program order, with addresses: where loads sit relative to barriers, whether a prefetch was hoisted or sunk, without reading the whole listing.
   Before measuring, confirm in this output that the change you meant actually happened. In the PGAIO runs so far, iterations were wasted on kernels whose SASS already showed the idea had not taken effect: `__ldg` removed but the load still `.CONSTANT`, a Triton load still scalar because an `if` merged its alignment, a tile spilling hundreds of `LDL`/`STL`. A `SPILLS:` line means shrink the tile or raise `num_warps` first.

   `python3 tools/pgaio.py check <run> <manifest>` on its own runs `arena --plan` over every timed and checked shape and prints each launch's grid, block, shared memory, scratch sizes and arguments, or the first expression error. It is free. Use it before the first eval of every new manifest: an expression error, or a `when` that leaves some checked size without a launch, otherwise costs a whole iteration.
3. Measure it when you expect it to help:

   ```bash
   python3 tools/pgaio.py eval <run> pgaio/runs/<run>/work/v3.json --hypothesis "<cause and effect>"
   ```

   This snapshots the manifest and source, checks correctness on held-out shapes, distributions and seeds, times it at a size larger than L2 with a cold cache, and prints the verdict, SOL with a 95% confidence interval, the spread at each shape, counters, SASS and a diagnosis. For compute-bound shapes the diagnosis leads with FFMA share, FMAs per shared load, registers and occupancy, issue active, top stalls and pipe use. Every eval spends one iteration of that DSL's budget, whatever the verdict. eval is the judge: only its score counts.
4. Profile it when the eval did not gain, or the largest shape is compute-bound:

   ```bash
   python3 tools/pgaio.py profile <run> 7                      # an evaluated iteration, largest shape
   python3 tools/pgaio.py profile <run> --reference cublas     # once per run, for matmul
   ```

   This runs Nsight Compute on arena for an iteration that passed eval and saves `profiles/<NNN>/summary.json` (plus csv exports and `report.ncu-rep`): stall reasons, pipe use, bank conflicts, the instructions carrying the most stalls with their SASS offsets, the hot loop's opcode mix, and a diff against the best or the reference. Form the next hypothesis from the stall diff and those instructions. ncu durations are at a locked base clock with replay, labelled as such, and never a score. Replay restores memory, so self-resetting state (split-K semaphores) profiles fine. It is free.
5. Compare any two measured iterations, free: `python3 tools/pgaio.py compare <run> 2 7`, or `compare <run> 7 ref:cublas`. It shows both CIs, stalls, pipes, clocks and, when both have profiles, the ncu stall diff.
6. Decide what the result says about the hypothesis before the next one. A slower result rules something out; say what.

When small and large shapes want different kernels, write both and dispatch with `"when"` in the manifest ([manifests.md](manifests.md)), rather than settling for one kernel that is mediocre everywhere. A launch can also name its own `"source"`, so one candidate can combine several Triton, cuTile or Warp kernels (one kernel per file), or send small sizes to a CUDA kernel and mid sizes to a Triton one. A candidate mixing DSLs counts as DSL `mixed`, with its own budget. In the scan and softmax runs the mixed candidate beat every single-DSL one. [playbook.md](playbook.md) lists what usually works for launch-bound, memory-bound and compute-bound shapes.

State hypotheses as cause and effect, citing the evidence: "Triton's SASS uses 4 shuffles and no shared memory per reduction where my CUDA kernel uses 10 shared ops and 9 syncs; a warp-shuffle reduction in CUDA should cut the syncs and gain a few percent."

Look up a DSL's API in the installed version, free: `python3 tools/pgaio.py api cutile load`, `api warp tile_load`, `api triton max`; without a name it lists what the module exports. Prefer this to guessing, since cuTile and Warp change between releases.

Moving ideas between DSLs is the point. When one DSL produces better SASS for the same algorithm, look at what it emitted and write the same thing in the other. When a DSL's abstraction blocks an idea, say so and move on.

Verdicts: `ok`, `incorrect` (a gated check failed; size, distribution, seed and error are listed), `compile_error`, `runtime_error` (includes a manifest whose arguments do not match the kernel; the message says which), `rejected_guard`. `pgaio.py status <run>` shows the history and the budget left per DSL.

Stop when the target is reached, the budget (if any) is spent, or the run reaches a plateau: `eval` and `status` count the evals since the best last made a real gain, and once that reaches 10 with every allowed DSL given at least 3 evals (or skipped, below), they say you may stop. In runs started now (plateau rule 2), an eval is a gain only if it beats the best by more than 0.5 points and by more than the two CIs' half-widths combined, or if an `ab` over every timed shape (no `--shape`) shows it faster than the best; an `ab` limited with `--shape` is for diagnosis and does not count. Older runs keep rule 1 (0.5 points). Before stopping on a plateau, look at the per-shape table: if one size range is far from its roof, an idea aimed at it (a size-dispatched kernel) is worth trying first. The user can stop the run at any time.

When a DSL is a dead end, mark it rather than spending evals to reach the minimum of 3:

```bash
python3 tools/pgaio.py skip-dsl <run> cutile --reason "ct.mma on f32 tiles lowers to FMUL+FADD, no FFMA" \
    --evidence "sass inspections 12, 14"
```

The evidence should be something you saw, such as a SASS inspection or earlier evals. A skipped DSL still accepts evals.

## 3. Rules

These keep the score meaningful, and the harness enforces most of them.

- Do not modify anything under `src/`, `include/`, `kernels/`, `tests/`, `tools/`, `CMakeLists.txt` or the Docker files during a run. `eval` hashes them and rejects the iteration if anything changed. `pgaio.py rehash` exists for maintainers who deliberately improve the harness mid-run, and records why in the journal; never use it to get your own change through. If a candidate needs a harness change, stop and say so.
- Candidates are kernels plus manifests. A manifest cannot run code, so verification, input data and the byte counts SOL is computed from are out of reach by design.
- Compute the full result from the inputs on every launch. Verification runs on inputs regenerated with a seed the kernel never saw, after timing, and checks the whole output, so caching, precomputing or skipping work fails.
- Keep accuracy within the gate on every distribution, including `adversarial`.
- No libraries that do the work for you: no cuBLAS, cuBLASLt, cuBLASDx or MathDx (including DSL calls that wrap them, such as Warp's `tile_matmul`), cuDNN, CUB, Thrust, CUTLASS or PyTorch ops. A compiler-generated call into one of these counts as library use. Language primitives (warp shuffles, `tl.max`, `ct.sum`, `tl.dot`) are fine.
- Do not commit anything. Runs live in pgaio/runs/, which git ignores; the learnings file that `pgaio.py learn` appends to is the only thing a run adds to the repo, and the user decides when to commit it.
- Measure through `pgaio.py eval` only. `sass`, `check`, `profile` and `ab` on evaluated iterations are free and allowed. Choosing what to submit from timings taken outside pgaio (arena run directly, ncu durations, your own scripts) is not.
- Scratch: under `legacy`, scratch is zeroed for free outside timing. Do not build on that in a `declared` run; mark every buffer the kernel needs zeroed with `"zero": true`. A kernel that spin-waits on an undeclared buffer (split-K semaphores, look-back flags) hangs on the `0xFF` fill instead of failing.

The timed size is chosen to be several times the L2 (softmax: 16384 x 4096, 512 MB in and out against 128 MB of L2). Cache-policy hints such as an evict_last store are allowed, but they cannot hide most of the output in L2 at this size; if the diagnosis says DRAM moved well under the ideal bytes, say so in the summary rather than presenting the SOL as pure bandwidth.

Every eval prints a 95% CI for the score and the spread at each shape. A difference inside the CIs is noise, not a result. When a candidate lands within the CI of the best, time the two directly instead of reading the numbers:

```bash
python3 tools/pgaio.py ab <run> 15 16 [--shape K=4096,M=4096,N=4096] [--rounds 6]
```

It alternates the two in rounds and reports the ratio with a CI. It is free, but only takes iterations that passed eval; drafts cannot be timed outside eval. Float atomics make summation order nondeterministic, so an atomic-heavy kernel can land on either side of a tolerance by a hair.

Each eval also samples SM clock and power during the timed runs, and for compute-bound shapes prints the SOL at the measured clock next to the nominal one. The nominal SOL stays the score. Power and input data move the clock: on the RTX PRO 6000, cuBLAS SGEMM runs at 2.2 to 2.4 GHz and 520 to 580 W, and was 6% faster at 8192 on uniform [0,1) inputs than on randn.

## 4. Finish

1. `python3 tools/pgaio.py status <run>` and `python3 tools/pgaio.py best <run>`. When the run has a reference, `status` shows it and `report` adds a "vs reference" column; if the best has not been profiled, `profile` it so the summary can say what still limits it.
2. Record one to three insights for future runs, with evidence:

   ```bash
   python3 tools/pgaio.py learn <run> --insight "cuTile softmax: TILE 4096 with one row per block beat 1024 by 6%" \
       --evidence "iter 9 vs 6"
   ```

   `learn` refuses when the learnings file is missing but git tracks it: restore it with `git checkout pgaio/learnings.jsonl` rather than passing `--create`.

3. `python3 tools/pgaio.py report --html pgaio/runs/<run>/progress.html` draws the progression: every iteration coloured by DSL, the best-so-far line, hypotheses and SASS changes. The markdown tables give iterations-to-50/70/80/90% SOL overall and per DSL.
4. Summarize for the user: start and final score, the per-size table of the best candidate (with the reference column when there is one), which iterations moved it and why, which DSL ideas transferred, what did not work, and whether the best candidate should be promoted to `kernels/` (arena-add-kernel). Promotion changes `kernels/`, so ask first. To compare the candidates by hand, point the user at the GUI in Docker, which has every DSL: `docker compose run --rm gui --gui --open pgaio/runs/<run>` (run `xhost +local:docker` first). A native build cannot run Triton, cuTile or Warp candidates.

## 5. CUDA documentation tools

Before `init`, check whether a CUDA documentation tool is connected: an MCP tool whose name mentions Nsight Copilot, CUDA docs or similar.

- If one is connected, pass `--docs on` and use it to check API signatures (cuTile and Warp change between versions), architecture limits for the GPU in `environment.arch`, and unfamiliar PTX or SASS instructions. A lookup is cheaper than a failed compile.
- If none is connected, tell the user once that a CUDA documentation MCP server would help (NVIDIA's Nsight Copilot CUDA server, for those with access) and that adding one takes effect in a new session. Do not add it yourself. Then continue with `--docs off`.

## Single-slot runs

`init --kernel <name>` instead optimizes one existing kernel's source in place: candidates are plain source files of that kernel's type, run with `--source-override` against its compiled descriptor, and `sass`/`eval` take the source path. Use it to tune a kernel without changing its launch shape. Everything else above applies.

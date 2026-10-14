---
name: arena-optimize
description: Profile and benchmark guided AI optimization (PGAIO) of a GPGPU Arena problem such as softmax, scan, reduce or matmul. Write as many candidate kernels as needed in CUDA, Triton, cuTile or Warp, compare their SASS for free, measure the promising ones against speed of light with correctness gates and a per-DSL budget, and borrow ideas across DSLs. Use when asked to optimize or speed up an arena kernel or category, run PGAIO, or measure iterations-to-SOL.
---

# PGAIO: optimizing an arena problem

You are the optimizer. `tools/pgaio.py` is the environment: it compiles and measures every candidate the same way, gates correctness on held-out inputs, scores against the roofline, diffs SASS, logs every iteration, and draws the progression. Read the arena-cli and arena-inspect skills for how the measurements work, and [manifests.md](manifests.md) for the candidate format.

Run everything from the repo root. By default `pgaio.py` drives arena through Docker (`docker compose run --rm -T cli`), which has all four DSLs. For a native build pass `--arena build/arena` (any build dir) to `init` or set `ARENA_CMD`.

## 1. Start a run

A run targets a problem: a category, its sizes, and its correctness checks. Candidates can be in any allowed DSL, in any order.

```bash
python3 tools/pgaio.py init --category softmax --budget-per-dsl 10 --target 0.9 \
    --mode isolated --docs off --model <your model id>
```

`init` measures the category's plainest kernel as iteration 0 (softmax: `softmax_naive`; reduce: `reduce_baseline`; scan: `scan_multiblock`; matmul: `matmul_naive`; override with `--baseline`), prints its observation and any learnings from earlier runs, and creates `pgaio/runs/<run>/` with `work/` for your drafts. Checks the baseline itself fails are dropped from the gate and listed. `--dsls cuda,triton` restricts the languages; `--budget` caps the total.

Options worth knowing:

- `--mode isolated`: do not read, inspect or time the existing kernels in `kernels/`, and do not use what you know of them. Your own candidates, in every DSL, are fair game.
- `--mode cross-dsl`: you may also inspect and time any existing kernel and say which one inspired an idea.
- `--docs on|off`: whether a CUDA documentation tool is available (section 5). Record it truthfully.

## 2. The loop

For each idea:

1. **Write a candidate** in `pgaio/runs/<run>/work/`: a kernel source plus a manifest saying how to launch it. The baseline's source is copied to `candidates/000_baseline/` to start from. Copy the matching example from [manifests.md](manifests.md) for a new DSL rather than guessing its argument format.
2. **Look at it for free**:

   ```bash
   python3 tools/pgaio.py sass <run> pgaio/runs/<run>/work/v3.json
   ```

   This compiles the draft and prints registers, spills and the SASS instruction mix, the difference against the best so far, and a `diff` command for the two SASS listings. It never runs or times the kernel, so it costs nothing. Use it as often as you like: to check that a change compiled the way you meant (128-bit loads, warp shuffles instead of shared memory, no spills), and to compare how the same idea compiles in different DSLs.
3. **Measure it** when you expect it to help:

   ```bash
   python3 tools/pgaio.py eval <run> pgaio/runs/<run>/work/v3.json --hypothesis "<cause and effect>"
   ```

   This snapshots the manifest and source, checks correctness on held-out shapes, distributions and seeds, times it at a size larger than L2 with a cold cache, and prints the verdict, SOL, counters, SASS and a heuristic diagnosis. Every eval spends one iteration of that DSL's budget, whatever the verdict.
4. **Compare** any two measured iterations, free: `python3 tools/pgaio.py compare <run> 2 7`.
5. **Decide** what the result says about the hypothesis before the next one. A slower result rules something out; say what.

State hypotheses as cause and effect, citing the evidence: "Triton's SASS uses 4 shuffles and no shared memory per reduction where my CUDA kernel uses 10 shared ops and 9 syncs; a warp-shuffle reduction in CUDA should cut the syncs and gain a few percent."

Moving ideas between DSLs is the point. When one DSL produces better SASS for the same algorithm, look at what it emitted and write the same thing in the other. When a DSL's abstraction blocks an idea, say so and move on.

Verdicts: `ok`, `incorrect` (a gated check failed; size, distribution, seed and error are listed), `compile_error`, `runtime_error` (includes a manifest whose arguments do not match the kernel; the message says which), `rejected_guard`. `pgaio.py status <run>` shows the history and the budget left per DSL.

Stop when the target is reached, the budget is spent, or several iterations in a row fail to improve while the diagnosis says you are within 15% of the roof.

## 3. Rules

These keep the score meaningful, and the harness enforces most of them.

- Do not modify anything under `src/`, `include/`, `kernels/`, `tests/`, `tools/`, `CMakeLists.txt` or the Docker files during a run. `eval` hashes them and rejects the iteration if anything changed.
- Candidates are kernels plus manifests. A manifest cannot run code, so verification, input data and the byte counts SOL is computed from are out of reach by design.
- Compute the full result from the inputs on every launch. Verification runs on inputs regenerated with a seed the kernel never saw, after timing, and checks the whole output, so caching, precomputing or skipping work fails.
- Keep accuracy within the gate on every distribution, including `adversarial`.
- No libraries that do the work for you: no cuDNN, CUB, Thrust, CUTLASS, cuBLAS or PyTorch ops. Language primitives (warp shuffles, `tl.max`, `ct.sum`) are fine.
- Measure through `pgaio.py eval` only. Do not time candidates with arena directly to choose what to submit; `sass` is the free tool, and it shows no timing.

Timing noise between runs is a few percent. Do not read a 2% change as a win or a loss without a SASS or counter difference to explain it. Float atomics make summation order nondeterministic, so an atomic-heavy kernel can land on either side of a tolerance by a hair.

## 4. Finish

1. `python3 tools/pgaio.py status <run>` and `python3 tools/pgaio.py best <run>`.
2. Record one to three insights for future runs, with evidence:

   ```bash
   python3 tools/pgaio.py learn <run> --insight "cuTile softmax: TILE 4096 with one row per block beat 1024 by 6%" \
       --evidence "iter 9 vs 6"
   ```

3. `python3 tools/pgaio.py report --html pgaio/runs/<run>/progress.html` draws the progression: every iteration coloured by DSL, the best-so-far line, hypotheses and SASS changes. The markdown tables give iterations-to-50/70/80/90% SOL overall and per DSL.
4. Summarize for the user: start and final SOL, which iterations moved it and why, which DSL ideas transferred, what did not work, and whether the best candidate should be promoted to `kernels/` (arena-add-kernel). Promotion changes `kernels/`, so ask first.

## 5. CUDA documentation tools

Before `init`, check whether a CUDA documentation tool is connected: an MCP tool whose name mentions Nsight Copilot, CUDA docs or similar.

- If one is connected, pass `--docs on` and use it to check API signatures (cuTile and Warp change between versions), architecture limits for the GPU in `environment.arch`, and unfamiliar PTX or SASS instructions. A lookup is cheaper than a failed compile.
- If none is connected, tell the user once that a CUDA documentation MCP server would help (NVIDIA's Nsight Copilot CUDA server, for those with access) and that adding one takes effect in a new session. Do not add it yourself. Then continue with `--docs off`.

## Single-slot runs

`init --kernel <name>` instead optimizes one existing kernel's source in place: candidates are plain source files of that kernel's type, run with `--source-override` against its compiled descriptor, and `sass`/`eval` take the source path. Use it to tune a kernel without changing its launch shape. Everything else above applies.

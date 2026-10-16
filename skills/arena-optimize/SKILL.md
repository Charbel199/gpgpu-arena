---
name: arena-optimize
description: Profile and benchmark guided AI optimization (PGAIO) of a GPGPU Arena problem such as softmax, scan, reduce, matmul or attention, for small sizes, large sizes or both. Write as many candidate kernels as needed in CUDA, Triton, cuTile or Warp, compare their SASS for free, measure the promising ones against speed of light with correctness gates and a per-DSL budget, and borrow ideas across DSLs. Use when asked to optimize or speed up an arena kernel or category, run PGAIO, or measure iterations-to-SOL.
---

# PGAIO: optimizing an arena problem

You are the optimizer. `tools/pgaio.py` is the environment: it compiles and measures every candidate the same way, gates correctness on held-out inputs, scores against the roofline at every size, diffs SASS, logs every iteration, and draws the progression. Read the arena-cli and arena-inspect skills for how the measurements work, [manifests.md](manifests.md) for the candidate format, and [playbook.md](playbook.md) for what to try.

## 0. Set the goal

Turn the request into run settings before anything else, and say back what you chose:

| The user says | Setting |
| --- | --- |
| "small inputs", "latency", "decode", "per-token" | `--sizes small` |
| "large", "throughput", "bandwidth", "big batches" | `--sizes large` |
| nothing about size, or "all sizes" | `--sizes all` (default) |
| specific shapes ("4096 x 4096") | `--train rows=4096,cols=4096`, repeatable |
| a language ("in Triton") | `--dsls triton` |
| "as fast as possible", no target | `--target 0.9` |
| a time or iteration budget | `--budget-per-dsl` / `--budget` |
| "look at the existing kernels for ideas" | `--mode cross-dsl`, otherwise `isolated` |

A run scores each candidate at every timed shape and combines them with a geometric mean, so a kernel that is fast at large sizes but slow at small ones cannot hide it. At small shapes the roof is the latency floor: an empty launch plus one cold DRAM read and one write (about 4 µs on an RTX PRO 6000), not bandwidth. Read the per-shape table in every observation, and optimize the shapes furthest from their roof first.

Run everything from the repo root. By default `pgaio.py` drives arena through Docker (`docker compose run --rm -T cli`), which has all four DSLs. For a native build pass `--arena build/arena` (any build dir) to `init` or set `ARENA_CMD`.

## 1. Start a run

A run targets a problem: a category, its sizes, and its correctness checks. Candidates can be in any allowed DSL, in any order.

```bash
python3 tools/pgaio.py init --category softmax --sizes all --budget-per-dsl 10 --target 0.9 \
    --mode isolated --docs off --model <your model id>
```

Categories: `reduce`, `scan`, `softmax`, `matmul` and `attention` (`[BH, S, D]` fp32, non-causal, D = 64 in the timed and checked shapes).

`init` prints where the run lives and two files that update after every eval: `journal.md` (each iteration's hypothesis, result and code diff against the previous version in that DSL) and `progress.html` (the progression chart plus every iteration's code). The user follows the run through them, so keep hypotheses clear.

`init` measures the category's plainest kernel as iteration 0 (softmax: `softmax_naive`; reduce: `reduce_baseline`; scan: `scan_multiblock`; matmul: `matmul_naive`; override with `--baseline`), prints its observation and any learnings from earlier runs, and creates `pgaio/runs/<run>/` with `work/` for your drafts. Checks the baseline itself fails are dropped from the gate and listed. `--dsls cuda,triton` restricts the languages; `--budget` caps the total.

Options worth knowing:

- `--mode isolated`: do not read, inspect or time the existing kernels in `kernels/`, and do not use what you know of them. Your own candidates, in every DSL, are fair game. Reading the harness is fine: the DSL helpers (`kernels/triton_base.py`, `cutile_base.py`, `warp_base.py`), `include/` and `src/`.
- `--mode cross-dsl`: you may also inspect and time any existing kernel and say which one inspired an idea.
- `--docs on|off`: whether a CUDA documentation tool is available (section 5). Record it truthfully.

## 2. The loop

For each idea:

1. **Write a candidate** in `pgaio/runs/<run>/work/`: a kernel source plus a manifest saying how to launch it. The baseline's source is copied to `candidates/000_baseline/` to start from. Copy the matching example from [manifests.md](manifests.md) for a new DSL rather than guessing its argument format.
2. **Look at it for free**:

   ```bash
   python3 tools/pgaio.py sass <run> pgaio/runs/<run>/work/v3.json
   ```

   This compiles the draft and prints registers, spills, the SASS instruction mix and its most frequent opcodes, the difference against the best so far, and a `diff` command for the two SASS listings. It saves `kernel.sass` and whatever else the compiler produced into `inspections/NNN_<draft>/`: `kernel.ptx` (CUDA, Triton), `kernel.ttgir` (Triton's IR, where layouts, vector widths and pipelining are decided) and `kernel.warp_gen.cu` (the CUDA C++ Warp generated). It never runs or times the kernel, so it costs nothing. Use it as often as you like: to check that a change compiled the way you meant (128-bit loads, warp shuffles instead of shared memory, no spills), and to compare how the same idea compiles in different DSLs.

   `--show LDG,STG,BAR` lists the SASS lines starting with those mnemonics in program order, with addresses: where loads sit relative to barriers, whether a prefetch was hoisted or sunk, without reading the whole listing.
   Before measuring, confirm in this output that the change you meant actually happened. In the PGAIO runs so far, iterations were wasted on kernels whose SASS already showed the idea had not taken effect: `__ldg` removed but the load still `.CONSTANT`, a Triton load still scalar because an `if` merged its alignment, a tile spilling hundreds of `LDL`/`STL`. A `SPILLS:` line means shrink the tile or raise `num_warps` first.
3. **Measure it** when you expect it to help:

   ```bash
   python3 tools/pgaio.py eval <run> pgaio/runs/<run>/work/v3.json --hypothesis "<cause and effect>"
   ```

   This snapshots the manifest and source, checks correctness on held-out shapes, distributions and seeds, times it at a size larger than L2 with a cold cache, and prints the verdict, SOL, counters, SASS and a heuristic diagnosis. Every eval spends one iteration of that DSL's budget, whatever the verdict.
4. **Compare** any two measured iterations, free: `python3 tools/pgaio.py compare <run> 2 7`.
5. **Decide** what the result says about the hypothesis before the next one. A slower result rules something out; say what.

When small and large shapes want different kernels, write both and dispatch with `"when"` in the manifest ([manifests.md](manifests.md)), rather than settling for one kernel that is mediocre everywhere. A launch can also name its own `"source"`, so one candidate can combine several Triton, cuTile or Warp kernels (one kernel per file), or send small sizes to a CUDA kernel and mid sizes to a Triton one. A candidate mixing DSLs counts as DSL `mixed`, with its own budget. In the scan and softmax runs the mixed candidate beat every single-DSL one. [playbook.md](playbook.md) lists what usually works for launch-bound, memory-bound and compute-bound shapes.

State hypotheses as cause and effect, citing the evidence: "Triton's SASS uses 4 shuffles and no shared memory per reduction where my CUDA kernel uses 10 shared ops and 9 syncs; a warp-shuffle reduction in CUDA should cut the syncs and gain a few percent."

Look up a DSL's API in the installed version, free: `python3 tools/pgaio.py api cutile load`, `api warp tile_load`, `api triton max`; without a name it lists what the module exports. Prefer this to guessing, since cuTile and Warp change between releases.

Moving ideas between DSLs is the point. When one DSL produces better SASS for the same algorithm, look at what it emitted and write the same thing in the other. When a DSL's abstraction blocks an idea, say so and move on.

Verdicts: `ok`, `incorrect` (a gated check failed; size, distribution, seed and error are listed), `compile_error`, `runtime_error` (includes a manifest whose arguments do not match the kernel; the message says which), `rejected_guard`. `pgaio.py status <run>` shows the history and the budget left per DSL.

Stop when the target is reached, the budget is spent, or several iterations in a row fail to improve while the diagnosis says you are within 15% of the roof.

## 3. Rules

These keep the score meaningful, and the harness enforces most of them.

- Do not modify anything under `src/`, `include/`, `kernels/`, `tests/`, `tools/`, `CMakeLists.txt` or the Docker files during a run. `eval` hashes them and rejects the iteration if anything changed. `pgaio.py rehash` exists for maintainers who deliberately improve the harness mid-run, and records why in the journal; never use it to get your own change through. If a candidate needs a harness change, stop and say so.
- Candidates are kernels plus manifests. A manifest cannot run code, so verification, input data and the byte counts SOL is computed from are out of reach by design.
- Compute the full result from the inputs on every launch. Verification runs on inputs regenerated with a seed the kernel never saw, after timing, and checks the whole output, so caching, precomputing or skipping work fails.
- Keep accuracy within the gate on every distribution, including `adversarial`.
- No libraries that do the work for you: no cuDNN, CUB, Thrust, CUTLASS, cuBLAS or PyTorch ops. Language primitives (warp shuffles, `tl.max`, `ct.sum`) are fine.
- Measure through `pgaio.py eval` only. Do not time candidates with arena directly to choose what to submit; `sass` is the free tool, and it shows no timing.

The timed size is chosen to be several times the L2 (softmax: 16384 x 4096, 512 MB in and out against 128 MB of L2). Cache-policy hints such as an evict_last store are allowed, but they cannot hide most of the output in L2 at this size; if the diagnosis says DRAM moved well under the ideal bytes, say so in the summary rather than presenting the SOL as pure bandwidth.

Timing noise between runs is a few percent. Do not read a 2% change as a win or a loss without a SASS or counter difference to explain it. Float atomics make summation order nondeterministic, so an atomic-heavy kernel can land on either side of a tolerance by a hair.

## 4. Finish

1. `python3 tools/pgaio.py status <run>` and `python3 tools/pgaio.py best <run>`.
2. Record one to three insights for future runs, with evidence:

   ```bash
   python3 tools/pgaio.py learn <run> --insight "cuTile softmax: TILE 4096 with one row per block beat 1024 by 6%" \
       --evidence "iter 9 vs 6"
   ```

3. `python3 tools/pgaio.py report --html pgaio/runs/<run>/progress.html` draws the progression: every iteration coloured by DSL, the best-so-far line, hypotheses and SASS changes. The markdown tables give iterations-to-50/70/80/90% SOL overall and per DSL.
4. Summarize for the user: start and final SOL, which iterations moved it and why, which DSL ideas transferred, what did not work, and whether the best candidate should be promoted to `kernels/` (arena-add-kernel). Promotion changes `kernels/`, so ask first. To compare the candidates by hand, point the user at the GUI in Docker, which has every DSL: `docker compose run --rm gui --gui --open pgaio/runs/<run>` (run `xhost +local:docker` first). A native build cannot run Triton, cuTile or Warp candidates.

## 5. CUDA documentation tools

Before `init`, check whether a CUDA documentation tool is connected: an MCP tool whose name mentions Nsight Copilot, CUDA docs or similar.

- If one is connected, pass `--docs on` and use it to check API signatures (cuTile and Warp change between versions), architecture limits for the GPU in `environment.arch`, and unfamiliar PTX or SASS instructions. A lookup is cheaper than a failed compile.
- If none is connected, tell the user once that a CUDA documentation MCP server would help (NVIDIA's Nsight Copilot CUDA server, for those with access) and that adding one takes effect in a new session. Do not add it yourself. Then continue with `--docs off`.

## Single-slot runs

`init --kernel <name>` instead optimizes one existing kernel's source in place: candidates are plain source files of that kernel's type, run with `--source-override` against its compiled descriptor, and `sass`/`eval` take the source path. Use it to tune a kernel without changing its launch shape. Everything else above applies.

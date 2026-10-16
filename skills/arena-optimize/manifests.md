# Kernel manifests

A manifest is a small JSON file next to a kernel source. It tells arena how to launch the kernel against a category's buffers. It is data, not code: it can choose the launch shape and the arguments, but it cannot compute, copy or verify anything. Verification, the CPU reference, input generation and the byte and FLOP counts always come from the category.

Run one with `$ARENA --manifest <file.json>` (plus the usual `-p`, `--dist`, `--seed`, `--l2`, `--profile`), and inspect it with `$ARENA --inspect --manifest <file.json>` or `--sass` / `--ptx`. `$ARENA --plan --manifest <file.json> --shape ...` prints each launch's resolved grid, block, shared memory, scratch sizes, arguments and defines without launching anything; in a PGAIO run, `pgaio.py check` does this for every timed and checked shape. It also checks each launch against the device and the compiled function (see Limits below). Relative paths are resolved from the current directory, or else the repo root; the manifest's own `source` is resolved from the manifest's folder.

## Fields

| Field | Meaning |
| --- | --- |
| `category` | `reduce`, `scan`, `softmax`, `layernorm`, `matmul` or `attention`. Fixes the buffers, sizes and verification. |
| `source` | Kernel file: `.cu`, `.triton.py`, `.cutile.py` or `.warp.py`. The extension picks the compiler. |
| `name`, `description` | Optional labels. |
| `scratch` | Optional list of `{"name": ..., "bytes": expr, "zero": true}` device buffers. `zero` is optional and only matters under `--scratch-policy declared` (below). |
| `launches` | One or more launches, run in order on one stream. |

How scratch starts each launch depends on `--scratch-policy`:

- `legacy` (the default): every scratch buffer is zeroed before every launch, outside the timed region, so a kernel gets a zeroed workspace for free.
- `declared`: a buffer with `"zero": true` is zeroed by an async memset at the start of the timed op, so `op_ms` pays for it (it is not a kernel, so `gpu_ms` and `launch_count` do not include it). Every other buffer is filled with `0xFF` bytes outside the timed region, so a kernel that relies on zeros it did not declare fails verification. Each result records the policy as `launch.scratch_policy`.

Under `declared`, a kernel that spin-waits on an undeclared buffer (split-K semaphores, look-back flags) does not fail: it hangs on the `0xFF` fill. Declare every buffer the kernel expects to start at zero.

Each launch:

| Field | Meaning |
| --- | --- |
| `when` | Optional expression; the launch runs only where it is non-zero, e.g. `"rows * cols < 1048576"`. Lets one manifest dispatch small and large inputs to different kernels. At least one launch must apply at every size. |
| `source` | Optional; this launch's kernel file, relative to the manifest, if not the top-level one. Each file is compiled separately, and the launch's expressions see that file's constants (`num_warps`, `BLOCK`, ...). Lets a candidate chain Triton/cuTile/Warp kernels (one kernel per file) or dispatch across DSLs. |
| `defines` | Optional `{"BLOCK": 256, "ROWS": 4}`: compile constants for this launch's source, so one file builds into several variants instead of one file per config. Each distinct set compiles once and is cached on its own. Keys are identifiers, values integers. CUDA gets `-DBLOCK=256` (guard the default with `#ifndef BLOCK`), Triton constexprs and `num_warps`/`num_stages`, cuTile `ct.Constant` parameters. Warp rejects them, since it fixes its constants in the source. The launch's expressions see these values. For the manifest's own source they go over any `--define`. |
| `entry` | Function name in the cubin. Omit for Triton, cuTile and Warp, whose compiler reports it. CUDA needs the `extern "C"` name when the file has more than one kernel. |
| `grid`, `block` | 1 to 3 sizes each, integers or expressions. At most 1024 threads per block. |
| `shared_mem` | Dynamic shared memory in bytes. Above 48 KB is opted in automatically, up to the device's per-block limit (static plus dynamic; 99 KB on most recent GPUs). |
| `args` | The kernel's parameters, in order (below). |

## Expressions

Integers with `+ - * / %`, comparisons `< <= > >= == !=` (1 or 0), `&& || !`, `c ? a : b`, parentheses, `cdiv(a, b)`, `min(...)`, `max(...)`. Precedence is C's, and `?:`, `&&` and `||` only evaluate the side they use, so `"M > 0 ? N / M : 0"` is fine at `M = 0`. Names available:

- Problem sizes: `n` (reduce, scan), `rows`, `cols` (softmax, layernorm), `M`, `K`, `N` (matmul), `BH`, `S`, `D` (attention).
- `sm_count`.
- The launch's own `defines`.
- What the compiler reported: `num_warps` (Triton), `block_dim` (cuTile's required threads per block), `shared_memory` (Triton's, and Warp's dynamic shared memory for tile kernels), and every constant: Triton constexprs, cuTile `ct.Constant` values, and upper-case `// arena: KEY=N` lines in a CUDA source.
- `<scratch name>_bytes` for each scratch buffer.

## Arguments

| Entry | Passes |
| --- | --- |
| `"input"`, `"output"` (reduce, scan, softmax), `"input"`, `"gamma"`, `"beta"`, `"output"` (layernorm), `"a"`, `"b"`, `"c"` (matmul), `"q"`, `"k"`, `"v"`, `"o"` (attention), or a scratch name | a device pointer |
| `"n"`, `"rows"`, `"cols"`, `"M"`, `"K"`, `"N"` | that size as int32 |
| `"null"` | a null pointer |
| `"..."` (last) | null pointers until the kernel's parameter count is reached; Triton adds hidden ones |
| `"eps"` (layernorm) | the category's float constant (1e-5) as f32 |
| `{"i32": expr}`, `{"u32": expr}`, `{"i64": expr}` | a scalar of that width |
| `{"f32": x}`, `{"f64": x}` | a float scalar (4 or 8 bytes): a number (`1e-5`), a float name (`"eps"`), or an integer expression (`"cols"`) |
| `{"cutile_array": "input", "shape": [...], "strides": [...]}` | cuTile's array expansion: pointer, then every shape, then every stride (u32, strides in elements, row-major if omitted) |
| `{"warp_array": "input", "shape": [...]}` | Warp's `array_t<float>` struct (strides in bytes, contiguous if omitted) |
| `{"warp_bounds": expr}` | Warp's 1D launch bounds; Warp kernels take it first |

Arena reads each compiled function's real parameter sizes from the driver and refuses a manifest whose argument count or sizes do not match, with a message saying which argument is wrong.

## Limits

For every launch that applies, `--plan` reports `"resources": {"registers", "static_shared_bytes", "max_threads_per_block"}` from the compiled function and `"problems"`, a list of what the device cannot run: static plus dynamic shared memory over the per-block opt-in limit, more threads than the function allows at its register count or than 1024, grid or block sizes over the device's limits or below 1. The output's top-level `"device"` holds the limits. Any problem becomes the shape's `"error"`, naming the launch:

```
manifest launch 4 (layernorm_split_9cc3159f_cuda_kernel_forward): needs 198832 B of shared memory (196640 dynamic + 2192 static), the device allows 101376 per block
```

A run checks the same before launching and fails with the same message, instead of CUDA's bare "invalid argument".

## Examples (each one run and verified)

CUDA, `softmax_naive.cu` with `extern "C" __global__ void softmax_naive(const float* input, float* output, int rows, int cols)`:

```json
{"category": "softmax", "source": "softmax_naive.cu",
 "launches": [{"entry": "softmax_naive", "grid": ["rows"], "block": [256],
               "shared_mem": "256 * 4", "args": ["input", "output", "rows", "cols"]}]}
```

Triton, `@triton.jit def softmax(input_ptr, output_ptr, rows, cols, BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr)`, compiled with `main(fn=softmax, signature={"input_ptr": "*fp32", "output_ptr": "*fp32", "rows": "i32", "cols": "i32"}, constants={...}, num_warps=8)` from `triton_base` (`num_warps`, `num_stages` and `divisibility={...}` are optional):

```json
{"category": "softmax", "source": "softmax.triton.py",
 "launches": [{"grid": ["cdiv(rows, ROWS_PER_PROGRAM)"], "block": ["num_warps * 32"],
               "shared_mem": "shared_memory", "args": ["input", "output", "rows", "cols", "..."]}]}
```

cuTile, `@ct.kernel def softmax(X, Y, ROWS: int, COLS: int, TILE: ct.Constant[int])` with 2D arrays, compiled with `main(kernel_fn=softmax, dummy_args=(x_2d, y_2d, 4, 4096, TILE), constants={"TILE": TILE})` from `cutile_base`. `ct.Constant` parameters take no argument slot:

```json
{"category": "softmax", "source": "softmax.cutile.py",
 "launches": [{"grid": ["rows"], "block": ["block_dim"],
               "args": [{"cutile_array": "input", "shape": ["rows", "cols"]},
                        {"cutile_array": "output", "shape": ["rows", "cols"]},
                        "rows", "cols"]}]}
```

Use `padding_mode=ct.PaddingMode.ZERO` (or `NEG_INF` for a max) on loads that can run past the edge; the default padding is undetermined. TMA loads only support zero padding, so mask a max by index instead of padding with `NEG_INF` when you want TMA.

## Alignment

Compiled ahead of time, a DSL does not know what a JIT launch would have seen, so the helpers state what is true of every arena buffer:

- Triton: pointer arguments are marked 16-byte aligned (`tt.divisibility = 16`), as Triton's JIT would. Add facts about other arguments with `main(..., divisibility={"cols": 16})`. `tl.multiple_of` inside the kernel is not a substitute.
- cuTile: arrays are declared contiguous (last stride 1) and 16-byte aligned. Add per-array facts with `main(..., array_hints={"X": {"stride_divisible_by": [4, 1]}})`, keyed by parameter name.
- CUDA: nothing implicit; use `float4` loads and the like yourself.

A hint that is false for some shape gives wrong results there, and verification will catch it. When a fact holds only for some shapes (for example `cols % 4 == 0`), pass the same buffer twice under two parameter names, hint only the second, and branch in the kernel.

Warp, `@wp.kernel def softmax(input: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32), rows: wp.int32, cols: wp.int32)`, compiled with `main(kernel_fn=softmax, constants={})` from `warp_base`:

```json
{"category": "softmax", "source": "softmax.warp.py",
 "launches": [{"grid": ["cdiv(rows, 256)"], "block": [256],
               "args": [{"warp_bounds": "rows"},
                        {"warp_array": "input", "shape": ["rows * cols"]},
                        {"warp_array": "output", "shape": ["rows * cols"]},
                        "rows", "cols"]}]}
```

A scan with a scratch buffer, CUDA, launch shape from `// arena: THREADS=256 GRID_PER_SM=4` in the source:

```json
{"category": "scan", "source": "scan.cu",
 "scratch": [{"name": "ws", "bytes": "64 + 16 * cdiv(n, 32)"}],
 "launches": [{"grid": ["sm_count * GRID_PER_SM"], "block": ["THREADS"],
               "args": ["input", "output", "n", "ws"]}]}
```

Size dispatch: a warp-per-row kernel for short rows and a block-per-row kernel otherwise, both in one CUDA file (softmax run, iteration 1):

```json
{"category": "softmax", "source": "v1.cu",
 "launches": [
   {"when": "cols <= 1024", "entry": "softmax_warp", "grid": ["cdiv(rows, 8)"], "block": [256],
    "args": ["input", "output", "rows", "cols"]},
   {"when": "cols > 1024 && cols <= 16384", "entry": "softmax_block", "grid": ["rows"], "block": [256],
    "args": ["input", "output", "rows", "cols"]}]}
```

Dispatch across DSLs, with three Triton kernels from three files for the middle sizes (scan run, iteration 6, the best scan candidate):

```json
{"category": "scan", "source": "v4.cu",
 "scratch": [{"name": "ws", "bytes": "64 + 16 * cdiv(n, 32)"},
             {"name": "sums", "bytes": "4 * cdiv(n, 4096) + 16"}, {"name": "offsets", "bytes": "4 * cdiv(n, 4096) + 16"}],
 "launches": [
   {"when": "n < 1048576 || n >= 16777216", "entry": "scan_v4", "grid": ["max(1, cdiv(n, 4096))"], "block": [256],
    "args": ["input", "output", "n", "ws"]},
   {"when": "n >= 1048576 && n < 16777216", "source": "t_sums.triton.py", "grid": ["max(1, cdiv(n, TILE))"],
    "block": ["num_warps * 32"], "shared_mem": "shared_memory", "args": ["input", "sums", "n", "..."]},
   {"when": "n >= 1048576 && n < 16777216", "source": "t_offsets.triton.py", "grid": [1],
    "block": ["num_warps * 32"], "shared_mem": "shared_memory",
    "args": ["sums", "offsets", {"i32": "max(1, cdiv(n, 4096))"}, "..."]},
   {"when": "n >= 1048576 && n < 16777216", "source": "t_scan.triton.py", "grid": ["max(1, cdiv(n, TILE))"],
    "block": ["num_warps * 32"], "shared_mem": "shared_memory", "args": ["input", "output", "offsets", "n", "..."]}]}
```

Variants of one Triton file, with `eps` passed as an argument (`signature={..., "eps": "fp32"}`) and a constexpr set per launch:

```json
{"category": "layernorm", "source": "ln.triton.py",
 "launches": [
   {"when": "cols <= 1024 && rows <= 2048", "defines": {"BLOCK": 1024, "ROWS": 1, "num_warps": 4},
    "grid": ["cdiv(rows, ROWS)"], "block": ["num_warps * 32"], "shared_mem": "shared_memory",
    "args": ["input", "gamma", "beta", "output", "rows", "cols", "eps", "..."]},
   {"when": "cols <= 1024 && rows > 2048", "defines": {"BLOCK": 1024, "ROWS": 4, "num_warps": 8},
    "grid": ["cdiv(rows, ROWS)"], "block": ["num_warps * 32"], "shared_mem": "shared_memory",
    "args": ["input", "gamma", "beta", "output", "rows", "cols", "eps", "..."]},
   {"when": "cols > 1024", "defines": {"BLOCK": 4096, "ROWS": 1, "num_warps": 8},
    "grid": ["cdiv(rows, ROWS)"], "block": ["num_warps * 32"], "shared_mem": "shared_memory",
    "args": ["input", "gamma", "beta", "output", "rows", "cols", {"f32": 1e-5}, "..."]}]}
```

`--inspect --artifacts-dir <dir>` saves the PTX, TTGIR or Warp's generated CUDA of every source and variant as `<file stem>[_DEFINES].ptx` and so on (`ln.ptx`, `ln_BLOCK1024_ROWS4_num_warps8.ptx`), plus `kernel.*` for the manifest's main source.

Two kernels from one file run as two launches, for example a block-sum pass and a fix-up pass:

```json
{"category": "scan", "source": "two_pass.cu",
 "scratch": [{"name": "sums", "bytes": "4 * cdiv(n, 1024)"}],
 "launches": [
   {"entry": "scan_blocks", "grid": ["cdiv(n, 1024)"], "block": [256], "args": ["input", "output", "n", "sums"]},
   {"entry": "add_offsets", "grid": ["cdiv(n, 1024)"], "block": [256], "args": ["output", "n", "sums"]}]}
```

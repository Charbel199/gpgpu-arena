# Kernel manifests

A manifest is a small JSON file next to a kernel source. It tells arena how to launch the kernel against a category's buffers. It is data, not code: it can choose the launch shape and the arguments, but it cannot compute, copy or verify anything. Verification, the CPU reference, input generation and the byte and FLOP counts always come from the category.

Run one with `$ARENA --manifest <file.json>` (plus the usual `-p`, `--dist`, `--seed`, `--l2`, `--profile`), and inspect it with `$ARENA --inspect --manifest <file.json>` or `--sass` / `--ptx`. Relative paths are resolved from the repo root; the manifest's own `source` is resolved from the manifest's folder.

## Fields

| Field | Meaning |
| --- | --- |
| `category` | `reduce`, `scan`, `softmax` or `matmul`. Fixes the buffers, sizes and verification. |
| `source` | Kernel file: `.cu`, `.triton.py`, `.cutile.py` or `.warp.py`. The extension picks the compiler. |
| `name`, `description` | Optional labels. |
| `scratch` | Optional list of `{"name": ..., "bytes": expr}` device buffers, zeroed before every launch. |
| `launches` | One or more launches, run in order on one stream. |

Each launch:

| Field | Meaning |
| --- | --- |
| `entry` | Function name in the cubin. Omit for Triton, cuTile and Warp, whose compiler reports it. CUDA needs the `extern "C"` name when the file has more than one kernel. |
| `grid`, `block` | 1 to 3 sizes each, integers or expressions. At most 1024 threads per block. |
| `shared_mem` | Dynamic shared memory in bytes. Above 48 KB is opted in automatically. |
| `args` | The kernel's parameters, in order (below). |

## Expressions

Integers with `+ - * / %`, parentheses, `cdiv(a, b)`, `min(...)`, `max(...)`. Names available:

- Problem sizes: `n` (reduce, scan), `rows`, `cols` (softmax), `M`, `K`, `N` (matmul).
- `sm_count`.
- What the compiler reported: `num_warps` (Triton), `block_dim` (cuTile's required threads per block), `shared_memory` (Triton), and every constant: Triton constexprs, cuTile `ct.Constant` values, and upper-case `// arena: KEY=N` lines in a CUDA source.
- `<scratch name>_bytes` for each scratch buffer.

## Arguments

| Entry | Passes |
| --- | --- |
| `"input"`, `"output"` (reduce, scan, softmax), `"a"`, `"b"`, `"c"` (matmul), or a scratch name | a device pointer |
| `"n"`, `"rows"`, `"cols"`, `"M"`, `"K"`, `"N"` | that size as int32 |
| `"null"` | a null pointer |
| `"..."` (last) | null pointers until the kernel's parameter count is reached; Triton adds hidden ones |
| `{"i32": expr}`, `{"u32": expr}`, `{"i64": expr}` | a scalar of that width |
| `{"cutile_array": "input", "shape": [...], "strides": [...]}` | cuTile's array expansion: pointer, then every shape, then every stride (u32, strides in elements, row-major if omitted) |
| `{"warp_array": "input", "shape": [...]}` | Warp's `array_t<float>` struct (strides in bytes, contiguous if omitted) |
| `{"warp_bounds": expr}` | Warp's 1D launch bounds; Warp kernels take it first |

Arena reads each compiled function's real parameter sizes from the driver and refuses a manifest whose argument count or sizes do not match, with a message saying which argument is wrong.

## Examples (each one run and verified)

CUDA, `softmax_naive.cu` with `extern "C" __global__ void softmax_naive(const float* input, float* output, int rows, int cols)`:

```json
{"category": "softmax", "source": "softmax_naive.cu",
 "launches": [{"entry": "softmax_naive", "grid": ["rows"], "block": [256],
               "shared_mem": "256 * 4", "args": ["input", "output", "rows", "cols"]}]}
```

Triton, `@triton.jit def softmax(input_ptr, output_ptr, rows, cols, BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr)`, compiled with `main(fn=softmax, signature={"input_ptr": "*fp32", "output_ptr": "*fp32", "rows": "i32", "cols": "i32"}, constants={...})` from `triton_base`:

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

Use `padding_mode=ct.PaddingMode.ZERO` (or `NEG_INF` for a max) on loads that can run past the edge; the default padding is undetermined.

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

Two kernels from one file run as two launches, for example a block-sum pass and a fix-up pass:

```json
{"category": "scan", "source": "two_pass.cu",
 "scratch": [{"name": "sums", "bytes": "4 * cdiv(n, 1024)"}],
 "launches": [
   {"entry": "scan_blocks", "grid": ["cdiv(n, 1024)"], "block": [256], "args": ["input", "output", "n", "sums"]},
   {"entry": "add_offsets", "grid": ["cdiv(n, 1024)"], "block": [256], "args": ["output", "n", "sums"]}]}
```

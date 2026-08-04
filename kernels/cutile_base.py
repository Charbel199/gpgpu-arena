#!/usr/bin/env python3
"""Compile a cuTile kernel to a cubin and report its metadata as JSON.

cuda-tile 1.5 replaced compile_tile(fn, dummy_args, ...) with an explicit
signature: every parameter is described by a constraint rather than inferred
from an example value. Kernel files still pass example arrays, which is the
friendlier thing to write, and this module turns them into constraints.
"""

import argparse
import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_import_t0 = time.perf_counter()
import cupy as cp
import cuda.tile as ct
from cuda.tile._compile import compile_tile
from cuda.tile.compilation import CallingConvention
from cuda.tile.compilation._signature import (
    KernelSignature, ArrayConstraint, ConstantConstraint, ScalarConstraint,
)
_IMPORT_MS = (time.perf_counter() - _import_t0) * 1000.0


# cupy dtype -> cuTile dtype, limited to what the kernels here use.
_DTYPES = {
    "float32": ct.float32,
    "float16": ct.float16,
    "int32":   ct.int32,
    "int64":   ct.int64,
    "uint32":  ct.uint32,
}


def _constraint(arg, is_constant):
    """Turn one example argument into the constraint describing it.

    Returns (constraint, launch slots it occupies). Whether a scalar is baked
    in or passed at launch comes from the kernel's own annotation, not from the
    value: ct.Constant[int] means baked, a plain int means runtime.
    """
    if isinstance(arg, (int, bool)):
        if is_constant:
            # Baked into the cubin, so it takes no launch slot. Confirmed with
            # cuobjdump: two 1D arrays plus one constant is six parameters,
            # not seven. It also locks the cubin to that value, which is why
            # a matmul K belongs in a runtime scalar instead.
            return ConstantConstraint(arg), 0
        return ScalarConstraint(ct.int32), 1

    name = str(arg.dtype)
    if name not in _DTYPES:
        raise TypeError(f"unsupported dtype {name}; add it to _DTYPES")

    # stride_lower_bound_incl=0 because negative strides are rejected outright.
    c = ArrayConstraint(
        _DTYPES[name], arg.ndim,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        alias_groups=[],
        may_alias_internally=False,
    )
    # One pointer, then a shape and a stride per dimension.
    return c, 1 + 2 * arg.ndim


def compile_kernel(kernel_fn, example_args):
    fn = getattr(kernel_fn, "_pyfunc", kernel_fn)

    params = list(inspect.signature(fn).parameters.values())
    constraints, slots = [], 0
    for a, p in zip(example_args, params):
        is_const = "Constant" in str(p.annotation)
        c, n = _constraint(a, is_const)
        constraints.append(c)
        slots += n

    symbol = fn.__name__

    sig = KernelSignature(
        parameters=constraints,
        calling_convention=CallingConvention.cutile_python_v2(),
        symbol=symbol,
    )

    result = compile_tile(fn, [sig])
    return symbol, result.cubin, slots


def query_block_dim(cubin_path, kernel_name):
    """Required threads per block, read from the cubin's .reqntid.

    Not CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: that reports the largest
    block the kernel could support (1024 here), while cuTile emits a
    .reqntid that the launch must match exactly. Launching at any other
    size returns CUDA_ERROR_INVALID_VALUE.
    """
    try:
        out = subprocess.run(["cuobjdump", "--dump-elf", str(cubin_path)],
                             capture_output=True, text=True).stdout
        m = re.search(r"\.reqntid\s+(\d+)", out)
        if m:
            return int(m.group(1))
        print("[cutile] no .reqntid in cubin, falling back to 128", file=sys.stderr)
    except Exception as e:
        print(f"[cutile] Could not read .reqntid, using 128: {e}", file=sys.stderr)
    return 128


def main(kernel_fn, dummy_args, constants=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, required=True)
    args = parser.parse_args()

    try:
        print(f"[cutile] Compiling {args.output_name} ...", file=sys.stderr)
        t0 = time.perf_counter()
        kernel_name, cubin, num_params = compile_kernel(kernel_fn, dummy_args)
        compile_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        print(f"[cutile] Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = output_dir / f"{args.output_name}.cubin"
    cubin_path.write_bytes(cubin)

    block_dim = query_block_dim(cubin_path, kernel_name)

    print(f"[cutile] {args.output_name} -> {cubin_path} "
          f"(kernel={kernel_name}, params={num_params}, block_dim={block_dim})",
          file=sys.stderr)

    print(json.dumps({
        "kernel_name": kernel_name,
        "num_warps": 0,
        "shared_memory": 0,
        "num_params": num_params,
        "block_dim": block_dim,
        "constants": constants or {},
        "import_ms": _IMPORT_MS,
        "compile_ms": compile_ms,
    }))

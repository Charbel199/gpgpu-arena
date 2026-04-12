import argparse
import json
import shutil
import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct
from cuda.tile._compile import compile_tile, CompilerOptions, default_tile_context


def compile_kernel(kernel_fn, dummy_args):
    """Compile a cuTile kernel to cubin. Returns (kernel_name, cubin_path, block_dim)."""
    lib = compile_tile(kernel_fn._pyfunc, dummy_args, CompilerOptions(), default_tile_context)

    # extract block size from the compiled cubin via CUDA driver API
    block_dim = 128  # fallback
    try:
        from cuda.bindings import driver as drv
        cubin_bytes = Path(lib.fname_cubin).read_bytes()
        err, mod = drv.cuModuleLoadData(cubin_bytes)
        err, func = drv.cuModuleGetFunction(mod, lib.func_name.encode())
        err, max_threads = drv.cuFuncGetAttribute(
            drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func)
        if max_threads > 0:
            block_dim = max_threads
        drv.cuModuleUnload(mod)
    except Exception as e:
        print(f"[cutile] Could not query block_dim, using {block_dim}: {e}", file=sys.stderr)

    return lib.func_name, str(lib.fname_cubin), block_dim


def main(kernel_fn, dummy_args, constants=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, required=True)
    args = parser.parse_args()

    try:
        print(f"[cutile] Compiling {args.output_name} ...", file=sys.stderr)
        kernel_name, cubin_path, block_dim = compile_kernel(kernel_fn, dummy_args)
    except Exception as e:
        print(f"[cutile] Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # copy cubin to output directory
    output_cubin = output_dir / f"{args.output_name}.cubin"
    shutil.copy2(cubin_path, output_cubin)

    # count params: the non-Constant args are the real kernel params
    num_params = sum(1 for a in dummy_args if not isinstance(a, int))

    print(f"[cutile] {args.output_name} -> {output_cubin} "
          f"(kernel={kernel_name}, block_dim={block_dim})", file=sys.stderr)

    # JSON metadata to stdout - parsed by KernelCompiler on the C++ side
    print(json.dumps({
        "kernel_name": kernel_name,
        "num_warps": 0,
        "shared_memory": 0,
        "num_params": num_params,
        "block_dim": block_dim,
        "constants": constants or {},
    }))

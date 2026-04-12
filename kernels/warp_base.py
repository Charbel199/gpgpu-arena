import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import warp as wp

# init warp BEFORE any @wp.kernel decorators run (avoids recursion)
wp.config.cuda_output = "cubin"
wp.init()


def count_cubin_params(cubin_path):
    """Count kernel params from cubin using cuobjdump."""
    result = subprocess.run(
        ["cuobjdump", "--dump-elf", str(cubin_path)],
        capture_output=True, text=True)
    return len(re.findall(r"EIATTR_KPARAM_INFO", result.stdout))


def compile_kernel(kernel_fn):
    # ompile a Warp kernel and return (cubin_path, kernel_name)
    device = wp.get_device("cuda:0")
    module = kernel_fn.module
    module.load(device)

    # warp writes the cubin to its cache directory
    module_id = module.get_module_identifier()
    output_name = module._get_compile_output_name(device)
    binary_path = os.path.join(wp.config.kernel_cache_dir, module_id, output_name)

    # kernel name is mangled: {name}_{hash}_cuda_kernel_forward
    kernel_name = kernel_fn.get_mangled_name() + "_cuda_kernel_forward"

    return binary_path, kernel_name


def main(kernel_fn, constants=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, required=True)
    args = parser.parse_args()

    try:
        print(f"[warp] Compiling {args.output_name} ...", file=sys.stderr)
        cubin_src, kernel_name = compile_kernel(kernel_fn)
    except Exception as e:
        print(f"[warp] Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cubin_path = output_dir / f"{args.output_name}.cubin"
    shutil.copy2(cubin_src, cubin_path)

    num_params = count_cubin_params(cubin_path)

    print(f"[warp] {args.output_name} -> {cubin_path} "
          f"(kernel={kernel_name}, params={num_params})",
          file=sys.stderr)

    # JSON metadata to stdout parsed by KernelCompiler on the C++ side
    print(json.dumps({
        "kernel_name": kernel_name,
        "num_warps": 0,
        "shared_memory": 0,
        "num_params": num_params,
        "constants": constants or {},
    }))

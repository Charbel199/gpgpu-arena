import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import triton
from triton.compiler import ASTSource


def check_gpu_supported():
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        cap = torch.cuda.get_device_capability()
        gpu_sm = f"sm_{cap[0]}{cap[1]}"
        supported = torch.cuda.get_arch_list()
        major = str(cap[0])
        for arch in supported:
            if arch.startswith(f"sm_{major}"):
                return True, gpu_sm
        return False, f"{gpu_sm} not supported by PyTorch (supported: {', '.join(supported)})"
    except Exception as e:
        return False, str(e)


def count_cubin_params(cubin_path):
    # count kernel params from cubin using cuobjdump
    result = subprocess.run(
        ["cuobjdump", "--dump-elf", str(cubin_path)],
        capture_output=True, text=True)
    return len(re.findall(r"EIATTR_KPARAM_INFO", result.stdout))


def compile_kernel(fn, signature, constants):
    src = ASTSource(fn=fn, signature=signature, constexprs=constants)
    compiled = triton.compile(src, target=triton.runtime.driver.active.get_current_target())
    cubin = compiled.asm["cubin"]
    metadata = compiled.metadata
    kernel_name = getattr(metadata, "name", fn.__name__)
    num_warps = getattr(metadata, "num_warps", 4)
    shared = getattr(metadata, "shared", 0)
    return cubin, kernel_name, num_warps, shared


def main(fn, signature, constants):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, required=True)
    args = parser.parse_args()

    supported, reason = check_gpu_supported()
    if not supported:
        print(f"[triton] Skipping {args.output_name}: {reason}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"[triton] Compiling {args.output_name} ...", file=sys.stderr)
        cubin, kernel_name, num_warps, shared = compile_kernel(fn, signature, constants)
    except Exception as e:
        print(f"[triton] Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cubin_path = output_dir / f"{args.output_name}.cubin"
    cubin_path.write_bytes(cubin)

    num_params = count_cubin_params(cubin_path)

    print(f"[triton] {args.output_name} -> {cubin_path} "
          f"(kernel={kernel_name}, warps={num_warps}, shmem={shared}, params={num_params})",
          file=sys.stderr)

    # JSON metadata to stdout parsed by KernelCompiler on the C++ side
    print(json.dumps({
        "kernel_name": kernel_name,
        "num_warps": num_warps,
        "shared_memory": shared,
        "num_params": num_params,
        "constants": constants,
    }))

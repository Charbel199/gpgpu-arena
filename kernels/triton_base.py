import argparse
import sys
from pathlib import Path

import triton
from triton.compiler import ASTSource


def compile_kernel(fn, signature, constants):
    src = ASTSource(fn=fn, signature=signature, constexprs=constants)
    compiled = triton.compile(src, target=triton.runtime.driver.active.get_current_target())
    ptx = compiled.asm["ptx"]
    kernel_name = getattr(compiled.metadata, "name", fn.__name__)
    return ptx, kernel_name


def main(fn, signature, constants):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, required=True)
    args = parser.parse_args()

    try:
        ptx, kernel_name = compile_kernel(fn, signature, constants)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ptx_path = output_dir / f"{args.output_name}.ptx"
    ptx_path.write_text(ptx)

    print(f"Generated: {ptx_path} (kernel: {kernel_name})")

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

try:
    import triton
    import triton.language as tl
    import torch
except ImportError as e:
    print(f"Error: {e}. Install with: pip install triton torch", file=sys.stderr)
    sys.exit(1)


@triton.jit
def reduce_sum(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data)
    tl.atomic_add(output_ptr, block_sum)



# TODO: should have a parent triton class that takes of everything after this comment
def compile_to_ptx(block_size=256):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    n = block_size * 4
    input_tensor = torch.ones(n, dtype=torch.float32, device="cuda")
    output_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")
    grid = ((n + block_size - 1) // block_size,)

    reduce_sum[grid](input_tensor, output_tensor, n, BLOCK_SIZE=block_size)


    if hasattr(reduce_sum, 'cache'):
        for _, compiled in reduce_sum.cache.items():
            if hasattr(compiled, 'asm') and 'ptx' in compiled.asm:
                ptx = compiled.asm['ptx']
                kernel_name = compiled.metadata.get('name', 'reduce_sum')
                return ptx, kernel_name, block_size

    cache_dir = Path.home() / ".triton" / "cache"
    if cache_dir.exists():
        ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ptx_files:
            ptx = ptx_files[0].read_text()
            # extract kernel name from PTX
            kernel_name = 'reduce_sum'
            for line in ptx.split("\n"):
                if ".visible .entry" in line:
                    parts = line.split()
                    if ".entry" in parts:
                        kernel_name = parts[parts.index(".entry") + 1].split("(")[0]
                        break
            return ptx, kernel_name, block_size

    raise RuntimeError("Could not extract PTX from compiled kernel")


def main():
    parser = argparse.ArgumentParser(description="Compile Triton reduce kernel to PTX")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, default="reduce_triton")
    args = parser.parse_args()

    if args.block_size <= 0 or (args.block_size & (args.block_size - 1)) != 0:
        print(f"Error: block-size must be a power of 2", file=sys.stderr)
        return 1

    ptx, kernel_name, block_size = compile_to_ptx(args.block_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ptx_path = output_dir / f"{args.output_name}.ptx"
    ptx_path.write_text(ptx)

    print(f"Generated: {ptx_path} (kernel: {kernel_name}, block_size: {block_size})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

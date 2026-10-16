import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_import_t0 = time.perf_counter()
import triton
from triton.compiler import ASTSource
_IMPORT_MS = (time.perf_counter() - _import_t0) * 1000.0


def check_gpu_supported():
    global _IMPORT_MS
    try:
        _torch_t0 = time.perf_counter()
        import torch
        _IMPORT_MS += (time.perf_counter() - _torch_t0) * 1000.0
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


def parse_defines(pairs):
    """--define KEY=int, repeated. Compile-time knobs from the C++ side."""
    out = {}
    for p in pairs or []:
        key, _, val = p.partition("=")
        if not key or not val:
            raise ValueError(f"malformed --define {p!r}, expected KEY=int")
        out[key] = int(val)
    return out


def divisibility_attrs(fn, signature, divisibility=None):
    """Alignment facts for the compiler, as Triton's JIT launcher would give them.

    Launched through triton's JIT, a pointer argument that happens to be
    16-byte aligned is specialized as tt.divisibility=16, which is what lets
    the compiler emit 128-bit loads. Compiling ahead of time from a signature
    loses that, so every load stays scalar. Arena's buffers all come from
    cuMemAlloc and are at least 256-byte aligned, so pointers get 16 here.
    `divisibility` adds or overrides by argument name, e.g. {"cols": 16},
    for facts the kernel author knows; a false one gives wrong results, which
    verification catches."""
    names = list(getattr(fn, "arg_names", signature.keys()))
    wanted = {n: 16 for n, t in signature.items() if str(t).startswith("*")}
    wanted.update(divisibility or {})
    attrs = {}
    for name, value in wanted.items():
        if name not in names:
            raise ValueError(f"divisibility names unknown argument {name!r}")
        if value and value > 1:
            attrs[(names.index(name),)] = [["tt.divisibility", int(value)]]
    return attrs


def compile_kernel(fn, signature, constants, options=None, divisibility=None):
    # num_warps is a compile option rather than a constexpr, so it is pulled
    # out of the defines instead of being passed through as one. Triton derives
    # the thread block from it, which is why a DSL kernel cannot be relaunched
    # at a different block size the way a hand-written one can.
    opts = dict(options or {})
    num_warps = opts.pop("num_warps", None)
    num_stages = opts.pop("num_stages", None)

    constants = {**constants, **opts}

    src = ASTSource(fn=fn, signature=signature, constexprs=constants,
                    attrs=divisibility_attrs(fn, signature, divisibility))
    compile_options = {}
    if num_warps is not None:
        compile_options["num_warps"] = num_warps
    if num_stages is not None:
        compile_options["num_stages"] = num_stages
    kwargs = {"options": compile_options} if compile_options else {}
    compiled = triton.compile(src, target=triton.runtime.driver.active.get_current_target(),
                              **kwargs)
    cubin = compiled.asm["cubin"]
    ptx = compiled.asm.get("ptx", "")
    ttgir = compiled.asm.get("ttgir", "")
    metadata = compiled.metadata
    kernel_name = getattr(metadata, "name", fn.__name__)
    num_warps = getattr(metadata, "num_warps", 4)
    shared = getattr(metadata, "shared", 0)
    return cubin, kernel_name, num_warps, shared, constants, ptx, ttgir


def main(fn, signature, constants, divisibility=None, num_warps=None, num_stages=None):
    """`divisibility` maps argument names to known divisibility, on top of
    the 16-byte pointer alignment every arena buffer has; see
    divisibility_attrs(). `num_warps` and `num_stages` are Triton compile
    options; a --define of the same name overrides them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--define", action="append", default=[])
    args = parser.parse_args()

    try:
        defines = parse_defines(args.define)
    except ValueError as e:
        print(f"[triton] {e}", file=sys.stderr)
        sys.exit(1)
    for key, value in (("num_warps", num_warps), ("num_stages", num_stages)):
        if value is not None:
            defines.setdefault(key, value)

    supported, reason = check_gpu_supported()
    if not supported:
        print(f"[triton] Skipping {args.output_name}: {reason}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"[triton] Compiling {args.output_name} ...", file=sys.stderr)
        _compile_t0 = time.perf_counter()
        cubin, kernel_name, num_warps, shared, constants, ptx, ttgir = compile_kernel(
            fn, signature, constants, defines, divisibility)
        compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
    except Exception as e:
        print(f"[triton] Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cubin_path = output_dir / f"{args.output_name}.cubin"
    cubin_path.write_bytes(cubin)
    # PTX sidecar for `arena --ptx`. Free here, since Triton produced it on the
    # way to the cubin anyway.
    if ptx:
        (output_dir / f"{args.output_name}.ptx").write_text(ptx)
    # Triton's GPU-level IR: layouts, pipelining and vectorization decisions
    # are visible here before they turn into PTX.
    if ttgir:
        (output_dir / f"{args.output_name}.ttgir").write_text(ttgir)

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
        "import_ms": _IMPORT_MS,
        "compile_ms": compile_ms,
    }))

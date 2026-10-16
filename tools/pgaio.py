#!/usr/bin/env python3
"""Profile and benchmark guided AI optimization (PGAIO) driver for GPGPU Arena.

An agent optimizes a problem (a category such as softmax) by writing
candidate kernels in any DSL, each with a small JSON manifest saying how to
launch it. This script is the environment around it: it measures every
candidate the same way, refuses the ways a candidate can cheat, keeps the
history, and turns raw counters into a short diagnosis the agent can act on.

    pgaio.py init    --category softmax [--sizes all] [--dsls cuda,triton] [--budget 40]
    pgaio.py sass    <run> <manifest.json>           free: SASS mix, no timing
    pgaio.py eval    <run> <manifest.json> --hypothesis "..."   one iteration
    pgaio.py compare <run> <iteration> <iteration|ref:lib>   free
    pgaio.py ab      <run> <iterA> <iterB> [--shape ..] [--rounds 6]   free: A/B timing
    pgaio.py check   <run> <manifest.json>           free: launch plan per shape
    pgaio.py skip-dsl <run> <dsl> --reason "..."     mark a DSL exhausted
    pgaio.py status  <run>
    pgaio.py best    <run>
    pgaio.py learn   <run> --insight "..." [--evidence "..."]
    pgaio.py report  [--csv curves.csv] [--html progress.html]

`init --kernel <name>` instead starts a single-slot run, where candidates
are plain sources run in that kernel's slot with --source-override.

Every eval counts as one iteration, including ones that fail to compile, fail
correctness, or touch a protected file. Looking at SASS does not: it shows
how a draft compiled, never how fast it is.

Only the standard library is used, so this runs on the host while arena runs
in Docker.
"""

import argparse
import datetime as _dt
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PGAIO_DIR = REPO / "pgaio"
RUNS_DIR = PGAIO_DIR / "runs"
LEARNINGS = PGAIO_DIR / "learnings.jsonl"

DEFAULT_ARENA_CMD = "docker compose run --rm -T cli"

# Files the agent must not change during a run. Candidates live in the run
# directory; anything here moving means the measurement itself was edited.
PROTECTED = [
    "src/*", "include/*", "kernels/*", "tests/*", "tools/*",
    "CMakeLists.txt", "Dockerfile", "docker-compose.yml", "docker/*",
]
# Build products that land under protected roots and are not sources.
UNPROTECTED_SUFFIXES = (".cubin", ".ptx", ".json", ".pyc", ".log")

# Correctness is gated on "check" shapes (a grid): odd sizes catch missing
# bounds checks, tiny ones catch grid math. Every timed shape is checked too.
# Checks the baseline itself fails are dropped at init, so the gate never
# demands more than the reference kernel delivers.
CATEGORIES = {
    "reduce":    {"check": {"n": [1, 1000, 1000003, 16777217]}},
    "scan":      {"check": {"n": [1, 1000, 1000003, 16777217]}},
    "softmax":   {"check": {"rows": [1, 333], "cols": [1000, 4097]}},
    "matmul":    {"check": {"M": [129], "N": [257], "K": [63, 512]}},
    "attention": {"check": {"BH": [1, 3], "S": [1, 100, 257], "D": [64]}},
}

# Timed shapes per size regime. "small" runs from a few KB to around L2 size,
# where launch latency and parallelism dominate; "large" is several times a
# 128 MB L2, where DRAM bandwidth (or FLOPs) does and a kernel cannot score
# by leaving its output in cache. "all" is both, the default when the user
# does not say which sizes matter.
SIZE_LADDERS = {
    "reduce": {
        "small": [{"n": 4096}, {"n": 262144}, {"n": 4194304}],
        "large": [{"n": 67108864}, {"n": 134217728}],
    },
    "scan": {
        "small": [{"n": 4096}, {"n": 262144}, {"n": 4194304}],
        "large": [{"n": 67108864}],
    },
    "softmax": {
        "small": [{"rows": 64, "cols": 1024}, {"rows": 1024, "cols": 1024}, {"rows": 4096, "cols": 512}],
        "large": [{"rows": 16384, "cols": 4096}, {"rows": 4096, "cols": 16384}],
    },
    "matmul": {
        "small": [{"M": 256, "N": 256, "K": 256}, {"M": 1024, "N": 1024, "K": 1024}],
        "large": [{"M": 4096, "N": 4096, "K": 4096}],
    },
    "attention": {
        "small": [{"BH": 8, "S": 128, "D": 64}, {"BH": 32, "S": 512, "D": 64}],
        "large": [{"BH": 32, "S": 2048, "D": 64}, {"BH": 8, "S": 4096, "D": 64}],
    },
}


def train_shapes(category, regime):
    ladder = SIZE_LADDERS.get(category, {})
    if regime == "all":
        return list(ladder.get("small", [])) + list(ladder.get("large", []))
    return list(ladder.get(regime, []))


def expand_grid(lists):
    """{"M": [1, 2], "N": [3]} -> [{"M": 1, "N": 3}, {"M": 2, "N": 3}]"""
    out = [{}]
    for k in sorted(lists):
        out = [{**o, k: v} for o in out for v in lists[k]]
    return out


def shape_args(shapes):
    """[{"rows": 64, "cols": 1024}] -> ["--shape", "cols=1024,rows=64"]"""
    out = []
    for sh in shapes:
        out += ["--shape", ",".join(f"{k}={v}" for k, v in sorted(sh.items()))]
    return out


def shape_label(params):
    return " ".join(f"{k}={v}" for k, v in params.items()) if params else "-"


CHECK_DISTS = ["uniform", "normal", "adversarial"]
SOL_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]

# The reference a problem run starts from: the plainest existing kernel of
# the category, measured as iteration 0.
CATEGORY_BASELINE = {
    "reduce": "reduce_baseline",
    "scan": "scan_multiblock",
    "softmax": "softmax_naive",
    "matmul": "matmul_naive",
    "attention": "attention_naive",
}
DSL_EXT = {".cu": "cuda", ".triton.py": "triton", ".cutile.py": "cutile", ".warp.py": "warp"}
DSLS = ["cuda", "triton", "cutile", "warp", "mixed"]


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tools/tests/test_pgaio.py)
# --------------------------------------------------------------------------

def params_args(params):
    """{"n": [1, 2]} -> ["-p", "n=1,2"]"""
    out = []
    for k, vs in sorted(params.items()):
        out += ["-p", f"{k}=" + ",".join(str(v) for v in vs)]
    return out


def combo_key(result):
    """Identifies one correctness check: sizes, distribution, seed."""
    inp = result.get("input", {})
    p = ",".join(f"{k}={v}" for k, v in sorted(inp.get("params", {}).items()))
    return f"{p}|{inp.get('distribution')}|{inp.get('seed')}"


def sol(result, env):
    """Speed of light: achieved throughput against the roofline at this
    kernel's arithmetic intensity. Uses op_ms throughput, the same figure the
    README's tables use, so launch overhead counts against the kernel."""
    tp = result.get("throughput", {})
    gflops = float(tp.get("gflops") or 0.0)
    gbps = float(tp.get("bandwidth_gbps") or 0.0)
    peak_bw = float(env.get("peak_dram_gbps") or 0.0)
    peak_fl = float(env.get("peak_fp32_gflops") or 0.0)
    out = {"fraction": 0.0, "bound": "unknown", "achieved": 0.0, "roof": 0.0,
           "unit": "", "intensity": None, "peak_dram_gbps": peak_bw,
           "ridge": (peak_fl / peak_bw) if peak_bw > 0 else None}
    if peak_bw <= 0 or gbps <= 0:
        return out
    ai = gflops / gbps if gbps > 0 else 0.0
    out["intensity"] = ai
    mem_roof_gflops = ai * peak_bw
    if peak_fl > 0 and mem_roof_gflops >= peak_fl:
        out.update(bound="compute", achieved=gflops, roof=peak_fl, unit="GFLOP/s",
                   fraction=gflops / peak_fl)
    else:
        out.update(bound="memory", achieved=gbps, roof=peak_bw, unit="GB/s",
                   fraction=gbps / peak_bw)
    return out


def size_sol(result, env):
    """SOL at one shape, against the slowest of three floors: moving the
    ideal bytes at peak DRAM bandwidth, doing the FLOPs at peak FP32, and the
    latency floor (an empty launch plus one cold DRAM read and one write).
    The last is what makes small shapes scorable: at a few KB no kernel can
    beat that latency, so it is the roof, not bandwidth. At large shapes this
    equals the plain roofline SOL."""
    tp = result.get("throughput", {})
    t = result.get("timing", {})
    median = float(t.get("op_ms") or 0.0)
    # Interquartile mean when arena reports it: steadier than the median for
    # microsecond kernels, whose timings cluster. Throughput is reported on
    # the median, so bytes and FLOPs are recovered with it.
    op = float(t.get("op_ms_trimmed") or median)
    gbps = float(tp.get("bandwidth_gbps") or 0.0)
    gflops = float(tp.get("gflops") or 0.0)
    nbytes = gbps * 1e6 * median        # GB/s x ms
    flops = gflops * 1e6 * median
    peak_bw = float(env.get("peak_dram_gbps") or 0.0)
    peak_fl = float(env.get("peak_fp32_gflops") or 0.0)
    # The latency roof is the round trip (launch, one cold DRAM read, one
    # write) when arena measured it, otherwise the empty launch.
    latency = float(env.get("roundtrip_floor_ms") or 0.0) or float(env.get("launch_floor_ms") or 0.0)
    floors = {
        "memory": nbytes / (peak_bw * 1e6) if peak_bw > 0 else 0.0,
        "compute": flops / (peak_fl * 1e6) if peak_fl > 0 else 0.0,
        "latency": latency,
    }
    bound = max(floors, key=floors.get)
    roof = floors[bound]
    return {
        "params": result.get("input", {}).get("params", {}),
        "op_ms": op,
        "gpu_ms": result.get("timing", {}).get("gpu_ms"),
        "fraction": (roof / op) if op > 0 and roof > 0 else 0.0,
        "bound": bound,
        "roof_ms": roof,
        "ideal_bytes": nbytes,
        "flops": flops,
    }


def geomean(values):
    """Geometric mean, so one shape far from its roof pulls the score down
    instead of being averaged away. A zero counts as a tiny positive value."""
    vals = [max(v, 1e-6) for v in values]
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= v
    return prod ** (1.0 / len(vals))


def classify_attempt(check_results, timing_results, check_exit, timing_exit,
                     guard_violations, required_checks):
    """Verdict for one iteration, in order of precedence."""
    if guard_violations:
        return "rejected_guard"
    all_res = (check_results or []) + (timing_results or [])
    errors = [r.get("error", "") for r in all_res if not r.get("success", False)]
    if errors or (not all_res and (check_exit or timing_exit)):
        text = " ".join(errors).lower()
        if any(w in text for w in ("nvcc", "compiler", "compilation", "syntaxerror",
                                   "error:", "traceback")):
            return "compile_error"
        return "runtime_error"
    ran = {combo_key(r) for r in check_results or []}
    missing = [k for k in required_checks if k not in ran]
    if missing:
        return "runtime_error"
    # Only required checks gate. A check the baseline already failed was
    # dropped at init and is reported but not held against the candidate.
    gated = [r for r in check_results or []
             if not required_checks or combo_key(r) in required_checks]
    if not all(r.get("verified", False) for r in gated + list(timing_results or [])):
        return "incorrect"
    return "ok"


def first_error(results):
    for r in results or []:
        if not r.get("success", False) and r.get("error"):
            return r["error"]
    return ""


def failed_checks(results, required=None):
    out = []
    for r in results or []:
        if required and combo_key(r) not in required:
            continue
        if r.get("success") and not r.get("verified"):
            n = r.get("numerics", {})
            out.append({"check": combo_key(r),
                        "max_rel_error": n.get("max_rel_error"),
                        "tolerance": n.get("tolerance")})
    return out


def guard_violations(before, after):
    """Protected files whose content changed, appeared or disappeared."""
    changed = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            changed.append(path)
    return changed


def is_protected(rel):
    if rel.endswith(UNPROTECTED_SUFFIXES) or "/__pycache__/" in f"/{rel}":
        return False
    # arena's compile cache for sources outside kernels/ (candidates), e.g.
    # kernels/ext_1a2b3c4d_mm.ttgir: written by every compile, not a source.
    if rel.startswith("kernels/") and os.path.basename(rel).startswith("ext_"):
        return False
    return any(fnmatch.fnmatch(rel, pat) for pat in PROTECTED)


def best_so_far(attempts):
    """Best score after each iteration, 0 until something passes."""
    best, curve = 0.0, []
    for a in attempts:
        if a.get("verdict") == "ok" and a.get("score") is not None:
            best = max(best, a["score"])
        curve.append(best)
    return curve


def iterations_to(curve, threshold, start_iteration=0):
    """First iteration whose best-so-far reaches threshold, or None.

    curve[0] is the baseline (iteration 0), so a baseline that already meets
    the threshold reports 0."""
    for i, v in enumerate(curve):
        if v >= threshold:
            return i + start_iteration
    return None


def sass_brief(inspect):
    s = (inspect or {}).get("sass", {})
    c = s.get("classes", {})
    return {
        "instructions": s.get("instructions", 0),
        "mean_global_load_bytes": s.get("mean_global_load_bytes", 0.0),
        "global_loads": sum(v for k, v in c.items() if k.startswith("global_load")),
        "global_load_128": c.get("global_load_128", 0),
        "global_atomic": c.get("global_atomic", 0),
        "warp_shuffle": c.get("warp_shuffle", 0),
        "warp_reduce": c.get("warp_reduce", 0),
        "shared_mem": c.get("shared_mem", 0),
        "sync": c.get("sync", 0),
        "tensor": c.get("tensor", 0),
        "local_mem": c.get("local_mem", 0),
    }


def diagnose(obs):
    """Heuristic reading of one observation. Each line names the evidence it
    rests on, so the agent can tell a measurement from a guess."""
    out = []
    t = obs.get("timing", {})
    s = obs.get("sol", {})
    c = obs.get("counters", {})
    r = obs.get("resources", {})
    b = obs.get("sass", {})
    ideal = obs.get("ideal_bytes") or 0

    frac = s.get("fraction", 0.0)
    if compute_bound(obs):
        return diagnose_compute(obs) + diagnose_noise(obs)
    if s.get("bound") == "memory":
        out.append(f"memory-bound: {frac:.1%} of the DRAM roof "
                   f"({s.get('achieved', 0):.0f} of {s.get('roof', 0):.0f} GB/s).")

    op, gpu = t.get("op_ms") or 0, t.get("gpu_ms") or 0
    if op > 0 and (op - gpu) / op > 0.2:
        out.append(f"{(op - gpu) / op:.0%} of op time is outside kernels "
                   f"({t.get('launch_count', 1)} launches): launch latency or gaps. "
                   "Fewer launches, or more work per launch, would recover it.")

    if c.get("available") and ideal > 0:
        moved = (c.get("dram_read_bytes") or 0) + (c.get("dram_write_bytes") or 0)
        ratio = moved / ideal
        if ratio > 1.3:
            out.append(f"DRAM moved {ratio:.2f}x the ideal bytes ({moved / 1e6:.1f} vs "
                       f"{ideal / 1e6:.1f} MB): uncoalesced, redundant or spilled access.")
        elif ratio < 0.7:
            out.append(f"DRAM moved only {ratio:.2f}x the ideal bytes: part of the data "
                       "came from L2, so the timing flatters this kernel.")
    if c.get("available") and c.get("occupancy") is not None and c["occupancy"] < 0.3:
        out.append(f"achieved occupancy {c['occupancy']:.0%}: too few warps to hide latency "
                   "(grid too small, or registers/shared memory limit residency).")

    if r.get("local", 0) > 0 or r.get("stack", 0) > 0 or b.get("local_mem", 0) > 0:
        out.append(f"{b.get('local_mem', 0)} local-memory instructions (LDL/STL) in SASS, "
                   f"{r.get('stack', 0)} B stack: register spills or a local array. Both go through "
                   "L1/L2 and cost bandwidth; shrink the per-thread working set.")

    if s.get("bound") == "memory" and frac < 0.7 and 0 < b.get("mean_global_load_bytes", 0) < 8:
        out.append(f"global loads average {b['mean_global_load_bytes']:.0f} B "
                   "(SASS). 128-bit loads cut load instructions 4x for fp32.")

    if b.get("global_atomic", 0) > 0 and s.get("bound") == "memory" and frac < 0.3:
        out.append(f"{b['global_atomic']} global atomic instruction(s) in SASS with low SOL: "
                   "if they run per element rather than per block, contention serializes them.")

    if obs.get("category") == "matmul" and b.get("tensor", 0) == 0:
        out.append("no tensor core instructions in SASS.")

    small = [p for p in obs.get("sizes", []) if p.get("bound") in ("latency", "launch")
             and p.get("fraction", 1) < 0.5]
    if small:
        worst = min(small, key=lambda p: p["fraction"])
        out.append(f"at {shape_label(worst['params'])} the kernel takes {1 / max(worst['fraction'], 1e-6):.1f}x "
                   f"the latency floor ({worst['op_ms'] * 1000:.1f} vs {worst['roof_ms'] * 1000:.1f} us): small "
                   "shapes pay for extra launches, block-wide syncs, idle threads and fixed setup, not bandwidth.")

    if frac >= 0.85:
        out.append("within 15% of the roof: further gains will be small.")
    return out + diagnose_noise(obs)


def diagnose_compute(obs):
    """The compute-bound reading: what feeds the FMA pipe, not what DRAM
    moved. Lines from counters appear only when arena collected them
    (--counters full for issue_active, stalls and pipes)."""
    out = []
    t = obs.get("timing", {})
    s = obs.get("sol", {})
    c = obs.get("counters", {}) or {}
    r = obs.get("resources", {}) or {}
    b = obs.get("sass", {}) or {}
    mix = obs.get("sass_mix") or {}
    ideal = obs.get("ideal_bytes") or 0
    frac = s.get("fraction", 0.0)

    line = (f"compute-bound: {frac:.1%} of the FP32 roof "
            f"({s.get('achieved', 0):.0f} of {s.get('roof', 0):.0f} GFLOP/s)")
    if s.get("fraction_at_clock") is not None and s.get("clock_mhz"):
        line += f", {s['fraction_at_clock']:.1%} at the measured {s['clock_mhz'] / 1000:.2f} GHz"
    out.append(line + ".")

    if mix:
        txt = f"FFMA is {mix['ffma_share']:.0%} of static SASS instructions"
        if mix.get("ffma_per_lds") is not None:
            txt += (f", {mix['ffma_per_lds']:.1f} FMAs per shared load ({mix['lds']} LDS, "
                    f"{mix['lds_128']} of them LDS.128)")
            if mix["ffma_per_lds"] < 4:
                txt += (": each shared load feeds few FMAs; a larger per-thread tile (8x8) "
                        "with 128-bit LDS raises it to 16")
        elif mix.get("ffma"):
            txt += ", no shared loads (operands from registers or global memory)"
        out.append(txt + ".")

    occ = c.get("occupancy") if c.get("available") else None
    if r.get("registers") or occ is not None:
        txt = f"{r.get('registers', '?')} registers"
        if occ is not None:
            txt += f", achieved occupancy {occ:.0%}"
        if occ is not None and occ < 0.3:
            txt += (": low occupancy is normal for a register-blocked kernel, as long as each "
                    "warp has independent FMAs to issue (check issue active and the stalls)")
        out.append(txt + ".")

    if c.get("issue_active") is not None:
        ia = c["issue_active"]
        txt = f"issue active {ia:.0%} of cycles"
        if ia < 0.6:
            txt += ": the schedulers idle often; the top stalls say on what"
        out.append(txt + ".")
    stalls = top_stalls(c, 3)
    if stalls:
        out.append("top stalls (warps per issue): " + ", ".join(f"{k} {v:.2f}" for k, v in stalls) + ".")
    pipes = c.get("pipes") or {}
    if pipes:
        top = sorted(((k, v) for k, v in pipes.items() if isinstance(v, (int, float))), key=lambda kv: -kv[1])[:4]
        out.append("pipe utilization: " + ", ".join(f"{k} {v:.0%}" for k, v in top) + ".")
    if c.get("shared_bank_conflicts"):
        wav = c.get("shared_wavefronts")
        out.append(f"{c['shared_bank_conflicts']:.3g} shared bank conflicts"
                   + (f" over {wav:.3g} shared wavefronts" if wav else "") + ".")

    if obs.get("tensor_forbidden"):
        out.append("tensor cores are forbidden in this run: the FP32 CUDA-core roof is the real limit.")
    elif obs.get("category") in ("matmul", "attention") and b.get("tensor", 0) == 0:
        out.append("no tensor core instructions in SASS.")
    elif b.get("tensor", 0):
        out.append(f"{b['tensor']} tensor core instructions in SASS: SOL above 100% is possible.")

    if r.get("local", 0) > 0 or r.get("stack", 0) > 0 or b.get("local_mem", 0) > 0:
        out.append(f"{b.get('local_mem', 0)} local-memory instructions (LDL/STL) in SASS, "
                   f"{r.get('stack', 0)} B stack: register spills in an FMA loop cost LSU issue slots.")

    if c.get("available") and ideal > 0:
        moved = (c.get("dram_read_bytes") or 0) + (c.get("dram_write_bytes") or 0)
        ratio = moved / ideal
        peak = s.get("peak_dram_gbps") or 0
        op = t.get("op_ms") or 0
        if ratio > 1.3:
            txt = (f"DRAM moved {ratio:.1f}x the ideal bytes ({moved / 1e6:.0f} vs {ideal / 1e6:.0f} MB), "
                   "but the kernel is compute-bound")
            if peak > 0 and op > 0:
                need = moved / (peak * 1e6)
                txt += f": that traffic needs {need:.2f} of the {op:.2f} ms at peak bandwidth"
            txt += (", so it matters only if it nears the op time or long_scoreboard stalls show "
                    "the FMAs waiting on L2/DRAM")
            out.append(txt + ".")

    op, gpu = t.get("op_ms") or 0, t.get("gpu_ms") or 0
    if op > 0 and (op - gpu) / op > 0.2:
        out.append(f"{(op - gpu) / op:.0%} of op time is outside kernels "
                   f"({t.get('launch_count', 1)} launches): launch latency or gaps.")
    if frac >= 0.85:
        out.append("within 15% of the roof: further gains will be small.")
    return out


def diagnose_noise(obs):
    """Lines about timing spread: two clusters, or many outliers, mean a
    single eval cannot resolve a small change."""
    out = []
    for p in obs.get("sizes", []) or []:
        sp = p.get("spread") or {}
        if sp.get("bimodal"):
            cl = sp.get("clusters") or {}
            out.append(f"timings at {shape_label(p.get('params'))} are bimodal: {cl.get('low_share', 0):.0%} "
                       f"of runs near {cl.get('low_ms', 0):.4f} ms, the rest near {cl.get('high_ms', 0):.4f} ms "
                       "(clock, power or placement changed between runs). Compare candidates with "
                       "`pgaio.py ab` rather than one eval.")
        elif sp.get("outliers", 0) >= 0.1 and sp.get("ci_half_frac", 0) * 2 > PLATEAU_GAIN:
            out.append(f"{sp['outliers']:.0%} of timed runs at {shape_label(p.get('params'))} are outliers "
                       f"({sp.get('min', 0):.4f}..{sp.get('max', 0):.4f} ms around a median of "
                       f"{sp.get('median', 0):.4f}); the score uses the interquartile mean, but changes "
                       f"under {sp.get('ci_half_frac', 0) * 2:.1%} here are within noise.")
    return out


def sass_delta(cur, ref):
    """Changed SASS counts, current minus reference."""
    keys = ["instructions", "global_loads", "global_load_128", "global_atomic",
            "warp_shuffle", "warp_reduce", "shared_mem", "sync", "tensor", "local_mem"]
    out = {}
    for k in keys:
        d = (cur or {}).get(k, 0) - (ref or {}).get(k, 0)
        if d:
            out[k] = d
    return out


BLOCK_DIRECTIVE = re.compile(r"arena:\s*block\s*=\s*(\d+)")


def block_directive(source_text):
    """A candidate may pick its own block size with a comment such as
    `// arena: block=512`, for slots whose block size is tunable. The slot's
    descriptor still decides whether that size is allowed."""
    m = BLOCK_DIRECTIVE.search(source_text or "")
    return int(m.group(1)) if m else None


def slugify(text, limit=40):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return (s[:limit].rstrip("-")) or "candidate"


def candidate_ext(path):
    """".triton.py" for x.triton.py, ".cu" for x.cu."""
    name = Path(path).name
    dot = name.find(".")
    return name[dot:] if dot >= 0 else ""


def dsl_of(path):
    """"cuda" for x.cu, "triton" for x.triton.py, None otherwise."""
    return DSL_EXT.get(candidate_ext(path))


def budget_error(cfg, attempts, dsl):
    """Why another iteration in this DSL is not allowed, or None.

    Baselines (iteration 0) do not count against any budget."""
    if cfg.get("kind") == "problem":
        allowed = cfg.get("dsls", DSLS)
        if dsl not in allowed:
            return f"{dsl} is not allowed in this run (allowed: {', '.join(allowed)})"
    used = [x for x in attempts if x.get("iteration", 0) > 0]
    if cfg.get("budget") and len(used) >= cfg["budget"]:
        return f"the run's budget of {cfg['budget']} iterations is used up"
    per = cfg.get("budget_per_dsl") or 0
    if per and sum(1 for x in used if x.get("dsl") == dsl) >= per:
        return f"the {dsl} budget of {per} iterations is used up; try another DSL"
    return None


# A run with no budget ends on a plateau: this many evals in a row without
# raising the best score by GAIN, after every allowed DSL had MIN_PER_DSL.
PLATEAU_EVALS = 10
PLATEAU_GAIN = 0.005
PLATEAU_MIN_PER_DSL = 3
# Runs started from now on record "plateau_rule": 2 (see plateau()). Runs
# without the key keep rule 1, so their history reads the same as before.
PLATEAU_RULE = 2


def score_half_width(att):
    """Half the width of an attempt's 95% score CI, in score units, or 0."""
    ci = att.get("score_ci") if att else None
    if not ci or len(ci) != 2 or ci[0] is None or ci[1] is None:
        return 0.0
    return max(0.0, (ci[1] - ci[0]) / 2)


def ab_says_faster(ab_results, new_it, old_it, shapes=None):
    """Whether a saved A/B comparison shows iteration new_it faster than
    old_it over all its shapes, with a CI that excludes no difference.
    With shapes (the run's timed shapes), only an A/B that timed every one
    of them counts: the score is a mean over all of them, so a gain at one
    shape (ab --shape) is not a gain in the score."""
    need = {pkey(s) for s in shapes or []}
    for r in ab_results or []:
        o = r.get("overall") or {}
        timed = {pkey(s.get("params")) for s in r.get("shapes") or [] if isinstance(s, dict)}
        if need - timed:
            continue
        if r.get("a") == old_it and r.get("b") == new_it and o.get("faster") == "b":
            return True
        if r.get("a") == new_it and r.get("b") == old_it and o.get("faster") == "a":
            return True
    return False


def plateau(cfg, attempts, ab_results=None):
    """(evals since the best last rose by a real gain, DSLs still short of
    PLATEAU_MIN_PER_DSL evals and not skipped, whether the run may stop on a
    plateau).

    Rule 1 (runs without "plateau_rule"): a gain is a score at least
    PLATEAU_GAIN above the best. Rule 2: the score must clear the best by
    max(PLATEAU_GAIN, half-width of its CI + half-width of the best's CI),
    i.e. the two 95% intervals must not overlap, which is conservative for a
    difference of two independent estimates. Timing spread on long kernels
    (1-3% interquartile) is larger than PLATEAU_GAIN, so rule 1 counted noise
    as progress. A saved `ab` result showing the iteration faster than the
    best of its time over all the timed shapes, with a CI excluding zero,
    also counts as a gain."""
    rule = cfg.get("plateau_rule", 1)
    timed = cfg.get("train_shapes")
    best, best_att, since = None, None, 0
    for x in attempts:
        if x.get("iteration", 0) == 0:
            if x.get("verdict") == "ok" and x.get("score") is not None:
                best, best_att = x["score"], x
            continue
        sc = x.get("score") if x.get("verdict") == "ok" else None
        if rule >= 2:
            margin = max(PLATEAU_GAIN, score_half_width(x) + score_half_width(best_att))
            gain = sc is not None and (best is None or sc > best + margin)
            if sc is not None and not gain and best_att is not None:
                gain = ab_says_faster(ab_results, x.get("iteration"), best_att.get("iteration"), timed)
        else:
            gain = sc is not None and (best is None or sc >= best + PLATEAU_GAIN)
        if gain:
            best, best_att, since = max(sc, best or 0), x, 0
        else:
            if sc is not None and sc > (best or 0):
                best, best_att = sc, x
            since += 1
    allowed = [d for d in cfg.get("dsls", []) if d != "mixed"] if cfg.get("kind") == "problem" else []
    skipped = {s.get("dsl") for s in cfg.get("skipped_dsls", [])}
    per = {}
    for x in attempts:
        if x.get("iteration", 0) > 0:
            per[x.get("dsl")] = per.get(x.get("dsl"), 0) + 1
    short = [d for d in allowed if per.get(d, 0) < PLATEAU_MIN_PER_DSL and d not in skipped]
    return since, short, since >= PLATEAU_EVALS and not short


def plateau_text(cfg):
    """How plateau() counts a gain, for status and eval output."""
    if cfg.get("plateau_rule", 1) >= 2:
        return f"the best last rose beyond its noise (CI) and {PLATEAU_GAIN:.1%}"
    return f"the best last rose by {PLATEAU_GAIN:.1%}"


# --------------------------------------------------------------------------
# Timing spread. Arena reports every timed run in timing.all_times_ms; the
# score uses their interquartile mean. These helpers say how far that mean
# could move on a rerun, so a 0.5% "gain" can be told from noise.
# --------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261016
# Two timing clusters: a gap between consecutive sorted times wider than this
# fraction of the median, with at least BIMODAL_SIDE of the runs on each side.
# Clock or power state changes give gaps of 10% or more; the largest gap
# inside one cluster of a 100-run matmul timing was under 3%.
BIMODAL_GAP = 0.04
BIMODAL_SIDE = 0.15


def trimmed_mean(values):
    """Interquartile mean, the same cut as arena's trimmed_mean(): drop the
    lowest and highest n/4 values."""
    v = sorted(values)
    n = len(v)
    if not n:
        return 0.0
    lo, hi = n // 4, n - n // 4
    if hi <= lo:
        lo, hi = 0, n
    return sum(v[lo:hi]) / (hi - lo)


def bimodal_gap(sorted_times, median):
    """The widest gap splitting the runs into two clusters that each hold at
    least BIMODAL_SIDE of them, if it exceeds BIMODAL_GAP of the median."""
    n = len(sorted_times)
    best = None
    for i in range(n - 1):
        left = i + 1
        if left < BIMODAL_SIDE * n or n - left < BIMODAL_SIDE * n:
            continue
        gap = sorted_times[i + 1] - sorted_times[i]
        if gap > BIMODAL_GAP * median and (best is None or gap > best[0]):
            best = (gap, i)
    if best is None:
        return None
    gap, i = best
    low, high = sorted_times[:i + 1], sorted_times[i + 1:]
    return {"gap_ms": gap, "low_ms": statistics.median(low), "high_ms": statistics.median(high),
            "low_share": len(low) / n}


def spread(times, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Dispersion of one shape's timed runs: quartiles, interquartile range
    as a fraction of the median, a bootstrap 95% CI of the interquartile mean
    (deterministic), the share of runs outside the 1.5 IQR fences, and a
    second cluster if there is one. None with fewer than 4 runs."""
    import random
    ts = sorted(float(t) for t in times or [] if t is not None)
    n = len(ts)
    if n < 4:
        return None
    q1, med, q3 = statistics.quantiles(ts, n=4)
    iqr = q3 - q1
    rng = random.Random(seed)
    means = sorted(trimmed_mean(rng.choices(ts, k=n)) for _ in range(resamples))
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(round(0.975 * (resamples - 1)))]
    tm = trimmed_mean(ts)
    out_lo, out_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    gap = bimodal_gap(ts, med)
    return {
        "n": n, "min": ts[0], "q1": q1, "median": med, "q3": q3, "max": ts[-1],
        "trimmed_ms": tm,
        "iqr_frac": iqr / med if med > 0 else 0.0,
        "ci_ms": [lo, hi],
        "ci_half_frac": (hi - lo) / 2 / tm if tm > 0 else 0.0,
        "outliers": sum(1 for t in ts if t < out_lo or t > out_hi) / n,
        "bimodal": gap is not None,
        "clusters": gap,
    }


def score_ci(sizes):
    """95% CI of the score: every shape at the slow end of its op time CI,
    then every shape at the fast end. Treating the shapes as moving together
    makes the interval a little wider than an independent combination.
    None when no shape has a CI (timings without all_times_ms, or fewer
    than 4 runs): a zero-width interval would claim no noise at all."""
    if not sizes:
        return None
    lows, highs, any_ci = [], [], False
    for p in sizes:
        roof = p.get("roof_ms") or 0
        ci = (p.get("spread") or {}).get("ci_ms")
        if not ci or roof <= 0 or ci[0] <= 0:
            lows.append(p.get("fraction", 0.0))
            highs.append(p.get("fraction", 0.0))
            continue
        any_ci = True
        lows.append(roof / ci[1])
        highs.append(roof / ci[0])
    return [geomean(lows), geomean(highs)] if any_ci else None


def add_spread(per, results):
    """Puts each shape's spread into the per-shape SOL rows (same order as
    the timing results) and returns the score CI."""
    for p, r in zip(per, results):
        p["spread"] = spread(r.get("timing", {}).get("all_times_ms"))
    return score_ci(per)


def fmt_ci(ci):
    return f"95% CI {ci[0]:.1%}..{ci[1]:.1%}" if ci else ""


def add_clock(per, results, env):
    """Clock and power during each shape's timed runs, when arena sampled
    them, and for compute-bound shapes the SOL at the measured clock: the
    FP32 peak assumes env["peak_fp32_gflops_at_mhz"], and a card running
    slower than that cannot reach it. The score stays on the nominal roof."""
    at = float(env.get("peak_fp32_gflops_at_mhz") or 0)
    for p, r in zip(per, results):
        q = r.get("quality") or {}
        clock = {k: q[k] for k in ("sm_clock_mean_mhz", "sm_clock_min_mhz", "sm_clock_max_mhz",
                                   "power_mean_w", "power_max_w", "clock_samples") if q.get(k) is not None}
        if clock:
            p["clock"] = clock
        mean = float(q.get("sm_clock_mean_mhz") or 0)
        if p.get("bound") == "compute" and at > 0 and mean > 0:
            p["fraction_at_clock"] = p.get("fraction", 0.0) * at / mean


def clock_text(p):
    """", 53.0% at the measured 2.35 GHz" for a shape with fraction_at_clock."""
    c = p.get("clock") or {}
    if p.get("fraction_at_clock") is None or not c.get("sm_clock_mean_mhz"):
        return ""
    return f", {p['fraction_at_clock']:.1%} at the measured {c['sm_clock_mean_mhz'] / 1000:.2f} GHz"


def power_text(p):
    c = p.get("clock") or {}
    if not c:
        return ""
    out = []
    if c.get("sm_clock_mean_mhz"):
        out.append(f"clock {c['sm_clock_mean_mhz']:.0f} MHz"
                   + (f" ({c.get('sm_clock_min_mhz', 0):.0f}..{c.get('sm_clock_max_mhz', 0):.0f})"
                      if c.get("sm_clock_min_mhz") is not None and c.get("sm_clock_max_mhz") is not None else ""))
    if c.get("power_mean_w"):
        out.append(f"{c['power_mean_w']:.0f} W" + (f" (max {c['power_max_w']:.0f})" if c.get("power_max_w") else ""))
    return ", ".join(out)


# --------------------------------------------------------------------------
# What the SASS says about a compute-bound kernel
# --------------------------------------------------------------------------

# Tensor-core instruction families. The arena classifier puts these in its
# "tensor" class; the list is repeated here so a run forbidding tensor cores
# also catches a family the classifier does not know yet (anything ending in
# MMA, e.g. a future UTCxMMA).
TENSOR_OPS = {"HMMA", "IMMA", "QMMA", "OMMA", "DMMA", "BMMA", "HGMMA", "IGMMA", "QGMMA",
              "BGMMA", "UTCHMMA", "UTCQMMA", "UTCOMMA", "UTCIMMA", "UTCMMA"}
SASS_MNEMONIC = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")


def is_tensor_op(opcode):
    base = opcode.split(".", 1)[0]
    return base in TENSOR_OPS or base.endswith("MMA")


def tensor_ops(insp, sass_text=""):
    """{opcode base: count} of tensor-core instructions, from the opcode
    counts of --inspect and, failing those, the SASS listing."""
    out = {}
    for op, n in ((insp or {}).get("sass", {}).get("opcodes") or {}).items():
        if is_tensor_op(op):
            base = op.split(".", 1)[0]
            out[base] = out.get(base, 0) + n
    if not out and sass_text:
        for m in SASS_MNEMONIC.finditer(sass_text):
            if is_tensor_op(m.group(1)):
                base = m.group(1).split(".", 1)[0]
                out[base] = out.get(base, 0) + 1
    if not out and sass_brief(insp)["tensor"]:
        out["tensor"] = sass_brief(insp)["tensor"]
    return out


def tensor_rejection(ops):
    return ("tensor-core instructions in SASS (" + ", ".join(f"{k} x{v}" for k, v in sorted(ops.items()))
            + ") but this run forbids tensor cores (init --forbid-tensor-cores); "
            "the candidate was not measured")


def sass_mix(insp):
    """Static FMA balance of a kernel: FFMA share of all SASS instructions
    and FFMAs per shared-memory load. Static counts over every entry, so a
    loop body that runs K/8 times weighs the same as setup code; still the
    ratio inside the unrolled main loop dominates for a register-blocked
    matmul. FFMA2 (two FMAs per instruction) counts twice in fma_ops."""
    ops = (insp or {}).get("sass", {}).get("opcodes") or {}
    total = (insp or {}).get("sass", {}).get("instructions") or sum(ops.values())
    ffma = sum(n for op, n in ops.items() if op.split(".", 1)[0] in ("FFMA", "FFMA2"))
    fma_ops = sum(n * (2 if op.split(".", 1)[0] == "FFMA2" else 1) for op, n in ops.items()
                  if op.split(".", 1)[0] in ("FFMA", "FFMA2"))
    lds = sum(n for op, n in ops.items() if op.split(".", 1)[0] in ("LDS", "LDSM"))
    lds128 = sum(n for op, n in ops.items() if op.split(".", 1)[0] == "LDS" and ".128" in op)
    if not ops:
        return {}
    return {"instructions": total, "ffma": ffma, "fma_ops": fma_ops, "lds": lds, "lds_128": lds128,
            "ffma_share": ffma / total if total else 0.0,
            "ffma_per_lds": fma_ops / lds if lds else None}


def top_stalls(counters, k=3):
    """[(reason, warps stalled per issue)], largest first."""
    st = (counters or {}).get("stalls") or {}
    return sorted(((r, v) for r, v in st.items() if isinstance(v, (int, float))),
                  key=lambda kv: -kv[1])[:k]


def compute_bound(obs):
    """Whether the observation should be read as compute-bound: the largest
    shape's roofline says so, or its per-shape SOL row does."""
    if (obs.get("sol") or {}).get("bound") == "compute":
        return True
    sizes = obs.get("sizes") or []
    if sizes:
        big = max(sizes, key=lambda p: (p.get("ideal_bytes") or 0) + (p.get("flops") or 0))
        return big.get("bound") == "compute"
    return False


# --------------------------------------------------------------------------
# Library references (written by pgaio_ncu.py reference) and A/B results
# --------------------------------------------------------------------------

def pkey(params):
    return ",".join(f"{k}={v}" for k, v in sorted((params or {}).items()))


def ref_relative(ref, sizes):
    """(geomean of ref op_ms / candidate op_ms over the shapes both have,
    {shape key: ratio}). 1.0 means as fast as the library."""
    ref_ms = {pkey(s.get("params")): s.get("op_ms") for s in (ref or {}).get("sizes", [])}
    per = {}
    for p in sizes or []:
        r, c = ref_ms.get(pkey(p.get("params"))), p.get("op_ms")
        if r and c:
            per[pkey(p.get("params"))] = r / c
    return (geomean(list(per.values())) if per else None), per


def ref_text(refs, sizes):
    """"85.0% of cublas" for every reference that shares a shape."""
    out = []
    for lib, ref in sorted((refs or {}).items()):
        g, _ = ref_relative(ref, sizes)
        if g is not None:
            out.append(f"{g:.1%} of {lib}")
    return ", ".join(out)


# Two-sided 97.5% quantiles of Student's t, by degrees of freedom.
T975 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def paired_ratio(a_ms, b_ms):
    """B/A from paired per-round times: the geometric mean of the per-round
    ratios, with a 95% t interval on their logs. Pairing within a round
    cancels drift that hits both candidates alike (clocks, temperature).
    Returns {"ratio", "ci": [lo, hi] or None, "rounds"}."""
    import math
    logs = [math.log(b / a) for a, b in zip(a_ms, b_ms) if a and b and a > 0 and b > 0]
    n = len(logs)
    if not n:
        return {"ratio": None, "ci": None, "rounds": 0}
    mean = sum(logs) / n
    if n < 2:
        return {"ratio": math.exp(mean), "ci": None, "rounds": n}
    sd = statistics.stdev(logs)
    t = T975[n - 1] if n - 1 < len(T975) else 1.96
    hw = t * sd / math.sqrt(n)
    return {"ratio": math.exp(mean), "ci": [math.exp(mean - hw), math.exp(mean + hw)], "rounds": n}


def ab_verdict(res):
    """("b" | "a" | None, text) for one paired_ratio result. B/A below 1
    with the whole CI below 1 means B is faster."""
    r, ci = res.get("ratio"), res.get("ci")
    if r is None:
        return None, "no data"
    if ci and ci[1] < 1:
        return "b", f"B faster by {1 - r:.1%} (CI {1 - ci[1]:.1%}..{1 - ci[0]:.1%})"
    if ci and ci[0] > 1:
        return "a", f"A faster by {1 - 1 / r:.1%} (CI {1 - 1 / ci[0]:.1%}..{1 - 1 / ci[1]:.1%})"
    span = f", CI {ci[0]:.3f}..{ci[1]:.3f}" if ci else ", too few rounds for a CI"
    return None, f"no difference resolved (B/A {r:.3f}{span})"


def ab_summary(rounds, shapes):
    """Per shape and overall verdicts from per-round trimmed means.
    rounds: [{"a": {shape key: ms}, "b": {shape key: ms}}]. Overall is the
    per-round geometric mean over shapes, which is how the score combines
    them."""
    import math
    out = {"shapes": [], "overall": None}
    keys = [pkey(s) for s in shapes]
    for s, k in zip(shapes, keys):
        res = paired_ratio([r["a"].get(k) for r in rounds], [r["b"].get(k) for r in rounds])
        res["faster"], res["verdict"] = ab_verdict(res)
        out["shapes"].append({"params": s, **res})
    ga, gb = [], []
    for r in rounds:
        if all(r["a"].get(k) and r["b"].get(k) for k in keys):
            ga.append(math.exp(sum(math.log(r["a"][k]) for k in keys) / len(keys)))
            gb.append(math.exp(sum(math.log(r["b"][k]) for k in keys) / len(keys)))
    res = paired_ratio(ga, gb)
    res["faster"], res["verdict"] = ab_verdict(res)
    out["overall"] = res
    return out


def ab_round_order(rounds):
    """["ab", "ba", "ab", ...]: alternating, so neither candidate always runs
    on a GPU the other one just warmed."""
    return ["ab" if i % 2 == 0 else "ba" for i in range(rounds)]


def parse_shape_spec(spec):
    """"K=63,M=129" -> {"K": 63, "M": 129}"""
    out = {}
    for item in spec.split(","):
        k, _, v = item.partition("=")
        if not k.strip() or not v.strip():
            raise ValueError(f"malformed shape '{spec}' (expected key=n,key=n)")
        out[k.strip()] = int(v)
    return out


# --------------------------------------------------------------------------
# Launch plans (arena --plan)
# --------------------------------------------------------------------------

def unknown_option(stderr):
    """The flag an arena build rejected as unknown, or None."""
    m = re.search(r"unknown option '([^']+)'", stderr or "")
    return m.group(1) if m else None


def plan_error(data):
    """The first error over all shapes of a --plan result, with its shape."""
    for p in (data or {}).get("plans", []):
        if p.get("error"):
            return f"{shape_label(p.get('params'))}: {p['error']}"
    return ""


def fmt_dims(d):
    return "x".join(str(x) for x in d) if isinstance(d, (list, tuple)) else str(d)


def render_plan(data):
    """One block per shape: the launches that apply, with grid, block and
    dynamic shared memory, and which ones are skipped by their "when"."""
    lines = []
    for p in (data or {}).get("plans", []):
        lines.append(f"{shape_label(p.get('params'))}:")
        if p.get("error"):
            lines.append(f"  ERROR {p['error']}")
        used = [l for l in p.get("launches", []) if l.get("applies", True)]
        skipped = [str(l.get("index")) for l in p.get("launches", []) if not l.get("applies", True)]
        for l in used:
            what = l.get("entry") or Path(l.get("source") or "").name or "?"
            lines.append(f"  #{l.get('index')} {what:22} grid {fmt_dims(l.get('grid')):12} "
                         f"block {fmt_dims(l.get('block')):9} smem {l.get('shared_mem_bytes', 0)}")
        if p.get("launches") and not used and not p.get("error"):
            lines.append("  no launch applies at this shape")
        if skipped:
            lines.append(f"  not applied: {', '.join(skipped)}")
        if p.get("scratch"):
            lines.append("  scratch: " + ", ".join(f"{s.get('name')} {s.get('bytes')} B" for s in p["scratch"]))
    return lines


def plan_summary(data):
    """One line for a plan with no error: which launches run at how many
    shapes."""
    plans = (data or {}).get("plans", [])
    use = {}
    for p in plans:
        for l in p.get("launches", []):
            if l.get("applies", True):
                use[l.get("index")] = use.get(l.get("index"), 0) + 1
    return (f"launch plan ok at {len(plans)} shapes ("
            + ", ".join(f"#{i} at {n}" for i, n in sorted(use.items(), key=lambda kv: str(kv[0]))) + ")")


def best_per_dsl(attempts):
    """Best passing score per DSL, over its own iterations. A DSL with no
    passing attempt is absent rather than reported as 0%."""
    out = {}
    for a in attempts:
        if a.get("iteration", 0) == 0 or a.get("verdict") != "ok" or a.get("score") is None:
            continue
        out[a["dsl"]] = max(out.get(a["dsl"], 0.0), a["score"])
    return out


def per_dsl_curves(attempts):
    """Best-so-far per DSL, over that DSL's own iterations only (the baseline
    excluded), so "iterations to 80%" means attempts spent in that language."""
    out = {}
    for a in attempts:
        if a.get("iteration", 0) == 0 or not a.get("dsl"):
            continue
        out.setdefault(a["dsl"], []).append(a)
    return {d: best_so_far(xs) for d, xs in out.items()}


def summarize_run(run_cfg, attempts):
    """One row of the report."""
    curve = best_so_far(attempts)
    row = {
        "run": run_cfg.get("name"),
        "kernel": run_cfg.get("kernel"),
        "dsl": run_cfg.get("dsl"),
        "mode": run_cfg.get("mode", "isolated"),
        "docs": bool(run_cfg.get("docs", False)),
        "category": run_cfg.get("category"),
        "device": run_cfg.get("device"),
        "model": run_cfg.get("model", ""),
        "iterations": max(0, len(attempts) - 1),
        "failed": sum(1 for a in attempts if a.get("iteration", 0) > 0
                      and a.get("verdict") in ("compile_error", "runtime_error")),
        "skipped_dsls": [x.get("dsl") for x in run_cfg.get("skipped_dsls", [])],
        "baseline_sol": curve[0] if curve else 0.0,
        "best_sol": curve[-1] if curve else 0.0,
        "curve": curve,
    }
    for th in SOL_THRESHOLDS:
        row[f"iters_to_{int(th * 100)}"] = iterations_to(curve, th)
    return row


def summarize_dsls(run_cfg, attempts):
    """For a problem run, one extra row per DSL tried: its own curve over its
    own iterations. curve[0] is that DSL's first attempt, so iterations_to
    counts from 1 here (an attempt that hits 80% at once took 1 iteration)."""
    if run_cfg.get("kind") != "problem":
        return []
    base = summarize_run(run_cfg, attempts)
    rows = []
    for dsl, curve in sorted(per_dsl_curves(attempts).items()):
        own = [a for a in attempts if a.get("iteration", 0) > 0 and a.get("dsl") == dsl]
        row = {**base, "run": f"{base['run']}:{dsl}", "dsl": dsl, "kernel": base["category"],
               "iterations": len(curve), "baseline_sol": base["baseline_sol"],
               "failed": sum(1 for a in own if a.get("verdict") in ("compile_error", "runtime_error")),
               "skipped_dsls": [d for d in base["skipped_dsls"] if d == dsl],
               "best_sol": curve[-1] if curve else 0.0, "curve": curve}
        for th in SOL_THRESHOLDS:
            row[f"iters_to_{int(th * 100)}"] = iterations_to(curve, th, start_iteration=1)
        rows.append(row)
    return rows


def group_key(row):
    """"cuda/isolated", or "cuda/isolated+docs" when documentation tools were
    available. Runs are only comparable within one group."""
    key = f"{row['dsl']}/{row.get('mode', 'isolated')}"
    key = key + "+docs" if row.get("docs") else key
    # Iterations-to-SOL only compares within one problem.
    return f"{row['category']}: {key}" if row.get("category") else key


def aggregate(rows):
    """Medians over runs, per DSL, mode and docs access. A threshold never
    reached counts as missing and is reported as how many runs reached it."""
    by = {}
    for r in rows:
        by.setdefault(group_key(r), []).append(r)
    out = {}
    for dsl, rs in sorted(by.items()):
        agg = {"runs": len(rs),
               "median_best_sol": statistics.median(r["best_sol"] for r in rs),
               "median_baseline_sol": statistics.median(r["baseline_sol"] for r in rs)}
        for th in SOL_THRESHOLDS:
            key = f"iters_to_{int(th * 100)}"
            hit = [r[key] for r in rs if r[key] is not None]
            agg[key] = statistics.median(hit) if hit else None
            agg[key + "_reached"] = f"{len(hit)}/{len(rs)}"
        out[dsl] = agg
    return out


DSL_COLORS = {"cuda": "#3f9d2f", "triton": "#d9822b", "cutile": "#2b7bd9", "warp": "#8e44ad"}


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_run_svg(cfg, attempts, width=760, height=300):
    """One run's progression: every iteration as a point coloured by DSL,
    failures as crosses on the floor, the best-so-far as a step line, and the
    target as a dashed rule. Pure string building, unit-tested."""
    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 34
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = max(1, len(attempts) - 1)
    scores = [a["score"] for a in attempts if a.get("score") is not None]
    top = max([1.0] + scores)
    x = lambda i: pad_l + w * i / n
    y = lambda v: pad_t + h * (1 - v / top)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="SOL per iteration for {_esc(cfg.get("name", ""))}">']
    for v in (0.0, 0.5, 0.7, 0.9, 1.0):
        if v <= top:
            out.append(f'<line class="grid" x1="{pad_l}" x2="{pad_l + w}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                       f'<text class="axis" x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    t = cfg.get("target")
    if t:
        out.append(f'<line class="target" x1="{pad_l}" x2="{pad_l + w}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="axis" x="{pad_l + w}" y="{y(t) - 5:.1f}" text-anchor="end">target {t:.0%}</text>')
    step = max(1, n // 10)
    for i in range(0, n + 1, step):
        out.append(f'<text class="axis" x="{x(i):.1f}" y="{pad_t + h + 18}" text-anchor="middle">{i}</text>')
    out.append(f'<text class="axis" x="{pad_l + w / 2:.1f}" y="{height - 2}" text-anchor="middle">iteration</text>')

    curve = best_so_far(attempts)
    pts = []
    for i, v in enumerate(curve):
        if i > 0:
            pts.append(f"{x(i):.1f},{y(curve[i - 1]):.1f}")
        pts.append(f"{x(i):.1f},{y(v):.1f}")
    out.append(f'<polyline class="best" points="{" ".join(pts)}"/>')

    for i, a in enumerate(attempts):
        color = DSL_COLORS.get(a.get("dsl"), "#888")
        tip = _esc(f"{i} · {a.get('dsl') or ''} · {a.get('verdict')}"
                   + (f" · {a['score']:.1%}" if a.get("score") is not None else "")
                   + f" · {a.get('hypothesis', '')}")
        if a.get("score") is not None:
            r = 6 if a.get("new_best") else 4
            out.append(f'<circle cx="{x(i):.1f}" cy="{y(a["score"]):.1f}" r="{r}" fill="{color}">'
                       f'<title>{tip}</title></circle>')
        else:
            cx, cy = x(i), pad_t + h - 6
            out.append(f'<g stroke="{color}" stroke-width="2"><title>{tip}</title>'
                       f'<line x1="{cx - 4:.1f}" y1="{cy - 4:.1f}" x2="{cx + 4:.1f}" y2="{cy + 4:.1f}"/>'
                       f'<line x1="{cx - 4:.1f}" y1="{cy + 4:.1f}" x2="{cx + 4:.1f}" y2="{cy - 4:.1f}"/></g>')
    out.append("</svg>")
    return "".join(out)


def render_html(runs):
    """The progression page for a list of (run config, attempts)."""
    sections = []
    for cfg, attempts in runs:
        curve = best_so_far(attempts)
        dsl_best = best_per_dsl(attempts)
        legend = "".join(f'<span class="chip"><i style="background:{DSL_COLORS.get(d, "#888")}"></i>'
                         f'{_esc(d)}' + (f" best {dsl_best[d]:.1%}" if d in dsl_best else "") + "</span>"
                         for d in (cfg.get("dsls") or [cfg.get("dsl")]))
        rows, prev_best = [], None
        for a in attempts:
            delta = ""
            if a.get("score") is not None and prev_best is not None and a.get("sass") and prev_best.get("sass"):
                d = sass_delta(a["sass"], prev_best["sass"])
                delta = ", ".join(f"{k} {v:+d}" for k, v in d.items())
            if a.get("score") is not None and (prev_best is None or a["score"] > prev_best["score"]):
                prev_best = a
            sc = f"{a['score']:.1%}" if a.get("score") is not None else "-"
            op = f"{a['op_ms']:.4f}" if a.get("op_ms") else "-"
            rows.append(
                f'<tr class="{"best" if a.get("new_best") else ""} {a.get("verdict")}">'
                f'<td>{a["iteration"]}</td>'
                f'<td><i class="dot" style="background:{DSL_COLORS.get(a.get("dsl"), "#888")}"></i>{_esc(a.get("dsl") or "")}</td>'
                f'<td>{_esc(a.get("verdict"))}</td><td class="num">{sc}</td><td class="num">{op}</td>'
                f'<td>{_esc(a.get("hypothesis", ""))}</td><td class="mono">{_esc(delta)}</td></tr>')
            body = a.get("diff") or a.get("code")
            if body:
                label = (f"diff against iteration {a['diff_against']}" if a.get("diff")
                         else f"source {_esc(Path(a.get('source_file', '')).name)}")
                rows.append(f'<tr class="code"><td></td><td colspan="6"><details><summary>{label}</summary>'
                            f'<pre>{_esc(body)}</pre></details></td></tr>')
        what = cfg["category"] if cfg.get("kind") == "problem" else cfg.get("kernel")
        sections.append(f"""
<section>
  <h2>{_esc(what)} <span class="muted">{_esc(cfg.get("name", ""))}</span></h2>
  <p class="meta">{_esc(cfg.get("device", ""))} · mode {_esc(cfg.get("mode", "isolated"))} ·
    docs {"on" if cfg.get("docs") else "off"} · model {_esc(cfg.get("model") or "unrecorded")} ·
    {max(0, len(attempts) - 1)} iterations · baseline {curve[0]:.1%} → best {curve[-1]:.1%}</p>
  <div class="legend">{legend}</div>
  <div class="chart">{render_run_svg(cfg, attempts)}</div>
  <div class="tablewrap"><table>
    <thead><tr><th>#</th><th>DSL</th><th>verdict</th><th>SOL</th><th>op ms</th>
      <th>hypothesis</th><th>SASS vs previous best</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PGAIO Progression</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4; --panel:#ffffff; --best:#f0fdf4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --line:#44403c; --panel:#292524; --best:#1a2e1f; }} }}
:root[data-theme="dark"] {{ --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --line:#44403c; --panel:#292524; --best:#1a2e1f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:0; }}
.muted, .meta {{ color:var(--muted); font-weight:400; }} .meta {{ margin:4px 0 10px; font-size:13px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; margin-top:20px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }}
.chip {{ font-size:13px; border:1px solid var(--line); border-radius:999px; padding:2px 10px; }}
.chip i, .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
.chart svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--line); }} .target {{ stroke:var(--muted); stroke-dasharray:4 4; }}
.axis {{ fill:var(--muted); font-size:11px; }}
.best {{ fill:none; stroke:var(--fg); stroke-width:1.5; opacity:.55; }}
.tablewrap {{ overflow-x:auto; margin-top:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
td:nth-child(2), td:nth-child(3) {{ white-space:nowrap; }}
th {{ color:var(--muted); font-weight:500; }}
tr.best td {{ background:var(--best); }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.mono {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:var(--muted); }}
tr.code td {{ border-bottom:1px solid var(--line); padding-top:0; }}
summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
pre {{ margin:6px 0 0; padding:10px; background:var(--bg); border:1px solid var(--line); border-radius:6px;
  font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-x:auto; max-height:480px; }}
</style></head>
<body><main>
<h1>PGAIO progression</h1>
<p class="muted">SOL is achieved throughput against the roofline at each kernel's arithmetic
intensity. Large points are new bests, crosses failed (compile, correctness or guard).
Hover a point for its hypothesis.</p>
{"".join(sections) or "<p>No runs yet.</p>"}
</main></body></html>
"""


# --------------------------------------------------------------------------
# Side effects
# --------------------------------------------------------------------------

def now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def hash_protected():
    out = {}
    for root, dirs, files in os.walk(REPO):
        rel_root = os.path.relpath(root, REPO)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if not d.startswith(".") and
                       not d.startswith("build") and d not in ("pgaio", "skills")]
        for f in files:
            rel = os.path.normpath(os.path.join(rel_root, f))
            if not is_protected(rel):
                continue
            with open(os.path.join(root, f), "rb") as fh:
                out[rel] = hashlib.sha1(fh.read()).hexdigest()
    return out


# After a timeout arena gets SIGTERM first: its handler restores locked
# clocks before exiting, and docker compose forwards the signal to the
# container. SIGKILL only if it is still running this many seconds later.
TERM_GRACE_S = 15


def run_with_timeout(argv, cwd, timeout, grace=TERM_GRACE_S):
    """(exit code, stdout, stderr) of argv. On timeout: SIGTERM, up to grace
    seconds to exit, then SIGKILL; the code is then nonzero and stderr says
    it timed out. subprocess.run(timeout=) would SIGKILL at once, leaving
    the GPU clock locked and a container behind."""
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        p.terminate()
        how = "SIGTERM"
        try:
            out, err = p.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            how = f"SIGTERM, then SIGKILL after {grace} s"
    code = p.returncode if p.returncode else 1
    msg = f"\n[pgaio] arena timed out after {timeout} s and was stopped ({how})\n"
    return (code if code > 0 else 128 - code), out or "", (err or "") + msg


def arena(cmd, args, timeout=1800):
    """Runs arena, returns (exit code, parsed JSON or None, stdout, stderr)."""
    full = shlex.split(cmd) + args
    cwd = REPO
    exe = Path(full[0]) if Path(full[0]).is_absolute() else REPO / full[0]
    if exe.is_file():
        # A native binary keeps its compile cache relative to its working
        # directory, so run it from its build dir rather than the repo root.
        full[0] = str(exe)
        cwd = exe.parent
    code, stdout, stderr = run_with_timeout(full, cwd, timeout)
    data = None
    out = stdout.strip()
    if out.startswith("{"):
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            data = None
    return code, data, stdout, stderr


def repo_rel(path, reported_root):
    """Paths from arena are absolute in its own filesystem (the container's)."""
    if not path:
        return ""
    if reported_root and path.startswith(reported_root.rstrip("/") + "/"):
        return path[len(reported_root.rstrip("/")) + 1:]
    try:
        return os.path.relpath(path, REPO)
    except ValueError:
        return path


def load_run(run):
    run_dir = Path(run)
    if not run_dir.is_absolute():
        cand = RUNS_DIR / run
        run_dir = cand if cand.exists() else (REPO / run)
    cfg_path = run_dir / "run.json"
    if not cfg_path.exists():
        sys.exit(f"no run at {run_dir} (expected run.json)")
    return run_dir, json.loads(cfg_path.read_text())


def load_attempts(run_dir):
    p = run_dir / "attempts.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_references(run_dir):
    """{lib: reference} from <run>/references/<lib>.json (pgaio_ncu.py
    reference). References are never iterations: they do not count against
    a budget and never become the best."""
    out = {}
    for p in sorted((Path(run_dir) / "references").glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out[data.get("lib") or p.stem] = data
    return out


def load_ab(run_dir):
    """Saved A/B comparisons of the run, oldest first."""
    out = []
    for p in sorted((Path(run_dir) / "ab").glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda r: r.get("time", ""))


def load_obs(run_dir, it):
    p = Path(run_dir) / "obs" / f"{it:03d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def target_args(cfg, rel_source=None, manifest=None):
    """What to pass arena to run and to inspect a candidate.

    A manifest names its own kernel. Otherwise the run's slot is used, with
    the candidate source swapped in when there is one."""
    if manifest:
        return ["--manifest", manifest], ["--manifest", manifest]
    ovr = ["--source-override", rel_source] if rel_source else []
    text = (REPO / (rel_source or cfg["slot_source"])).read_text(errors="replace")
    block = block_directive(text)
    if block:
        ovr += ["--block", str(block)]
    for k, v in sorted(cfg.get("defines", {}).items()):
        ovr += ["--define", f"{k}={v}"]
    return ["--run", cfg["kernel"]] + ovr, [cfg["kernel"]] + ovr


def inspect_static(cfg, rel_source=None, manifest=None, artifacts_dir=None):
    """Compile only: the --inspect summary and the SASS text. No launch, no
    timing, so this is free to call on any draft. With artifacts_dir (a path
    under the repo), arena also copies PTX, Triton IR and Warp's generated
    CUDA there as kernel.ptx, kernel.ttgir, kernel.warp_gen.cu."""
    _, insp_args = target_args(cfg, rel_source, manifest)
    extra = ["--artifacts-dir", str(Path(artifacts_dir).relative_to(REPO))] if artifacts_dir else []
    code, insp, _, err = arena(cfg["arena_cmd"], ["--inspect"] + insp_args + extra)
    if code != 0:
        return None, "", (err or "").strip()[-3000:]
    code, _, sass_text, err = arena(cfg["arena_cmd"], ["--sass"] + insp_args)
    return insp, (sass_text if code == 0 else ""), ""


def copied_artifacts(insp):
    """File names arena copied for --artifacts-dir."""
    return [Path(x).name for x in (insp or {}).get("artifacts_copied", [])]


def top_opcodes(insp, limit=14):
    """The most frequent full mnemonics, e.g. LDG.E.128.CONSTANT x8."""
    ops = (insp or {}).get("sass", {}).get("opcodes", {})
    return sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def show_lines(sass_text, prefixes, limit=80):
    """SASS lines whose mnemonic starts with one of prefixes, in program
    order with their addresses: where loads, barriers and stores sit relative
    to each other, without reading the whole listing."""
    pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)[^;]*;)")
    out = []
    for line in sass_text.splitlines():
        m = pat.search(line)
        if m and any(m.group(3).startswith(p) for p in prefixes):
            out.append(f"{m.group(1)}  {m.group(2).strip()}")
            if len(out) >= limit:
                out.append("...")
                break
    return out


def run_shapes(cfg):
    """(timed shapes, checked shapes), also for runs started before shapes."""
    if "train_shapes" in cfg:
        return cfg["train_shapes"], cfg["check_shapes"]
    train = expand_grid(cfg.get("train_params", {}))
    check = expand_grid(cfg.get("check_params", {}))
    return train, check + [t for t in train if t not in check]


def run_flags(cfg, timing):
    """Run-level arena settings fixed at init. Each is passed only when the
    run recorded it, so runs started before these settings existed measure
    exactly as they did. Counters need --profile, so they go with timing."""
    out = []
    if cfg.get("scratch_policy"):
        out += ["--scratch-policy", cfg["scratch_policy"]]
    if timing and cfg.get("lock_clocks"):
        out += ["--lock-clocks", str(cfg["lock_clocks"])]
    if timing and cfg.get("profile") and cfg.get("counters"):
        out += ["--counters", cfg["counters"]]
    return out


def timing_args(cfg, run_args, shapes, runs=None, profile=None):
    """Arguments of a timed arena call, the same for eval and ab (which
    passes profile=False: counters are not needed to compare times)."""
    profile = bool(cfg.get("profile")) if profile is None else profile
    out = (run_args + shape_args(shapes) +
           ["--l2", "cold", "--runs", str(runs or cfg.get("runs", 30))])
    if profile:
        out.append("--profile")
    return out + run_flags({**cfg, "profile": profile}, timing=True)


def measure(cfg, label, rel_source=None, manifest=None, artifacts_dir=None):
    """Correctness gate, cold-L2 timing and SASS inspection for one candidate."""
    cmd = cfg["arena_cmd"]
    run_args, _ = target_args(cfg, rel_source, manifest)
    empty = {"label": label, "check_results": [], "check_exit": None, "check_stderr": "",
             "timing_results": [], "timing_exit": None, "timing_stderr": "",
             "environment": {}, "inspect": None, "sass_text": ""}

    # 0. a run that forbids tensor cores looks at the SASS first, so the
    # rule holds whether or not the candidate would pass its checks
    insp, sass_text, inspected = None, "", False
    if cfg.get("forbid_tensor_cores"):
        insp, sass_text, _ = inspect_static(cfg, rel_source, manifest, artifacts_dir)
        inspected = True
        ops = tensor_ops(insp, sass_text) if insp else {}
        if ops:
            return {**empty, "inspect": insp, "sass_text": sass_text, "tensor_ops": ops}

    # 1. correctness over held-out sizes, distributions and seeds
    train, check = run_shapes(cfg)
    chk_args = (run_args + shape_args(check) +
                ["--dist", ",".join(cfg["check_dists"]),
                 "--seed", ",".join(str(s) for s in cfg["check_seeds"]),
                 "--runs", "1", "--warmup", "1"] + run_flags(cfg, timing=False))
    chk_exit, chk, _, chk_err = arena(cmd, chk_args)
    chk_results = (chk or {}).get("results", [])
    required = [k for k in cfg.get("required_checks", [])]

    # 2. timing on the train size, cold L2, counters if available
    tim_results, tim_exit, env, tim_err = [], None, (chk or {}).get("environment", {}), ""
    verdict_so_far = classify_attempt(chk_results, [], chk_exit, 0, [], required)
    if verdict_so_far == "ok":
        tim_exit, tim, _, tim_err = arena(cmd, timing_args(cfg, run_args, train))
        tim_results = (tim or {}).get("results", [])
        env = (tim or {}).get("environment", env)

    # 3. static view of what the compiler made of it
    if verdict_so_far == "ok" and not inspected:
        insp, sass_text, _ = inspect_static(cfg, rel_source, manifest, artifacts_dir)

    return {
        "label": label,
        "check_results": chk_results, "check_exit": chk_exit, "check_stderr": chk_err[-4000:],
        "timing_results": tim_results, "timing_exit": tim_exit, "timing_stderr": tim_err[-4000:],
        "environment": env, "inspect": insp, "sass_text": sass_text,
    }


def observe(cfg, m, guard):
    """Turns raw measurements into the observation the agent reads."""
    verdict = classify_attempt(m["check_results"], m["timing_results"], m["check_exit"],
                               m["timing_exit"], guard, cfg.get("required_checks", []))
    tensor = m.get("tensor_ops") if cfg.get("forbid_tensor_cores") and not guard else None
    if tensor:
        verdict = "rejected_guard"
    obs = {"verdict": verdict, "category": cfg["category"], "guard_violations": guard}
    if tensor:
        obs["tensor_ops"] = tensor
    checks = m["check_results"]
    req = cfg.get("required_checks", [])
    gated = [r for r in checks if not req or combo_key(r) in req]
    obs["checks"] = {
        "passed": sum(1 for r in gated if r.get("success") and r.get("verified")),
        "total": len(gated),
        "failed": failed_checks(checks, req)[:8],
        "max_total_error": max([r.get("numerics", {}).get("max_total_error", 0) or 0
                                for r in checks] or [0]),
    }
    err = first_error(checks) or first_error(m["timing_results"])
    if not err and verdict in ("compile_error", "runtime_error"):
        err = (m["check_stderr"] or m["timing_stderr"])[-2000:]
    if tensor:
        err = tensor_rejection(tensor)
    obs["error"] = err[-3000:] if err else ""

    if verdict != "ok":
        obs["score"] = None
        return obs

    rs = m["timing_results"]
    env = dict(m["environment"])
    # One set of floors for the whole run, recorded at init, so an
    # iteration's score cannot move because a floor was measured a little
    # differently.
    for key in ("launch_floor_ms", "roundtrip_floor_ms"):
        if cfg.get(key):
            env[key] = cfg[key]
    per = [size_sol(r, env) for r in rs]
    obs["sizes"] = per
    obs["score"] = geomean([p["fraction"] for p in per])
    obs["score_ci"] = add_spread(per, rs)
    add_clock(per, rs, env)
    # Counters and the diagnosis describe the largest timed shape: that is
    # where bandwidth or FLOPs are the limit and the counters say most.
    i = max(range(len(rs)), key=lambda k: per[k]["ideal_bytes"] + per[k]["flops"])
    r = rs[i]
    obs["sol"] = sol(r, env)
    if per[i].get("fraction_at_clock") is not None:
        mhz = per[i]["clock"]["sm_clock_mean_mhz"]
        obs["sol"]["fraction_at_clock"] = obs["sol"]["fraction"] * float(env["peak_fp32_gflops_at_mhz"]) / mhz
        obs["sol"]["clock_mhz"] = mhz
    if r.get("launched_kernels"):
        obs["launched_kernels"] = r["launched_kernels"]
    obs["timing"] = {k: r["timing"].get(k) for k in ("op_ms", "gpu_ms", "overhead_ms", "launch_count")}
    obs["throughput"] = r.get("throughput", {})
    obs["counters"] = r.get("counters", {})
    obs["launch"] = r.get("launch", {})
    obs["quality"] = r.get("quality", {})
    tp = r.get("throughput", {})
    op = r["timing"].get("op_ms") or 0
    obs["ideal_bytes"] = (tp.get("bandwidth_gbps") or 0) * 1e9 * op / 1e3
    insp = m["inspect"] or {}
    obs["resources"] = insp.get("resources", {})
    obs["sass"] = sass_brief(insp)
    obs["sass_mix"] = sass_mix(insp)
    if cfg.get("forbid_tensor_cores"):
        obs["tensor_forbidden"] = True
    obs["diagnosis"] = diagnose(obs)
    return obs


def size_line(p, refs=None):
    """One shape of the per-shape table."""
    sp = p.get("spread") or {}
    pm = f" ±{sp['ci_half_frac']:.1%}" if sp.get("ci_half_frac") is not None else ""
    line = (f"  {shape_label(p['params']):>16}: {p['op_ms']:.4f} ms{pm}, "
            f"{p['fraction']:6.1%} of the {p['bound']} roof ({p['roof_ms']:.4f} ms){clock_text(p)}")
    for lib, ref in sorted((refs or {}).items()):
        _, per = ref_relative(ref, [p])
        if per:
            line += f", {list(per.values())[0]:.0%} of {lib}"
    if sp.get("bimodal"):
        line += ", BIMODAL"
    pw = power_text(p)
    return line + (f"\n  {'':>16}  {pw}" if pw else "")


def render(obs, it, cfg, best, baseline, refs=None):
    """Short text for the agent. Everything in it is in the JSON too."""
    dsl = obs.get("dsl") or cfg.get("dsl", "")
    lines = [f"iteration {it}{'/' + str(cfg['budget']) if cfg.get('budget') else ''} ({dsl})  verdict: {obs['verdict'].upper()}"
             + ("  (new best)" if obs.get("new_best") else "")]
    if obs["guard_violations"]:
        lines.append("protected files changed, iteration rejected: " +
                     ", ".join(obs["guard_violations"][:10]))
        lines.append("restore edited files (git checkout -- <file>) and delete new ones "
                     "before the next eval.")
    ch = obs["checks"]
    lines.append(f"correctness: {ch['passed']}/{ch['total']} checks passed"
                 + (f", max total error {ch['max_total_error']:.2e}" if ch["total"] else ""))
    for f in ch["failed"]:
        lines.append(f"  FAILED {f['check']}: rel error {f['max_rel_error']} > tol {f['tolerance']}")
    if obs.get("error"):
        lines.append("error:\n" + "\n".join("  " + l for l in obs["error"].strip().splitlines()[-25:]))
    if obs["verdict"] != "ok":
        return "\n".join(lines)

    t, s = obs["timing"], obs["sol"]
    if obs.get("sizes") and (len(obs["sizes"]) > 1 or obs.get("score_ci") or refs):
        extra = [x for x in (fmt_ci(obs.get("score_ci")), ref_text(refs, obs["sizes"])) if x]
        lines.append(f"score: {obs['score']:.1%}" + (f" ({', '.join(extra)})" if extra else "")
                     + f", geometric mean of SOL over {len(obs['sizes'])} shape(s)")
        for p in obs["sizes"]:
            lines.append(size_line(p, refs))
        lines.append("largest shape:")
    line = f"time: op {t['op_ms']:.4f} ms, gpu {t['gpu_ms']:.4f} ms, {t['launch_count']} launch(es)"
    if baseline and baseline.get("timing"):
        line += f"  | {baseline['timing']['op_ms'] / t['op_ms']:.2f}x vs baseline"
    if best and best.get("timing") and not obs.get("new_best"):
        line += (f", {best['timing']['op_ms'] / t['op_ms']:.2f}x vs best (iter {best['iteration']}"
                 + (f", {obs['best_dsl']}" if obs.get("best_dsl") else "") + ")")
    lines.append(line)
    ai = s.get("intensity")
    at_clock = (f", {s['fraction_at_clock']:.1%} at the measured {s['clock_mhz'] / 1000:.2f} GHz"
                if s.get("fraction_at_clock") is not None and s.get("clock_mhz") else "")
    lines.append(f"SOL: {s['fraction']:.1%} of nominal{at_clock} ({s['bound']}-bound, "
                 f"{s['achieved']:.0f}/{s['roof']:.0f} {s['unit']}, intensity {ai:.3f} FLOP/B vs "
                 f"ridge {s['ridge']:.1f})"
                 if ai is not None and s.get("ridge") else f"SOL: {s['fraction']:.1%}{at_clock}")
    lines.append(f"target {cfg['target']:.0%}" + (", reached" if s["fraction"] >= cfg["target"] else ""))
    c = obs["counters"]
    if c.get("available"):
        lines.append(f"counters: DRAM read {c['dram_read_bytes'] / 1e6:.1f} MB, write "
                     f"{c['dram_write_bytes'] / 1e6:.1f} MB (ideal {obs['ideal_bytes'] / 1e6:.1f} MB), "
                     f"occupancy {c['occupancy']:.0%}, IPC {c['ipc']:.2f}"
                     + (f", issue active {c['issue_active']:.0%}" if c.get("issue_active") is not None else ""))
    for k in obs.get("launched_kernels") or []:
        lines.append(f"  launched {k.get('name')}: {k.get('count_per_op')} per op, {k.get('gpu_ms') or 0:.4f} ms")
    r = obs["resources"]
    if r:
        lines.append(f"resources: {r.get('registers')} regs, {r.get('shared')} B shared, "
                     f"{r.get('local')} B local, launch grid {obs['launch'].get('grid')} "
                     f"block {obs['launch'].get('block')}")
    b = obs["sass"]
    lines.append(f"SASS: {b['instructions']} instr, {b['global_loads']} global loads "
                 f"(mean {b['mean_global_load_bytes']:.0f} B), {b['global_atomic']} global atomics, "
                 f"{b['warp_shuffle']} shuffles, {b['shared_mem']} shared ops, {b['sync']} syncs")
    if obs.get("sass_vs_best"):
        lines.append("SASS vs best: " + ", ".join(f"{k} {v:+d}" for k, v in obs["sass_vs_best"].items()))
    if obs["diagnosis"]:
        lines.append("diagnosis (heuristic):")
        lines += ["  - " + d for d in obs["diagnosis"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def parse_sizes(specs, fallback):
    if not specs:
        return {k: list(v) for k, v in fallback.items()}
    out = {}
    for spec in specs:
        k, _, v = spec.partition("=")
        out[k] = [int(x) for x in v.split(",") if x]
    return out


def cmd_init(a):
    if bool(a.kernel) == bool(a.category):
        sys.exit("pass exactly one of --category (any DSL, candidates with manifests) "
                 "or --kernel (one slot)")
    arena_cmd = a.arena or os.environ.get("ARENA_CMD") or DEFAULT_ARENA_CMD
    problem = bool(a.category)
    kernel = a.kernel or a.baseline or CATEGORY_BASELINE.get(a.category)
    if not kernel:
        sys.exit(f"no baseline kernel known for category {a.category}; pass --baseline")

    # First call builds (Docker entrypoint) and describes the baseline kernel.
    print(f"[pgaio] inspecting {kernel} via: {arena_cmd}", file=sys.stderr)
    code, insp, out, err = arena(arena_cmd, ["--inspect", kernel], timeout=3600)
    if code != 0 or not insp:
        sys.exit(f"--inspect {kernel} failed (exit {code}):\n{err[-3000:]}")
    if "docker compose" in arena_cmd and "ARENA_SKIP_BUILD" not in arena_cmd:
        # The binary is built now; later calls skip the cmake check.
        arena_cmd = arena_cmd.replace("run --rm", "run --rm -e ARENA_SKIP_BUILD=1", 1)
    if a.lock_clocks and "docker compose" in arena_cmd and "SYS_ADMIN" not in arena_cmd:
        # Root in a container cannot set GPU clocks without this capability.
        # Only runs that ask to lock clocks get it.
        arena_cmd = arena_cmd.replace("run --rm", "run --rm --cap-add SYS_ADMIN", 1)

    root = insp.get("repo_root", "")
    slot_source = repo_rel(insp.get("slot_source", ""), root)
    category = insp["category"]
    if problem and category != a.category:
        sys.exit(f"baseline {kernel} is a {category} kernel, not {a.category}")
    if not problem and not slot_source:
        sys.exit(f"{kernel} has no source slot (it is compiled into the binary)")
    defaults = CATEGORIES.get(category, {"check": {}})

    if a.train:
        train = []
        for spec in a.train:
            shape = {}
            for item in spec.split(","):
                k, _, v = item.partition("=")
                shape[k.strip()] = int(v)
            train.append(shape)
        regime = "custom"
    else:
        train = train_shapes(category, a.sizes)
        regime = a.sizes
    if not train:
        sys.exit(f"no timed shapes for category {category}; pass --train")
    check = parse_sizes(a.check, defaults["check"])
    check_shapes = expand_grid(check)
    for sh in train:   # every timed shape is checked too
        if sh not in check_shapes:
            check_shapes.append(sh)

    dsls = [d.strip() for d in (a.dsls or ",".join(DSLS)).split(",") if d.strip()]
    for d in dsls:
        if d not in DSLS:
            sys.exit(f"unknown DSL {d}; choose from {', '.join(DSLS)}")
    per_dsl = a.budget_per_dsl if problem else 0
    budget = a.budget or None   # None: no limit; the run ends on the target or a plateau

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = a.name or f"{stamp}-{a.category or kernel}"
    run_dir = RUNS_DIR / name
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists")
    for sub in ("candidates", "obs", "work", "inspections"):
        (run_dir / sub).mkdir(parents=True)

    seed_base = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    cfg = {
        "name": name, "created": now(),
        "kind": "problem" if problem else "slot",
        "kernel": kernel, "category": category,
        "dsl": "any" if problem and len(dsls) > 1 else (dsls[0] if problem else insp.get("dsl")),
        "dsls": dsls if problem else [insp.get("dsl")],
        "entry_point": insp.get("entry_point"),
        "slot_source": slot_source,
        "descriptor_file": repo_rel(insp.get("descriptor_file", ""), root),
        "arena_cmd": arena_cmd,
        "budget": budget, "budget_per_dsl": per_dsl,
        "target": a.target, "runs": a.runs, "mode": a.mode,
        "docs": a.docs == "on",
        "sizes": regime, "train_shapes": train, "check_params": check, "check_shapes": check_shapes,
        "check_dists": CHECK_DISTS,
        "check_seeds": [seed_base % 100000 + i for i in range(a.seeds)],
        "defines": {k: int(v) for k, v in (kv.split("=") for kv in (a.define or []))},
        "model": a.model or os.environ.get("PGAIO_MODEL", ""),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
        "profile": True,
        "plateau_rule": PLATEAU_RULE,
        "forbid_tensor_cores": bool(a.forbid_tensor_cores),
        "scratch_policy": a.scratch_policy or "legacy",
        "counters": a.counters or "full",
        "protected_hashes": hash_protected(),
    }
    if a.lock_clocks:
        cfg["lock_clocks"] = a.lock_clocks
    # Settings the user did not ask for are dropped if this arena build does
    # not know them yet; one the user asked for stops the run instead.
    asked = {"--scratch-policy": "scratch_policy" if a.scratch_policy else None,
             "--counters": "counters" if a.counters else None,
             "--lock-clocks": "lock_clocks" if a.lock_clocks else None}
    defaulted = {"--scratch-policy": "scratch_policy", "--counters": "counters", "--lock-clocks": "lock_clocks"}

    # Iteration 0. A slot run starts from the slot's source or --start; a
    # problem run measures the baseline kernel as it is in the repo.
    base_rel = None
    if problem:
        base_copy = run_dir / "candidates" / "000_baseline"
        base_copy.mkdir()
        if slot_source:
            shutil.copy2(REPO / slot_source, base_copy / Path(slot_source).name)
        cfg["start_source"] = slot_source or kernel
    else:
        start = Path(a.start).resolve() if a.start else REPO / slot_source
        if a.start and candidate_ext(start.name) != candidate_ext(slot_source):
            shutil.rmtree(run_dir)
            sys.exit(f"--start must be a {candidate_ext(slot_source)} file for slot {kernel}")
        base_copy = run_dir / "candidates" / f"000_baseline{candidate_ext(slot_source)}"
        shutil.copy2(start, base_copy)
        base_rel = str(base_copy.relative_to(REPO)) if a.start else None
        cfg["start_source"] = str(start.relative_to(REPO)) if start.is_relative_to(REPO) else str(start)

    print(f"[pgaio] measuring baseline {cfg['start_source']} ...", file=sys.stderr)
    m = measure(cfg, "baseline", rel_source=base_rel)
    for _ in range(len(defaulted)):
        flag = unknown_option(m["check_stderr"] + m["timing_stderr"])
        if flag not in defaulted:
            break
        if asked[flag]:
            shutil.rmtree(run_dir)
            sys.exit(f"this arena build does not know {flag}; rebuild arena or drop the option")
        print(f"[pgaio] this arena build does not know {flag}; the run goes on without it",
              file=sys.stderr)
        cfg.pop(defaulted[flag], None)
        m = measure(cfg, "baseline", rel_source=base_rel)
    if m.get("tensor_ops"):
        # measure() stopped before the checks; without this the run would
        # end on "passes none of the correctness checks" with no reason
        shutil.rmtree(run_dir)
        sys.exit(f"baseline {cfg['start_source']}: " + tensor_rejection(m["tensor_ops"]).replace(
            "the candidate was not measured", "start from a baseline without them"))
    if not m["timing_results"] or not all(r.get("success") for r in m["timing_results"]):
        # Counters unavailable (no perf access) is not a reason to stop.
        cfg["profile"] = False
        m = measure(cfg, "baseline", rel_source=base_rel)

    # Drop checks the baseline itself cannot pass, so the gate asks for
    # nothing the reference kernel does not deliver.
    passing = [r for r in m["check_results"] if r.get("success") and r.get("verified")]
    dropped = sorted({combo_key(r) for r in m["check_results"]} - {combo_key(r) for r in passing})
    cfg["required_checks"] = sorted({combo_key(r) for r in passing})
    cfg["device"] = m["environment"].get("device_name", "")
    cfg["launch_floor_ms"] = m["environment"].get("launch_floor_ms", 0.0)
    cfg["roundtrip_floor_ms"] = m["environment"].get("roundtrip_floor_ms", 0.0)
    cfg["environment"] = {k: m["environment"].get(k) for k in
                          ("peak_dram_gbps", "peak_fp32_gflops", "l2_cache_bytes", "arch", "device_name")}
    # newer arena builds: the clock the FP32 peak assumes, and the clock lock
    cfg["environment"].update({k: m["environment"][k] for k in
                               ("max_sm_clock_mhz", "peak_fp32_gflops_at_mhz", "clock_lock")
                               if m["environment"].get(k) is not None})
    cfg["dropped_checks"] = dropped
    if not passing:
        shutil.rmtree(run_dir)
        sys.exit("the baseline passes none of the correctness checks; nothing to gate on:\n" +
                 (first_error(m["check_results"]) or m["check_stderr"])[-2000:])

    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))
    obs = observe(cfg, m, [])
    if obs["verdict"] != "ok":
        print(render(obs, 0, cfg, None, None))
        shutil.rmtree(run_dir)
        sys.exit("baseline measurement failed")
    obs["iteration"] = 0
    obs["new_best"] = True
    obs["dsl"] = insp.get("dsl")
    record(run_dir, cfg, 0, base_copy, f"baseline: {kernel} as it is in the repo"
           if problem else "baseline: the slot's own source", obs, m, insp.get("dsl"))

    update_journal(run_dir, cfg)
    l2 = m["environment"].get("l2_cache_bytes") or 0
    ideal = obs.get("ideal_bytes") or 0
    if l2 and ideal and ideal < 2 * l2:
        print(f"WARNING: the timed working set is {ideal / 1e6:.0f} MB against a "
              f"{l2 / 1e6:.0f} MB L2. Output can stay in cache through the timed loop and "
              f"flatter SOL; pass a larger --train size.")
    print(f"run: {run_dir.relative_to(REPO)}  (draft candidates in {(run_dir / 'work').relative_to(REPO)}/)")
    print(f"journal: {(run_dir / 'journal.md').relative_to(REPO)}  "
          f"progression: {(run_dir / 'progress.html').relative_to(REPO)} (both update after every eval)")
    if problem:
        print(f"problem: {category}, DSLs {', '.join(dsls)}, baseline {kernel} "
              f"({cfg['start_source']}); candidates are manifests (see the arena-optimize skill)")
        print((f"budget {budget} iterations" if budget else "no iteration budget") +
              (f", at most {per_dsl} per DSL" if per_dsl else ""))
    else:
        print(f"kernel {kernel} ({cfg['dsl']}), slot {slot_source}, entry point "
              f"{cfg['entry_point']}, descriptor {cfg['descriptor_file']}")
        print(f"budget {budget} iterations" if budget else "no iteration budget")
    print(f"mode {cfg['mode']}, docs {'on' if cfg['docs'] else 'off'}, target {cfg['target']:.0%} SOL, "
          f"train {train}, counters {cfg.get('counters', 'on') if cfg['profile'] else 'off'}")
    print(settings_line(cfg))
    if dropped:
        print(f"dropped {len(dropped)} checks the baseline fails: {', '.join(dropped[:6])}")
    print(render(obs, 0, cfg, None, None))
    ls = matching_learnings(category, cfg["dsl"])
    if ls:
        print("\nlearnings from earlier runs (same category or DSL):")
        for l in ls:
            print(f"  - [{l.get('dsl')}/{l.get('category')}, {l.get('device', '')}] {l['insight']}")


def settings_line(cfg):
    """Run-level settings that change what arena measures."""
    out = [f"scratch policy {cfg.get('scratch_policy') or 'legacy (arena default)'}"]
    if cfg.get("lock_clocks"):
        lock = (cfg.get("environment") or {}).get("clock_lock") or {}
        out.append(f"clocks locked at {cfg['lock_clocks']}"
                   + (" (applied)" if lock.get("applied") else
                      f" (NOT applied: {lock.get('error') or 'arena did not say'})" if lock else ""))
    if cfg.get("forbid_tensor_cores"):
        out.append("tensor cores forbidden")
    out.append(f"plateau rule {cfg.get('plateau_rule', 1)}")
    return "settings: " + ", ".join(out)


def record(run_dir, cfg, it, cand_path, hypothesis, obs, m, dsl):
    (run_dir / "obs" / f"{it:03d}.json").write_text(json.dumps(
        {**obs, "raw": {k: m[k] for k in ("check_results", "timing_results", "inspect")}}, indent=1))
    if m.get("sass_text"):
        (run_dir / "obs" / f"{it:03d}.sass").write_text(m["sass_text"])
    append_jsonl(run_dir / "attempts.jsonl", {
        "iteration": it, "time": now(), "dsl": dsl,
        "candidate": str(Path(cand_path).relative_to(REPO)),
        "hypothesis": hypothesis, "verdict": obs["verdict"], "score": obs.get("score"),
        "score_ci": obs.get("score_ci"),
        "op_ms": obs.get("timing", {}).get("op_ms"),
        "gpu_ms": obs.get("timing", {}).get("gpu_ms"),
        "new_best": bool(obs.get("new_best")),
        "sass": obs.get("sass"),
        "error": (obs.get("error") or "")[:300],
    })


def resolve_candidate(path):
    src = Path(path)
    if not src.is_absolute():
        src = Path.cwd() / src
    if not src.exists():
        sys.exit(f"not found: {src}")
    return src.resolve()


def manifest_dsl(manifest_path):
    """The DSL a manifest counts against: its source's, or "mixed" when its
    launches use sources in different DSLs."""
    try:
        spec = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    used = {dsl_of(spec.get("source", ""))}
    for launch in spec.get("launches", []):
        if "source" in launch:
            used.add(dsl_of(launch["source"]))
    used.discard(None)
    if len(used) > 1:
        return "mixed"
    return used.pop() if used else None


def snapshot_manifest(manifest_path, dest_dir, name=None, description=None):
    """Copies a manifest and the source it names into dest_dir, with the
    copy's "source" pointing at the copied file, so the snapshot is
    self-contained and later edits to the draft cannot change history.
    `name` and `description`, when given, replace the draft's, so the
    snapshot lists as "<run>/<iteration>-<file>" wherever it is opened.
    Returns (manifest copy, dsl)."""
    try:
        spec = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{manifest_path} is not valid JSON: {e} (this did not count as an iteration)")
    if "source" not in spec:
        sys.exit(f"{manifest_path} has no \"source\" (this did not count as an iteration)")
    src = Path(spec["source"])
    if not src.is_absolute():
        src = manifest_path.parent / src
    if not src.exists():
        sys.exit(f"manifest source not found: {src} (this did not count as an iteration)")
    dsl = dsl_of(src.name)
    if not dsl:
        sys.exit(f"{src.name}: unknown DSL; use .cu, .triton.py, .cutile.py or .warp.py")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)
    spec["source"] = src.name
    if name:
        spec["name"] = name
    if description:
        spec["description"] = description
    dsls_used = {dsl}
    # Launches may name their own sources (several DSL kernels, or dispatch
    # across DSLs); those are copied too, so the snapshot is self-contained.
    for launch in spec.get("launches", []):
        if "source" not in launch:
            continue
        ls = Path(launch["source"])
        if not ls.is_absolute():
            ls = manifest_path.parent / ls
        if not ls.exists():
            sys.exit(f"launch source not found: {ls} (this did not count as an iteration)")
        shutil.copy2(ls, dest_dir / ls.name)
        launch["source"] = ls.name
        dsls_used.add(dsl_of(ls.name))
    # A candidate that dispatches across DSLs is its own kind, with its own
    # budget, rather than being charged to whichever file is listed first.
    if len(dsls_used) > 1:
        dsl = "mixed"
    out = dest_dir / manifest_path.name
    out.write_text(json.dumps(spec, indent=2))
    return out, dsl


def candidate_name(run_dir, it, manifest):
    """How a measured candidate is listed in arena (--open, the GUI)."""
    return f"{run_dir.name}/{it:03d}-{Path(manifest).stem}"


def describe_snapshot(snap, it, dsl, hypothesis, obs):
    """Puts the iteration's outcome in the snapshot's description, which the
    GUI shows when hovering the kernel."""
    spec = json.loads(snap.read_text())
    outcome = obs.get("verdict", "?")
    if obs.get("score") is not None:
        outcome += f", score {obs['score']:.1%} of SOL"
    spec["description"] = f"PGAIO iteration {it} ({dsl}): {outcome}. {hypothesis}"
    snap.write_text(json.dumps(spec, indent=2))


def best_of(run_dir, attempts, dsl=None):
    ok = [x for x in attempts if x["verdict"] == "ok" and x.get("score") is not None
          and (dsl is None or x.get("dsl") == dsl)]
    if not ok:
        return None, None
    b = max(ok, key=lambda x: x["score"])
    return b, json.loads((run_dir / "obs" / f"{b['iteration']:03d}.json").read_text())


def cmd_eval(a):
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    it = len(attempts)
    src = resolve_candidate(a.candidate)
    problem = cfg.get("kind") == "problem"

    if problem:
        if not src.name.endswith(".json"):
            sys.exit("a problem run evaluates manifests: pass the .json that names the source "
                     "(this did not count as an iteration)")
        dsl = manifest_dsl(src)
    else:
        dsl = cfg["dsl"]
        want = candidate_ext(cfg["slot_source"])
        if candidate_ext(src.name) != want:
            sys.exit(f"candidate must be a {want} file to fit slot {cfg['kernel']} "
                     "(this did not count as an iteration)")
    why = budget_error(cfg, attempts, dsl)
    if why:
        sys.exit(why + "; see `pgaio.py status`")
    skipped = [x for x in cfg.get("skipped_dsls", []) if x.get("dsl") == dsl]
    if skipped:
        print(f"note: {dsl} was marked exhausted after iteration {skipped[-1].get('after_iteration')} "
              f"({skipped[-1].get('reason')}); measuring anyway", file=sys.stderr)

    slug = f"{it:03d}_{slugify(a.hypothesis)}"
    if problem:
        snap, dsl = snapshot_manifest(src, run_dir / "candidates" / slug,
                                      name=candidate_name(run_dir, it, src),
                                      description=a.hypothesis)
        target = {"manifest": str(snap.relative_to(REPO))}
    else:
        snap = run_dir / "candidates" / f"{slug}{candidate_ext(src.name)}"
        shutil.copy2(src, snap)
        target = {"rel_source": str(snap.relative_to(REPO))}

    guard = guard_violations(cfg["protected_hashes"], hash_protected())
    if guard:
        m = {k: [] if k.endswith("results") else None for k in
             ("check_results", "timing_results", "inspect")}
        m.update(check_exit=None, timing_exit=None, check_stderr="", timing_stderr="",
                 environment={}, sass_text="")
    else:
        (run_dir / "obs" / f"{it:03d}").mkdir(parents=True, exist_ok=True)
        m = measure(cfg, f"iter {it}", artifacts_dir=run_dir / "obs" / f"{it:03d}", **target)
    obs = observe(cfg, m, guard)
    obs["iteration"] = it
    obs["dsl"] = dsl

    best_att, best_obs = best_of(run_dir, attempts)
    base_obs = json.loads((run_dir / "obs" / "000.json").read_text()) \
        if (run_dir / "obs" / "000.json").exists() else None
    if obs["verdict"] == "ok":
        obs["new_best"] = best_att is None or obs["score"] > best_att["score"]
        if best_obs:
            obs["sass_vs_best"] = sass_delta(obs.get("sass"), best_obs.get("sass"))
            obs["best_dsl"] = best_att.get("dsl")

    record(run_dir, cfg, it, snap, a.hypothesis, obs, m, dsl)
    if problem:
        describe_snapshot(snap, it, dsl, a.hypothesis, obs)
    update_journal(run_dir, cfg)
    print(render(obs, it, cfg, best_obs, base_obs, load_references(run_dir)))
    attempts.append({"iteration": it, "dsl": dsl})
    used = len([x for x in attempts if x.get("iteration", 0) > 0])
    left = f"{cfg['budget'] - used} iteration(s) left" if cfg.get("budget") else f"{used} iteration(s) used"
    if cfg.get("budget_per_dsl"):
        per = {d: sum(1 for x in attempts if x.get("dsl") == d and x.get("iteration", 0) > 0)
               for d in cfg.get("dsls", [])}
        left += " (" + ", ".join(f"{d} {cfg['budget_per_dsl'] - n}" for d, n in per.items()) + ")"
    reached = obs.get("score") is not None and obs["score"] >= cfg["target"]
    since, short, flat = plateau(cfg, load_attempts(run_dir), load_ab(run_dir))
    left += f"; {since} eval(s) since {plateau_text(cfg)}"
    if short:
        left += f"; DSLs with under {PLATEAU_MIN_PER_DSL} evals (and not skipped): {', '.join(short)}"
    if reached:
        left += "; target reached, you may stop"
    elif flat and not cfg.get("budget"):
        left += (f"; plateau ({PLATEAU_EVALS} evals without gain, every DSL tried): "
                 "you may stop, or keep going if an untried idea remains")
    print(f"\n{left}")
    print(f"snapshot: {snap.relative_to(REPO)}   full observation: "
          f"{(run_dir / 'obs' / f'{it:03d}.json').relative_to(REPO)}")
    print(f"journal: {(run_dir / 'journal.md').relative_to(REPO)}   "
          f"progression: {(run_dir / 'progress.html').relative_to(REPO)}")


def cmd_sass(a):
    """Free: how a draft compiled, against the best so far. Never timed."""
    run_dir, cfg = load_run(a.run)
    src = resolve_candidate(a.candidate)
    n = sum(1 for d in (run_dir / "inspections").iterdir() if d.is_dir())
    out = run_dir / "inspections" / f"{n:03d}_{src.stem}"
    # Created here, not by arena: inside Docker arena runs as root, and files
    # in a root-owned folder could not be cleaned up from the host.
    out.mkdir(parents=True, exist_ok=True)
    if cfg.get("kind") == "problem":
        snap, dsl = snapshot_manifest(src, out)
        insp, sass_text, err = inspect_static(cfg, manifest=str(snap.relative_to(REPO)), artifacts_dir=out)
    else:
        dsl = cfg["dsl"]
        rel = str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)
        insp, sass_text, err = inspect_static(cfg, rel_source=rel, artifacts_dir=out)
    if insp is None:
        print(f"did not compile:\n{err}")
        sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inspect.json").write_text(json.dumps(insp, indent=1))
    (out / "kernel.sass").write_text(sass_text)
    extra = copied_artifacts(insp)
    append_jsonl(run_dir / "inspections.jsonl", {"time": now(), "draft": str(src), "dsl": dsl,
                                                 "sass": sass_brief(insp)})

    brief = sass_brief(insp)
    r = insp.get("resources", {})
    print(f"{dsl} draft {src.name}: compiled (not timed, not counted)")
    print(f"resources: {r.get('registers')} regs, {r.get('shared')} B shared, {r.get('local')} B local, "
          f"{r.get('stack')} B stack")
    if brief["local_mem"]:
        print(f"SPILLS: {brief['local_mem']} local-memory instructions (LDL/STL) in SASS: registers are "
              "spilling; shrink the tile or raise num_warps before measuring")
    print(f"SASS: {brief['instructions']} instr, {brief['global_loads']} global loads (mean "
          f"{brief['mean_global_load_bytes']:.0f} B), {brief['global_atomic']} global atomics, "
          f"{brief['warp_shuffle']} shuffles, {brief['warp_reduce']} warp reduces, "
          f"{brief['shared_mem']} shared ops, {brief['sync']} syncs, {brief['tensor']} tensor")
    for e in insp.get("entries", []):
        print(f"  entry {e['entry']}: {e['instructions']} instr, {e['registers']} regs")
    print("top opcodes: " + ", ".join(f"{op} x{c}" for op, c in top_opcodes(insp)))
    mix = sass_mix(insp)
    if mix.get("ffma") and mix.get("ffma_per_lds") is not None:
        print(f"FMA balance: FFMA {mix['ffma_share']:.0%} of instructions, "
              f"{mix['ffma_per_lds']:.1f} FMAs per shared load")
    ops = tensor_ops(insp, sass_text)
    if ops and cfg.get("forbid_tensor_cores"):
        print("WARNING: " + tensor_rejection(ops).replace("the candidate was not measured",
                                                          "eval would reject this draft"))
    elif ops:
        print("tensor cores: " + ", ".join(f"{k} x{v}" for k, v in sorted(ops.items())))
    if cfg.get("kind") == "problem":
        data, note = run_plan(cfg, str(snap.relative_to(REPO)))
        if data is None:
            print(note)
        elif data.get("ok", True) and not plan_error(data):
            print(plan_summary(data))
        else:
            print("\n".join(render_plan(data)))
            print(f"first plan error: {plan_error(data)}")
    if a.show:
        prefixes = [x.strip().upper() for x in a.show.split(",") if x.strip()]
        print(f"SASS lines starting with {', '.join(prefixes)}, in program order:")
        for line in show_lines(sass_text, prefixes):
            print("  " + line)
    best_att, best_obs = best_of(run_dir, load_attempts(run_dir))
    if best_obs:
        d = sass_delta(brief, best_obs.get("sass"))
        print(f"vs best (iteration {best_att['iteration']}, {best_att.get('dsl')}): " +
              (", ".join(f"{k} {v:+d}" for k, v in d.items()) or "same instruction mix"))
        best_sass = run_dir / "obs" / f"{best_att['iteration']:03d}.sass"
        print(f"diff the SASS: diff {best_sass.relative_to(REPO)} {(out / 'kernel.sass').relative_to(REPO)}")
    print(f"saved: {out.relative_to(REPO)}/ (kernel.sass, inspect.json"
          + "".join(", " + x for x in extra) + ")")


def run_plan(cfg, manifest_rel):
    """(arena --plan result, notice). The result is None when this arena
    build has no --plan or the call failed; the notice then says why in one
    line. Plans every timed and checked shape."""
    train, check = run_shapes(cfg)
    shapes = list(train) + [c for c in check if c not in train]
    code, data, out, err = arena(cfg["arena_cmd"], ["--plan", "--manifest", manifest_rel] + shape_args(shapes))
    if data is not None and "plans" in data:
        return data, ""
    if unknown_option(err) == "--plan":
        return None, "launch plan: this arena build has no --plan (rebuild arena for it); skipped"
    last = [l for l in (err or out or "").strip().splitlines() if l.strip()]
    return None, f"launch plan failed (exit {code}): {last[-1] if last else 'no output'}"


def cmd_check(a):
    """Free: where each launch of a manifest goes at every timed and checked
    shape (arena --plan): it compiles, but launches nothing and times nothing."""
    run_dir, cfg = load_run(a.run)
    if cfg.get("kind") != "problem":
        sys.exit("check is for problem runs, whose candidates are manifests")
    src = resolve_candidate(a.manifest)
    if not src.is_relative_to(REPO):
        sys.exit(f"{src} is outside the repo; put drafts in {(run_dir / 'work').relative_to(REPO)}/")
    data, note = run_plan(cfg, str(src.relative_to(REPO)))
    if data is None:
        print(note)
        return
    print("\n".join(render_plan(data)))
    err = plan_error(data)
    if err:
        print(f"first error: {err}")
        sys.exit(1)
    print(plan_summary(data))


API_LOOKUP = r"""
import importlib, inspect, sys
dsl, name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
roots = {"triton": ["triton.language", "triton"], "cutile": ["cuda.tile"], "warp": ["warp"]}[dsl]
mods = [importlib.import_module(r) for r in roots]
if not name:
    names = sorted({n for m in mods for n in dir(m) if not n.startswith("_")})
    print(" ".join(names))
    sys.exit(0)
obj = None
for m in mods:
    o = m
    try:
        for part in name.split("."):
            o = getattr(o, part)
        obj = o
        break
    except AttributeError:
        continue
doc = ""
if dsl == "warp":
    for path in ("warp._src.context", "warp.context"):
        try:
            b = importlib.import_module(path).builtin_functions
            if name in b:
                f = b[name]
                doc = getattr(f, "doc", "") or ""
                for ov in getattr(f, "overloads", [])[:6]:
                    print("overload:", name, getattr(ov, "input_types", ""))
            break
        except Exception:
            pass
if obj is None and not doc:
    print(f"{name}: not found in {', '.join(roots)}")
    sys.exit(1)
if obj is not None:
    try:
        print(f"{name}{inspect.signature(obj)}")
    except (TypeError, ValueError):
        print(name)
    doc = doc or inspect.getdoc(obj) or ""
print(doc[:6000] if doc else "(no docstring)")
"""


def cmd_api(a):
    """Free: a DSL's own signature and docstring, from the installed version
    inside the arena environment, so the answer matches what will compile."""
    arena_cmd = a.arena or os.environ.get("ARENA_CMD") or DEFAULT_ARENA_CMD
    if "docker compose" in arena_cmd:
        base = ["docker", "compose", "run", "--rm", "-T", "--entrypoint", "python3",
                "-e", "PYTHONPATH=/workspace/gpgpu-arena/kernels", "cli"]
    else:
        base = [sys.executable]
    args = base + ["-c", API_LOOKUP, a.dsl] + ([a.name] if a.name else [])
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=600)
    noise = ("Container ", "Warp ", "   CUDA Toolkit", "   Devices", '     "cpu"', '     "cuda:0"',
             "   Kernel cache", "   CUDA")
    out = "\n".join(l for l in p.stdout.splitlines() if not l.startswith(noise))
    print(out.strip() or p.stderr.strip()[-2000:])
    sys.exit(p.returncode)


def candidate_source(run_dir, attempt):
    """The kernel source behind an attempt: the candidate file in a slot run,
    the non-manifest file of the candidate folder in a problem run."""
    p = REPO / attempt.get("candidate", "")
    if p.is_file() and not p.name.endswith(".json"):
        return p
    folder = p if p.is_dir() else p.parent
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file() and not f.name.endswith(".json") and dsl_of(f.name):
                return f
    return None


def source_diff(old_text, new_text, old_name, new_name, limit=200):
    lines = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                      old_name, new_name, lineterm="", n=2))
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines"]
    return "\n".join(lines)


def annotate_code(run_dir, attempts):
    """Adds "code" and "diff" to each attempt: its source, and the change
    against the previous attempt in the same DSL (the baseline for the first
    CUDA attempt), which is what reading the run as a progression needs."""
    last = {}
    out = []
    for a in attempts:
        a = dict(a)
        src = candidate_source(run_dir, a)
        text = src.read_text(errors="replace") if src else ""
        a["code"] = text
        a["source_file"] = str(src.relative_to(REPO)) if src else ""
        prev = last.get(a.get("dsl"))
        if prev is not None and text:
            a["diff"] = source_diff(prev[1], text, prev[0], a["source_file"])
            a["diff_against"] = prev[2]
        if text:
            last[a.get("dsl")] = (a["source_file"], text, a["iteration"])
        out.append(a)
    return out


def update_journal(run_dir, cfg):
    """Rewrites journal.md and progress.html from the attempt log, so both
    are always complete and can be watched while a run is going."""
    attempts = annotate_code(run_dir, load_attempts(run_dir))
    insp = []
    p = run_dir / "inspections.jsonl"
    if p.exists():
        insp = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    what = cfg["category"] if cfg.get("kind") == "problem" else cfg["kernel"]
    curve = best_so_far(attempts)
    lines = [f"# PGAIO journal: {what}", "",
             f"Run `{cfg['name']}` on {cfg.get('device', '')}. Mode {cfg.get('mode', 'isolated')}, "
             f"docs {'on' if cfg.get('docs') else 'off'}, target {cfg['target']:.0%} SOL. "
             f"Best so far {curve[-1]:.1%} after {len(attempts) - 1} iteration(s), "
             f"{len(insp)} free SASS inspection(s). Live view: [progress.html](progress.html).", ""]
    for hc in cfg.get("harness_changes", []):
        lines.append(f"> Harness changed after iteration {hc['after_iteration']}: {hc['reason']} "
                     f"({', '.join(hc['files'])})")
        lines.append("")
    for sk in cfg.get("skipped_dsls", []):
        lines.append(f"> {sk['dsl']} marked exhausted after iteration {sk.get('after_iteration')}: "
                     f"{sk.get('reason', '')}" + (f" (evidence: {sk['evidence']})" if sk.get("evidence") else ""))
        lines.append("")
    for r in load_ab(run_dir):
        o = r.get("overall") or {}
        lines.append(f"> A/B iteration {r.get('a')} (A) vs {r.get('b')} (B), {r.get('rounds')} rounds: "
                     f"{o.get('verdict', '')}")
        lines.append("")
    times = [a.get("time", "") for a in attempts]
    for i, a in enumerate(attempts):
        sc = f"{a['score']:.1%}" if a.get("score") is not None else "-"
        best = " (new best)" if a.get("new_best") else ""
        lines.append(f"## {a['iteration']}. {a.get('dsl') or ''}: {a['verdict']}, SOL {sc}{best}")
        lines.append("")
        lines.append(a.get("hypothesis", ""))
        lines.append("")
        if a.get("op_ms"):
            lines.append(f"op {a['op_ms']:.4f} ms, gpu {a.get('gpu_ms') or 0:.4f} ms"
                         + (f", score {fmt_ci(a['score_ci'])}" if a.get("score_ci") else "") + ".")
        if a.get("error"):
            lines.append(f"Error: `{a['error'][:200]}`")
        if a.get("source_file"):
            rel = os.path.relpath(REPO / a["source_file"], run_dir)
            lines.append(f"Source: [{Path(a['source_file']).name}]({rel}). "
                         f"Observation: [obs/{a['iteration']:03d}.json](obs/{a['iteration']:03d}.json).")
        # drafts inspected between the previous eval and this one
        lo = times[i - 1] if i > 0 else ""
        drafts = [Path(x["draft"]).name for x in insp if lo < x.get("time", "") <= times[i]]
        if drafts:
            lines.append(f"Inspected before this eval: {', '.join(drafts)}.")
        if a.get("diff"):
            lines += ["", f"<details><summary>diff against iteration {a['diff_against']}</summary>",
                      "", "```diff", a["diff"], "```", "", "</details>"]
        lines.append("")
    (run_dir / "journal.md").write_text("\n".join(lines))
    (run_dir / "progress.html").write_text(render_html([(cfg, attempts)]))


def cmd_rehash(a):
    """Accepts the current harness for this run after a deliberate change to
    arena or pgaio by its maintainer. Recorded in run.json and the journal,
    so it is never silent. Not for an optimizing agent: a candidate that
    needs a harness change has to say so instead."""
    run_dir, cfg = load_run(a.run)
    now_hashes = hash_protected()
    changed = guard_violations(cfg["protected_hashes"], now_hashes)
    if not changed:
        print("nothing changed")
        return
    cfg["protected_hashes"] = now_hashes
    cfg.setdefault("harness_changes", []).append(
        {"time": now(), "after_iteration": len(load_attempts(run_dir)) - 1,
         "reason": a.reason, "files": changed})
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))
    update_journal(run_dir, cfg)
    print(f"accepted {len(changed)} harness change(s): {', '.join(changed)}")


def cmd_rescore(a):
    """Recomputes every iteration's score from its stored raw timings with the
    run's current scoring settings, e.g. after fixing the launch floor."""
    run_dir, cfg = load_run(a.run)
    if a.launch_floor:
        cfg["launch_floor_ms"] = a.launch_floor
        (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))
    attempts = load_attempts(run_dir)
    best = None
    for att in attempts:
        p = run_dir / "obs" / f"{att['iteration']:03d}.json"
        if not p.exists() or att.get("verdict") != "ok":
            continue
        obs = json.loads(p.read_text())
        raw = obs.get("raw", {}).get("timing_results", [])
        if not raw:
            continue
        env = {**cfg.get("environment", {}), "launch_floor_ms": cfg.get("launch_floor_ms", 0.0),
               "roundtrip_floor_ms": cfg.get("roundtrip_floor_ms", 0.0)}
        per = [size_sol(r, env) for r in raw]
        old = att.get("score")
        att["score"] = obs["score"] = geomean([x["fraction"] for x in per])
        att["score_ci"] = obs["score_ci"] = add_spread(per, raw)
        add_clock(per, raw, env)
        obs["sizes"] = per
        if obs.get("raw", {}).get("inspect"):
            obs["sass_mix"] = sass_mix(obs["raw"]["inspect"])
        p.write_text(json.dumps(obs, indent=1))
        att["new_best"] = best is None or att["score"] > best
        best = att["score"] if att["new_best"] else best
        ci = f" ({fmt_ci(att['score_ci'])})" if att.get("score_ci") else ""
        print(f"iteration {att['iteration']}: {old:.1%} -> {att['score']:.1%}{ci}" if old is not None
              else f"iteration {att['iteration']}: {att['score']:.1%}{ci}")
    (run_dir / "attempts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in attempts))
    update_journal(run_dir, cfg)


def compare_column(run_dir, spec, attempts):
    """One column of `compare`: an iteration number, or ref:<lib> for a
    library reference (only the timings it has)."""
    spec = str(spec)
    if spec.startswith("ref:"):
        lib = spec[4:]
        p = run_dir / "references" / f"{lib}.json"
        if not p.exists():
            sys.exit(f"no reference {lib} (expected {p.relative_to(REPO)}; see pgaio_ncu.py reference)")
        ref = json.loads(p.read_text())
        obs = {"verdict": "reference", "score": ref.get("score"), "sizes": ref.get("sizes", []),
               "lib": lib, "kernel": ref.get("kernel")}
        return {"label": f"ref {lib}", "obs": obs, "att": {"dsl": ref.get("kernel")},
                "profiles": stored_profiles(run_dir, reference=lib)}
    try:
        it = int(spec)
    except ValueError:
        sys.exit(f"{spec}: expected an iteration number or ref:<lib>")
    obs = load_obs(run_dir, it)
    if obs is None:
        sys.exit(f"no observation for iteration {it}")
    att = next((x for x in attempts if x.get("iteration") == it), {})
    return {"label": f"iteration {it}", "obs": obs, "att": att,
            "profiles": stored_profiles(run_dir, iteration=it)}


def stored_profiles(run_dir, iteration=None, reference=None):
    """{shape key: summary} of the ncu summaries pgaio_ncu.py saved for an
    iteration or a library reference, at any shape. The folder name
    depends on the shape (profiles/016, profiles/016-K2048-M2048-N2048,
    profiles/ref-cublas-K2048-M2048-N2048), so the summaries are matched on
    what they say they are."""
    out = {}
    for p in sorted((Path(run_dir) / "profiles").glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if reference is not None:
            if s.get("reference") != reference:
                continue
        elif s.get("reference") or s.get("iteration") != iteration:
            continue
        out[pkey(s.get("params"))] = s
    return out


def shape_size(params):
    n = 1
    for v in (params or {}).values():
        n *= v if isinstance(v, (int, float)) else 1
    return n


def union_top(dicts, k=5):
    """Keys in the top k of any of the dicts, by the largest value."""
    keys = {}
    for d in dicts:
        for name, v in sorted(((n, v) for n, v in (d or {}).items() if isinstance(v, (int, float))),
                              key=lambda kv: -kv[1])[:k]:
            keys[name] = max(keys.get(name, 0), v)
    return [n for n, _ in sorted(keys.items(), key=lambda kv: -kv[1])]


def cmd_compare(a):
    """Free: two measured iterations (or one and a library reference) side
    by side."""
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    cols = [compare_column(run_dir, x, attempts) for x in (a.a, a.b)]
    refs = load_references(run_dir)

    def row(label, f):
        vals = []
        for c in cols:
            try:
                v = f(c["obs"], c)
                vals.append("-" if v is None else v)
            except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
                vals.append("-")
        if all(v == "-" for v in vals):
            return
        print(f"{label:28} {str(vals[0]):>24} {str(vals[1]):>24}")

    def size_of(o, params):
        # KeyError (shown as "-") when a column lacks the shape, e.g. a
        # reference timed at fewer shapes; next() would raise StopIteration,
        # which row() does not catch.
        for p in o["sizes"]:
            if pkey(p.get("params")) == pkey(params):
                return p
        raise KeyError(pkey(params))

    def op_cell(p):
        sp = p.get("spread") or {}
        return f"{p['op_ms']:.4f}" + (f" ±{sp['ci_half_frac']:.1%}" if sp.get("ci_half_frac") is not None else "") \
            + (" bimodal" if sp.get("bimodal") else "")

    print(f"{'':28} {cols[0]['label']:>24} {cols[1]['label']:>24}")
    row("dsl", lambda o, c: c["att"].get("dsl"))
    row("verdict", lambda o, c: o["verdict"])
    row("SOL", lambda o, c: f"{o['score']:.1%}")
    row("SOL 95% CI", lambda o, c: f"{o['score_ci'][0]:.1%}..{o['score_ci'][1]:.1%}")
    for lib, ref in sorted(refs.items()):
        row(f"% of {lib}", lambda o, c, ref=ref: f"{ref_relative(ref, o['sizes'])[0]:.1%}"
            if "lib" not in o else None)
    shapes = []
    for c in cols:
        for p in c["obs"].get("sizes") or []:
            if pkey(p.get("params")) not in [pkey(x) for x in shapes]:
                shapes.append(p.get("params"))
    for params in shapes:
        row(f"op_ms {shape_label(params)}", lambda o, c, params=params: op_cell(size_of(o, params)))
        row(f"  SOL {shape_label(params)}", lambda o, c, params=params: f"{size_of(o, params)['fraction']:.1%}")
        row(f"  clock MHz {shape_label(params)}",
            lambda o, c, params=params: f"{size_of(o, params)['clock']['sm_clock_mean_mhz']:.0f}")
    row("op_ms (largest)", lambda o, c: f"{o['timing']['op_ms']:.4f}")
    row("gpu_ms (largest)", lambda o, c: f"{o['timing']['gpu_ms']:.4f}")
    row("launches", lambda o, c: o["timing"]["launch_count"])
    row("sm clock mean MHz", lambda o, c: f"{o['quality']['sm_clock_mean_mhz']:.0f}")
    row("DRAM read MB", lambda o, c: f"{o['counters']['dram_read_bytes'] / 1e6:.1f}")
    row("occupancy", lambda o, c: f"{o['counters']['occupancy']:.0%}")
    row("issue active", lambda o, c: f"{o['counters']['issue_active']:.0%}")
    row("shared bank conflicts", lambda o, c: f"{o['counters']['shared_bank_conflicts']:.3g}")
    row("registers", lambda o, c: o["resources"]["registers"])
    row("local bytes", lambda o, c: o["resources"]["local"])
    for k in union_top([c["obs"].get("counters", {}).get("pipes") for c in cols], k=6):
        row(f"pipe {k}", lambda o, c, k=k: f"{o['counters']['pipes'][k]:.0%}")
    for k in union_top([c["obs"].get("counters", {}).get("stalls") for c in cols]):
        row(f"stall {k}", lambda o, c, k=k: f"{o['counters']['stalls'][k]:.2f}")
    for k in ["instructions", "global_loads", "global_load_128", "mean_global_load_bytes",
              "global_atomic", "warp_shuffle", "warp_reduce", "shared_mem", "sync", "tensor"]:
        row(f"sass {k}", lambda o, c, k=k: o["sass"][k])
    for c in cols:
        c["mix"] = c["obs"].get("sass_mix") or sass_mix((c["obs"].get("raw") or {}).get("inspect"))
    row("FFMA share", lambda o, c: f"{c['mix']['ffma_share']:.0%}")
    row("FMAs per LDS", lambda o, c: f"{c['mix']['ffma_per_lds']:.1f}")

    common = set(cols[0]["profiles"]) & set(cols[1]["profiles"])
    if common:
        # the largest shape both were profiled at
        key = max(common, key=lambda k: shape_size(cols[0]["profiles"][k].get("params")))
        profs = [c["profiles"][key] for c in cols]
        # summary.json keeps each stall as a share of the stall samples (0..1)
        print(f"\nncu stalls at {shape_label(profs[0].get('params'))}, share of stall samples "
              "(profiles/*/summary.json); diff in points:")
        print(f"{'':28} {cols[0]['label']:>24} {cols[1]['label']:>24} {'diff':>8}")
        for k in union_top([p.get("stalls") for p in profs]):
            va, vb = (p.get("stalls", {}).get(k, 0) or 0 for p in profs)
            print(f"{k:28} {va:>24.1%} {vb:>24.1%} {(vb - va) * 100:>+8.1f}")
        for k in ("issue_active", "clock_mhz", "duration_ms"):
            if all(p.get(k) is not None for p in profs):
                print(f"{k:28} {profs[0][k]:>24.4g} {profs[1][k]:>24.4g}")
    elif any(c["profiles"] for c in cols):
        have = [c["label"] + " at " + "; ".join(shape_label(s.get("params")) for s in c["profiles"].values())
                for c in cols if c["profiles"]]
        print(f"\nncu profile only for {', '.join(have)}; profile the other at the same shape "
              "to diff stalls")

    hyp = [(c["label"], c["att"].get("hypothesis", "")) for c in cols if c["att"].get("hypothesis")]
    if hyp:
        print("\nhypotheses:\n" + "\n".join(f"  {l}: {h}" for l, h in hyp))
    its = [c["label"].split()[-1] for c in cols if c["label"].startswith("iteration")]
    if len(its) == 2:
        sa, sb = (run_dir / "obs" / f"{int(its[0]):03d}.sass", run_dir / "obs" / f"{int(its[1]):03d}.sass")
        if sa.exists() and sb.exists():
            print(f"SASS: diff {sa.relative_to(REPO)} {sb.relative_to(REPO)}")


def ab_target(cfg, attempts, it):
    """arena arguments that run measured iteration `it` exactly as its eval
    did. Only passing iterations: a draft that was never measured, or one
    that failed its checks, has no score an A/B could revise."""
    att = next((x for x in attempts if x.get("iteration") == it), None)
    if att is None:
        raise ValueError(f"iteration {it} does not exist in this run")
    if att.get("verdict") != "ok":
        raise ValueError(f"iteration {it} is {att.get('verdict')}; ab compares passing iterations only")
    cand = att.get("candidate", "")
    if cfg.get("kind") == "problem":
        if it == 0 or not cand.endswith(".json"):
            return target_args(cfg)[0]            # the baseline kernel, as init ran it
        return target_args(cfg, manifest=cand)[0]
    # a slot run's iteration 0 is the slot's own source unless --start gave one
    own = it == 0 and cfg.get("start_source") == cfg.get("slot_source")
    return target_args(cfg, None if own else cand)[0]


def ab_commands(cfg, args_a, args_b, shapes, runs, rounds):
    """[[("a"|"b", argv), (...)], ...]: one arena process per candidate per
    round (arena takes one --manifest per call), the order alternating."""
    out = []
    for order in ab_round_order(rounds):
        out.append([(side, timing_args(cfg, args_a if side == "a" else args_b, shapes,
                                       runs=runs, profile=False)) for side in order])
    return out


def round_times(results):
    """{shape key: interquartile mean ms} from one timed arena call."""
    out = {}
    for r in results or []:
        t = r.get("timing", {})
        ms = t.get("op_ms_trimmed") or (trimmed_mean(t["all_times_ms"]) if t.get("all_times_ms") else None) \
            or t.get("op_ms")
        out[pkey(r.get("input", {}).get("params"))] = ms
    return out


def cmd_ab(a):
    """Free: times two passing iterations in alternating order over several
    rounds and says whether one is faster than the timing noise. Not an
    iteration; saved under <run>/ab/ and read by the plateau rule."""
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    if a.a == a.b:
        sys.exit("pass two different iterations")
    try:
        args_a, args_b = ab_target(cfg, attempts, a.a), ab_target(cfg, attempts, a.b)
        shapes = [parse_shape_spec(x) for x in a.shape] if a.shape else run_shapes(cfg)[0]
    except ValueError as e:
        sys.exit(str(e))
    if a.rounds < 3:
        # Two rounds give one degree of freedom: the t interval is so wide
        # that even a 10x difference is "not resolved".
        sys.exit("--rounds must be at least 3 for a useful confidence interval")
    plan = ab_commands(cfg, args_a, args_b, shapes, a.runs, a.rounds)
    base = shlex.split(cfg["arena_cmd"])
    if a.dry_run:
        for i, rnd in enumerate(plan):
            for side, argv in rnd:
                print(f"round {i + 1} {side.upper()}: {shlex.join(base + argv)}")
        return

    rounds = []
    for i, rnd in enumerate(plan):
        got = {}
        for side, argv in rnd:
            print(f"[pgaio] ab round {i + 1}/{len(plan)}: {side.upper()} "
                  f"(iteration {a.a if side == 'a' else a.b})", file=sys.stderr)
            code, data, _, err = arena(cfg["arena_cmd"], argv)
            res = (data or {}).get("results", [])
            bad = [r for r in res if not (r.get("success") and r.get("verified", True))]
            if not res or bad:
                msg = first_error(res) or (err or "").strip()[-1500:] or f"exit {code}"
                sys.exit(f"round {i + 1}, {side.upper()} failed: {msg}")
            got[side] = round_times(res)
        rounds.append(got)

    summary = ab_summary(rounds, shapes)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = {"a": a.a, "b": a.b, "time": now(), "rounds": len(rounds), "runs": a.runs,
           "shapes": shapes, "order": ab_round_order(len(rounds)), "per_round": rounds, **summary}
    path = run_dir / "ab" / f"{a.a}-vs-{a.b}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))

    dsl = {x.get("iteration"): x.get("dsl") for x in attempts}
    print(f"A = iteration {a.a} ({dsl.get(a.a)}), B = iteration {a.b} ({dsl.get(a.b)}); "
          f"{len(rounds)} rounds x {a.runs} runs, cold L2, order alternating")
    for sh in summary["shapes"]:
        k = pkey(sh["params"])
        ma = statistics.median(r["a"][k] for r in rounds)
        mb = statistics.median(r["b"][k] for r in rounds)
        print(f"  {shape_label(sh['params']):>16}: A {ma:.4f} ms, B {mb:.4f} ms: {sh['verdict']}")
    print(f"overall (geometric mean over shapes, as the score): {summary['overall']['verdict']}")
    print(f"saved: {path.relative_to(REPO)}")
    update_journal(run_dir, cfg)


def cmd_status(a):
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    curve = best_so_far(attempts)
    what = cfg["category"] if cfg.get("kind") == "problem" else cfg["kernel"]
    print(f"{cfg['name']}: {what} ({cfg['dsl']}, {cfg.get('mode', 'isolated')}, "
          f"docs {'on' if cfg.get('docs') else 'off'}), "
          f"{len(attempts) - 1}{'/' + str(cfg['budget']) if cfg.get('budget') else ''} "
          f"iterations used, target {cfg['target']:.0%}")
    print(settings_line(cfg))
    if cfg.get("kind") == "problem":
        per = {}
        for x in attempts[1:]:
            per[x.get("dsl")] = per.get(x.get("dsl"), 0) + 1
        cap = cfg.get("budget_per_dsl") or 0
        print("per DSL: " + ", ".join(f"{d} {per.get(d, 0)}" + (f"/{cap}" if cap else "")
                                      for d in cfg.get("dsls", [])))
        best = best_per_dsl(attempts)
        if best:
            print("best per DSL: " + ", ".join(f"{d} {v:.1%}" for d, v in sorted(best.items())))
    for sk in cfg.get("skipped_dsls", []):
        print(f"skipped DSL {sk['dsl']} after iteration {sk.get('after_iteration')}: {sk.get('reason', '')}"
              + (f" (evidence: {sk['evidence']})" if sk.get("evidence") else ""))
    ab = load_ab(run_dir)
    since, short, flat = plateau(cfg, attempts, ab)
    print(f"{since} eval(s) since {plateau_text(cfg)}"
          + (f"; DSLs with under {PLATEAU_MIN_PER_DSL} evals (and not skipped): {', '.join(short)}" if short else "")
          + ("; plateau reached" if flat else ""))
    for r in ab:
        print(f"A/B {r.get('a')} vs {r.get('b')} ({r.get('rounds')} rounds, {r.get('time', '')}): "
              f"{(r.get('overall') or {}).get('verdict', '')}")

    refs = load_references(run_dir)
    for lib, ref in sorted(refs.items()):
        sizes = ", ".join(f"{shape_label(x.get('params'))} {x.get('op_ms', 0):.4f} ms" for x in ref.get("sizes", []))
        print(f"reference {lib} ({ref.get('kernel', '')}): score {ref.get('score') or 0:.1%}; {sizes}")
    ref_cols = sorted(refs)
    print(f"{'it':>3} {'dsl':7} {'verdict':15} {'SOL':>7} {'±CI':>6} {'best':>7} {'op_ms':>10}"
          + "".join(f" {('%' + lib)[:8]:>8}" for lib in ref_cols) + "  hypothesis")
    for x, b in zip(attempts, curve):
        sc = f"{x['score']:.1%}" if x.get("score") is not None else "-"
        hw = f"{score_half_width(x):.1%}" if x.get("score_ci") else "-"
        op = f"{x['op_ms']:.4f}" if x.get("op_ms") else "-"
        rc = ""
        if ref_cols:
            obs = load_obs(run_dir, x["iteration"]) if x.get("verdict") == "ok" else None
            for lib in ref_cols:
                g = ref_relative(refs[lib], (obs or {}).get("sizes"))[0] if obs else None
                rc += f" {(f'{g:.1%}' if g is not None else '-'):>8}"
        print(f"{x['iteration']:>3} {str(x.get('dsl') or ''):7} {x['verdict']:15} {sc:>7} {hw:>6} {b:>7.1%} "
              f"{op:>10}{rc}  {x['hypothesis'][:70]}")


def cmd_best(a):
    run_dir, cfg = load_run(a.run)
    ok = [x for x in load_attempts(run_dir) if x["verdict"] == "ok" and x.get("score") is not None]
    if not ok:
        sys.exit("no passing iteration yet")
    b = max(ok, key=lambda x: x["score"])
    print(json.dumps({"iteration": b["iteration"], "candidate": b["candidate"], "dsl": b.get("dsl"),
                      "score": b["score"], "op_ms": b["op_ms"], "hypothesis": b["hypothesis"],
                      "slot_source": cfg["slot_source"],
                      "descriptor_file": cfg["descriptor_file"]}, indent=2))


def matching_learnings(category, dsl, limit=15):
    if not LEARNINGS.exists():
        return []
    rows = [json.loads(l) for l in LEARNINGS.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("category") == category or r.get("dsl") == dsl]
    return rows[-limit:]


def git_tracked(path, repo=None):
    """Whether git tracks path (a file that may no longer exist on disk)."""
    repo = Path(repo or REPO)
    try:
        rel = str(Path(path).resolve().relative_to(repo.resolve()))
        p = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo,
                           capture_output=True, text=True)
    except (OSError, ValueError):
        return False
    return p.returncode == 0


def learnings_refusal(path, repo=None, create=False):
    """Why `learn` must not write path, or None. A tracked learnings file
    that is missing was deleted by accident (or by a checkout); appending
    would start a new file holding one line and lose the history."""
    path = Path(path)
    if create or path.exists() or not git_tracked(path, repo):
        return None
    rel = os.path.relpath(path, repo or REPO)
    return (f"{rel} is tracked in git but missing from the working tree. Restore it with "
            f"`git checkout -- {rel}`, or pass --create to start a new file on purpose.")


def cmd_learn(a):
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    why = learnings_refusal(LEARNINGS, REPO, a.create)
    if why:
        sys.exit(why)
    append_jsonl(LEARNINGS, {
        "time": now(), "run": cfg["name"], "kernel": cfg["kernel"],
        "category": cfg["category"], "dsl": cfg["dsl"], "device": cfg.get("device", ""),
        "insight": a.insight, "evidence": a.evidence or "",
        "best_sol": max(best_so_far(attempts) or [0]),
    })
    print(f"recorded in {LEARNINGS.relative_to(REPO)}")


def skip_entry(cfg, attempts, dsl, reason, evidence=""):
    """The run.json record of a DSL marked exhausted; raises ValueError for
    a DSL the run does not allow."""
    allowed = cfg.get("dsls", [])
    if cfg.get("kind") != "problem":
        raise ValueError("skip-dsl is for problem runs, which have several DSLs")
    if dsl not in allowed:
        raise ValueError(f"{dsl} is not one of this run's DSLs ({', '.join(allowed)})")
    if not reason.strip():
        raise ValueError("say why the DSL is a dead end (--reason)")
    return {"dsl": dsl, "reason": reason, "evidence": evidence or "", "time": now(),
            "after_iteration": max([x.get("iteration", 0) for x in attempts] or [0])}


def cmd_skip_dsl(a):
    """Marks a DSL as exhausted: it then counts as having its minimum evals
    for the plateau rule. Evals in it stay allowed. Recorded, never silent."""
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    try:
        entry = skip_entry(cfg, attempts, a.dsl, a.reason, a.evidence)
    except ValueError as e:
        sys.exit(str(e))
    n = sum(1 for x in attempts if x.get("dsl") == a.dsl and x.get("iteration", 0) > 0)
    cfg.setdefault("skipped_dsls", []).append(entry)
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))
    update_journal(run_dir, cfg)
    print(f"{a.dsl} marked exhausted after {n} eval(s) (iteration {entry['after_iteration']}); "
          "it now counts as tried for the plateau rule. Evals in it are still allowed.")


def collect_rows(pattern):
    rows = []
    for cfg_path in sorted(RUNS_DIR.glob(f"{pattern}/run.json")):
        cfg = json.loads(cfg_path.read_text())
        attempts = load_attempts(cfg_path.parent)
        if attempts:
            row = summarize_run(cfg, attempts)
            row["vs_ref"] = best_vs_references(cfg_path.parent, attempts)
            rows.append(row)
            rows.extend(summarize_dsls(cfg, attempts))
    return rows


def best_vs_references(run_dir, attempts):
    """"92.1% of cublas" for the run's best iteration, or "" without
    references."""
    refs = load_references(run_dir)
    ok = [x for x in attempts if x.get("verdict") == "ok" and x.get("score") is not None]
    if not refs or not ok:
        return ""
    obs = load_obs(run_dir, max(ok, key=lambda x: x["score"])["iteration"])
    return ref_text(refs, (obs or {}).get("sizes"))


def fmt_it(v):
    return "-" if v is None else str(v)


def cmd_report(a):
    rows = collect_rows(a.runs)
    if not rows:
        sys.exit(f"no runs under {RUNS_DIR.relative_to(REPO)}/{a.runs}")
    th_cols = [f"iters_to_{int(t * 100)}" for t in SOL_THRESHOLDS]
    print("| run | group | kernel | iters | failed | baseline SOL | best SOL | vs reference | " +
          " | ".join(f"to {int(t * 100)}%" for t in SOL_THRESHOLDS) + " | skipped DSLs |")
    print("|" + " --- |" * (9 + len(SOL_THRESHOLDS)))
    for r in rows:
        print(f"| {r['run']} | {group_key(r)} | {r['kernel']} | {r['iterations']} | {r.get('failed', 0)} | "
              f"{r['baseline_sol']:.1%} | {r['best_sol']:.1%} | {r.get('vs_ref') or '-'} | " +
              " | ".join(fmt_it(r[c]) for c in th_cols) + f" | {', '.join(r.get('skipped_dsls') or []) or '-'} |")
    print()
    print("| group | runs | median baseline | median best | " +
          " | ".join(f"median to {int(t * 100)}% (reached)" for t in SOL_THRESHOLDS) + " |")
    print("|" + " --- |" * (4 + len(SOL_THRESHOLDS)))
    for dsl, g in aggregate(rows).items():
        print(f"| {dsl} | {g['runs']} | {g['median_baseline_sol']:.1%} | {g['median_best_sol']:.1%} | " +
              " | ".join(f"{fmt_it(g[c])} ({g[c + '_reached']})" for c in th_cols) + " |")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("run,dsl,mode,docs,kernel,model,iteration,best_sol\n")
            for r in rows:
                for i, v in enumerate(r["curve"]):
                    f.write(f"{r['run']},{r['dsl']},{r['mode']},{int(r['docs'])},{r['kernel']},"
                            f"{r['model']},{i},{v:.6f}\n")
        print(f"\ncurves written to {a.csv}")
    if a.html:
        runs = []
        for cfg_path in sorted(RUNS_DIR.glob(f"{a.runs}/run.json")):
            cfg = json.loads(cfg_path.read_text())
            attempts = load_attempts(cfg_path.parent)
            if attempts:
                runs.append((cfg, annotate_code(cfg_path.parent, attempts)))
        Path(a.html).write_text(render_html(runs))
        print(f"\nprogression page written to {a.html}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="start a run and measure its baseline")
    i.add_argument("--category", help="problem run: optimize this category in any DSL, "
                   "candidates are manifests")
    i.add_argument("--kernel", help="slot run: optimize this one kernel's source")
    i.add_argument("--baseline", help="problem run: kernel measured as iteration 0 "
                   "(default: the category's plainest kernel)")
    i.add_argument("--dsls", help="problem run: comma list of allowed DSLs (default all)")
    i.add_argument("--budget-per-dsl", type=int, default=0,
                   help="problem run: iterations allowed per DSL (default 0: no per-DSL cap)")
    i.add_argument("--budget", type=int, default=0,
                   help="total iterations after the baseline (default 0: no limit; the run "
                        "ends at the target or on a plateau)")
    i.add_argument("--target", type=float, default=1.0,
                   help="SOL fraction to aim for (default 1.0: as close to the roof as it gets)")
    i.add_argument("--runs", type=int, default=100,
                   help="timed runs per shape; many, because microsecond timings are noisy")
    i.add_argument("--seeds", type=int, default=2, help="held-out seeds per check")
    i.add_argument("--sizes", choices=["small", "large", "all"], default="all",
                   help="which sizes to optimize for: small (latency and parallelism), large "
                        "(several times L2: bandwidth or FLOPs) or all (default)")
    i.add_argument("--train", action="append",
                   help="an explicit timed shape instead of --sizes, e.g. rows=64,cols=1024 (repeatable)")
    i.add_argument("--check", action="append", help="checked sizes, e.g. n=1,1000003")
    i.add_argument("--define", action="append", help="compile define KEY=n for the slot")
    i.add_argument("--start", help="source to start from instead of the slot's own "
                   "(same file type); becomes iteration 0")
    i.add_argument("--name", help="run directory name")
    i.add_argument("--model", help="model doing the optimizing, recorded for the report")
    i.add_argument("--docs", choices=["on", "off"], default="off",
                   help="whether a CUDA documentation tool (MCP) is available to the agent. "
                        "Recorded for the report.")
    i.add_argument("--mode", choices=["isolated", "cross-dsl"], default="cross-dsl",
                   help="cross-dsl (default): existing kernels' code, SASS and timings may be "
                        "used as hints. isolated: only the run's own candidates. Recorded for "
                        "the report.")
    i.add_argument("--forbid-tensor-cores", action="store_true",
                   help="reject (as rejected_guard) any candidate whose SASS has tensor-core "
                        "instructions (HMMA, HGMMA, UTCHMMA, QMMA, ...)")
    i.add_argument("--scratch-policy", choices=["legacy", "declared"],
                   help="arena scratch handling, fixed for the run: legacy (default) zeroes every "
                        "scratch buffer outside the timed op; declared zeroes only entries marked "
                        "\"zero\": true, inside the timed op, and fills the rest with 0xFF")
    i.add_argument("--counters", choices=["basic", "full"],
                   help="counter level with --profile (default full: adds issue active, pipes, "
                        "stalls, shared bank conflicts)")
    i.add_argument("--lock-clocks", metavar="MHZ|auto",
                   help="ask arena to lock the SM clock while timing (default: no lock)")
    i.add_argument("--arena", help=f"command that runs arena (default: $ARENA_CMD or '{DEFAULT_ARENA_CMD}')")
    i.set_defaults(fn=cmd_init)

    e = sub.add_parser("eval", help="measure one candidate (counts as an iteration)")
    e.add_argument("run")
    e.add_argument("candidate", help="a manifest (problem run) or a source (slot run)")
    e.add_argument("--hypothesis", required=True, help="what this change should do and why")
    e.set_defaults(fn=cmd_eval)

    x = sub.add_parser("sass", help="compile a draft and show its SASS mix; free, never timed")
    x.add_argument("run")
    x.add_argument("candidate", help="a manifest (problem run) or a source (slot run)")
    x.add_argument("--show", help="also list SASS lines whose mnemonic starts with these, "
                   "comma separated, e.g. LDG,STG,BAR")
    x.set_defaults(fn=cmd_sass)

    ap = sub.add_parser("api", help="look up a DSL's Python API inside the arena container; free")
    ap.add_argument("dsl", choices=["triton", "cutile", "warp"])
    ap.add_argument("name", nargs="?", help="e.g. load, tile_load, language.max; omit to list names")
    ap.add_argument("--arena", help="command that runs arena, as for init")
    ap.set_defaults(fn=cmd_api)

    rh = sub.add_parser("rehash", help="accept a deliberate harness change for a run (maintainers only, recorded)")
    rh.add_argument("run")
    rh.add_argument("--reason", required=True)
    rh.set_defaults(fn=cmd_rehash)

    rs = sub.add_parser("rescore", help="recompute scores from stored timings (e.g. after fixing the launch floor)")
    rs.add_argument("run")
    rs.add_argument("--launch-floor", type=float, help="launch floor in ms to use from now on")
    rs.set_defaults(fn=cmd_rescore)

    c = sub.add_parser("compare", help="two measured iterations, or one and a library reference, side by side; free")
    c.add_argument("run")
    c.add_argument("a", help="an iteration, or ref:<lib> for a library reference")
    c.add_argument("b", help="an iteration, or ref:<lib> for a library reference")
    c.set_defaults(fn=cmd_compare)

    ab = sub.add_parser("ab", help="time two passing iterations in alternating rounds and say "
                        "whether one is faster beyond the noise; free, not an iteration")
    ab.add_argument("run")
    ab.add_argument("a", type=int, help="iteration A")
    ab.add_argument("b", type=int, help="iteration B")
    ab.add_argument("--shape", action="append",
                    help="a shape to time, e.g. K=4096,M=4096,N=4096 (repeatable; default: the timed shapes)")
    ab.add_argument("--rounds", type=int, default=6, help="rounds, each timing A and B once (default 6, at least 3)")
    ab.add_argument("--runs", type=int, default=30, help="timed runs per shape per process (default 30)")
    ab.add_argument("--dry-run", action="store_true", help="print the arena command lines and stop")
    ab.set_defaults(fn=cmd_ab)

    ck = sub.add_parser("check", help="where each launch of a manifest goes at every shape "
                        "(arena --plan); free, problem runs")
    ck.add_argument("run")
    ck.add_argument("manifest")
    ck.set_defaults(fn=cmd_check)

    sk = sub.add_parser("skip-dsl", help="mark a DSL exhausted: it then counts as tried for the plateau rule")
    sk.add_argument("run")
    sk.add_argument("dsl")
    sk.add_argument("--reason", required=True, help="why more evals in it would be wasted")
    sk.add_argument("--evidence", help="iterations or measurements behind the reason")
    sk.set_defaults(fn=cmd_skip_dsl)

    s = sub.add_parser("status", help="history of a run")
    s.add_argument("run")
    s.set_defaults(fn=cmd_status)

    b = sub.add_parser("best", help="best passing candidate of a run")
    b.add_argument("run")
    b.set_defaults(fn=cmd_best)

    l = sub.add_parser("learn", help="record an insight for future runs")
    l.add_argument("run")
    l.add_argument("--insight", required=True)
    l.add_argument("--evidence")
    l.add_argument("--create", action="store_true",
                   help="start a new learnings file even though git tracks a missing one")
    l.set_defaults(fn=cmd_learn)

    r = sub.add_parser("report", help="iterations-to-SOL across runs, per DSL")
    r.add_argument("--runs", default="*", help="glob over run names")
    r.add_argument("--csv", help="write best-so-far curves here")
    r.add_argument("--html", help="write a progression page here")
    r.set_defaults(fn=cmd_report)

    # ncu profiles and library references live in their own module; a
    # checkout without it (or with a broken copy) still has every command here.
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        import pgaio_ncu
        pgaio_ncu.add_parsers(sub)
    except ImportError:
        pass
    except Exception as e:   # noqa: BLE001
        print(f"[pgaio] pgaio_ncu.py not loaded: {e}", file=sys.stderr)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

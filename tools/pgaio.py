#!/usr/bin/env python3
"""Profile and benchmark guided AI optimization (PGAIO) driver for GPGPU Arena.

An agent optimizes one kernel slot by writing candidate sources. This script
is the environment around it: it measures each candidate the same way, refuses
the ways a candidate can cheat, keeps the history, and turns raw counters into
a short diagnosis the agent can act on.

    pgaio.py init   --kernel reduce_baseline [--budget 20] [--target 0.9]
    pgaio.py eval   <run> <candidate file> --hypothesis "vectorize loads"
    pgaio.py status <run>
    pgaio.py best   <run>
    pgaio.py learn  <run> --insight "..." [--evidence "..."]
    pgaio.py report [--csv curves.csv]

Every eval counts as one iteration, including ones that fail to compile, fail
correctness, or touch a protected file. That is what makes iterations-to-SOL
comparable across DSLs.

Only the standard library is used, so this runs on the host while arena runs
in Docker.
"""

import argparse
import datetime as _dt
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

# Per-category sizes. "train" is what gets timed and scored. "check" is where
# correctness is gated: odd sizes catch missing bounds checks, tiny ones catch
# grid math, and the train size is always included. Sizes the baseline itself
# fails on are dropped at init, so the gate never demands more than the slot
# could ever deliver.
CATEGORIES = {
    "reduce":  {"train": {"n": [67108864]},
                "check": {"n": [1, 1000, 1000003, 16777217]}},
    "scan":    {"train": {"n": [67108864]},
                "check": {"n": [1, 1000, 1000003, 16777217]}},
    "softmax": {"train": {"rows": [4096], "cols": [4096]},
                "check": {"rows": [1, 333], "cols": [1000, 4097]}},
    "matmul":  {"train": {"M": [4096], "N": [4096], "K": [4096]},
                "check": {"M": [129], "N": [257], "K": [63, 512]}},
}
CHECK_DISTS = ["uniform", "normal", "adversarial"]
SOL_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]


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
           "unit": "", "intensity": None,
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
    if s.get("bound") == "memory":
        out.append(f"memory-bound: {frac:.1%} of the DRAM roof "
                   f"({s.get('achieved', 0):.0f} of {s.get('roof', 0):.0f} GB/s).")
    elif s.get("bound") == "compute":
        out.append(f"compute-bound: {frac:.1%} of the FP32 roof "
                   f"({s.get('achieved', 0):.0f} of {s.get('roof', 0):.0f} GFLOP/s).")

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

    if r.get("local", 0) > 0 or b.get("local_mem", 0) > 0:
        out.append(f"{r.get('local', 0)} B of local memory per thread: register spills or a "
                   "local array. Both go through L1/L2 and cost bandwidth.")

    if s.get("bound") == "memory" and frac < 0.7 and 0 < b.get("mean_global_load_bytes", 0) < 8:
        out.append(f"global loads average {b['mean_global_load_bytes']:.0f} B "
                   "(SASS). 128-bit loads cut load instructions 4x for fp32.")

    if b.get("global_atomic", 0) > 0 and s.get("bound") == "memory" and frac < 0.3:
        out.append(f"{b['global_atomic']} global atomic instruction(s) in SASS with low SOL: "
                   "if they run per element rather than per block, contention serializes them.")

    if obs.get("category") == "matmul" and b.get("tensor", 0) == 0:
        out.append("no tensor core instructions in SASS.")

    if frac >= 0.85:
        out.append("within 15% of the roof: further gains will be small.")
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
        "baseline_sol": curve[0] if curve else 0.0,
        "best_sol": curve[-1] if curve else 0.0,
        "curve": curve,
    }
    for th in SOL_THRESHOLDS:
        row[f"iters_to_{int(th * 100)}"] = iterations_to(curve, th)
    return row


def group_key(row):
    """"cuda/isolated", or "cuda/isolated+docs" when documentation tools were
    available. Runs are only comparable within one group."""
    key = f"{row['dsl']}/{row.get('mode', 'isolated')}"
    return key + "+docs" if row.get("docs") else key


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
    p = subprocess.run(full, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    data = None
    out = p.stdout.strip()
    if out.startswith("{"):
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            data = None
    return p.returncode, data, p.stdout, p.stderr


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


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def measure(cfg, rel_source, label):
    """Correctness gate, cold-L2 timing and SASS inspection for one source."""
    cmd = cfg["arena_cmd"]
    ovr = ["--source-override", rel_source] if rel_source else []
    text = (REPO / (rel_source or cfg["slot_source"])).read_text(errors="replace")
    block = block_directive(text)
    if block:
        ovr += ["--block", str(block)]
    defines = []
    for k, v in sorted(cfg.get("defines", {}).items()):
        defines += ["--define", f"{k}={v}"]

    # 1. correctness over held-out sizes, distributions and seeds
    chk_args = (["--run", cfg["kernel"]] + ovr + defines + params_args(cfg["check_params"]) +
                ["--dist", ",".join(cfg["check_dists"]),
                 "--seed", ",".join(str(s) for s in cfg["check_seeds"]),
                 "--runs", "1", "--warmup", "1"])
    chk_exit, chk, _, chk_err = arena(cmd, chk_args)
    chk_results = (chk or {}).get("results", [])
    required = [k for k in cfg.get("required_checks", [])]

    # 2. timing on the train size, cold L2, counters if available
    tim_results, tim_exit, env, tim_err = [], None, (chk or {}).get("environment", {}), ""
    verdict_so_far = classify_attempt(chk_results, [], chk_exit, 0, [], required)
    if verdict_so_far == "ok":
        tim_args = (["--run", cfg["kernel"]] + ovr + defines + params_args(cfg["train_params"]) +
                    ["--l2", "cold", "--runs", str(cfg.get("runs", 30))])
        if cfg.get("profile"):
            tim_args.append("--profile")
        tim_exit, tim, _, tim_err = arena(cmd, tim_args)
        tim_results = (tim or {}).get("results", [])
        env = (tim or {}).get("environment", env)

    # 3. static view of what the compiler made of it
    insp = None
    sass_text = ""
    if verdict_so_far == "ok":
        code, insp, _, _ = arena(cmd, ["--inspect", cfg["kernel"]] + ovr + defines)
        if code != 0:
            insp = None
        code, _, sass_text, _ = arena(cmd, ["--sass", cfg["kernel"]] + ovr + defines)
        if code != 0:
            sass_text = ""

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
    obs = {"verdict": verdict, "category": cfg["category"], "guard_violations": guard}
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
    obs["error"] = err[-3000:] if err else ""

    if verdict != "ok":
        obs["score"] = None
        return obs

    rs = m["timing_results"]
    sols = [sol(r, m["environment"]) for r in rs]
    i = min(range(len(sols)), key=lambda k: sols[k]["fraction"])   # worst train size scores
    r, s = rs[i], sols[i]
    obs["score"] = s["fraction"]
    obs["sol"] = s
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
    obs["diagnosis"] = diagnose(obs)
    return obs


def render(obs, it, cfg, best, baseline):
    """Short text for the agent. Everything in it is in the JSON too."""
    lines = [f"iteration {it}/{cfg['budget']}  verdict: {obs['verdict'].upper()}"
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
    line = f"time: op {t['op_ms']:.4f} ms, gpu {t['gpu_ms']:.4f} ms, {t['launch_count']} launch(es)"
    if baseline and baseline.get("timing"):
        line += f"  | {baseline['timing']['op_ms'] / t['op_ms']:.2f}x vs baseline"
    if best and best.get("timing") and not obs.get("new_best"):
        line += f", {best['timing']['op_ms'] / t['op_ms']:.2f}x vs best (iter {best['iteration']})"
    lines.append(line)
    ai = s.get("intensity")
    lines.append(f"SOL: {s['fraction']:.1%} ({s['bound']}-bound, {s['achieved']:.0f}/{s['roof']:.0f} "
                 f"{s['unit']}, intensity {ai:.3f} FLOP/B vs ridge {s['ridge']:.1f})"
                 if ai is not None and s.get("ridge") else f"SOL: {s['fraction']:.1%}")
    lines.append(f"target {cfg['target']:.0%}" + (", reached" if s["fraction"] >= cfg["target"] else ""))
    c = obs["counters"]
    if c.get("available"):
        lines.append(f"counters: DRAM read {c['dram_read_bytes'] / 1e6:.1f} MB, write "
                     f"{c['dram_write_bytes'] / 1e6:.1f} MB (ideal {obs['ideal_bytes'] / 1e6:.1f} MB), "
                     f"occupancy {c['occupancy']:.0%}, IPC {c['ipc']:.2f}")
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

def cmd_init(a):
    arena_cmd = a.arena or os.environ.get("ARENA_CMD") or DEFAULT_ARENA_CMD

    # First call builds (Docker entrypoint) and describes the slot.
    print(f"[pgaio] inspecting {a.kernel} via: {arena_cmd}", file=sys.stderr)
    code, insp, out, err = arena(arena_cmd, ["--inspect", a.kernel], timeout=3600)
    if code != 0 or not insp:
        sys.exit(f"--inspect {a.kernel} failed (exit {code}):\n{err[-3000:]}")
    if "docker compose" in arena_cmd and "ARENA_SKIP_BUILD" not in arena_cmd:
        # The binary is built now; later calls skip the cmake check.
        arena_cmd = arena_cmd.replace("run --rm", "run --rm -e ARENA_SKIP_BUILD=1", 1)

    root = insp.get("repo_root", "")
    slot_source = repo_rel(insp.get("slot_source", ""), root)
    if not slot_source:
        sys.exit(f"{a.kernel} has no source slot (it is compiled into the binary)")
    category = insp["category"]
    defaults = CATEGORIES.get(category, {"train": {}, "check": {}})

    def parse_p(specs, fallback):
        if not specs:
            return {k: list(v) for k, v in fallback.items()}
        out = {}
        for s in specs:
            k, _, v = s.partition("=")
            out[k] = [int(x) for x in v.split(",") if x]
        return out

    train = parse_p(a.train, defaults["train"])
    check = parse_p(a.check, defaults["check"])
    for k, vs in train.items():   # the train size is always checked too
        check.setdefault(k, [])
        for v in vs:
            if v not in check[k]:
                check[k].append(v)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = a.name or f"{stamp}-{a.kernel}"
    run_dir = RUNS_DIR / name
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists")
    (run_dir / "candidates").mkdir(parents=True)
    (run_dir / "obs").mkdir()
    (run_dir / "work").mkdir()   # where the agent drafts candidates

    seed_base = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    cfg = {
        "name": name, "created": now(),
        "kernel": a.kernel, "category": category, "dsl": insp.get("dsl"),
        "entry_point": insp.get("entry_point"),
        "slot_source": slot_source,
        "descriptor_file": repo_rel(insp.get("descriptor_file", ""), root),
        "arena_cmd": arena_cmd,
        "budget": a.budget, "target": a.target, "runs": a.runs, "mode": a.mode,
        "docs": a.docs == "on",
        "train_params": train, "check_params": check,
        "check_dists": CHECK_DISTS,
        "check_seeds": [seed_base % 100000 + i for i in range(a.seeds)],
        "defines": dict(kv.split("=") for kv in (a.define or [])),
        "model": a.model or os.environ.get("PGAIO_MODEL", ""),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
        "profile": True,
        "protected_hashes": hash_protected(),
    }
    cfg["defines"] = {k: int(v) for k, v in cfg["defines"].items()}

    # Baseline: the slot's own source, copied in as iteration 0.
    # Baseline: the slot's own source, or --start, copied in as iteration 0.
    start = Path(a.start).resolve() if a.start else REPO / slot_source
    if a.start and candidate_ext(start.name) != candidate_ext(slot_source):
        shutil.rmtree(run_dir)
        sys.exit(f"--start must be a {candidate_ext(slot_source)} file for slot {a.kernel}")
    base_copy = run_dir / "candidates" / f"000_baseline{candidate_ext(slot_source)}"
    shutil.copy2(start, base_copy)
    base_rel = str(base_copy.relative_to(REPO)) if a.start else None
    cfg["start_source"] = str(start.relative_to(REPO)) if start.is_relative_to(REPO) else str(start)

    print(f"[pgaio] measuring baseline {cfg['start_source']} ...", file=sys.stderr)
    m = measure(cfg, base_rel, "baseline")
    if not m["timing_results"] or not all(r.get("success") for r in m["timing_results"]):
        # Counters unavailable (no perf access) is not a reason to stop.
        cfg["profile"] = False
        m = measure(cfg, base_rel, "baseline")

    # Drop checks the slot itself cannot pass, so the gate asks for nothing
    # the original kernel does not deliver.
    passing = [r for r in m["check_results"] if r.get("success") and r.get("verified")]
    dropped = sorted({combo_key(r) for r in m["check_results"]} - {combo_key(r) for r in passing})
    cfg["required_checks"] = sorted({combo_key(r) for r in passing})
    cfg["device"] = m["environment"].get("device_name", "")
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
    record(run_dir, cfg, 0, base_copy, "baseline: the slot's own source", obs, m)

    print(f"run: {run_dir.relative_to(REPO)}  (draft candidates in {(run_dir / 'work').relative_to(REPO)}/)")
    print(f"kernel {a.kernel} ({cfg['dsl']}), slot {slot_source}, entry point "
          f"{cfg['entry_point']}, descriptor {cfg['descriptor_file']}")
    print(f"mode {cfg['mode']}, docs {'on' if cfg['docs'] else 'off'}, budget {cfg['budget']} iterations, target {cfg['target']:.0%} SOL, "
          f"train {train}, counters {'on' if cfg['profile'] else 'off'}")
    if dropped:
        print(f"dropped {len(dropped)} checks the baseline fails: {', '.join(dropped[:6])}")
    print(render(obs, 0, cfg, None, None))
    ls = matching_learnings(category, cfg["dsl"])
    if ls:
        print("\nlearnings from earlier runs (same category or DSL):")
        for l in ls:
            print(f"  - [{l.get('dsl')}/{l.get('category')}, {l.get('device', '')}] {l['insight']}")


def record(run_dir, cfg, it, cand_path, hypothesis, obs, m):
    (run_dir / "obs" / f"{it:03d}.json").write_text(json.dumps(
        {**obs, "raw": {k: m[k] for k in ("check_results", "timing_results", "inspect")}}, indent=1))
    if m.get("sass_text"):
        (run_dir / "obs" / f"{it:03d}.sass").write_text(m["sass_text"])
    append_jsonl(run_dir / "attempts.jsonl", {
        "iteration": it, "time": now(),
        "candidate": str(Path(cand_path).relative_to(REPO)),
        "hypothesis": hypothesis, "verdict": obs["verdict"], "score": obs.get("score"),
        "op_ms": obs.get("timing", {}).get("op_ms"),
        "gpu_ms": obs.get("timing", {}).get("gpu_ms"),
        "new_best": bool(obs.get("new_best")),
        "sass": obs.get("sass"),
        "error": (obs.get("error") or "")[:300],
    })


def cmd_eval(a):
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    it = len(attempts)
    if it > cfg["budget"]:
        sys.exit(f"budget of {cfg['budget']} iterations is used up; see `pgaio.py status`")

    src = Path(a.candidate)
    if not src.is_absolute():
        src = (Path.cwd() / src)
    if not src.exists():
        sys.exit(f"candidate not found: {src}")
    want = candidate_ext(cfg["slot_source"])
    if candidate_ext(src.name) != want:
        sys.exit(f"candidate must be a {want} file to fit slot {cfg['kernel']} "
                 "(this did not count as an iteration)")

    snap = run_dir / "candidates" / f"{it:03d}_{slugify(a.hypothesis)}{want}"
    shutil.copy2(src, snap)
    rel = str(snap.relative_to(REPO))

    guard = guard_violations(cfg["protected_hashes"], hash_protected())
    if guard:
        m = {k: [] if k.endswith("results") else None for k in
             ("check_results", "timing_results", "inspect")}
        m.update(check_exit=None, timing_exit=None, check_stderr="", timing_stderr="",
                 environment={}, sass_text="")
    else:
        m = measure(cfg, rel, f"iter {it}")
    obs = observe(cfg, m, guard)
    obs["iteration"] = it

    ok = [x for x in attempts if x["verdict"] == "ok" and x.get("score") is not None]
    best_att = max(ok, key=lambda x: x["score"]) if ok else None
    best_obs = json.loads((run_dir / "obs" / f"{best_att['iteration']:03d}.json").read_text()) \
        if best_att else None
    base_obs = json.loads((run_dir / "obs" / "000.json").read_text()) \
        if (run_dir / "obs" / "000.json").exists() else None
    if obs["verdict"] == "ok":
        obs["new_best"] = best_att is None or obs["score"] > best_att["score"]
        if best_obs:
            obs["sass_vs_best"] = sass_delta(obs.get("sass"), best_obs.get("sass"))

    record(run_dir, cfg, it, snap, a.hypothesis, obs, m)
    print(render(obs, it, cfg, best_obs, base_obs))
    left = cfg["budget"] - it
    reached = obs.get("score") is not None and obs["score"] >= cfg["target"]
    print(f"\n{left} iteration(s) left" + ("; target reached, you may stop" if reached else ""))
    print(f"snapshot: {rel}   full observation: {(run_dir / 'obs' / f'{it:03d}.json').relative_to(REPO)}")


def cmd_status(a):
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    curve = best_so_far(attempts)
    print(f"{cfg['name']}: {cfg['kernel']} ({cfg['dsl']}, {cfg.get('mode', 'isolated')}, "
          f"docs {'on' if cfg.get('docs') else 'off'}), "
          f"{len(attempts) - 1}/{cfg['budget']} "
          f"iterations used, target {cfg['target']:.0%}")
    print(f"{'it':>3} {'verdict':15} {'SOL':>7} {'best':>7} {'op_ms':>10}  hypothesis")
    for x, b in zip(attempts, curve):
        sc = f"{x['score']:.1%}" if x.get("score") is not None else "-"
        op = f"{x['op_ms']:.4f}" if x.get("op_ms") else "-"
        print(f"{x['iteration']:>3} {x['verdict']:15} {sc:>7} {b:>7.1%} {op:>10}  {x['hypothesis'][:70]}")


def cmd_best(a):
    run_dir, cfg = load_run(a.run)
    ok = [x for x in load_attempts(run_dir) if x["verdict"] == "ok" and x.get("score") is not None]
    if not ok:
        sys.exit("no passing iteration yet")
    b = max(ok, key=lambda x: x["score"])
    print(json.dumps({"iteration": b["iteration"], "candidate": b["candidate"],
                      "score": b["score"], "op_ms": b["op_ms"], "hypothesis": b["hypothesis"],
                      "slot_source": cfg["slot_source"],
                      "descriptor_file": cfg["descriptor_file"]}, indent=2))


def matching_learnings(category, dsl, limit=15):
    if not LEARNINGS.exists():
        return []
    rows = [json.loads(l) for l in LEARNINGS.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("category") == category or r.get("dsl") == dsl]
    return rows[-limit:]


def cmd_learn(a):
    run_dir, cfg = load_run(a.run)
    attempts = load_attempts(run_dir)
    append_jsonl(LEARNINGS, {
        "time": now(), "run": cfg["name"], "kernel": cfg["kernel"],
        "category": cfg["category"], "dsl": cfg["dsl"], "device": cfg.get("device", ""),
        "insight": a.insight, "evidence": a.evidence or "",
        "best_sol": max(best_so_far(attempts) or [0]),
    })
    print(f"recorded in {LEARNINGS.relative_to(REPO)}")


def collect_rows(pattern):
    rows = []
    for cfg_path in sorted(RUNS_DIR.glob(f"{pattern}/run.json")):
        cfg = json.loads(cfg_path.read_text())
        attempts = load_attempts(cfg_path.parent)
        if attempts:
            rows.append(summarize_run(cfg, attempts))
    return rows


def fmt_it(v):
    return "-" if v is None else str(v)


def cmd_report(a):
    rows = collect_rows(a.runs)
    if not rows:
        sys.exit(f"no runs under {RUNS_DIR.relative_to(REPO)}/{a.runs}")
    th_cols = [f"iters_to_{int(t * 100)}" for t in SOL_THRESHOLDS]
    print("| run | group | kernel | iters | baseline SOL | best SOL | " +
          " | ".join(f"to {int(t * 100)}%" for t in SOL_THRESHOLDS) + " |")
    print("|" + " --- |" * (6 + len(SOL_THRESHOLDS)))
    for r in rows:
        print(f"| {r['run']} | {group_key(r)} | {r['kernel']} | {r['iterations']} | "
              f"{r['baseline_sol']:.1%} | {r['best_sol']:.1%} | " +
              " | ".join(fmt_it(r[c]) for c in th_cols) + " |")
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="start a run on one kernel slot and measure its baseline")
    i.add_argument("--kernel", required=True, help="kernel name from `arena --list`")
    i.add_argument("--budget", type=int, default=20, help="iterations allowed after the baseline")
    i.add_argument("--target", type=float, default=0.9, help="SOL fraction to aim for")
    i.add_argument("--runs", type=int, default=30, help="timed runs per measurement")
    i.add_argument("--seeds", type=int, default=2, help="held-out seeds per check")
    i.add_argument("--train", action="append", help="timed size, e.g. n=67108864 (repeatable)")
    i.add_argument("--check", action="append", help="checked sizes, e.g. n=1,1000003")
    i.add_argument("--define", action="append", help="compile define KEY=n for the slot")
    i.add_argument("--start", help="source to start from instead of the slot's own "
                   "(same file type); becomes iteration 0")
    i.add_argument("--name", help="run directory name")
    i.add_argument("--model", help="model doing the optimizing, recorded for the report")
    i.add_argument("--docs", choices=["on", "off"], default="off",
                   help="whether a CUDA documentation tool (MCP) is available to the agent. "
                        "Recorded for the report.")
    i.add_argument("--mode", choices=["isolated", "cross-dsl"], default="isolated",
                   help="isolated: only this slot may be studied. cross-dsl: other kernels' "
                        "SASS and timings may be used as hints. Recorded for the report.")
    i.add_argument("--arena", help=f"command that runs arena (default: $ARENA_CMD or '{DEFAULT_ARENA_CMD}')")
    i.set_defaults(fn=cmd_init)

    e = sub.add_parser("eval", help="measure one candidate (counts as an iteration)")
    e.add_argument("run")
    e.add_argument("candidate")
    e.add_argument("--hypothesis", required=True, help="what this change should do and why")
    e.set_defaults(fn=cmd_eval)

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
    l.set_defaults(fn=cmd_learn)

    r = sub.add_parser("report", help="iterations-to-SOL across runs, per DSL")
    r.add_argument("--runs", default="*", help="glob over run names")
    r.add_argument("--csv", help="write best-so-far curves here")
    r.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

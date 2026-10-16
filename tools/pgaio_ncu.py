#!/usr/bin/env python3
"""Nsight Compute for PGAIO: the microscope next to arena's judge.

arena is the judge: only `pgaio.py eval` produces scores. This module never
produces or changes one. It looks inside kernels that eval already measured,
and times library references the same way eval times candidates.

    pgaio.py profile   <run> <iteration> [--shape K=..,M=..,N=..] [--clock-control base|none]
    pgaio.py profile   <run> --reference cublas [--shape ...]
    pgaio.py reference <run> --lib cublas

`profile` is free (no budget, no attempt record) and only takes iterations
that eval measured with verdict ok, so drafts are never profiled and "measure
through eval" keeps its meaning. It runs ncu on the arena binary itself, with
one warmup and one timed run, and keeps the timed run's launches. Durations
come from ncu, which by default locks clocks to base and replays the kernel
once per counter pass: every duration here is labelled "locked clock,
replayed" and is not comparable to eval's op_ms.

For memory-bound kernels the summary also has DRAM, L2 and L1/tex
throughput as a share of peak, achieved occupancy, waves and L2 slice
balance (busiest slice over the mean), all from `--set full`. Its time
breakdown puts the iteration's own eval timing at the shape (op_ms, the
kernels' gpu_ms and the launch and gap time between them, real clock)
next to ncu's view of the kernel body (duration, and the tail: average
over the longest SM's active cycles), labelled by where each number came
from.

arena's own CUPTI profiler cannot run while ncu holds the counters, so no
--profile is passed; arena warns and carries on.

Kernel replay saves and restores device memory between passes, so candidates
with self-resetting state (split-K semaphores, counters that the last block
clears) see the same memory on every pass and profile correctly.

`reference` times a library kernel (cuBLAS for matmul) through arena with
eval's timing settings and scores it per shape against the run's floors, so
a candidate can be read as "% of cuBLAS". It is not an iteration either.

Standalone use, with the same arguments:

    python3 tools/pgaio_ncu.py profile <run> 16 --shape K=2048,M=2048,N=2048

Only the standard library is used.
"""

import argparse
import csv
import io
import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pgaio  # noqa: E402

NCU = "/usr/local/cuda/bin/ncu"
# Where the repo and the arena binary live inside the compose service.
CONTAINER_REPO = "/workspace/gpgpu-arena"
CONTAINER_BUILD = CONTAINER_REPO + "/build"

# Library reference kernels arena can run, per category and library.
REFERENCES = {"matmul": {"cublas": "matmul_cublas"}}

TOP_INSTRUCTIONS = 15
HOT_COVER = 0.8


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tools/tests/test_pgaio_ncu.py)
# --------------------------------------------------------------------------

def number(text):
    """"1,024" -> 1024.0, "-" or "" -> None. ncu prints thousands separators."""
    t = (text or "").strip().replace(",", "")
    if t in ("", "-", "n/a", "N/A"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


# Unit -> (kind, factor to the base unit). Bytes go to bytes, times to ms,
# frequencies to MHz.
UNITS = {
    "byte": ("bytes", 1.0), "kbyte": ("bytes", 1e3), "mbyte": ("bytes", 1e6),
    "gbyte": ("bytes", 1e9), "tbyte": ("bytes", 1e12),
    "ns": ("ms", 1e-6), "us": ("ms", 1e-3), "usecond": ("ms", 1e-3), "ms": ("ms", 1.0),
    "msecond": ("ms", 1.0), "s": ("ms", 1e3), "second": ("ms", 1e3), "nsecond": ("ms", 1e-6),
    "hz": ("mhz", 1e-6), "khz": ("mhz", 1e-3), "mhz": ("mhz", 1.0), "ghz": ("mhz", 1e3),
    "cycle/second": ("mhz", 1e-6), "cycle/nsecond": ("mhz", 1e3), "cycle/usecond": ("mhz", 1.0),
    "byte/s": ("gbps", 1e-9), "kbyte/s": ("gbps", 1e-6), "mbyte/s": ("gbps", 1e-3),
    "gbyte/s": ("gbps", 1.0), "tbyte/s": ("gbps", 1e3), "byte/second": ("gbps", 1e-9),
    "gbyte/second": ("gbps", 1.0),
}


def to_base(value, unit, kind):
    """value in `unit` converted to the base unit of `kind` ("bytes", "ms",
    "mhz" or "gbps"). A missing unit is taken as already in the base unit."""
    if value is None:
        return None
    u = (unit or "").strip().lower()
    if not u:
        return value
    k, f = UNITS.get(u, (None, None))
    if k != kind:
        raise ValueError(f"unit {unit!r} is not a {kind} unit")
    return value * f


def parse_raw_csv(text):
    """`ncu --page raw --csv`: a header row, a units row, then one row per
    profiled kernel launch. Returns [{"kernel", "metrics": {name: (value,
    unit)}}], values as floats (None when ncu printed none)."""
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    # ncu may print notes ("==PROF== ...") before the table.
    start = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if start is None or start + 1 >= len(rows):
        return []
    header, units = rows[start], rows[start + 1]
    out = []
    for r in rows[start + 2:]:
        if len(r) < len(header) or not r[0].strip().isdigit():
            continue
        cell = dict(zip(header, r))
        metrics = {}
        for i, name in enumerate(header):
            if "__" not in name:
                continue
            metrics[name] = (number(r[i]), units[i] if i < len(units) else "")
        out.append({"kernel": cell.get("Kernel Name", ""), "metrics": metrics})
    return out


def metric(launch, name, kind=None):
    """One metric of one launch, in base units when kind is given."""
    v, unit = launch["metrics"].get(name, (None, ""))
    if v is None:
        return None
    return to_base(v, unit, kind) if kind else v


STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
PIPE_RE = re.compile(r"^sm__inst_executed_pipe_([a-z0-9_]+)\.avg\.pct_of_peak_sustained_active$")
# Sub-pipes and per-type splits double count their parent pipe.
PIPE_SKIP = ("_op_", "_type_", "subpipe", "pred_on", "_size_")


def stall_samples(launch):
    """{reason: warp samples} from the sampled warp states, issued samples
    ("selected") excluded: those are the warp issuing, not stalling."""
    out = {}
    for prefix in (STALL_PREFIX, "warpsampling:" + STALL_PREFIX):
        for name, (v, _) in launch["metrics"].items():
            if not name.startswith(prefix) or name.endswith("_not_issued") or v is None:
                continue
            reason = name[len(prefix):]
            if reason == "selected":
                continue
            out[reason] = v
        if out:
            return out
    return out


def summarize_metrics(launches):
    """Totals over the profiled launches of one operation. Counts add up;
    rates (issue active, pipe utilization, clock) are duration-weighted."""
    if not launches:
        return {}
    durs = [metric(l, "gpu__time_duration.sum", "ms") or 0.0 for l in launches]
    total = sum(durs)
    weights = [d / total if total > 0 else 1.0 / len(launches) for d in durs]

    def wavg(name, kind=None, scale=1.0):
        vals = [(metric(l, name, kind), w) for l, w in zip(launches, weights)]
        vals = [(v, w) for v, w in vals if v is not None]
        if not vals:
            return None
        wsum = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / wsum * scale if wsum > 0 else None

    def total_of(names, kind=None):
        seen, acc = False, 0.0
        for l in launches:
            for n in names:
                v = metric(l, n, kind)
                if v is not None:
                    acc, seen = acc + v, True
        return acc if seen else None

    clock = wavg("sm__cycles_elapsed.avg.per_second", "mhz")
    issue = wavg("smsp__issue_active.avg.pct_of_peak_sustained_active", scale=0.01)

    pipes = {}
    names = {m.group(1) for l in launches for n in l["metrics"] for m in [PIPE_RE.match(n)] if m}
    for p in sorted(names):
        if any(s in p for s in PIPE_SKIP):
            continue
        v = wavg(f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_active", scale=0.01)
        if v is not None and v >= 0.001:
            pipes[p] = round(v, 4)
    pipes = dict(sorted(pipes.items(), key=lambda kv: -kv[1]))

    samples = {}
    for l in launches:
        for k, v in stall_samples(l).items():
            samples[k] = samples.get(k, 0.0) + v
    stall_total = sum(samples.values())
    stalls = {k: round(v / stall_total, 4) for k, v in
              sorted(samples.items(), key=lambda kv: -kv[1]) if v > 0} if stall_total > 0 else {}

    dram = total_of(["dram__bytes.sum"], "bytes")
    if dram is None:
        dram = total_of(["dram__bytes_op_read.sum", "dram__bytes_op_write.sum"], "bytes")
    return {
        "duration_ms": total if total > 0 else None,
        "clock_mhz": round(clock, 1) if clock is not None else None,
        "issue_active": round(issue, 4) if issue is not None else None,
        "pipes": pipes,
        "stalls": stalls,
        "stall_samples": int(stall_total),
        "shared_bank_conflicts": total_of(["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"]),
        "shared_wavefronts": total_of(["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"]),
        "dram_bytes": dram,
        "l2_sectors": total_of(["lts__t_sectors.sum"]),
    }


def _ratio(a, b):
    return a / b if a is not None and b else None


def _round(v, nd=4):
    return round(v, nd) if v is not None else None


def first_metric(launch, names, kind=None):
    """The first of `names` this launch has a value for."""
    for n in names:
        v = metric(launch, n, kind)
        if v is not None:
            return v
    return None


# Throughputs as a fraction of peak; the first name ncu exported wins.
DRAM_PCT = ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed")
L2_PCT = ("lts__throughput.avg.pct_of_peak_sustained_elapsed",)
L1_PCT = ("l1tex__throughput.avg.pct_of_peak_sustained_active",
          "l1tex__throughput.avg.pct_of_peak_sustained_elapsed")
OCCUPANCY = ("sm__warps_active.avg.pct_of_peak_sustained_active",)
OCC_LIMITS = ("launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
              "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
              "launch__occupancy_limit_barriers")


def launch_detail(launch):
    """Shape, waves, SM balance and memory throughput of one profiled launch.

    waves = grid / (blocks per SM x SMs), blocks per SM being the tightest
    occupancy limit. sm_tail = sm__cycles_active avg / max: 1.0 when every
    SM was busy as long as the busiest one, lower when a few SMs still run
    after the others are done (a tail, or uneven work). sm_busy = average
    SM active cycles / elapsed cycles: the rest is ramp up, drain and idle
    SMs. l2_slice_max_over_avg = sectors in the busiest L2 slice / the mean
    slice: 1.0 is perfectly spread, a hot slice shows up above it."""
    def pct(names):
        v = first_metric(launch, names)
        return _round(v / 100.0) if v is not None else None

    grid = metric(launch, "launch__grid_size")
    sms = first_metric(launch, ["launch__sm_count", "device__attribute_multiprocessor_count"])
    limits = [v for v in (metric(launch, n) for n in OCC_LIMITS) if v]
    per_sm = min(limits) if limits else None
    waves = grid / (per_sm * sms) if grid and per_sm and sms else metric(
        launch, "launch__waves_per_multiprocessor")
    act_avg, act_max = metric(launch, "sm__cycles_active.avg"), metric(launch, "sm__cycles_active.max")
    elapsed = metric(launch, "sm__cycles_elapsed.avg")
    sec_avg, sec_max = metric(launch, "lts__t_sectors.avg"), metric(launch, "lts__t_sectors.max")
    sec_min = metric(launch, "lts__t_sectors.min")
    hot = metric(launch, "lts__t_sectors.max.pct_of_peak_sustained_elapsed")
    return {
        "kernel": launch["kernel"],
        "duration_ms": metric(launch, "gpu__time_duration.sum", "ms"),
        "grid": int(grid) if grid is not None else None,
        "block": int(b) if (b := metric(launch, "launch__block_size")) is not None else None,
        "sm_count": int(sms) if sms is not None else None,
        "blocks_per_sm": int(per_sm) if per_sm is not None else None,
        "waves": _round(waves, 3),
        "sm_active_avg_cycles": act_avg,
        "sm_active_max_cycles": act_max,
        "sm_elapsed_cycles": elapsed,
        "sm_tail": _round(_ratio(act_avg, act_max)),
        "sm_busy": _round(_ratio(act_avg, elapsed)),
        "dram_pct": pct(DRAM_PCT),
        "dram_gbps": _round(metric(launch, "dram__bytes.sum.per_second", "gbps"), 1),
        "l2_pct": pct(L2_PCT),
        "l1_pct": pct(L1_PCT),
        "occupancy": pct(OCCUPANCY),
        "l2_slice_sectors": {"avg": sec_avg, "max": sec_max, "min": sec_min}
        if sec_avg is not None else None,
        "l2_slice_max_over_avg": _round(_ratio(sec_max, sec_avg)),
        "l2_hot_slice_pct": _round(hot / 100.0) if hot is not None else None,
    }


def memory_summary(launches):
    """The memory-bound view of one operation, over its profiled launches:
    throughputs as a fraction of peak (duration-weighted), hit rates,
    achieved occupancy, waves and L2 slice balance (sectors in the busiest
    slice over the mean slice, summed over launches). None when ncu
    exported none of it."""
    if not launches:
        return None
    durs = [metric(l, "gpu__time_duration.sum", "ms") or 0.0 for l in launches]
    total = sum(durs)
    weights = [d / total if total > 0 else 1.0 / len(launches) for d in durs]

    def wavg(names, scale=1.0):
        vals = [(first_metric(l, names), w) for l, w in zip(launches, weights)]
        vals = [(v, w) for v, w in vals if v is not None]
        wsum = sum(w for _, w in vals)
        return _round(sum(v * w for v, w in vals) / wsum * scale) if vals and wsum > 0 else None

    def total_of(name):
        vals = [metric(l, name) for l in launches]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    details = [launch_detail(l) for l in launches]
    sec_max, sec_avg = total_of("lts__t_sectors.max"), total_of("lts__t_sectors.avg")
    sec_min = total_of("lts__t_sectors.min")
    cyc_max, cyc_avg = total_of("lts__cycles_active.max"), total_of("lts__cycles_active.avg")
    waves = [d["waves"] for d in details if d["waves"] is not None]
    out = {
        "dram_pct": wavg(DRAM_PCT, 0.01),
        # bytes over time: the duration-weighted mean of each launch's rate
        "dram_gbps": _round(sum(d["dram_gbps"] * w for d, w in zip(details, weights)), 1)
        if all(d["dram_gbps"] is not None for d in details) else None,
        "l2_pct": wavg(L2_PCT, 0.01),
        "l2_hit_rate": wavg(("lts__t_sector_hit_rate.pct",), 0.01),
        "l1_pct": wavg(L1_PCT, 0.01),
        "l1_hit_rate": wavg(("l1tex__t_sector_hit_rate.pct",), 0.01),
        "occupancy": wavg(OCCUPANCY, 0.01),
        "waves": _round(max(waves), 3) if waves else None,
        "l2_slices": first_metric(launches[0], ["device__attribute_l2s_count"]),
        "l2_slice_max_over_avg": _round(_ratio(sec_max, sec_avg)),
        "l2_slice_min_over_avg": _round(_ratio(sec_min, sec_avg)),
        "l2_hot_slice_pct": wavg(("lts__t_sectors.max.pct_of_peak_sustained_elapsed",), 0.01),
        "l2_active_max_over_avg": _round(_ratio(cyc_max, cyc_avg)),
    }
    if out["l2_slices"] is not None:
        out["l2_slices"] = int(out["l2_slices"])
    return out if any(v is not None for v in out.values()) else None


EVAL_NOTE = "eval: real clock, CUDA events (op_ms) and summed kernel durations (gpu_ms), cold L2"


def eval_timing(obs, params):
    """The iteration's own eval timing at `params`, from its obs/<NNN>.json:
    op_ms (the measured time by the run's score statistic, "op_stat":
    the interquartile mean under score rule 1, the mean of the fastest half
    under rule 2, without rule 2's parked-write charge, kept apart as
    parked_ms), the median op time, gpu_ms (kernel durations of one
    profiled run), arena's overhead_ms (median op time - gpu_ms, at least
    0), launch_count and the mean SM clock. None when eval did not time the
    shape."""
    size = next((s for s in (obs or {}).get("sizes") or [] if s.get("params") == params), None)
    if not size or not size.get("op_ms"):
        return None
    t = next((r.get("timing") or {} for r in ((obs.get("raw") or {}).get("timing_results") or [])
              if (r.get("input") or {}).get("params") == params), {})
    gpu = size.get("gpu_ms") if size.get("gpu_ms") is not None else t.get("gpu_ms")
    rule = obs.get("score_rule", 1)
    out = {"op_ms": size.get("op_ms_raw") or size["op_ms"],
           "op_stat": "mean of the fastest half" if rule >= 2 else "interquartile mean",
           "op_ms_median": t.get("op_ms"), "gpu_ms": gpu, "overhead_ms": t.get("overhead_ms"),
           "launch_count": t.get("launch_count"),
           "sm_clock_mhz": (size.get("clock") or {}).get("sm_clock_mean_mhz")}
    if size.get("parked_ms"):
        out["parked_ms"] = size["parked_ms"]
    return out


def _time(ms):
    if ms is None:
        return "-"
    return f"{ms * 1e3:.1f} us" if abs(ms) < 1.0 else f"{ms:.3f} ms"


def time_breakdown(evalt, details, duration_ms=None, clock_mhz=None, note="locked clock, replayed"):
    """Where an operation's time goes. From eval (real clock): op_ms, the
    kernels' share (gpu_ms) and launch and gaps (op_ms - gpu_ms). From ncu
    (its clock, replayed): the duration and, per launch, the SM tail
    (average over the longest SM's active cycles) and waves."""
    out = {}
    if evalt:
        # gpu_ms is a median-like single profiled run, so launch and gaps
        # come from the median op time (arena's overhead_ms when it has it),
        # never the score statistic, which can sit below gpu_ms
        gpu = evalt.get("gpu_ms")
        op = evalt.get("op_ms_median") or evalt.get("op_ms")
        gap = evalt.get("overhead_ms")
        if gap is None and op is not None and gpu is not None:
            gap = max(0.0, op - gpu)
        out["eval"] = dict(evalt, launch_gap_ms=gap, kernel_share=_round(_ratio(gpu, op)),
                           note=EVAL_NOTE)
    acts = [(d["sm_active_avg_cycles"], d["sm_active_max_cycles"], d["sm_elapsed_cycles"])
            for d in details or []]
    acts = [a for a in acts if None not in a]
    tail = _round(_ratio(sum(a[0] for a in acts), sum(a[1] for a in acts))) if acts else None
    busy = _round(_ratio(sum(a[0] for a in acts), sum(a[2] for a in acts))) if acts else None
    peak = _round(_ratio(sum(a[1] for a in acts), sum(a[2] for a in acts))) if acts else None
    waves = [d["waves"] for d in details or [] if d.get("waves") is not None]
    if duration_ms is not None or acts:
        out["ncu"] = {"duration_ms": duration_ms, "clock_mhz": clock_mhz, "sm_tail": tail,
                      "sm_busy": busy, "busiest_sm_busy": peak,
                      "waves": _round(max(waves), 3) if waves else None, "note": note}
    if not out:
        return None
    parts = []
    e = out.get("eval")
    if e:
        clk = f", {e['sm_clock_mhz']:.0f} MHz" if e.get("sm_clock_mhz") else ""
        n = e.get("launch_count")
        stat = f"{e['op_stat']}, " if e.get("op_stat") else ""
        med = f", median {_time(e['op_ms_median'])}" if e.get("op_ms_median") else ""
        parts.append(f"op {_time(e['op_ms'])} (eval, {stat}real clock{clk}{med}): "
                     f"kernels {_time(e.get('gpu_ms'))}, "
                     f"launch and gaps {_time(e.get('launch_gap_ms'))}"
                     + (f" ({n} launch{'es' if n != 1 else ''})" if n is not None else "")
                     + (f"; the score adds {_time(e['parked_ms'])} for writes parked in L2"
                        if e.get("parked_ms") else ""))
    n = out.get("ncu")
    if n:
        clk = f" at {n['clock_mhz']:.0f} MHz" if n.get("clock_mhz") else ""
        s = f"in the kernel (ncu{clk}, {note}: {_time(n['duration_ms'])})"
        if tail is not None:
            s += (f", SMs busy on average {tail:.0%} of the longest SM's time (tail); "
                  f"the longest SM active {peak:.0%} of elapsed cycles")
        if n.get("waves") is not None:
            s += f"; {n['waves']:.2f} waves"
        parts.append(s)
    out["text"] = "; ".join(parts)
    return out


SOURCE_STALL = re.compile(r"^stall_([a-z_]+)$")
OPCODE = re.compile(r"^\s*(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")


def parse_source_csv(text):
    """`ncu --page source --print-source sass --csv`: per kernel, a
    "Kernel Name" line, a header, then one row per SASS instruction.
    Returns [{"kernel", "address", "offset", "opcode", "source", "samples",
    "stalls": {reason: samples}}]. offset is from the kernel's first
    instruction, as in arena's SASS listings (/*0c40*/)."""
    out = []
    kernel, header, base = "", None, None
    for r in csv.reader(io.StringIO(text)):
        if not r or not any(c.strip() for c in r):
            continue
        if r[0] == "Kernel Name":
            kernel, header, base = (r[1] if len(r) > 1 else ""), None, None
            continue
        if r[0] == "Address":
            header = r
            continue
        if header is None:
            continue
        cell = dict(zip(header, r))
        addr = cell.get("Address", "")
        try:
            a = int(addr, 16)
        except ValueError:
            continue
        base = a if base is None else base
        src = cell.get("Source", "").strip()
        m = OPCODE.match(src)
        stalls = {}
        for col, v in cell.items():
            sm = SOURCE_STALL.match(col)
            n = number(v)
            if sm and n:
                stalls[sm.group(1)] = n
        out.append({
            "kernel": kernel,
            "address": addr,
            "offset": f"{a - base:04x}",
            "opcode": m.group(1) if m else src.split(" ")[0],
            "source": src,
            "samples": number(cell.get("Warp Stall Sampling (All Samples)")) or 0.0,
            "stalls": stalls,
        })
    return out


def main_stall(stalls):
    """The reason holding most of an instruction's samples, issuing
    ("selected") only when nothing else was sampled."""
    real = {k: v for k, v in stalls.items() if k != "selected" and v > 0}
    if real:
        return max(real, key=real.get)
    return "selected" if stalls.get("selected") else ""


def top_instructions(instrs, limit=TOP_INSTRUCTIONS):
    total = sum(i["samples"] for i in instrs)
    ranked = sorted(instrs, key=lambda i: -i["samples"])[:limit]
    return [{"address": i["address"], "offset": i["offset"], "kernel": i["kernel"],
             "opcode": i["opcode"], "stall": main_stall(i["stalls"]),
             "samples": int(i["samples"]),
             "share": round(i["samples"] / total, 4) if total > 0 else 0.0}
            for i in ranked if i["samples"] > 0]


def hot_loop_opcodes(instrs, cover=HOT_COVER):
    """Opcode mix of the fewest instructions that hold `cover` of all
    samples: in a matmul, the main loop's FFMA, LDS and barrier balance."""
    total = sum(i["samples"] for i in instrs)
    if total <= 0:
        return {}
    acc, out = 0.0, {}
    for i in sorted(instrs, key=lambda i: -i["samples"]):
        if acc >= cover * total or i["samples"] <= 0:
            break
        acc += i["samples"]
        out[i["opcode"]] = out.get(i["opcode"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def clock_note(clock_control):
    return ("locked clock, replayed" if clock_control == "base"
            else "unlocked clock, replayed")


def summarize(raw_text, source_text, iteration=None, reference=None, params=None,
              clock_control="base", evalt=None):
    """summary.json from the raw and source exports. evalt is eval's own
    timing at this shape (eval_timing), for the time breakdown."""
    launches = parse_raw_csv(raw_text)
    instrs = parse_source_csv(source_text or "")
    kernels = []
    for l in launches:
        if l["kernel"] not in kernels:
            kernels.append(l["kernel"])
    out = {"iteration": iteration, "reference": reference, "params": params or {},
           "kernels": kernels, "launches": len(launches)}
    m = summarize_metrics(launches)
    out.update({
        "duration_ms": m.get("duration_ms"),
        "clock_mhz": m.get("clock_mhz"),
        "note": clock_note(clock_control),
        "issue_active": m.get("issue_active"),
        "pipes": m.get("pipes", {}),
        "stalls": m.get("stalls", {}),
        "stall_samples": m.get("stall_samples", 0),
        "shared_bank_conflicts": m.get("shared_bank_conflicts"),
        "shared_wavefronts": m.get("shared_wavefronts"),
        "dram_bytes": m.get("dram_bytes"),
        "l2_sectors": m.get("l2_sectors"),
        "top_instructions": top_instructions(instrs),
        "hot_loop_opcodes": hot_loop_opcodes(instrs),
    })
    # Added later: older summary.json files lack these keys.
    details = [launch_detail(l) for l in launches]
    out["memory"] = memory_summary(launches)
    out["launch_details"] = details
    out["time_breakdown"] = time_breakdown(evalt, details, m.get("duration_ms"), m.get("clock_mhz"),
                                           clock_note(clock_control))
    return out


def _pct(v):
    return "-" if v is None else f"{v:.0%}" if v >= 0.1 or v == 0 else f"{v:.1%}"


def _big(v, unit=""):
    if v is None:
        return "-"
    for f, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= f:
            return f"{v / f:.1f}{s}{unit}"
    return f"{v:.0f}{unit}"


def subject(s):
    what = f"reference {s['reference']}" if s.get("reference") else f"iteration {s.get('iteration')}"
    return f"{what} at {pgaio.shape_label(s.get('params') or {})}"


def render_summary(s):
    """Human-readable summary.txt."""
    lines = [f"{subject(s)}: ncu, {s['note']} (not a score; eval's op_ms is the measurement)"]
    lines.append(f"kernels: {', '.join(s.get('kernels') or ['-'])} ({s.get('launches', 0)} launch(es))")
    d = s.get("duration_ms")
    lines.append(f"duration {d:.4f} ms ({s['note']}) at {s.get('clock_mhz') or 0:.0f} MHz"
                 if d is not None else "duration -")
    lines.append(f"issue active {_pct(s.get('issue_active'))} of cycles")
    if s.get("pipes"):
        lines.append("pipes (share of peak, active cycles): " +
                     ", ".join(f"{k} {_pct(v)}" for k, v in list(s["pipes"].items())[:8]))
    if s.get("stalls"):
        lines.append(f"stalls (share of {s.get('stall_samples', 0)} stall samples): " +
                     ", ".join(f"{k} {_pct(v)}" for k, v in list(s["stalls"].items())[:10]))
    wf, bc = s.get("shared_wavefronts"), s.get("shared_bank_conflicts")
    if wf:
        lines.append(f"shared memory: {_big(wf)} wavefronts, {_big(bc)} bank conflicts"
                     + (f" ({bc / wf:.1%} of wavefronts)" if bc is not None else ""))
    lines.append(f"DRAM {_big(s.get('dram_bytes'), 'B')}, L2 {_big(s.get('l2_sectors'))} sectors")
    lines += render_memory(s)
    if s.get("top_instructions"):
        lines.append("top instructions by stall samples (offset, opcode, share, main stall):")
        multi = len(s.get("kernels") or []) > 1
        for i in s["top_instructions"]:
            lines.append(f"  {i['offset']}  {i['opcode']:<16} {i['share']:6.1%}  {i['stall']}"
                         + (f"  [{i['kernel']}]" if multi else ""))
    if s.get("hot_loop_opcodes"):
        lines.append(f"hot loop opcode mix (instructions holding {HOT_COVER:.0%} of samples): " +
                     ", ".join(f"{k} x{v}" for k, v in list(s["hot_loop_opcodes"].items())[:12]))
    return "\n".join(lines)


def render_memory(s):
    """summary.txt lines for the memory and time sections (empty for
    summaries written before they existed)."""
    lines = []
    m = s.get("memory") or {}
    if m:
        parts = []
        if m.get("dram_pct") is not None:
            parts.append(f"DRAM {_pct(m['dram_pct'])} of peak"
                         + (f" ({m['dram_gbps']:.0f} GB/s)" if m.get("dram_gbps") is not None else ""))
        if m.get("l2_pct") is not None:
            parts.append(f"L2 {_pct(m['l2_pct'])} of peak"
                         + (f" (hit rate {_pct(m['l2_hit_rate'])})" if m.get("l2_hit_rate") is not None else ""))
        if m.get("l1_pct") is not None:
            parts.append(f"L1/tex {_pct(m['l1_pct'])} of peak"
                         + (f" (hit rate {_pct(m['l1_hit_rate'])})" if m.get("l1_hit_rate") is not None else ""))
        if m.get("occupancy") is not None:
            parts.append(f"achieved occupancy {_pct(m['occupancy'])}")
        if m.get("waves") is not None:
            d = (s.get("launch_details") or [{}])[0]
            parts.append(f"{m['waves']:.2f} waves" + (
                f" ({d['grid']} blocks / ({d['blocks_per_sm']} per SM x {d['sm_count']} SMs))"
                if len(s.get("launch_details") or []) == 1 and d.get("blocks_per_sm") else ""))
        if parts:
            lines.append(f"memory ({s['note']}): " + ", ".join(parts))
        r = m.get("l2_slice_max_over_avg")
        if r is not None:
            n = f" ({m['l2_slices']})" if m.get("l2_slices") else ""
            line = f"L2 slices{n}: busiest {r:.2f}x the mean slice's sectors"
            if m.get("l2_slice_min_over_avg") is not None:
                line += f", quietest {m['l2_slice_min_over_avg']:.2f}x"
            if m.get("l2_hot_slice_pct") is not None:
                line += f"; busiest slice at {_pct(m['l2_hot_slice_pct'])} of its peak"
            if m.get("l2_active_max_over_avg") is not None:
                line += f"; active cycles busiest {m['l2_active_max_over_avg']:.2f}x the mean"
            lines.append(line)
    t = s.get("time_breakdown") or {}
    if t.get("text"):
        lines.append("time: " + t["text"])
    details = s.get("launch_details") or []
    if len(details) > 1:
        lines.append("launches (ncu): kernel, duration, waves, SM tail (avg/max active), DRAM, L2:")
        for d in details:
            lines.append(f"  {d['kernel']}  {_time(d.get('duration_ms'))}  "
                         f"{'-' if d.get('waves') is None else format(d['waves'], '.2f')} waves  "
                         f"tail {_pct(d.get('sm_tail'))}  DRAM {_pct(d.get('dram_pct'))}  "
                         f"L2 {_pct(d.get('l2_pct'))}")
    return lines


def diff_summaries(cur, ref, min_points=0.02):
    """Lines comparing two summaries, current first: "stall X: 23% vs 11%".
    Stall and pipe shares are listed when they differ by min_points."""
    label = subject(ref)
    out = [f"vs {label}" + ("" if cur.get("params") == ref.get("params")
                            else " (a different shape)") + ":"]
    a, b = cur.get("duration_ms"), ref.get("duration_ms")
    if a and b:
        out.append(f"  duration: {a:.4f} vs {b:.4f} ms ({a / b:.2f}x the time; both {cur['note']})"
                   if cur.get("note") == ref.get("note") else
                   f"  duration: {a:.4f} ms ({cur['note']}) vs {b:.4f} ms ({ref['note']})")
    a, b = cur.get("issue_active"), ref.get("issue_active")
    if a is not None and b is not None:
        out.append(f"  issue active: {_pct(a)} vs {_pct(b)}")
    for key, word in (("stalls", "stall"), ("pipes", "pipe")):
        x, y = cur.get(key) or {}, ref.get(key) or {}
        rows = [(k, x.get(k, 0.0), y.get(k, 0.0)) for k in set(x) | set(y)]
        rows = [r for r in rows if abs(r[1] - r[2]) >= min_points]
        for k, va, vb in sorted(rows, key=lambda r: -abs(r[1] - r[2])):
            out.append(f"  {word} {k}: {_pct(va)} vs {_pct(vb)}")
    for key, word in (("shared_bank_conflicts", "shared bank conflicts"),
                      ("shared_wavefronts", "shared wavefronts"),
                      ("dram_bytes", "DRAM bytes"), ("l2_sectors", "L2 sectors")):
        a, b = cur.get(key), ref.get(key)
        if a is not None and b is not None and (a or b):
            out.append(f"  {word}: {_big(a)} vs {_big(b)}")
    ma, mb = cur.get("memory") or {}, ref.get("memory") or {}
    for key, word in (("dram_pct", "DRAM throughput (of peak)"), ("l2_pct", "L2 throughput (of peak)"),
                      ("l2_hit_rate", "L2 hit rate"), ("l1_pct", "L1/tex throughput (of peak)"),
                      ("occupancy", "achieved occupancy"), ("l2_hot_slice_pct", "busiest L2 slice (of peak)")):
        a, b = ma.get(key), mb.get(key)
        if a is not None and b is not None:
            out.append(f"  {word}: {_pct(a)} vs {_pct(b)}")
    for key, word in (("l2_slice_max_over_avg", "L2 busiest slice / mean"), ("waves", "waves")):
        a, b = ma.get(key), mb.get(key)
        if a is not None and b is not None:
            out.append(f"  {word}: {a:.2f} vs {b:.2f}")
    na = (cur.get("time_breakdown") or {}).get("ncu") or {}
    nb = (ref.get("time_breakdown") or {}).get("ncu") or {}
    if na.get("sm_tail") is not None and nb.get("sm_tail") is not None:
        out.append(f"  SM tail (avg/max active cycles): {_pct(na['sm_tail'])} vs {_pct(nb['sm_tail'])}")
    ea = (cur.get("time_breakdown") or {}).get("eval") or {}
    eb = (ref.get("time_breakdown") or {}).get("eval") or {}
    for key, word in (("op_ms", "op (eval)"), ("gpu_ms", "kernels (eval)"),
                      ("launch_gap_ms", "launch and gaps (eval)")):
        a, b = ea.get(key), eb.get(key)
        if a is not None and b is not None:
            out.append(f"  {word}: {_time(a)} vs {_time(b)}")
    x, y = cur.get("hot_loop_opcodes") or {}, ref.get("hot_loop_opcodes") or {}
    changed = [(k, x.get(k, 0), y.get(k, 0)) for k in sorted(set(x) | set(y)) if x.get(k, 0) != y.get(k, 0)]
    if changed:
        out.append("  hot loop opcodes: " + ", ".join(f"{k} {va} vs {vb}" for k, va, vb in changed[:12]))
    return out


def parse_shape(spec):
    """"K=2048,M=2048,N=2048" -> {"K": 2048, "M": 2048, "N": 2048}"""
    out = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        if not v:
            raise ValueError(f"bad shape item {item!r}; expected k=v")
        out[k.strip()] = int(v)
    return out


def largest_shape(shapes):
    """The timed shape with the most work: the product of its sizes."""
    def size(sh):
        p = 1
        for v in sh.values():
            p *= v
        return p
    return max(shapes, key=size) if shapes else {}


def shape_slug(params):
    return "-".join(f"{k}{v}" for k, v in sorted(params.items())) or "default"


def profile_dir_name(iteration=None, reference=None, params=None, default=True):
    """profiles/016 for an iteration at the run's largest timed shape,
    profiles/016-K2048-M2048-N2048 at another shape, profiles/ref-cublas-<shape>."""
    if reference:
        return f"ref-{reference}-{shape_slug(params or {})}"
    return f"{iteration:03d}" if default else f"{iteration:03d}-{shape_slug(params or {})}"


IDENT = re.compile(r"^[A-Za-z_$][\w$.]*$")


def function_name(name):
    """The bare function name of a demangled kernel name, as ncu's
    --kernel-name-base function shows it: no return type, namespace,
    class, template arguments or parameters.
    "void ns::Outer<int>::gemm<float>(float*, int)" -> "gemm",
    "(anonymous namespace)::reduce<256>" -> "reduce"."""
    s, depth = [], 0
    for c in name:                      # template arguments, at any depth
        if c == "<":
            depth += 1
        elif c == ">" and depth:
            depth -= 1
        elif not depth:
            s.append(c)
    s = "".join(s).strip()
    if s.endswith(")"):                 # the parameter list
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(s[i], 0)
            if depth == 0:
                s = s[:i] if i > 0 else s
                break
    s = s.split("::")[-1].strip()
    return s.split()[-1] if s.split() else s


def kernel_filter(names):
    """ncu kernel filter for the operation's kernels: an anchored regex on
    the mangled name, which for extern "C" kernels is the plain name. Names
    CUPTI reports demangled fall back to their function name."""
    uniq, base = [], "mangled"
    for n in names:
        if not IDENT.match(n):
            base = "function"
        if n not in uniq:
            uniq.append(n)
    if base == "function":
        uniq = list(dict.fromkeys(function_name(n) for n in uniq))
    return base, "regex:^(" + "|".join(re.escape(n) for n in uniq) + ")$"


def ncu_args(names, launches_per_op, out_rel, warmup=1, clock_control="base", ncu_log=None):
    """ncu options profiling the timed run's launches. arena runs `warmup`
    warmup operations first, so that many operations' launches are skipped."""
    base, flt = kernel_filter(names)
    args = ["--target-processes", "all", "--set", "full", "--import-source", "yes",
            "--section", "SourceCounters", "--kernel-name-base", base, "-k", flt,
            "--launch-skip", str(warmup * launches_per_op), "--launch-count", str(launches_per_op),
            "--clock-control", clock_control, "-f", "-o", f"{out_rel}/report"]
    if ncu_log:
        args += ["--log-file", ncu_log]
    return args


def is_docker(arena_cmd):
    toks = shlex.split(arena_cmd)
    return bool(toks) and Path(toks[0]).name == "docker" and "run" in toks


def build_script(arena_cmd, profile_args, arena_args, out_dir, repo=None, uid=None, gid=None,
                 host_ncu=NCU):
    """(argv, cwd) that profiles with ncu and exports the report as csv.

    Docker: the run's arena command with sh as the entrypoint (in place of
    the build-and-run script), the working directory set to arena's build
    dir, and a chown at the end, since the container runs as root. Native:
    the same script through the host's sh, from the binary's directory, as
    pgaio.arena runs it."""
    repo = Path(repo or pgaio.REPO)
    out_dir = Path(out_dir)
    toks = shlex.split(arena_cmd)
    if is_docker(arena_cmd):
        ncu, arena_bin = NCU, CONTAINER_BUILD + "/arena"
        out = CONTAINER_REPO + "/" + str(out_dir.relative_to(repo))
    else:
        exe = Path(toks[0]) if Path(toks[0]).is_absolute() else repo / toks[0]
        ncu, arena_bin, out = host_ncu, str(exe), str(out_dir)
        arena_args = toks[1:] + list(arena_args)
    q = shlex.quote
    rep = f"{out}/report.ncu-rep"
    prof = [ncu] + [a.replace("{out}", out) for a in profile_args] + [arena_bin] + list(arena_args)
    lines = [
        shlex.join(prof) + f" > {q(out + '/arena_stdout.json')} 2> {q(out + '/arena_stderr.txt')}",
        "rc=$?",
        f"if [ -f {q(rep)} ]; then",
        f"  {q(ncu)} --import {q(rep)} --page details --csv > {q(out + '/details.csv')}",
        f"  {q(ncu)} --import {q(rep)} --page raw --csv > {q(out + '/raw.csv')}",
        f"  {q(ncu)} --import {q(rep)} --page source --print-source sass --csv > {q(out + '/source.csv')}",
        "fi",
    ]
    if is_docker(arena_cmd) and uid is not None:
        lines.append(f"chown -R {uid}:{gid} {q(out)}")
    lines.append("exit $rc")
    script = "\n".join(lines)
    if is_docker(arena_cmd):
        service = toks[-1]
        argv = toks[:-1] + ["-w", CONTAINER_BUILD, "--entrypoint", "/bin/sh", service, "-c", script]
        return argv, repo
    return ["sh", "-c", script], exe.parent


def reference_kernel(category, lib):
    """The arena kernel for a library reference, or an error message."""
    libs = REFERENCES.get(category)
    if not libs:
        return None, f"no library reference is known for category {category}"
    if lib not in libs:
        return None, f"no {lib} reference for {category} (known: {', '.join(sorted(libs))})"
    return libs[lib], None


def timing_args(cfg, run_args, shapes):
    """eval's timing step (pgaio.measure, step 2), without the correctness
    gate: cold L2, the run's number of runs, counters when the run has them."""
    return pgaio.timing_args(cfg, run_args, shapes)


def run_env(cfg, environment):
    """The environment with the run's own floors, as observe() scores."""
    env = dict(environment or {})
    for key in ("launch_floor_ms", "roundtrip_floor_ms"):
        if cfg.get(key):
            env[key] = cfg[key]
    return env


def op_launches(result):
    """One kernel name per launch of one operation, from an arena result:
    "launched_kernels" repeated by count_per_op, or the activity run's
    "sub_kernels". Its length is the number of launches ncu must skip per
    warmup operation and then profile."""
    out = []
    for k in result.get("launched_kernels") or []:
        if isinstance(k, dict):
            if k.get("name"):
                out += [k["name"]] * max(1, int(k.get("count_per_op") or 1))
        elif k:
            out.append(k)
    if not out:
        out = [k.get("name") for k in result.get("sub_kernels") or [] if k.get("name")]
    return out


def kernel_names(launched):
    """Kernel names from arena's "launched_kernels" ([{"name", ...}]) or a
    plain list of names, in order, without repeats."""
    names = [k.get("name") if isinstance(k, dict) else k for k in launched or []]
    return list(dict.fromkeys(n for n in names if n))


def reference_sizes(results, env, rule=1):
    """Per-shape SOL of a reference's timing results, scored like observe()
    under the run's score rule. op_ms is the measured time by that rule's
    statistic, without rule 2's parked-write charge (which stays in the
    fraction), so it compares with a candidate's measured time."""
    sizes = []
    for r in results:
        s = pgaio.size_sol(r, env, rule)
        names = kernel_names(r.get("launched_kernels")) or None
        if names is None and r.get("sub_kernels"):
            names = [k.get("name") for k in r["sub_kernels"]]
        sizes.append({"params": s["params"], "op_ms": s.get("op_ms_raw") or s["op_ms"], "fraction": s["fraction"],
                      "bound": s["bound"], "roof_ms": s["roof_ms"],
                      "verified": r.get("verified"), "launched_kernels": names})
    return sizes


def percent_of_reference(ref_sizes, cand_sizes):
    """[(params, ref op_ms, candidate op_ms, ref/candidate)] per shared shape.
    Above 1.0 the candidate is faster than the library."""
    by = {json.dumps(s["params"], sort_keys=True): s for s in cand_sizes or []}
    out = []
    for r in ref_sizes:
        c = by.get(json.dumps(r["params"], sort_keys=True))
        cm = (c or {}).get("op_ms_raw") or (c or {}).get("op_ms")    # measured, as for the reference
        if cm and r.get("op_ms"):
            out.append((r["params"], r["op_ms"], cm, r["op_ms"] / cm))
    return out


# --------------------------------------------------------------------------
# Side effects
# --------------------------------------------------------------------------

def iteration_target(cfg, attempt):
    """arena arguments that run an evaluated iteration, as eval ran it."""
    cand = attempt.get("candidate", "")
    if cfg.get("kind") == "problem":
        if cand.endswith(".json"):
            return pgaio.target_args(cfg, manifest=cand)[0]
        return pgaio.target_args(cfg)[0]         # the baseline kernel, as init measured it
    if attempt["iteration"] == 0 and cfg.get("start_source") == cfg.get("slot_source"):
        return pgaio.target_args(cfg)[0]
    return pgaio.target_args(cfg, rel_source=cand)[0]


def stored_kernels(run_dir, iteration, params):
    """(kernel names, launches per operation) from eval's own timing of this
    shape, when the shape was timed."""
    p = run_dir / "obs" / f"{iteration:03d}.json"
    if not p.exists():
        return None
    obs = json.loads(p.read_text())
    for r in obs.get("raw", {}).get("timing_results", []):
        if r.get("input", {}).get("params") == params and r.get("sub_kernels"):
            return [k["name"] for k in r["sub_kernels"]]
    return None


def preflight_kernels(cfg, run_args, params):
    """One untimed arena run at the shape to learn which kernels the
    operation launches. Nothing from it is kept."""
    code, data, _, err = pgaio.arena(cfg["arena_cmd"], run_args + pgaio.shape_args([params]) +
                                     ["--runs", "1", "--warmup", "1"])
    rs = (data or {}).get("results", [])
    if code != 0 or not rs or not rs[0].get("success"):
        msg = (rs[0].get("error") if rs else "") or err[-2000:]
        sys.exit(f"arena could not run the kernel at {pgaio.shape_label(params)}:\n{msg}")
    names = op_launches(rs[0])
    if not names:
        sys.exit("arena reported no kernel launches for this operation (CUPTI unavailable?)")
    return names


def find_profile(run_dir, params, iteration=None, reference=None):
    """A stored summary.json for this iteration or reference at these params."""
    root = run_dir / "profiles"
    if not root.is_dir():
        return None
    for p in sorted(root.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if s.get("params") != params:
            continue
        if reference and s.get("reference") == reference:
            return s
        if iteration is not None and not reference and s.get("iteration") == iteration \
                and not s.get("reference"):
            return s
    return None


def cmd_profile(a):
    run_dir, cfg = pgaio.load_run(a.run)
    attempts = pgaio.load_attempts(run_dir)
    train, _ = pgaio.run_shapes(cfg)
    default = largest_shape(train)
    try:
        params = parse_shape(a.shape) if a.shape else default
    except ValueError as e:
        sys.exit(str(e))
    if not params:
        sys.exit("the run has no timed shapes; pass --shape")

    if a.reference:
        if a.iteration is not None:
            sys.exit("pass an iteration or --reference, not both")
        kernel, err = reference_kernel(cfg["category"], a.reference)
        if err:
            sys.exit(err)
        run_args = ["--run", kernel]
        iteration, names = None, None
        what = f"reference {a.reference} ({kernel})"
    else:
        if a.iteration is None:
            sys.exit("pass an iteration number, or --reference <lib>")
        att = next((x for x in attempts if x.get("iteration") == a.iteration), None)
        if att is None:
            sys.exit(f"iteration {a.iteration} was never evaluated; profile only measured "
                     f"iterations (drafts are inspected with `pgaio.py sass`)")
        if att.get("verdict") != "ok":
            sys.exit(f"iteration {a.iteration} has verdict {att.get('verdict')}; only iterations "
                     "that passed eval are profiled")
        run_args = iteration_target(cfg, att)
        iteration = a.iteration
        names = stored_kernels(run_dir, iteration, params)
        what = f"iteration {iteration} ({att.get('dsl')})"

    out_dir = run_dir / "profiles" / profile_dir_name(iteration, a.reference, params,
                                                      default=params == default)
    # The run's own arena settings (the scratch policy), as eval passes them.
    run_args = run_args + pgaio.run_flags(cfg, timing=False)
    arena_args = run_args + pgaio.shape_args([params]) + ["--runs", "1", "--warmup", "1"]

    if a.dry_run:
        shown = names or ["<kernels from a preflight arena run>"]
        prof = ncu_args(shown, len(shown) if names else 1, "{out}", clock_control=a.clock_control,
                        ncu_log="{out}/ncu.log")
        argv, cwd = build_script(cfg["arena_cmd"], prof, arena_args, out_dir,
                                 uid=os.getuid(), gid=os.getgid())
        print(f"would profile {what} at {pgaio.shape_label(params)} into "
              f"{out_dir.relative_to(pgaio.REPO)}/" + ("" if names else
              "\n(kernel names and launch count come from one untimed arena run first)"))
        print(f"cwd: {cwd}")
        print(shlex.join(argv))
        return

    if names is None:
        print(f"[pgaio] finding the kernels {what} launches at {pgaio.shape_label(params)} ...",
              file=sys.stderr)
        names = preflight_kernels(cfg, run_args, params)

    # Made here, not by ncu: in Docker it runs as root.
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in ("report.ncu-rep", "raw.csv", "source.csv", "details.csv", "summary.json", "summary.txt",
                "ncu.log", "arena_stdout.json", "arena_stderr.txt"):
        try:
            (out_dir / old).unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            sys.exit(f"{out_dir / old} is not writable (left by a root container?); remove it first")
    prof = ncu_args(names, len(names), "{out}", clock_control=a.clock_control, ncu_log="{out}/ncu.log")
    argv, cwd = build_script(cfg["arena_cmd"], prof, arena_args, out_dir,
                             uid=os.getuid(), gid=os.getgid())
    print(f"[pgaio] ncu on {what} at {pgaio.shape_label(params)}: {', '.join(dict.fromkeys(names))}, "
          f"{len(names)} launch(es) per op, clock control {a.clock_control} ...", file=sys.stderr)
    # SIGTERM before SIGKILL on timeout, so arena can release a clock lock
    # and the container is stopped.
    code, _, stderr = pgaio.run_with_timeout(argv, cwd, 3600)
    raw = out_dir / "raw.csv"
    if not raw.exists() or not parse_raw_csv(raw.read_text()):
        log = out_dir / "ncu.log"
        tail = (log.read_text(errors="replace")[-2000:] if log.exists() else "") or stderr[-2000:]
        sys.exit(f"ncu produced no profile (exit {code}):\n{tail}")

    got = len(parse_raw_csv(raw.read_text()))
    if code != 0 or got != len(names):
        print(f"[pgaio] warning: ncu exited {code} and profiled {got} launch(es) where the "
              f"operation makes {len(names)}; the profile may be partial (see "
              f"{(out_dir / 'ncu.log').relative_to(pgaio.REPO)})", file=sys.stderr)

    src = out_dir / "source.csv"
    evalt = None
    if iteration is not None:
        obs_path = run_dir / "obs" / f"{iteration:03d}.json"
        try:
            evalt = eval_timing(json.loads(obs_path.read_text()), params)
        except (OSError, json.JSONDecodeError):
            evalt = None
        if evalt:
            evalt["source"] = str(obs_path.relative_to(run_dir))
    s = summarize(raw.read_text(), src.read_text() if src.exists() else "",
                  iteration=iteration, reference=a.reference, params=params,
                  clock_control=a.clock_control, evalt=evalt)
    s.update({"time": pgaio.now(), "ncu_filter": kernel_filter(names)[1],
              "launches_per_op": len(names)})
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1))
    text = render_summary(s)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)

    # Compare with the best iteration, or the reference when this is the best.
    best_att, _ = pgaio.best_of(run_dir, attempts)
    other = None
    if a.reference:
        if best_att:
            other = find_profile(run_dir, params, iteration=best_att["iteration"])
    else:
        if best_att and best_att["iteration"] != iteration:
            other = find_profile(run_dir, params, iteration=best_att["iteration"])
        if other is None:
            for lib in REFERENCES.get(cfg["category"], {}):
                other = find_profile(run_dir, params, reference=lib)
                if other:
                    break
    if other:
        print("\n" + "\n".join(diff_summaries(s, other)))
    print(f"\nsaved: {out_dir.relative_to(pgaio.REPO)}/ (summary.json, summary.txt, raw.csv, "
          f"source.csv, details.csv, report.ncu-rep: open it in ncu-ui)")


def cmd_reference(a):
    run_dir, cfg = pgaio.load_run(a.run)
    kernel, err = reference_kernel(cfg["category"], a.lib)
    if err:
        sys.exit(err)
    train, _ = pgaio.run_shapes(cfg)
    args = timing_args(cfg, ["--run", kernel], train)
    if a.dry_run:
        print(f"would time {kernel} with eval's timing settings over {len(train)} shape(s):")
        print(f"{cfg['arena_cmd']} {shlex.join(args)}")
        print(f"and write {(run_dir / 'references' / (a.lib + '.json')).relative_to(pgaio.REPO)}")
        return
    print(f"[pgaio] timing {kernel} ...", file=sys.stderr)
    code, data, _, err = pgaio.arena(cfg["arena_cmd"], args)
    results = (data or {}).get("results", [])
    if not results:
        sys.exit(f"arena did not run {kernel} (exit {code}); is it built into this binary?\n{err[-2000:]}")
    failed = [r for r in results if not r.get("success")]
    if failed:
        sys.exit(f"{kernel} failed: {failed[0].get('error', '')[-2000:]}")
    env = run_env(cfg, (data or {}).get("environment", {}))
    rule = pgaio.score_rule(cfg)
    sizes = reference_sizes(results, env, rule)
    out = {"lib": a.lib, "kernel": kernel, "time": pgaio.now(), "sizes": sizes,
           "score": pgaio.geomean([s["fraction"] for s in sizes]),
           "environment": {k: env.get(k) for k in ("device_name", "arch", "peak_dram_gbps",
                                                   "peak_fp32_gflops", "launch_floor_ms",
                                                   "roundtrip_floor_ms", "driver_version",
                                                   "cuda_toolkit_version")},
           "note": "reference only: timed like eval, never an iteration or a score of the run"}
    if rule >= 2:
        out["score_rule"] = rule
    path = run_dir / "references" / f"{a.lib}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))

    print(f"{a.lib} ({kernel}): SOL {out['score']:.1%} (geometric mean over {len(sizes)} shapes, "
          "same floors and timing as eval)")
    best_att, best_obs = pgaio.best_of(run_dir, pgaio.load_attempts(run_dir))
    rel = {json.dumps(p, sort_keys=True): (r, c, x) for p, r, c, x in
           percent_of_reference(sizes, (best_obs or {}).get("sizes"))}
    head = f"{'shape':>22} {'op ms':>10} {'SOL':>7}  kernels"
    if best_att:
        head += f"   | iteration {best_att['iteration']}: op ms, % of {a.lib}"
    print(head)
    for s in sizes:
        line = (f"{pgaio.shape_label(s['params']):>22} {s['op_ms']:>10.4f} {s['fraction']:>7.1%}  "
                f"{', '.join(kernel_names(s['launched_kernels'])) or '-'}")
        hit = rel.get(json.dumps(s["params"], sort_keys=True))
        if hit:
            line += f"   | {hit[1]:.4f} ms, {hit[2]:.1%}"
        if s.get("verified") is False:
            line += "   (did not verify)"
        print(line)
    if best_att:
        print(f"% of {a.lib} = {a.lib} op_ms / candidate op_ms; above 100% the candidate is faster")
    print(f"saved: {path.relative_to(pgaio.REPO)}")


def add_parsers(sub):
    """Adds `profile` and `reference` to pgaio.py's subcommands."""
    pr = sub.add_parser("profile", help="ncu on an evaluated iteration or a library reference; "
                        "free, never a score (locked clock, replayed)")
    pr.add_argument("run")
    pr.add_argument("iteration", type=int, nargs="?",
                    help="an iteration that passed eval (verdict ok)")
    pr.add_argument("--shape", help="e.g. K=2048,M=2048,N=2048 (default: the run's largest timed shape)")
    pr.add_argument("--clock-control", choices=["base", "none"], default="base",
                    help="base (default) locks clocks to base so profiles compare; none leaves "
                         "them free")
    pr.add_argument("--reference", metavar="LIB", help="profile the library reference instead, e.g. cublas")
    pr.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    pr.set_defaults(fn=cmd_profile)

    rf = sub.add_parser("reference", help="time a library reference with eval's settings; "
                        "free, not an iteration")
    rf.add_argument("run")
    rf.add_argument("--lib", required=True, help="e.g. cublas")
    rf.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    rf.set_defaults(fn=cmd_reference)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_parsers(p.add_subparsers(dest="cmd", required=True))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

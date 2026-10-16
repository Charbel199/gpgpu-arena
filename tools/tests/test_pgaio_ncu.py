"""Unit tests for the pure parts of tools/pgaio_ncu.py. No GPU, no ncu.

    python3 -m unittest discover -s tools/tests

The CSV snippets are cut from real ncu 2025.4 exports of a matmul kernel
(`--page raw --csv` and `--page source --print-source sass --csv`), with
most columns dropped.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pgaio_ncu as ncu  # noqa: E402

HEAD = '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC"'
UNIT_HEAD = '"","","","","","","","","","",""'


def raw_csv(kernels):
    """kernels: [(name, {metric: (unit, value)})], same metrics for each."""
    names = list(kernels[0][1])
    lines = [HEAD + "," + ",".join(f'"{n}"' for n in names),
             UNIT_HEAD + "," + ",".join(f'"{kernels[0][1][n][0]}"' for n in names)]
    for i, (k, ms) in enumerate(kernels):
        lines.append(f'"{i}","72","arena","127.0.0.1","{k}","1","7","(256, 1, 1)","(512, 1, 1)","0","12.0",'
                     + ",".join(f'"{ms[n][1]}"' for n in names))
    return "\n".join(lines) + "\n"


def metrics(duration_us="446.62", issue="70.20", fma="62.32", lsu="35.74", read_mb="34.49",
            write_mb="6.53", conflicts="3,218,030", wavefronts="51,064,179",
            long_sb="1548", short_sb="1481", selected="9451", clock="1.47"):
    return {
        "gpu__time_duration.sum": ("us", duration_us),
        "sm__cycles_elapsed.avg.per_second": ("Ghz", clock),
        "smsp__issue_active.avg.pct_of_peak_sustained_active": ("%", issue),
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": ("%", fma),
        "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active": ("%", "0.13"),
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": ("%", lsu),
        "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": ("%", "0"),
        "dram__bytes_op_read.sum": ("Mbyte", read_mb),
        "dram__bytes_op_write.sum": ("Mbyte", write_mb),
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": ("", conflicts),
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum": ("", wavefronts),
        "lts__t_sectors.sum": ("sector", "16768651.000000"),
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard": ("warp", long_sb),
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued": ("warp", "1507"),
        "smsp__pcsamp_warps_issue_stalled_short_scoreboard": ("warp", short_sb),
        "smsp__pcsamp_warps_issue_stalled_selected": ("warp", selected),
        "warpsampling:smsp__pcsamp_warps_issue_stalled_long_scoreboard": ("warp", "12408.000000"),
    }


SOURCE = '''"Kernel Name","sgemm_big",
"Address","Source","Warp Stall Sampling (All Samples)","Warp Stall Sampling (Not-issued Samples)","# Samples","Instructions Executed","stall_barrier","stall_long_sb","stall_not_selected","stall_selected","stall_short_sb","stall_barrier (Not Issued)","stall_short_sb (Not Issued)"
"0x71cd5b6f4c00","      LDC R1, c[0x0][0x37c]","38","38","38","4096","0","0","0","38","0","0","0"
"0x71cd5b6f4c10","      FFMA R3, R4, R5, R3","968","400","968","4096","0","0","200","368","400","0","400"
"0x71cd5b6f4c20","      @!P0 BAR.SYNC.DEFER_BLOCKING 0x0","500","500","500","4096","500","0","0","0","0","500","0"
"0x71cd5b6f4c30","      LDS.128 R8, [R2+0x10]","300","100","300","4096","0","0","0","200","100","0","100"
"0x71cd5b6f4c40","      EXIT","0","0","0","4096","0","0","0","0","0","0","0"

"Kernel Name","second_kernel",
"Address","Source","Warp Stall Sampling (All Samples)","Warp Stall Sampling (Not-issued Samples)","# Samples","Instructions Executed","stall_barrier","stall_long_sb","stall_not_selected","stall_selected","stall_short_sb","stall_barrier (Not Issued)","stall_short_sb (Not Issued)"
"0x71cd5b700000","      LDG.E R0, desc[UR4][R2.64]","194","194","194","128","0","194","0","0","0","0","0"
'''


class TestNumbersAndUnits(unittest.TestCase):
    def test_number(self):
        self.assertEqual(ncu.number("3,218,030"), 3218030.0)
        self.assertEqual(ncu.number("446.62"), 446.62)
        self.assertIsNone(ncu.number("-"))
        self.assertIsNone(ncu.number(""))
        self.assertIsNone(ncu.number("PolicySpread"))

    def test_units(self):
        self.assertAlmostEqual(ncu.to_base(446.62, "us", "ms"), 0.44662)
        self.assertAlmostEqual(ncu.to_base(459360, "ns", "ms"), 0.45936)
        self.assertAlmostEqual(ncu.to_base(34.49, "Mbyte", "bytes"), 34.49e6)
        self.assertAlmostEqual(ncu.to_base(1.47, "Ghz", "mhz"), 1470.0)
        self.assertEqual(ncu.to_base(5.0, "", "bytes"), 5.0)
        with self.assertRaises(ValueError):
            ncu.to_base(1.0, "Mbyte", "ms")


class TestRaw(unittest.TestCase):
    def test_parse(self):
        rows = ncu.parse_raw_csv("==PROF== note\n" + raw_csv([("sgemm_big", metrics())]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kernel"], "sgemm_big")
        self.assertEqual(rows[0]["metrics"]["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"],
                         (3218030.0, ""))
        self.assertEqual(rows[0]["metrics"]["gpu__time_duration.sum"], (446.62, "us"))
        self.assertEqual(ncu.parse_raw_csv(""), [])

    def test_summary_one_kernel(self):
        m = ncu.summarize_metrics(ncu.parse_raw_csv(raw_csv([("sgemm_big", metrics())])))
        self.assertAlmostEqual(m["duration_ms"], 0.44662)
        self.assertAlmostEqual(m["clock_mhz"], 1470.0)
        self.assertAlmostEqual(m["issue_active"], 0.702)
        # sub-pipes and empty pipes are left out, biggest first
        self.assertEqual(list(m["pipes"]), ["fma", "lsu"])
        self.assertAlmostEqual(m["pipes"]["fma"], 0.6232)
        # issued samples are not stalls; not_issued splits and warpsampling are not double counted
        self.assertEqual(set(m["stalls"]), {"long_scoreboard", "short_scoreboard"})
        self.assertAlmostEqual(m["stalls"]["long_scoreboard"], round(1548 / 3029, 4))
        self.assertEqual(m["stall_samples"], 3029)
        self.assertAlmostEqual(m["dram_bytes"], 41.02e6)
        self.assertEqual(m["shared_bank_conflicts"], 3218030.0)
        self.assertEqual(m["shared_wavefronts"], 51064179.0)
        self.assertEqual(m["l2_sectors"], 16768651.0)

    def test_summary_sums_counts_and_weights_rates(self):
        text = raw_csv([("a", metrics(duration_us="300", issue="80", conflicts="10", long_sb="100",
                                      short_sb="0")),
                        ("b", metrics(duration_us="100", issue="40", conflicts="5", long_sb="0",
                                      short_sb="100"))])
        m = ncu.summarize_metrics(ncu.parse_raw_csv(text))
        self.assertAlmostEqual(m["duration_ms"], 0.4)
        self.assertAlmostEqual(m["issue_active"], 0.7)        # 0.75 * 0.8 + 0.25 * 0.4
        self.assertEqual(m["shared_bank_conflicts"], 15.0)
        self.assertEqual(m["stalls"], {"long_scoreboard": 0.5, "short_scoreboard": 0.5})

    def test_warpsampling_fallback(self):
        ms = {k: v for k, v in metrics().items() if not k.startswith("smsp__pcsamp")}
        m = ncu.summarize_metrics(ncu.parse_raw_csv(raw_csv([("k", ms)])))
        self.assertEqual(m["stalls"], {"long_scoreboard": 1.0})


class TestSource(unittest.TestCase):
    def setUp(self):
        self.instrs = ncu.parse_source_csv(SOURCE)

    def test_parse(self):
        self.assertEqual(len(self.instrs), 6)
        first, ffma, bar = self.instrs[:3]
        self.assertEqual((first["opcode"], first["offset"], first["samples"]), ("LDC", "0000", 38.0))
        self.assertEqual(ffma["offset"], "0010")
        self.assertEqual(ffma["stalls"], {"not_selected": 200.0, "selected": 368.0, "short_sb": 400.0})
        self.assertEqual(bar["opcode"], "BAR.SYNC.DEFER_BLOCKING")      # predicate stripped
        # offsets restart at each kernel
        self.assertEqual((self.instrs[5]["kernel"], self.instrs[5]["offset"]), ("second_kernel", "0000"))

    def test_main_stall(self):
        self.assertEqual(ncu.main_stall({"selected": 900, "short_sb": 400, "not_selected": 200}), "short_sb")
        self.assertEqual(ncu.main_stall({"selected": 38}), "selected")
        self.assertEqual(ncu.main_stall({}), "")

    def test_top_instructions(self):
        top = ncu.top_instructions(self.instrs, limit=3)
        self.assertEqual([t["opcode"] for t in top], ["FFMA", "BAR.SYNC.DEFER_BLOCKING", "LDS.128"])
        self.assertEqual(top[0]["stall"], "short_sb")
        self.assertEqual(top[0]["samples"], 968)
        self.assertAlmostEqual(top[0]["share"], round(968 / 2000, 4))
        # instructions never sampled are not listed
        self.assertNotIn("EXIT", [t["opcode"] for t in ncu.top_instructions(self.instrs)])

    def test_hot_loop(self):
        # 968 + 500 = 73% of 2000, so LDS.128 is needed to pass 80%
        self.assertEqual(ncu.hot_loop_opcodes(self.instrs, cover=0.8),
                         {"BAR.SYNC.DEFER_BLOCKING": 1, "FFMA": 1, "LDS.128": 1})
        self.assertEqual(ncu.hot_loop_opcodes(self.instrs, cover=0.4), {"FFMA": 1})
        self.assertEqual(ncu.hot_loop_opcodes([]), {})


class TestSummary(unittest.TestCase):
    def summary(self, **kw):
        return ncu.summarize(raw_csv([("sgemm_big", metrics(**kw))]), SOURCE, iteration=16,
                             params={"K": 2048, "M": 2048, "N": 2048})

    def test_fields(self):
        s = self.summary()
        for key in ("iteration", "reference", "params", "kernels", "duration_ms", "clock_mhz", "note",
                    "issue_active", "pipes", "stalls", "shared_bank_conflicts", "shared_wavefronts",
                    "dram_bytes", "l2_sectors", "top_instructions", "hot_loop_opcodes"):
            self.assertIn(key, s)
        self.assertEqual(s["note"], "locked clock, replayed")
        self.assertEqual(s["kernels"], ["sgemm_big"])
        unlocked = ncu.summarize("", "", clock_control="none")
        self.assertEqual(unlocked["note"], "unlocked clock, replayed")
        self.assertIsNone(unlocked["duration_ms"])

    def test_render_labels_duration(self):
        text = ncu.render_summary(self.summary())
        self.assertIn("locked clock, replayed", text)
        self.assertIn("not a score", text)
        self.assertIn("long_scoreboard", text)
        self.assertNotIn("\u2014", text)

    def test_diff(self):
        cur = self.summary(long_sb="230", short_sb="770")
        ref = dict(self.summary(long_sb="110", short_sb="890"), iteration=None, reference="cublas")
        lines = ncu.diff_summaries(cur, ref)
        self.assertIn("vs reference cublas at K=2048 M=2048 N=2048:", lines[0])
        self.assertIn("  stall long_scoreboard: 23% vs 11%", lines)
        self.assertIn("  stall short_scoreboard: 77% vs 89%", lines)
        other = dict(ref, params={"K": 4096})
        self.assertIn("a different shape", ncu.diff_summaries(cur, other)[0])


def ln_metrics(duration_us="7.040000", dram="35.292982", gbps="598.363636", l2="17.902680",
               act_avg="6418.505319", act_max="7263.000000", sec_max="7423.000000", grid="1024"):
    """Memory and launch metrics of a layernorm kernel (ln_block128x2p at
    rows=1024 cols=1024 on an RTX PRO 6000 Blackwell), from a real `--set
    full` raw export."""
    return {
        "gpu__time_duration.sum": ("us", duration_us),
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("%", dram),
        "dram__bytes.sum.per_second": ("Gbyte/s", gbps),
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": ("%", l2),
        "lts__t_sector_hit_rate.pct": ("%", "26.268971"),
        "l1tex__throughput.avg.pct_of_peak_sustained_active": ("%", "23.082195"),
        "l1tex__t_sector_hit_rate.pct": ("%", "40.820312"),
        "sm__warps_active.avg.pct_of_peak_sustained_active": ("%", "40.306422"),
        "launch__grid_size": ("", grid),
        "launch__block_size": ("", "128"),
        "launch__sm_count": ("SM", "188"),
        "launch__waves_per_multiprocessor": ("", "0.45"),
        "launch__occupancy_limit_blocks": ("block", "24.000000"),
        "launch__occupancy_limit_registers": ("block", "12.000000"),
        "launch__occupancy_limit_shared_mem": ("block", "28.000000"),
        "launch__occupancy_limit_warps": ("block", "12.000000"),
        "launch__occupancy_limit_barriers": ("block", "24.000000"),
        "sm__cycles_active.avg": ("cycle", act_avg),
        "sm__cycles_active.max": ("cycle", act_max),
        "sm__cycles_elapsed.avg": ("cycle", "10422.553191"),
        "lts__t_sectors.avg": ("sector", "5020.062500"),
        "lts__t_sectors.max": ("sector", sec_max),
        "lts__t_sectors.min": ("sector", "4213.000000"),
        "lts__t_sectors.max.pct_of_peak_sustained_elapsed": ("%", "26.465345"),
        "lts__cycles_active.avg": ("cycle", "12431.921875"),
        "lts__cycles_active.max": ("cycle", "13022.000000"),
        "device__attribute_l2s_count": ("", "96"),
    }


# obs/018.json of the layernorm run, cut down to what eval_timing reads.
OBS = {
    "sizes": [{"params": {"cols": 1024, "rows": 64}, "op_ms": 0.0057, "gpu_ms": 0.0038},
              {"params": {"cols": 1024, "rows": 1024}, "op_ms": 0.0085, "gpu_ms": 0.0074,
               "clock": {"sm_clock_mean_mhz": 2812.0}}],
    "raw": {"timing_results": [
        {"input": {"params": {"cols": 1024, "rows": 1024}},
         "timing": {"op_ms": 0.0084, "gpu_ms": 0.0074, "launch_count": 1, "overhead_ms": 0.001}}]},
}


class TestMemoryAndTime(unittest.TestCase):
    PARAMS = {"rows": 1024, "cols": 1024}

    def launches(self, *kernels):
        return ncu.parse_raw_csv(raw_csv(list(kernels) or [("ln_block128x2p", ln_metrics())]))

    def test_bandwidth_units(self):
        self.assertAlmostEqual(ncu.to_base(598.36, "Gbyte/s", "gbps"), 598.36)
        self.assertAlmostEqual(ncu.to_base(1.2, "Tbyte/s", "gbps"), 1200.0)

    def test_launch_detail(self):
        d = ncu.launch_detail(self.launches()[0])
        self.assertEqual((d["grid"], d["block"], d["sm_count"], d["blocks_per_sm"]), (1024, 128, 188, 12))
        self.assertAlmostEqual(d["waves"], round(1024 / (12 * 188), 3))
        self.assertAlmostEqual(d["sm_tail"], round(6418.505319 / 7263, 4))
        self.assertAlmostEqual(d["sm_busy"], round(6418.505319 / 10422.553191, 4))
        self.assertAlmostEqual(d["dram_pct"], 0.3529)
        self.assertAlmostEqual(d["dram_gbps"], 598.4)
        self.assertAlmostEqual(d["l2_slice_max_over_avg"], round(7423 / 5020.0625, 4))
        self.assertAlmostEqual(d["duration_ms"], 0.00704)

    def test_memory_summary(self):
        m = ncu.memory_summary(self.launches())
        self.assertAlmostEqual(m["dram_pct"], 0.3529)
        self.assertAlmostEqual(m["l2_pct"], 0.179)
        self.assertAlmostEqual(m["l2_hit_rate"], 0.2627)
        self.assertAlmostEqual(m["l1_pct"], 0.2308)
        self.assertAlmostEqual(m["occupancy"], 0.4031)
        self.assertEqual(m["l2_slices"], 96)
        self.assertAlmostEqual(m["l2_slice_min_over_avg"], round(4213 / 5020.0625, 4))
        self.assertAlmostEqual(m["l2_hot_slice_pct"], 0.2647)
        self.assertIsNone(ncu.memory_summary([]))
        # a profile without these metrics (an older or narrower set) has none
        self.assertIsNone(ncu.memory_summary(ncu.parse_raw_csv(raw_csv([("k", metrics())]))))

    def test_memory_summary_weights_by_duration(self):
        m = ncu.memory_summary(self.launches(("a", ln_metrics(duration_us="30", dram="80", gbps="1000",
                                                               sec_max="10040.125")),
                                             ("b", ln_metrics(duration_us="10", dram="40", gbps="600"))))
        self.assertAlmostEqual(m["dram_pct"], 0.7)
        self.assertAlmostEqual(m["dram_gbps"], 900.0)
        # slices: summed busiest over summed mean
        self.assertAlmostEqual(m["l2_slice_max_over_avg"], round((10040.125 + 7423) / (2 * 5020.0625), 4))

    def test_eval_timing(self):
        e = ncu.eval_timing(OBS, self.PARAMS)
        self.assertEqual(e, {"op_ms": 0.0085, "op_stat": "interquartile mean", "op_ms_median": 0.0084,
                             "gpu_ms": 0.0074, "overhead_ms": 0.001, "launch_count": 1,
                             "sm_clock_mhz": 2812.0})
        self.assertIsNone(ncu.eval_timing(OBS, {"rows": 1, "cols": 1}))
        self.assertIsNone(ncu.eval_timing(None, self.PARAMS))

    def test_eval_timing_under_rule_2(self):
        # the score statistic can sit below the profiled kernel time, and
        # the parked-write charge is not launch overhead
        obs = {"score_rule": 2,
               "sizes": [{"params": self.PARAMS, "op_ms": 0.366, "op_ms_raw": 0.341, "parked_ms": 0.025,
                          "gpu_ms": 0.3547},
                         {"params": {"rows": 64, "cols": 1024}, "op_ms": 0.00385, "gpu_ms": 0.00429}],
               "raw": {"timing_results": [
                   {"input": {"params": self.PARAMS}, "timing": {"op_ms": 0.3525, "overhead_ms": 0.0}},
                   {"input": {"params": {"rows": 64, "cols": 1024}}, "timing": {"op_ms": 0.0044}}]}}
        e = ncu.eval_timing(obs, self.PARAMS)
        self.assertEqual((e["op_ms"], e["op_stat"], e["parked_ms"]), (0.341, "mean of the fastest half", 0.025))
        t = ncu.time_breakdown(e, [])
        self.assertEqual(t["eval"]["launch_gap_ms"], 0.0)
        self.assertIn("the score adds 25.0 us for writes parked in L2", t["text"])
        t = ncu.time_breakdown(ncu.eval_timing(obs, {"rows": 64, "cols": 1024}), [])
        self.assertAlmostEqual(t["eval"]["launch_gap_ms"], 0.0044 - 0.00429)
        self.assertGreaterEqual(t["eval"]["launch_gap_ms"], 0.0)

    def test_time_breakdown(self):
        details = [ncu.launch_detail(l) for l in self.launches()]
        t = ncu.time_breakdown(ncu.eval_timing(OBS, self.PARAMS), details, 0.00704, 1480.0)
        self.assertAlmostEqual(t["eval"]["launch_gap_ms"], 0.001)
        self.assertAlmostEqual(t["eval"]["kernel_share"], round(0.0074 / 0.0084, 4))
        self.assertIn("real clock", t["eval"]["note"])
        self.assertEqual(t["ncu"]["note"], "locked clock, replayed")
        self.assertAlmostEqual(t["ncu"]["sm_tail"], round(6418.505319 / 7263, 4))
        self.assertTrue(t["text"].startswith("op 8.5 us (eval, interquartile mean, real clock, 2812 MHz, "
                                             "median 8.4 us): kernels 7.4 us, "
                                             "launch and gaps 1.0 us (1 launch)"), t["text"])
        self.assertIn("ncu at 1480 MHz, locked clock, replayed: 7.0 us", t["text"])
        self.assertIn("SMs busy on average 88% of the longest SM's time (tail)", t["text"])
        # a reference has no eval observation: ncu only
        t = ncu.time_breakdown(None, details, 0.00704, 1480.0)
        self.assertNotIn("eval", t)
        self.assertTrue(t["text"].startswith("in the kernel"))
        self.assertIsNone(ncu.time_breakdown(None, [], None))

    def test_summary_render_and_diff(self):
        p = self.PARAMS
        cur = ncu.summarize(raw_csv([("ln_block128x2p", ln_metrics())]), "", iteration=18, params=p,
                            evalt=ncu.eval_timing(OBS, p))
        for key in ("memory", "launch_details", "time_breakdown"):
            self.assertIn(key, cur)
        text = ncu.render_summary(cur)
        self.assertIn("memory (locked clock, replayed): DRAM 35% of peak (598 GB/s), L2 18% of peak "
                      "(hit rate 26%)", text)
        self.assertIn("0.45 waves (1024 blocks / (12 per SM x 188 SMs))", text)
        self.assertIn("L2 slices (96): busiest 1.48x the mean slice's sectors", text)
        self.assertIn("time: op 8.5 us (eval", text)
        self.assertNotIn("\u2014", text)
        ref = ncu.summarize(raw_csv([("ln_block128x2p", ln_metrics(dram="41.0", sec_max="5522.0"))]), "",
                            iteration=16, params=p)
        lines = ncu.diff_summaries(cur, ref)
        self.assertIn("  DRAM throughput (of peak): 35% vs 41%", lines)
        self.assertIn("  L2 busiest slice / mean: 1.48 vs 1.10", lines)
        self.assertIn("  SM tail (avg/max active cycles): 88% vs 88%", lines)
        # the reference side has no eval timing, so no eval rows
        self.assertFalse(any("(eval)" in l for l in lines))

    def test_old_summaries_still_render_and_diff(self):
        p = self.PARAMS
        cur = ncu.summarize(raw_csv([("ln_block128x2p", ln_metrics())]), "", iteration=18, params=p)
        old = ncu.summarize(raw_csv([("sgemm_big", metrics())]), SOURCE, iteration=2, params=p)
        for key in ("memory", "launch_details", "time_breakdown"):
            old.pop(key)
        self.assertNotIn("memory (", ncu.render_summary(old))
        lines = ncu.diff_summaries(cur, old)
        self.assertFalse(any("DRAM throughput" in l for l in lines))


class TestCommandBuilding(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(ncu.parse_shape("K=2048,M=1024,N=512"), {"K": 2048, "M": 1024, "N": 512})
        with self.assertRaises(ValueError):
            ncu.parse_shape("K")
        shapes = [{"M": 2048, "N": 2048, "K": 2048}, {"M": 8192, "N": 8192, "K": 8192},
                  {"M": 4096, "N": 4096, "K": 4096}]
        self.assertEqual(ncu.largest_shape(shapes)["M"], 8192)
        self.assertEqual(ncu.profile_dir_name(16, params=shapes[1]), "016")
        self.assertEqual(ncu.profile_dir_name(16, params=shapes[0], default=False), "016-K2048-M2048-N2048")
        self.assertEqual(ncu.profile_dir_name(reference="cublas", params=shapes[0]),
                         "ref-cublas-K2048-M2048-N2048")

    def test_kernel_filter(self):
        self.assertEqual(ncu.kernel_filter(["sgemm_big"]), ("mangled", "regex:^(sgemm_big)$"))
        self.assertEqual(ncu.kernel_filter(["a", "b.x", "a"]), ("mangled", r"regex:^(a|b\.x)$"))
        base, flt = ncu.kernel_filter(["void ns::gemm<float>(float*, int)"])
        self.assertEqual((base, flt), ("function", "regex:^(gemm)$"))

    def test_kernel_filter_function_names(self):
        # arena's short names keep namespaces and template arguments
        self.assertEqual(ncu.kernel_filter(["cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>"]),
                         ("function", "regex:^(Kernel2)$"))
        # a template function in the global namespace keeps its return type when demangled
        self.assertEqual(ncu.kernel_filter(["void gemm<float>(float*, int)"])[1], "regex:^(gemm)$")
        # anonymous namespaces and class templates
        self.assertEqual(ncu.function_name("(anonymous namespace)::reduce<256>"), "reduce")
        self.assertEqual(ncu.function_name("void (anonymous namespace)::k<(int)4>(float*)"), "k")
        self.assertEqual(ncu.function_name("ns::Outer<int>::inner"), "inner")
        self.assertEqual(ncu.function_name("plain"), "plain")

    def test_ncu_args_skip_the_warmup_op(self):
        args = ncu.ncu_args(["k1", "k2"], 2, "{out}", warmup=1)
        self.assertEqual(args[args.index("--launch-skip") + 1], "2")
        self.assertEqual(args[args.index("--launch-count") + 1], "2")
        self.assertEqual(args[args.index("--clock-control") + 1], "base")
        self.assertIn("SourceCounters", args)
        self.assertNotIn("--profile", args)

    def test_docker_command(self):
        repo = Path("/repo")
        argv, cwd = ncu.build_script("docker compose run --rm -e ARENA_SKIP_BUILD=1 -T cli",
                                     ["-o", "{out}/report"], ["--run", "matmul_cublas"],
                                     repo / "pgaio/runs/r/profiles/016", repo=repo, uid=1000, gid=100)
        self.assertEqual(cwd, repo)
        i = argv.index("cli")
        self.assertEqual(argv[i - 2:i], ["--entrypoint", "/bin/sh"])
        self.assertIn("-w", argv[:i])
        script = argv[i + 2]
        out = "/workspace/gpgpu-arena/pgaio/runs/r/profiles/016"
        self.assertIn(f"/usr/local/cuda/bin/ncu -o {out}/report /workspace/gpgpu-arena/build/arena "
                      "--run matmul_cublas", script)
        self.assertIn("--page raw --csv", script)
        self.assertIn("--page source --print-source sass --csv", script)
        self.assertIn(f"chown -R 1000:100 {out}", script)
        self.assertTrue(script.endswith("exit $rc"))

    def test_native_command(self):
        repo = Path("/repo")
        argv, cwd = ncu.build_script("build-native/arena", ["-o", "{out}/report"], ["--run", "x"],
                                     repo / "p", repo=repo, uid=1000, gid=100)
        self.assertEqual(argv[:2], ["sh", "-c"])
        self.assertEqual(cwd, repo / "build-native")
        self.assertIn("/usr/local/cuda/bin/ncu -o /repo/p/report /repo/build-native/arena --run x", argv[2])
        self.assertNotIn("chown", argv[2])


class TestReference(unittest.TestCase):
    def test_known(self):
        self.assertEqual(ncu.reference_kernel("matmul", "cublas"), ("matmul_cublas", None))
        k, err = ncu.reference_kernel("softmax", "cublas")
        self.assertIsNone(k)
        self.assertIn("no library reference", err)
        self.assertIn("known: cublas", ncu.reference_kernel("matmul", "cutlass")[1])

    def test_kernel_names_from_arena_objects(self):
        launched = [{"name": "cutlass_sgemm", "count_per_op": 1, "gpu_ms": 0.25},
                    {"name": "splitKreduce", "count_per_op": 1, "gpu_ms": 0.01}]
        self.assertEqual(ncu.kernel_names(launched), ["cutlass_sgemm", "splitKreduce"])
        self.assertEqual(ncu.kernel_names(["a", "a", "b"]), ["a", "b"])
        self.assertEqual(ncu.kernel_names(None), [])

    def test_op_launches_count_repeats(self):
        # a kernel launched twice per operation is two launches to skip and profile
        r = {"launched_kernels": [{"name": "reduce_pass", "count_per_op": 2},
                                  {"name": "finish", "count_per_op": 1}],
             "sub_kernels": [{"name": "reduce_pass"}, {"name": "reduce_pass"}, {"name": "finish"}]}
        self.assertEqual(ncu.op_launches(r), ["reduce_pass", "reduce_pass", "finish"])
        self.assertEqual(ncu.op_launches({"launched_kernels": [], "sub_kernels": r["sub_kernels"]}),
                         ["reduce_pass", "reduce_pass", "finish"])
        self.assertEqual(ncu.op_launches({}), [])

    def test_preflight_keeps_every_launch(self):
        r = {"success": True, "launched_kernels": [{"name": "reduce_pass", "count_per_op": 2}],
             "sub_kernels": [{"name": "reduce_pass"}, {"name": "reduce_pass"}]}
        saved = ncu.pgaio.arena
        ncu.pgaio.arena = lambda cmd, args, timeout=1800: (0, {"results": [r]}, "", "")
        try:
            names = ncu.preflight_kernels({"arena_cmd": "x"}, ["--run", "k"], {"N": 1})
        finally:
            ncu.pgaio.arena = saved
        args = ncu.ncu_args(names, len(names), "{out}")
        self.assertEqual(args[args.index("--launch-skip") + 1], "2")
        self.assertEqual(args[args.index("--launch-count") + 1], "2")
        self.assertEqual(args[args.index("-k") + 1], "regex:^(reduce_pass)$")

    def test_problem_baseline_runs_as_init_measured_it(self):
        cfg = {"kind": "problem", "kernel": "matmul_naive", "slot_source": "kernels/matmul/fp32/naive.cu",
               "defines": {"TILE": "16"}}
        att = {"iteration": 0, "candidate": "pgaio/runs/r/candidates/000_baseline"}
        self.assertEqual(ncu.iteration_target(cfg, att), ncu.pgaio.target_args(cfg)[0])
        self.assertIn("TILE=16", ncu.iteration_target(cfg, att))

    def test_timing_args_match_eval(self):
        cfg = {"runs": 100, "profile": True}
        args = ncu.timing_args(cfg, ["--run", "matmul_cublas"], [{"M": 2, "N": 2, "K": 2}])
        self.assertEqual(args, ["--run", "matmul_cublas", "--shape", "K=2,M=2,N=2",
                                "--l2", "cold", "--runs", "100", "--profile"])

    def test_sizes_use_the_run_floors(self):
        cfg = {"launch_floor_ms": 0.002, "roundtrip_floor_ms": 0.004}
        env = ncu.run_env(cfg, {"peak_dram_gbps": 1800.0, "peak_fp32_gflops": 126000.0,
                                "roundtrip_floor_ms": 0.5})
        self.assertEqual(env["roundtrip_floor_ms"], 0.004)
        r = {"input": {"params": {"M": 2048}}, "verified": True,
             "timing": {"op_ms": 0.25}, "throughput": {"gflops": 63000.0, "bandwidth_gbps": 100.0},
             "sub_kernels": [{"name": "cutlass_80_simt_sgemm"}]}
        s = ncu.reference_sizes([r], env)[0]
        self.assertAlmostEqual(s["fraction"], 0.5)
        self.assertEqual(s["launched_kernels"], ["cutlass_80_simt_sgemm"])
        r["launched_kernels"] = ["from_arena"]
        self.assertEqual(ncu.reference_sizes([r], env)[0]["launched_kernels"], ["from_arena"])

    def test_sizes_follow_the_run_score_rule(self):
        # a rule 2 run scores its references with the fastest-half mean too,
        # so % of the library compares one statistic with the same one
        env = {"peak_dram_gbps": 1000.0, "peak_fp32_gflops": 100000.0, "launch_floor_ms": 0.003,
               "roundtrip_floor_ms": 0.004, "l2_cache_bytes": 100e6}
        times = [0.010] * 6 + [0.020] * 4
        r = {"input": {"params": {"n": 1}}, "timing": {"op_ms": 0.010, "op_ms_trimmed": 0.012,
                                                        "all_times_ms": times},
             "throughput": {"bandwidth_gbps": 400.0, "gflops": 0.0,
                            "ideal_read_bytes": 2e6, "ideal_write_bytes": 2e6}}
        self.assertAlmostEqual(ncu.reference_sizes([r], env)[0]["op_ms"], 0.012)
        self.assertAlmostEqual(ncu.reference_sizes([r], env, 2)[0]["op_ms"], 0.010)

    def test_percent_of_reference_uses_measured_times(self):
        ref = [{"params": {"M": 1}, "op_ms": 0.9}]
        cand = [{"params": {"M": 1}, "op_ms": 1.2, "op_ms_raw": 1.0}]
        self.assertEqual(ncu.percent_of_reference(ref, cand), [({"M": 1}, 0.9, 1.0, 0.9)])

    def test_percent_of_reference(self):
        ref = [{"params": {"M": 1}, "op_ms": 0.9}, {"params": {"M": 2}, "op_ms": 2.0}]
        cand = [{"params": {"M": 1}, "op_ms": 1.0}]
        self.assertEqual(ncu.percent_of_reference(ref, cand), [({"M": 1}, 0.9, 1.0, 0.9)])
        self.assertEqual(ncu.percent_of_reference(ref, None), [])


if __name__ == "__main__":
    unittest.main()

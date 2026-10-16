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

    def test_percent_of_reference(self):
        ref = [{"params": {"M": 1}, "op_ms": 0.9}, {"params": {"M": 2}, "op_ms": 2.0}]
        cand = [{"params": {"M": 1}, "op_ms": 1.0}]
        self.assertEqual(ncu.percent_of_reference(ref, cand), [({"M": 1}, 0.9, 1.0, 0.9)])
        self.assertEqual(ncu.percent_of_reference(ref, None), [])


if __name__ == "__main__":
    unittest.main()

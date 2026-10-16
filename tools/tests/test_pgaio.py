"""Unit tests for the pure parts of tools/pgaio.py. No GPU, no arena binary.

    python3 -m unittest discover -s tools/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pgaio  # noqa: E402

ENV = {"peak_dram_gbps": 1800.0, "peak_fp32_gflops": 126000.0}


def result(kernel="k", n=1000, dist="uniform", seed=1, success=True, verified=True,
           error="", gflops=0.0, gbps=0.0, op_ms=1.0, gpu_ms=1.0):
    return {
        "kernel": kernel, "success": success, "verified": verified, "error": error,
        "input": {"params": {"n": n}, "distribution": dist, "seed": seed},
        "throughput": {"gflops": gflops, "bandwidth_gbps": gbps},
        "timing": {"op_ms": op_ms, "gpu_ms": gpu_ms, "launch_count": 1},
        "numerics": {"max_rel_error": 0.5, "tolerance": 0.1, "max_total_error": 1e-6},
    }


class TestParamsArgs(unittest.TestCase):
    def test_lists_and_order(self):
        self.assertEqual(pgaio.params_args({"n": [1, 2]}), ["-p", "n=1,2"])
        self.assertEqual(pgaio.params_args({"N": [3], "M": [1]}), ["-p", "M=1", "-p", "N=3"])
        self.assertEqual(pgaio.params_args({}), [])


class TestComboKey(unittest.TestCase):
    def test_distinguishes_every_field(self):
        keys = {pgaio.combo_key(result(n=1)), pgaio.combo_key(result(n=2)),
                pgaio.combo_key(result(dist="normal")), pgaio.combo_key(result(seed=9))}
        self.assertEqual(len(keys), 4)

    def test_param_order_does_not_matter(self):
        a = {"input": {"params": {"M": 1, "N": 2}, "distribution": "d", "seed": 1}}
        b = {"input": {"params": {"N": 2, "M": 1}, "distribution": "d", "seed": 1}}
        self.assertEqual(pgaio.combo_key(a), pgaio.combo_key(b))


class TestSol(unittest.TestCase):
    def test_memory_bound_reduce(self):
        # reduce: 0.25 FLOP/byte, far left of the ridge
        s = pgaio.sol(result(gflops=225.0, gbps=900.0), ENV)
        self.assertEqual(s["bound"], "memory")
        self.assertAlmostEqual(s["fraction"], 0.5)
        self.assertAlmostEqual(s["intensity"], 0.25)
        self.assertAlmostEqual(s["ridge"], 70.0)

    def test_compute_bound_matmul(self):
        # intensity 1000 FLOP/B is past the 70 FLOP/B ridge
        s = pgaio.sol(result(gflops=63000.0, gbps=63.0), ENV)
        self.assertEqual(s["bound"], "compute")
        self.assertAlmostEqual(s["fraction"], 0.5)

    def test_missing_data_scores_zero(self):
        self.assertEqual(pgaio.sol(result(), ENV)["fraction"], 0.0)
        self.assertEqual(pgaio.sol(result(gbps=100.0), {})["fraction"], 0.0)


class TestClassifyAttempt(unittest.TestCase):
    def setUp(self):
        self.checks = [result(n=1), result(n=1000)]
        self.required = [pgaio.combo_key(r) for r in self.checks]

    def classify(self, checks=None, timing=None, guard=(), ce=0, te=0, required=None):
        return pgaio.classify_attempt(
            self.checks if checks is None else checks,
            [result(n=64)] if timing is None else timing,
            ce, te, list(guard), self.required if required is None else required)

    def test_ok(self):
        self.assertEqual(self.classify(), "ok")

    def test_guard_wins_over_everything(self):
        self.assertEqual(self.classify(guard=["src/runner.cpp"]), "rejected_guard")

    def test_compile_error(self):
        bad = [result(success=False, error="nvcc for /x.cu failed:\nerror: expected ';'")]
        self.assertEqual(self.classify(checks=bad), "compile_error")
        bad = [result(success=False, error="Triton compiler for x failed: Traceback")]
        self.assertEqual(self.classify(checks=bad), "compile_error")

    def test_runtime_error(self):
        bad = [result(success=False, error="kernel launch failed: an illegal memory access")]
        self.assertEqual(self.classify(checks=bad), "runtime_error")

    def test_no_output_at_all_is_an_error(self):
        self.assertEqual(self.classify(checks=[], timing=[], ce=2), "runtime_error")

    def test_missing_required_check(self):
        self.assertEqual(self.classify(checks=self.checks[:1]), "runtime_error")

    def test_incorrect(self):
        checks = [result(n=1), result(n=1000, verified=False)]
        self.assertEqual(self.classify(checks=checks), "incorrect")

    def test_timing_run_failing_verification_is_incorrect(self):
        self.assertEqual(self.classify(timing=[result(n=64, verified=False)]), "incorrect")

    def test_dropped_check_does_not_gate(self):
        # n=7 failed for the baseline, so it was not made a requirement
        checks = self.checks + [result(n=7, verified=False)]
        self.assertEqual(self.classify(checks=checks), "ok")


class TestGuard(unittest.TestCase):
    def test_changed_added_removed(self):
        before = {"src/a.cpp": "1", "src/b.cpp": "2"}
        after = {"src/a.cpp": "1", "src/b.cpp": "X", "src/c.cpp": "3"}
        self.assertEqual(pgaio.guard_violations(before, after), ["src/b.cpp", "src/c.cpp"])
        self.assertEqual(pgaio.guard_violations(after, {}), sorted(after))
        self.assertEqual(pgaio.guard_violations(before, dict(before)), [])

    def test_is_protected(self):
        for p in ["src/arena/runner.cpp", "kernels/reduce/fp32/baseline.cu",
                  "kernels/reduce/fp32/baseline.cpp", "kernels/triton_base.py",
                  "tools/pgaio.py", "CMakeLists.txt", "tests/test_sass.cpp"]:
            self.assertTrue(pgaio.is_protected(p), p)
        for p in ["pgaio/runs/x/candidates/001.cu", "README.md", "kernels/ext_063ddc11_mm.ttgir",
                  "kernels/ext_2714a1c9_warp_mm.warp_gen.cu", "kernels/reduce_fp32_x.cubin",
                  "kernels/reduce_fp32_x.json", "kernels/__pycache__/triton_base.cpython-312.pyc"]:
            self.assertFalse(pgaio.is_protected(p), p)


class TestCurves(unittest.TestCase):
    def attempts(self, scores):
        return [{"verdict": "ok" if s is not None else "incorrect", "score": s} for s in scores]

    def test_best_so_far_ignores_failures(self):
        curve = pgaio.best_so_far(self.attempts([0.1, None, 0.5, 0.3, None, 0.8]))
        self.assertEqual(curve, [0.1, 0.1, 0.5, 0.5, 0.5, 0.8])

    def test_iterations_to(self):
        curve = [0.1, 0.1, 0.5, 0.5, 0.8]
        self.assertEqual(pgaio.iterations_to(curve, 0.5), 2)
        self.assertEqual(pgaio.iterations_to(curve, 0.8), 4)
        self.assertIsNone(pgaio.iterations_to(curve, 0.9))
        self.assertEqual(pgaio.iterations_to([0.95], 0.9), 0)

    def test_summarize_and_aggregate(self):
        cfg = {"name": "r1", "kernel": "k", "dsl": "cuda", "category": "reduce"}
        rows = [pgaio.summarize_run(cfg, self.attempts([0.1, 0.6, 0.85])),
                pgaio.summarize_run({**cfg, "name": "r2"}, self.attempts([0.1, 0.2, 0.75])),
                pgaio.summarize_run({**cfg, "name": "r3", "dsl": "triton"}, self.attempts([0.7]))]
        self.assertEqual(rows[0]["iterations"], 2)
        self.assertEqual(rows[0]["iters_to_50"], 1)
        self.assertEqual(rows[0]["iters_to_80"], 2)
        self.assertIsNone(rows[1]["iters_to_80"])
        agg = pgaio.aggregate(rows)
        self.assertEqual(agg["reduce: cuda/isolated"]["runs"], 2)
        self.assertEqual(agg["reduce: cuda/isolated"]["iters_to_70"], 2)
        self.assertEqual(agg["reduce: cuda/isolated"]["iters_to_80_reached"], "1/2")
        self.assertEqual(agg["reduce: triton/isolated"]["iters_to_50"], 0)
        self.assertIsNone(agg["reduce: triton/isolated"]["iters_to_90"])

    def test_modes_are_aggregated_separately(self):
        cfg = {"name": "a", "kernel": "k", "dsl": "cuda", "category": "reduce"}
        rows = [pgaio.summarize_run({**cfg, "mode": "isolated"}, self.attempts([0.1])),
                pgaio.summarize_run({**cfg, "name": "b", "mode": "cross-dsl"}, self.attempts([0.9]))]
        agg = pgaio.aggregate(rows)
        self.assertEqual(set(agg), {"reduce: cuda/isolated", "reduce: cuda/cross-dsl"})

    def test_docs_access_is_its_own_group(self):
        cfg = {"name": "a", "kernel": "k", "dsl": "triton", "category": "reduce", "mode": "isolated"}
        rows = [pgaio.summarize_run(cfg, self.attempts([0.5])),
                pgaio.summarize_run({**cfg, "name": "b", "docs": True}, self.attempts([0.6]))]
        self.assertFalse(rows[0]["docs"])
        self.assertEqual(set(pgaio.aggregate(rows)), {"reduce: triton/isolated", "reduce: triton/isolated+docs"})


class TestDiagnose(unittest.TestCase):
    def base_obs(self, **over):
        obs = {
            "category": "reduce",
            "timing": {"op_ms": 1.0, "gpu_ms": 0.95, "launch_count": 1},
            "sol": {"bound": "memory", "fraction": 0.3, "achieved": 540, "roof": 1800},
            "counters": {"available": True, "dram_read_bytes": 268e6, "dram_write_bytes": 0,
                         "occupancy": 0.8},
            "resources": {"local": 0},
            "sass": {"mean_global_load_bytes": 16, "global_atomic": 0, "tensor": 0},
            "ideal_bytes": 268e6,
        }
        for k, v in over.items():
            obs[k] = {**obs[k], **v} if isinstance(v, dict) else v
        return obs

    def has(self, lines, text):
        return any(text in l for l in lines)

    def test_quiet_when_nothing_is_wrong(self):
        lines = pgaio.diagnose(self.base_obs())
        self.assertEqual(len(lines), 1)   # just the SOL statement

    def test_launch_overhead(self):
        d = pgaio.diagnose(self.base_obs(timing={"op_ms": 1.0, "gpu_ms": 0.5, "launch_count": 2}))
        self.assertTrue(self.has(d, "outside kernels"))

    def test_excess_and_short_dram(self):
        self.assertTrue(self.has(pgaio.diagnose(self.base_obs(
            counters={"dram_read_bytes": 600e6})), "the ideal bytes"))
        self.assertTrue(self.has(pgaio.diagnose(self.base_obs(
            counters={"dram_read_bytes": 10e6})), "came from L2"))

    def test_scalar_loads_spills_occupancy_atomics(self):
        d = pgaio.diagnose(self.base_obs(
            sass={"mean_global_load_bytes": 4, "global_atomic": 2},
            sol={"fraction": 0.1}, resources={"local": 32},
            counters={"occupancy": 0.1}))
        for text in ("128-bit", "register spills", "occupancy", "global atomic"):
            self.assertTrue(self.has(d, text), text)

    def test_near_roof(self):
        self.assertTrue(self.has(pgaio.diagnose(self.base_obs(sol={"fraction": 0.9})), "within 15%"))

    def test_matmul_without_tensor_cores(self):
        d = pgaio.diagnose(self.base_obs(category="matmul"))
        self.assertTrue(self.has(d, "tensor core"))


class TestSmallHelpers(unittest.TestCase):
    def test_block_directive(self):
        self.assertEqual(pgaio.block_directive("// arena: block=512\nint x;"), 512)
        self.assertEqual(pgaio.block_directive("# arena:block = 128"), 128)
        self.assertIsNone(pgaio.block_directive("// block=512"))

    def test_candidate_ext(self):
        self.assertEqual(pgaio.candidate_ext("a/b/x.triton.py"), ".triton.py")
        self.assertEqual(pgaio.candidate_ext("x.cu"), ".cu")
        self.assertEqual(pgaio.candidate_ext("Makefile"), "")

    def test_slugify(self):
        self.assertEqual(pgaio.slugify("Vectorize loads (float4)!"), "vectorize-loads-float4")
        self.assertEqual(pgaio.slugify(""), "candidate")
        self.assertLessEqual(len(pgaio.slugify("x" * 100)), 40)

    def test_sass_delta(self):
        d = pgaio.sass_delta({"instructions": 50, "global_load_128": 4, "sync": 1},
                             {"instructions": 60, "global_load_128": 0, "sync": 1})
        self.assertEqual(d, {"instructions": -10, "global_load_128": 4})

    def test_repo_rel(self):
        self.assertEqual(pgaio.repo_rel("/workspace/gpgpu-arena/kernels/a.cu", "/workspace/gpgpu-arena"),
                         "kernels/a.cu")
        self.assertEqual(pgaio.repo_rel("", "/x"), "")

    def test_sass_brief(self):
        b = pgaio.sass_brief({"sass": {"instructions": 10, "mean_global_load_bytes": 16.0,
                                       "classes": {"global_load_128": 2, "global_load_32": 1,
                                                   "global_atomic": 1}}})
        self.assertEqual(b["global_loads"], 3)
        self.assertEqual(b["global_load_128"], 2)
        self.assertEqual(pgaio.sass_brief(None)["instructions"], 0)


if __name__ == "__main__":
    unittest.main()


class TestProblemRuns(unittest.TestCase):
    def cfg(self, **over):
        c = {"kind": "problem", "name": "r", "category": "softmax", "dsls": ["cuda", "triton"],
             "budget": 4, "budget_per_dsl": 2, "target": 0.9, "mode": "isolated"}
        c.update(over)
        return c

    def att(self, it, dsl, score=None, verdict=None):
        return {"iteration": it, "dsl": dsl, "score": score,
                "verdict": verdict or ("ok" if score is not None else "incorrect"),
                "hypothesis": f"h{it}"}

    def test_dsl_of(self):
        self.assertEqual(pgaio.dsl_of("a/b.cu"), "cuda")
        self.assertEqual(pgaio.dsl_of("x.triton.py"), "triton")
        self.assertEqual(pgaio.dsl_of("x.cutile.py"), "cutile")
        self.assertEqual(pgaio.dsl_of("x.warp.py"), "warp")
        self.assertIsNone(pgaio.dsl_of("x.py"))
        self.assertIsNone(pgaio.dsl_of("x.json"))

    def test_budget(self):
        cfg = self.cfg()
        base = [self.att(0, "cuda", 0.7)]
        self.assertIsNone(pgaio.budget_error(cfg, base, "triton"))
        self.assertIn("not allowed", pgaio.budget_error(cfg, base, "warp"))
        two_triton = base + [self.att(1, "triton", 0.5), self.att(2, "triton")]
        self.assertIn("triton budget", pgaio.budget_error(cfg, two_triton, "triton"))
        self.assertIsNone(pgaio.budget_error(cfg, two_triton, "cuda"))
        full = two_triton + [self.att(3, "cuda"), self.att(4, "cuda")]
        self.assertIn("run's budget", pgaio.budget_error(cfg, full, "cuda"))

    def test_budget_baseline_is_free(self):
        cfg = self.cfg(budget=1, budget_per_dsl=1)
        self.assertIsNone(pgaio.budget_error(cfg, [self.att(0, "cuda", 0.7)], "cuda"))

    def test_per_dsl_curves_exclude_the_baseline(self):
        atts = [self.att(0, "cuda", 0.7), self.att(1, "triton", 0.5), self.att(2, "cuda"),
                self.att(3, "triton", 0.8), self.att(4, "cuda", 0.75)]
        curves = pgaio.per_dsl_curves(atts)
        self.assertEqual(curves["triton"], [0.5, 0.8])
        self.assertEqual(curves["cuda"], [0.0, 0.75])

    def test_best_per_dsl_skips_dsls_with_no_pass(self):
        atts = [self.att(0, "cuda", 0.7), self.att(1, "triton", 0.6), self.att(2, "cuda"),
                self.att(3, "triton", 0.8)]
        self.assertEqual(pgaio.best_per_dsl(atts), {"triton": 0.8})

    def test_summarize_dsls(self):
        atts = [self.att(0, "cuda", 0.7), self.att(1, "triton", 0.85), self.att(2, "cuda", 0.6)]
        rows = pgaio.summarize_dsls(self.cfg(), atts)
        by = {r["dsl"]: r for r in rows}
        self.assertEqual(by["triton"]["iters_to_80"], 1)   # first attempt, counted from 1
        self.assertEqual(by["cuda"]["iters_to_50"], 1)
        self.assertIsNone(by["cuda"]["iters_to_70"])
        self.assertEqual(pgaio.summarize_dsls(self.cfg(kind="slot"), atts), [])

    def test_snapshot_manifest(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "draft").mkdir()
            (d / "draft" / "k.triton.py").write_text("kernel v1")
            m = d / "draft" / "m.json"
            m.write_text(json.dumps({"category": "softmax", "source": "k.triton.py", "launches": []}))
            out, dsl = pgaio.snapshot_manifest(m, d / "snap")
            self.assertEqual(dsl, "triton")
            self.assertEqual(json.loads(out.read_text())["source"], "k.triton.py")
            self.assertEqual((d / "snap" / "k.triton.py").read_text(), "kernel v1")
            # later edits to the draft do not touch the snapshot
            (d / "draft" / "k.triton.py").write_text("kernel v2")
            self.assertEqual((d / "snap" / "k.triton.py").read_text(), "kernel v1")

    def test_no_budget_means_no_limit(self):
        cfg = {"kind": "problem", "dsls": ["cuda"], "budget": None, "budget_per_dsl": 0}
        atts = [{"iteration": i, "dsl": "cuda"} for i in range(500)]
        self.assertIsNone(pgaio.budget_error(cfg, atts, "cuda"))
        cfg["budget"] = 3
        self.assertIn("used up", pgaio.budget_error(cfg, atts, "cuda"))

    def test_plateau(self):
        cfg = {"kind": "problem", "dsls": ["cuda", "triton"]}
        def a(i, dsl, score, verdict="ok"):
            return {"iteration": i, "dsl": dsl, "score": score, "verdict": verdict}
        atts = [a(0, "cuda", 0.1), a(1, "cuda", 0.5), a(2, "cuda", 0.6)]
        since, short, flat = pgaio.plateau(cfg, atts)
        self.assertEqual((since, short, flat), (0, ["cuda", "triton"], False))
        # gains under PLATEAU_GAIN and failures do not reset the count
        atts += [a(3 + k, "triton", 0.601) for k in range(5)]
        atts += [a(8 + k, "cuda", None, "incorrect") for k in range(5)]
        since, short, flat = pgaio.plateau(cfg, atts)
        self.assertEqual(since, 10)
        self.assertEqual(short, [])
        self.assertTrue(flat)
        atts.append(a(13, "cuda", 0.7))
        self.assertEqual(pgaio.plateau(cfg, atts)[0], 0)

    def test_snapshot_is_named_for_the_run(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "k.cu").write_text("x")
            m = d / "v3.json"
            m.write_text(json.dumps({"name": "mine", "category": "scan", "source": "k.cu", "launches": []}))
            run = d / "scan-1"
            name = pgaio.candidate_name(run, 6, m)
            self.assertEqual(name, "scan-1/006-v3")
            out, _ = pgaio.snapshot_manifest(m, d / "snap", name=name, description="why")
            self.assertEqual(json.loads(out.read_text())["name"], "scan-1/006-v3")
            pgaio.describe_snapshot(out, 6, "cuda", "why", {"verdict": "ok", "score": 0.5})
            desc = json.loads(out.read_text())["description"]
            self.assertIn("iteration 6 (cuda): ok, score 50.0% of SOL. why", desc)
            # the draft keeps its own name
            self.assertEqual(json.loads(m.read_text())["name"], "mine")

    def test_snapshot_rejects_unknown_dsl(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "k.py").write_text("x")
            m = d / "m.json"
            m.write_text(json.dumps({"source": "k.py"}))
            with self.assertRaises(SystemExit):
                pgaio.snapshot_manifest(m, d / "snap")

    def test_html_page(self):
        atts = [self.att(0, "cuda", 0.7), self.att(1, "triton", 0.85), self.att(2, "cuda")]
        atts[1]["new_best"] = True
        atts[1]["hypothesis"] = "vectorize <loads> & fuse"
        page = pgaio.render_html([(self.cfg(), atts)])
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<title>PGAIO Progression</title>", page)
        self.assertIn("prefers-color-scheme: dark", page)
        self.assertIn("vectorize &lt;loads&gt; &amp; fuse", page)   # escaped
        self.assertNotIn("<loads>", page)
        self.assertEqual(page.count("<circle"), 2)                 # two passing points
        self.assertIn("triton best 85.0%", page)
        self.assertIn("<svg", pgaio.render_html([]) + pgaio.render_run_svg(self.cfg(), atts))


class TestInspectionHelpers(unittest.TestCase):
    SASS = """
        /*0000*/                   LDC R1, c[0x0][0x37c] ;            /* 0x0 */
        /*0010*/                   LDG.E.128 R4, desc[UR4][R2.64] ;   /* 0x0 */
        /*0020*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;      /* 0x0 */
        /*0030*/              @!P0 LDG.E R8, desc[UR4][R6.64] ;       /* 0x0 */
        /*0040*/                   STG.E.128 desc[UR4][R2.64], R4 ;   /* 0x0 */
        /*0050*/                   EXIT ;                             /* 0x0 */
"""

    def test_show_lines_keeps_program_order_and_predicates(self):
        lines = pgaio.show_lines(self.SASS, ["LDG", "BAR"])
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("0010  LDG.E.128"))
        self.assertTrue(lines[1].startswith("0020  BAR.SYNC"))
        self.assertTrue(lines[2].startswith("0030  @!P0 LDG.E"))
        self.assertEqual(pgaio.show_lines(self.SASS, ["STG"]), ["0040  STG.E.128 desc[UR4][R2.64], R4 ;"])

    def test_show_lines_limit(self):
        many = "\n".join(f"/*{i:04x}*/ LDG.E R{i}, desc[UR4][R2.64] ;" for i in range(10))
        out = pgaio.show_lines(many, ["LDG"], limit=3)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[-1], "...")

    def test_top_opcodes(self):
        insp = {"sass": {"opcodes": {"FADD": 3, "LDG.E.128": 8, "BAR.SYNC": 3, "EXIT": 1}}}
        self.assertEqual(pgaio.top_opcodes(insp, limit=3), [("LDG.E.128", 8), ("BAR.SYNC", 3), ("FADD", 3)])
        self.assertEqual(pgaio.top_opcodes(None), [])

    def test_copied_artifacts(self):
        insp = {"artifacts_copied": ["/w/r/inspections/003_t2/kernel.ptx", "/w/r/inspections/003_t2/kernel.ttgir"]}
        self.assertEqual(pgaio.copied_artifacts(insp), ["kernel.ptx", "kernel.ttgir"])
        self.assertEqual(pgaio.copied_artifacts({}), [])

    def test_source_diff(self):
        d = pgaio.source_diff("a\nb\nc\n", "a\nB\nc\n", "old.cu", "new.cu")
        self.assertIn("--- old.cu", d)
        self.assertIn("-b", d)
        self.assertIn("+B", d)
        long = pgaio.source_diff("\n".join(map(str, range(500))), "", "x", "y", limit=10)
        self.assertIn("more lines", long)


class TestSizes(unittest.TestCase):
    ENV = {"peak_dram_gbps": 1800.0, "peak_fp32_gflops": 126000.0, "launch_floor_ms": 0.002}

    def res(self, op_ms, gbps, gflops, params=None):
        return {"input": {"params": params or {"n": 1}},
                "timing": {"op_ms": op_ms, "gpu_ms": op_ms},
                "throughput": {"bandwidth_gbps": gbps, "gflops": gflops}}

    def test_large_shape_is_plain_roofline(self):
        # 180 MB in 0.2 ms: memory roof 0.1 ms, so 50%
        s = pgaio.size_sol(self.res(0.2, 900.0, 225.0), self.ENV)
        self.assertEqual(s["bound"], "memory")
        self.assertAlmostEqual(s["fraction"], 0.5, places=6)
        self.assertAlmostEqual(s["ideal_bytes"], 180e6, delta=1)

    def test_small_shape_uses_the_latency_floor(self):
        # 16 KB in 0.008 ms: bandwidth would say 0.1%, the launch floor 25%
        s = pgaio.size_sol(self.res(0.008, 2.048, 0.5), self.ENV)
        self.assertEqual(s["bound"], "latency")
        self.assertAlmostEqual(s["fraction"], 0.25, places=6)
        # with a measured round trip, that is the roof instead
        s = pgaio.size_sol(self.res(0.008, 2.048, 0.5), {**self.ENV, "roundtrip_floor_ms": 0.004})
        self.assertAlmostEqual(s["fraction"], 0.5, places=6)

    def test_compute_bound(self):
        s = pgaio.size_sol(self.res(2.0, 50.0, 63000.0), self.ENV)
        self.assertEqual(s["bound"], "compute")
        self.assertAlmostEqual(s["fraction"], 0.5, places=6)

    def test_missing_floor_still_scores(self):
        s = pgaio.size_sol(self.res(0.2, 900.0, 225.0), {"peak_dram_gbps": 1800.0})
        self.assertAlmostEqual(s["fraction"], 0.5, places=6)
        self.assertEqual(pgaio.size_sol(self.res(0.0, 0.0, 0.0), self.ENV)["fraction"], 0.0)

    def test_geomean(self):
        self.assertAlmostEqual(pgaio.geomean([0.5, 0.5]), 0.5)
        self.assertAlmostEqual(pgaio.geomean([0.9, 0.1]), 0.3)
        self.assertLess(pgaio.geomean([1.0, 0.0]), 0.01)   # a zero is not averaged away
        self.assertEqual(pgaio.geomean([]), 0.0)

    def test_ladders(self):
        for cat in pgaio.SIZE_LADDERS:
            small = pgaio.train_shapes(cat, "small")
            large = pgaio.train_shapes(cat, "large")
            self.assertTrue(small and large, cat)
            self.assertEqual(pgaio.train_shapes(cat, "all"), small + large)
            self.assertIn(cat, pgaio.CATEGORIES)
            self.assertIn(cat, pgaio.CATEGORY_BASELINE)
        self.assertEqual(pgaio.train_shapes("nope", "all"), [])

    def test_shape_helpers(self):
        self.assertEqual(pgaio.expand_grid({"M": [1, 2], "N": [3]}),
                         [{"M": 1, "N": 3}, {"M": 2, "N": 3}])
        self.assertEqual(pgaio.expand_grid({}), [{}])
        self.assertEqual(pgaio.shape_args([{"rows": 64, "cols": 1024}, {"n": 5}]),
                         ["--shape", "cols=1024,rows=64", "--shape", "n=5"])
        self.assertEqual(pgaio.shape_label({"rows": 64, "cols": 1024}), "rows=64 cols=1024")

    def test_run_shapes_for_old_runs(self):
        new = {"train_shapes": [{"n": 1}], "check_shapes": [{"n": 2}]}
        self.assertEqual(pgaio.run_shapes(new), ([{"n": 1}], [{"n": 2}]))
        old = {"train_params": {"n": [64]}, "check_params": {"n": [1, 64]}}
        train, check = pgaio.run_shapes(old)
        self.assertEqual(train, [{"n": 64}])
        self.assertEqual(check, [{"n": 1}, {"n": 64}])

    def test_small_shape_diagnosis(self):
        obs = {"category": "reduce", "timing": {"op_ms": 1.0, "gpu_ms": 1.0, "launch_count": 1},
               "sol": {"bound": "memory", "fraction": 0.8, "achieved": 1, "roof": 1},
               "counters": {}, "resources": {}, "sass": {}, "ideal_bytes": 0,
               "sizes": [{"params": {"n": 4096}, "bound": "latency", "fraction": 0.2,
                          "op_ms": 0.01, "roof_ms": 0.002}]}
        self.assertTrue(any("latency floor" in d for d in pgaio.diagnose(obs)))


class TestMultiSourceSnapshot(unittest.TestCase):
    def test_launch_sources_are_copied(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.triton.py").write_text("A")
            (d / "sub").mkdir()
            (d / "sub" / "b.triton.py").write_text("B")
            m = d / "m.json"
            m.write_text(json.dumps({"category": "scan", "source": "a.triton.py",
                                     "launches": [{"grid": [1]}, {"source": "sub/b.triton.py", "grid": [1]}]}))
            out, dsl = pgaio.snapshot_manifest(m, d / "snap")
            spec = json.loads(out.read_text())
            self.assertEqual(spec["launches"][1]["source"], "b.triton.py")
            self.assertEqual((d / "snap" / "b.triton.py").read_text(), "B")
            self.assertEqual(dsl, "triton")


class TestMixed(unittest.TestCase):
    def test_manifest_dsl(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("a.cu", "b.triton.py", "c.triton.py"):
                (d / name).write_text("x")
            m = d / "m.json"
            m.write_text(json.dumps({"source": "a.cu", "launches": [{}, {"source": "b.triton.py"}]}))
            self.assertEqual(pgaio.manifest_dsl(m), "mixed")
            out, dsl = pgaio.snapshot_manifest(m, d / "snap")
            self.assertEqual(dsl, "mixed")
            m.write_text(json.dumps({"source": "b.triton.py", "launches": [{}, {"source": "c.triton.py"}]}))
            self.assertEqual(pgaio.manifest_dsl(m), "triton")
            self.assertIsNone(pgaio.manifest_dsl(d / "missing.json"))


# Timings at M=N=K=4096 from iteration 10 of the matmul-fp32-vs-public run
# (sorted, rounded to 1 us): a main body, 5 fast and 9 slow outliers.
MATMUL_4096 = [
    1.979, 2.002, 2.004, 2.011, 2.012, 2.256, 2.259, 2.263, 2.265, 2.265, 2.265, 2.269, 2.272, 2.273,
    2.276, 2.278, 2.28, 2.282, 2.282, 2.282, 2.285, 2.286, 2.292, 2.293, 2.293, 2.294, 2.296, 2.3,
    2.301, 2.305, 2.306, 2.306, 2.307, 2.309, 2.31, 2.313, 2.313, 2.315, 2.315, 2.316, 2.322, 2.324,
    2.324, 2.326, 2.326, 2.326, 2.327, 2.328, 2.334, 2.334, 2.335, 2.336, 2.336, 2.341, 2.345, 2.348,
    2.349, 2.352, 2.353, 2.36, 2.36, 2.369, 2.386, 2.39, 2.39, 2.399, 2.401, 2.405, 2.41, 2.414, 2.423,
    2.429, 2.43, 2.439, 2.441, 2.442, 2.446, 2.453, 2.464, 2.478, 2.479, 2.48, 2.482, 2.485, 2.547,
    2.551, 2.596, 2.597, 2.613, 2.615, 2.632, 2.957, 2.974, 2.984, 2.988, 3.002, 3.016, 3.072, 3.205,
    3.242]


class TestSpread(unittest.TestCase):
    def test_trimmed_mean_matches_arena(self):
        self.assertAlmostEqual(pgaio.trimmed_mean([1, 2, 3, 4, 100, 5, 6, 7]), 4.5)
        self.assertEqual(pgaio.trimmed_mean([2.0]), 2.0)
        self.assertEqual(pgaio.trimmed_mean([]), 0.0)

    def test_matmul_4096_has_outliers_not_two_modes(self):
        s = pgaio.spread(MATMUL_4096)
        self.assertFalse(s["bimodal"])
        self.assertAlmostEqual(s["outliers"], 0.14, places=2)
        self.assertGreater(s["iqr_frac"], 0.03)
        lo, hi = s["ci_ms"]
        self.assertLess(lo, s["trimmed_ms"])
        self.assertGreater(hi, s["trimmed_ms"])
        self.assertLess(s["ci_half_frac"], 0.02)
        self.assertEqual(pgaio.spread(MATMUL_4096), s)   # deterministic
        self.assertIn("outliers", " ".join(pgaio.diagnose_noise(
            {"sizes": [{"params": {"M": 4096}, "spread": s}]})))

    def test_two_clusters_are_bimodal(self):
        times = [2.0 + 0.002 * (i % 7) for i in range(40)] + [2.3 + 0.002 * (i % 5) for i in range(60)]
        s = pgaio.spread(times)
        self.assertTrue(s["bimodal"])
        self.assertAlmostEqual(s["clusters"]["low_share"], 0.4)
        self.assertLess(s["clusters"]["low_ms"], 2.02)
        self.assertGreater(s["clusters"]["high_ms"], 2.29)
        d = pgaio.diagnose_noise({"sizes": [{"params": {"n": 1}, "spread": s}]})
        self.assertTrue(any("bimodal" in x for x in d))

    def test_small_cluster_is_not_a_mode(self):
        # 10% of runs far away: outliers, not a second mode
        times = [1.0 + 0.001 * i for i in range(90)] + [1.5] * 10
        self.assertFalse(pgaio.spread(times)["bimodal"])

    def test_too_few_runs(self):
        self.assertIsNone(pgaio.spread([1.0, 2.0, 3.0]))
        self.assertIsNone(pgaio.spread(None))

    def test_score_ci_brackets_the_score(self):
        sizes = [{"roof_ms": 1.0, "fraction": 0.5, "spread": {"ci_ms": [1.9, 2.1]}},
                 {"roof_ms": 1.0, "fraction": 0.25, "spread": None}]
        lo, hi = pgaio.score_ci(sizes)
        score = pgaio.geomean([0.5, 0.25])
        self.assertLess(lo, score)
        self.assertGreater(hi, score)
        self.assertAlmostEqual(lo, pgaio.geomean([1 / 2.1, 0.25]))
        self.assertIsNone(pgaio.score_ci([]))

    def test_observe_and_render_carry_the_ci(self):
        cfg = {"category": "matmul", "required_checks": [], "target": 1.0}
        tim = {"input": {"params": {"M": 4096}}, "success": True, "verified": True,
               "throughput": {"gflops": 60000.0, "bandwidth_gbps": 80.0},
               "timing": {"op_ms": 2.33, "op_ms_trimmed": 2.34, "gpu_ms": 2.3, "launch_count": 1,
                          "all_times_ms": MATMUL_4096},
               "quality": {"sm_clock_mean_mhz": 2400, "power_mean_w": 550, "power_max_w": 590}}
        m = {"check_results": [], "timing_results": [tim], "check_exit": 0, "timing_exit": 0,
             "check_stderr": "", "timing_stderr": "", "inspect": None, "sass_text": "",
             "environment": {"peak_dram_gbps": 1800.0, "peak_fp32_gflops": 126000.0,
                             "peak_fp32_gflops_at_mhz": 2820}}
        obs = pgaio.observe(cfg, m, [])
        self.assertEqual(obs["verdict"], "ok")
        lo, hi = obs["score_ci"]
        self.assertTrue(lo < obs["score"] < hi)
        p = obs["sizes"][0]
        self.assertAlmostEqual(p["fraction_at_clock"], p["fraction"] * 2820 / 2400)
        ref = {"sizes": [{"params": {"M": 4096}, "op_ms": 2.0}]}
        text = pgaio.render(obs, 3, cfg, None, None, {"cublas": ref})
        self.assertIn("95% CI", text)
        self.assertIn("of cublas", text)
        self.assertIn("at the measured 2.40 GHz", text)
        self.assertIn("550 W", text)


class TestPlateauRule2(unittest.TestCase):
    def a(self, i, score, ci=None, dsl="cuda", verdict="ok"):
        return {"iteration": i, "dsl": dsl, "score": score, "verdict": verdict, "score_ci": ci}

    def test_old_runs_ignore_the_ci(self):
        cfg = {"kind": "problem", "dsls": ["cuda"]}
        atts = [self.a(0, 0.40, [0.39, 0.41]), self.a(1, 0.406, [0.396, 0.416])]
        self.assertEqual(pgaio.plateau(cfg, atts)[0], 0)   # +0.6 points is a gain under rule 1

    def test_rule_2_needs_the_intervals_apart(self):
        cfg = {"kind": "problem", "dsls": ["cuda"], "plateau_rule": 2}
        atts = [self.a(0, 0.40, [0.39, 0.41]), self.a(1, 0.406, [0.396, 0.416])]
        self.assertEqual(pgaio.plateau(cfg, atts)[0], 1)
        atts.append(self.a(2, 0.43, [0.425, 0.435]))      # clears 0.406 + 0.01 + 0.005
        self.assertEqual(pgaio.plateau(cfg, atts)[0], 0)

    def test_rule_2_without_ci_falls_back_to_the_gain(self):
        cfg = {"kind": "problem", "dsls": ["cuda"], "plateau_rule": 2}
        self.assertEqual(pgaio.plateau(cfg, [self.a(0, 0.40), self.a(1, 0.406)])[0], 0)
        self.assertEqual(pgaio.plateau(cfg, [self.a(0, 0.40), self.a(1, 0.404)])[0], 1)

    def test_ab_result_counts_as_a_gain(self):
        cfg = {"kind": "problem", "dsls": ["cuda"], "plateau_rule": 2}
        atts = [self.a(0, 0.40, [0.39, 0.41]), self.a(1, 0.406, [0.396, 0.416])]
        ab = [{"a": 0, "b": 1, "overall": {"faster": "b"}}]
        self.assertEqual(pgaio.plateau(cfg, atts, ab)[0], 0)
        self.assertEqual(pgaio.plateau(cfg, atts, [{"a": 1, "b": 0, "overall": {"faster": "a"}}])[0], 0)
        self.assertEqual(pgaio.plateau(cfg, atts, [{"a": 0, "b": 1, "overall": {"faster": None}}])[0], 1)
        # rule 1 runs do not read A/B results
        self.assertEqual(pgaio.plateau({**cfg, "plateau_rule": 1}, [self.a(0, 0.40), self.a(1, 0.401)], ab)[0], 1)

    def test_skipped_dsl_counts_as_tried(self):
        cfg = {"kind": "problem", "dsls": ["cuda", "cutile", "mixed"]}
        atts = [self.a(0, 0.4)] + [self.a(i, 0.4) for i in range(1, 12)]
        since, short, flat = pgaio.plateau(cfg, atts)
        self.assertEqual(short, ["cutile"])
        self.assertFalse(flat)
        cfg["skipped_dsls"] = [{"dsl": "cutile", "reason": "no FFMA from ct.mma"}]
        since, short, flat = pgaio.plateau(cfg, atts)
        self.assertEqual(short, [])
        self.assertTrue(flat)

    def test_skip_entry(self):
        cfg = {"kind": "problem", "dsls": ["cuda", "warp"]}
        e = pgaio.skip_entry(cfg, [self.a(0, 0.1), self.a(4, 0.2)], "warp", "tile API caps at 32%", "iters 14, 20")
        self.assertEqual((e["dsl"], e["after_iteration"], e["evidence"]), ("warp", 4, "iters 14, 20"))
        for bad in [(cfg, "triton", "x"), (cfg, "warp", " "), ({**cfg, "kind": "slot"}, "warp", "x")]:
            with self.assertRaises(ValueError):
                pgaio.skip_entry(bad[0], [], bad[1], bad[2])


class TestAB(unittest.TestCase):
    def test_paired_ratio_resolves_a_consistent_gain(self):
        a = [2.00, 2.05, 1.98, 2.10, 2.02, 2.01]
        b = [x * 0.98 * (1 + 0.001 * (-1) ** i) for i, x in enumerate(a)]
        res = pgaio.paired_ratio(a, b)
        self.assertAlmostEqual(res["ratio"], 0.98, places=3)
        self.assertLess(res["ci"][1], 1)
        faster, text = pgaio.ab_verdict(res)
        self.assertEqual(faster, "b")
        self.assertIn("B faster by 2.0%", text)

    def test_noise_is_no_difference(self):
        a = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0]
        b = [2.02, 2.06, 1.93, 2.03, 1.97, 1.99]
        faster, text = pgaio.ab_verdict(pgaio.paired_ratio(a, b))
        self.assertIsNone(faster)
        self.assertIn("no difference resolved", text)
        self.assertEqual(pgaio.ab_verdict(pgaio.paired_ratio(b, [x * 1.05 for x in b]))[0], "a")
        self.assertIsNone(pgaio.paired_ratio([1.0], [0.9])["ci"])

    def test_summary_and_order(self):
        shapes = [{"M": 1}, {"M": 2}]
        rounds = [{"a": {"M=1": 1.0 + 0.01 * i, "M=2": 4.0}, "b": {"M=1": 0.9 + 0.01 * i, "M=2": 4.0 + 0.001 * i}}
                  for i in range(6)]
        s = pgaio.ab_summary(rounds, shapes)
        self.assertEqual(s["shapes"][0]["faster"], "b")
        self.assertEqual(s["overall"]["faster"], "b")
        self.assertEqual(pgaio.ab_round_order(4), ["ab", "ba", "ab", "ba"])

    def cfg(self, **over):
        c = {"kind": "problem", "kernel": "matmul_naive", "slot_source": "kernels/matmul/fp32/naive.cu",
             "runs": 100, "profile": True, "train_shapes": [{"M": 64}, {"M": 128}], "check_shapes": [],
             "arena_cmd": "build/arena"}
        c.update(over)
        return c

    def attempts(self):
        return [{"iteration": 0, "verdict": "ok", "candidate": "pgaio/runs/r/candidates/000_baseline"},
                {"iteration": 1, "verdict": "ok", "candidate": "pgaio/runs/r/candidates/001_x/v1.json"},
                {"iteration": 2, "verdict": "incorrect", "candidate": "pgaio/runs/r/candidates/002_y/v2.json"}]

    def test_targets(self):
        cfg, atts = self.cfg(), self.attempts()
        self.assertEqual(pgaio.ab_target(cfg, atts, 1), ["--manifest", "pgaio/runs/r/candidates/001_x/v1.json"])
        self.assertEqual(pgaio.ab_target(cfg, atts, 0)[:2], ["--run", "matmul_naive"])
        with self.assertRaises(ValueError):
            pgaio.ab_target(cfg, atts, 2)       # failed its checks
        with self.assertRaises(ValueError):
            pgaio.ab_target(cfg, atts, 9)       # never evaluated

    def test_commands_alternate_and_skip_profiling(self):
        cfg = self.cfg(scratch_policy="declared", lock_clocks="auto", counters="full")
        plan = pgaio.ab_commands(cfg, ["--manifest", "A"], ["--manifest", "B"], cfg["train_shapes"], 30, 3)
        self.assertEqual([[s for s, _ in r] for r in plan], [["a", "b"], ["b", "a"], ["a", "b"]])
        argv = plan[0][0][1]
        self.assertEqual(argv[:2], ["--manifest", "A"])
        self.assertIn("--l2", argv)
        self.assertEqual(argv[argv.index("--runs") + 1], "30")
        self.assertNotIn("--profile", argv)
        self.assertNotIn("--counters", argv)
        self.assertEqual(argv[argv.index("--scratch-policy") + 1], "declared")
        self.assertEqual(argv[argv.index("--lock-clocks") + 1], "auto")
        self.assertEqual(argv.count("--shape"), 2)

    def test_parse_shape_spec(self):
        self.assertEqual(pgaio.parse_shape_spec("K=63,M=129"), {"K": 63, "M": 129})
        with self.assertRaises(ValueError):
            pgaio.parse_shape_spec("K=")

    def test_dry_run_prints_commands(self):
        import contextlib
        import io
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({**self.cfg(), "name": "r"}))
            (run / "attempts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.attempts()))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                pgaio.main(["ab", str(run), "0", "1", "--rounds", "3", "--shape", "M=64", "--dry-run"])
            lines = out.getvalue().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertTrue(lines[0].startswith("round 1 A: build/arena --run matmul_naive"))
            self.assertTrue(lines[2].startswith("round 2 B: build/arena --manifest"))
            self.assertIn("--shape M=64 --l2 cold --runs 30", lines[1])
            self.assertFalse((run / "ab").exists())
            with self.assertRaises(SystemExit):
                pgaio.main(["ab", str(run), "1", "2", "--dry-run"])
            with self.assertRaises(SystemExit):
                pgaio.main(["ab", str(run), "0", "1", "--rounds", "2", "--dry-run"])


class TestRunFlags(unittest.TestCase):
    def test_old_runs_pass_nothing_new(self):
        old = {"runs": 100, "profile": True}
        self.assertEqual(pgaio.run_flags(old, True), [])
        self.assertEqual(pgaio.timing_args(old, ["--run", "k"], [{"n": 4}]),
                         ["--run", "k", "--shape", "n=4", "--l2", "cold", "--runs", "100", "--profile"])

    def test_new_settings(self):
        cfg = {"runs": 10, "profile": True, "scratch_policy": "declared", "counters": "full", "lock_clocks": "2400"}
        self.assertEqual(pgaio.run_flags(cfg, False), ["--scratch-policy", "declared"])
        args = pgaio.timing_args(cfg, [], [])
        for flag, v in (("--scratch-policy", "declared"), ("--counters", "full"), ("--lock-clocks", "2400")):
            self.assertEqual(args[args.index(flag) + 1], v)
        # counters only come with --profile
        self.assertNotIn("--counters", pgaio.timing_args({**cfg, "profile": False}, [], []))

    def test_unknown_option(self):
        self.assertEqual(pgaio.unknown_option("Error: unknown option '--plan'\n"), "--plan")
        self.assertIsNone(pgaio.unknown_option("Error: something else"))


class TestTensorGuard(unittest.TestCase):
    def test_classifier(self):
        for op in ["HMMA.16816.F32", "HGMMA.64x128x16.F32", "UTCHMMA", "UTCQMMA.2CTA", "QMMA.16832",
                   "IMMA.8816.S8", "OMMA.SP", "DMMA.884"]:
            self.assertTrue(pgaio.is_tensor_op(op), op)
        for op in ["FFMA", "FFMA2", "LDS.128", "MUFU.EX2", "IMAD.WIDE", "HFMA2.MMA"]:
            self.assertFalse(pgaio.is_tensor_op(op), op)

    def test_tensor_ops_from_opcodes_text_and_classes(self):
        insp = {"sass": {"opcodes": {"HMMA.16816.F32": 8, "HMMA.1688.F32": 2, "FFMA": 100}}}
        self.assertEqual(pgaio.tensor_ops(insp), {"HMMA": 10})
        text = "  /*0040*/  @P0 UTCHMMA gdesc[UR4], tmem[UR6] ;\n  /*0050*/  FFMA R1, R2, R3, R4 ;"
        self.assertEqual(pgaio.tensor_ops({}, text), {"UTCHMMA": 1})
        self.assertEqual(pgaio.tensor_ops({"sass": {"classes": {"tensor": 3}}}), {"tensor": 3})
        self.assertEqual(pgaio.tensor_ops({"sass": {"opcodes": {"FFMA": 9}}}, "FFMA"), {})

    def test_observe_rejects(self):
        cfg = {"category": "matmul", "forbid_tensor_cores": True, "required_checks": ["x"]}
        m = {"check_results": [], "timing_results": [], "check_exit": None, "timing_exit": None,
             "check_stderr": "", "timing_stderr": "", "environment": {}, "inspect": None,
             "sass_text": "", "tensor_ops": {"HMMA": 4}}
        obs = pgaio.observe(cfg, m, [])
        self.assertEqual(obs["verdict"], "rejected_guard")
        self.assertIn("HMMA x4", obs["error"])
        self.assertIsNone(obs["score"])
        # without the setting the same SASS is fine
        self.assertNotEqual(pgaio.observe({**cfg, "forbid_tensor_cores": False}, m, [])["verdict"],
                            "rejected_guard")


class TestPlan(unittest.TestCase):
    DATA = {"ok": False, "plans": [
        {"params": {"K": 63, "M": 129, "N": 257}, "scratch": [{"name": "ws", "bytes": 1024}], "error": "",
         "launches": [{"index": 0, "entry": "sgemm_big", "applies": False, "grid": [1, 1, 1], "block": [256, 1, 1],
                       "shared_mem_bytes": 49664, "args": []},
                      {"index": 1, "entry": "sgemm_generic", "applies": True, "grid": [9, 5, 1],
                       "block": [256, 1, 1], "shared_mem_bytes": 9280, "args": [{"name": "a", "buffer": "a"}]}]},
        {"params": {"K": 4096, "M": 4096, "N": 4096}, "scratch": [], "error": "launch 0: unknown name 'BM2'",
         "launches": []}]}

    def test_render(self):
        lines = pgaio.render_plan(self.DATA)
        text = "\n".join(lines)
        self.assertTrue(lines[0].startswith("K=63 M=129 N=257"))
        self.assertIn("#1 sgemm_generic", text)
        self.assertIn("grid 9x5x1", text)
        self.assertIn("smem 9280", text)
        self.assertNotIn("#0 sgemm_big", text)
        self.assertIn("not applied: 0", text)
        self.assertIn("scratch: ws 1024 B", text)
        self.assertIn("ERROR launch 0: unknown name 'BM2'", text)

    def test_first_error_and_summary(self):
        self.assertEqual(pgaio.plan_error(self.DATA), "K=4096 M=4096 N=4096: launch 0: unknown name 'BM2'")
        ok = {"ok": True, "plans": [self.DATA["plans"][0], self.DATA["plans"][0]]}
        self.assertEqual(pgaio.plan_error(ok), "")
        self.assertEqual(pgaio.plan_summary(ok), "launch plan ok at 2 shapes (#1 at 2)")

    def test_no_launch_applies(self):
        data = {"plans": [{"params": {"n": 1}, "error": "", "launches": [{"index": 0, "applies": False}]}]}
        self.assertIn("no launch applies", "\n".join(pgaio.render_plan(data)))


class TestComputeDiagnosis(unittest.TestCase):
    def obs(self, **counters):
        c = {"available": True, "dram_read_bytes": 3.0e9, "dram_write_bytes": 0.9e9, "occupancy": 0.17}
        c.update(counters)
        return {
            "category": "matmul",
            "timing": {"op_ms": 18.8, "gpu_ms": 18.8, "launch_count": 1},
            "sol": {"bound": "compute", "fraction": 0.464, "achieved": 58462, "roof": 125951,
                    "peak_dram_gbps": 1792.0, "fraction_at_clock": 0.52, "clock_mhz": 2520},
            "counters": c, "resources": {"registers": 227, "local": 0},
            "sass": {"tensor": 0, "local_mem": 0, "mean_global_load_bytes": 15},
            "sass_mix": pgaio.sass_mix({"sass": {"instructions": 2345, "opcodes": {
                "FFMA": 1280, "LDS.128": 54, "LDS": 153, "BAR.SYNC": 6}}}),
            "ideal_bytes": 805e6,
        }

    def test_leads_with_compute_signals(self):
        d = pgaio.diagnose(self.obs(issue_active=0.55, stalls={"short_scoreboard": 0.23, "mio_throttle": 0.11,
                                                                "barrier": 0.08, "wait": 0.02},
                                    pipes={"fma": 0.61, "alu": 0.12, "lsu": 0.09}))
        self.assertTrue(d[0].startswith("compute-bound: 46.4%"))
        self.assertIn("52.0% at the measured 2.52 GHz", d[0])
        self.assertIn("FFMA is 55%", d[1])
        self.assertIn("6.2 FMAs per shared load", d[1])
        text = "\n".join(d)
        self.assertIn("227 registers, achieved occupancy 17%: low occupancy is normal", text)
        self.assertIn("issue active 55%", text)
        self.assertIn("short_scoreboard 0.23, mio_throttle 0.11, barrier 0.08", text)
        self.assertNotIn("wait 0.02", text)
        self.assertIn("pipe utilization: fma 61%", text)
        self.assertIn("no tensor core", text)
        dram = [x for x in d if x.startswith("DRAM moved")][0]
        self.assertIn("but the kernel is compute-bound", dram)
        self.assertIn("2.18 of the 18.80 ms", dram)
        self.assertNotIn("uncoalesced", text)
        self.assertGreater(d.index(dram), 2)

    def test_basic_counters_and_forbidden_tensor_cores(self):
        o = self.obs()
        o["tensor_forbidden"] = True
        text = "\n".join(pgaio.diagnose(o))
        self.assertNotIn("of cycles", text)
        self.assertNotIn("top stalls", text)
        self.assertIn("tensor cores are forbidden", text)

    def test_largest_shape_decides(self):
        o = self.obs()
        o["sol"] = {"bound": "unknown"}
        o["sizes"] = [{"bound": "latency", "flops": 1, "ideal_bytes": 1},
                      {"bound": "compute", "flops": 1e12, "ideal_bytes": 1e8}]
        self.assertTrue(pgaio.compute_bound(o))
        o["sizes"].reverse()
        self.assertTrue(pgaio.compute_bound(o))
        o["sizes"][0]["bound"] = "memory"
        self.assertFalse(pgaio.compute_bound(o))

    def test_sass_mix(self):
        m = pgaio.sass_mix({"sass": {"instructions": 10, "opcodes": {"FFMA2": 2, "FFMA": 4, "LDS.128": 2, "EXIT": 2}}})
        self.assertEqual((m["ffma"], m["fma_ops"], m["lds"], m["lds_128"]), (6, 8, 2, 2))
        self.assertAlmostEqual(m["ffma_per_lds"], 4.0)
        self.assertEqual(pgaio.sass_mix(None), {})


class TestClockAndReferences(unittest.TestCase):
    def test_add_clock(self):
        per = [{"bound": "compute", "fraction": 0.47}, {"bound": "latency", "fraction": 0.5}]
        res = [{"quality": {"sm_clock_mean_mhz": 2350, "power_mean_w": 580}},
               {"quality": {"sm_clock_mean_mhz": 2800}}]
        pgaio.add_clock(per, res, {"peak_fp32_gflops_at_mhz": 2617})
        self.assertAlmostEqual(per[0]["fraction_at_clock"], 0.47 * 2617 / 2350)
        self.assertNotIn("fraction_at_clock", per[1])
        self.assertIn("at the measured 2.35 GHz", pgaio.clock_text(per[0]))
        self.assertEqual(pgaio.clock_text(per[1]), "")
        old = [{"bound": "compute", "fraction": 0.47}]
        pgaio.add_clock(old, [{"quality": {"sm_clock_end_mhz": 2820}}], {})
        self.assertEqual(old, [{"bound": "compute", "fraction": 0.47}])

    def test_ref_relative(self):
        ref = {"sizes": [{"params": {"M": 1, "N": 2}, "op_ms": 1.0}, {"params": {"M": 9}, "op_ms": 1.0}]}
        g, per = pgaio.ref_relative(ref, [{"params": {"N": 2, "M": 1}, "op_ms": 2.0},
                                          {"params": {"M": 3}, "op_ms": 1.0}])
        self.assertAlmostEqual(g, 0.5)
        self.assertEqual(per, {"M=1,N=2": 0.5})
        self.assertEqual(pgaio.ref_relative(ref, [])[0], None)
        self.assertEqual(pgaio.ref_text({"cublas": ref}, [{"params": {"M": 9}, "op_ms": 1.25}]), "80.0% of cublas")

    def test_report_counts_failed_evals(self):
        atts = [{"iteration": 0, "verdict": "ok", "score": 0.1},
                {"iteration": 1, "verdict": "compile_error", "score": None},
                {"iteration": 2, "verdict": "runtime_error", "score": None},
                {"iteration": 3, "verdict": "incorrect", "score": None}]
        row = pgaio.summarize_run({"name": "r", "kernel": "k", "dsl": "cuda", "category": "scan"}, atts)
        self.assertEqual(row["failed"], 2)


class TestLearnGuard(unittest.TestCase):
    def test_refuses_a_deleted_tracked_file(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            f = repo / "pgaio" / "learnings.jsonl"
            f.parent.mkdir()
            f.write_text("{}\n")
            subprocess.run(["git", "add", "pgaio/learnings.jsonl"], cwd=repo, check=True)
            self.assertIsNone(pgaio.learnings_refusal(f, repo))          # present
            f.unlink()
            why = pgaio.learnings_refusal(f, repo)
            self.assertIn("git checkout -- pgaio/learnings.jsonl", why)
            self.assertIn("--create", why)
            self.assertIsNone(pgaio.learnings_refusal(f, repo, create=True))
            self.assertIsNone(pgaio.learnings_refusal(repo / "pgaio" / "other.jsonl", repo))  # untracked

            # through the command, with the paths pointed at the temp repo
            saved = pgaio.LEARNINGS, pgaio.REPO, pgaio.load_run, pgaio.load_attempts
            try:
                pgaio.LEARNINGS, pgaio.REPO = f, repo
                pgaio.load_run = lambda run: (repo, {"name": "r", "kernel": "k", "category": "c", "dsl": "cuda"})
                pgaio.load_attempts = lambda run_dir: []
                with self.assertRaises(SystemExit):
                    pgaio.main(["learn", "r", "--insight", "x"])
                self.assertFalse(f.exists())
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    pgaio.main(["learn", "r", "--insight", "x", "--create"])
                self.assertTrue(f.exists())
            finally:
                pgaio.LEARNINGS, pgaio.REPO, pgaio.load_run, pgaio.load_attempts = saved


class TestReviewFixes(unittest.TestCase):
    """Regressions for defects found in review."""

    def test_score_ci_without_any_shape_ci_is_none(self):
        # old timings without all_times_ms: no interval rather than a zero-width one
        self.assertIsNone(pgaio.score_ci([{"roof_ms": 1.0, "fraction": 0.5, "spread": None}]))
        self.assertIsNone(pgaio.add_spread([{"roof_ms": 1.0, "fraction": 0.5}], [{"timing": {}}]))
        self.assertIsNone(pgaio.add_spread([{"roof_ms": 1.0, "fraction": 0.5}],
                                           [{"timing": {"all_times_ms": [1.0, 1.1, 1.2]}}]))

    def test_ab_on_part_of_the_shapes_is_not_a_score_gain(self):
        cfg = {"kind": "problem", "dsls": ["cuda"], "plateau_rule": 2,
               "train_shapes": [{"M": 1}, {"M": 2}]}
        atts = [{"iteration": 0, "dsl": "cuda", "verdict": "ok", "score": 0.40, "score_ci": [0.39, 0.41]},
                {"iteration": 1, "dsl": "cuda", "verdict": "ok", "score": 0.406, "score_ci": [0.396, 0.416]}]
        one = [{"a": 0, "b": 1, "overall": {"faster": "b"}, "shapes": [{"params": {"M": 2}}]}]
        both = [{"a": 0, "b": 1, "overall": {"faster": "b"},
                 "shapes": [{"params": {"M": 2}}, {"params": {"M": 1}}]}]
        self.assertEqual(pgaio.plateau(cfg, atts, one)[0], 1)
        self.assertEqual(pgaio.plateau(cfg, atts, both)[0], 0)

    def test_min_max_opcodes_are_not_tensor_ops(self):
        for op in ["HMNMX2", "HMNMX2.NAN", "IMNMX", "IMNMX.U32", "FMNMX", "FMNMX3", "VIMNMX"]:
            self.assertFalse(pgaio.is_tensor_op(op), op)
        for op in ["BMMA.88128.AND.POPC", "IGMMA.64x8x32.S8", "QGMMA.64x8x32.F32.E4M3", "UTCOMMA"]:
            self.assertTrue(pgaio.is_tensor_op(op), op)
        text = ("  /*0040*/  @!PT HMNMX2 R1, R2, R3, PT ;\n"
                "  /*0050*/  IMNMX.U32 R4, R5, R6, PT ;\n")
        self.assertEqual(pgaio.tensor_ops({}, text), {})

    def test_compare_finds_profiles_and_survives_a_missing_shape(self):
        import contextlib
        import io
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "r"
            (run / "obs").mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"name": "r", "kind": "problem"}))
            (run / "attempts.jsonl").write_text(json.dumps({"iteration": 1, "dsl": "cuda", "verdict": "ok"}) + "\n")
            small, big = {"M": 2}, {"M": 4}
            (run / "obs" / "001.json").write_text(json.dumps(
                {"verdict": "ok", "score": 0.3, "sizes": [
                    {"params": small, "op_ms": 1.0, "fraction": 0.3},
                    {"params": big, "op_ms": 2.0, "fraction": 0.3}]}))
            (run / "references").mkdir()
            (run / "references" / "cublas.json").write_text(json.dumps(
                {"lib": "cublas", "score": 0.5, "sizes": [{"params": small, "op_ms": 0.5, "fraction": 0.5}]}))
            # folder names as pgaio_ncu.py writes them
            for name, s in (("001-M2", {"iteration": 1, "reference": None, "params": small,
                                         "stalls": {"long_scoreboard": 0.40}}),
                            ("ref-cublas-M2", {"iteration": None, "reference": "cublas", "params": small,
                                               "stalls": {"long_scoreboard": 0.10}})):
                (run / "profiles" / name).mkdir(parents=True)
                (run / "profiles" / name / "summary.json").write_text(json.dumps(s))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                pgaio.main(["compare", str(run), "1", "ref:cublas"])
            text = out.getvalue()
            self.assertIn("ncu stalls at M=2", text)
            self.assertRegex(text, r"long_scoreboard\s+40\.0%\s+10\.0%\s+-30\.0")
            self.assertRegex(text, r"op_ms M=4\s+2\.0000\s+-")


class TestInitFallback(unittest.TestCase):
    """cmd_init with arena and the measurement faked: what an older arena
    build that rejects a new flag, or a tensor-core baseline, does to init."""

    def init(self, measure, extra=()):
        import contextlib
        import io
        import json
        import tempfile
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        repo = Path(d.name)
        (repo / "kernels").mkdir()
        (repo / "kernels" / "naive.cu").write_text("// baseline\n")
        insp = {"repo_root": str(repo), "slot_source": str(repo / "kernels" / "naive.cu"),
                "category": "softmax", "dsl": "cuda", "entry_point": "k", "descriptor_file": ""}
        saved = (pgaio.REPO, pgaio.RUNS_DIR, pgaio.LEARNINGS, pgaio.arena, pgaio.measure,
                 pgaio.hash_protected)
        calls = []

        def fake_measure(cfg, label, **kw):
            calls.append(dict(cfg))
            return measure(cfg)
        try:
            pgaio.REPO, pgaio.RUNS_DIR = repo, repo / "pgaio" / "runs"
            pgaio.LEARNINGS = repo / "pgaio" / "learnings.jsonl"
            pgaio.arena = lambda cmd, args, timeout=0: (0, insp, "", "")
            pgaio.measure = fake_measure
            pgaio.hash_protected = lambda: {}
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    pgaio.main(["init", "--category", "softmax", "--name", "t", "--arena", "./arena",
                                "--train", "rows=64,cols=1024", "--dsls", "cuda"] + list(extra))
                exit_msg = None
            except SystemExit as e:
                exit_msg = str(e.code)
            run = pgaio.RUNS_DIR / "t"
            cfg = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else None
            return cfg, calls, exit_msg
        finally:
            (pgaio.REPO, pgaio.RUNS_DIR, pgaio.LEARNINGS, pgaio.arena, pgaio.measure,
             pgaio.hash_protected) = saved

    @staticmethod
    def ok_m():
        res = {"success": True, "verified": True,
               "input": {"params": {"rows": 64, "cols": 1024}, "distribution": "uniform", "seed": 1},
               "throughput": {"gflops": 1.0, "bandwidth_gbps": 900.0},
               "timing": {"op_ms": 0.01, "gpu_ms": 0.01, "launch_count": 1}}
        return {"label": "baseline", "check_results": [res], "check_exit": 0, "check_stderr": "",
                "timing_results": [res], "timing_exit": 0, "timing_stderr": "",
                "environment": {"peak_dram_gbps": 1800.0, "peak_fp32_gflops": 126000.0,
                                "device_name": "gpu"},
                "inspect": None, "sass_text": ""}

    def old_build(self, cfg):
        """An arena build that knows neither --scratch-policy nor --counters."""
        m = self.ok_m()
        if cfg.get("scratch_policy"):
            m.update(check_results=[], timing_results=[], check_exit=1,
                     check_stderr="Error: unknown option '--scratch-policy'\n")
        elif cfg.get("counters") and cfg.get("profile"):
            m.update(timing_results=[], timing_exit=1,
                     timing_stderr="Error: unknown option '--counters'\n")
        return m

    def test_defaulted_flags_are_dropped(self):
        cfg, calls, msg = self.init(self.old_build)
        self.assertIsNone(msg)
        self.assertNotIn("scratch_policy", cfg)
        self.assertNotIn("counters", cfg)
        self.assertTrue(cfg["profile"])
        self.assertEqual(len(calls), 3)

    def test_an_asked_flag_stops_the_run(self):
        cfg, calls, msg = self.init(self.old_build, ["--scratch-policy", "declared"])
        self.assertIsNone(cfg)
        self.assertIn("does not know --scratch-policy", msg)

    def test_tensor_core_baseline_says_why(self):
        def m(cfg):
            return {**self.ok_m(), "check_results": [], "timing_results": [], "check_exit": None,
                    "tensor_ops": {"HMMA": 4}}
        cfg, calls, msg = self.init(m, ["--forbid-tensor-cores"])
        self.assertIsNone(cfg)
        self.assertIn("HMMA x4", msg)
        self.assertEqual(len(calls), 1)


class TestArenaTimeout(unittest.TestCase):
    """arena() on timeout: SIGTERM first (arena resets locked clocks in its
    handler), SIGKILL only after a grace period."""

    def test_sigterm_reaches_the_process(self):
        import shlex
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "got_sigterm"
            code = ("import signal, sys, time\n"
                    "def h(*_):\n"
                    f"    open({str(marker)!r}, 'w').write('x')\n"
                    "    sys.exit(3)\n"
                    "signal.signal(signal.SIGTERM, h)\n"
                    "print('ready', flush=True)\n"
                    "time.sleep(60)\n")
            rc, data, out, err = pgaio.arena(shlex.join([sys.executable, "-c", code]), [], timeout=1)
            self.assertTrue(marker.exists())
            self.assertEqual(rc, 3)
            self.assertIn("ready", out)
            self.assertIn("timed out after 1 s", err)
            self.assertIsNone(data)

    def test_kill_after_the_grace_period(self):
        code = ("import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(60)\n")
        rc, out, err = pgaio.run_with_timeout([sys.executable, "-c", code], None, 1, grace=1)
        self.assertNotEqual(rc, 0)
        self.assertIn("SIGKILL", err)

    def test_normal_exit_is_untouched(self):
        rc, out, err = pgaio.run_with_timeout([sys.executable, "-c", "print('{}')"], None, 30)
        self.assertEqual((rc, out.strip(), err), (0, "{}", ""))

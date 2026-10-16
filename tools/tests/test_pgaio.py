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
        for p in ["pgaio/runs/x/candidates/001.cu", "README.md", "kernels/reduce_fp32_x.cubin",
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

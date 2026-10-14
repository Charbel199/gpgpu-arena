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
        self.assertEqual(agg["cuda/isolated"]["runs"], 2)
        self.assertEqual(agg["cuda/isolated"]["iters_to_70"], 2)
        self.assertEqual(agg["cuda/isolated"]["iters_to_80_reached"], "1/2")
        self.assertEqual(agg["triton/isolated"]["iters_to_50"], 0)
        self.assertIsNone(agg["triton/isolated"]["iters_to_90"])

    def test_modes_are_aggregated_separately(self):
        cfg = {"name": "a", "kernel": "k", "dsl": "cuda", "category": "reduce"}
        rows = [pgaio.summarize_run({**cfg, "mode": "isolated"}, self.attempts([0.1])),
                pgaio.summarize_run({**cfg, "name": "b", "mode": "cross-dsl"}, self.attempts([0.9]))]
        agg = pgaio.aggregate(rows)
        self.assertEqual(set(agg), {"cuda/isolated", "cuda/cross-dsl"})

    def test_docs_access_is_its_own_group(self):
        cfg = {"name": "a", "kernel": "k", "dsl": "triton", "category": "reduce", "mode": "isolated"}
        rows = [pgaio.summarize_run(cfg, self.attempts([0.5])),
                pgaio.summarize_run({**cfg, "name": "b", "docs": True}, self.attempts([0.6]))]
        self.assertFalse(rows[0]["docs"])
        self.assertEqual(set(pgaio.aggregate(rows)), {"triton/isolated", "triton/isolated+docs"})


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
        for text in ("128-bit", "local memory", "occupancy", "global atomic"):
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

# GPGPU Arena, notes for coding agents

Runs the same GPU kernel written in CUDA C++, Triton, cuTile or Warp under one measurement harness. C++17 host code in `src/` and `include/`, kernels in `kernels/<category>/<input dtype>/`, PGAIO tooling in `tools/` and `pgaio/`.

Task-specific instructions live in `skills/` (also reachable as `.agents/skills` and `.claude/skills`):

| Skill | Use it to |
| --- | --- |
| `skills/arena-cli` | run kernels headless and read the JSON results |
| `skills/arena-inspect` | read SASS, PTX, registers, counters; try a source in an existing slot |
| `skills/arena-add-kernel` | add or promote a kernel, in any of the four DSLs |
| `skills/arena-optimize` | run a PGAIO optimization loop with `tools/pgaio.py` |

Build and test:

```bash
cmake -S . -B build -DBUILD_GUI=OFF && cmake --build build -j
ctest --test-dir build --output-on-failure          # C++ unit tests, no GPU needed
python3 -m unittest discover -s tools/tests         # PGAIO unit tests
docker compose run --rm -T cli --list               # full environment with all DSLs
```

Conventions: comments explain why, not what. Measurement code keeps the three timing tiers (per invocation, per process, per build) separate. Do not commit `build*/`, cubins, or `pgaio/runs/`.

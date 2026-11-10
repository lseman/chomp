"""Known KKT optima with interacting equalities, inequalities, and bounds."""
from bench_mixed_qp import run

for n in (3, 6, 10):
    for seed in range(5):
        for mode in ("ip", "sqp"):
            result = run(n, seed, mode)
            assert result["passed"], result

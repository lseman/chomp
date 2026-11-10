"""Projected CG must not retain a full-space warm start across active sets."""

from bench_nlp_constraints import run


for n in (2, 10, 50):
    for equality in (False, True):
        for feasible_start in (False, True):
            result = run(n, "sqp", equality, feasible_start, repeats=1, max_iter=200)
            assert result["passed"], result

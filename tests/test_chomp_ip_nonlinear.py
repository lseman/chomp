"""IP must solve nonlinear slack equations before driving the barrier to zero."""

from bench_nlp_constraints import run


for n in (2, 10, 50):
    for equality in (False, True):
        for feasible_start in (False, True):
            result = run(n, "ip", equality, feasible_start, repeats=1, max_iter=100)
            assert result["passed"], result

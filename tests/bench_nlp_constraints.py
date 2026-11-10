"""Nonlinear equality/inequality projection problems with analytic KKT checks.

Run with PYTHONPATH=build python tests/bench_nlp_constraints.py.
The solution is the projection of a target outside the unit ball onto its sphere.
"""

import argparse
import json
import time
from statistics import median

import numpy as np
from chomp import ChompConfig, chomp


def run(n, mode, equality, feasible_start, repeats, max_iter):
    target = np.linspace(2.0, 1.0, n)
    optimum = target / np.linalg.norm(target)

    def objective(x):
        return sum((x[i] - target[i]) ** 2 for i in range(n))

    def constraint(x):
        return sum(x[i] ** 2 for i in range(n)) - 1.0

    x0 = np.full(n, 0.2 / np.sqrt(n) if feasible_start else 2.0)
    cfg = ChompConfig()
    cfg.mode = mode
    cfg.verbose = False
    cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
    times, build_times, solve_times, errors, residuals, iterations, statuses = [], [], [], [], [], [], []
    exception = None
    for _ in range(repeats):
        try:
            start = time.perf_counter()
            with chomp(objective, [] if equality else [constraint],
                       [constraint] if equality else [], None, None, x0, cfg) as solver:
                built = time.perf_counter()
                x = np.asarray(solver.solve(max_iter=max_iter, verbose=False)).copy()
                solved = time.perf_counter()
                if hasattr(solver, "stats"):
                    iterations.append(solver.stats["iterations"])
                    statuses.append(solver.stats["status"])
            times.append(time.perf_counter() - start)
            build_times.append(built - start)
            solve_times.append(solved - built)
            gradient = 2 * (x - target)
            jacobian = 2 * x
            multiplier = -np.dot(jacobian, gradient) / np.dot(jacobian, jacobian)
            c = float(constraint(x))
            stationarity = float(np.max(np.abs(gradient + multiplier * jacobian)))
            feasibility = abs(c) if equality else max(0.0, c)
            complementarity = 0.0 if equality else abs(multiplier * c)
            dual_feasibility = 0.0 if equality else max(0.0, -multiplier)
            residuals.append(max(stationarity, feasibility, complementarity, dual_feasibility))
            errors.append(float(np.max(np.abs(x - optimum))))
        except Exception as exc:
            exception = str(exc)
            break
    solution_valid = (exception is None and all(np.isfinite(residuals))
              and max(residuals, default=float("inf")) < 1e-6
              and max(errors, default=float("inf")) < 1e-5)
    passed = solution_valid and all(status == "converged" for status in statuses)
    return dict(n=n, mode=mode, equality=equality, feasible_start=feasible_start,
                passed=bool(passed), solution_valid=bool(solution_valid),
                median_seconds=median(times) if times else None,
                median_build_seconds=median(build_times) if build_times else None,
                median_solve_seconds=median(solve_times) if solve_times else None,
                max_kkt_residual=max(residuals) if residuals else None,
                max_solution_error=max(errors) if errors else None,
                max_iterations=max(iterations) if iterations else None,
                statuses=statuses, exception=exception)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iter < 1 or any(n < 2 for n in args.sizes):
        parser.error("sizes must be at least 2; repeats and max-iter must be positive")
    failed = False
    for n in args.sizes:
        for mode in ("ip", "sqp"):
            for equality in (False, True):
                for feasible_start in (True, False):
                    result = run(n, mode, equality, feasible_start, args.repeats, args.max_iter)
                    print(json.dumps(result), flush=True)
                    failed |= not result["passed"]
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()

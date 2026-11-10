"""Compare CHOMP with SciPy SLSQP and trust-constr on analytic projections.

All solvers receive exact derivatives. Every result is checked against the
known optimizer and original KKT equations. Construction is separated from
solve time for CHOMP; SciPy's solve time includes its minimize setup.
"""

import argparse
import json
import time
from statistics import median

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from scipy.sparse import csr_matrix, eye

from bench_nlp_constraints import run


def scipy_run(n, method, equality, feasible_start, repeats, max_iter):
    target = np.linspace(2.0, 1.0, n)
    optimum = target / np.linalg.norm(target)
    x0 = np.full(n, 0.2 / np.sqrt(n) if feasible_start else 2.0)
    identity = eye(n, format="csr")
    objective = lambda x: float(np.dot(x - target, x - target))
    gradient = lambda x: 2 * (x - target)
    circle = lambda x: float(np.dot(x, x) - 1)
    if method == "SLSQP":
        sign = 1.0 if equality else -1.0
        constraint = {"type": "eq" if equality else "ineq",
                      "fun": lambda x: sign * circle(x), "jac": lambda x: sign * 2 * x}
        kwargs = dict(constraints=[constraint], options=dict(maxiter=max_iter, ftol=1e-12))
    else:
        constraint = NonlinearConstraint(
            circle, 0.0 if equality else -np.inf, 0.0,
            jac=lambda x: csr_matrix(2 * x.reshape(1, -1)),
            hess=lambda x, v: 2 * v[0] * identity)
        kwargs = dict(constraints=[constraint], hess=lambda x: 2 * identity,
                      options=dict(maxiter=max_iter, gtol=1e-9, xtol=1e-12, barrier_tol=1e-7,
                                   initial_barrier_parameter=1e-7, initial_barrier_tolerance=1e-7))
    times, errors, residuals, iterations, success, messages = [], [], [], [], [], []
    exception = None
    for _ in range(repeats):
        try:
            start = time.perf_counter()
            result = minimize(objective, x0.copy(), jac=gradient, method=method, **kwargs)
            times.append(time.perf_counter() - start)
            x = result.x
            g, j = gradient(x), 2 * x
            multiplier = -np.dot(j, g) / np.dot(j, j)
            c = circle(x)
            residuals.append(float(max(np.max(np.abs(g + multiplier * j)),
                                       abs(c) if equality else max(0.0, c),
                                       0.0 if equality else abs(multiplier * c),
                                       0.0 if equality else max(0.0, -multiplier))))
            errors.append(float(np.max(np.abs(x - optimum))))
            iterations.append(result.nit)
            success.append(bool(result.success))
            messages.append(str(result.message))
        except Exception as exc:
            exception = str(exc)
            break
    solution_valid = (exception is None and all(np.isfinite(residuals))
              and max(residuals, default=float("inf")) < 1e-6
              and max(errors, default=float("inf")) < 1e-5)
    return dict(n=n, mode=method, equality=equality, feasible_start=feasible_start,
                passed=bool(solution_valid and all(success)), solution_valid=bool(solution_valid),
                median_seconds=median(times) if times else None,
                median_build_seconds=None, median_solve_seconds=median(times) if times else None,
                max_kkt_residual=max(residuals) if residuals else None,
                max_solution_error=max(errors) if errors else None,
                max_iterations=max(iterations) if iterations else None,
                statuses=success, messages=messages, options=kwargs["options"], exception=exception)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 10, 50, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--solvers", nargs="+", choices=["ip", "sqp", "SLSQP", "trust-constr"],
                        default=["ip", "sqp", "SLSQP", "trust-constr"])
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iter < 1 or any(n < 2 for n in args.sizes):
        parser.error("sizes must be at least 2; repeats and max-iter must be positive")
    failed = False
    for n in args.sizes:
        for equality in (False, True):
            for feasible_start in (True, False):
                for method in args.solvers:
                    solve = run if method in ("ip", "sqp") else scipy_run
                    result = solve(n, method, equality, feasible_start, args.repeats, args.max_iter)
                    print(json.dumps(result), flush=True)
                    failed |= not result["passed"]
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()

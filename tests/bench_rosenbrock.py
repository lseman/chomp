"""Time to a verified local minimum on growing Rosenbrock chains.

Unlike a global-target gate, this distinguishes an accurate local minimum
from an iteration-limit return. SciPy references receive analytic derivatives.
"""

import argparse
import json
import time
from statistics import median

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess_prod

from chomp import ChompConfig, chomp


def check_solution(x):
    gradient = float(np.max(np.abs(rosen_der(x))))
    diagonal = np.r_[2 + 1200 * x[:-1] ** 2 - 400 * x[1:], 0.0]
    diagonal[1:] += 200
    curvature = float(eigvalsh_tridiagonal(diagonal, -400 * x[:-1],
                                         select="i", select_range=(0, 0))[0])
    objective = float(rosen(x))
    valid = bool(np.all(np.isfinite(x)) and gradient < 1e-5 and curvature >= -1e-6)
    return valid, objective, gradient, curvature


def run(n, method, repeats=3, max_iter=2000):
    x0 = np.ones(n)
    x0[::2] = -1.2

    def objective(x):
        return sum((1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
                   for i in range(n - 1))

    times, build_times, iterations, checks, statuses = [], [], [], [], []
    options = {"maxiter": max_iter, "gtol": 1e-6}
    if method == "L-BFGS-B":
        options.update(ftol=1e-15, maxls=50)
    exception = None
    for _ in range(repeats):
        try:
            if method in ("ip", "sqp"):
                cfg = ChompConfig()
                cfg.mode = method
                cfg.verbose = False
                cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-6
                start = time.perf_counter()
                with chomp(objective, [], [], None, None, x0.copy(), cfg) as solver:
                    built = time.perf_counter()
                    x = np.asarray(solver.solve(max_iter=max_iter, verbose=False)).copy()
                    solved = time.perf_counter()
                    stats = solver.stats
                build_times.append(built - start)
                times.append(solved - built)
                statuses.append(stats["status"])
                iterations.append(stats["iterations"])
            else:
                start = time.perf_counter()
                kwargs = {"hessp": rosen_hess_prod} if method == "trust-krylov" else {}
                result = minimize(rosen, x0.copy(), jac=rosen_der, method=method,
                                  options=options, **kwargs)
                times.append(time.perf_counter() - start)
                x = result.x
                statuses.append(str(result.message))
                iterations.append(result.nit)
            checks.append(check_solution(x))
        except Exception as exc:
            exception = str(exc)
            break
    solution_valid = exception is None and bool(checks) and all(c[0] for c in checks)
    return dict(n=n, method=method, solution_valid=solution_valid,
                median_solve_seconds=median(times) if times else None,
                median_build_seconds=median(build_times) if build_times else None,
                max_iterations=max(iterations) if iterations else None,
                max_objective=max(c[1] for c in checks) if checks else None,
                max_stationarity=max(c[2] for c in checks) if checks else None,
                min_curvature=min(c[3] for c in checks) if checks else None,
                statuses=statuses, options=options, exception=exception)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--solvers", nargs="+", choices=["ip", "sqp", "trust-krylov", "L-BFGS-B"],
                        default=["ip", "sqp", "trust-krylov", "L-BFGS-B"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iter < 1 or any(n < 2 for n in args.sizes):
        parser.error("sizes must be at least 2; repeats and max-iter must be positive")
    failed = False
    for n in args.sizes:
        for method in args.solvers:
            result = run(n, method, args.repeats, args.max_iter)
            print(json.dumps(result), flush=True)
            failed |= not result["solution_valid"]
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()

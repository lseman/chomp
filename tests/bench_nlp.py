"""End-to-end NLP baseline: PYTHONPATH=build python tests/bench_nlp.py.

Reports median construction + solve time and independently checks solutions.
Failed solves remain in the output and cause a nonzero exit status.
"""

import argparse
import json
import time
from statistics import median

import numpy as np
from chomp import ChompConfig, chomp


def cases(n):
    def sphere(x):
        return sum((x[i] - 1.0) ** 2 for i in range(n))

    def equality(x):
        return sum(x[i] for i in range(n)) - n / 2

    def rosenbrock(x):
        return sum((1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
                   for i in range(n - 1))

    yield "sphere", sphere, [], None, None, np.zeros(n), np.ones(n)
    yield "equality", sphere, [equality], None, None, np.zeros(n), np.full(n, 0.5)
    yield "bounds", sphere, [], np.full(n, -2.0), np.full(n, 0.5), np.zeros(n), np.full(n, 0.5)
    x0 = np.ones(n)
    x0[::2] = -1.2
    yield "rosenbrock", rosenbrock, [], None, None, x0, np.ones(n)


def stationarity(name, x, lb, ub):
    """Independent analytic first/second-order checks for these test models."""
    complementarity = 0.0
    if name == "rosenbrock":
        gradient = np.zeros_like(x)
        gradient[:-1] += 2 * (x[:-1] - 1) - 400 * x[:-1] * (x[1:] - x[:-1] ** 2)
        gradient[1:] += 200 * (x[1:] - x[:-1] ** 2)
        hessian = np.diag(np.r_[2 + 1200 * x[:-1] ** 2 - 400 * x[1:], 0.0])
        hessian[np.arange(1, len(x)), np.arange(1, len(x))] += 200
        i = np.arange(len(x) - 1)
        hessian[i, i + 1] = hessian[i + 1, i] = -400 * x[:-1]
        minimum_curvature = float(np.linalg.eigvalsh(hessian)[0])
    else:
        gradient = 2 * (x - 1)
        minimum_curvature = 2.0
        if name == "equality":
            gradient -= np.mean(gradient)
        if lb is not None:
            dual = np.maximum(gradient, 0.0)
            complementarity = max(complementarity, float(np.max(np.abs(dual * (x - lb)))))
            gradient -= dual
        if ub is not None:
            dual = np.maximum(-gradient, 0.0)
            complementarity = max(complementarity, float(np.max(np.abs(dual * (ub - x)))))
            gradient += dual
    return float(np.max(np.abs(gradient))), minimum_curvature, complementarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 10, 50])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--cases", nargs="+", choices=["sphere", "equality", "bounds", "rosenbrock"],
                        default=["sphere", "equality", "bounds", "rosenbrock"])
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 2 for n in args.sizes):
        parser.error("repeats must be positive and sizes must be at least 2")
    failed = False
    for n in args.sizes:
        for name, objective, eq, lb, ub, x0, optimum in cases(n):
            if name not in args.cases:
                continue
            for mode in ("ip", "sqp"):
                cfg = ChompConfig()
                cfg.mode = mode
                cfg.verbose = False
                cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-6
                times, build_times, solve_times, errors, violations = [], [], [], [], []
                residuals, curvatures, complementarities = [], [], []
                exception = None
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    try:
                        with chomp(objective, [], eq, lb, ub, x0.copy(), cfg) as solver:
                            built = time.perf_counter()
                            x = np.asarray(solver.solve(max_iter=args.max_iter, verbose=False)).copy()
                            solved = time.perf_counter()
                        times.append(time.perf_counter() - start)
                        build_times.append(built - start)
                        solve_times.append(solved - built)
                        errors.append(float(np.max(np.abs(x - optimum))))
                        residual, curvature, complementarity = stationarity(name, x, lb, ub)
                        residuals.append(residual)
                        curvatures.append(curvature)
                        complementarities.append(complementarity)
                        violations.append(max([0.0] + [abs(float(c(x))) for c in eq]
                                              + ([] if lb is None else [float(np.max(lb - x))])
                                              + ([] if ub is None else [float(np.max(x - ub))])))
                    except Exception as exc:
                        exception = str(exc)
                        break
                passed = (exception is None and all(np.isfinite(errors))
                          and max(errors, default=float("inf")) < 1e-3
                          and max(violations, default=float("inf")) < 1e-6
                          and max(residuals, default=float("inf")) < 1e-5
                          and max(complementarities, default=float("inf")) < 1e-6
                          and min(curvatures, default=-float("inf")) >= -1e-6)
                failed |= not passed
                print(json.dumps(dict(case=name, n=n, mode=mode, passed=passed,
                                      median_seconds=median(times) if times else None,
                                      median_build_seconds=median(build_times) if build_times else None,
                                      median_solve_seconds=median(solve_times) if solve_times else None,
                                      max_stationarity=max(residuals) if residuals else None,
                                      max_complementarity=max(complementarities) if complementarities else None,
                                      min_curvature=min(curvatures) if curvatures else None,
                                      max_solution_error=max(errors) if errors else None,
                                      max_violation=max(violations) if violations else None,
                                      exception=exception)), flush=True)
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()

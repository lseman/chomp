"""Audit IP/SQP on strictly convex QPs with planted mixed-constraint KKT points.

Positive inequality/bound duals and H >= I make the planted point the unique
minimum. This oracle needs no external optimizer and checks the returned point
independently of native stopping statistics. Exit nonzero on any missed case.
"""
import argparse
import json
import time

import numpy as np
from chomp import ChompConfig, chomp


def run(n, seed, mode, max_iter=300):
    rng = np.random.default_rng(seed)
    target = rng.uniform(-0.3, 0.3, n)
    factor = rng.normal(size=(n, n))
    hessian = factor.T @ factor + np.eye(n)
    inequality = rng.normal(size=(2, n))
    equality = rng.normal(size=(1, n))
    lower, upper = np.full(n, -2.0), np.full(n, 2.0)
    upper[0], lower[1] = target[0], target[1]
    lam, nu = np.array([0.5, 1.0]), np.array([0.3])
    bound_normal = np.zeros(n)
    bound_normal[0], bound_normal[1] = 0.7, -0.9
    gradient = -inequality.T @ lam - equality.T @ nu - bound_normal

    def objective(x):
        return (sum(gradient[i] * (x[i] - target[i]) for i in range(n))
                + sum(0.5 * hessian[i, j] * (x[i] - target[i]) * (x[j] - target[j])
                      for i in range(n) for j in range(n)))

    ci = [lambda x, k=k: sum(inequality[k, j] * (x[j] - target[j])
                             for j in range(n)) for k in range(2)]
    ce = [lambda x: sum(equality[0, j] * (x[j] - target[j]) for j in range(n))]
    cfg = ChompConfig()
    cfg.mode, cfg.verbose = mode, False
    cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
    started = time.perf_counter()
    with chomp(objective, ci, ce, lower, upper, np.zeros(n), cfg) as solver:
        built = time.perf_counter()
        x = np.asarray(solver.solve(max_iter=max_iter, verbose=False)).copy()
        solved = time.perf_counter()
        stats = solver.stats
    delta = x - target
    stationarity = hessian @ delta + gradient + inequality.T @ lam + equality.T @ nu + bound_normal
    feasibility = max(float(np.max(np.abs(equality @ delta))),
                      float(np.max(np.maximum(inequality @ delta, 0))),
                      float(np.max(np.maximum(lower - x, 0))),
                      float(np.max(np.maximum(x - upper, 0))))
    complementarity = max(float(np.max(np.abs(lam * (inequality @ delta)))),
                          0.7 * abs(upper[0] - x[0]), 0.9 * abs(x[1] - lower[1]))
    error = float(np.max(np.abs(delta)))
    residual = max(float(np.max(np.abs(stationarity))), feasibility, complementarity)
    valid = bool(np.all(np.isfinite(x)) and error < 1e-5 and residual < 1e-5)
    return dict(n=n, seed=seed, mode=mode, solution_valid=valid,
                passed=valid and stats['status'] == 'converged',
                max_solution_error=error, planted_kkt_residual=residual,
                feasibility=feasibility, complementarity=complementarity,
                build_seconds=built-started, solve_seconds=solved-built, stats=stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[3, 6, 10])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--modes', nargs='+', choices=['ip', 'sqp'], default=['ip', 'sqp'])
    parser.add_argument('--max-iter', type=int, default=300)
    args = parser.parse_args()
    if min(args.sizes) < 3 or args.seeds < 1 or args.max_iter < 1:
        parser.error('sizes must be >=3; seeds and max-iter must be positive')
    failed = False
    for n in args.sizes:
        for seed in range(args.seeds):
            for mode in args.modes:
                result = run(n, seed, mode, args.max_iter)
                print(json.dumps(result), flush=True)
                failed |= not result['passed']
    raise SystemExit(int(failed))


if __name__ == '__main__':
    main()

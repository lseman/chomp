import numpy as np
from chomp import ChompConfig, chomp


for blocks in (1, 8, 64):
    n = 2 * blocks
    for side in ("upper", "lower", "fixed"):
        sign = -1.0 if side == "lower" else 1.0

        def objective(x):
            return sum((sign * x[2 * i] + x[2 * i + 1] - 1) ** 2
                       + (sign * x[2 * i] - 2) ** 2 for i in range(blocks))

        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
        initial = np.zeros(n)
        initial[::2] = -0.5 * sign
        if side == "upper":
            upper[::2] = 0.0
        elif side == "lower":
            lower[::2] = 0.0
        else:
            lower[::2] = upper[::2] = 0.0
            initial[::2] = 0.0
        cfg = ChompConfig()
        cfg.mode = "sqp"
        cfg.verbose = False
        cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
        with chomp(objective, [], [], lower, upper, initial, cfg) as solver:
            x = np.asarray(solver.solve(max_iter=100, verbose=False))
            assert solver.stats["status"] == "converged", (blocks, side, solver.stats)
            assert np.all(x >= lower) and np.all(x <= upper), (blocks, side, x)
            assert np.max(np.abs(x[::2])) < 1e-7, (blocks, side, x)
            assert np.max(np.abs(x[1::2] - 1.0)) < 1e-6, (blocks, side, x)
            assert abs(objective(x) - 4 * blocks) < 1e-7, (blocks, side, x)

# Equality feasibility must progress along free coordinates after a bound hit.
# The former pullback/recompute normal step stalled for 200 iterations here.
for sign in (1.0, -1.0):
    for initial in ((-0.5 * sign, 0.0), (0.0, 1.0), (-2.0 * sign, 3.0)):
        lower = np.array([-np.inf, -np.inf])
        upper = np.array([np.inf, np.inf])
        if sign > 0:
            upper[0] = 0.0
        else:
            lower[0] = 0.0
        cfg = ChompConfig()
        cfg.mode = "sqp"
        cfg.verbose = False
        cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-8
        with chomp(lambda x: (sign * x[0] - 2) ** 2 + x[1] ** 2,
                   [], [lambda x: sign * x[0] + x[1] - 1],
                   lower, upper, np.array(initial), cfg) as solver:
            x = np.asarray(solver.solve(max_iter=30, verbose=False))
            assert solver.stats["status"] == "converged", (initial, solver.stats)
            assert np.all(x >= lower) and np.all(x <= upper)
            assert np.max(np.abs(x - [0, 1])) < 1e-8, x
            assert abs(sign * x[0] + x[1] - 1) < 1e-8
            # Independent KKT residual at the known solution: nu=-2, z=6.
            gradient = np.array([2 * sign * (sign * x[0] - 2), 2 * x[1]])
            residual = gradient - 2 * np.array([sign, 1]) + np.array([6 * sign, 0])
            assert np.max(np.abs(residual)) < 1e-7, residual

# A bound must not suppress motion along unrelated equality-tangent axes.
for sign in (1.0, -1.0):
    for initial in ([0.0, 1.0, 0.0], [-0.5 * sign, 0.0, 0.0]):
        lower = np.full(3, -np.inf)
        upper = np.full(3, np.inf)
        if sign > 0:
            upper[0] = 0.0
        else:
            lower[0] = 0.0
        cfg = ChompConfig()
        cfg.mode = "sqp"
        cfg.verbose = False
        cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
        with chomp(lambda x: (sign * x[0] - 2) ** 2 + x[1] ** 2 + (x[2] - 3) ** 2,
                   [], [lambda x: sign * x[0] + x[1] - 1], lower, upper,
                   np.array(initial), cfg) as solver:
            x = np.asarray(solver.solve(max_iter=30, verbose=False))
            assert solver.stats["status"] == "converged", solver.stats
            assert np.max(np.abs(x - [0, 1, 3])) < 1e-7, x
            assert abs(sign * x[0] + x[1] - 1) < 1e-8
            gradient = np.array([2 * sign * (sign * x[0] - 2), 2 * x[1], 2 * (x[2] - 3)])
            residual = gradient - 2 * np.array([sign, 1, 0]) + np.array([6 * sign, 0, 0])
            assert np.max(np.abs(residual)) < 1e-7, residual

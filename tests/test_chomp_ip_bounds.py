"""Restoration must keep the IP iterate inside its variable bounds."""

import numpy as np
from chomp import ChompConfig, chomp


def objective(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2


cfg = ChompConfig()
cfg.mode = "ip"
cfg.verbose = False
cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
with chomp(objective, [], [], np.full(2, -2.0), np.full(2, 0.5),
           np.zeros(2), cfg) as solver:
    # Check every returned iterate: the old restoration step reused the
    # Newton direction's step limit and crossed the upper bound.
    for _ in range(200):
        x = np.asarray(solver.solve(max_iter=1, verbose=False))
        assert np.all(np.isfinite(x)), x
        assert np.all(x >= -2.0) and np.all(x <= 0.5), x


# With many active bounds, the final step must also preserve bound-dual
# positivity. Otherwise clipping negative duals causes repeated excursions
# away from the minimizer even though the primal box remains feasible.
for n in (2, 10, 50):
    def separable(x):
        return sum((x[i] - 1.0) ** 2 for i in range(n))

    with chomp(separable, [], [], np.full(n, -2.0), np.full(n, 0.5),
               np.zeros(n), cfg) as solver:
        x = np.asarray(solver.solve(max_iter=200, verbose=False))
        assert np.max(np.abs(x - 0.5)) < 1e-6, (n, x)

# Boundary/outside starts need interior initialization; interior starts must
# recover from restoration without pinning free coordinates or exploding duals.
for side in ("upper", "lower", "two"):
    sign = -1.0 if side == "lower" else 1.0
    lower = np.array([0.0 if side == "lower" else (-1.0 if side == "two" else -np.inf), -np.inf])
    upper = np.array([np.inf if side == "lower" else 0.0, np.inf])
    starts = (0.0, -1.0, 1.0)
    for start in starts:
        with chomp(lambda x: (sign * x[0] - 2) ** 2 + x[1] ** 2,
                   [], [lambda x: sign * x[0] + x[1] - 1],
                   lower, upper, np.array([start, 1.0]), cfg) as solver:
            x = np.asarray(solver.solve(max_iter=100, verbose=False))
            assert solver.stats["status"] == "converged", (side, start, solver.stats)
            assert np.all(x >= lower) and np.all(x <= upper), x
            assert np.max(np.abs(x - [0.0, 1.0])) < 1e-7, x
            assert abs(sign * x[0] + x[1] - 1.0) < 1e-7
            assert 6 * abs(x[0]) < 1e-7  # independent bound complementarity

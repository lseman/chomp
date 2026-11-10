"""A positive Hessian diagonal does not rule out negative curvature."""

import numpy as np
from chomp import ChompConfig, chomp


def objective(x):
    return (x[0] ** 2 + x[1] ** 2 + 4 * x[0] * x[1]
            + 0.25 * (x[0] ** 2 + x[1] ** 2) ** 2)


# At (0.1, -0.1), the Hessian has positive diagonal but a negative
# eigenvalue. Unmodified Newton steps lead toward the saddle at the origin.
# The two minima are (1, -1) and (-1, 1), with objective -1.
for mode in ("ip", "sqp"):
    cfg = ChompConfig()
    cfg.mode = mode
    cfg.verbose = False
    cfg.tol_stat = 1e-7
    with chomp(objective, [], [], None, None, np.array([0.1, -0.1]), cfg) as solver:
        x = np.asarray(solver.solve(max_iter=200, verbose=False))
        assert np.all(np.isfinite(x)), (mode, x)
        assert abs(objective(x) + 1.0) < 1e-10, (mode, x)
        assert abs(x[0] + x[1]) < 1e-6, (mode, x)

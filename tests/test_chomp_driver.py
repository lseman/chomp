"""Generic driver dispatch and constructor-failure cleanup regressions."""
import numpy as np
from chomp import ChompConfig, chomp, _cleanup_live_solvers


def objective(x):
    return (x[0] - 1.0) ** 2


for bounds, inequalities, equalities, expected in (
    ((None, None), [], [], "sqp"),
    ((None, None), [], [lambda x: x[0] - 1.0], "sqp"),
    ((np.array([-2.0]), np.array([2.0])), [], [], "ip"),
    ((None, None), [lambda x: x[0] - 2.0], [], "ip"),
):
    with chomp(objective, inequalities, equalities, *bounds,
               np.array([0.0]), None) as solver:
        assert solver.stats["mode"] == expected, solver.stats
        result = solver.solve(max_iter=100, verbose=False)
        assert solver.stats["status"] == "converged", solver.stats
        np.testing.assert_allclose(result, [1.0], atol=1e-4)


def invalid_objective(x):
    raise RuntimeError("deliberate construction failure")


for mode, f in (("dfo", objective), ("ip", invalid_objective)):
    cfg = ChompConfig()
    cfg.mode = mode
    try:
        chomp(f, [], [], None, None, np.array([0.0]), cfg)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("invalid solver construction succeeded")
    # A failed constructor must never leave a dangling live-instance entry.
    _cleanup_live_solvers()

# Shutdown cleanup closes registered solvers and remains idempotent.
solver = chomp(objective, [], [], None, None, np.array([0.0]), None)
_cleanup_live_solvers()
_cleanup_live_solvers()
assert solver.solve(verbose=False).size == 0

# Both algorithms share the extracted quasi-Newton implementation.
for mode in ("ip", "sqp"):
    cfg = ChompConfig()
    cfg.mode = mode
    cfg.hessian_mode = "lbfgs"
    with chomp(objective, [], [], None, None, np.array([0.0]), cfg) as solver:
        result = solver.solve(max_iter=100, verbose=False)
        assert solver.stats["status"] == "converged", solver.stats
        np.testing.assert_allclose(result, [1.0], atol=1e-4)

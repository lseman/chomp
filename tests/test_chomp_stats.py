import numpy as np
from chomp import ChompConfig, chomp


def objective(x):
    return (x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2


for mode in ("ip", "sqp"):
    cfg = ChompConfig()
    cfg.mode = mode
    cfg.verbose = False
    cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
    with chomp(objective, [], [], None, None, np.array([-10.0, 10.0]), cfg) as solver:
        assert solver.stats["status"] == "not_started"
        assert solver.stats["iterations"] == 0
        solver.solve(max_iter=1, verbose=False)
        first = solver.stats
        assert first["status"] == "iteration_limit", first
        assert first["iterations"] == 1
        x = np.asarray(solver.solve(max_iter=100, verbose=False))
        result = solver.stats
        assert result["status"] == "converged", result
        assert result["iterations"] > 0
        assert result["stationarity"] <= cfg.tol_stat, result
        assert abs(objective(x) - result["objective"]) < 1e-12
        assert first["status"] == "iteration_limit"  # independent snapshot
    assert solver.stats == result  # diagnostics survive resource cleanup


def constant_objective(x):
    return 0.0 * x[0]


# At the box center, stationarity and barrier-centered complementarity are
# both zero initially. Original-NLP complementarity is still mu_init > 0.
cfg = ChompConfig()
cfg.mode = "ip"
cfg.verbose = False
cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
with chomp(constant_objective, [], [], np.array([-1.0]), np.array([1.0]),
           np.array([0.0]), cfg) as solver:
    solver.solve(max_iter=1, verbose=False)
    assert solver.stats["status"] == "iteration_limit", solver.stats
    assert solver.stats["complementarity"] > cfg.tol_comp, solver.stats
    solver.solve(max_iter=100, verbose=False)
    assert solver.stats["status"] == "converged", solver.stats
    assert solver.stats["complementarity"] <= cfg.tol_comp, solver.stats

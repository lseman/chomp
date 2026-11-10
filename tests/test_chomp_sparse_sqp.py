import numpy as np
from chomp import ChompConfig, chomp


for n in (65, 128):
    target = np.linspace(-1.0, 2.0, n)
    weights = np.geomspace(0.1, 100.0, n)
    for dense in (False, True):
        def objective(x):
            y = [x[i] - target[i] for i in range(n)]
            value = sum(weights[i] * y[i] ** 2 for i in range(n))
            value += 0.05 * sum((y[i + 1] - y[i]) ** 2 for i in range(n - 1))
            return value + (0.1 * sum(y) ** 2 if dense else 0.0)

        for radius in (1e-3, 0.8 * np.linalg.norm(target), 2 * np.linalg.norm(target)):
            cfg = ChompConfig()
            cfg.mode = "sqp"
            cfg.verbose = False
            cfg.delta0 = radius
            cfg.tol_stat = cfg.tol_feas = cfg.tol_comp = 1e-7
            with chomp(objective, [], [], None, None, np.zeros(n), cfg) as solver:
                x = np.asarray(solver.solve(max_iter=100, verbose=False))
                assert solver.stats["status"] == "converged", solver.stats
                assert np.max(np.abs(x - target)) < 1e-6, (n, dense, radius, x)
                assert objective(x) < 1e-12, (n, dense, radius, x)
                # With a sufficiently large radius the first sparse/dense
                # dogleg step must recover the exact Newton minimizer.
                if radius > np.linalg.norm(target):
                    assert solver.stats["iterations"] == 1, solver.stats
            if radius > np.linalg.norm(target):
                with chomp(objective, [], [], np.full(n, -10.0), np.full(n, 10.0),
                           np.zeros(n), cfg) as solver:
                    x = np.asarray(solver.solve(max_iter=1, verbose=False))
                    assert solver.stats["status"] == "converged", solver.stats
                    assert np.max(np.abs(x - target)) < 1e-6

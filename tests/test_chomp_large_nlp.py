from bench_rosenbrock import run


for n in (100, 200, 500):
    for method in ("ip", "sqp"):
        result = run(n, method, repeats=1, max_iter=4 * n + 100)
        assert result["solution_valid"], result
        assert result["statuses"] == ["converged"], result
        assert result["max_objective"] < 4.0, result

import ad
import numpy as np
import pytest


def make_objective(kind, n):
    def objective(x):
        if kind == "linear":
            return 2 * sum(x[i] for i in range(n))
        if kind == "dense":
            return ad.exp(sum(x[i] for i in range(n)) / n)
        if kind == "blocks":
            return sum(ad.exp(0.1 * sum(x[i] for i in range(start, min(start + 20, n))))
                       for start in range(0, n, 20))
        if kind == "branches":
            return sum(ad.max((x[i] - 1) ** 2, (x[i] + 1) ** 2)
                       + ad.relu(x[i]) ** 3 for i in range(n))
        if kind == "quotient":
            return sum(x[i] / (2 + x[i + 1] ** 2) for i in range(n - 1))
        return (sum((x[i] - 1) ** 2 for i in range(n))
                + sum(0.5 * ad.sin(x[i]) * x[i + 1] for i in range(n - 1)))
    return objective


@pytest.mark.parametrize("n", [33, 65, 128])
@pytest.mark.parametrize("kind", ["linear", "banded", "quotient", "branches", "blocks", "dense"])
def test_compressed_lagrangian_hessian_matches_basis_sweeps(n, kind):
    f = make_objective(kind, n)
    inequalities = [lambda x: sum(x[i] ** 3 for i in range(n)) / n]
    equalities = [lambda x: x[0] * x[n - 1] + sum(ad.exp(0.05 * x[i]) for i in range(n))]
    compiled = ad.sym_laghess(f, inequalities, equalities, n)
    rng = np.random.default_rng(24)
    # Reuse one structural plan across different points and initially zero,
    # then nonzero and negative multipliers. Numeric sparsity is insufficient.
    for lam, nu in ((0.0, 0.0), (1.3, -0.2), (0.7, 0.9)):
        x = rng.uniform(-0.3, 0.3, n)
        multipliers = np.array([lam]), np.array([nu])
        sparse = compiled.hess_sparse(x, *multipliers).toarray()
        dense = compiled.hess(x, *multipliers)
        np.testing.assert_allclose(sparse, dense, rtol=1e-11, atol=1e-11)
        v = rng.standard_normal(n)
        np.testing.assert_allclose(sparse @ v, compiled.hvp_numpy(x, *multipliers, v),
                                   rtol=1e-10, atol=1e-10)
        if kind == "banded":
            expected = np.diag(2 + 6 * lam * x / n + nu * 0.0025 * np.exp(0.05 * x))
            expected[np.arange(n - 1), np.arange(n - 1)] -= 0.5 * np.sin(x[:-1]) * x[1:]
            expected[np.arange(n - 1), np.arange(1, n)] = 0.5 * np.cos(x[:-1])
            expected[np.arange(1, n), np.arange(n - 1)] = 0.5 * np.cos(x[:-1])
            expected[0, n - 1] += nu
            expected[n - 1, 0] += nu
            np.testing.assert_allclose(sparse, expected, rtol=1e-11, atol=1e-11)

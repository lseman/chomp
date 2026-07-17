"""Verify sparse-input HessFn.hess() matches the dense-input path."""
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse

sys.path.insert(0, "build")
import ad

try:
    import torch
    import torch.autograd.functional as af
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def make_obj(n):
    """Sum of coupled nonlinear terms — tridiagonal-ish Hessian."""
    def f(z):
        acc = 0.0
        for i in range(n):
            acc = acc + (z[i] - 1.0) ** 2
        for i in range(n - 1):
            acc = acc + 0.5 * ad.sin(z[i]) * z[i + 1] + 0.1 * ad.exp(z[i] * 0.1)
        return acc
    return f


def _sparse_input(x):
    return sparse.csr_matrix(x.reshape(-1, 1))


def test_sparse_dense_match():
    n = 20
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)

    hess_fn = ad.sym_hess(make_obj(n), n, vector_input=True)

    # Dense
    H_dense = hess_fn(x)
    # Sparse (COO: rows, cols, data)
    H_sparse = hess_fn.hess(_sparse_input(x))
    H_from_sparse = np.asarray(H_sparse.todense())

    assert np.allclose(H_dense, H_from_sparse, atol=1e-9), \
        f"Max diff: {np.max(np.abs(H_dense - H_from_sparse))}"

    # Symmetry check on sparse
    assert np.allclose(H_from_sparse, H_from_sparse.T, atol=1e-9), "Sparse Hessian not symmetric"

    nnz = H_sparse.nnz
    print(f"n={n}, nnz={nnz}, density={nnz/(n*n):.4f}")
    print("PASS  sparse_dense_match")


def test_sparse_vs_torch():
    if not HAVE_TORCH:
        print("SKIP  sparse_vs_torch (torch not available)")
        return

    n = 10
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)

    def torch_obj(z):
        acc = ((z - 1.0) ** 2).sum()
        acc = acc + (0.5 * torch.sin(z[:-1]) * z[1:]).sum()
        acc = acc + (0.1 * torch.exp(z[:-1] * 0.1)).sum()
        return acc

    H_torch = af.hessian(torch_obj, torch.tensor(x, dtype=torch.float64))

    hess_fn = ad.sym_hess(make_obj(n), n, vector_input=True)
    H_sparse = hess_fn.hess(_sparse_input(x))
    H_from_sparse = np.asarray(H_sparse.todense())

    assert np.allclose(H_from_sparse, H_torch, rtol=1e-6, atol=1e-7), \
        f"Max diff: {np.max(np.abs(H_from_sparse - H_torch))}"

    print(f"n={n}, nnz={H_sparse.nnz}")
    print("PASS  sparse_vs_torch")


def test_sparse_large():
    """Test scalability: n=100 with mostly zero Hessian."""
    n = 100
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)

    def f(z):
        acc = 0.0
        for i in range(n):
            acc = acc + (z[i] - 1.0) ** 2
        for i in range(n - 1):
            acc = acc + 0.5 * ad.sin(z[i]) * z[i + 1]
        return acc

    hess_fn = ad.sym_hess(f, n, vector_input=True)
    H_sparse = hess_fn.hess(_sparse_input(x))
    H_dense = hess_fn(x)
    H_from_sparse = np.asarray(H_sparse.todense())

    assert np.allclose(H_dense, H_from_sparse, atol=1e-8), \
        f"Max diff: {np.max(np.abs(H_dense - H_from_sparse))}"

    nnz = H_sparse.nnz
    print(f"n={n}, nnz={nnz}, density={nnz/(n*n):.4f}")
    print("PASS  sparse_large")


def test_sparse_tolerance():
    """Test that tol parameter filters small entries."""
    n = 5
    x = np.ones(n)

    def f(z):
        acc = 0.0
        for i in range(n):
            acc = acc + (z[i] - 1.0) ** 2
        for i in range(n - 1):
            acc = acc + 1e-10 * ad.sin(z[i]) * z[i + 1]
        return acc

    hess_fn = ad.sym_hess(f, n, vector_input=True)

    H0 = hess_fn.hess(_sparse_input(x), tol=0.0)
    H1e6 = hess_fn.hess(_sparse_input(x), tol=1e-6)
    H1e3 = hess_fn.hess(_sparse_input(x), tol=1e-3)

    def count_nnz(h):
        return h.nnz

    assert count_nnz(H0) >= count_nnz(H1e6) >= count_nnz(H1e3), \
        f"tol ordering wrong: {count_nnz(H0)} >= {count_nnz(H1e6)} >= {count_nnz(H1e3)}"

    print(f"tol=0: {count_nnz(H0)} nnz, tol=1e-6: {count_nnz(H1e6)}, tol=1e-3: {count_nnz(H1e3)}")
    print("PASS  sparse_tolerance")


def test_sparse_tridiagonal_pattern():
    """Verify the coupling pattern: only i,i and i,i+1 entries are nonzero."""
    n = 15
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)

    hess_fn = ad.sym_hess(make_obj(n), n, vector_input=True)
    H = hess_fn.hess(_sparse_input(x), tol=0.0)

    # Only main diagonal + one super/sub-diagonal should be nonzero
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and H[i, j] != 0:
                raise AssertionError(f"Unexpected nonzero at ({i},{j}) = {H[i,j]}")

    print(f"n={n}, nnz={H.nnz}, tridiagonal pattern verified")
    print("PASS  sparse_tridiagonal_pattern")


def test_shared_hessian_is_thread_safe_across_dense_and_sparse_calls():
    n = 12
    hess_fn = ad.sym_hess(make_obj(n), n, vector_input=True)
    points = [np.linspace(-1.0, 1.0, n) + 0.01 * i for i in range(24)]
    expected = [hess_fn(x) for x in points]

    def evaluate(item):
        i, x = item
        if i % 2:
            return hess_fn.hess(_sparse_input(x)).toarray()
        return hess_fn(x)

    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(evaluate, enumerate(points)))

    for got, want in zip(actual, expected):
        assert np.allclose(got, want, atol=1e-9)


if __name__ == "__main__":
    test_sparse_dense_match()
    test_sparse_vs_torch()
    test_sparse_large()
    test_sparse_tolerance()
    test_sparse_tridiagonal_pattern()
    test_shared_hessian_is_thread_safe_across_dense_and_sparse_calls()
    print("\nALL PASSED")

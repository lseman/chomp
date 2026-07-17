"""Compare the `ad` library against PyTorch autograd.

For each test function we build the *same* expression twice: once with the
`ad` operator overloads and once with `torch`. We then compare value, gradient,
Hessian, and (for vector-valued maps) the Jacobian.

Run with:  PYTHONPATH=build python tests/test_ad_vs_torch.py
(or via pytest if the harness allows it).
"""

import math

import numpy as np
import pytest
from scipy import sparse

import ad

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# torch reference helpers
# --------------------------------------------------------------------------- #
def torch_value_grad_hess(fn, x):
    """Return (value, grad, hess) for scalar fn: R^n -> R using torch autograd."""
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = fn(xt)
    grad = torch.autograd.grad(y, xt, create_graph=True)[0]
    n = x.shape[0]
    hess_rows = []
    for i in range(n):
        row = torch.autograd.grad(grad[i], xt, retain_graph=True)[0]
        hess_rows.append(row)
    hess = torch.stack(hess_rows)
    return float(y.detach()), grad.detach().numpy(), hess.detach().numpy()


def torch_value_jac(fns, x):
    """Return (values, jac) for a list of scalar fns: R^n -> R via torch."""
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    values = []
    jac_rows = []
    for fn in fns:
        if xt.grad is not None:
            xt.grad = None
        y = fn(xt)
        (g,) = torch.autograd.grad(y, xt, retain_graph=True)
        values.append(float(y.detach()))
        jac_rows.append(g.detach().numpy())
    return np.array(values), np.stack(jac_rows)


def ad_sparse_hessian(fn, x):
    """Compile fn and request a sparse Hessian with a sparse input vector."""
    hess_fn = ad.sym_hess(fn, x.shape[0], vector_input=True)
    x_sparse = sparse.csr_matrix(x.reshape(-1, 1))
    return hess_fn.hess(x_sparse).toarray()


# --------------------------------------------------------------------------- #
# paired functions: (ad_fn, torch_fn) computing the same expression
# --------------------------------------------------------------------------- #
def _smooth_ad(z):
    return (z[0] - 1.5) ** 2 + ad.sin(z[1]) * ad.exp(z[0]) + z[0] * z[1]


def _smooth_torch(z):
    return (z[0] - 1.5) ** 2 + torch.sin(z[1]) * torch.exp(z[0]) + z[0] * z[1]


def _trig_ad(z):
    return ad.tanh(z[0]) * ad.cos(z[1]) + ad.tan(z[0] * 0.3) + ad.log(z[1] + 5.0)


def _trig_torch(z):
    return torch.tanh(z[0]) * torch.cos(z[1]) + torch.tan(z[0] * 0.3) + torch.log(z[1] + 5.0)


def _poly_ad(z):
    return z[0] ** 3 * z[1] - 2.0 * z[0] * z[1] ** 2 + z[1] ** 4


def _poly_torch(z):
    return z[0] ** 3 * z[1] - 2.0 * z[0] * z[1] ** 2 + z[1] ** 4


def _pow_ad(z):
    return ad.pow(z[0], 2.5) + ad.pow(z[1] + 3.0, 1.5)


def _pow_torch(z):
    return torch.pow(z[0], 2.5) + torch.pow(z[1] + 3.0, 1.5)


def _activations_ad(z):
    return ad.gelu(z[0]) + ad.silu(z[1]) + ad.tanh(z[0] * z[1])


def _activations_torch(z):
    return (
        torch.nn.functional.gelu(z[0])
        + torch.nn.functional.silu(z[1])
        + torch.tanh(z[0] * z[1])
    )


def _div_ad(z):
    return z[0] / (z[1] + 2.0) + ad.exp(-z[0]) / z[1]


def _div_torch(z):
    return z[0] / (z[1] + 2.0) + torch.exp(-z[0]) / z[1]


SCALAR_CASES = [
    ("smooth", _smooth_ad, _smooth_torch, np.array([0.4, -0.7])),
    ("trig_log", _trig_ad, _trig_torch, np.array([0.6, 1.2])),
    ("poly", _poly_ad, _poly_torch, np.array([2.0, -3.0])),
    ("pow", _pow_ad, _pow_torch, np.array([1.3, 0.8])),
    ("activations", _activations_ad, _activations_torch, np.array([0.7, -1.1])),
    ("div", _div_ad, _div_torch, np.array([0.9, 1.4])),
]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,ad_fn,torch_fn,x", SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_value_grad_hessian_match_torch(name, ad_fn, torch_fn, x):
    x = x.astype(float)

    value, grad = ad.valgrad(ad_fn, x)
    hess = ad_sparse_hessian(ad_fn, x)

    t_value, t_grad, t_hess = torch_value_grad_hess(torch_fn, x)

    assert abs(value - t_value) < 1e-9, f"{name}: value {value} vs {t_value}"
    assert np.allclose(grad, t_grad, rtol=1e-8, atol=1e-9), (
        f"{name}: grad {grad} vs {t_grad}"
    )
    assert np.allclose(hess, t_hess, rtol=1e-7, atol=1e-8), (
        f"{name}: hess\n{hess}\nvs\n{t_hess}"
    )


def test_hessian_is_symmetric():
    for name, ad_fn, _, x in SCALAR_CASES:
        h = ad_sparse_hessian(ad_fn, x.astype(float))
        assert np.allclose(h, h.T, atol=1e-9), f"{name}: Hessian not symmetric"


def test_batch_valgrad_jacobian_matches_torch():
    x = np.array([2.0, 3.0], dtype=float)
    ad_fns = [lambda z: z[0] * z[1], lambda z: (z[0] - 1.0) ** 2 + z[1] * z[1]]
    torch_fns = [lambda z: z[0] * z[1], lambda z: (z[0] - 1.0) ** 2 + z[1] * z[1]]

    values, jac = ad.batch_valgrad(ad_fns, x)
    t_values, t_jac = torch_value_jac(torch_fns, x)

    assert np.allclose(values, t_values, rtol=1e-9, atol=1e-10)
    assert np.allclose(jac, t_jac, rtol=1e-8, atol=1e-9)


def test_relu_gradient_matches_torch_away_from_kink():
    for xv in (-2.0, 2.0, 0.5):
        x = np.array([xv], dtype=float)
        value, grad = ad.valgrad(lambda z: ad.relu(z[0]), x)
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        y = torch.relu(xt[0])
        (g,) = torch.autograd.grad(y, xt)
        assert abs(value - float(y.detach())) < 1e-12
        assert np.allclose(grad, g.detach().numpy(), atol=1e-12)


def test_gradient_matches_finite_difference():
    """Sanity check independent of torch: central differences."""
    x = np.array([0.4, -0.7], dtype=float)
    _, grad = ad.valgrad(_smooth_ad, x)
    eps = 1e-6
    fd = np.empty_like(x)
    for i in range(x.shape[0]):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        vp, _ = ad.valgrad(_smooth_ad, xp)
        vm, _ = ad.valgrad(_smooth_ad, xm)
        fd[i] = (vp - vm) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    import sys

    failures = 0
    for name, ad_fn, torch_fn, x in SCALAR_CASES:
        try:
            test_value_grad_hessian_match_torch(name, ad_fn, torch_fn, x)
            print(f"PASS  value/grad/hess  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  value/grad/hess  {name}\n      {e}")

    for fn in (
        test_hessian_is_symmetric,
        test_batch_valgrad_jacobian_matches_torch,
        test_relu_gradient_matches_torch_away_from_kink,
        test_gradient_matches_finite_difference,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {e}")

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)

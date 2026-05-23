import math

import numpy as np

import ad


def smooth_objective(x):
    return (x[0] - 1.5) ** 2 + ad.sin(x[1]) * ad.exp(x[0]) + x[0] * x[1]


def expected_smooth(x):
    ex = math.exp(x[0])
    sy = math.sin(x[1])
    cy = math.cos(x[1])
    value = (x[0] - 1.5) ** 2 + sy * ex + x[0] * x[1]
    grad = np.array(
        [
            2.0 * (x[0] - 1.5) + sy * ex + x[1],
            cy * ex + x[0],
        ]
    )
    hess = np.array(
        [
            [2.0 + sy * ex, 1.0 + cy * ex],
            [1.0 + cy * ex, -sy * ex],
        ]
    )
    return value, grad, hess


def assert_close(actual, expected, *, atol=1e-10):
    assert np.allclose(actual, expected, rtol=1e-10, atol=atol), (
        f"actual={actual}, expected={expected}"
    )


def test_valgrad_and_hessian_match_analytic_formula():
    x = np.array([0.4, -0.7], dtype=float)
    value_expected, grad_expected, hess_expected = expected_smooth(x)

    value, grad = ad.valgrad(smooth_objective, x)
    hess = ad.hess(smooth_objective, x)

    assert abs(value - value_expected) < 1e-12
    assert_close(grad, grad_expected)
    assert_close(hess, hess_expected)


def test_basic_polynomial_hessian_and_cross_terms():
    x = np.array([2.0, -3.0], dtype=float)

    value, grad = ad.valgrad(lambda z: z[0] * z[1] + z[1] * z[1], x)
    hess = ad.hess(lambda z: z[0] * z[1] + z[1] * z[1], x)

    assert value == -6.0 + 9.0
    assert_close(grad, np.array([-3.0, -4.0]))
    assert_close(hess, np.array([[0.0, 1.0], [1.0, 2.0]]))


def test_batch_valgrad_for_python_callables():
    x = np.array([2.0, 3.0], dtype=float)
    values, jac = ad.batch_valgrad(
        [
            lambda z: z[0] * z[1],
            lambda z: (z[0] - 1.0) ** 2 + z[1] * z[1],
        ],
        x,
    )

    assert_close(values, np.array([6.0, 10.0]))
    assert_close(jac, np.array([[3.0, 2.0], [2.0, 6.0]]))


def test_relu_piecewise_gradient_away_from_kink():
    neg_value, neg_grad = ad.valgrad(lambda z: ad.relu(z[0]), np.array([-2.0]))
    pos_value, pos_grad = ad.valgrad(lambda z: ad.relu(z[0]), np.array([2.0]))

    assert neg_value == 0.0
    assert_close(neg_grad, np.array([0.0]))
    assert pos_value == 2.0
    assert_close(pos_grad, np.array([1.0]))

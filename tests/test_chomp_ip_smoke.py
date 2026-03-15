from pathlib import Path

import numpy as np

import ad
from chomp import ChompConfig, chomp


ROOT = Path(__file__).resolve().parents[1]


def quadratic(x):
    return (x[0] - 0.25) ** 2


def rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def upper_bound_zero(x):
    return x[0]


def make_cfg(use_funnel: bool) -> ChompConfig:
    cfg = ChompConfig()
    cfg.mode = "ip"
    cfg.verbose = False
    cfg.use_filter = not use_funnel
    cfg.use_funnel = use_funnel
    cfg.tol_stat = 1e-6
    cfg.tol_feas = 1e-6
    cfg.tol_comp = 1e-6
    cfg.ip_mu_init = 1e-1
    cfg.ip_tau = 0.995
    cfg.ip_tau_pri = 0.995
    cfg.ip_tau_dual = 0.995
    cfg.barrier_strategy = "mehrotra_safeguarded"
    return cfg


def smoke_solve(use_funnel: bool) -> np.ndarray:
    solver = chomp(
        quadratic,
        [upper_bound_zero],
        [],
        None,
        None,
        np.array([0.5], dtype=float),
        make_cfg(use_funnel=use_funnel),
    )
    return np.asarray(solver.solve(max_iter=2, tol=1e-8, verbose=False))


def assert_source_regressions_fixed() -> None:
    stepper_src = (ROOT / "include/ip/stepper.h").read_text()
    ls_src = (ROOT / "include/blocks/linesearch.h").read_text()
    helpers_src = (ROOT / "include/ip/helpers.h").read_text()

    assert "ls_->search(m_, x, dx, (mI ? ds : dvec()), (mI ? s : dvec()), mu," in stepper_src
    assert "d_phi -= mu * ds[i] / std::max(s[i], consts::EPS_POS);" in stepper_src
    assert "if (theta0 <= filter_theta_min)" in ls_src
    assert "filter_->add_if_acceptable(theta_t, phi_t);" in ls_src
    assert "phi0 -= mu * bound_log_sum0;" in ls_src
    assert "bound_barrier_directional_derivative(x, dx, bounds, mu, barrier_eps)" in ls_src
    assert "rhs_x[i] += bound_corr;" in helpers_src


def rosenbrock_filter_solve() -> np.ndarray:
    cfg = make_cfg(use_funnel=False)
    cfg.use_trust_region = True
    cfg.delta0 = 1.0
    cfg.norm_type = "2"
    cfg.piqp_verbose = False
    cfg.ip_sigma_power = 3
    cfg.ip_switch_theta = 1e-5
    cfg.ip_switch_mu = 1e-8
    cfg.ip_stall_iters = 5
    solver = chomp(
        rosenbrock,
        [],
        [],
        np.array([-2.0, -1.0], dtype=float),
        np.array([2.0, 3.0], dtype=float),
        np.array([-1.2, 1.0], dtype=float),
        cfg,
    )
    return np.asarray(solver.solve(max_iter=60, tol=1e-8, verbose=False))


filter_sol = smoke_solve(use_funnel=False)
funnel_sol = smoke_solve(use_funnel=True)
rosenbrock_filter_sol = rosenbrock_filter_solve()

assert np.all(np.isfinite(filter_sol)), f"non-finite filter solve output: {filter_sol}"
assert np.all(np.isfinite(funnel_sol)), f"non-finite funnel solve output: {funnel_sol}"
assert rosenbrock(rosenbrock_filter_sol) < 1e-8, (
    f"filter IP failed to converge on Rosenbrock: {rosenbrock_filter_sol}"
)
assert_source_regressions_fixed()

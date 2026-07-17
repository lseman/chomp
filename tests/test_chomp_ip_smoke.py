from pathlib import Path

import numpy as np
from chomp import ChompConfig, chomp
from scipy.optimize import minimize

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


def solve_with_scipy(
    objective_fn,
    ineq_fns,
    eq_fns,
    x0: np.ndarray,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
) -> np.ndarray:
    constraints = []
    for fn in ineq_fns:
        constraints.append({"type": "ineq", "fun": lambda x, fn=fn: -float(fn(x))})
    for fn in eq_fns:
        constraints.append({"type": "eq", "fun": lambda x, fn=fn: float(fn(x))})

    bounds = None
    if lb is not None or ub is not None:
        x0_arr = np.asarray(x0, dtype=float)
        lb_arr = (
            np.full_like(x0_arr, -np.inf) if lb is None else np.asarray(lb, dtype=float)
        )
        ub_arr = (
            np.full_like(x0_arr, np.inf) if ub is None else np.asarray(ub, dtype=float)
        )
        bounds = list(zip(lb_arr.tolist(), ub_arr.tolist(), strict=True))

    result = minimize(
        objective_fn,
        np.asarray(x0, dtype=float),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 400, "disp": False},
    )
    assert result.success, result.message
    return np.asarray(result.x, dtype=float)


def print_comparison(
    name: str, ours: np.ndarray, scipy_sol: np.ndarray, objective_fn, ineq_fns, eq_fns
) -> None:
    ours = np.asarray(ours, dtype=float)
    scipy_sol = np.asarray(scipy_sol, dtype=float)
    ours_ineq = [float(fn(ours)) for fn in ineq_fns]
    scipy_ineq = [float(fn(scipy_sol)) for fn in ineq_fns]
    ours_eq = [float(fn(ours)) for fn in eq_fns]
    scipy_eq = [float(fn(scipy_sol)) for fn in eq_fns]

    print(f"[{name}] chomp x      = {ours}")
    print(f"[{name}] scipy x      = {scipy_sol}")
    print(f"[{name}] |dx|         = {np.linalg.norm(ours - scipy_sol):.3e}")
    print(f"[{name}] chomp f      = {float(objective_fn(ours)):.12e}")
    print(f"[{name}] scipy f      = {float(objective_fn(scipy_sol)):.12e}")
    if ineq_fns:
        print(f"[{name}] chomp ineq   = {ours_ineq}")
        print(f"[{name}] scipy ineq   = {scipy_ineq}")
    if eq_fns:
        print(f"[{name}] chomp eq     = {ours_eq}")
        print(f"[{name}] scipy eq     = {scipy_eq}")


def assert_matches_scipy(
    name: str, ours: np.ndarray, scipy_sol: np.ndarray, atol: float = 1e-5
) -> None:
    delta = np.linalg.norm(
        np.asarray(ours, dtype=float) - np.asarray(scipy_sol, dtype=float)
    )
    assert delta < atol, (
        f"{name}: chomp/scipy mismatch {delta:.3e}; chomp={ours}, scipy={scipy_sol}"
    )


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


def compare_solve(use_funnel: bool) -> np.ndarray:
    solver = chomp(
        quadratic,
        [upper_bound_zero],
        [],
        None,
        None,
        np.array([0.5], dtype=float),
        make_cfg(use_funnel=use_funnel),
    )
    return np.asarray(solver.solve(max_iter=60, tol=1e-8, verbose=False), dtype=float)


def assert_source_regressions_fixed() -> None:
    stepper_src = (ROOT / "include/ip/stepper.h").read_text()
    ls_src = (ROOT / "include/blocks/linesearch.h").read_text()
    mg_solver_src = (ROOT / "include/ip/mg_solver.h").read_text()

    assert (
        "ls_->search(m_, x, dx, (mI ? ds : dvec()), (mI ? s : dvec()), mu,"
        in stepper_src
    )
    assert "d_phi -= mu * ds[i] / std::max(s[i] + sh, consts::EPS_POS);" in stepper_src
    assert "if (theta0 <= filter_theta_min)" in ls_src
    assert "filter_->add_if_acceptable(theta_t, phi_t);" in ls_src
    assert "phi0 -= mu * bound_log_sum0;" in ls_src
    assert (
        "bound_barrier_directional_derivative(x, dx, bounds, mu, barrier_eps)" in ls_src
    )
    assert "rhs_x[i] += bound_corr;" in mg_solver_src


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
filter_compare_sol = compare_solve(use_funnel=False)
funnel_compare_sol = compare_solve(use_funnel=True)
rosenbrock_filter_sol = rosenbrock_filter_solve()

quadratic_scipy_sol = solve_with_scipy(
    quadratic,
    [upper_bound_zero],
    [],
    np.array([0.5], dtype=float),
)
rosenbrock_scipy_sol = solve_with_scipy(
    rosenbrock,
    [],
    [],
    np.array([-1.2, 1.0], dtype=float),
    lb=np.array([-2.0, -1.0], dtype=float),
    ub=np.array([2.0, 3.0], dtype=float),
)

print_comparison(
    "ip-filter-quadratic",
    filter_compare_sol,
    quadratic_scipy_sol,
    quadratic,
    [upper_bound_zero],
    [],
)
print_comparison(
    "ip-funnel-quadratic",
    funnel_compare_sol,
    quadratic_scipy_sol,
    quadratic,
    [upper_bound_zero],
    [],
)
print_comparison(
    "ip-filter-rosenbrock",
    rosenbrock_filter_sol,
    rosenbrock_scipy_sol,
    rosenbrock,
    [],
    [],
)

assert np.all(np.isfinite(filter_sol)), f"non-finite filter solve output: {filter_sol}"
assert np.all(np.isfinite(funnel_sol)), f"non-finite funnel solve output: {funnel_sol}"
assert np.all(np.isfinite(filter_compare_sol)), (
    f"non-finite filter comparison solve output: {filter_compare_sol}"
)
assert np.all(np.isfinite(funnel_compare_sol)), (
    f"non-finite funnel comparison solve output: {funnel_compare_sol}"
)
assert rosenbrock(rosenbrock_filter_sol) < 1e-8, (
    f"filter IP failed to converge on Rosenbrock: {rosenbrock_filter_sol}"
)
assert_matches_scipy(
    "ip-filter-rosenbrock", rosenbrock_filter_sol, rosenbrock_scipy_sol
)
assert_source_regressions_fixed()

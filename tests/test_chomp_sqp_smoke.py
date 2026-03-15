from pathlib import Path

import numpy as np

import ad
from chomp import ChompConfig, chomp


ROOT = Path(__file__).resolve().parents[1]


def objective(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def equality(x):
    return x[0] + x[1] - 3.0


def make_cfg() -> ChompConfig:
    cfg = ChompConfig()
    cfg.mode = "sqp"
    cfg.verbose = False
    cfg.use_trust_region = True
    cfg.use_filter = True
    cfg.feasibility_emphasis = True
    cfg.delta0 = 1.0
    cfg.norm_type = "2"
    cfg.hessian_mode = "exact"
    cfg.tol_stat = 1e-7
    cfg.tol_feas = 1e-7
    cfg.tol_comp = 1e-7
    return cfg


def assert_source_regressions_fixed() -> None:
    manager_src = (ROOT / "include/trustregion/manager.h").read_text()
    stepper_src = (ROOT / "include/sqp/stepper.h").read_text()

    assert "current_theta_ = *theta_old;" in manager_src
    assert "const double funnel_ref =" in manager_src
    assert "std::isfinite(current_theta_) ? current_theta_ : theta0;" in manager_src
    assert "std::max(actual, theta_actual)" not in manager_src
    assert (
        "if (!std::isfinite(actual) && theta_old && theta_trial &&"
        in manager_src
    )
    assert "exact_penalty_merit_(" in manager_src
    assert "restoration_phase_(" in manager_src
    assert '"ls-merit"' in manager_src
    assert '"restoration-filter"' in manager_src
    assert "restoration_max_iter" in stepper_src
    assert '"feasibility_emphasis"' in stepper_src


solver = chomp(
    objective,
    [],
    [equality],
    None,
    None,
    np.array([0.0, 0.0], dtype=float),
    make_cfg(),
)
sol = np.asarray(solver.solve(max_iter=60, tol=1e-8, verbose=False), dtype=float)

assert np.all(np.isfinite(sol)), f"non-finite SQP solve output: {sol}"
assert np.linalg.norm(sol - np.array([1.0, 2.0], dtype=float)) < 1e-5, sol
assert abs(float(equality(sol))) < 1e-7, sol
assert float(objective(sol)) < 1e-10, sol

assert_source_regressions_fixed()

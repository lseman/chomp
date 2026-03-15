import os
import subprocess
import sys


def run_case(expr: str) -> str:
    script = f"""
import gc
import numpy as np
import ad
from chomp import ChompConfig, chomp

cfg = ChompConfig()
cfg.print_convexity_analysis = True

def objective(x):
    return {expr}

solver = chomp(
    objective,
    [],
    [],
    np.array([-2.0], dtype=float),
    np.array([2.0], dtype=float),
    np.array([0.0], dtype=float),
    cfg,
)
del solver
del cfg
gc.collect()
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "nanobind: leaked" not in combined, combined
    if "Problem is CONVEX" in combined:
        return "CONVEX"
    if "Problem is NON-CONVEX" in combined:
        return "NON-CONVEX"
    raise AssertionError(combined)


assert run_case("x[0] * x[0]") == "CONVEX"
assert run_case("ad.exp(x[0])") == "CONVEX"
assert run_case("ad.pow(1.0 + x[0] * x[0], 0.5)") == "CONVEX"
assert run_case("ad.pow(1.0 + ((x[0] - 1.0) * (x[0] - 1.0)), 0.5)") == "CONVEX"
assert run_case("(x[0] * x[0]) / (x[0] + 3.0)") == "CONVEX"
assert run_case("((x[0] - 1.0) * (x[0] - 1.0)) / (x[0] + 3.0)") == "CONVEX"
assert run_case("1.0 / (1.0 + x[0] * x[0])") == "NON-CONVEX"
assert run_case("ad.exp(-(x[0] * x[0]))") == "NON-CONVEX"
assert run_case("ad.max(-(x[0] * x[0]), -((x[0] - 1.0) * (x[0] - 1.0)))") == "NON-CONVEX"

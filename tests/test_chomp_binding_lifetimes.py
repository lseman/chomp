import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SCRIPT = r"""
import gc
import numpy as np
import ad
from chomp import ChompConfig, chomp

def objective(x):
    return (x[0] - 1.0) ** 2

cfg = ChompConfig()
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
print("binding cleanup ok")
"""

GLOBAL_SCRIPT = r"""
import numpy as np
import ad
from chomp import ChompConfig, chomp

def objective(x):
    return (x[0] - 1.0) ** 2

cfg = ChompConfig()
solver = chomp(
    objective,
    [],
    [],
    np.array([-2.0], dtype=float),
    np.array([2.0], dtype=float),
    np.array([0.0], dtype=float),
    cfg,
)
print("binding global ok")
"""

WITH_SCRIPT = r"""
import numpy as np
import ad
from chomp import ChompConfig, chomp

def objective(x):
    return (x[0] - 1.0) ** 2

cfg = ChompConfig()
with chomp(
    objective,
    [],
    [],
    np.array([-2.0], dtype=float),
    np.array([2.0], dtype=float),
    np.array([0.0], dtype=float),
    cfg,
) as solver:
    pass
print("binding with ok")
"""

VERBOSE_SCRIPT = r"""
import numpy as np
import ad
from chomp import ChompConfig, chomp

def objective(x):
    return (x[0] - 1.0) ** 2

cfg = ChompConfig()
cfg.print_convexity_analysis = True
solver = chomp(
    objective,
    [],
    [],
    np.array([-2.0], dtype=float),
    np.array([2.0], dtype=float),
    np.array([0.0], dtype=float),
    cfg,
)
print("binding verbose ok")
"""


env = os.environ.copy()
pythonpath_entries = [str(ROOT / "build"), str(ROOT)]
if env.get("PYTHONPATH"):
    pythonpath_entries.append(env["PYTHONPATH"])
env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
proc = subprocess.run(
    [sys.executable, "-c", SCRIPT],
    capture_output=True,
    text=True,
    env=env,
    check=False,
)

combined = proc.stdout + proc.stderr

assert proc.returncode == 0, combined
assert "binding cleanup ok" in proc.stdout, combined
assert "nanobind: leaked" not in combined, combined
assert "SUSPECT Convexity Analysis" not in combined, combined

global_proc = subprocess.run(
    [sys.executable, "-c", GLOBAL_SCRIPT],
    capture_output=True,
    text=True,
    env=env,
    check=False,
)

global_combined = global_proc.stdout + global_proc.stderr

assert global_proc.returncode == 0, global_combined
assert "binding global ok" in global_proc.stdout, global_combined
assert "nanobind: leaked" not in global_combined, global_combined

with_proc = subprocess.run(
    [sys.executable, "-c", WITH_SCRIPT],
    capture_output=True,
    text=True,
    env=env,
    check=False,
)

with_combined = with_proc.stdout + with_proc.stderr

assert with_proc.returncode == 0, with_combined
assert "binding with ok" in with_proc.stdout, with_combined
assert "nanobind: leaked" not in with_combined, with_combined

verbose_proc = subprocess.run(
    [sys.executable, "-c", VERBOSE_SCRIPT],
    capture_output=True,
    text=True,
    env=env,
    check=False,
)

verbose_combined = verbose_proc.stdout + verbose_proc.stderr

assert verbose_proc.returncode == 0, verbose_combined
assert "binding verbose ok" in verbose_proc.stdout, verbose_combined
assert "SUSPECT Convexity Analysis" in verbose_combined, verbose_combined

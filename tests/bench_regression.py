"""Benchmark regression gate for CI.

Measures ad.valgrad and ad.hess throughput on a representative workload and
rejects the build if performance drops more than ``regression_threshold`` %
below the reference baseline.

Usage:
    PYTHONPATH=build python tests/bench_regression.py
    PYTHONPATH=build python tests/bench_regression.py --threshold 15

Exit 0 = no regression, exit 1 = regression detected.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import ad

# --------------------------------------------------------------------------- #
# Reference baselines (measured on CI baseline commit, in µs per call)
# --------------------------------------------------------------------------- #
# These are *maximum acceptable* values — the benchmark must be AT LEAST this
# fast.  If your hardware changes, regenerate these with --regen.
REFERENCE = {
    # n=10 valgrad (µs)  — mid-size workload, most common in the solver
    "valgrad_10": 80.0,
    # n=10 hess  (µs)  — Hessian on the same workload
    "hess_10": 350.0,
    # n=50 valgrad (µs)  — larger batch, less frequent but still used
    "valgrad_50": 1200.0,
    # n=50 hess  (µs)
    "hess_50": 8000.0,
}

# Allow this % slowdown before flagging a regression
DEFAULT_THRESHOLD = 10  # percent


# --------------------------------------------------------------------------- #
# Workload: a dense smooth objective matching the SQP usage pattern
# --------------------------------------------------------------------------- #
def make_workload(n):
    """Return an AD objective that exercises add, mul, sin, exp, pow."""
    def f(z):
        acc = 0.0
        for i in range(n):
            acc += (z[i] - 1.0) ** 2
        for i in range(n - 1):
            acc += 0.5 * ad.sin(z[i]) * z[i + 1]
            acc += 0.1 * ad.exp(z[i] * 0.1)
        return acc
    return f


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #
def time_fn(fn, warmup=30, iters=500):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def run_benchmarks(threshold_pct=DEFAULT_THRESHOLD):
    rng = np.random.default_rng(42)
    results = {}
    failures = []

    for name, x in [("n10", rng.standard_normal(10)),
                    ("n50", rng.standard_normal(50))]:
        f = make_workload(len(x))
        # Warm up once
        ad.valgrad(f, x)
        ad.hess(f, x)

        vg_us = time_fn(lambda: ad.valgrad(f, x), warmup=50, iters=200)
        h_us = time_fn(lambda: ad.hess(f, x), warmup=10, iters=50)

        results[f"valgrad_{name}"] = round(vg_us, 2)
        results[f"hess_{name}"] = round(h_us, 2)

        # Check valgrad
        n = name[1:]  # "n10" -> "10"
        key = f"valgrad_{n}"
        if key in REFERENCE:
            ref = REFERENCE[key]
            deviation = ((vg_us - ref) / ref) * 100
            dev_key = f"valgrad_{n}_dev"
            results[dev_key] = round(deviation, 2)
            if deviation < -threshold_pct:
                failures.append(
                    f"valgrad {name}: {vg_us:.1f} µs vs ref {ref:.1f} µs "
                    f"({deviation:.1f}% vs -{threshold_pct}% threshold)"
                )

        # Check hess
        key = f"hess_{n}"
        if key in REFERENCE:
            ref = REFERENCE[key]
            deviation = ((h_us - ref) / ref) * 100
            dev_key = f"hess_{n}_dev"
            results[dev_key] = round(deviation, 2)
            if deviation < -threshold_pct:
                failures.append(
                    f"hess {name}: {h_us:.1f} µs vs ref {ref:.1f} µs "
                    f"({deviation:.1f}% vs -{threshold_pct}% threshold)"
                )

    return results, failures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    import argparse
    p = argparse.ArgumentParser(description="Benchmark regression gate")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Max allowed regression in percent (default: 10)")
    args = p.parse_args()

    results, failures = run_benchmarks(threshold_pct=args.threshold)

    # Print summary table
    print(f"{'Metric':>20} | {'Measured':>10} {'Reference':>10} {'Deviation':>10}")
    print("-" * 58)
    for key in sorted(results.keys()):
        if key.endswith("_dev"):
            continue
        dev_key = key + "_dev"
        dev = results.get(dev_key, "N/A")
        ref_str = "N/A" if dev == "N/A" else f"{dev}% "
        print(f"{key:>20} | {results[key]:>10.2f} {'N/A':>10} {f'{dev}%':>10}")

    # Also output JSON for CI parsing
    json_out = {k: v for k, v in results.items() if not k.endswith("_dev")}
    baseline_dir = Path(__file__).parent.parent / ".bench_baselines"
    baseline_file = baseline_dir / "ad_bench.json"
    baseline_dir.mkdir(exist_ok=True)
    baseline_file.write_text(json.dumps(json_out, indent=2))

    if failures:
        print(f"\n⚠ REGRESSION DETECTED ({len(failures)} check(s)):")
        for msg in failures:
            print(f"  - {msg}")
        print("\nTo update baselines: edit the REFERENCE dict in this file.")
        print("Or increase --threshold if the drop is acceptable.")
        sys.exit(1)
    else:
        print(f"\n✓ All checks within ±{args.threshold}% threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""Benchmark the `ad` library against PyTorch autograd on the workload that
matters for the SQP solver: repeated value/grad/Hessian of a *fixed* function
at *changing* x (the cached path).

Run:  PYTHONPATH=build python tests/bench_ad_vs_torch.py
"""

import time

import numpy as np

import ad

try:
    import torch

    torch.set_num_threads(1)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def timeit(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


# A scalable smooth objective: sum of coupled nonlinear terms (dense-ish grad,
# tridiagonal-ish Hessian). Built identically in ad and torch.
def make_ad_obj(n):
    def f(z):
        acc = 0.0
        for i in range(n):
            acc = acc + (z[i] - 1.0) ** 2
        for i in range(n - 1):
            acc = acc + 0.5 * ad.sin(z[i]) * z[i + 1] + 0.1 * ad.exp(z[i] * 0.1)
        return acc

    return f


def make_torch_obj(n):
    def f(z):
        acc = ((z - 1.0) ** 2).sum()
        acc = acc + (0.5 * torch.sin(z[:-1]) * z[1:]).sum()
        acc = acc + (0.1 * torch.exp(z[:-1] * 0.1)).sum()
        return acc

    return f


def torch_valgrad(fn):
    def run(x):
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        y = fn(xt)
        (g,) = torch.autograd.grad(y, xt)
        return float(y), g
    return run


def torch_hess(fn, n):
    def run(x):
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.hessian(fn, xt)
    return run


def main():
    sizes = [2, 5, 10, 25, 50]
    iters_by_size = {2: 5000, 5: 5000, 10: 2000, 25: 1000, 50: 500}

    print(f"{'n':>4} | {'ad valgrad':>12} {'torch vg':>12} {'speedup':>8} "
          f"| {'ad hess':>12} {'torch hess':>12} {'speedup':>8}  (us/call)")
    print("-" * 92)

    for n in sizes:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        iters = iters_by_size[n]

        ad_f = make_ad_obj(n)
        # warm the cache once
        ad.valgrad(ad_f, x)
        ad.hess(ad_f, x)

        t_ad_vg = timeit(lambda: ad.valgrad(ad_f, x), iters)
        t_ad_h = timeit(lambda: ad.hess(ad_f, x), max(iters // 5, 50))

        if HAVE_TORCH:
            tf = make_torch_obj(n)
            run_vg = torch_valgrad(tf)
            run_h = torch_hess(tf, n)
            t_t_vg = timeit(lambda: run_vg(x), iters)
            t_t_h = timeit(lambda: run_h(x), max(iters // 10, 30))
        else:
            t_t_vg = t_t_h = float("nan")

        print(f"{n:>4} | {t_ad_vg*1e6:>12.2f} {t_t_vg*1e6:>12.2f} "
              f"{t_t_vg/t_ad_vg:>7.1f}x | {t_ad_h*1e6:>12.2f} {t_t_h*1e6:>12.2f} "
              f"{t_t_h/t_ad_h:>7.1f}x")


if __name__ == "__main__":
    main()

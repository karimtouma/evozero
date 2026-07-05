#!/usr/bin/env python3
"""Fair head-to-head benchmark: evozero (GPU) vs Operon / PySR / gplearn (CPU).

Same problem, same operators (+ - * / sin cos), same max size, same wall-clock
budget, same data; CPU methods get many threads. We sweep the dataset size N to
find the regime where the GPU-batched fitness wins. Reports median test-R^2 and
wall time over several seeds.

Honest by construction: on small N the optimized CPU tools (Operon C++/SIMD,
PySR Julia) are expected to win; the GPU should pull ahead as N grows.

Run (on the GPU box, after warming up PySR/Julia once):
    JULIA_NUM_THREADS=32 python benchmarks/benchmark.py --budget 20 --seeds 3
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JULIA_NUM_THREADS", "32")  # must be set before importing pysr

import numpy as np

BINARY = ["+", "-", "*", "/"]
UNARY = ["sin", "cos"]
MAXSIZE = 30
NTHREADS = int(os.environ.get("BENCH_THREADS", "32"))
N_TEST = 2000


def target(X: np.ndarray) -> np.ndarray:
    # y = x0^2 + x0*x1 + sin(x1)  — needs only + - * / sin, supported by all tools
    return X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # float64: native for Operon/PySR (they degrade badly on float32). evozero casts
    # to float32 internally by design — a fair, honest precision difference.
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(-3, 3, size=(n, 2))
    Xte = rng.uniform(-3, 3, size=(N_TEST, 2))
    return Xtr, target(Xtr), Xte, target(Xte)


def warmup(methods: list[str]) -> None:
    """One discarded fit per method: pays Julia JIT / CUDA kernel compile off-clock."""
    Xtr, ytr, Xte, yte = make_data(200, 99999)
    for name in methods:
        try:
            METHODS[name](Xtr, ytr, Xte, yte, 3.0, 99999)
        except Exception:  # noqa: BLE001
            pass


def r2(yt: np.ndarray, yp: np.ndarray) -> float:
    yt = np.asarray(yt, float).ravel()
    yp = np.asarray(yp, float).ravel()
    if not np.isfinite(yp).all():
        return float("nan")
    return float(1.0 - ((yt - yp) ** 2).sum() / (((yt - yt.mean()) ** 2).sum() + 1e-12))


# ── methods: each returns (test_r2, wall_seconds) ───────────────────────────
def run_evozero(Xtr, ytr, Xte, yte, budget, seed):
    from evozero import SymbolicRegressor

    m = SymbolicRegressor(
        binary_operators=tuple(BINARY), unary_operators=tuple(UNARY),
        max_size=MAXSIZE, population_size=10000, generations=10_000_000,
        max_time=budget, device="auto", random_state=seed,
    )
    t = time.time()
    m.fit(Xtr, ytr)
    return r2(yte, m.predict(Xte)), time.time() - t


def run_operon(Xtr, ytr, Xte, yte, budget, seed):
    from pyoperon.sklearn import SymbolicRegressor as OP

    m = OP(
        allowed_symbols="add,sub,mul,div,sin,cos,constant,variable",
        max_length=MAXSIZE, population_size=1000, generations=10_000_000,
        max_evaluations=1_000_000_000, max_time=int(budget), n_threads=NTHREADS,
        random_state=seed,
    )
    t = time.time()
    m.fit(Xtr, ytr)
    return r2(yte, m.predict(Xte)), time.time() - t


def run_pysr(Xtr, ytr, Xte, yte, budget, seed):
    from pysr import PySRRegressor

    m = PySRRegressor(
        binary_operators=list(BINARY), unary_operators=list(UNARY),
        maxsize=MAXSIZE, niterations=10_000_000, timeout_in_seconds=budget,
        verbosity=0, progress=False, random_state=seed, temp_equation_file=True,
    )
    t = time.time()
    m.fit(Xtr, ytr)
    return r2(yte, m.predict(Xte)), time.time() - t


def run_gplearn(Xtr, ytr, Xte, yte, budget, seed):
    from gplearn.genetic import SymbolicRegressor as GP

    m = GP(
        function_set=("add", "sub", "mul", "div", "sin", "cos"),
        population_size=2000, generations=1, warm_start=True,
        n_jobs=NTHREADS, random_state=seed, verbose=0,
    )
    t = time.time()
    gen = 0
    while time.time() - t < budget and gen < 1000:
        gen += 1
        m.set_params(generations=gen)
        m.fit(Xtr, ytr)
    return r2(yte, m.predict(Xte)), time.time() - t


METHODS = {"evozero(GPU)": run_evozero, "operon(CPU)": run_operon,
           "pysr(CPU)": run_pysr, "gplearn(CPU)": run_gplearn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=20.0, help="wall-clock seconds per run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sizes", type=str, default="1000,10000,100000,1000000")
    ap.add_argument("--methods", type=str, default="evozero(GPU),operon(CPU),pysr(CPU),gplearn(CPU)")
    ap.add_argument("--skip-slow-above", type=int, default=10000,
                    help="skip gplearn above this N (it is the slow Python baseline)")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    methods = [m for m in args.methods.split(",") if m in METHODS]
    print(f"budget={args.budget}s  seeds={args.seeds}  threads(CPU)={NTHREADS}  "
          f"ops={BINARY + UNARY}  maxsize={MAXSIZE}\n")
    print("warming up (Julia JIT / CUDA compile, off-clock)...", flush=True)
    warmup(methods)
    header = f"{'N':>9} | " + " | ".join(f"{m:>18}" for m in methods)
    print(header)
    print("-" * len(header))

    results = []
    for n in sizes:
        cells = []
        for name in methods:
            if name == "gplearn(CPU)" and n > args.skip_slow_above:
                cells.append(f"{'—':>18}")
                continue
            r2s, times = [], []
            for seed in range(args.seeds):
                Xtr, ytr, Xte, yte = make_data(n, seed)
                try:
                    r, dt = METHODS[name](Xtr, ytr, Xte, yte, args.budget, seed)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {name} N={n} seed={seed}: {type(e).__name__}: {e}")
                    r, dt = float("nan"), float("nan")
                r2s.append(r)
                times.append(dt)
                results.append({"N": n, "method": name, "seed": seed, "r2": r, "time": dt})
            med_r2 = float(np.nanmedian(r2s))
            med_t = float(np.nanmedian(times))
            cells.append(f"{med_r2:>7.4f} ({med_t:>4.0f}s)")
        print(f"{n:>9} | " + " | ".join(cells))

    # csv
    import csv
    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "method", "seed", "r2", "time"])
        w.writeheader()
        w.writerows(results)
    print("\nwrote benchmark_results.csv")


if __name__ == "__main__":
    main()

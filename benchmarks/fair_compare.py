#!/usr/bin/env python3
"""Fairness-controlled comparison, resolving the BENCHMARK.md caveat that evozero's
large-N speed mixed two advantages (GPU + subsampling) vs CPU tools on full data.

No competitor has a GPU backend (Operon/PySR/gplearn are CPU-only), so a literal
"GPU vs GPU" is impossible. Instead we DECOMPOSE the advantage:

  A) CPU vs CPU, SAME hardware, SAME (full) data  -> is evozero's *search* competitive?
  B) evozero GPU vs CPU (subsampling on)           -> what does the GPU actually buy?
  C) subsampling isolation: give the CPU tools the SAME 2048-row subsample
                                                   -> is the speed from subsampling, not GPU?

Same target/protocol as benchmark.py. Budget 30 s, 2 seeds, fp64.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JULIA_NUM_THREADS", "32")
sys.path.insert(0, "benchmarks")

import numpy as np
import torch
from benchmark import BINARY, MAXSIZE, UNARY, make_data, r2

BUDGET = 30.0
SEEDS = 2
POP = 3000
SUB = 2048


def evozero(Xtr, ytr, Xte, yte, seed, device, subsample):
    from evozero import SymbolicRegressor
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    kw = {"subsample_size": SUB, "subsample_threshold": 0} if subsample \
        else {"subsample_size": None, "subsample_threshold": 10**12}  # off: full data
    m = SymbolicRegressor(
        binary_operators=tuple(BINARY), unary_operators=tuple(UNARY), max_size=MAXSIZE,
        population_size=POP, generations=10_000_000, max_time=BUDGET,
        device=device, random_state=seed, **kw,
    )
    t = time.time()
    m.fit(Xtr, ytr)
    dt = time.time() - t
    peak = (torch.cuda.max_memory_allocated() / 1e9) if (device != "cpu" and torch.cuda.is_available()) else 0.0
    return r2(yte, m.predict(Xte)), dt, peak


def operon(Xtr, ytr, Xte, yte, seed):
    from pyoperon.sklearn import SymbolicRegressor as OP
    m = OP(allowed_symbols="add,sub,mul,div,sin,cos,constant,variable", max_length=MAXSIZE,
           population_size=1000, generations=10_000_000, max_evaluations=1_000_000_000,
           max_time=int(BUDGET), n_threads=32, random_state=seed)
    t = time.time()
    m.fit(Xtr, ytr)
    dt = time.time() - t
    return r2(yte, m.predict(Xte)), dt, 0.0


def pysr(Xtr, ytr, Xte, yte, seed):
    from pysr import PySRRegressor
    m = PySRRegressor(binary_operators=list(BINARY), unary_operators=list(UNARY), maxsize=MAXSIZE,
                      niterations=10_000_000, timeout_in_seconds=BUDGET, verbosity=0,
                      progress=False, random_state=seed, temp_equation_file=True)
    t = time.time()
    m.fit(Xtr, ytr)
    dt = time.time() - t
    return r2(yte, m.predict(Xte)), dt, 0.0


def bench(label, fn, n, seeds=SEEDS, subset=None):
    r2s, ts, pks = [], [], []
    for s in range(seeds):
        Xtr, ytr, Xte, yte = make_data(n, s)
        if subset is not None:                       # subsampling isolation: SAME rows the engine sees
            rng = np.random.default_rng(s)
            idx = rng.permutation(len(Xtr))[:subset]
            Xtr, ytr = Xtr[idx], ytr[idx]
        try:
            rr, dt, pk = fn(Xtr, ytr, Xte, yte, s)
        except Exception as e:  # noqa: BLE001
            print(f"    {label} seed {s} FAILED: {type(e).__name__}: {str(e)[:90]}")
            rr, dt, pk = float("nan"), float("nan"), float("nan")
        r2s.append(rr)
        ts.append(dt)
        pks.append(pk)
    print(f"{label:>28} | {np.nanmedian(r2s):>10.5f} | {np.nanmedian(ts):>5.0f}s | {np.nanmedian(pks):>6.2f}")


def hdr(title):
    print(f"\n### {title} ###")
    print(f"{'method':>28} | {'test R2':>10} | {'sec':>6} | {'GB':>6}")
    print("-" * 60)


def main():
    print(f"target=x0^2+x0*x1+sin(x1)  budget={BUDGET}s  seeds={SEEDS}  pop(evozero)={POP}  "
          f"cuda={torch.cuda.is_available()}\nwarming up...", flush=True)
    for fn in (lambda a, b, c, d, s: evozero(a, b, c, d, s, "cpu", True), operon, pysr):
        try:
            X, y, Xt, yt = make_data(200, 42)
            fn(X, y, Xt, yt, 42)
        except Exception:  # noqa: BLE001
            pass

    hdr("A) CPU vs CPU — same hardware, FULL data")
    for n in (10_000, 100_000):
        print(f"  -- N = {n:,} --")
        bench("evozero(CPU, full)", lambda a, b, c, d, s: evozero(a, b, c, d, s, "cpu", False), n)
        bench("operon(CPU)", operon, n)
        bench("pysr(CPU)", pysr, n)

    hdr("B) evozero GPU vs CPU — subsampling ON (what the GPU buys)")
    for n in (100_000, 1_000_000):
        print(f"  -- N = {n:,} --")
        bench("evozero(GPU, sub 2048)", lambda a, b, c, d, s: evozero(a, b, c, d, s, "cuda", True), n)
        bench("evozero(CPU, sub 2048)", lambda a, b, c, d, s: evozero(a, b, c, d, s, "cpu", True), n)

    hdr("C) subsampling isolation @ N=1e6 — CPU tools given the SAME 2048 rows")
    n = 1_000_000
    bench("evozero(GPU, sub 2048)", lambda a, b, c, d, s: evozero(a, b, c, d, s, "cuda", True), n)
    bench("operon(CPU, 2048 rows)", operon, n, subset=SUB)
    bench("pysr(CPU, 2048 rows)", pysr, n, subset=SUB)
    print("\n(if operon/pysr on 2048 rows are also fast+accurate -> the speed edge is "
          "SUBSAMPLING, not the GPU: exactly the honest caveat, now measured.)")


if __name__ == "__main__":
    main()

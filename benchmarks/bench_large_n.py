#!/usr/bin/env python3
"""Large-N head-to-head on benchmark.py's EXACT target, WITH case subsampling active.
Confirms the OOM fix + gap-close on the canonical target vs Operon / PySR, and records
peak GPU memory. Writes measured numbers to substantiate the BENCHMARK.md † note.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JULIA_NUM_THREADS", "32")
sys.path.insert(0, "benchmarks")

import numpy as np
import torch
from benchmark import BINARY, MAXSIZE, UNARY, make_data, r2  # exact target/protocol

BUDGET = 60.0
SEEDS = 3
SIZES = [100_000, 1_000_000]


def run_evozero(Xtr, ytr, Xte, yte, seed, selection):
    from evozero import SymbolicRegressor
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = SymbolicRegressor(
        binary_operators=tuple(BINARY), unary_operators=tuple(UNARY),
        max_size=MAXSIZE, population_size=10000, generations=10_000_000,
        max_time=BUDGET, selection=selection, device="auto", random_state=seed,
    )
    t = time.time()
    m.fit(Xtr, ytr)
    dt = time.time() - t
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    return r2(yte, m.predict(Xte)), dt, peak


def run_operon(Xtr, ytr, Xte, yte, seed):
    from pyoperon.sklearn import SymbolicRegressor as OP
    m = OP(allowed_symbols="add,sub,mul,div,sin,cos,constant,variable",
           max_length=MAXSIZE, population_size=1000, generations=10_000_000,
           max_evaluations=1_000_000_000, max_time=int(BUDGET), n_threads=32, random_state=seed)
    t = time.time()
    m.fit(Xtr, ytr)
    dt = time.time() - t
    return r2(yte, m.predict(Xte)), dt, 0.0


def run_pysr(Xtr, ytr, Xte, yte, seed):
    from pysr import PySRRegressor
    m = PySRRegressor(binary_operators=list(BINARY), unary_operators=list(UNARY),
                      maxsize=MAXSIZE, niterations=10_000_000, timeout_in_seconds=BUDGET,
                      verbosity=0, progress=False, random_state=seed, temp_equation_file=True)
    t = time.time()
    m.fit(Xtr, ytr)
    dt = time.time() - t
    return r2(yte, m.predict(Xte)), dt, 0.0


METHODS = [
    ("evozero/tourn", lambda a, b, c, d, s: run_evozero(a, b, c, d, s, "tournament")),
    ("evozero/DALex", lambda a, b, c, d, s: run_evozero(a, b, c, d, s, "dalex")),
    ("operon", lambda a, b, c, d, s: run_operon(a, b, c, d, s)),
    ("pysr", lambda a, b, c, d, s: run_pysr(a, b, c, d, s)),
]


def warmup():
    Xtr, ytr, Xte, yte = make_data(200, 99999)
    for _, fn in METHODS:
        try:
            fn(Xtr, ytr, Xte, yte, 99999)
        except Exception:  # noqa: BLE001
            pass


def main():
    print(f"target=x0^2+x0*x1+sin(x1)  budget={BUDGET}s  seeds={SEEDS}  fp64 data  "
          f"subsampling AUTO (evozero)\nwarming up...", flush=True)
    warmup()
    for n in SIZES:
        print(f"\n### N = {n:,} ###")
        print(f"{'method':>16} | {'test R2 (med)':>13} | {'R2 worst':>9} | "
              f"{'sec':>5} | {'peak GB':>7}")
        print("-" * 62)
        for name, fn in METHODS:
            r2s, ts, pks = [], [], []
            for s in range(SEEDS):
                Xtr, ytr, Xte, yte = make_data(n, s)
                try:
                    rr, dt, pk = fn(Xtr, ytr, Xte, yte, s)
                except Exception as e:  # noqa: BLE001
                    print(f"    {name} seed {s} FAILED: {type(e).__name__}: {str(e)[:90]}")
                    rr, dt, pk = float("nan"), float("nan"), float("nan")
                r2s.append(rr)
                ts.append(dt)
                pks.append(pk)
            print(f"{name:>16} | {np.nanmedian(r2s):>13.5f} | "
                  f"{np.nanmin(r2s):>9.5f} | {np.nanmedian(ts):>5.0f} | {np.nanmedian(pks):>7.2f}")


if __name__ == "__main__":
    main()

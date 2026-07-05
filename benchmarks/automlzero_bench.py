#!/usr/bin/env python3
"""AutoML-Zero benchmark: is evozero's evolutionary search efficient, and do the
FEC / Hurdles upgrades help — measured HONESTLY on the training-steps axis?

This reuses evozero's AutoML-Zero VM primitives (Mem, run, random_algo, mutate,
make_tasks, ...) and implements the *search* variants on top so the vendored engine
stays untouched (same pattern as benchmarks/ab_selection.py).

Why training steps, not wall-clock or #evaluations (per Real et al. 2020, ICML):
  wall-clock is hardware-specific, and #evaluations is NOT a fixed unit of work once
  FEC skips full evals and Hurdles truncate them. The fair, hardware-agnostic budget is
  the total number of inner LEARN iterations executed. Every curve is best-fitness-so-far
  vs cumulative training steps, capped at the same step budget.

Searchers (identical VM search space + identical step budget):
  - random_search          : all-random algorithms, keep the best (no selection).
  - one_plus_one_es        : P=1, always-mutate, elitist accept (added baseline; NOT in
                             the paper — labeled as such).
  - aging_evolution        : regularized/aging evolution (evict OLDEST, tournament, U=0.9).
      vanilla  = aging with FEC off, Hurdles off (the honest ablation baseline).
      +FEC / +Hurdles / full = the upgrades, toggled ONE at a time for attribution.

Honesty notes baked in:
  - FEC charges its 10-step probe on EVERY candidate (hit AND miss); a hit then skips the
    full eval. FEC is therefore EXPECTED to shift the fitness-vs-steps curve left — we
    report full-evals-saved, we do not pretend the curve is unchanged.
  - The main curves run aging_evolution with FEC/Hurdles OFF, so vanilla vs RS vs (1+1)-ES
    differ only by the search operator. FEC/Hurdles appear only in their own ablation.
  - Cross-seed bands use a shared step grid with forward-filled best-so-far (event-driven
    histories are ragged otherwise).

Run on the GPU box:  python benchmarks/automlzero_bench.py
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch

from evozero.core import _automlzero_engine as az

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMS = (4, 4, 4)           # (nS, nV, nP), matches the facade's _DIMS
T_INNER = 30               # inner training steps per task
PROBE_STEPS = 10           # FEC fingerprint probe = 10 train steps (paper)
AMIN = -9.0                # degenerate-algorithm floor (matches eval_algo)


# ---------------------------------------------------------------------------
# Evaluation with training-step accounting (+ optional Hurdle early-stop)
# ---------------------------------------------------------------------------
def eval_with_steps(algo, tasks, T, dims, device, hurdle=None):
    """Mean R2 over `tasks`, plus the number of LEARN iterations actually executed.

    Mirrors az.eval_algo's semantics (clamp [-9,1], -9 on non-finite/exception) but
    counts training steps and, if `hurdle` is given, stops early once the running mean
    over the tasks seen so far drops below the hurdle (assigning that low partial score).
    """
    nS, nV, nP = dims
    r2s, steps = [], 0
    for i, (Xtr, ytr, Xval, yval) in enumerate(tasks):
        try:
            F = Xtr.shape[1]
            mem = az.Mem(nS, nV, nP, F, Xtr.shape[0], device)
            az.run(algo["setup"], mem, Xtr, ytr, nS, nV, nP)
            for _ in range(T):
                az.run(algo["predict"], mem, Xtr, ytr, nS, nV, nP)
                az.run(algo["learn"], mem, Xtr, ytr, nS, nV, nP)
                steps += 1
            memv = az.Mem(nS, nV, nP, F, Xval.shape[0], device)
            memv.S = mem.S.clone()
            memv.V = mem.V.clone()
            az.run(algo["predict"], memv, Xval, yval, nS, nV, nP)
            yhat = memv.P[az.OUT_P]
            if not torch.isfinite(yhat).all():
                r2s.append(AMIN)
            else:
                var = ((yval - yval.mean()) ** 2).mean() + 1e-12
                r2 = float(1.0 - ((yhat - yval) ** 2).mean() / var)
                r2s.append(max(AMIN, min(1.0, r2)))
        except Exception:  # noqa: BLE001
            r2s.append(AMIN)
        if hurdle is not None and i >= 1 and float(np.mean(r2s)) < hurdle:
            break                                     # too weak so far -> stop (paper's hurdle)
    return float(np.mean(r2s)), steps


def fingerprint(algo, probe, dims, device, trunc=4, ncols=8):
    """FEC behavioral fingerprint: run setup + PROBE_STEPS train steps on a FIXED small
    probe task, hash the truncated p0 PREDICTION VECTOR (first `ncols` examples) at each
    step. Using the vector, not just its mean, is what keeps distinct algorithms from
    colliding to one key (the failure mode a too-coarse fingerprint causes). Returns
    (fp_int, steps_charged); the probe is charged on EVERY call (hit AND miss)."""
    nS, nV, nP = dims
    Xtr, ytr, Xval, yval = probe
    vals: list[float] = []

    def snap(p):  # first ncols predictions, as plain floats
        return p[:ncols].detach().float().cpu().numpy().tolist()

    try:
        F = Xtr.shape[1]
        mem = az.Mem(nS, nV, nP, F, Xtr.shape[0], device)
        az.run(algo["setup"], mem, Xtr, ytr, nS, nV, nP)
        for _ in range(PROBE_STEPS):
            az.run(algo["predict"], mem, Xtr, ytr, nS, nV, nP)
            vals += snap(mem.P[az.OUT_P])
            az.run(algo["learn"], mem, Xtr, ytr, nS, nV, nP)
        memv = az.Mem(nS, nV, nP, F, Xval.shape[0], device)
        memv.S = mem.S.clone()
        memv.V = mem.V.clone()
        az.run(algo["predict"], memv, Xval, yval, nS, nV, nP)
        vals += snap(memv.P[az.OUT_P])
    except Exception:  # noqa: BLE001
        vals = [float("nan")]
    key = tuple(int(round(v, trunc) * 10 ** trunc) if np.isfinite(v) else 0 for v in vals)
    return hash(key), PROBE_STEPS


# ---------------------------------------------------------------------------
# History helper: record best-so-far vs cumulative training steps
# ---------------------------------------------------------------------------
class History:
    def __init__(self):
        self.steps, self.best = [], []
        self._b = -1e18

    def record(self, cum_steps, fitness):
        if fitness > self._b:
            self._b = fitness
        self.steps.append(cum_steps)
        self.best.append(self._b)


def _copy(algo):
    return {k: list(v) for k, v in algo.items()}


# ---------------------------------------------------------------------------
# Searchers (all cap at `budget` cumulative training steps)
# ---------------------------------------------------------------------------
def random_search(tasks, budget, seed):
    rng = np.random.default_rng(seed)
    h, cum, h.full_evals = History(), 0, 0
    while cum < budget:
        algo = az.random_algo(*DIMS, rng)
        fit, st = eval_with_steps(algo, tasks, T_INNER, DIMS, DEV)
        cum += st
        h.full_evals += 1
        h.record(cum, fit)
    return h


def one_plus_one_es(tasks, budget, seed):
    rng = np.random.default_rng(seed)
    h, cum, h.full_evals = History(), 0, 0
    inc = az.random_algo(*DIMS, rng)                # single random incumbent, hill-climb
    inc_fit, st = eval_with_steps(inc, tasks, T_INNER, DIMS, DEV)
    cum += st
    h.full_evals += 1
    h.record(cum, inc_fit)
    while cum < budget:
        child = az.mutate(inc, *DIMS, rng)          # P=1, always mutate
        fit, st = eval_with_steps(child, tasks, T_INNER, DIMS, DEV)
        cum += st
        h.full_evals += 1
        if fit >= inc_fit:                          # elitist accept
            inc, inc_fit = child, fit
        h.record(cum, inc_fit)
    return h


def aging_evolution(tasks, budget, seed, pop_size=100, tournament=10, u=0.9,
                    fec=False, hurdle=False, probe=None, trunc=4):
    """Regularized/aging evolution: evict the OLDEST, tournament-select the parent,
    copy, mutate with prob u, append. Optional FEC (cache) and Hurdles (rolling 75th
    pct of distinct pop fitnesses)."""
    rng = np.random.default_rng(seed)
    h, cum = History(), 0
    pop = deque(maxlen=pop_size)                    # (fitness, algo, age-order by insertion)
    cache: dict[int, float] = {}
    full_evals = fec_hits = 0

    def evaluate(algo, hurdle_thr):
        nonlocal cum, full_evals, fec_hits
        if fec and probe is not None:
            fp, pst = fingerprint(algo, probe, DIMS, DEV, trunc)
            cum += pst                              # probe charged on hit AND miss
            if fp in cache:
                fec_hits += 1
                return cache[fp]
            fit, st = eval_with_steps(algo, tasks, T_INNER, DIMS, DEV, hurdle_thr)
            cum += st
            full_evals += 1
            cache[fp] = fit
            return fit
        fit, st = eval_with_steps(algo, tasks, T_INNER, DIMS, DEV, hurdle_thr)
        cum += st
        full_evals += 1
        return fit

    while len(pop) < pop_size and cum < budget:     # seed the population (random, like the
        algo = az.random_algo(*DIMS, rng)           # shipping engine, so it captures the rare
        fit = evaluate(algo, None)                  # high-fitness structures RS also relies on)
        pop.append((fit, algo))
        h.record(cum, fit)

    while cum < budget:
        thr = None
        if hurdle and len(pop) >= 8:
            uniq = np.unique([f for f, _ in pop])
            thr = float(np.percentile(uniq, 75))    # rolling hurdle
        idx = rng.integers(0, len(pop), size=min(tournament, len(pop)))
        parent = max((pop[i] for i in idx), key=lambda fa: fa[0])[1]
        child = az.mutate(parent, *DIMS, rng) if rng.random() < u else _copy(parent)
        fit = evaluate(child, thr)
        pop.append((fit, child))                    # deque(maxlen) auto-evicts the oldest
        h.record(cum, fit)
    h.full_evals, h.fec_hits = full_evals, fec_hits
    return h


# ---------------------------------------------------------------------------
# Cross-seed aggregation on a shared step grid (forward-fill best-so-far)
# ---------------------------------------------------------------------------
def aggregate(histories, budget, points=40):
    grid = np.linspace(budget / points, budget, points)
    curves = []
    for h in histories:
        steps = np.asarray(h.steps)
        best = np.asarray(h.best)
        idx = np.searchsorted(steps, grid, side="right") - 1
        curves.append(np.where(idx >= 0, best[idx.clip(0)], AMIN))
    curves = np.asarray(curves)
    return grid, np.median(curves, axis=0), np.percentile(curves, 25, axis=0), \
        np.percentile(curves, 75, axis=0)


def final_at(histories, budget):
    _, med, lo, hi = aggregate(histories, budget)
    return med[-1], lo[-1], hi[-1]


# ---------------------------------------------------------------------------
def main():
    budget = 400_000       # total training steps per run (raise for a fuller sweep)
    seeds = 3
    tasks = az.to_device(az.make_tasks(5, F=4, N=200, seed=123, kind="relu"), DEV)
    probe = az.to_device(az.make_tasks(1, F=4, N=16, seed=999, kind="relu"), DEV)[0]

    print(f"\nAutoML-Zero search efficiency — budget={budget:,} training steps, "
          f"{seeds} seeds, 5 relu tasks, device={DEV}")
    print(f"{'searcher':>22} | {'best fit (median)':>17} | {'IQR':>15} | {'full evals':>10}")
    print("-" * 76)

    def run_many(fn, **kw):
        return [fn(tasks, budget, s, **kw) if kw else fn(tasks, budget, s) for s in range(seeds)]

    configs = [
        ("random_search", lambda: run_many(random_search)),
        ("(1+1)-ES [added]", lambda: run_many(one_plus_one_es)),
        ("aging (vanilla)", lambda: run_many(aging_evolution)),
    ]
    t0 = time.time()
    for name, fn in configs:
        hs = fn()
        med, lo, hi = final_at(hs, budget)
        fe = int(np.median([getattr(h, "full_evals", 0) for h in hs]))
        print(f"{name:>22} | {med:>17.4f} | [{lo:>6.4f},{hi:>6.4f}] | {fe:>10}")

    print("\nFEC / Hurdles ablation (aging evolution; one upgrade at a time):")
    print(f"{'config':>22} | {'best fit (median)':>17} | {'full evals':>10} | {'fec hits':>9}")
    print("-" * 72)
    abls = [
        ("vanilla", {"fec": False, "hurdle": False}),
        ("+FEC", {"fec": True, "hurdle": False, "probe": probe}),
        ("+Hurdles", {"fec": False, "hurdle": True}),
        ("full (FEC+Hurdles)", {"fec": True, "hurdle": True, "probe": probe}),
    ]
    for name, kw in abls:
        hs = [aging_evolution(tasks, budget, s, **kw) for s in range(seeds)]
        med, _, _ = final_at(hs, budget)
        fe = int(np.median([h.full_evals for h in hs]))
        fh = int(np.median([h.fec_hits for h in hs]))
        print(f"{name:>22} | {med:>17.4f} | {fe:>10} | {fh:>9}")

    # sanity: the hand-coded linreg-GD reference should score well on these tasks
    ref_fit, _ = eval_with_steps(az.handcoded_linreg_gd(), tasks, T_INNER, DIMS, DEV)
    print(f"\nreference handcoded_linreg_gd fitness = {ref_fit:.4f}  "
          f"(sanity: a real learner should beat most random algos)")
    print(f"total wall-clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

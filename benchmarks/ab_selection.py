#!/usr/bin/env python3
"""A/B experiment: does GPU-enabled lexicase (DALex) selection beat tournament?

Isolates ONLY the selection operator in a single-population GP that reuses evozero's
tensorized [P,N] interpreter. Same generations, same variation, same data — swap
tournament <-> DALex. Measures R²-vs-generation and wall-clock/generation (to check the
"lexicase is near-free on GPU" claim). This is the falsifiable test from the research
plan: if tournament ties DALex on R²-vs-generations, lexicase is NOT the SR lever.

Run on the GPU box:  python benchmarks/ab_selection.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from evozero.core import _sr_engine as E

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target(X, kind):
    if kind == "clean":  # no free constants
        return X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])
    return 2.5 * X[:, 0] ** 2 - 1.3 * X[:, 0] * X[:, 1] + 0.8 * np.sin(1.5 * X[:, 1]) + 0.4


def r2_np(yt, yp):
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    return 1 - ((yt - yp) ** 2).sum() / (((yt - yt.mean()) ** 2).sum() + 1e-12)


def dalex_parents(err, k, sigma, gen):
    """DALex: lexicase-as-a-matmul. err [P,N] per-case error (lower better) -> k parents."""
    n = err.shape[1]
    w = torch.softmax(torch.randn(k, n, generator=gen, device=err.device) * sigma, dim=1)  # [k,N]
    return (err @ w.t()).argmin(dim=0)  # [k] best program per weighted event


def tournament_parents(mse, k, tsize, gen):
    p = mse.shape[0]
    idx = torch.randint(0, p, (k, tsize), generator=gen, device=mse.device)
    return idx.gather(1, mse[idx].argmin(dim=1, keepdim=True)).squeeze(1)  # [k]


def run_gp(select, kind, n, generations, pop_size, seed, sigma=1.0, elite=8, tsize=6):
    rng = np.random.default_rng(seed)
    gen_t = torch.Generator(device=DEV).manual_seed(seed)
    ps = E.PrimSet(2, ["sin", "cos"], ["add", "sub", "mul"], named=[])
    Xtr = rng.uniform(-3, 3, size=(n, 2))
    Xte = rng.uniform(-3, 3, size=(2000, 2))
    ytr = target(Xtr, kind)
    yte = target(Xte, kind)
    Xtr_t = torch.tensor(Xtr.T, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)

    pop = []
    while len(pop) < pop_size:
        c, k = E.gen_tree(ps, int(rng.integers(2, 5)), "grow" if rng.random() < 0.5 else "full", rng)
        c, k = E.simplify_prefix(c, k, ps)
        if E._ok_size(c, ps, 40, 9):
            pop.append((c, k))

    best_ind, best_r2_tr = None, -1e9
    curve = {}
    t0 = time.time()
    for g in range(generations):
        codes, consts, _ = E.batch_postfix(pop, ps)
        yhat = E.run_population(codes, consts, Xtr_t, ps, DEV)  # [P, N]
        a, b, mse, r2 = E.fit_ab(yhat, ytr_t)
        gi = int(torch.argmax(r2))
        if float(r2[gi]) > best_r2_tr:
            best_r2_tr = float(r2[gi])
            best_ind = (pop[gi][0], pop[gi][1], float(a[gi]), float(b[gi]))
        curve[g] = float(r2[gi])

        need = pop_size - elite
        if select == "dalex":
            err = (a[:, None] * yhat + b[:, None] - ytr_t[None, :]) ** 2  # [P, N]
            parents = dalex_parents(err, 2 * need, sigma, gen_t).cpu().numpy()
        else:
            parents = tournament_parents(mse, 2 * need, tsize, gen_t).cpu().numpy()

        order = torch.argsort(mse).cpu().numpy()
        new_pop = [pop[i] for i in order[:elite]]
        pi = 0
        while len(new_pop) < pop_size:
            p1 = pop[int(parents[pi % len(parents)])]; pi += 1
            if rng.random() < 0.7:
                p2 = pop[int(parents[pi % len(parents)])]; pi += 1
                child = E.crossover(p1, p2, ps, 40, 9, rng)
            else:
                child = p1
            r = rng.random()
            if r < 0.4:
                child = E.mut_subtree(child, ps, 40, 9, rng)
            elif r < 0.7:
                child = E.mut_point(child, ps, rng)
            elif r < 0.9:
                child = E.mut_const(child, ps, rng)
            child = E.simplify_prefix(list(child[0]), list(child[1]), ps)
            if not E._ok_size(child[0], ps, 40, 9):
                child = (child[0][:40], child[1][:40])
            new_pop.append((list(child[0]), list(child[1])))
        pop = new_pop

    elapsed = time.time() - t0
    # test R2 of the best-on-train individual
    cc, kk, aa, bb = best_ind
    codes, consts, _ = E.batch_postfix([(cc, kk)], ps)
    Xte_t = torch.tensor(Xte.T, dtype=torch.float32, device=DEV)
    yp = (aa * E.run_population(codes, consts, Xte_t, ps, DEV)[0] + bb).cpu().numpy()
    return {"curve": curve, "test_r2": r2_np(yte, yp), "s_per_gen": elapsed / generations}


def main():
    GENS, POP, SEEDS = 40, 2000, 3
    for kind in ("clean", "constants"):
        print(f"\n===== problema: {kind}  (pop={POP}, gens={GENS}, N=3000, {SEEDS} seeds) =====")
        print(f"{'selector':>16} | {'R2@gen10':>9} | {'R2@gen20':>9} | {'R2@gen40(test)':>15} | {'ms/gen':>7}")
        print("-" * 72)
        for sel, label in [("tournament", "tournament"), ("dalex", "DALex σ=1"),
                           ("dalex3", "DALex σ=3")]:
            g10, g20, tr2, spg = [], [], [], []
            for s in range(SEEDS):
                sigma = 3.0 if sel == "dalex3" else 1.0
                res = run_gp("dalex" if sel.startswith("dalex") else sel, kind, 3000,
                             GENS, POP, s, sigma=sigma)
                g10.append(res["curve"][10]); g20.append(res["curve"][20])
                tr2.append(res["test_r2"]); spg.append(res["s_per_gen"])
            print(f"{label:>16} | {np.median(g10):>9.4f} | {np.median(g20):>9.4f} | "
                  f"{np.median(tr2):>15.4f} | {1000 * np.median(spg):>7.1f}")


if __name__ == "__main__":
    main()

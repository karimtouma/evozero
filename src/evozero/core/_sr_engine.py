#!/usr/bin/env python3
"""
symreg_gpu.py — Motor de regresion simbolica evolutiva acelerado por GPU.

Diseno (patron hibrido recomendado en research.md):
  * BUCLE EVOLUTIVO en CPU (barato): individuos = arboles prefijos;
    seleccion/crossover/mutacion de SUBARBOLES siempre producen arboles validos.
  * EVALUACION DE FITNESS de TODA la poblacion en GPU (caro): interprete
    POSTFIJO TENSORIZADO, pila [profundidad, P, N] en VRAM.
  * LINEAR SCALING (Keijzer) en forma cerrada; a,b se ajustan SOLO en train.

Robustez / expresividad:
  * Split TRAIN/VAL/TEST; el modelo se selecciona por VALIDACION y se reporta TEST.
  * TERMINALES ricos: constantes con nombre (pi, e) como terminales reales +
    pool de constantes simples sesgando las efimeras.
  * EQUIVALENCIAS: simplificacion algebraica estructural cada generacion
    (plegado de constantes + reglas de identidad que RESPETAN la semantica
    protegida de los operadores). Verificada por test de propiedad.
  * FITNESS con COMPLEJIDAD PONDERADA (coste por operador, estilo MDL) y
    COHERENCIA MATEMATICA (penaliza anidamiento de transcendentes, gap
    train-validacion, e inconsistencia dimensional).
  * Export a funcion NumPy, LaTeX y frente de Pareto CSV.

Licencia del codigo: tuya (permisiva). Dependencias: torch (BSD), numpy, sympy.
"""
import argparse
import csv
import json
import math
import os
import time
import numpy as np
import torch

# ----------------------------------------------------------------------------
# 1. CONJUNTO DE PRIMITIVAS
# ----------------------------------------------------------------------------
UNARY_ALL = ["sin", "cos", "exp", "log", "sqrt", "neg", "square", "cube", "tanh", "abs", "inv"]
BINARY_ALL = ["add", "sub", "mul", "div", "aq"]
NAMED_VALUES = {"pi": math.pi, "e": math.e}
TRANSCENDENTAL = {"sin", "cos", "exp", "log", "tanh"}
# coste por operador para la complejidad ponderada (mas alto = mas fragil/costoso)
OP_COST = {"add": 1, "sub": 1, "mul": 2, "div": 3, "aq": 3,
           "neg": 1, "abs": 1, "square": 2, "cube": 3, "sqrt": 3,
           "sin": 4, "cos": 4, "tanh": 4, "exp": 5, "log": 5, "inv": 3}
CONST_POOL = [1.0, 2.0, 3.0, 5.0, 10.0, 0.5, -1.0, -2.0, 0.1, 0.25]


class PrimSet:
    def __init__(self, n_vars, unary, binary, named=("pi", "e"),
                 const_low=-3.0, const_high=3.0):
        self.n_vars = n_vars
        self.unary = list(unary)
        self.binary = list(binary)
        self.named = [nm for nm in named if nm in NAMED_VALUES]
        self.const_low = const_low
        self.const_high = const_high

        self.PAD = 0
        self.var_codes = list(range(1, n_vars + 1))
        self.CONST = n_vars + 1
        base = n_vars + 2
        self.named_codes = {nm: base + i for i, nm in enumerate(self.named)}   # nombre->codigo
        base = base + len(self.named)
        self.ucode = {nm: base + i for i, nm in enumerate(self.unary)}
        base2 = base + len(self.unary)
        self.bcode = {nm: base2 + i for i, nm in enumerate(self.binary)}
        self.num_codes = base2 + len(self.binary)

        arity = np.zeros(self.num_codes, dtype=np.int64)
        for c in self.ucode.values():
            arity[c] = 1
        for c in self.bcode.values():
            arity[c] = 2
        self.arity = arity

        # tablas auxiliares indexadas por codigo
        self.code2name = {}
        for nm, c in self.ucode.items():
            self.code2name[c] = nm
        for nm, c in self.bcode.items():
            self.code2name[c] = nm
        self.named_vals = {c: NAMED_VALUES[nm] for nm, c in self.named_codes.items()}  # codigo->valor
        self.named_sym = {c: nm for nm, c in self.named_codes.items()}                 # codigo->'pi'/'e'
        self.op_cost = np.ones(self.num_codes, dtype=np.float64)
        for nm, c in list(self.ucode.items()) + list(self.bcode.items()):
            self.op_cost[c] = OP_COST.get(nm, 2)
        self.trans_codes = set(c for nm, c in list(self.ucode.items()) if nm in TRANSCENDENTAL)

    def terminals(self):
        return self.var_codes + [self.CONST] + list(self.named_codes.values())

    def is_const_terminal(self, c):
        return c == self.CONST or c in self.named_vals


# ----------------------------------------------------------------------------
# 2. GENERACION Y OPERADORES GENETICOS (prefijo, CPU)
# ----------------------------------------------------------------------------
def subtree_end(code, i, arity):
    need, j, n = 1, i, len(code)
    while need > 0 and j < n:
        need += arity[code[j]] - 1
        j += 1
    return j


def tree_depth(code, arity):
    def rec(i):
        c = code[i]
        nxt = i + 1
        a = int(arity[c])
        if a == 0:
            return 1, nxt
        best = 0
        for _ in range(a):
            d, nxt = rec(nxt)
            best = max(best, d)
        return best + 1, nxt
    return rec(0)[0]


def _sample_const(ps, rng):
    if rng.random() < 0.5:
        return float(CONST_POOL[rng.integers(len(CONST_POOL))])
    return float(rng.uniform(ps.const_low, ps.const_high))


def gen_tree(ps, max_depth, method, rng, p_term=0.3):
    code, const = [], []
    terms = ps.terminals()
    funcs = list(ps.ucode.values()) + list(ps.bcode.values())

    def add_terminal():
        c = terms[rng.integers(len(terms))]
        code.append(int(c))
        const.append(_sample_const(ps, rng) if c == ps.CONST else 0.0)

    def rec(depth):
        if (depth <= 0) or (method == "grow" and rng.random() < p_term) or not funcs:
            add_terminal()
            return
        f = funcs[rng.integers(len(funcs))]
        code.append(int(f))
        const.append(0.0)
        for _ in range(int(ps.arity[f])):
            rec(depth - 1)

    rec(max_depth)
    return code, const


def _ok_size(code, ps, max_len, max_depth):
    return len(code) <= max_len and tree_depth(code, ps.arity) <= max_depth


def crossover(A, B, ps, max_len, max_depth, rng, tries=12):
    ac, ak = A
    bc, bk = B
    for _ in range(tries):
        i = int(rng.integers(len(ac)))
        ei = subtree_end(ac, i, ps.arity)
        j = int(rng.integers(len(bc)))
        ej = subtree_end(bc, j, ps.arity)
        nc = ac[:i] + bc[j:ej] + ac[ei:]
        if _ok_size(nc, ps, max_len, max_depth):
            return nc, ak[:i] + bk[j:ej] + ak[ei:]
    return list(ac), list(ak)


def mut_subtree(ind, ps, max_len, max_depth, rng, sub_depth=3):
    code, const = ind
    for _ in range(6):
        i = int(rng.integers(len(code)))
        ei = subtree_end(code, i, ps.arity)
        sc, sk = gen_tree(ps, int(rng.integers(1, sub_depth + 1)), "grow", rng)
        nc = code[:i] + sc + code[ei:]
        if _ok_size(nc, ps, max_len, max_depth):
            return nc, const[:i] + sk + const[ei:]
    return list(code), list(const)


def mut_point(ind, ps, rng):
    code, const = list(ind[0]), list(ind[1])
    i = int(rng.integers(len(code)))
    a = ps.arity[code[i]]
    if a == 0:
        terms = ps.terminals()
        c = terms[rng.integers(len(terms))]
        code[i] = int(c)
        const[i] = _sample_const(ps, rng) if c == ps.CONST else 0.0
    elif a == 1:
        u = list(ps.ucode.values())
        code[i] = int(u[rng.integers(len(u))])
    else:
        b = list(ps.bcode.values())
        code[i] = int(b[rng.integers(len(b))])
    return code, const


def mut_const(ind, ps, rng, sigma=0.5):
    code, const = list(ind[0]), list(ind[1])
    for i, c in enumerate(code):
        if c == ps.CONST:
            const[i] += float(rng.normal(0, sigma))
    return code, const


def mut_hoist(ind, ps, rng):
    code, const = ind
    if len(code) <= 1:
        return list(code), list(const)
    i = int(rng.integers(len(code)))
    ei = subtree_end(code, i, ps.arity)
    return code[i:ei], const[i:ei]


# ----------------------------------------------------------------------------
# 2b. SIMPLIFICACION ALGEBRAICA (equivalencias) — respeta semantica protegida
#     Se opera sobre un arbol anidado (code, value, [hijos]) y se vuelve a prefijo.
# ----------------------------------------------------------------------------
def _parse_tree(code, const, arity):
    def rec(i):
        c = code[i]
        v = const[i]
        nxt = i + 1
        ch = []
        for _ in range(int(arity[c])):
            node, nxt = rec(nxt)
            ch.append(node)
        return [c, v, ch], nxt
    return rec(0)[0]


def _flatten(node, out_c, out_k):
    c, v, ch = node
    out_c.append(int(c))
    out_k.append(float(v))
    for x in ch:
        _flatten(x, out_c, out_k)


def _same(a, b):
    if a[0] != b[0] or len(a[2]) != len(b[2]):
        return False
    if not a[2] and a[1] != b[1]:
        return False
    return all(_same(x, y) for x, y in zip(a[2], b[2]))


def _clampf(x):
    if not math.isfinite(x):
        return 0.0 if math.isnan(x) else (1e6 if x > 0 else -1e6)
    return max(-1e6, min(1e6, x))


def _fold_unary(name, x):
    if name == "sin": return _clampf(math.sin(x))
    if name == "cos": return _clampf(math.cos(x))
    if name == "exp": return _clampf(math.exp(max(-20.0, min(20.0, x))))
    if name == "log": return _clampf(math.log(abs(x) + 1e-6))
    if name == "sqrt": return _clampf(math.sqrt(abs(x)))
    if name == "neg": return _clampf(-x)
    if name == "square": return _clampf(x * x)
    if name == "cube": return _clampf(x * x * x)
    if name == "tanh": return _clampf(math.tanh(x))
    if name == "abs": return _clampf(abs(x))
    if name == "inv": return _clampf(1.0 / x) if abs(x) > 1e-6 else 1.0
    raise ValueError(name)


def _fold_binary(name, a, b):
    if name == "add": return _clampf(a + b)
    if name == "sub": return _clampf(a - b)
    if name == "mul": return _clampf(a * b)
    if name == "div": return _clampf(a / b) if abs(b) > 1e-6 else 1.0
    if name == "aq": return _clampf(a / math.sqrt(1.0 + b * b))
    raise ValueError(name)


def _const_val(node, ps):
    c = node[0]
    if c == ps.CONST:
        return node[1]
    if c in ps.named_vals:
        return ps.named_vals[c]
    return None


def simplify_node(node, ps):
    c, v, ch = node
    a = int(ps.arity[c])
    if a == 0:
        return node
    ch = [simplify_node(x, ps) for x in ch]
    node = [c, v, ch]
    name = ps.code2name.get(c)

    # plegado de constantes SOLO entre efimeras (preserva pi/e simbolicos)
    if all(x[0] == ps.CONST for x in ch):
        if a == 1:
            return [ps.CONST, _fold_unary(name, ch[0][1]), []]
        return [ps.CONST, _fold_binary(name, ch[0][1], ch[1][1]), []]

    def is_c(x, val):
        cv = _const_val(x, ps)
        return cv is not None and abs(cv - val) < 1e-12

    # reglas de identidad (validadas por semantica protegida y por selftest)
    if name == "add":
        if is_c(ch[0], 0.0): return ch[1]
        if is_c(ch[1], 0.0): return ch[0]
    elif name == "sub":
        if is_c(ch[1], 0.0): return ch[0]
        if _same(ch[0], ch[1]): return [ps.CONST, 0.0, []]
    elif name == "mul":
        if is_c(ch[0], 1.0): return ch[1]
        if is_c(ch[1], 1.0): return ch[0]
        if is_c(ch[0], 0.0) or is_c(ch[1], 0.0): return [ps.CONST, 0.0, []]
    elif name == "div":
        if is_c(ch[1], 1.0): return ch[0]
        if is_c(ch[0], 0.0): return [ps.CONST, 0.0, []]
        if _same(ch[0], ch[1]): return [ps.CONST, 1.0, []]  # div protegida -> 1 en todo el dominio
    elif name == "neg":
        if ch[0][0] == ps.ucode.get("neg"): return ch[0][2][0]      # neg(neg(x)) -> x
    elif name == "sqrt":
        if ch[0][0] == ps.ucode.get("square") and "abs" in ps.ucode:  # sqrt(square(x)) -> abs(x)
            return [ps.ucode["abs"], 0.0, [ch[0][2][0]]]
    elif name == "square":
        if ch[0][0] == ps.ucode.get("sqrt") and "abs" in ps.ucode:    # square(sqrt(x)) -> abs(x)
            return [ps.ucode["abs"], 0.0, [ch[0][2][0]]]
    elif name == "abs":
        if ch[0][0] == ps.ucode.get("abs"): return ch[0]              # abs(abs(x)) -> abs(x)
    return node


def simplify_prefix(code, const, ps):
    node = _parse_tree(code, const, ps.arity)
    node = simplify_node(node, ps)
    oc, ok = [], []
    _flatten(node, oc, ok)
    return oc, ok


# ----------------------------------------------------------------------------
# 2c. METRICAS DE COMPLEJIDAD / COHERENCIA (para el fitness)
# ----------------------------------------------------------------------------
def weighted_complexity(code, ps):
    return float(ps.op_cost[np.asarray(code)].sum())


def nested_transcendental_count(code, ps):
    trans = ps.trans_codes
    arity = ps.arity
    cnt = [0]

    def rec(i, anc):
        c = code[i]
        nxt = i + 1
        is_t = c in trans
        if is_t and anc:
            cnt[0] += 1
        for _ in range(int(arity[c])):
            nxt = rec(nxt, anc or is_t)
        return nxt
    rec(0, False)
    return cnt[0]


# ----------------------------------------------------------------------------
# 3. PREFIJO -> POSTFIJO
# ----------------------------------------------------------------------------
def prefix_to_postfix(code, const, arity):
    out_c, out_k = [], []

    def rec(i):
        c = code[i]
        nxt = i + 1
        for _ in range(int(arity[c])):
            nxt = rec(nxt)
        out_c.append(c)
        out_k.append(const[i])
        return nxt
    rec(0)
    return out_c, out_k


def batch_postfix(pop, ps):
    posts = [prefix_to_postfix(c, k, ps.arity) for (c, k) in pop]
    L = max(len(pc) for pc, _ in posts)
    P = len(posts)
    codes = np.zeros((P, L), dtype=np.int64)
    consts = np.zeros((P, L), dtype=np.float32)
    for p, (pc, pk) in enumerate(posts):
        codes[p, : len(pc)] = pc
        consts[p, : len(pk)] = pk
    return codes, consts, L


# ----------------------------------------------------------------------------
# 4. INTERPRETE POSTFIJO TENSORIZADO EN GPU
# ----------------------------------------------------------------------------
def _apply_unary(name, x):
    if name == "sin": return torch.sin(x)
    if name == "cos": return torch.cos(x)
    if name == "exp": return torch.exp(torch.clamp(x, -20.0, 20.0))
    if name == "log": return torch.log(torch.abs(x) + 1e-6)
    if name == "sqrt": return torch.sqrt(torch.abs(x))
    if name == "neg": return -x
    if name == "square": return x * x
    if name == "cube": return x * x * x
    if name == "tanh": return torch.tanh(x)
    if name == "abs": return torch.abs(x)
    if name == "inv": return torch.where(x.abs() > 1e-6, 1.0 / x, torch.ones_like(x))
    raise ValueError(name)


def _apply_binary(name, a, b):
    if name == "add": return a + b
    if name == "sub": return a - b
    if name == "mul": return a * b
    if name == "div": return torch.where(b.abs() > 1e-6, a / b, torch.ones_like(a))
    if name == "aq": return a / torch.sqrt(1.0 + b * b)
    raise ValueError(name)


@torch.no_grad()
def run_population(codes, consts, X, ps, device, chunk=512):
    """Interprete postfijo tensorizado: devuelve yhat [P, N] en GPU."""
    P, L = codes.shape
    N = X.shape[1]
    D = L // 2 + 2
    # auto-acota el chunk para que la pila [D, chunk, N] quepa (evita OOM con N grande)
    _budget = 1_500_000_000  # ~1.5 GB de working set para la pila
    chunk = max(1, min(chunk, _budget // (max(D, 1) * max(N, 1) * 4)))
    arity_t = torch.tensor(ps.arity, dtype=torch.long, device=device)
    yhat = torch.empty((P, N), dtype=torch.float32, device=device)
    codes_t_full = torch.from_numpy(codes).to(device)
    consts_t_full = torch.from_numpy(consts).to(device)
    n_vars = ps.n_vars
    CONST = ps.CONST
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        pc = codes_t_full[s:e]
        kc = consts_t_full[s:e]
        C = e - s
        S = torch.zeros((D, C, N), dtype=torch.float32, device=device)
        sp = torch.zeros(C, dtype=torch.long, device=device)
        prog = torch.arange(C, device=device)
        for t in range(L):
            tok = pc[:, t]
            cval = kc[:, t]
            ar = arity_t[tok]
            top1 = S[(sp - 1).clamp(min=0), prog]
            top2 = S[(sp - 2).clamp(min=0), prog]
            var_idx = (tok - 1).clamp(min=0, max=max(n_vars - 1, 0))
            push_val = X[var_idx]
            push_val = torch.where((tok == CONST).unsqueeze(1),
                                   cval.unsqueeze(1).expand(-1, N), push_val)
            for code_nc, val_nc in ps.named_vals.items():     # constantes con nombre (pi, e)
                push_val = torch.where((tok == code_nc).unsqueeze(1),
                                       torch.full_like(push_val, val_nc), push_val)
            res = push_val
            for name, c in ps.ucode.items():
                res = torch.where((tok == c).unsqueeze(1), _apply_unary(name, top1), res)
            for name, c in ps.bcode.items():
                res = torch.where((tok == c).unsqueeze(1), _apply_binary(name, top2, top1), res)
            res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            target = torch.where(ar == 0, sp,
                     torch.where(ar == 1, (sp - 1).clamp(min=0), (sp - 2).clamp(min=0)))
            S[target, prog] = res
            is_pad = tok == ps.PAD
            delta = torch.where(is_pad, torch.zeros_like(sp),
                    torch.where(ar == 0, torch.ones_like(sp),
                    torch.where(ar == 1, torch.zeros_like(sp), -torch.ones_like(sp))))
            sp = (sp + delta).clamp(0, D - 1)
        yhat[s:e] = S[(sp - 1).clamp(min=0), prog]
    return yhat


@torch.no_grad()
def run_population_reduce(codes, consts, X, y, ps, device, a=None, b=None, chunk_n=None):
    """Memory-safe fitness over FULL data: tiles the N (case) axis and folds each
    yhat tile into streaming reductions, so the [P, N] prediction tensor is never
    materialized (this is the fix for the large-N OOM). Reuses the validated
    ``run_population`` interpreter per tile, so the interpreter itself is untouched.

    ``a, b is None`` -> FIT mode: returns the Keijzer linear-scaling (a, b) fit on the
    full data (closed form). Otherwise SCORE mode: scores with the supplied (a, b).
    Returns ``(a, b, mse, r2)`` as GPU float32 ``[P]`` tensors, matching fit_ab/score_ab.

    Numerics: raw moments and residuals accumulate in float64 (predictions are clamped
    to +/-1e6 -> f^2 up to 1e12); MSE uses direct residual accumulation
    ``sum((a*f + b - y)^2)`` (not the expanded-moment form) to avoid catastrophic
    cancellation, and is guarded exactly like score_ab (clamp >= 0, non-finite -> 1e18).
    """
    P = codes.shape[0]
    N = X.shape[1]
    if chunk_n is None:
        # bound the returned [P, chunk_n] fp32 tile to ~2 GB (run_population still chunks
        # P internally for its own interpreter stack; this only caps the tile it returns).
        chunk_n = max(1, min(N, 2_000_000_000 // (max(P, 1) * 4)))
    y = y.to(torch.float64)
    n = float(N)
    Sy = y.sum()
    Syy = (y * y).sum()
    ybar = Sy / n
    vary = Syy / n - ybar * ybar + 1e-12

    if a is None or b is None:                       # FIT mode: pass 1 = raw moments
        Sf = torch.zeros(P, dtype=torch.float64, device=device)
        Sff = torch.zeros(P, dtype=torch.float64, device=device)
        Sfy = torch.zeros(P, dtype=torch.float64, device=device)
        for s in range(0, N, chunk_n):
            e = min(N, s + chunk_n)
            f = run_population(codes, consts, X[:, s:e], ps, device).to(torch.float64)
            Sf += f.sum(dim=1)
            Sff += (f * f).sum(dim=1)
            Sfy += (f * y[s:e].unsqueeze(0)).sum(dim=1)
            del f
        fbar = Sf / n
        Sxx = Sff - Sf * Sf / n
        Sxy = Sfy - Sf * Sy / n
        ok = Sxx > n * 1e-12                          # == fit_ab's var = Sxx/n > 1e-12
        a = torch.where(ok, Sxy / Sxx, torch.zeros_like(Sxx))
        b = ybar - a * fbar
    else:
        a = a.to(torch.float64)
        b = b.to(torch.float64)

    SSE = torch.zeros(P, dtype=torch.float64, device=device)   # residual-accumulation pass
    for s in range(0, N, chunk_n):
        e = min(N, s + chunk_n)
        f = run_population(codes, consts, X[:, s:e], ps, device).to(torch.float64)
        res = a.unsqueeze(1) * f + b.unsqueeze(1) - y[s:e].unsqueeze(0)
        SSE += (res * res).sum(dim=1)
        del f, res
    mse = (SSE / n).clamp_min(0.0)
    mse = torch.where(torch.isfinite(mse), mse, torch.full_like(mse, 1e18))
    r2 = 1.0 - mse / vary
    return a.to(torch.float32), b.to(torch.float32), mse.to(torch.float32), r2.to(torch.float32)


@torch.no_grad()
def fit_ab(yhat, y):
    ybar = y.mean()
    yc = y - ybar
    fbar = yhat.mean(dim=1, keepdim=True)
    fc = yhat - fbar
    cov = (fc * yc.unsqueeze(0)).mean(dim=1)
    var = (fc * fc).mean(dim=1)
    a = torch.where(var > 1e-12, cov / var, torch.zeros_like(cov))
    b = ybar - a * fbar.squeeze(1)
    mse, r2 = score_ab(yhat, y, a, b)
    return a, b, mse, r2


@torch.no_grad()
def score_ab(yhat, y, a, b):
    vary = ((y - y.mean()) ** 2).mean() + 1e-12
    scaled = a.unsqueeze(1) * yhat + b.unsqueeze(1)
    mse = ((scaled - y.unsqueeze(0)) ** 2).mean(dim=1)
    mse = torch.where(torch.isfinite(mse), mse, torch.full_like(mse, 1e18))
    return mse, 1.0 - mse / vary


# ----------------------------------------------------------------------------
# 4c. ANALISIS DIMENSIONAL OPCIONAL (estilo PhySO, "lite")
# ----------------------------------------------------------------------------
def units_consistent(code, ps, var_units, ndim, tol=1e-6):
    def rec(i):
        c = code[i]
        nxt = i + 1
        if c == ps.CONST or c in ps.named_vals:
            return True, np.zeros(ndim), nxt
        if 1 <= c <= ps.n_vars:
            return True, var_units[c - 1].astype(float).copy(), nxt
        name = ps.code2name[c]
        us, ok = [], True
        for _ in range(int(ps.arity[c])):
            o, u, nxt = rec(nxt)
            ok = ok and o
            us.append(u)
        if not ok:
            return False, None, nxt
        if name in ("add", "sub"):
            return (np.max(np.abs(us[0] - us[1])) <= tol, us[0], nxt)
        if name == "mul": return True, us[0] + us[1], nxt
        if name == "div": return True, us[0] - us[1], nxt
        if name == "aq":
            return (np.max(np.abs(us[1])) <= tol, us[0], nxt)
        if name in ("neg", "abs"): return True, us[0], nxt
        if name == "square": return True, 2 * us[0], nxt
        if name == "cube": return True, 3 * us[0], nxt
        if name == "sqrt": return True, 0.5 * us[0], nxt
        if name == "inv": return True, -us[0], nxt
        if name in ("sin", "cos", "exp", "log", "tanh"):
            return (np.max(np.abs(us[0])) <= tol, np.zeros(ndim), nxt)
        return True, us[0], nxt
    ok, _, _ = rec(0)
    return ok


# ----------------------------------------------------------------------------
# 5. PARETO Y RECONSTRUCCION SIMBOLICA
# ----------------------------------------------------------------------------
def pareto_update(archive, cand, err_key="mse_val"):
    seen = {}
    for d in archive + cand:
        key = (d["complexity"], round(float(d[err_key]), 10))
        seen.setdefault(key, d)
    items = list(seen.values())
    nd = []
    for d in items:
        if not any(o is not d and o["complexity"] <= d["complexity"] and o[err_key] <= d[err_key]
                   and (o["complexity"] < d["complexity"] or o[err_key] < d[err_key])
                   for o in items):
            nd.append(d)
    nd.sort(key=lambda d: d["complexity"])
    return nd


def round_sig(v, sig=4):
    if v == 0 or not math.isfinite(v):
        return 0.0
    return round(v, -int(math.floor(math.log10(abs(v)))) + (sig - 1))


def to_sympy(code, const, ps, a=1.0, b=0.0, simplify=True):
    import sympy as sp
    xs = sp.symbols(f"x0:{ps.n_vars}")
    named = {"pi": sp.pi, "e": sp.E}
    funcs = {
        "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
        "log": lambda z: sp.log(sp.Abs(z) + sp.Rational(1, 1000000)),
        "sqrt": lambda z: sp.sqrt(sp.Abs(z)),
        "neg": lambda z: -z, "square": lambda z: z ** 2, "cube": lambda z: z ** 3,
        "tanh": sp.tanh, "abs": sp.Abs, "inv": lambda z: 1 / z,
        "add": lambda u, v: u + v, "sub": lambda u, v: u - v,
        "mul": lambda u, v: u * v, "div": lambda u, v: u / v,
        "aq": lambda u, v: u / sp.sqrt(1 + v ** 2),
    }

    def rec(i):
        c = code[i]
        nxt = i + 1
        if c == ps.CONST:
            return sp.Float(round_sig(const[i], 4)), nxt
        if c in ps.named_sym:
            return named[ps.named_sym[c]], nxt
        if 1 <= c <= ps.n_vars:
            return xs[c - 1], nxt
        name = ps.code2name[c]
        args = []
        for _ in range(int(ps.arity[c])):
            arg, nxt = rec(nxt)
            args.append(arg)
        return funcs[name](*args), nxt

    expr, _ = rec(0)
    full = sp.Float(round_sig(a, 4)) * expr + sp.Float(round_sig(b, 4))
    if simplify:
        try:
            full = sp.simplify(full)
        except Exception:
            pass
    return full


# ----------------------------------------------------------------------------
# 6. BUCLE EVOLUTIVO
# ----------------------------------------------------------------------------
def optimize_constants(inds, ps, Xtr_t, ytr_t, device, rng, trials=48, rounds=3, sigma0=0.5):
    """Afina las constantes efimeras de cada individuo por ES paralelo en GPU:
    perturbar -> evaluar TODO el lote a la vez -> quedarse con la mejor variante.
    Sin autodiff (robusto). Sube el R2 de una estructura dada ajustando sus constantes."""
    out = [(list(c), list(k)) for c, k in inds]
    for r in range(rounds):
        sigma = sigma0 * (0.35 ** r)
        batch, owners = [], []
        for i, (c, k) in enumerate(out):
            cpos = [j for j, cc in enumerate(c) if cc == ps.CONST]
            batch.append((c, k)); owners.append(i)                 # variante original
            if not cpos:
                continue
            for _ in range(trials):
                k2 = list(k)
                for j in cpos:
                    k2[j] += float(rng.normal(0.0, sigma * (abs(k2[j]) + 1.0)))
                batch.append((c, k2)); owners.append(i)
        codes, consts, _ = batch_postfix(batch, ps)
        yhat = run_population(codes, consts, Xtr_t, ps, device)
        _, _, mse, _ = fit_ab(yhat, ytr_t)
        mse = mse.cpu().numpy()
        bestpos = {}
        for bi, i in enumerate(owners):
            if i not in bestpos or mse[bi] < mse[bestpos[i]]:
                bestpos[i] = bi
        out = [(batch[bestpos[i]][0], batch[bestpos[i]][1]) for i in range(len(out))]
    return out


def _eval_expr_grad(code, ps, theta, X):
    """Evaluacion DIFERENCIABLE de UNA expresion prefija.
    theta = [n_consts] tensor con los valores de las constantes CONST en orden prefijo;
    X = [n_vars, N]. Devuelve yhat [N] con grafo de autograd sobre theta."""
    named = ps.named_vals
    ci = [0]

    def rec(i):
        c = code[i]
        nxt = i + 1
        if c == ps.CONST:
            k = ci[0]
            ci[0] += 1
            return theta[k], nxt
        if c in named:
            return X.new_tensor(float(named[c])), nxt
        if 1 <= c <= ps.n_vars:
            return X[c - 1], nxt
        name = ps.code2name[c]
        if int(ps.arity[c]) == 1:
            v, nxt = rec(nxt)
            return _apply_unary(name, v), nxt
        v1, nxt = rec(nxt)
        v2, nxt = rec(nxt)
        return _apply_binary(name, v1, v2), nxt  # prefijo "op A B" -> op(A, B)

    y, _ = rec(0)
    return y


def local_optimize_constants(inds, ps, Xtr_t, ytr_t, device, steps=20):
    """Optimizacion LOCAL de constantes por gradiente (LBFGS a traves del interprete
    diferenciable), con linear scaling en forma cerrada. Es la palanca que iguala la
    eficiencia-por-evaluacion de Operon (que hace lo mismo con SGD/Levenberg)."""
    y = ytr_t
    ybar = y.mean()
    yc = y - ybar
    out = []
    with torch.enable_grad():
        for code, const in inds:
            cpos = [j for j, c in enumerate(code) if c == ps.CONST]
            if not cpos:
                out.append((list(code), list(const)))
                continue
            theta = torch.tensor([float(const[j]) for j in cpos], dtype=torch.float32,
                                 device=device, requires_grad=True)
            opt = torch.optim.LBFGS([theta], max_iter=steps, line_search_fn="strong_wolfe")

            def closure(_code=code, _theta=theta):
                opt.zero_grad()
                yhat = _eval_expr_grad(_code, ps, _theta, Xtr_t)
                yhat = torch.nan_to_num(yhat, nan=0.0, posinf=1e6, neginf=-1e6)
                fbar = yhat.mean()
                fc = yhat - fbar
                a = (fc * yc).sum() / ((fc * fc).sum() + 1e-12)
                b = ybar - a * fbar
                loss = ((a * yhat + b - y) ** 2).mean()
                loss = torch.nan_to_num(loss, nan=1e18, posinf=1e18, neginf=1e18)
                loss.backward()
                return loss

            try:
                opt.step(closure)
            except Exception:
                pass
            tv = theta.detach().cpu().numpy()
            newk = list(const)
            for k, j in enumerate(cpos):
                v = float(tv[k])
                newk[j] = v if np.isfinite(v) else float(const[j])
            out.append((list(code), newk))
    return out


def _dalex_parents(err, k, sigma, device, rng):
    """DALex (lexicase-como-un-matmul): err [P,N] error por-caso -> k indices de programa.
    Casi-gratis en GPU porque `err` ya esta materializado (en CPU es O(T*N^2), prohibitivo)."""
    n = err.shape[1]
    g = torch.Generator(device=device).manual_seed(int(rng.integers(1 << 31)))
    w = torch.softmax(torch.randn(k, n, generator=g, device=device) * sigma, dim=1)  # [k,N]
    return (err @ w.t()).argmin(dim=0).cpu().numpy()  # [k] mejor programa por evento ponderado


def evolve(Xtr, ytr, Xval, yval, ps, device, pop_size=3000, generations=100000,
           max_len=40, max_depth=9, tournament=6, cx_prob=0.7, mut_prob=0.35,
           elite=4, init_depth=4, cw=0.006, gap_pen=0.5, nest_pen=0.3, seed=0,
           time_budget=None, var_units=None, ndim=0, do_simplify=True,
           n_islands=6, migration_interval=8, n_migrants=6, restart_patience=30,
           const_opt_interval=5, const_opt_top=48, const_opt_trials=48,
           selection="tournament", dalex_sigma=3.0,
           subsample_size=2048, val_subsample_size=8192, subsample_threshold=50000,
           subsample_resample_interval=1, subsample_refit_full=True,
           on_generation=None, gen_delay=0.0, verbose=True):
    rng = np.random.default_rng(seed)
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    vary_tr = float(((ytr_t - ytr_t.mean()) ** 2).mean()) + 1e-12
    has_val = Xval is not None
    if has_val:
        Xval_t = torch.from_numpy(Xval.astype(np.float32)).to(device)
        yval_t = torch.from_numpy(yval.astype(np.float32)).to(device)
        vary_val = float(((yval_t - yval_t.mean()) ** 2).mean()) + 1e-12
    else:
        Xval_t = yval_t = None
        vary_val = vary_tr

    # --- CASE SUBSAMPLING (memory-safe fitness for large N; OFF below threshold) ---
    # Below the gate every *_cur is the full tensor and the executed code path is
    # behaviorally identical to no-subsampling. The train subset is re-drawn each gen
    # (down-sampled-lexicase diversity); the val subset is drawn ONCE (fixed seeded)
    # so best/Pareto/const-opt acceptance sit on a stationary yardstick.
    Ntr = Xtr_t.shape[1]
    use_sub = subsample_size is not None and Ntr >= subsample_threshold
    Xtr_cur, ytr_cur, vary_tr_cur = Xtr_t, ytr_t, vary_tr
    Xval_cur, yval_cur, vary_val_cur = Xval_t, yval_t, vary_val
    if use_sub and has_val:
        Nval = Xval_t.shape[1]
        vgen = torch.Generator(device=device).manual_seed(seed + 1)
        vidx = torch.randperm(Nval, generator=vgen, device=device)[:min(val_subsample_size, Nval)]
        Xval_cur = Xval_t[:, vidx]
        yval_cur = yval_t[vidx]
        vary_val_cur = float(((yval_cur - yval_cur.mean()) ** 2).mean()) + 1e-12

    def maybe_simplify(ind):
        if not do_simplify:
            return ind
        try:
            return simplify_prefix(ind[0], ind[1], ps)
        except Exception:
            return ind

    def new_ind():
        for _ in range(50):
            d = int(rng.integers(2, init_depth + 1))
            c, k = gen_tree(ps, d, "full" if rng.random() < 0.5 else "grow", rng)
            c, k = maybe_simplify((c, k))
            if _ok_size(c, ps, max_len, max_depth):
                return (c, k)
        return maybe_simplify(gen_tree(ps, 2, "grow", rng))

    def eval_flat(flat, return_yhat=False):
        codes, consts, _ = batch_postfix(flat, ps)
        yh = run_population(codes, consts, Xtr_cur, ps, device)
        a, b, mse_tr, r2_tr = fit_ab(yh, ytr_cur)
        if has_val:
            yhv = run_population(codes, consts, Xval_cur, ps, device)
            mse_val, r2_val = score_ab(yhv, yval_cur, a, b)
        else:
            mse_val, r2_val = mse_tr, r2_tr
        base = (a.cpu().numpy(), b.cpu().numpy(), mse_tr.cpu().numpy(), r2_tr.cpu().numpy(),
                mse_val.cpu().numpy(), r2_val.cpu().numpy())
        return (*base, a, b, yh) if return_yhat else base

    def locate(bounds, pos):
        for j, (s, e) in enumerate(bounds):
            if s <= pos < e:
                return j, pos - s
        return 0, 0

    # --- MODELO DE ISLAS (subpoblaciones) con parsimonia HETEROGENEA ---
    n_islands = max(1, n_islands)
    isize = max(6, pop_size // n_islands)
    cw_mult = np.geomspace(0.25, 4.0, n_islands) if n_islands > 1 else np.array([1.0])
    islands = [[new_ind() for _ in range(isize)] for _ in range(n_islands)]
    isl_best = [1e18] * n_islands   # mejor mse_val visto por isla (para detectar estancamiento)
    isl_stall = [0] * n_islands
    n_restarts = 0

    archive = []
    best = None
    t0 = time.time()
    for gen in range(generations):
        flat, bounds = [], []
        for isl in islands:
            bounds.append((len(flat), len(flat) + len(isl)))
            flat.extend(isl)

        if use_sub and gen % subsample_resample_interval == 0:
            # independent per-gen RNG: never advances the numpy GP stream, so the
            # small-N (gate-off) trajectory stays byte-for-byte reproducible.
            tgen = torch.Generator(device=device).manual_seed(seed * 1_000_003 + gen)
            tidx = torch.randperm(Ntr, generator=tgen, device=device)[:min(subsample_size, Ntr)]
            Xtr_cur = Xtr_t[:, tidx]
            ytr_cur = ytr_t[tidx]
            vary_tr_cur = float(((ytr_cur - ytr_cur.mean()) ** 2).mean()) + 1e-12

        if selection == "dalex":
            (a_np, b_np, mse_tr_np, r2_tr_np, mse_val_np, r2_val_np,
             a_t, b_t, yh_t) = eval_flat(flat, return_yhat=True)
            Efull = (a_t.unsqueeze(1) * yh_t + b_t.unsqueeze(1) - ytr_cur.unsqueeze(0)) ** 2  # [P,S]
        else:
            a_np, b_np, mse_tr_np, r2_tr_np, mse_val_np, r2_val_np = eval_flat(flat)
            Efull = None
        size = np.array([len(c) for c, _ in flat], dtype=np.float64)
        wc = np.array([weighted_complexity(c, ps) for c, _ in flat], dtype=np.float64)
        nest = np.array([nested_transcendental_count(c, ps) for c, _ in flat], dtype=np.float64)
        dim_ok = (np.array([units_consistent(c, ps, var_units, ndim) for c, _ in flat], bool)
                  if var_units is not None else np.ones(len(flat), bool))
        nmse_tr = mse_tr_np / vary_tr_cur
        gap = np.maximum(0.0, mse_val_np / vary_val_cur - nmse_tr)
        cw_arr = np.empty(len(flat))
        for j, (s, e) in enumerate(bounds):
            cw_arr[s:e] = cw * cw_mult[j]
        sel_fit = nmse_tr + cw_arr * wc + gap_pen * gap + nest_pen * nest + np.where(dim_ok, 0.0, 1e6)
        eff_val = np.where(dim_ok, mse_val_np, 1e18)

        def mkcand(i):
            return {"complexity": int(size[i]), "wc": float(wc[i]), "nest": int(nest[i]),
                    "mse_val": float(eff_val[i]), "r2_val": float(r2_val_np[i]),
                    "mse_tr": float(mse_tr_np[i]), "r2_tr": float(r2_tr_np[i]),
                    "a": float(a_np[i]), "b": float(b_np[i]),
                    "code": flat[i][0], "const": flat[i][1]}
        archive = pareto_update(archive, [mkcand(i) for i in range(len(flat))])
        gi = int(np.argmin(eff_val))
        if best is None or eff_val[gi] < best["mse_val"]:
            best = mkcand(gi)

        # --- OPTIMIZACION DE CONSTANTES sobre los mejores (palanca de R2) ---
        if const_opt_interval and gen % const_opt_interval == 0:
            order = np.argsort(eff_val)[:min(const_opt_top, len(flat))]
            opt = local_optimize_constants([flat[i] for i in order], ps, Xtr_cur, ytr_cur, device)
            ao, bo, mto, rto, mvo, rvo = eval_flat(opt)
            for idx, pos in enumerate(order):
                if mvo[idx] < eff_val[pos] - 1e-12:
                    j, loc = locate(bounds, pos)
                    islands[j][loc] = opt[idx]
                    flat[pos] = opt[idx]
                    eff_val[pos] = mvo[idx]; mse_val_np[pos] = mvo[idx]; r2_val_np[pos] = rvo[idx]
                    mse_tr_np[pos] = mto[idx]; r2_tr_np[pos] = rto[idx]
                    a_np[pos] = ao[idx]; b_np[pos] = bo[idx]
                    c2 = mkcand(pos)
                    archive = pareto_update(archive, [c2])
                    if c2["mse_val"] < best["mse_val"]:
                        best = c2
            gi = int(np.argmin(eff_val))

        if verbose:
            print(f"gen {gen:3d} | best val R2={best['r2_val']:.5f} | genR2={r2_val_np[gi]:.4f} "
                  f"| pareto={len(archive)} | restarts={n_restarts} | mean_nodes={size.mean():.1f} "
                  f"| t={time.time()-t0:.1f}s", flush=True)

        if gen_delay:
            time.sleep(gen_delay)
        if on_generation is not None:
            on_generation({
                "gen": gen, "elapsed": time.time() - t0, "pop": len(flat),
                "best": best, "archive": archive,
                "cloud_c": size, "cloud_r2": r2_val_np,
                "mean_nodes": float(size.mean()), "mean_wc": float(wc.mean()),
                "mean_nest": float(nest.mean()), "restarts": n_restarts, "islands": n_islands,
            })

        if best["r2_val"] > 1 - 1e-9 or (time_budget and time.time() - t0 > time_budget):
            break
        if gen == generations - 1:
            break

        # --- NUEVA GENERACION: elitismo + migracion (anillo) + restart por estancamiento ---
        migrating = (n_islands > 1 and migration_interval and gen % migration_interval == 0 and gen > 0)
        new_islands = []
        for j, isl in enumerate(islands):
            s, e = bounds[j]
            fit = sel_fit[s:e]
            order = np.argsort(fit)

            cur = float(np.min(eff_val[s:e]))
            if cur < isl_best[j] - 1e-9:
                isl_best[j] = cur; isl_stall[j] = 0
            else:
                isl_stall[j] += 1

            if isl_stall[j] >= restart_patience:               # RESTART de la isla estancada
                n_restarts += 1
                isl_stall[j] = 0; isl_best[j] = 1e18
                hof = archive[:max(2, n_migrants)]
                newpop = [(list(isl[o][0]), list(isl[o][1])) for o in order[:2]]
                newpop += [(list(d["code"]), list(d["const"])) for d in hof]
                while len(newpop) < len(isl):
                    newpop.append(new_ind())
                new_islands.append(newpop[:len(isl)])
                continue

            seeds = [isl[o] for o in order[:elite]]            # ELITISMO
            if migrating:                                      # MIGRACION en anillo
                src = (j - 1) % n_islands
                ss, se = bounds[src]
                for o in np.argsort(sel_fit[ss:se])[:n_migrants]:
                    seeds.append((list(islands[src][o][0]), list(islands[src][o][1])))
            newpop = list(seeds)

            def tourney(_isl=isl, _fit=fit):
                idx = rng.integers(0, len(_isl), size=min(tournament, len(_isl)))
                return _isl[idx[np.argmin(_fit[idx])]]

            if selection == "dalex" and Efull is not None:
                _pool = _dalex_parents(Efull[s:e], 2 * len(isl) + 8, dalex_sigma, device, rng)

                def pick(_isl=isl, _pool=_pool, _c=[0]):  # DALex-selected parents (cycled)
                    v = _isl[int(_pool[_c[0] % len(_pool)])]
                    _c[0] += 1
                    return v
            else:
                pick = tourney

            while len(newpop) < len(isl):
                child = crossover(pick(), pick(), ps, max_len, max_depth, rng) \
                    if rng.random() < cx_prob else pick()
                if rng.random() < mut_prob:
                    r = rng.random()
                    if r < 0.4:
                        child = mut_subtree(child, ps, max_len, max_depth, rng)
                    elif r < 0.7:
                        child = mut_point(child, ps, rng)
                    elif r < 0.9:
                        child = mut_const(child, ps, rng)
                    else:
                        child = mut_hoist(child, ps, rng)
                child = maybe_simplify((list(child[0]), list(child[1])))
                if not _ok_size(child[0], ps, max_len, max_depth):
                    child = (list(child[0])[:max_len], list(child[1])[:max_len])
                newpop.append((list(child[0]), list(child[1])))
            new_islands.append(newpop[:len(isl)])
        islands = new_islands

    # --- FULL-DATA REFIT of linear scaling (a, b) for EXPORTED models ---
    # During subsampling, archive/best store a,b fit on a per-gen S-case subset; predict()
    # applies them to full data, so re-fit them on FULL train (memory-safe, amortized once).
    if subsample_refit_full and use_sub and archive:
        ents = archive + ([best] if best is not None else [])
        rcodes, rconsts, _ = batch_postfix([(d["code"], d["const"]) for d in ents], ps)
        af, bf, _, rtf = run_population_reduce(rcodes, rconsts, Xtr_t, ytr_t, ps, device)
        for i, d in enumerate(ents):
            d["a"] = float(af[i])
            d["b"] = float(bf[i])
            d["r2_tr"] = float(rtf[i])

    return best, archive


# ----------------------------------------------------------------------------
# 7. EXPORT
# ----------------------------------------------------------------------------
def expr_to_numpy_src(expr, ps):
    import sympy as sp
    try:
        from sympy.printing.numpy import NumPyPrinter
        code = NumPyPrinter().doprint(expr).replace("numpy.", "np.")
    except Exception:
        code = sp.pycode(expr, fully_qualified_modules=False).replace("math.", "np.")
    args = ", ".join(f"x{i}" for i in range(ps.n_vars))
    src = ["import numpy as np", "", f"def predict({args}):", f"    return {code}", "",
           "def predict_matrix(X):", f"    # X: array [n_samples, {ps.n_vars}]"]
    for i in range(ps.n_vars):
        src.append(f"    x{i} = X[:, {i}]")
    src.append(f"    return predict({args})")
    return "\n".join(src) + "\n"


def export_results(best, archive, ps, outdir):
    import sympy as sp
    os.makedirs(outdir, exist_ok=True)
    expr = to_sympy(best["code"], best["const"], ps, best["a"], best["b"])
    with open(os.path.join(outdir, "best_formula.py"), "w") as f:
        f.write(f"# R2_val={best['r2_val']:.6f}  R2_test={best.get('r2_test', float('nan')):.6f}"
                f"  nodos={best['complexity']}  wc={best['wc']:.0f}\n# y = {expr}\n\n")
        f.write(expr_to_numpy_src(expr, ps))
    with open(os.path.join(outdir, "best_formula.tex"), "w") as f:
        f.write(sp.latex(expr) + "\n")
    with open(os.path.join(outdir, "pareto.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nodes", "weighted_complexity", "nested", "r2_train", "r2_val",
                    "r2_test", "expression", "latex"])
        for d in archive:
            e = to_sympy(d["code"], d["const"], ps, d["a"], d["b"])
            w.writerow([d["complexity"], f"{d['wc']:.0f}", d["nest"], f"{d['r2_tr']:.6f}",
                        f"{d['r2_val']:.6f}", f"{d.get('r2_test', float('nan')):.6f}",
                        str(e), sp.latex(e)])
    print(f"[export] escrito en {outdir}/  (best_formula.py, best_formula.tex, pareto.csv)")


# ----------------------------------------------------------------------------
# 8. ENTORNO / DATOS / SELFTEST
# ----------------------------------------------------------------------------
def pick_device(prefer=None):
    if prefer is not None:
        return torch.device(prefer)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, bestfree = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > bestfree:
            best, bestfree = i, free
    return torch.device(f"cuda:{best}")


def make_synthetic(n_points, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-3, 3, n_points)
    x1 = rng.uniform(-3, 3, n_points)
    y = x0 ** 2 + x0 * x1 + np.sin(x1)
    return np.vstack([x0, x1]), y, "x0^2 + x0*x1 + sin(x1)"


def split_data(X, y, seed=0, val=0.2, test=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[1])
    n_te, n_va = int(X.shape[1] * test), int(X.shape[1] * val)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return (X[:, tr], y[tr]), (X[:, va], y[va]), (X[:, te], y[te])


def score_models_on(entries, X_np, y_np, ps, device):
    if not entries:
        return
    X_t = torch.from_numpy(X_np.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
    codes, consts, _ = batch_postfix([(d["code"], d["const"]) for d in entries], ps)
    a = torch.tensor([d["a"] for d in entries], dtype=torch.float32, device=device)
    b = torch.tensor([d["b"] for d in entries], dtype=torch.float32, device=device)
    _, _, _, r2 = run_population_reduce(codes, consts, X_t, y_t, ps, device, a=a, b=b)
    r2 = r2.cpu().numpy()
    for i, d in enumerate(entries):
        d["r2_test"] = float(r2[i])


def selftest(seed=0, n_trees=4000, N=256, tol=1e-2):
    """Test de PROPIEDAD: simplificar un arbol no debe cambiar su salida numerica.
    Compara la prediccion del arbol original contra la del simplificado sobre datos
    aleatorios; asi garantizamos que las equivalencias preservan la semantica (bajo
    operadores protegidos)."""
    rng = np.random.default_rng(seed)
    ps = PrimSet(3, UNARY_ALL, BINARY_ALL, named=["pi", "e"])
    X = rng.uniform(-3, 3, size=(3, N)).astype(np.float32)
    Xd = torch.from_numpy(X)
    dev = torch.device("cpu")
    pop, popS = [], []
    for _ in range(n_trees):
        c, k = gen_tree(ps, int(rng.integers(2, 6)), "grow", rng)
        sc, sk = simplify_prefix(c, k, ps)
        pop.append((c, k))
        popS.append((sc, sk))
    co, ko, _ = batch_postfix(pop, ps)
    cs, ks, _ = batch_postfix(popS, ps)
    y0 = run_population(co, ko, Xd, ps, dev).numpy()
    y1 = run_population(cs, ks, Xd, ps, dev).numpy()
    diff = np.abs(y0 - y1)
    rel = (diff / np.maximum(1.0, np.abs(y0))).max(axis=1)
    n_bad = int((rel > tol).sum())
    shrink = 1 - np.mean([len(s[0]) for s in popS]) / np.mean([len(p[0]) for p in pop])
    print(f"[selftest] {n_trees} arboles | fallos(rel>{tol})={n_bad} "
          f"| max_rel={rel.max():.2e} | reduccion_media_nodos={shrink*100:.1f}%")
    if n_bad == 0:
        print("[selftest] OK: la simplificacion PRESERVA la semantica.")
    else:
        worst = int(np.argmax(rel))
        print(f"[selftest] FALLA en arbol {worst}: rel={rel[worst]:.3e}")
    return n_bad == 0


def load_units(path, feature_names):
    with open(path) as f:
        spec = json.load(f)
    dims = spec.get("dims", [])
    vmap = spec.get("vars", {})
    var_units = np.zeros((len(feature_names), len(dims)), dtype=float)
    for i, name in enumerate(feature_names):
        if name in vmap:
            var_units[i] = np.array(vmap[name], dtype=float)
    return var_units, len(dims), dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--npoints", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=3000)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=9)
    ap.add_argument("--parsimony", type=float, default=0.006, help="coef de complejidad ponderada")
    ap.add_argument("--gap-penalty", type=float, default=0.5, help="penaliza gap train-val (overfit)")
    ap.add_argument("--nest-penalty", type=float, default=0.3, help="penaliza transcendentes anidados")
    ap.add_argument("--no-simplify", action="store_true", help="desactiva simplificacion algebraica")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--units", type=str, default=None)
    ap.add_argument("--export", type=str, default=None)
    ap.add_argument("--named", type=str, default="pi,e", help="constantes con nombre (terminales)")
    ap.add_argument("--unary", type=str, default="sin,cos,exp,log,sqrt,neg,square")
    ap.add_argument("--binary", type=str, default="add,sub,mul,aq")
    ap.add_argument("--islands", type=int, default=6, help="subpoblaciones (islas) con migracion")
    ap.add_argument("--migration", type=int, default=8, help="cada cuantas gens migran (anillo)")
    ap.add_argument("--restart-patience", type=int, default=30, help="gens sin mejora -> restart de isla")
    ap.add_argument("--const-opt", type=int, default=5, help="cada cuantas gens optimizar constantes (0=off)")
    ap.add_argument("--const-opt-trials", type=int, default=48)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    device = pick_device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device.index)
        print(f"[dev] {device} {torch.cuda.get_device_name(device.index)} "
              f"free={free/1e9:.1f}GB/{total/1e9:.1f}GB")
    else:
        print(f"[dev] {device}")

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        tcol = args.target or df.columns[-1]
        feat = [c for c in df.columns if c != tcol]
        X, y = df[feat].to_numpy().T, df[tcol].to_numpy()
        n_vars, truth = len(feat), None
        print(f"[data] csv={args.csv} vars={feat} target={tcol} N={len(y)}")
    else:
        X, y, truth = make_synthetic(args.npoints, args.seed)
        n_vars = X.shape[0]
        feat = [f"x{i}" for i in range(n_vars)]
        print(f"[data] demo sintetica  y = {truth}  (N={len(y)}, vars={n_vars})")

    ps = PrimSet(n_vars, args.unary.split(","), args.binary.split(","),
                 named=[s for s in args.named.split(",") if s])
    print(f"[prims] unary={ps.unary} binary={ps.binary} named={ps.named} "
          f"simplify={not args.no_simplify}")

    var_units, ndim = None, 0
    if args.units:
        var_units, ndim, dims = load_units(args.units, feat)
        print(f"[units] analisis dimensional activo dims={dims}")

    if args.benchmark:
        r = _bench(ps, device)
        print(f"\n[bench] GPU {r['g']:.3e} vs CPU {r['c']:.3e} node-evals/s "
              f"-> {r['s']:.1f}x\n")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y, args.seed, args.val, args.test)
    print(f"[split] train={ytr.size} val={yva.size} test={yte.size}")

    print("\n=== EVOLUCION (fitness: precision + complejidad ponderada + coherencia) ===")
    t0 = time.time()
    best, archive = evolve(Xtr, ytr, Xva, yva, ps, device, pop_size=args.pop,
                           generations=args.generations, max_len=args.max_len,
                           max_depth=args.max_depth, cw=args.parsimony,
                           gap_pen=args.gap_penalty, nest_pen=args.nest_penalty,
                           time_budget=args.time_budget, seed=args.seed,
                           var_units=var_units, ndim=ndim,
                           do_simplify=not args.no_simplify,
                           n_islands=args.islands, migration_interval=args.migration,
                           restart_patience=args.restart_patience,
                           const_opt_interval=args.const_opt, const_opt_trials=args.const_opt_trials)
    dt = time.time() - t0
    score_models_on(archive + [best], Xte, yte, ps, device)

    print(f"\n=== RESULTADO ({dt:.1f}s) ===")
    if truth:
        print(f"objetivo real : y = {truth}")
    print(f"R2 train={best['r2_tr']:.6f} val={best['r2_val']:.6f} TEST={best['r2_test']:.6f}"
          f"  (nodos={best['complexity']}, wc={best['wc']:.0f}, nest={best['nest']})")
    try:
        print(f"mejor formula : y = {to_sympy(best['code'], best['const'], ps, best['a'], best['b'])}")
    except Exception as e:
        print(f"  (no se pudo simplificar: {e})")

    print("\n=== FRENTE DE PARETO (nodos | wc | R2 val | R2 test) ===")
    for d in archive:
        try:
            s = str(to_sympy(d["code"], d["const"], ps, d["a"], d["b"]))
        except Exception:
            s = "<expr>"
        if len(s) > 74:
            s = s[:71] + "..."
        print(f"  n={d['complexity']:3d} wc={d['wc']:3.0f} | val={d['r2_val']:.4f} "
              f"test={d['r2_test']:.4f} | y = {s}")

    if args.export:
        export_results(best, archive, ps, args.export)


def _bench(ps, device, P=20000, N=5000, seed=0):
    rng = np.random.default_rng(seed)
    Xn = rng.uniform(-2, 2, size=(ps.n_vars, N)).astype(np.float32)
    X = torch.from_numpy(Xn).to(device)
    pop = [gen_tree(ps, 4, "full", rng) for _ in range(P)]
    codes, consts, _ = batch_postfix(pop, ps)
    nodes = int((codes != 0).sum())
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    run_population(codes, consts, X, ps, device, chunk=2048)
    if device.type == "cuda":
        torch.cuda.synchronize()
    g = nodes * N / (time.time() - t)
    sub = min(P, 2000)
    t = time.time()
    run_population(codes[:sub], consts[:sub], torch.from_numpy(Xn), ps, torch.device("cpu"), 512)
    c = int((codes[:sub] != 0).sum()) * N / (time.time() - t)
    return {"g": g, "c": c, "s": g / c}


if __name__ == "__main__":
    main()

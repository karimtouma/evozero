# How it works

This page explains the architecture that lets `evozero` run evolutionary search on the
GPU.

## The core idea: batched fitness on the GPU

Genetic programming spends almost all its time **evaluating candidate expressions**. The
key move in `evozero` is to evaluate a *whole population* of expressions against a *whole
dataset* as **one batched tensor operation**, instead of interpreting trees one at a time.

Each expression is stored in **postfix (reverse-Polish) form**, padded to a fixed length.
A tensorized stack interpreter then walks the token positions `t = 0 … L`, and at every
step applies the right operation to every program × every data row at once:

```text
stack  S : tensor of shape [depth, P, N]     P = programs, N = data rows
for t in range(L):
    token[p]  → push a variable/constant, or apply a unary/binary op to the
                top of each program's stack, dispatched by masking on the op code
```

A step is `O(P × N)` elementwise work done in parallel — exactly what a GPU is for. On an
H100 this reaches billions of node-evaluations per second, ~30× a vectorized CPU baseline
on large populations.

## The hybrid CPU/GPU loop

The evolutionary bookkeeping (selection, crossover, mutation) is cheap and irregular, so
it stays on the **CPU**; only the expensive fitness evaluation goes to the **GPU**. Trees
are manipulated as prefix arrays where subtree crossover and mutation are always valid (no
stack underflow), then serialized to postfix for the batched GPU pass. Only the genome
(a few kilobytes of integers) crosses to the GPU each generation — the data lives there
permanently.

## Linear scaling

For each candidate `f(x)`, `evozero` fits the best `a, b` in `y ≈ a·f(x) + b` in closed
form (Keijzer scaling) — vectorized across the whole population on the GPU. This removes
the burden of discovering global scale/offset and dramatically improves fitness for a
given structure. The `a, b` are fit on the training split only and applied unchanged to
validation, so reported scores are honest.

## The island model (why the search doesn't stall)

A single population converges prematurely and gets stuck. `evozero` uses the standard
robustness machinery from PySR/Operon/Bingo:

- **Islands.** Several semi-independent sub-populations evolve in parallel.
- **Heterogeneous parsimony.** Each island uses a different complexity pressure
  (`0.25× … 4×`): one island chases accuracy with big expressions, another chases
  simplicity. This keeps a spread of solutions alive and lets accurate structures emerge.
- **Ring migration.** Every few generations each island sends its best individuals to a
  neighbour, injecting good genes and maintaining diversity.
- **Stagnation restart.** If an island stops improving for `restart_patience` generations,
  it is reinitialized — keeping its elite plus a global hall-of-fame — to escape local
  optima. The global best is never lost.

## Constant optimization

Structure discovery (discrete GP) and numeric fitting (continuous) are separated. Every
few generations the numeric constants of the top models are refined by a **GPU-batched
evolution strategy**: perturb the constants, evaluate the whole batch at once, keep the
best, anneal. This is the single biggest lever on accuracy for a fixed structure, and it
needs no autodiff (robust).

## The Pareto front

Fitness is multi-objective: **accuracy** and **complexity**. `evozero` maintains the
non-dominated set across the whole run and exposes it as `pareto_front_` — the same
accuracy-vs-complexity trade-off curve that made Eureqa's UI famous. You choose the knee.

## Optional dimensional analysis

If you supply physical units for the variables, expressions that are dimensionally
inconsistent (e.g. `sin(mass)`, `length + time`) are penalized — a lightweight take on
PhySO. Internal consistency is enforced; the global units are absorbed by linear scaling.

## AutoML-Zero: a second VM

The AutoML-Zero engine reuses the same tensorized philosophy but with a richer genome: a
program of typed-register instructions (scalars, vectors, per-example buffers) split into
`Setup` / `Predict` / `Learn`. Evaluating a candidate means *running* it — `Setup`, then a
loop of `Predict`+`Learn` on the training data (this is where it *learns*), then `Predict`
on held-out data. Fitness is averaged across many tasks so the discovered algorithm
generalizes. See {doc}`../guide/automl_zero`.

## Package architecture

```text
evozero/
├── core/          the only subpackage that imports torch at top level
│   ├── the tensorized interpreter, island loop, constant optimizer
│   └── the AutoML-Zero VM
├── sr/            SymbolicRegressor facade (sklearn-style)
├── automlzero/    LearnerSearch / EvolvedLearner / Task
└── dashboard/     stdlib HTTP server + packaged index.html
```

`torch` is isolated to `core/`; the public API is lazy ([PEP 562](https://peps.python.org/pep-0562/)),
so `import evozero` never imports `torch`. This is why LaTeX/NumPy export works on a
machine with no GPU and no torch.

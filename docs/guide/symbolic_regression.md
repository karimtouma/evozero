# Symbolic regression

`SymbolicRegressor` searches the space of mathematical expressions for a formula
`y = f(X)` that balances **accuracy** and **simplicity**. It follows the scikit-learn
estimator contract, so it drops into pipelines and model-selection tools.

## Basic usage

```python
from evozero import SymbolicRegressor

model = SymbolicRegressor(device="auto", random_state=0)
model.fit(X, y)               # X: [n_samples, n_features], y: [n_samples]
model.predict(X)
model.score(X, y)             # R^2
```

## Parameters

### Search budget

`population_size` (default `3000`)
: Total individuals across all islands. Larger populations explore more, and are exactly
  where the GPU pays off.

`generations` (default `200`)
: Maximum number of generations.

`max_time` (default `None`)
: Wall-clock budget in seconds. When reached, the search stops and returns the best model
  so far. Handy for reproducible time-boxed runs.

### Primitives (the building blocks)

`binary_operators` (default `("+", "-", "*", "/")`)
: Binary primitives. You may pass symbols (`"+"`, `"-"`, `"*"`, `"/"`) or engine names
  (`"add"`, `"sub"`, `"mul"`, `"div"`, `"aq"`). `aq` is the *analytic quotient*
  `a / sqrt(1 + b²)` — a smooth, pole-free alternative to division.

`unary_operators` (default `("sin", "cos", "exp", "log", "sqrt")`)
: Unary primitives. Available: `sin cos exp log sqrt neg square cube tanh abs inv`.

Named constants `pi` and `e` are always available as terminals, alongside evolved
numeric constants.

### Complexity control (robustness)

`max_size` (default `40`)
: Maximum number of nodes in an expression.

`max_depth` (default `9`)
: Maximum tree depth.

`parsimony_coefficient` (default `0.006`)
: Weight of the (per-operator) *complexity* term in the selection fitness. Higher →
  simpler formulas. Operators are weighted by fragility (`exp`/`log` cost more than `+`),
  an MDL-flavoured pressure. Lower it (e.g. `0.002`) to allow bigger, more accurate models.

### Diversity (escaping local optima)

`n_islands` (default `6`)
: Number of semi-independent sub-populations. Islands have **heterogeneous** parsimony
  pressure — one chases accuracy, another chases simplicity — and periodically exchange
  their best individuals. This is what keeps the search from stalling.

`migration_interval` (default `8`)
: Migrate the best individuals between islands every this many generations (ring topology).

`restart_patience` (default `30`)
: If an island doesn't improve for this many generations it is **restarted** (keeping its
  elite + a global hall-of-fame), escaping local optima. The global best is never lost.

### Fine-tuning constants

`const_opt_interval` (default `5`)
: Every this many generations, the numeric constants of the top models are optimized by a
  GPU-batched evolution strategy. This is the single biggest lever on `R²` for a given
  structure. Set to `0` to disable.

### Runtime

`device` (default `"auto"`)
: `"auto"` | `"cpu"` | `"cuda"` | `"cuda:N"` | `"mps"`. `"auto"` picks the CUDA device with
  the most free memory, else MPS, else CPU.

`random_state` (default `None`)
: Seed for reproducibility.

`verbose` (default `0`)
: Print per-generation progress when `> 0`.

## Fitted attributes

After `fit`, these carry a trailing underscore:

`best_equation_` : {class}`~evozero.Equation`
: The best model by validation error. `str(best_equation_)` is the readable formula.

`pareto_front_` : list of dict
: The non-dominated set, each `{"complexity", "loss", "r2", "equation"}`, sorted by
  complexity. This is the Eureqa-style trade-off curve.

`device_` : `torch.device`
: The device the search actually ran on.

`n_features_in_` : int
: Number of input features seen during `fit`.

## Exporting the equation

```python
model.to_sympy()             # SymPy expression (symbolic manipulation, simplification)
model.to_latex()             # LaTeX string for papers/reports
model.to_numpy_func()        # pure-NumPy callable f(X) -> y, needs no torch
```

Every exporter accepts an `index=` argument to export a specific Pareto entry instead of
the best one:

```python
simplest_good = model.pareto_front_[4]["equation"]   # pick a knee
model.to_latex(index=4)
```

## Working with a Pareto entry

```python
eq = model.pareto_front_[4]["equation"]     # an Equation
eq.complexity        # node count
eq.r2                # validation R^2
eq.to_sympy()        # SymPy
eq.to_latex()        # LaTeX
str(eq)              # readable string
```

## Tips

- **Start broad, then tighten.** Run with defaults; if the formula is too complex, raise
  `parsimony_coefficient`; if it underfits, lower it and/or raise `max_size`.
- **Give it operators that fit the domain.** Fewer, well-chosen operators search faster.
  Drop `sin`/`cos` if the signal isn't periodic.
- **Use `max_time` for reproducible runs**, and `random_state` for reproducible results.
- **Interpretable knees live at low complexity.** The most useful models are usually in
  the first few Pareto rows, not the highest-`R²` one.
- **Named constants can add clutter.** On very simple targets you may see `+ pi - e - c`
  terms that net to ~0; they don't hurt `R²` but you can post-simplify with SymPy.

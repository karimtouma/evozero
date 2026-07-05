# evozero

**Evolve formulas and learning algorithms from zero — GPU-native evolutionary computation.**

`evozero` is a GPU-accelerated evolutionary-computation toolkit. Three engines share
one tensorized core:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🔎 Symbolic regression
`SymbolicRegressor` discovers an **interpretable equation** `y = f(X)` — a modern,
open-source take on Eureqa. Island-model genetic programming, a tensorized postfix
interpreter, linear scaling and constant optimization, returning a full
complexity-vs-accuracy Pareto front.
+++
{doc}`guide/symbolic_regression`
:::

:::{grid-item-card} 🧬 AutoML-Zero
`LearnerSearch` evolves the **learning algorithm itself** (`Setup`/`Predict`/`Learn`
over a typed-register VM), meta-trained across tasks so the discovered learner
generalizes to unseen problems.
+++
{doc}`guide/automl_zero`
:::

:::{grid-item-card} 📊 Live dashboard
`launch_dashboard` serves a dependency-free (stdlib) live web view of the search:
the Pareto cloud, convergence curve, and predicted-vs-actual fit.
+++
{doc}`guide/dashboard`
:::
::::

## Why evozero

- **Interpretable by default.** The output of symbolic regression *is* the formula —
  something you can read, audit, and put in a report, not a black box.
- **GPU-native.** A whole population of candidate expressions is evaluated as one
  batched tensor operation. On large populations × datasets this is where the GPU wins.
- **`import evozero` never imports `torch`.** The public API is lazy
  ([PEP 562](https://peps.python.org/pep-0562/)), so export utilities (LaTeX, NumPy)
  work on machines without a GPU. PyTorch is only touched when you run a search.
- **scikit-learn-style API.** `fit` / `predict` / `score`, learned attributes with a
  trailing underscore, `get_params` / `set_params`.
- **Permissive.** Apache-2.0.

## Install

```bash
pip install evozero                 # CPU (brings the CPU torch wheel)
pip install "evozero[cuda]" --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
```

See {doc}`installation` for details.

## 30-second example

```python
import numpy as np
from evozero import SymbolicRegressor

X = np.random.randn(500, 2)
y = X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])

model = SymbolicRegressor(device="auto", random_state=0).fit(X, y)
print(model.best_equation_)      # x0*(x0 + x1) + sin(x1)
print(model.to_latex())
```

Continue with the {doc}`quickstart`.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
guide/symbolic_regression
guide/automl_zero
guide/dashboard
concepts/how_it_works
api
contributing
changelog
```

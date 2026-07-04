# evozero

**Evolve formulas and learning algorithms from zero — GPU-native evolutionary computation.**

[![CI](https://github.com/karimtouma/evozero/actions/workflows/ci.yml/badge.svg)](https://github.com/karimtouma/evozero/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evozero.svg)](https://pypi.org/project/evozero/)
[![Python](https://img.shields.io/pypi/pyversions/evozero.svg)](https://pypi.org/project/evozero/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`evozero` is a GPU-accelerated evolutionary-computation toolkit with three engines that
share one tensorized core:

- **`SymbolicRegressor`** — Eureqa-style *symbolic regression*: discovers an interpretable
  equation `y = f(X)`. Island-model GP + tensorized postfix interpreter + linear scaling +
  constant optimization, returning a full complexity-vs-accuracy Pareto front.
- **`LearnerSearch`** — *AutoML-Zero*: evolves the **learning algorithm itself**
  (`Setup`/`Predict`/`Learn` over a typed-register VM), meta-trained across tasks so the
  discovered learner generalizes to unseen problems.
- **`launch_dashboard`** — a dependency-free (stdlib) live web dashboard of the search.

`import evozero` never imports `torch`: export utilities (LaTeX, NumPy) work on machines
without a GPU. PyTorch is only imported when you actually run a search.

## Install

```bash
pip install evozero                 # CPU (brings the CPU torch wheel)
pip install "evozero[cuda]" --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
uv sync --extra cuda                # from a clone, for development
```

`evozero` does **not** pin a CUDA build of PyTorch — you choose the wheel that matches your
driver. See <https://pytorch.org/get-started/>.

## Quickstart — symbolic regression

```python
import numpy as np
from evozero import SymbolicRegressor

X = np.random.randn(500, 2)
y = X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])

model = SymbolicRegressor(population_size=3000, generations=200,
                          device="auto", random_state=0, verbose=1)
model.fit(X, y)

print(model.best_equation_)     # x0*(x0 + x1) + sin(x1)
print(model.to_latex())         # x_{0} \left(x_{0} + x_{1}\right) + \sin{x_{1}}
f = model.to_numpy_func()       # pure-NumPy callable, no torch needed
model.pareto_front_             # [{'complexity', 'loss', 'r2', 'equation'}, ...]
```

## Quickstart — AutoML-Zero

```python
from evozero import LearnerSearch, Task

tasks = [Task(Xi, yi) for Xi, yi in my_meta_training_data]
search = LearnerSearch(population_size=200, max_time=60, device="auto", random_state=0)
search.fit(tasks)

learner = search.export_learner()      # a discovered EvolvedLearner
learner.fit(X_train, y_train)
preds = learner.predict(X_test)
print(learner.to_python_source())      # the evolved learning algorithm, as code
```

## Live dashboard

```python
from evozero import launch_dashboard
with launch_dashboard(port=8080) as dash:
    print("open", dash.url)
    ...  # push metrics with dash.update({...}) from a training callback
```
or `evozero dashboard --port 8080`.

## Documentation

<https://evozero.readthedocs.io> · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Citing

If `evozero` helps your research, please cite it — see [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0 © Karim Touma. See [LICENSE](LICENSE).

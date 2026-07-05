# Quickstart

This 5-minute tour fits an equation, reads the Pareto front, and exports the result.

## Fit a model

```python
import numpy as np
from evozero import SymbolicRegressor

# A dataset generated from a known law (x2 is deliberately irrelevant).
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(2000, 3))
y = X[:, 0] ** 2 - X[:, 0] * X[:, 1] + np.sin(X[:, 1])

model = SymbolicRegressor(
    population_size=3000,
    generations=200,
    unary_operators=("sin", "cos", "exp", "square"),
    device="auto",
    random_state=0,
    max_time=45,        # optional wall-clock budget in seconds
)
model.fit(X, y)
```

## Read the results

```python
print(model.best_equation_)   # e.g. x0*(x0 - x1) + sin(x1)
print(model.score(X, y))      # R^2, e.g. 0.99+
print(model.device_)          # cuda:0 / mps / cpu
```

`evozero` never returns a single answer — it returns the **Pareto front** of the
accuracy/complexity trade-off, exactly like Eureqa:

```python
for row in model.pareto_front_:
    print(f"nodes={row['complexity']:2d}  R2={row['r2']:.4f}  {row['equation']}")
```

```text
nodes= 1  R2=-0.00  0.067*x0 + 3.1
nodes= 2  R2= 0.51  1.03*x0**2 - 0.036
nodes= 3  R2= 0.58  -1.03*x0*x1 + 3.0
nodes= 5  R2= 0.97  1.0*x0*(x0 - x1) - 0.019
nodes= 9  R2= 0.99  1.0*(x0 - 0.36)*(x0 - x1 + 0.35) + 0.13
```

Pick the "knee" that balances simplicity and accuracy for your use case.

## Predict and export

```python
# Predict with the best equation (or any Pareto entry via index=)
y_hat = model.predict(X)

# Export to other formats
model.to_latex()            # 'x_{0} (x_{0} - x_{1}) + \\sin(x_{1})'
model.to_sympy()            # a SymPy expression you can manipulate
f = model.to_numpy_func()   # a pure-NumPy callable, no torch required
f(X)                        # == model.predict(X)
```

## Command line

```bash
evozero fit data.csv          # last column is the target
evozero --version
```

## Where to next

- {doc}`guide/symbolic_regression` — every parameter and exporter.
- {doc}`guide/automl_zero` — evolve the learning algorithm itself.
- {doc}`guide/dashboard` — watch the search live.
- {doc}`concepts/how_it_works` — the GPU architecture under the hood.

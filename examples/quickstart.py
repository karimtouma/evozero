"""Minimal end-to-end example: discover an equation and export it.

Run with:  python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from evozero import SymbolicRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(600, 2))
    y = X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])

    model = SymbolicRegressor(
        population_size=3000,
        generations=120,
        unary_operators=("sin", "cos", "square"),
        device="auto",
        random_state=0,
        verbose=1,
    )
    model.fit(X, y)

    print("best equation :", model.best_equation_)
    print("R^2           :", round(model.score(X, y), 6))
    print("LaTeX         :", model.to_latex())
    print("Pareto front  :")
    for row in model.pareto_front_:
        print(f"  n={row['complexity']:2d}  R2={row['r2']:.4f}  {row['equation']}")


if __name__ == "__main__":
    main()

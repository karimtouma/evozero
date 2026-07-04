"""Tests for the symbolic regression estimator (run on CPU in CI)."""

from __future__ import annotations

import numpy as np

from evozero import SymbolicRegressor


def test_fit_predict_and_recover(toy_regression: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = toy_regression
    model = SymbolicRegressor(
        population_size=800, generations=60, n_islands=4,
        unary_operators=("sin", "cos", "square"), device="cpu", random_state=0,
    )
    model.fit(X, y)

    assert model.n_features_in_ == 2
    assert hasattr(model, "best_equation_")
    assert len(model.pareto_front_) >= 1

    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    # this target is exactly representable -> should fit very well
    assert model.score(X, y) > 0.9


def test_export_roundtrip(toy_regression: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = toy_regression
    model = SymbolicRegressor(population_size=400, generations=30, device="cpu", random_state=1)
    model.fit(X, y)

    latex = model.to_latex()
    assert isinstance(latex, str) and len(latex) > 0

    f = model.to_numpy_func()
    np_pred = f(X)
    torch_pred = model.predict(X)
    # NumPy export and the torch path should agree closely
    assert np.allclose(np_pred, torch_pred, rtol=1e-3, atol=1e-3)


def test_get_set_params() -> None:
    model = SymbolicRegressor(population_size=123)
    assert model.get_params()["population_size"] == 123
    model.set_params(population_size=456)
    assert model.population_size == 456

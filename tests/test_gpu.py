"""GPU-only tests. Auto-skipped without a CUDA device (see conftest.py)."""

from __future__ import annotations

import numpy as np
import pytest

from evozero import SymbolicRegressor


@pytest.mark.gpu
def test_symbolic_regression_on_cuda() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(2000, 2))
    y = X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])

    model = SymbolicRegressor(
        population_size=3000,
        generations=120,
        max_time=45,
        unary_operators=("sin", "cos", "square"),
        device="cuda",
        random_state=0,
    )
    model.fit(X, y)

    assert model.device_.type == "cuda"
    assert model.score(X, y) > 0.95


@pytest.mark.gpu
def test_predict_matches_numpy_export_on_cuda() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(500, 2))
    y = X[:, 0] ** 2 + X[:, 1]
    model = SymbolicRegressor(
        population_size=1000, generations=40, device="cuda", random_state=0
    ).fit(X, y)
    # the pure-NumPy export (float64) must agree with the CUDA predict path (float32) to ~1%
    assert np.allclose(model.to_numpy_func()(X), model.predict(X), rtol=1e-2, atol=1e-2)

"""scikit-learn interoperability (skipped if scikit-learn is not installed)."""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402

from evozero import SymbolicRegressor  # noqa: E402


def test_clone_roundtrips_params() -> None:
    est = SymbolicRegressor(population_size=321, n_islands=3, device="cpu", random_state=1)
    cloned = clone(est)
    assert isinstance(cloned, SymbolicRegressor)
    assert cloned.get_params() == est.get_params()
    # clone must return a fresh, unfitted estimator
    assert not hasattr(cloned, "best_equation_")


def test_get_params_are_constructor_args() -> None:
    est = SymbolicRegressor()
    # every reported param must be accepted by __init__ (so clone/GridSearchCV work)
    SymbolicRegressor(**est.get_params())

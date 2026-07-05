"""Tests for the symbolic regression estimator (run on CPU in CI)."""

from __future__ import annotations

import numpy as np
import pytest

from evozero import Equation, SymbolicRegressor


@pytest.fixture(scope="module")
def fitted(toy_regression: tuple[np.ndarray, np.ndarray]) -> SymbolicRegressor:
    """Fit one model once and share it across assertions (keeps the suite fast)."""
    X, y = toy_regression
    model = SymbolicRegressor(
        population_size=800,
        generations=60,
        n_islands=4,
        unary_operators=("sin", "cos", "square"),
        device="cpu",
        random_state=0,
    )
    return model.fit(X, y)


def test_fit_sets_attributes(fitted: SymbolicRegressor) -> None:
    assert fitted.n_features_in_ == 2
    assert isinstance(fitted.best_equation_, Equation)
    assert len(fitted.pareto_front_) >= 1
    assert str(fitted.device_) == "cpu"


def test_recovers_signal(fitted: SymbolicRegressor, toy_regression) -> None:
    X, y = toy_regression
    assert fitted.score(X, y) > 0.9  # this target is exactly representable


def test_predict_shape(fitted: SymbolicRegressor, toy_regression) -> None:
    X, _ = toy_regression
    pred = fitted.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred).all()


def test_pareto_front_sorted(fitted: SymbolicRegressor) -> None:
    comps = [row["complexity"] for row in fitted.pareto_front_]
    assert comps == sorted(comps)
    for row in fitted.pareto_front_:
        assert set(row) >= {"complexity", "loss", "r2", "equation"}
        assert isinstance(row["equation"], Equation)


def test_export_roundtrip(fitted: SymbolicRegressor, toy_regression) -> None:
    X, _ = toy_regression
    assert isinstance(fitted.to_latex(), str)
    assert fitted.to_latex()
    f = fitted.to_numpy_func()  # pure NumPy (float64), no torch
    # the engine computes in float32, the NumPy export in float64 -> agree to ~1%
    assert np.allclose(f(X), fitted.predict(X), rtol=1e-2, atol=1e-2)


def test_predict_by_index(fitted: SymbolicRegressor, toy_regression) -> None:
    X, _ = toy_regression
    simplest = fitted.predict(X, index=0)
    assert simplest.shape == (X.shape[0],)


def test_reproducible(toy_regression) -> None:
    X, y = toy_regression
    kw = {"population_size": 400, "generations": 30, "device": "cpu", "random_state": 7}
    a = SymbolicRegressor(**kw).fit(X, y)
    b = SymbolicRegressor(**kw).fit(X, y)
    assert str(a.best_equation_) == str(b.best_equation_)


def test_operator_name_aliases(toy_regression) -> None:
    X, y = toy_regression
    # engine names instead of symbols must work identically
    model = SymbolicRegressor(
        binary_operators=("add", "sub", "mul"),
        unary_operators=("square", "sin"),
        population_size=400,
        generations=30,
        device="cpu",
        random_state=0,
    )
    model.fit(X, y)
    assert model.score(X, y) > 0.5


def test_get_set_params() -> None:
    model = SymbolicRegressor(population_size=123)
    assert model.get_params()["population_size"] == 123
    assert model.set_params(population_size=456) is model
    assert model.population_size == 456


def test_reduce_matches_full_path() -> None:
    # run_population_reduce (streamed, memory-safe) must match the full [P,N] fit_ab /
    # score_ab path it replaces at large N -- FIT mode and SCORE mode, to fp32 tolerance.
    import torch

    from evozero.core import _sr_engine as engine

    dev = torch.device("cpu")
    rng = np.random.default_rng(0)
    ps = engine.PrimSet(2, ["sin", "cos"], ["add", "sub", "mul"], named=[])
    pop = []
    while len(pop) < 120:
        c, k = engine.gen_tree(ps, int(rng.integers(2, 5)), "grow", rng)
        c, k = engine.simplify_prefix(c, k, ps)
        if engine._ok_size(c, ps, 40, 9):
            pop.append((c, k))
    n = 4096
    x = rng.uniform(-3, 3, size=(n, 2))
    y = x[:, 0] ** 2 + x[:, 0] * x[:, 1] + np.sin(x[:, 1])
    xt = torch.tensor(x.T, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    codes, consts, _ = engine.batch_postfix(pop, ps)

    yh = engine.run_population(codes, consts, xt, ps, dev)
    a0, b0, _, r20 = engine.fit_ab(yh, yt)
    a1, b1, mse1, r21 = engine.run_population_reduce(codes, consts, xt, yt, ps, dev, chunk_n=512)
    assert (a0 - a1).abs().max() < 1e-4
    assert (b0 - b1).abs().max() < 1e-4
    assert (r20 - r21).abs().max() < 1e-4
    _, r2_s = engine.score_ab(yh, yt, a0, b0)
    _, _, _, r22 = engine.run_population_reduce(
        codes, consts, xt, yt, ps, dev, a=a0, b=b0, chunk_n=512
    )
    assert (r2_s - r22).abs().max() < 1e-4
    # honesty invariants: streamed reduction never reports r2>1 or negative mse
    assert not bool((r21 > 1 + 1e-5).any())
    assert not bool((mse1 < 0).any())


def test_case_subsampling_fits_and_is_inert_when_gated_off(toy_regression) -> None:
    # Forcing the subsample gate on (threshold=0) must still fit; and with the default
    # threshold on small data it must be inert (identical to not passing subsample_size).
    X, y = toy_regression
    common = {
        "population_size": 500,
        "generations": 25,
        "n_islands": 3,
        "device": "cpu",
        "random_state": 0,
    }
    forced = SymbolicRegressor(subsample_size=256, subsample_threshold=0, **common).fit(X, y)
    assert forced.score(X, y) > 0.9  # full-data refit keeps predict accurate

    base = SymbolicRegressor(**common).fit(X, y)
    inert = SymbolicRegressor(subsample_size=256, **common).fit(X, y)  # gated off (N < 50k)
    assert str(base.best_equation_) == str(inert.best_equation_)


def test_dalex_selection_fits(toy_regression) -> None:
    # GPU-native lexicase selection must run end-to-end and be sklearn-round-trippable
    # even on CPU (the [P, N] matmul path just runs on the CPU tensor there).
    X, y = toy_regression
    model = SymbolicRegressor(
        population_size=600,
        generations=40,
        n_islands=3,
        selection="dalex",
        dalex_sigma=3.0,
        device="cpu",
        random_state=0,
    )
    params = model.get_params()
    assert params["selection"] == "dalex"
    assert params["dalex_sigma"] == 3.0
    model.fit(X, y)
    assert model.score(X, y) > 0.9  # this target is exactly representable


def test_1d_X_raises() -> None:
    model = SymbolicRegressor(device="cpu")
    with pytest.raises(ValueError, match="2-D"):
        model.fit(np.arange(10.0), np.arange(10.0))

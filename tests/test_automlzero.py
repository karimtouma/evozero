"""Tests for the AutoML-Zero engine (LearnerSearch / EvolvedLearner / Task)."""

from __future__ import annotations

import numpy as np
import pytest

from evozero import EvolvedLearner, LearnerSearch, Task


def _linear_task(seed: int, n: int = 200, f: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=f)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = X @ w
    return X, y


def test_task_split() -> None:
    X, y = _linear_task(0)
    t = Task(X, y, val_fraction=0.25, random_state=0)
    assert len(t.Xtr) + len(t.Xval) == len(y)
    assert t.Xtr.shape[1] == X.shape[1]
    assert abs(float(np.mean(np.concatenate([t.ytr, t.yval])))) < 1e-4  # centered


def test_handcoded_vm_learns() -> None:
    """The VM itself must be able to train (hand-coded linear-GD reaches high R²)."""
    import torch

    from evozero.core import _automlzero_engine as az

    tasks = az.to_device(az.make_tasks(3, 4, 200, seed=1), torch.device("cpu"))
    algo = az.handcoded_linreg_gd(lr=0.5)
    r2, _per = az.eval_algo(algo, tasks, 30, 4, 4, 4, torch.device("cpu"))
    assert r2 > 0.9


def test_learner_search_and_reuse() -> None:
    tasks = [Task(*_linear_task(s)) for s in range(4)]
    search = LearnerSearch(
        population_size=80,
        n_meta_generations=40,
        steps=20,
        device="cpu",
        random_state=0,
    )
    search.fit(tasks)
    assert isinstance(search.best_program_, EvolvedLearner)
    assert np.isfinite(search.best_score_)

    learner = search.export_learner()
    src = learner.to_python_source()
    assert "def setup" in src and "def predict" in src and "def learn" in src

    X, y = _linear_task(99)
    learner.fit(X, y)
    pred = learner.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred).all()


@pytest.mark.gpu
def test_learner_search_gpu() -> None:
    tasks = [Task(*_linear_task(s)) for s in range(4)]
    search = LearnerSearch(
        population_size=120,
        n_meta_generations=60,
        max_time=20,
        device="cuda",
        random_state=0,
    )
    search.fit(tasks)
    assert str(search._device).startswith("cuda")

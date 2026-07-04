"""Shared test fixtures; auto-skip ``gpu``-marked tests without a CUDA device."""

from __future__ import annotations

import numpy as np
import pytest


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GPU tests when no accelerator is present."""
    if _cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device; run under nightly-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def toy_regression() -> tuple[np.ndarray, np.ndarray]:
    """A small, exactly-recoverable regression problem."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2)).astype(np.float64)
    y = X[:, 0] ** 2 + X[:, 0] * X[:, 1] + np.sin(X[:, 1])
    return X, y

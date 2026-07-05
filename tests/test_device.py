"""Tests for device resolution and the torch-optionality guard."""

from __future__ import annotations

import pytest


def test_resolve_explicit_cpu() -> None:
    import torch

    from evozero._device import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_a_device() -> None:
    import torch

    from evozero._device import resolve_device

    dev = resolve_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_require_torch_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from evozero import _import_utils

    monkeypatch.setattr(_import_utils, "HAS_TORCH", False)
    with pytest.raises(ImportError, match="PyTorch"):
        _import_utils.require_torch()


def test_is_torch_available() -> None:
    from evozero._import_utils import is_torch_available

    assert is_torch_available() is True  # torch is installed in the test env

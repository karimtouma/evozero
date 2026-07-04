"""Smoke tests for the public API surface and torch isolation."""

from __future__ import annotations

import importlib


def test_public_api_exports() -> None:
    import evozero

    for name in ("SymbolicRegressor", "LearnerSearch", "EvolvedLearner", "Task",
                 "launch_dashboard", "DashboardHandle", "__version__"):
        assert name in dir(evozero)
        assert getattr(evozero, name) is not None


def test_import_does_not_import_torch() -> None:
    # A fresh `import evozero` must not pull in torch (PEP 562 lazy attributes).
    import sys

    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    sys.modules.pop("evozero", None)
    importlib.import_module("evozero")
    assert "torch" not in sys.modules, "importing evozero must not import torch"


def test_dashboard_asset_packaged() -> None:
    import importlib.resources

    html = (importlib.resources.files("evozero.dashboard")
            .joinpath("_static/index.html").read_text(encoding="utf-8"))
    assert "<canvas" in html or "<!doctype" in html.lower()

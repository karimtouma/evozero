"""Smoke tests for the public API surface and torch isolation."""

from __future__ import annotations

import subprocess
import sys

PUBLIC = (
    "SymbolicRegressor",
    "Equation",
    "LearnerSearch",
    "EvolvedLearner",
    "Task",
    "launch_dashboard",
    "DashboardHandle",
    "__version__",
)


def test_public_api_exports() -> None:
    import evozero

    for name in PUBLIC:
        assert name in dir(evozero)
        assert getattr(evozero, name) is not None


def test_dir_matches_all() -> None:
    import evozero

    assert set(dir(evozero)) == set(evozero.__all__)


def test_unknown_attribute_raises() -> None:
    import evozero

    try:
        evozero.NoSuchThing  # noqa: B018
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for an unknown attribute")


def test_import_does_not_import_torch() -> None:
    # Run in a clean interpreter: deleting torch from sys.modules in-process would
    # corrupt torch's native extensions. `import evozero` must be torch-free (PEP 562).
    code = "import evozero, sys; assert 'torch' not in sys.modules; print(evozero.__version__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip(), "evozero should expose a version"


def test_dashboard_asset_packaged() -> None:
    import importlib.resources

    html = (
        importlib.resources.files("evozero.dashboard")
        .joinpath("_static/index.html")
        .read_text(encoding="utf-8")
    )
    assert "<canvas" in html

"""Tests for the ``evozero`` command-line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evozero.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "evozero" in out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_fit_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2))
    y = X[:, 0] ** 2 + X[:, 1]
    data = np.column_stack([X, y])
    csv = tmp_path / "d.csv"
    np.savetxt(csv, data, delimiter=",", header="x0,x1,y", comments="")

    rc = main(["fit", str(csv), "--pop", "400", "--generations", "40", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "R^2" in out and "best:" in out

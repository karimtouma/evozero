"""Task automation for evozero (run with ``nox``)."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
PYTHONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session
def lint(session: nox.Session) -> None:
    """Lint and format-check with Ruff."""
    session.install("ruff>=0.6")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Static type-check the public package with mypy (strict)."""
    session.install("-e", ".", "mypy>=1.11")
    session.run("mypy")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the CPU test suite with coverage."""
    session.install("-e", ".", "pytest>=8.0", "pytest-cov>=5.0", "hypothesis>=6.100")
    session.run("pytest", "-m", "not gpu", "--cov=evozero", "--cov-branch", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the Sphinx documentation."""
    session.install("-e", ".", "--group", "docs")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")

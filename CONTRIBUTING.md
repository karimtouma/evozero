# Contributing to evozero

Thanks for your interest! evozero is developed in the open.

## Development setup

```bash
git clone https://github.com/karimtouma/evozero
cd evozero
uv sync --group dev            # or: pip install -e ".[all]" and the dev tools
pre-commit install
```

## Workflow

1. Create a branch from `main`.
2. Make your change with tests and NumPy-style docstrings.
3. Run the checks locally:
   ```bash
   uv run ruff check . && uv run ruff format .
   uv run mypy
   uv run pytest -m "not gpu"
   ```
4. Add a changelog fragment in `changelog.d/` (e.g. `changelog.d/123.feature.md`).
5. Open a pull request. CI runs lint + types + the CPU test matrix.

## Conventions

- **Style**: Ruff (lint + format), line length 100.
- **Types**: `mypy --strict` on `src/evozero`; ship `py.typed`.
- **Docstrings**: NumPy convention.
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` …).
- **`torch` isolation**: only `evozero.core` may import torch at module top level.
- **GPU tests**: mark with `@pytest.mark.gpu`; they auto-skip without CUDA.

## Releasing (maintainers)

See the release runbook in the project docs — tag `vX.Y.Z`, and
`release.yml` builds and publishes to PyPI via Trusted Publishing (OIDC).

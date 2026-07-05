# Installation

## Requirements

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.4 (the compute engine). `evozero` does not pin a CUDA build — you
  choose the wheel that matches your driver.
- **NumPy** ≥ 1.24 and **SymPy** ≥ 1.12 (pulled automatically).

A CUDA-capable GPU is optional but recommended: the engine runs on CPU too, just slower.

## From PyPI

::::{tab-set}
:::{tab-item} CPU
```bash
pip install evozero
```
This brings the CPU build of PyTorch from PyPI — fine for small problems, development,
and CI.
:::

:::{tab-item} NVIDIA GPU (CUDA)
```bash
pip install "evozero[cuda]" --index-url https://download.pytorch.org/whl/cu128
```
Pick the index URL that matches your CUDA version (`cu121`, `cu124`, `cu128`, …). See the
[PyTorch install matrix](https://pytorch.org/get-started/locally/).
:::

:::{tab-item} Apple Silicon (MPS)
```bash
pip install evozero
```
The macOS PyTorch wheel includes the Metal (MPS) backend; `device="auto"` will use it.
:::
::::

Optional extras:

| Extra | Adds | For |
|-------|------|-----|
| `[cuda]` | GPU PyTorch (via the index URL) | NVIDIA GPUs |
| `[dashboard]` | `pandas` | richer dashboard tables |
| `[sklearn]` | `scikit-learn` | `Pipeline` / `GridSearchCV` interop |
| `[all]` | dashboard + sklearn | everything |

## From source (development)

`evozero` uses [uv](https://docs.astral.sh/uv/) for environment and lock management and
[hatchling](https://hatch.pypa.io/) + `hatch-vcs` for git-tag versioning.

```bash
git clone https://github.com/karimtouma/evozero
cd evozero
uv sync --extra cuda        # or `uv sync` for CPU
uv run pytest -m "not gpu"  # run the CPU test suite
```

## Verifying the install

```python
import evozero
print(evozero.__version__)          # e.g. 0.1.0

from evozero import SymbolicRegressor
m = SymbolicRegressor(device="auto")
# m.device_ is set after .fit(); resolve manually to check the accelerator:
from evozero._device import resolve_device
print(resolve_device("auto"))       # cuda:0 / mps / cpu
```

```{note}
`import evozero` does **not** import `torch`. The first `torch` import happens when you
construct a `SymbolicRegressor()` or call `.fit()`. If PyTorch is missing you'll get a
clear `ImportError` with install instructions.
```

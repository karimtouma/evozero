"""Optional-dependency helpers so ``import evozero`` never requires torch."""

from __future__ import annotations

import importlib.util
from functools import lru_cache

#: ``True`` if PyTorch is importable. Computed with ``find_spec`` (imports nothing).
HAS_TORCH: bool = importlib.util.find_spec("torch") is not None


@lru_cache(maxsize=1)
def is_torch_available() -> bool:
    """Return whether PyTorch can be imported (cached)."""
    return HAS_TORCH


def require_torch() -> None:
    """Raise a helpful ``ImportError`` if PyTorch is missing.

    Raises
    ------
    ImportError
        If ``torch`` is not installed.
    """
    if not HAS_TORCH:
        raise ImportError(
            "This evozero feature needs PyTorch. Install it with:\n"
            "  pip install 'evozero[cuda]'   # GPU (CUDA)\n"
            "  pip install torch             # CPU\n"
            "See https://pytorch.org/get-started/ for the right wheel."
        )

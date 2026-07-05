"""evozero: GPU-native evolutionary computation.

Evolve **formulas** (symbolic regression) and **learning algorithms**
(AutoML-Zero) from zero, on the GPU.

The public API is imported lazily (:pep:`562`) so ``import evozero`` never
triggers ``import torch`` — export utilities work on machines without a GPU.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source checkout without build metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "DashboardHandle",
    "Equation",
    "EvolvedLearner",
    "LearnerSearch",
    "SymbolicRegressor",
    "Task",
    "__version__",
    "launch_dashboard",
]

_LAZY = {
    "SymbolicRegressor": "evozero.sr",
    "Equation": "evozero.sr",
    "LearnerSearch": "evozero.automlzero",
    "EvolvedLearner": "evozero.automlzero",
    "Task": "evozero.automlzero",
    "launch_dashboard": "evozero.dashboard",
    "DashboardHandle": "evozero.dashboard",
}


def __getattr__(name: str) -> object:  # :pep:`562`
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'evozero' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # let type checkers see the real symbols
    from .automlzero import EvolvedLearner, LearnerSearch, Task
    from .dashboard import DashboardHandle, launch_dashboard
    from .sr import Equation, SymbolicRegressor

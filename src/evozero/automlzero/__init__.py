"""AutoML-Zero: evolve the learning algorithm itself.

The genome is three instruction sequences (``Setup`` / ``Predict`` / ``Learn``)
over a typed-register VM. Meta-trained across tasks so the discovered algorithm
generalizes to *unseen* tasks (learning to learn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["LearnerSearch", "EvolvedLearner", "Task"]

_DIMS = (4, 4, 4)  # (n_scalars, n_vectors, n_per_example) registers


class Task:
    """A single meta-training task ``(X, y)`` split into train/val internally."""

    def __init__(self, X: "ArrayLike", y: "ArrayLike", *, val_fraction: float = 0.3,
                 random_state: int = 0) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        y = y - y.mean()  # the VM has no bias term; center the target
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(y))
        cut = int(len(y) * (1 - val_fraction))
        self.Xtr, self.ytr = X[idx[:cut]], y[idx[:cut]]
        self.Xval, self.yval = X[idx[cut:]], y[idx[cut:]]

    def _to_device(self, device: Any) -> tuple[Any, Any, Any, Any]:
        import torch

        return (
            torch.tensor(self.Xtr, device=device), torch.tensor(self.ytr, device=device),
            torch.tensor(self.Xval, device=device), torch.tensor(self.yval, device=device),
        )


class EvolvedLearner:
    """A discovered learning algorithm; a reusable ``fit``/``predict`` estimator."""

    def __init__(self, algo: dict[str, Any], device: Any, steps: int = 30,
                 dims: tuple[int, int, int] = _DIMS) -> None:
        self._algo = algo
        self._device = device
        self._steps = steps
        self._nS, self._nV, self._nP = dims

    def fit(self, X: "ArrayLike", y: "ArrayLike") -> "EvolvedLearner":
        """Train from scratch on ``(X, y)`` using the evolved ``Learn`` rule."""
        import torch

        from ..core import _automlzero_engine as az

        X = torch.tensor(np.asarray(X, dtype=np.float32), device=self._device)
        y = torch.tensor(np.asarray(y, dtype=np.float32).ravel(), device=self._device)
        mem = az.Mem(self._nS, self._nV, self._nP, X.shape[1], X.shape[0], self._device)
        az.run(self._algo["setup"], mem, X, y, self._nS, self._nV, self._nP)
        for _ in range(self._steps):
            az.run(self._algo["predict"], mem, X, y, self._nS, self._nV, self._nP)
            az.run(self._algo["learn"], mem, X, y, self._nS, self._nV, self._nP)
        self._S, self._V = mem.S.clone(), mem.V.clone()
        return self

    def predict(self, X: "ArrayLike") -> "NDArray[np.float64]":
        """Predict with the learned weights."""
        import torch

        from ..core import _automlzero_engine as az

        X = torch.tensor(np.asarray(X, dtype=np.float32), device=self._device)
        mem = az.Mem(self._nS, self._nV, self._nP, X.shape[1], X.shape[0], self._device)
        mem.S, mem.V = self._S.clone(), self._V.clone()
        az.run(self._algo["predict"], mem, X, X.new_zeros(X.shape[0]),
               self._nS, self._nV, self._nP)
        return mem.P[az.OUT_P].detach().cpu().numpy().astype(np.float64)

    def to_python_source(self) -> str:
        """Return the evolved algorithm as readable pseudo-code."""
        from ..core import _automlzero_engine as az

        lines: list[str] = []
        for fn in ("setup", "predict", "learn"):
            lines.append(f"def {fn}:")
            body = self._algo[fn]
            lines.extend(az.fmt_instr(ins) for ins in body) if body else lines.append("    (empty)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_python_source()


class LearnerSearch:
    """Evolve a learning algorithm over a distribution of tasks.

    Parameters
    ----------
    population_size : int, default=200
    n_meta_generations : int, default=400
    steps : int, default=30
        Inner training steps per task when scoring an algorithm.
    max_time : float or None, default=None
        Wall-clock budget in seconds.
    device : str, default="auto"
    random_state : int or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    best_program_ : EvolvedLearner
    best_score_ : float
    """

    def __init__(self, *, population_size: int = 200, n_meta_generations: int = 400,
                 steps: int = 30, max_time: float | None = None, device: str = "auto",
                 random_state: int | None = None, verbose: int = 0) -> None:
        self.population_size = population_size
        self.n_meta_generations = n_meta_generations
        self.steps = steps
        self.max_time = max_time
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, tasks: "list[Task]") -> "LearnerSearch":
        """Meta-evolve a learning algorithm on ``tasks``."""
        from .._device import resolve_device
        from ..core import _automlzero_engine as az

        device = resolve_device(self.device)
        seed = 0 if self.random_state is None else int(self.random_state)
        converted = [t._to_device(device) for t in tasks]
        # hold out the last task for generalization reporting (or reuse all if few)
        test = converted[-1:] if len(converted) > 2 else converted
        best, _hof = az.evolve(
            converted, test, device, T=self.steps,
            pop_size=self.population_size, generations=self.n_meta_generations,
            time_budget=self.max_time, seed=seed, verbose=bool(self.verbose),
        )
        self.best_score_ = float(best[0])
        self.best_program_ = EvolvedLearner(best[1], device, steps=self.steps)
        self._device = device
        return self

    def export_learner(self) -> EvolvedLearner:
        """Return the discovered :class:`EvolvedLearner`."""
        return self.best_program_

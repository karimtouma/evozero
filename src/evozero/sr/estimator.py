"""Scikit-learn-style symbolic regression estimator.

Thin, well-typed facade over the proven GPU engine in
:mod:`evozero.core._sr_engine` (island-model GP + tensorized postfix
interpreter + linear scaling + constant optimization + Pareto front).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["Equation", "SymbolicRegressor"]

# Public operator symbols -> engine primitive names.
_BINARY_ALIASES = {"+": "add", "-": "sub", "*": "mul", "/": "div", "aq": "aq"}


class Equation:
    """A single evolved expression from the Pareto front.

    Attributes
    ----------
    complexity : int
        Number of nodes.
    r2 : float
        Validation coefficient of determination.
    """

    def __init__(self, entry: dict[str, Any], primset: Any) -> None:
        self._entry = entry
        self._ps = primset
        self.complexity: int = int(entry["complexity"])
        self.r2: float = float(entry.get("r2_val", float("nan")))

    def to_sympy(self, simplify: bool = True) -> Any:
        """Return the expression as a SymPy object."""
        from ..core import _sr_engine as engine

        return engine.to_sympy(
            self._entry["code"],
            self._entry["const"],
            self._ps,
            self._entry["a"],
            self._entry["b"],
            simplify=simplify,
        )

    def to_latex(self) -> str:
        """Return the expression as a LaTeX string."""
        import sympy as sp

        return sp.latex(self.to_sympy())

    def __str__(self) -> str:
        return str(self.to_sympy())

    def __repr__(self) -> str:
        return f"Equation(complexity={self.complexity}, r2={self.r2:.4f}, expr={self})"


class SymbolicRegressor:
    """Evolve an interpretable equation ``y = f(X)`` on the GPU.

    Follows the scikit-learn estimator contract: no work in ``__init__``,
    ``fit`` returns ``self``, learned attributes carry a trailing underscore.

    Parameters
    ----------
    population_size : int, default=3000
        Total number of individuals across all islands.
    n_islands : int, default=6
        Number of semi-independent sub-populations (ring migration).
    migration_interval : int, default=8
        Migrate every this many generations.
    generations : int, default=200
        Maximum number of generations.
    max_time : float or None, default=None
        Wall-clock budget in seconds (overrides ``generations`` when reached).
    binary_operators : tuple of str, default=("+", "-", "*", "/")
        Binary primitives (symbols or engine names).
    unary_operators : tuple of str, default=("sin", "cos", "exp", "log", "sqrt")
        Unary primitives.
    max_size : int, default=40
        Maximum number of nodes per expression.
    max_depth : int, default=9
        Maximum tree depth.
    parsimony_coefficient : float, default=0.006
        Weight of the (per-operator) complexity term in the selection fitness.
    restart_patience : int, default=30
        Generations without island improvement before that island restarts.
    const_opt_interval : int, default=5
        Optimize constants of the best models every this many generations (``0`` disables).
    selection : {"tournament", "dalex"}, default="tournament"
        Parent-selection operator. ``"tournament"`` is the classic size-``tournament``
        tournament. ``"dalex"`` is GPU-native lexicase (DALex): each parent is chosen by
        a randomly weighted aggregation over the per-case error matrix, computed as a
        single ``[P, N] @ [N, k]`` matmul. Lexicase is O(T·N²) and impractical on CPU,
        but near-free on GPU since the per-case error tensor is already materialized;
        it improves fit on problems with many free constants.
    dalex_sigma : float, default=3.0
        Particularity pressure for ``selection="dalex"`` (softmax temperature of the
        random case weights). Higher values concentrate selection on fewer cases
        (stronger lexicase behavior); lower values approach mean-error selection.
    subsample_size : int or None, default=None
        Number of training cases used to evaluate fitness each generation when the
        training set is large (``>= subsample_threshold``). ``None`` means auto
        (``2048``). This is what lets the engine scale to ``N >= 1e5`` without
        materializing the ``[P, N]`` prediction tensor; below the threshold the full
        data is used and behavior is unchanged. Exported models' linear-scaling
        coefficients are re-fit on the full training set (see ``subsample_refit_full``).
    val_subsample_size : int, default=8192
        Number of validation cases (drawn once, fixed) used for model selection when
        subsampling is active — a stationary yardstick for the best/Pareto choice.
    subsample_threshold : int, default=50000
        Training-set size at/above which case subsampling activates. Set very large to
        force full-data evaluation regardless of ``N``.
    subsample_resample_interval : int, default=1
        Redraw the training subsample every this many generations (``1`` = every gen).
    subsample_refit_full : bool, default=True
        Re-fit the linear-scaling coefficients of the exported models on the full
        training set at the end of the search (memory-safe, streamed), so ``predict``
        uses full-data coefficients rather than subsample-fit ones.
    device : str, default="auto"
        ``"auto" | "cpu" | "cuda" | "cuda:N" | "mps"``.
    random_state : int or None, default=None
        Seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    best_equation_ : Equation
    pareto_front_ : list of dict
        ``[{"complexity", "loss", "r2", "equation"}, ...]`` sorted by complexity.
    device_ : torch.device
    n_features_in_ : int
    """

    def __init__(
        self,
        *,
        population_size: int = 3000,
        n_islands: int = 6,
        migration_interval: int = 8,
        generations: int = 200,
        max_time: float | None = None,
        binary_operators: tuple[str, ...] = ("+", "-", "*", "/"),
        unary_operators: tuple[str, ...] = ("sin", "cos", "exp", "log", "sqrt"),
        max_size: int = 40,
        max_depth: int = 9,
        parsimony_coefficient: float = 0.006,
        restart_patience: int = 30,
        const_opt_interval: int = 5,
        selection: str = "tournament",
        dalex_sigma: float = 3.0,
        subsample_size: int | None = None,
        val_subsample_size: int = 8192,
        subsample_threshold: int = 50000,
        subsample_resample_interval: int = 1,
        subsample_refit_full: bool = True,
        device: str = "auto",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.population_size = population_size
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.generations = generations
        self.max_time = max_time
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.max_size = max_size
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.restart_patience = restart_patience
        self.const_opt_interval = const_opt_interval
        self.selection = selection
        self.dalex_sigma = dalex_sigma
        self.subsample_size = subsample_size
        self.val_subsample_size = val_subsample_size
        self.subsample_threshold = subsample_threshold
        self.subsample_resample_interval = subsample_resample_interval
        self.subsample_refit_full = subsample_refit_full
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters (scikit-learn compatible)."""
        return {
            k: getattr(self, k)
            for k in (
                "population_size",
                "n_islands",
                "migration_interval",
                "generations",
                "max_time",
                "binary_operators",
                "unary_operators",
                "max_size",
                "max_depth",
                "parsimony_coefficient",
                "restart_patience",
                "const_opt_interval",
                "selection",
                "dalex_sigma",
                "subsample_size",
                "val_subsample_size",
                "subsample_threshold",
                "subsample_resample_interval",
                "subsample_refit_full",
                "device",
                "random_state",
                "verbose",
            )
        }

    def set_params(self, **params: Any) -> SymbolicRegressor:
        """Set estimator parameters (scikit-learn compatible)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Any = None) -> SymbolicRegressor:
        """Run the evolutionary search and store the discovered equations."""
        from .._device import resolve_device
        from ..core import _sr_engine as engine

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D [n_samples, n_features].")
        self.n_features_in_ = X.shape[1]
        seed = 0 if self.random_state is None else int(self.random_state)
        device = resolve_device(self.device)

        binary = [_BINARY_ALIASES.get(o, o) for o in self.binary_operators]
        unary = list(self.unary_operators)
        primset = engine.PrimSet(self.n_features_in_, unary, binary)

        xcol = X.T  # engine expects [n_features, n_samples]
        (xtr, ytr), (xva, yva), _ = engine.split_data(xcol, y, seed=seed)
        best, archive = engine.evolve(
            xtr,
            ytr,
            xva,
            yva,
            primset,
            device,
            pop_size=self.population_size,
            generations=self.generations,
            max_len=self.max_size,
            max_depth=self.max_depth,
            cw=self.parsimony_coefficient,
            n_islands=self.n_islands,
            migration_interval=self.migration_interval,
            restart_patience=self.restart_patience,
            const_opt_interval=self.const_opt_interval,
            selection=self.selection,
            dalex_sigma=self.dalex_sigma,
            subsample_size=(self.subsample_size if self.subsample_size is not None else 2048),
            val_subsample_size=self.val_subsample_size,
            subsample_threshold=self.subsample_threshold,
            subsample_resample_interval=self.subsample_resample_interval,
            subsample_refit_full=self.subsample_refit_full,
            time_budget=self.max_time,
            seed=seed,
            verbose=bool(self.verbose),
        )

        self._ps = primset
        self._device = device
        self._best = best
        self._archive = archive
        self.device_ = device
        self.best_equation_ = Equation(best, primset)
        self.pareto_front_ = [
            {
                "complexity": int(d["complexity"]),
                "loss": float(d["mse_val"]),
                "r2": float(d["r2_val"]),
                "equation": Equation(d, primset),
            }
            for d in archive
        ]
        return self

    def _entry(self, index: int | None) -> dict[str, Any]:
        return self._best if index is None else self._archive[index]

    def predict(self, X: ArrayLike, index: int | None = None) -> NDArray[np.float64]:
        """Predict with the best equation (or Pareto entry ``index``)."""
        import torch

        from ..core import _sr_engine as engine

        entry = self._entry(index)
        xarr = np.asarray(X, dtype=np.float32)
        xt = torch.from_numpy(xarr.T).to(self._device)
        codes, consts, _ = engine.batch_postfix([(entry["code"], entry["const"])], self._ps)
        yhat = engine.run_population(codes, consts, xt, self._ps, self._device)
        pred = entry["a"] * yhat[0] + entry["b"]
        return pred.detach().cpu().numpy().astype(np.float64)

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: Any = None) -> float:
        """Return the :math:`R^2` of the best equation on ``(X, y)``."""
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot

    def to_sympy(self, index: int | None = None) -> Any:
        """Best (or ``index``-th) equation as a SymPy object."""
        return Equation(self._entry(index), self._ps).to_sympy()

    def to_latex(self, index: int | None = None) -> str:
        """Best (or ``index``-th) equation as LaTeX."""
        return Equation(self._entry(index), self._ps).to_latex()

    def to_numpy_func(self, index: int | None = None) -> Callable[..., NDArray[np.float64]]:
        """Compile the equation to a pure-NumPy callable ``f(X)`` (no torch)."""
        import sympy as sp

        expr = self.to_sympy(index)
        xs = sp.symbols(f"x0:{self.n_features_in_}")
        fn = sp.lambdify(xs, expr, "numpy")

        def predict(X: ArrayLike) -> NDArray[np.float64]:
            arr = np.asarray(X, dtype=np.float64)
            return np.asarray(fn(*[arr[:, i] for i in range(arr.shape[1])]), dtype=np.float64)

        return predict

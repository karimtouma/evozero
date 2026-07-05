# Changelog

All notable changes to `evozero` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [Unreleased]

### Added

- **GPU-native lexicase selection (DALex)** via `SymbolicRegressor(selection="dalex",
  dalex_sigma=3.0)`. Lexicase is O(T·N²) and impractical on CPU, but near-free on GPU
  because the per-case error matrix `[P, N]` is already materialized — parent choice
  becomes a single `[P, N] @ [N, k]` matmul. Measured (full island engine, constants-heavy
  target, N=10⁴, 30 s budget, H100): median held-out R² **0.99477 → 0.99924** vs tournament
  at **+10 % wall-clock**. Still does not beat Operon on pure SR — this is an internal
  strategy improvement, not a victory claim.
- `benchmarks/ab_selection.py` — isolated A/B that validated the DALex-vs-tournament lever.
- `benchmarks/` — an **honest** head-to-head SR benchmark (evozero vs Operon / PySR /
  gplearn) with `BENCHMARK.md`. Finding: evozero ties the CPU tools on small data and
  **loses to Operon at N ≥ 10⁵**; it is not the fastest SR tool.

### Changed

- Local constant optimization is now **gradient-based** (LBFGS through the differentiable
  interpreter), replacing the perturbation ES — better per-evaluation search quality.

### Fixed

- Memory-safe **auto-chunking** of the tensorized interpreter stack (avoids one OOM
  source on large `N`).

## [0.1.0] - 2026-07-04

### Added

- `SymbolicRegressor` — GPU symbolic regression (island-model GP, tensorized postfix
  interpreter, linear scaling, constant optimization, Pareto front) with a scikit-learn-style
  API and `to_sympy` / `to_latex` / `to_numpy_func` exporters.
- `LearnerSearch` / `EvolvedLearner` / `Task` — AutoML-Zero: evolve a learning algorithm
  from primitive ops, meta-trained across tasks.
- `launch_dashboard` — stdlib-only live web dashboard.
- Lazy public API (PEP 562): `import evozero` does not import `torch`.

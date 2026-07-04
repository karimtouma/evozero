# Changelog

All notable changes to `evozero` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.1.0] - 2026-07-04

### Added

- `SymbolicRegressor` — GPU symbolic regression (island-model GP, tensorized postfix
  interpreter, linear scaling, constant optimization, Pareto front) with a scikit-learn-style
  API and `to_sympy` / `to_latex` / `to_numpy_func` exporters.
- `LearnerSearch` / `EvolvedLearner` / `Task` — AutoML-Zero: evolve a learning algorithm
  from primitive ops, meta-trained across tasks.
- `launch_dashboard` — stdlib-only live web dashboard.
- Lazy public API (PEP 562): `import evozero` does not import `torch`.

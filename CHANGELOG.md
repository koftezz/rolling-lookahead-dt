# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.1.0] - 2026-08-11

### Added
- Optional globally exact complete depth-3 initialization via `initial_depth=3`, implemented by enumerating the root feature and solving its two independent OCT-2 child subtrees.
- Reproducible per-subproblem feature budgets with `max_features` and `random_state`.
- A fit-wide `total_time_limit` that returns the last valid tree with a `ConvergenceWarning` when possible.
- Public `fit_status_`, `fit_time_`, `actual_depth_`, and `subproblem_diagnostics_` attributes.
- A paired cross-validation benchmark comparing depth-2 and exact depth-3 seeds through requested depths 3, 4, and 5.
- Pull-request/main CI across Python 3.9, 3.11, and 3.13, including package build checks.

### Fixed
- Enforced `min_samples_leaf` instead of silently restoring invalid feature pairs.
- Distinguished feasible time-limit incumbents from solver errors.
- Validated binary values, feature counts, and DataFrame column identity/order at prediction time.
- Made `depth` changes through `set_params()` affect the next fit.
- Counted all populated terminal leaves and reported actual realized depth/results.
- Reported `completed` when the requested maximum depth is reached, even when terminal leaves remain mixed.
- Added a safe sequential fallback when multiprocessing primitives are unavailable.
- Unified runtime and package metadata on one version source.

### Performance
- Replaced the coefficient-construction hot path with vectorized class-count matrix operations.
- Cached compiled tree-routing metadata until the tree changes.

## [2.0.1] - 2026-02-19

### Fixed
- Restored pandas 3.0 compatibility in the test data loader used on Python 3.13.

## [2.0.0] - 2026-02-16

### Added
- sklearn-style parameter handling, class probabilities, feature importances, tree inspection and visualization, and joblib persistence.
- Binary-input validation, optional Numba routing, example notebooks, and expanded tests.

## [1.3.1] - 2026-02-15

### Fixed
- Kept the tree structurally valid when a rolling subproblem is infeasible.

## [1.3.0] - 2026-02-15

### Added
- `n_jobs` parameter on `RollingOCT` for parallel solving of independent OCT-2 subproblems during rolling expansion via `concurrent.futures.ProcessPoolExecutor`.
- `rollotree/rolling/parallel.py` — worker function and data classes for process-safe subproblem dispatch.
- `benchmarks/bench_parallel.py` — wall-clock benchmark comparing sequential vs parallel execution.

### Performance
- At depth 4+ on larger datasets (e.g. wdbc, 512 samples / 300 features), `n_jobs=-1` yields 13-22% wall-clock speedup. Speedup scales with the number of independent parent nodes per level and the per-subproblem solve time.

### Changed
- Refactored `_rolling_expand()` inner loop into three phases (build inputs → parallel solve → sequential merge) for cleaner separation of concerns.
- Moved the unprune block from per-parent to once-per-level execution (idempotent, no behavior change).

## [1.2.0] - 2026-02-15

### Performance
- Vectorized `predict()` and `get_misclassified_leaves()` — batch-route all samples through the tree using NumPy instead of per-sample Python loops (~20x speedup).
- Replaced `feature_vector.dot(x)` with direct array index lookup in tree routing, eliminating O(n_features) dot product per node per sample.
- Precomputed boolean feature masks in impurity computation, reducing temporary array allocations in the innermost loop (~2-3x speedup).
- Matrix-multiply variable elimination in the MIP solver — replaced O(|P|² × n_samples) nested loops with a single `F.T @ F` BLAS call (~45x speedup).
- Optional Numba JIT compilation for tree routing via `pip install rollotree[fast]` (additional ~2-5x on top of NumPy vectorization).

### Added
- `rollotree/tree/_numba.py` — optional Numba-accelerated tree routing with graceful fallback.
- `fast` optional dependency extra: `pip install rollotree[fast]`.

## [1.1.0] - 2026-02-14

### Changed
- Renamed package from `rollo-oct` to `rollotree` (PyPI name, import name, brand name).
- Full OOP refactor of the codebase.

### Added
- sklearn-style `fit()` / `predict()` / `score()` API via `RollingOCT`.
- Solver-agnostic MIP backend via PuLP (HiGHS, Gurobi, CBC).
- Gini and misclassification impurity criteria.
- Rolling subtree (RST) algorithm for building trees deeper than depth 2.
- Bundled Wine dataset (binarized) for examples and tests.
- 70+ pytest test cases.
- Jupyter notebook examples (`01_quickstart`, `02_advanced`).
- GitHub Actions CI/CD for automated PyPI publishing on tag push.
- CONTRIBUTING.md and CHANGELOG.md.

## [1.0.0] - 2024-03-23

### Added
- Initial release as `rollo-oct`.

[1.3.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.3.0
[2.1.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v2.1.0
[2.0.1]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v2.0.1
[2.0.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v2.0.0
[1.3.1]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.3.1
[1.2.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.2.0
[1.1.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.1.0
[1.0.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.0.0

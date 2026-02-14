# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[1.1.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.1.0
[1.0.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v1.0.0

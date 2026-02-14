# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-02-14

### Added
- Initial public release.
- `RollingOCT` classifier with sklearn-style `fit()` / `predict()` / `score()` API.
- Solver-agnostic MIP backend via PuLP (HiGHS, Gurobi, CBC).
- Gini and misclassification impurity criteria.
- Rolling subtree (RST) algorithm for building trees deeper than depth 2.
- Bundled Wine dataset (binarized) for examples and tests.
- 70 pytest test cases.
- Jupyter notebook examples (`01_quickstart`, `02_advanced`).

[0.1.0]: https://github.com/koftezz/rolling-lookahead-dt/releases/tag/v0.1.0

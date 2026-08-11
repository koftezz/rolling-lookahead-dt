# RolloTree — Rolling Lookahead Optimal Classification Trees

[![PyPI version](https://img.shields.io/pypi/v/rollotree)](https://pypi.org/project/rollotree/)
[![Python](https://img.shields.io/pypi/pyversions/rollotree)](https://pypi.org/project/rollotree/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

An implementation of the rolling subtree lookahead algorithm from ["Rolling Lookahead Learning for Optimal Classification Trees"](https://www.tandfonline.com/doi/abs/10.1080/24725854.2026.2613786) (published in *IISE Transactions*).

RolloTree builds interpretable decision trees by solving a sequence of small mixed-integer programs (MIPs). It starts with an optimal depth-2 tree and iteratively expands misclassified leaves, combining the scalability of greedy methods with the optimality guarantees of MIP-based approaches.

## Features

- **Solver-agnostic** — uses [PuLP](https://github.com/coin-or/pulp) so you can choose between:
  - **HiGHS** (open-source, installed by default)
  - **Gurobi** (commercial, optional — install `gurobipy` separately)
  - **CBC** (open-source, bundled with PuLP)
- **Full sklearn compatibility** — `fit()` / `predict()` / `score()` / `predict_proba()` / `get_params()` / `set_params()` — works with `GridSearchCV`, `Pipeline`, `cross_val_score`, and `clone()`
- **Class probabilities** — `predict_proba()` returns per-class probabilities from leaf distributions
- **Feature importances** — `feature_importances_` based on split frequency across branch nodes
- **Tree visualization** — `export_text()` for ASCII and `export_graphviz()` for DOT/Graphviz output
- **Tree inspection** — `apply()`, `decision_path()`, `get_n_leaves()`, `get_depth()`
- **Model persistence** — `save()` / `load()` via joblib, plus standard pickle support
- **Input validation** — clear error messages for non-binary features, single-class targets, and more
- **Exact depth-3 option** — `initial_depth=3` globally optimizes a complete depth-3 tree over the selected feature set
- **Two impurity criteria** — Gini index or misclassification error
- **Arbitrary depth** — depth-2 base tree extended via rolling subtree optimization
- **Parallel solving** — `n_jobs=-1` to solve independent subproblems across CPU cores
- **Bounded wide-data search** — reproducible `max_features` sampling and a fit-wide `total_time_limit`
- **Fit diagnostics** — solver status, objective, runtime, feature count, and skip reason for each subproblem

## Installation

```bash
pip install rollotree
```

Or in editable/development mode (from a local clone):

```bash
pip install -e ".[dev]"
```

Dependencies: `pulp`, `highspy`, `numpy`, `pandas`, `scipy`, `joblib`, `scikit-learn`.

For faster prediction and tree building with [Numba](https://numba.pydata.org/) JIT compilation:

```bash
pip install "rollotree[fast]"
```

To use the Gurobi solver backend:

```bash
pip install "rollotree[gurobi]"
```

## Quick Start

```python
import pandas as pd
from rollotree import RollingOCT

# Load data
train = pd.read_csv("rollotree/data/train.csv")
X_train = train.drop("y", axis=1)
y_train = train["y"]

test = pd.read_csv("rollotree/data/test.csv")
X_test = test.drop("y", axis=1)
y_test = test["y"]

# Train a depth-3 tree using the open-source HiGHS solver
model = RollingOCT(depth=3, criterion="gini", solver="highs")
model.fit(X_train, y_train)

# Evaluate
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# Class probabilities
proba = model.predict_proba(X_test)
print(f"Probabilities shape: {proba.shape}")

# Feature importances
importances = model.feature_importances_
print(f"Top feature: {importances.argmax()}, importance: {importances.max():.3f}")

# Tree visualization
from rollotree import export_text
print(export_text(model.tree_))
```

### sklearn Integration

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv)
print(f"CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Grid search
grid = GridSearchCV(
    RollingOCT(solver="highs"),
    {"depth": [2, 3, 4], "criterion": ["gini", "misclassification"]},
    cv=cv,
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

### Model Persistence

```python
# Save and load
model.save("my_tree.joblib")
loaded = RollingOCT.load("my_tree.joblib")

# Standard pickle also works
import pickle
data = pickle.dumps(model)
loaded = pickle.loads(data)
```

### Parallel Execution

For deeper trees on larger datasets, use `n_jobs` to solve independent
subproblems in parallel:

```python
# Use all CPU cores for parallel subproblem solving
model = RollingOCT(depth=5, solver="highs", n_jobs=-1)
model.fit(X_train, y_train)

# Or specify an exact number of workers
model = RollingOCT(depth=5, solver="highs", n_jobs=4)
model.fit(X_train, y_train)
```

The rolling expansion at each depth level solves independent OCT-2 MIP
subproblems for each parent node. With `n_jobs > 1` these are dispatched
across processes via `ProcessPoolExecutor`. Speedup scales with the number
of parents per level — most effective at depth 4+.

### Exact Depth 3 and Wide Data

The default remains the paper's rolling OCT-2 strategy. To optimize a complete
depth-3 seed globally, use `initial_depth=3`:

```python
model = RollingOCT(
    depth=5,
    initial_depth=3,       # exact depth-3 seed, then rolling expansion
    max_features="sqrt",  # fixed seeded subset within each subproblem
    random_state=42,
    total_time_limit=300,  # wall-clock budget for the complete fit
    n_jobs=-1,
)
model.fit(X_train, y_train)

print(model.fit_status_, model.actual_depth_, model.fit_time_)
for diagnostic in model.subproblem_diagnostics_:
    print(diagnostic.status, diagnostic.objective_value, diagnostic.elapsed_time)
```

With `depth=3, initial_depth=3`, optimal status certifies the globally optimal
complete depth-3 tree over the selected feature subset. For larger requested
depths, only that seed has a global certificate; subsequent levels use rolling
OCT-2 expansion. Every populated leaf strictly honors `min_samples_leaf`, so
infeasible complete trees raise a clear error instead of relaxing the contract.

Predictive accuracy is not guaranteed to improve merely because the training
objective is more optimal. The bundled paired Wine benchmark found lower Gini
but no held-out accuracy gain, at roughly 2.5-3.7x runtime. See
[`benchmarks/QUALITY_RESULTS.md`](benchmarks/QUALITY_RESULTS.md) for the
depth-3/4/5 comparison and run `python benchmarks/bench_quality.py` on the
target workload before choosing the seed strategy.

## API Reference

### `RollingOCT`

```python
RollingOCT(
    depth=2,              # Maximum tree depth (>= 2)
    criterion="gini",     # "gini" or "misclassification"
    solver="highs",       # "highs", "gurobi", or "cbc"
    time_limit=1800,      # Max seconds per depth-2 subproblem
    mip_gap=None,         # MIP optimality gap (e.g. 0.01 for 1%)
    big_m=99,             # Penalty for empty-leaf splits
    log_to_console=False,
    min_samples_split=2,  # Min samples to solve a subproblem
    min_samples_leaf=1,   # Min samples per leaf node
    n_jobs=1,             # Parallel workers: 1=sequential, -1=all cores
    max_features=None,    # None, int, fraction, "sqrt", or "log2"
    random_state=None,    # Reproducible feature subsets
    total_time_limit=None,# Wall-clock budget for the whole fit
    initial_depth=2,      # Exact seed depth: 2 or 3
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the model. `X` is a binary feature matrix (DataFrame or array), `y` is the target. Returns `self`. |
| `predict(X)` | Return class predictions as a numpy array. |
| `predict_proba(X)` | Return class probability estimates (n_samples × n_classes). |
| `score(X, y)` | Return accuracy (fraction correct). |
| `apply(X)` | Return leaf node IDs for each sample. |
| `decision_path(X)` | Return sparse CSR matrix of nodes visited by each sample. |
| `get_n_leaves()` | Return the number of populated terminal leaves. |
| `get_depth()` | Return the actual depth of the deepest populated path. |
| `get_params()` | Return dict of estimator parameters (sklearn protocol). |
| `set_params(**params)` | Set estimator parameters (sklearn protocol). Returns `self`. |
| `save(path)` | Save the fitted model to a file (joblib). |
| `RollingOCT.load(path)` | Load a saved model (class method). |

**Attributes (after fitting):**

| Attribute | Description |
|-----------|-------------|
| `tree_` | The fitted `DecisionTree` object |
| `depth_results_` | Dict mapping depth → `DepthResult(depth, training_accuracy, test_accuracy, elapsed_time)` |
| `classes_` | Sorted numpy array of unique class labels |
| `features_` | List of feature indices used |
| `feature_importances_` | Feature importance array (split frequency, normalized to sum to 1) |
| `n_features_in_` | Number of features seen during `fit()` |
| `feature_names_in_` | Feature names (when `X` is a DataFrame) |
| `fit_status_` | `"completed"`, `"early_stopped"`, or `"time_limit"` |
| `fit_time_` | Complete wall-clock fit time in seconds |
| `actual_depth_` | Deepest populated terminal path |
| `subproblem_diagnostics_` | Ordered per-solve status, timing, feature-count, and candidate-feature records |

**Visualization functions:**

| Function | Description |
|----------|-------------|
| `export_text(tree, feature_names=None)` | ASCII tree representation |
| `export_graphviz(tree, feature_names=None, class_names=None)` | DOT format string for Graphviz |

### Solver Options

| Solver | Install | Notes |
|--------|---------|-------|
| `"highs"` | `pip install highspy` (included in requirements) | Best open-source MIP solver. Default. |
| `"gurobi"` | `pip install gurobipy` + license | Fastest commercial solver. |
| `"cbc"` | Bundled with PuLP | Fallback open-source solver. |

## Examples

See the `examples/` directory for Jupyter notebooks (all with saved outputs):

- **[01_quickstart.ipynb](https://github.com/koftezz/rolling-lookahead-dt/blob/main/examples/01_quickstart.ipynb)** — Fit/predict/score, `predict_proba`, `feature_importances_`
- **[02_visualization.ipynb](https://github.com/koftezz/rolling-lookahead-dt/blob/main/examples/02_visualization.ipynb)** — `export_text`, `export_graphviz`, `apply`, `decision_path`, tree inspection
- **[03_sklearn_integration.ipynb](https://github.com/koftezz/rolling-lookahead-dt/blob/main/examples/03_sklearn_integration.ipynb)** — `GridSearchCV`, `Pipeline`, `cross_val_score`, model persistence
- **[04_advanced.ipynb](https://github.com/koftezz/rolling-lookahead-dt/blob/main/examples/04_advanced.ipynb)** — Depth analysis, preprocessing, solver tuning, parallel execution, tree internals, Numba, per-leaf stats, criteria comparison, early stopping

## Dataset

An example dataset is provided under `rollotree/data/` — a binarized version of the [Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine) (3-class, 130 binary features).

## How It Works

1. **OCT-2 Formulation**: Solves a MIP to find the optimal depth-2 binary classification tree. Binary variables select which feature to split on at each node, and the objective minimizes total leaf impurity.

2. **Rolling Subtree (RST) Algorithm**: Identifies misclassified leaves, groups them by parent, then solves a new OCT-2 subproblem for each parent's data subset. The resulting subtrees are merged back into the main tree. This repeats level-by-level until the target depth is reached.

3. **LP Relaxation Integrality**: The OCT-2 formulation's feasible region is an integral polyhedron (Proposition 2 in the paper), so the LP relaxation always yields an integer-optimal solution.

## Project Structure

```
rollotree/
    __init__.py              # Public API: RollingOCT, SolverConfig, etc.
    classifier.py            # RollingOCT (sklearn-like fit/predict)
    tree/
        nodes.py             # DecisionNode, LeafNode, DecisionTree
        export.py            # export_text(), export_graphviz()
        impurity.py          # GiniCriterion, MisclassificationCriterion
        utils.py             # Node generation, index mapping helpers
        _numba.py            # Optional Numba-accelerated tree routing
    solver/
        base.py              # SolverStatus, OCT2Solution, SolverConfig
        pulp_solver.py       # PuLPOCT2Solver (HiGHS / Gurobi / CBC)
        depth3.py            # Exact depth-3 root enumeration
    rolling/
        optimizer.py         # RollingOptimizer, DepthResult
        parallel.py          # Multiprocessing worker for subproblem solving
    preprocessing/
        helpers.py           # Data binarization and preprocessing
    data/
        train.csv, test.csv  # Example Wine dataset
tests/                       # Unit, integration, solver, and API regression tests
benchmarks/                  # Performance benchmarks
examples/                    # Jupyter notebooks
```

## Citation

```bibtex
@article{organ2026rolling,
      title={Rolling Lookahead Learning for Optimal Classification Trees},
      author={Zeynel Batuhan Organ and Enis Kayış and Taghi Khaniyev},
      journal={IISE Transactions},
      year={2026},
      publisher={Taylor \& Francis},
      doi={10.1080/24725854.2026.2613786},
      url={https://www.tandfonline.com/doi/abs/10.1080/24725854.2026.2613786}
}
```

A preprint is also available on [arXiv (2304.10830)](https://arxiv.org/abs/2304.10830).

## Contact

Feel free to [reach out](mailto:batuhan.organ@ozu.edu.tr) with questions or feedback.

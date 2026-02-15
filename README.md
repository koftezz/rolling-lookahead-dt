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
- **sklearn-style API** — `fit()` / `predict()` / `score()`
- **Two impurity criteria** — Gini index or misclassification error
- **Arbitrary depth** — depth-2 base tree extended via rolling subtree optimization
- **Parallel solving** — `n_jobs=-1` to solve independent subproblems across CPU cores

## Installation

```bash
pip install rollotree
```

Or in editable/development mode (from a local clone):

```bash
pip install -e ".[dev]"
```

Dependencies: `pulp`, `highspy`, `numpy`, `pandas`.

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
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the model. `X` is a binary feature matrix (DataFrame or array), `y` is the target. Returns `self`. |
| `predict(X)` | Return class predictions as a numpy array. |
| `score(X, y)` | Return accuracy (fraction correct). |

**Attributes (after fitting):**

| Attribute | Description |
|-----------|-------------|
| `tree_` | The fitted `DecisionTree` object |
| `depth_results_` | Dict mapping depth → `DepthResult(depth, training_accuracy, test_accuracy, elapsed_time)` |
| `classes_` | Sorted list of unique class labels |
| `features_` | List of feature indices used |

### Solver Options

| Solver | Install | Notes |
|--------|---------|-------|
| `"highs"` | `pip install highspy` (included in requirements) | Best open-source MIP solver. Default. |
| `"gurobi"` | `pip install gurobipy` + license | Fastest commercial solver. |
| `"cbc"` | Bundled with PuLP | Fallback open-source solver. |

## Examples

See the `examples/` directory for Jupyter notebooks:

- **[01_quickstart.ipynb](https://github.com/koftezz/rolling-lookahead-dt/blob/main/examples/01_quickstart.ipynb)** — Basic usage: load data, train, predict, evaluate
- **[02_advanced.ipynb](https://github.com/koftezz/rolling-lookahead-dt/blob/main/examples/02_advanced.ipynb)** — Comparing solvers, criteria, and tree depths

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
        impurity.py          # GiniCriterion, MisclassificationCriterion
        utils.py             # Node generation, index mapping helpers
        _numba.py            # Optional Numba-accelerated tree routing
    solver/
        base.py              # SolverStatus, OCT2Solution, SolverConfig
        pulp_solver.py       # PuLPOCT2Solver (HiGHS / Gurobi / CBC)
    rolling/
        optimizer.py         # RollingOptimizer, DepthResult
        parallel.py          # Multiprocessing worker for subproblem solving
    preprocessing/
        helpers.py           # Data binarization and preprocessing
    data/
        train.csv, test.csv  # Example Wine dataset
tests/                       # 90+ pytest test cases
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

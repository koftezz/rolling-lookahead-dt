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
OCT-2 expansion. The exact initializer explicitly uses a zero MIP gap; passing
a positive `mip_gap` with `initial_depth=3` raises an error because it cannot
support a global-optimality certificate. Every populated leaf strictly honors
`min_samples_leaf`, so infeasible complete trees raise a clear error instead
of relaxing the contract.

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

### Direct local optimization

`RollingOCT(solver="direct")` opts into exact additive OCT-2 minimization.
For each root, feasible left and right child choices are independent: minimize
both costs and then choose the cheapest root. Support filtering is unchanged.
Ties follow supplied candidate order (root, then left child, then right child);
equal-objective trees from different backends can predict differently.
The scan is O(p²), in addition to coefficient calculation, and initially stores
quadratic pair tables. External-solver `time_limit` does not interrupt direct
coefficient work. HiGHS remains the default.

This is local exactness over supplied binary candidates, not global optimality
of the final rolling tree. Cross-branch constraints can break the decomposition.
Misclassification objectives are counts, not rates. Gini is weighted by leaf
sample fraction; combining subset objectives requires subset-size weighting.
The corresponding LP has integral vertices, but tied optimal vertices can have
fractional convex combinations that are also optimal points.

### Acceptance and search traces

`acceptance_policy="accuracy"` retains the existing level-wide accuracy guard:
all replacements in a level are reverted if full training accuracy falls by
more than 1e-10. Individual replacements may trade accuracy within that level.
`acceptance_policy="objective"` independently scores each affected subset before
and after replacement and rejects objective increases above 1e-10. Scores use
raw routed labels, not solver coefficients. Gini trace scores are normalized
within that subset; multiply by `n_samples / training_n` before adding across
subsets. Misclassification trace scores are counts and add directly. Neither
policy promises held-out improvement.

Set `trace=True` for JSON-serializable `search_trace_` events: initial solves,
initial acceptance, attempted updates, and final stopping status. No raw rows
are recorded. Update records include node/depth, subset/candidate sizes, backend,
status, before/after objective and accuracy, timings, decision/reason, and budget
remaining. A level rollback marks its attempted replacements rejected.
Existing `subproblem_diagnostics_` remains available independently.

`time_limit` applies per external solver call. `total_time_limit` is a cooperative
fit budget checked at solve/expansion boundaries; coefficient calculation,
model construction and dispatched calls can overrun it. This is not a hard
wall-clock deadline. An incumbent is retained on exhaustion; initialization
without an incumbent raises. Coefficient timing covers coefficients (and direct
support filtering); assembly covers external support filtering/model setup;
solve timing covers the backend call or direct cost scan. Missing timings are
`None`, not zero. Direct assembly is zero because no external model is built.

### Blocked direct optimization

`solver="direct", direct_block_size=8` processes root candidates eight at a
time. Temporary pair statistics use O(bp) storage and stream over classes;
input matrices and class-selected copies use O(np) separately. No quadratic
support precheck is used on the blocked depth-3 path. The default `None` retains
the unblocked implementation. Blocked fits use sequential solves even when
`n_jobs` requests parallel execution, to avoid multiplying memory consumption.

The fit deadline is checked between root blocks, not inside matrix operations.
Partial searches return TIME_LIMIT with a feasible incumbent when available,
no optimality certificate and no invented gap. Before the first feasible root,
there may be no incumbent. Candidate-order tie-breaking matches unblocked direct.
Run `python benchmarks/bench_direct.py` for fresh-process RSS and timing data;
RSS includes interpreter, dependencies, and inputs, not just pair costs.

### Experimental adaptive lookahead and offline diagnostics

`adaptive_lookahead=None` is the unchanged default. With `solver="direct"`, opt
into `"gap"`, `"impurity"`, `"samples"`, `"random"`, or `"fixed"`. The initial
OCT-2 and each rolling replacement retain a feasible baseline. An eligible node
with at least three remaining levels may receive one exact depth-3 audit;
independent routed scoring must agree with its reported objective, and the
configured acceptance policy must accept it against the baseline. Subsequent
replacement/level acceptance still applies. Failed or unusable audits retain
the baseline. This does not revise ancestors that were already committed.

`audit_budget=2` limits the number of audits across the fit, including the root.
`audit_min_samples=32` is a sample floor; samples selection audits every node
above this floor. Fixed selection uses the same floor and audits every eligible
node until the budget is used. Gap selection requires the best two root costs
to differ by at most `audit_threshold=0.02`. Impurity selection requires residual
objective at least that threshold. Counts are divided by subset size for these
heuristics. Random selection uses a seeded 50% decision per eligible node and
may spend less than its count allowance. Nodes are visited deterministically,
not globally ranked. `audit_count_` and `audit_time_` expose actual expenditure.
Budgets are allowances, not a promise that different selectors spend equal time.
There is no learned allocator. Controlled-perturbation signals are offline only.

`rollotree.diagnostics.commitment_regret(...)` performs expensive shallow,
fixed-root depth-3 and free-root depth-3 solves with identical candidates,
support rules and objective units. `ExactDepth3Solver.solve(fixed_root=...)`
restricts only the top root, retaining all child candidates. Exact regret is
reported only when all required solves are optimal; otherwise it is `None`.
It measures the cost of keeping the shallow root at depth 3, not final rolling
tree quality. A small root gap is not a regret certificate and requires empirical
evaluation. See the reproducible experiments for failures of this hypothesis.

Related work predates this extension: [anytime lookahead induction](https://csaws.cs.technion.ac.il/~shaulm/papers/abstracts/Esmeir-2004-LBA.html),
[variable-depth lookahead](https://www.bgu.ac.il/en/researcher/mark-last/publications/39521646/),
and [MurTree](https://jmlr.org/papers/v23/20-520.html). We make no novelty claim.
This is a bounded experimental implementation, not evidence of improved
held-out accuracy or an established quality–compute advantage.

### Reproducible evidence

See [the lookahead pilot report](benchmarks/LOOKAHEAD_RESULTS.md) for measured
runtime/RSS, exact commitment diagnostics, paired train/test experiments,
negative results, and reproduction commands. The study establishes no general
adaptive-selector advantage. Tracing overhead counts toward cooperative fit
budgets, so results near a binding deadline can vary with system load.

### Experimental Jev audit selection

Jev can optionally advise whether an eligible node should receive a depth-3
audit. The numerical solver still builds the candidate and the existing
acceptance policy checks it. This is an experimental routing policy, not an
optimality certificate or evidence of improved accuracy/runtime. Small local
fits can be much faster than an API request.

Install from a checkout containing this feature (Python **3.10+** for the
optional TypeSafe SDK; base RolloTree continues to support Python 3.9):

```bash
python -m pip install -e ".[jev]"
export TYPESAFE_API_KEY="your-key"
```

```python
from rollotree import RollingOCT, JevAuditSelector

model = RollingOCT(
    depth=4,
    solver="direct",
    adaptive_lookahead="policy",
    audit_selector=JevAuditSelector(
        model="jev-1.13.0",
        timeout=0.5,          # HTTP operation timeout in seconds
        min_confidence=0.8,  # experimental threshold; calibrate on your data
        fallback="skip",
    ),
    audit_budget=2,
    policy_max_calls=4,
    policy_time_limit=2.0,    # total cooperative policy allowance per fit
    total_time_limit=30.0,    # includes policy work and API waiting
    acceptance_policy="objective",
    trace=True,
)
model.fit(X_train, y_train)
print(model.policy_call_count_, model.policy_time_, model.audit_count_)
print([e for e in model.search_trace_ if e["event"] == "policy"])
predictions = model.predict(X_test)  # no API call or key required
```

Only aggregate node statistics are sent: sizes, sorted class proportions,
normalized loss/accuracy, complete root gap/ties/spread when available,
shallow status/runtime, leaf support balance, depth, and remaining budgets.
No raw rows, feature names, or class names are sent. A time-limited shallow
search reports incomplete root statistics and omits its gap. State snapshots
and raw provider errors are not embedded in traces.

For experimental allocation across competing nodes, use `JevSearchController`:

```python
from rollotree import JevSearchController, RollingOCT

model = RollingOCT(
    depth=5, solver="direct", adaptive_lookahead="policy",
    audit_selector=JevSearchController(
        timeout=2.0,
        min_confidence=0.0,  # diagnostic default; calibrate on development data
        # audit_seconds_estimate=1.0,  # optional: supply a measured estimate
        # reference_cases=frozen_examples,  # optional: up to 16 AuditExample records
        min_objective_gain=0.01,  # whole-tree normalized training-objective target
        max_frontier_nodes=16,
    ),
    audit_budget=2, policy_max_calls=3, policy_time_limit=6.0,
    total_time_limit=20.0, acceptance_policy="objective", trace=True,
).fit(X_train, y_train)
```

After the initial root decision, the controller compares eligible disjoint
regions at each rolling level in one request. Each node receives separate
typed questions for audit benefit and relative priority. Python visits proposed
audits in descending ordinal priority, breaks ties by training-weighted residual
loss, and enforces the remaining audit/time budgets before every solve. This
can allocate a scarce audit to a later node instead of spending it in node-ID
order. Nodes outside the configured request cap are skipped; the cap retains
nodes with the largest training-weighted residual. Skipped nodes still receive
the ordinary shallow rolling update.

Controller state schema 3 adds the node's fraction of training rows, current
committed-region loss, and aggregate per-leaf support, class proportions, Gini,
and best feasible one-split Gini gain. It also receives up to 16 earlier audit
observations with measured duration, independently scored local objective and
current commitment status. History exists even with tracing disabled and resets
on every fit. These observations do not establish causal final-tree gains, and
one-split evidence cannot identify every higher-order interaction. Test labels,
raw rows and feature/class names are not included.

The controller has no calibrated confidence threshold by default; explicit
abstention still skips. Priorities are ordinal judgments, not predicted gains.
The optional duration estimate must come from separate measurements; otherwise
only observed past durations are supplied. Enabling broader control does not
establish an advantage over local rules given the same evidence.

`AuditExample(state, final_objective_gain, audit_seconds)` supplies offline
evidence from paired full continuations with and without one audit. Freeze
these examples before evaluating new cases; use training-objective differences,
not test labels or local incumbent regret. Examples, a gain floor and
code-computed evidence buckets are sent explicitly as development context.
They are optional and remain configuration on the saved controller, so include
only aggregate evidence intended for reuse and provider submission.

Custom controllers implement the ordinary `select(state, *, timeout)` root
interface and `plan(states, *, history, timeout) -> AuditPlan` for frontiers.
An `AuditPlan` maps every supplied node ID to an `AuditDecision`, whose optional
`priority` is in `[0, 1]`. It owns the batch's `input_tokens`; individual plan
decisions must not duplicate usage. Invalid plans fail closed for the whole
frontier. A batch consumes one policy call, with its charged time and token
usage in a `policy_plan` trace event; per-node policy events reference that
call. Several per-node fallbacks can therefore follow one failed plan call.

The selector is called after ordinary eligibility checks. At most
`policy_max_calls` attempts are made per fit; the default is 8, with a default
`policy_time_limit` of 1 second. Both counters reset on refit. Request timeouts
are bounded by the remaining fit/policy allowance, retries are disabled, and
budgets are rechecked before the deeper solve. These limits are cooperative:
SDK HTTP timeouts apply per operation, and neither requests nor numerical work
are forcibly cancelled at an exact wall-clock deadline.

Low confidence, `insufficient_evidence`, unavailable SDK/key, malformed results,
and API errors retain the shallow baseline. Failures emit one warning per fit;
check the trace rather than assuming that a successful fit proves Jev worked.
The first supported fallback is `skip`. Confidence describes the distribution
over actions, not measured correctness or predicted objective improvement.

`policy_call_count_` counts attempted selector invocations (including failures
before an HTTP request); `policy_time_` includes pre-decision scoring, state
extraction, fingerprinting, and client work; `policy_input_tokens_` sums known reported input usage only;
`policy_fallback_count_` counts fallback decisions after attempted calls.
Trace records include action, original proposed action when available,
probabilities/confidence, effective action after budget checks, latency, requested
and returned models, prompt/state versions, state hash, and explicit fallback
reason. `cache_status="disabled"` identifies uncached calls. Audit events report
local `adopted_as_candidate` separately from `final_commit` and
`final_commit_reason`, including region rejection, level rollback, and later
replacement of the audited region. Expanding a child region preserves its
ancestor audit's commitment. An audit candidate's adoption alone is not a
final commitment. Candidate structure, class/feature membership, routed sample
support, and objective agreement are checked independently before adoption.

`JevAuditSelector` is sklearn-cloneable and stores only configuration. The SDK
is imported lazily, the key is read from the environment, and clients are closed
after each request. Saved models can predict without the SDK or credentials.
The default `adaptive_lookahead=None` makes no calls. To use another policy,
provide a selector with `select(state: AuditState, *, timeout: float)` returning
an `AuditDecision` with action `audit`, `skip`, or `insufficient_evidence`.
Custom selectors must honor the cooperative timeout and be cloneable/pickleable
if used with sklearn or model persistence.

Run `python examples/jev_smoke.py` after setting the key to exercise one real
request, a complete fit, and the baseline comparison. It incurs API usage and
fails if no usable Jev response is recorded; abstention is a valid response.
This toy smoke test checks connectivity, not whether Jev improves performance.
No hosted-model benchmark advantage is claimed. The model is pinned, but a
fixed random seed does not guarantee identical hosted responses.

This integration implements the optional policy hook and provider adapter.
The offline paired-continuation dataset, learned baselines, response cache/replay,
and held-out quality/runtime experiments proposed in the integration plan remain
research work. Online usefulness has not been established.

Official API contract: [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python).

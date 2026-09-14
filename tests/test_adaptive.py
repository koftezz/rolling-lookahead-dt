from itertools import product
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from rollotree import RollingOCT
from rollotree.diagnostics import commitment_regret
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.depth3 import ExactDepth3Solver, OCT3Solution
from rollotree.tree.impurity import get_criterion
from tests.test_direct import raw_cost


def fixture(seed=1):
    X = np.array(list(product((0, 1), repeat=4))*4)
    y = np.tile(np.random.default_rng(seed).integers(0, 2, 16), 4)
    return X, y, pd.DataFrame(np.column_stack([y, X]))


@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
def test_fixed_and_free_raw_depth3_oracle(criterion):
    X, y, data = fixture()
    features = [1, 2, 3, 4]
    arr = np.asarray(data)
    costs = []
    for root in features:
        subcost = 0
        for first in [1, 0]:
            subset = arr[arr[:, root] == first]
            best = min(raw_cost(subset, t, criterion) for t in product(features, repeat=3))
            subcost += best * (len(subset)/len(arr) if criterion == "gini" else 1)
        costs.append(subcost)
    solver = ExactDepth3Solver(SolverConfig(solver_name="direct", direct_block_size=2), get_criterion(criterion))
    for root in features:
        fixed = solver.solve(data, features, [0,1], fixed_root=root)
        assert fixed.objective_value == pytest.approx(costs[root-1])
        assert fixed.branch_features[1] == root
    free = solver.solve(data, features, [0,1])
    assert free.objective_value == pytest.approx(min(costs))
    record = commitment_regret(data, features, [0,1], get_criterion(criterion))
    assert record["exact"]
    assert record["commitment_regret"] > 0


@pytest.mark.parametrize("mode", [None, "gap", "samples", "impurity", "fixed", "random"])
def test_adaptive_budget_depth_reproducibility(mode):
    X, y, _ = fixture()
    config = dict(depth=4, solver="direct", direct_block_size=2, adaptive_lookahead=mode,
                  audit_budget=1, random_state=42, trace=True, acceptance_policy="objective")
    a = clone(RollingOCT(**config)).fit(X, y)
    b = RollingOCT(**config).fit(X, y)
    assert a.get_depth() <= 4
    assert a.audit_count_ <= 1
    assert np.array_equal(a.predict(X), b.predict(X))
    assert [(e.get("node"),e.get("selected")) for e in a.search_trace_] == [(e.get("node"),e.get("selected")) for e in b.search_trace_]


def test_unsuccessful_audit_preserves_baseline(monkeypatch):
    X, y, _ = fixture()
    baseline = RollingOCT(solver="direct", depth=3).fit(X, y)
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: OCT3Solution(status=SolverStatus.TIME_LIMIT))
    adaptive = RollingOCT(solver="direct", depth=3, adaptive_lookahead="fixed", audit_budget=1, trace=True).fit(X, y)
    assert np.array_equal(baseline.predict(X), adaptive.predict(X))
    assert any(e.get("outcome") == "no_usable_deeper_incumbent" for e in adaptive.search_trace_)


def test_no_budget_and_depth2_are_identical():
    X, y, _ = fixture()
    for depth in [2, 4]:
        plain = RollingOCT(solver="direct", depth=depth).fit(X,y)
        adaptive = RollingOCT(solver="direct", depth=depth, adaptive_lookahead="fixed", audit_budget=0).fit(X,y)
        assert np.array_equal(plain.predict(X), adaptive.predict(X))
        assert adaptive.audit_count_ == 0


def test_inexact_regret_is_not_claimed(monkeypatch):
    _, _, data = fixture()
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: OCT3Solution(status=SolverStatus.TIME_LIMIT))
    record = commitment_regret(data, [1,2,3,4], [0,1], get_criterion("gini"))
    assert not record["exact"]
    assert record["commitment_regret"] is None


def test_parallel_direct_matches_serial():
    X, y, _ = fixture(12)
    a = RollingOCT(solver="direct", depth=4, n_jobs=1).fit(X, y)
    b = RollingOCT(solver="direct", depth=4, n_jobs=2).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))

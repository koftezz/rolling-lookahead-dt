import time
from itertools import product
import numpy as np
import pandas as pd
import pytest
from rollotree import RollingOCT
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import get_criterion
from tests.test_direct import raw_cost


@pytest.mark.parametrize("block", [1, 2, 4, 20])
@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
@pytest.mark.parametrize("support", [1, 4, 100])
def test_blocks_match_oracle(block, criterion, support):
    rng = np.random.default_rng(37)
    arr = np.column_stack([rng.integers(0, 3, 40), rng.integers(0, 2, (40, 5))])
    features = [5, 3, 1, 2, 4]
    config = SolverConfig(solver_name="direct", min_samples_leaf=support)
    solver = PuLPOCT2Solver(config, get_criterion(criterion))
    full = solver.solve(pd.DataFrame(arr), features, [0, 1, 2])
    blocked = PuLPOCT2Solver(config.copy_with(direct_block_size=block), get_criterion(criterion)).solve(pd.DataFrame(arr), features, [0, 1, 2])
    expected = min(raw_cost(arr, t, criterion, support) for t in product(features, repeat=3))
    assert full.status == blocked.status
    if np.isfinite(expected):
        assert blocked.objective_value == pytest.approx(expected, abs=1e-10)
        assert (blocked.root_feature, blocked.left_feature, blocked.right_feature) == (full.root_feature, full.left_feature, full.right_feature)
    else:
        assert blocked.status == SolverStatus.INFEASIBLE


def test_interruption_preserves_incumbent(monkeypatch):
    from rollotree.solver import blocked
    X = np.array(list(product((0, 1), repeat=3)))
    arr = np.column_stack([X[:, 0], X])
    ticks = iter([0, 2])
    monkeypatch.setattr(blocked.time, "time", lambda: next(ticks))
    sol = PuLPOCT2Solver(SolverConfig(solver_name="direct", direct_block_size=1, deadline=1), get_criterion("gini")).solve(pd.DataFrame(arr), [1, 2, 3], [0, 1])
    assert sol.status == SolverStatus.TIME_LIMIT
    assert sol.objective_value == 0
    assert sol.mip_gap is None
    assert len(sol.root_costs) == 1


def test_expired_without_incumbent():
    arr = pd.DataFrame([[0, 0, 1], [1, 1, 0]])
    sol = PuLPOCT2Solver(SolverConfig(solver_name="direct", direct_block_size=1, deadline=time.time()-1), get_criterion("gini")).solve(arr, [1, 2], [0, 1])
    assert sol.status == SolverStatus.TIME_LIMIT
    assert sol.root_feature is None


def test_depth3_blocking_avoids_quadratic_precheck(monkeypatch):
    monkeypatch.setattr(PuLPOCT2Solver, "_compute_valid_pairs", lambda *a: pytest.fail("quadratic precheck"))
    X = np.array(list(product((0, 1), repeat=3)) * 2)
    m = RollingOCT(solver="direct", direct_block_size=1, initial_depth=3, depth=3, n_jobs=2).fit(X, X[:,0])
    assert m.score(X, X[:,0]) == 1

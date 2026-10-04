"""Cooperative solver budgets must preserve only completed incumbents."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.depth3 import ExactDepth3Solver
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import get_criterion


def problem():
    X = np.array(list(product((0, 1), repeat=3)))
    return pd.DataFrame(np.column_stack([X[:, 0], X]))


def test_expired_dense_request_avoids_coefficient_allocation(monkeypatch):
    from rollotree.solver import direct

    criterion = get_criterion("gini")
    monkeypatch.setattr(criterion, "compute_leaf_coefficients",
                        lambda *a, **k: pytest.fail("expired request computed coefficients"))
    monkeypatch.setattr(direct, "time", SimpleNamespace(
        time=lambda: 2.0, perf_counter=lambda: 0.0))
    config = SolverConfig(solver_name="direct", deadline=1.0)
    solution = PuLPOCT2Solver(config, criterion).solve(problem(), [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.root_feature is None
    assert solution.root_costs == {}


def test_dense_expiration_after_coefficients_avoids_feasibility_work(monkeypatch):
    from rollotree.solver import direct

    clock = {"now": 0.0}
    criterion = get_criterion("gini")
    compute = criterion.compute_leaf_coefficients

    def compute_and_expire(*args, **kwargs):
        result = compute(*args, **kwargs)
        clock["now"] = 2.0
        return result

    monkeypatch.setattr(criterion, "compute_leaf_coefficients", compute_and_expire)
    monkeypatch.setattr(direct, "time", SimpleNamespace(
        time=lambda: clock["now"], perf_counter=lambda: 0.0))
    monkeypatch.setattr(PuLPOCT2Solver, "_compute_valid_pairs",
                        lambda *a, **k: pytest.fail("expired request computed feasibility"))
    solution = PuLPOCT2Solver(
        SolverConfig(solver_name="direct", deadline=1.0), criterion,
    ).solve(problem(), [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.root_feature is None


def test_dense_partial_scan_preserves_one_complete_root(monkeypatch):
    from rollotree.solver import direct

    ticks = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(direct, "time", SimpleNamespace(
        time=lambda: next(ticks), perf_counter=lambda: 0.0))
    solution = PuLPOCT2Solver(
        SolverConfig(solver_name="direct", deadline=1.0), get_criterion("gini"),
    ).solve(problem(), [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.root_feature == 1
    assert set(solution.root_costs) == {1}
    assert set(solution.leaf_classes) == {4, 5, 6, 7}
    assert solution.objective_value == 0.0
    assert solution.mip_gap is None


def test_dense_per_solve_budget_includes_coefficients(monkeypatch):
    from rollotree.solver import direct

    clock = {"now": 0.0}
    criterion = get_criterion("gini")
    compute = criterion.compute_leaf_coefficients

    def compute_and_expire(*args, **kwargs):
        result = compute(*args, **kwargs)
        clock["now"] = 2.0
        return result

    monkeypatch.setattr(criterion, "compute_leaf_coefficients", compute_and_expire)
    monkeypatch.setattr(direct, "time", SimpleNamespace(
        time=lambda: 0.0, perf_counter=lambda: clock["now"]))
    solution = PuLPOCT2Solver(
        SolverConfig(solver_name="direct", time_limit=1.0), criterion,
    ).solve(problem(), [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.root_feature is None
    assert solution.coefficient_time == 2.0


def test_expired_blocked_request_avoids_data_conversion(monkeypatch):
    from rollotree.solver import blocked

    data = problem()
    monkeypatch.setattr(blocked, "time", SimpleNamespace(
        time=lambda: 2.0, perf_counter=lambda: 0.0))
    original_asarray = blocked.np.asarray

    def reject_training_conversion(value, *args, **kwargs):
        if value is data:
            pytest.fail("expired request converted training data")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(blocked.np, "asarray", reject_training_conversion)
    solution = PuLPOCT2Solver(
        SolverConfig(solver_name="direct", direct_block_size=1, deadline=1.0),
        get_criterion("gini"),
    ).solve(data, [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.root_feature is None


def test_blocked_per_solve_budget_preserves_a_complete_root(monkeypatch):
    from rollotree.solver import blocked

    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(blocked, "time", SimpleNamespace(
        time=lambda: next(ticks), perf_counter=lambda: 0.0))
    solution = PuLPOCT2Solver(
        SolverConfig(solver_name="direct", direct_block_size=1, time_limit=1.0),
        get_criterion("gini"),
    ).solve(problem(), [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.root_feature == 1
    assert set(solution.root_costs) == {1}
    assert set(solution.leaf_classes) == {4, 5, 6, 7}
    assert solution.objective_value == 0.0
    assert solution.mip_gap is None


@pytest.mark.parametrize("configured, limit", [(1.0, 10.0), (10.0, 1.0)])
def test_blocked_uses_earlier_fit_or_per_solve_deadline(monkeypatch, configured, limit):
    from rollotree.solver import blocked

    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(blocked, "time", SimpleNamespace(
        time=lambda: next(ticks), perf_counter=lambda: 0.0))
    solution = PuLPOCT2Solver(
        SolverConfig(solver_name="direct", direct_block_size=1,
                     deadline=configured, time_limit=limit), get_criterion("gini"),
    ).solve(problem(), [1, 2, 3], [0, 1])
    assert solution.status == SolverStatus.TIME_LIMIT
    assert set(solution.root_costs) == {1}


@pytest.mark.parametrize("block", [None, 1])
def test_depth3_inherits_expired_config_deadline_before_dispatch(monkeypatch, block):
    from rollotree.solver import depth3

    monkeypatch.setattr(depth3, "time", SimpleNamespace(
        time=lambda: 2.0, perf_counter=lambda: 0.0))
    monkeypatch.setattr(depth3, "_solve_candidate",
                        lambda *a: pytest.fail("expired request dispatched a candidate"))
    config = SolverConfig(solver_name="direct", deadline=1.0, direct_block_size=block)
    solution = ExactDepth3Solver(config, get_criterion("gini")).solve(
        problem(), [1, 2, 3], [0, 1],
    )
    assert solution.status == SolverStatus.TIME_LIMIT
    assert solution.n_candidates == 3
    assert solution.n_complete_candidates == 0


@pytest.mark.parametrize("configured, explicit, expected", [
    (1.0, None, 1.0), (None, 2.0, 2.0),
    (1.0, 2.0, 1.0), (2.0, 1.0, 1.0), (None, None, None),
])
def test_depth3_uses_the_earliest_deadline(configured, explicit, expected):
    solver = ExactDepth3Solver(
        SolverConfig(solver_name="direct", deadline=configured),
        get_criterion("gini"), deadline=explicit,
    )
    assert solver.deadline == solver.config.deadline == expected

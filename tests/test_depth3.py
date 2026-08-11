"""Tests for the decomposed exact depth-3 solver."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.depth3 import ExactDepth3Solver
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion


def _make_oracle_dataset():
    patterns = np.array(list(product([0, 1], repeat=4)), dtype=int)
    labels = np.array(
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
    )
    frame = pd.DataFrame(patterns, columns=[1, 2, 3, 4])
    frame.insert(0, "y", labels)
    return frame, [1, 2, 3, 4], [0, 1]


def _tree_objective(data, branch_features, criterion, min_samples_leaf=1):
    arr = np.asarray(data)
    leaves = np.ones(len(arr), dtype=int)
    for _depth in range(3):
        next_leaves = leaves.copy()
        for node_id in np.unique(leaves):
            mask = leaves == node_id
            feature = branch_features[node_id]
            next_leaves[mask] = np.where(
                arr[mask, feature] == 1, node_id * 2, node_id * 2 + 1
            )
        leaves = next_leaves

    objective = 0.0
    for leaf_id in range(8, 16):
        labels = arr[leaves == leaf_id, 0]
        if len(labels) < min_samples_leaf:
            return None
        _values, counts = np.unique(labels, return_counts=True)
        if isinstance(criterion, GiniCriterion):
            probabilities = counts / len(labels)
            objective += (len(labels) / len(arr)) * (
                1.0 - float(np.sum(probabilities**2))
            )
        else:
            objective += len(labels) - int(counts.max())
    return objective


def _brute_force_depth3(data, features, criterion, min_samples_leaf=1):
    best = None
    for assignment in product(features, repeat=7):
        branch_features = {
            node_id: assignment[node_id - 1] for node_id in range(1, 8)
        }
        objective = _tree_objective(
            data, branch_features, criterion, min_samples_leaf
        )
        if objective is not None and (best is None or objective < best):
            best = objective
    return best


@pytest.mark.parametrize(
    "criterion", [GiniCriterion(), MisclassificationCriterion()]
)
def test_matches_brute_force_oracle_for_both_criteria(criterion):
    data, features, classes = _make_oracle_dataset()
    expected = _brute_force_depth3(data, features, criterion)
    solver = ExactDepth3Solver(
        SolverConfig(solver_name="highs", time_limit=60), criterion
    )

    solution = solver.solve(data, features, classes)

    assert solution.status == SolverStatus.OPTIMAL
    assert set(solution.branch_features) == set(range(1, 8))
    assert set(solution.leaf_classes) == set(range(8, 16))
    assert solution.objective_value == pytest.approx(expected)
    assert _tree_objective(data, solution.branch_features, criterion) == pytest.approx(
        expected
    )


def test_no_strict_complete_tree_is_infeasible():
    patterns = np.array(list(product([0, 1], repeat=2)), dtype=int)
    data = pd.DataFrame(patterns, columns=[1, 2])
    data.insert(0, "y", [0, 1, 1, 0])
    solver = ExactDepth3Solver(
        SolverConfig(solver_name="highs", min_samples_leaf=1),
        GiniCriterion(),
    )

    solution = solver.solve(data, [1, 2], [0, 1])

    assert solution.status == SolverStatus.INFEASIBLE
    assert solution.branch_features == {}
    assert solution.leaf_classes == {}
    assert solution.n_complete_candidates == 0
    assert all(
        diagnostic.status == SolverStatus.INFEASIBLE
        for diagnostic in solution.candidate_diagnostics
    )


def test_fixed_feature_subset_is_used_at_every_node():
    data, features, classes = _make_oracle_dataset()
    solver = ExactDepth3Solver(
        SolverConfig(solver_name="highs", time_limit=60),
        MisclassificationCriterion(),
    )

    solution = solver.solve(
        data, features, classes, feature_subset=[1, 2, 3]
    )

    assert solution.status == SolverStatus.OPTIMAL
    assert solution.features == (1, 2, 3)
    assert solution.n_candidates == 3
    assert set(solution.branch_features.values()) <= {1, 2, 3}
    assert [d.root_feature for d in solution.candidate_diagnostics] == [1, 2, 3]


def test_sequential_and_parallel_solutions_are_deterministic():
    data, features, classes = _make_oracle_dataset()
    config = SolverConfig(solver_name="highs", time_limit=60)
    sequential = ExactDepth3Solver(
        config, MisclassificationCriterion(), n_jobs=1
    ).solve(data, features, classes)
    parallel = ExactDepth3Solver(
        config, MisclassificationCriterion(), n_jobs=2
    ).solve(data, features, classes)

    assert sequential.status == parallel.status == SolverStatus.OPTIMAL
    assert sequential.objective_value == pytest.approx(parallel.objective_value)
    assert sequential.branch_features == parallel.branch_features
    assert sequential.leaf_classes == parallel.leaf_classes
    assert [d.root_feature for d in sequential.candidate_diagnostics] == [
        d.root_feature for d in parallel.candidate_diagnostics
    ]


@pytest.mark.parametrize(
    "feature_subset, message",
    [([], "at least one"), ([1, 1], "duplicates"), ([1, 5], "unknown")],
)
def test_invalid_feature_subset_raises(feature_subset, message):
    data, features, classes = _make_oracle_dataset()
    solver = ExactDepth3Solver(SolverConfig(), GiniCriterion())
    with pytest.raises(ValueError, match=message):
        solver.solve(data, features, classes, feature_subset=feature_subset)


def test_exact_solver_requires_and_sets_zero_mip_gap():
    with pytest.raises(ValueError, match="mip_gap"):
        ExactDepth3Solver(
            SolverConfig(mip_gap=0.01), GiniCriterion()
        )

    solver = ExactDepth3Solver(SolverConfig(mip_gap=None), GiniCriterion())
    assert solver.config.mip_gap == 0.0

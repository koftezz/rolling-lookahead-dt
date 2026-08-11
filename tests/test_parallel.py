"""Tests for the parallel subproblem execution module."""

import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion
from rollotree.rolling.parallel import (
    SubproblemInput,
    SubproblemResult,
    _solve_subproblem,
    _resolve_n_jobs,
)


def _make_parent_data(n=20, n_features=5, seed=42):
    """Create a small DataFrame matching the internal format (y at col 0)."""
    np.random.seed(seed)
    features = list(range(1, n_features + 1))
    X = np.random.randint(0, 2, size=(n, n_features))
    y = np.random.randint(1, 3, size=n)
    df = pd.DataFrame(X, columns=features)
    df.insert(0, "y", y)
    classes = sorted(df["y"].unique().tolist())
    return df, features, classes


class TestSubproblemInputPicklable:
    def test_roundtrip(self):
        df, features, classes = _make_parent_data()
        inp = SubproblemInput(
            parent_node=2,
            leaf_ids=[4, 5],
            parent_data=df,
            features=features,
            sub_K=classes,
            y_idx=0,
            solver_config=SolverConfig(solver_name="highs", time_limit=30),
            criterion=GiniCriterion(),
        )
        restored = pickle.loads(pickle.dumps(inp))
        assert restored.parent_node == 2
        assert restored.leaf_ids == [4, 5]
        assert len(restored.parent_data) == len(df)


class TestSubproblemResultPicklable:
    def test_roundtrip(self):
        df, features, classes = _make_parent_data()
        result = SubproblemResult(
            parent_node=2,
            leaf_ids=[4, 5],
            sub_solution=None,
            sub_misclassified=[],
            skipped=True,
            n_samples=20,
            parent_data=df,
        )
        restored = pickle.loads(pickle.dumps(result))
        assert restored.parent_node == 2
        assert restored.skipped is True


class TestSolveSubproblem:
    def test_deadline_uses_process_independent_wall_clock(self, monkeypatch):
        """Worker deadline and parent deadline must share the same epoch."""
        captured = {}

        class FakeSolver:
            def __init__(self, config, criterion):
                captured["time_limit"] = config.time_limit

            def solve(self, **_kwargs):
                from rollotree.solver.base import OCT2Solution

                return OCT2Solution(status=SolverStatus.ERROR)

        monkeypatch.setattr(
            "rollotree.solver.pulp_solver.PuLPOCT2Solver", FakeSolver
        )
        df, features, classes = _make_parent_data(n=30)
        inp = SubproblemInput(
            parent_node=2,
            leaf_ids=[4, 5],
            parent_data=df,
            features=features,
            sub_K=classes,
            y_idx=0,
            solver_config=SolverConfig(time_limit=60),
            criterion=GiniCriterion(),
            deadline=time.time() + 5,
        )

        _solve_subproblem(inp)

        assert 0 < captured["time_limit"] <= 5

    def test_expired_deadline_skips_solver(self):
        df, features, classes = _make_parent_data(n=30)
        inp = SubproblemInput(
            parent_node=2,
            leaf_ids=[4, 5],
            parent_data=df,
            features=features,
            sub_K=classes,
            y_idx=0,
            solver_config=SolverConfig(time_limit=60),
            criterion=GiniCriterion(),
            deadline=0.0,
        )

        result = _solve_subproblem(inp)

        assert result.timed_out is True
        assert result.sub_solution is None

    def test_skips_small_datasets(self):
        """With min_samples_split > n_samples, should return skipped=True."""
        df, features, classes = _make_parent_data(n=5)
        inp = SubproblemInput(
            parent_node=2,
            leaf_ids=[4, 5],
            parent_data=df,
            features=features,
            sub_K=classes,
            y_idx=0,
            solver_config=SolverConfig(
                solver_name="highs", time_limit=30, min_samples_split=100
            ),
            criterion=GiniCriterion(),
        )
        result = _solve_subproblem(inp)
        assert result.skipped is True
        assert result.sub_solution is None
        assert result.n_samples == 5

    def test_solves_normal_dataset(self):
        """Normal dataset should return a valid OCT2Solution."""
        df, features, classes = _make_parent_data(n=30)
        inp = SubproblemInput(
            parent_node=2,
            leaf_ids=[4, 5],
            parent_data=df,
            features=features,
            sub_K=classes,
            y_idx=0,
            solver_config=SolverConfig(solver_name="highs", time_limit=60),
            criterion=GiniCriterion(),
        )
        result = _solve_subproblem(inp)
        assert result.skipped is False
        assert result.sub_solution is not None
        assert result.sub_solution.status in (
            SolverStatus.OPTIMAL,
            SolverStatus.TIME_LIMIT,
        )
        assert result.sub_solution.root_feature is not None

    def test_works_with_misclassification_criterion(self):
        df, features, classes = _make_parent_data(n=30)
        inp = SubproblemInput(
            parent_node=3,
            leaf_ids=[6, 7],
            parent_data=df,
            features=features,
            sub_K=classes,
            y_idx=0,
            solver_config=SolverConfig(solver_name="highs", time_limit=60),
            criterion=MisclassificationCriterion(),
        )
        result = _solve_subproblem(inp)
        assert result.skipped is False
        assert result.sub_solution.status in (
            SolverStatus.OPTIMAL,
            SolverStatus.TIME_LIMIT,
        )


class TestResolveNJobs:
    def test_minus_one(self):
        assert _resolve_n_jobs(-1) == (os.cpu_count() or 1)

    def test_positive(self):
        assert _resolve_n_jobs(4) == 4

    def test_one(self):
        assert _resolve_n_jobs(1) == 1

    def test_zero_clamps_to_one(self):
        assert _resolve_n_jobs(0) == 1

    def test_minus_two(self):
        expected = max(1, (os.cpu_count() or 1) - 1)
        assert _resolve_n_jobs(-2) == expected

"""Tests for the rolling subtree optimizer."""

import numpy as np
import pandas as pd
import pytest

from rollotree.solver.base import SolverConfig
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion
from rollotree.rolling.optimizer import RollingOptimizer, DepthResult


def _make_dataset(n=40, n_features=5, n_classes=2, seed=42):
    """Create a synthetic binary dataset."""
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(n, n_features))
    y = np.random.randint(1, n_classes + 1, size=n)
    features = list(range(1, n_features + 1))
    data = pd.DataFrame(X, columns=features)
    data.insert(0, "y", y)
    classes = sorted(data["y"].unique().tolist())
    return data, features, classes


class TestRollingOptimizer:
    def test_depth2_no_expansion(self):
        """Target depth=2 should just solve the initial OCT-2."""
        data, features, classes = _make_dataset()
        config = SolverConfig(solver_name="highs", time_limit=60)
        optimizer = RollingOptimizer(
            solver_config=config, criterion=GiniCriterion()
        )

        tree, results = optimizer.build_tree(
            train_data=data,
            test_data=data,
            features=features,
            classes=classes,
            target_depth=2,
        )

        assert tree.depth == 2
        assert 2 in results
        assert results[2].training_accuracy >= 0.0
        assert results[2].training_accuracy <= 1.0

    def test_depth3_expands_once(self):
        """Target depth=3 should produce one round of expansion."""
        data, features, classes = _make_dataset()
        config = SolverConfig(solver_name="highs", time_limit=60)
        optimizer = RollingOptimizer(
            solver_config=config, criterion=GiniCriterion()
        )

        tree, results = optimizer.build_tree(
            train_data=data,
            test_data=data,
            features=features,
            classes=classes,
            target_depth=3,
        )

        assert tree.depth == 3
        assert 2 in results
        assert 3 in results

    def test_training_accuracy_nondecreasing(self):
        """Deeper trees should not decrease training accuracy."""
        data, features, classes = _make_dataset(n=60)
        config = SolverConfig(solver_name="highs", time_limit=60)
        optimizer = RollingOptimizer(
            solver_config=config, criterion=GiniCriterion()
        )

        tree, results = optimizer.build_tree(
            train_data=data,
            test_data=data,
            features=features,
            classes=classes,
            target_depth=3,
        )

        if 3 in results:
            assert results[3].training_accuracy >= results[2].training_accuracy - 1e-10

    def test_results_dict_has_all_depths(self):
        data, features, classes = _make_dataset()
        config = SolverConfig(solver_name="highs", time_limit=60)
        optimizer = RollingOptimizer(
            solver_config=config, criterion=GiniCriterion()
        )

        tree, results = optimizer.build_tree(
            train_data=data,
            test_data=data,
            features=features,
            classes=classes,
            target_depth=4,
        )

        assert 2 in results
        # Deeper results may or may not exist depending on whether
        # misclassified leaves were found
        for depth in results:
            assert isinstance(results[depth], DepthResult)
            assert results[depth].elapsed_time >= 0

    def test_infeasible_subproblem_prunes_leaves(self):
        """When a subproblem is infeasible the tree should stay consistent.

        Uses very few samples and many features so that deep expansions
        exhaust valid feature pairs, triggering INFEASIBLE.  Before the
        fix the tree would be left with unpopulated branch nodes and
        predictions would break.
        """
        # Small dataset with many binary features → forces infeasibility
        # at deeper levels because leaf subsets become tiny.
        np.random.seed(99)
        n, n_features = 20, 10
        X = np.random.randint(0, 2, size=(n, n_features))
        y = np.random.randint(1, 4, size=n)  # 3 classes
        features = list(range(1, n_features + 1))
        data = pd.DataFrame(X, columns=features)
        data.insert(0, "y", y)
        classes = sorted(data["y"].unique().tolist())

        config = SolverConfig(
            solver_name="highs", time_limit=60, min_samples_leaf=1
        )
        optimizer = RollingOptimizer(
            solver_config=config, criterion=GiniCriterion()
        )

        # Depth 6 on 20 samples will almost certainly hit infeasible
        tree, results = optimizer.build_tree(
            train_data=data,
            test_data=data,
            features=features,
            classes=classes,
            target_depth=6,
        )

        # Tree must predict without errors for every sample
        X_arr = np.array(data[features])
        preds = tree.predict(X_arr)
        assert len(preds) == n
        assert all(p in classes for p in preds)

        # Training accuracy should never decrease with depth
        depths = sorted(results.keys())
        for i in range(1, len(depths)):
            assert (
                results[depths[i]].training_accuracy
                >= results[depths[i - 1]].training_accuracy - 1e-10
            ), (
                f"Training accuracy decreased from depth {depths[i-1]} "
                f"({results[depths[i-1]].training_accuracy}) to "
                f"depth {depths[i]} ({results[depths[i]].training_accuracy})"
            )

    def test_with_misclassification_criterion(self):
        data, features, classes = _make_dataset()
        config = SolverConfig(solver_name="highs", time_limit=60)
        optimizer = RollingOptimizer(
            solver_config=config, criterion=MisclassificationCriterion()
        )

        tree, results = optimizer.build_tree(
            train_data=data,
            test_data=data,
            features=features,
            classes=classes,
            target_depth=3,
        )

        assert tree.depth >= 2
        assert 2 in results

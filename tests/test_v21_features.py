"""Regression tests for the 2.1 correctness and search-control release."""

from decimal import Decimal
from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning

from rollotree import RollingOCT, __version__
from rollotree.rolling.optimizer import RollingOptimizer
from rollotree.solver.base import SolverConfig
from rollotree.tree.impurity import GiniCriterion
from rollotree.tree.nodes import DecisionTree


def _complete_binary_dataset(n_features=4):
    patterns = np.array(list(product([0, 1], repeat=n_features)), dtype=int)
    X = pd.DataFrame(patterns, columns=[f"f{i}" for i in range(n_features)])
    y = np.array([sum(row) % 2 for row in patterns])
    return X, y


def _always_mixed_dataset(n_features=4):
    patterns = np.array(list(product([0, 1], repeat=n_features)), dtype=int)
    X = np.repeat(patterns, 2, axis=0)
    y = np.tile([0, 1], len(patterns))
    features = list(range(1, n_features + 1))
    data = pd.DataFrame(X, columns=features)
    data.insert(0, "y", y)
    return data, features, [0, 1]


def test_native_sklearn_classifier_and_clone_protocol():
    estimator = RollingOCT(
        max_features="sqrt", random_state=7, total_time_limit=10
    )

    assert is_classifier(estimator)
    assert clone(estimator).get_params() == estimator.get_params()
    assert {"max_features", "random_state", "total_time_limit", "initial_depth"} <= set(
        estimator.get_params()
    )


def test_exact_depth3_is_available_through_classifier():
    X, y = _complete_binary_dataset()
    model = RollingOCT(
        depth=3,
        initial_depth=3,
        criterion="misclassification",
        time_limit=60,
        random_state=11,
    ).fit(X, y)

    assert model.fit_status_ == "completed"
    assert model.actual_depth_ == model.get_depth() == 3
    assert model.get_n_leaves() == 8
    assert set(model.tree_.branch_nodes) >= set(range(1, 8))
    assert len(model.subproblem_diagnostics_) == X.shape[1]
    assert all(d.depth == 3 for d in model.subproblem_diagnostics_)
    assert {d.candidate_feature for d in model.subproblem_diagnostics_} == set(
        model.features_
    )


def test_set_params_depth_controls_the_next_fit():
    data, features, _classes = _always_mixed_dataset()
    X = data[features].to_numpy()
    y = data["y"].to_numpy()
    model = RollingOCT(depth=2, time_limit=60).set_params(depth=3).fit(X, y)

    assert model.depth == 3
    assert model.actual_depth_ == 3
    assert 3 in model.depth_results_
    assert model.fit_status_ == "completed"


def test_seeded_feature_budget_is_reproducible_and_reported():
    X, y = _complete_binary_dataset(n_features=5)
    left = RollingOCT(max_features=3, random_state=42, time_limit=60).fit(X, y)
    right = RollingOCT(max_features=3, random_state=42, time_limit=60).fit(X, y)

    assert left.tree_.get_var_a_dict() == right.tree_.get_var_a_dict()
    assert left.subproblem_diagnostics_[0].n_features == 3
    assert left.tree_.branch_nodes[1].feature_index in left.features_


def test_predict_validates_binary_values_and_dataframe_schema():
    X, y = _complete_binary_dataset()
    model = RollingOCT(time_limit=60).fit(X, y)

    with pytest.raises(ValueError, match="binary"):
        model.predict(X.assign(f0=2))
    with pytest.raises(ValueError, match="feature names"):
        model.predict(X[["f1", "f0", "f2", "f3"]])
    with pytest.raises(ValueError, match="features"):
        model.predict(X.drop(columns="f3").to_numpy())


def test_object_backed_binary_values_use_safe_routing_fallback():
    X, y = _complete_binary_dataset()
    object_X = np.asarray(
        [[Decimal(int(value)) for value in row] for row in X.to_numpy()],
        dtype=object,
    )

    model = RollingOCT(time_limit=60).fit(object_X, y)

    np.testing.assert_array_equal(
        model.predict(object_X), model.predict(X.to_numpy())
    )


def test_mixed_object_values_report_binary_validation_error():
    X, y = _complete_binary_dataset()
    mixed = X.to_numpy(dtype=object, copy=True)
    mixed[0, 0] = "not-binary"

    with pytest.raises(ValueError, match="binary"):
        RollingOCT().fit(mixed, y)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"depth": 1}, "depth"),
        ({"initial_depth": 3, "depth": 2}, "initial_depth"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"max_features": 0}, "max_features"),
        ({"max_features": 99}, "number of input features"),
        ({"total_time_limit": 0}, "total_time_limit"),
        ({"random_state": -1}, "random_state"),
        ({"time_limit": float("nan")}, "time_limit"),
        ({"total_time_limit": float("inf")}, "total_time_limit"),
    ],
)
def test_set_params_values_are_revalidated_at_fit(params, message):
    X, y = _complete_binary_dataset()
    model = RollingOCT().set_params(**params)
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_global_deadline_returns_initial_valid_tree(monkeypatch):
    data, features, classes = _always_mixed_dataset()
    remaining = iter([10.0, -1.0])
    monkeypatch.setattr(
        RollingOptimizer,
        "_remaining_time",
        lambda self: next(remaining, -1.0),
    )
    optimizer = RollingOptimizer(
        SolverConfig(time_limit=60),
        GiniCriterion(),
        total_time_limit=60,
    )

    with pytest.warns(ConvergenceWarning, match="last valid tree"):
        tree, results = optimizer.build_tree(
            data, data, features, classes, target_depth=4
        )

    assert optimizer.fit_status_ == "time_limit"
    assert tree.get_depth() == 2
    assert set(results) == {2}


def test_frozen_leaf_is_still_a_populated_terminal_leaf():
    tree = DecisionTree(depth=2, features=[1, 2])
    tree.set_branch_feature(1, 1)
    tree.set_branch_feature(2, 2)
    tree.set_branch_feature(3, 2)
    for leaf_id in range(4, 8):
        tree.set_leaf_class(leaf_id, leaf_id % 2)

    tree.prune_leaf(4)

    assert tree.get_n_leaves() == 4
    assert tree.get_depth() == 2


def test_public_version_uses_release_source():
    assert __version__ == "2.1.0"

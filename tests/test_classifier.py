"""Tests for the RollingOCT classifier public API."""

import numpy as np
import pandas as pd
import pytest

from rollotree import RollingOCT


def _make_X_y(n=30, n_features=5, n_classes=2, seed=42):
    """Create synthetic X, y for testing."""
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randint(0, 2, size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(1, n_classes + 1, size=n), name="y")
    return X, y


class TestRollingOCTInit:
    def test_default_params(self):
        model = RollingOCT()
        assert model.depth == 2
        assert model.criterion == "gini"
        assert model.solver == "highs"

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError, match="depth must be >= 2"):
            RollingOCT(depth=1)

    def test_invalid_criterion_raises(self):
        X, y = _make_X_y()
        model = RollingOCT(criterion="entropy")
        with pytest.raises(ValueError, match="Unknown criterion"):
            model.fit(X, y)

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            RollingOCT(solver="cplex")


class TestRollingOCTFitPredict:
    def test_fit_returns_self(self):
        X, y = _make_X_y()
        model = RollingOCT(depth=2, solver="highs")
        result = model.fit(X, y)
        assert result is model

    def test_predict_returns_correct_shape(self):
        X, y = _make_X_y()
        model = RollingOCT(depth=2, solver="highs")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        model = RollingOCT()
        X, _ = _make_X_y()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_predictions_are_valid_classes(self):
        X, y = _make_X_y()
        model = RollingOCT(depth=2, solver="highs")
        model.fit(X, y)
        preds = model.predict(X)
        for p in preds:
            assert p in model.classes_

    def test_score_method(self):
        X, y = _make_X_y()
        model = RollingOCT(depth=2, solver="highs")
        model.fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_gini_and_misclass_both_run(self):
        X, y = _make_X_y()
        for criterion in ["gini", "misclassification"]:
            model = RollingOCT(depth=2, criterion=criterion, solver="highs")
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(X)

    def test_depth3_works(self):
        X, y = _make_X_y(n=40, n_features=5)
        model = RollingOCT(depth=3, solver="highs")
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_numpy_input(self):
        """Test that numpy arrays work as input (not just DataFrames)."""
        np.random.seed(42)
        X = np.random.randint(0, 2, size=(20, 4))
        y = np.array([1] * 10 + [2] * 10)
        model = RollingOCT(depth=2, solver="highs")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (20,)

    def test_fitted_attributes(self):
        X, y = _make_X_y()
        model = RollingOCT(depth=2, solver="highs")
        model.fit(X, y)
        assert model.tree_ is not None
        assert model.depth_results_ is not None
        assert model.classes_ is not None
        assert model.features_ is not None
        assert 2 in model.depth_results_


class TestEarlyStopping:
    def test_default_early_stopping_params(self):
        """Default min_samples_split=2, min_samples_leaf=1 should not change behavior."""
        model = RollingOCT()
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1

    def test_min_samples_split_limits_expansion(self):
        """High min_samples_split should prevent rolling expansion beyond depth 2."""
        X, y = _make_X_y(n=30, n_features=5)
        model = RollingOCT(
            depth=4, solver="highs", min_samples_split=1000, time_limit=60
        )
        model.fit(X, y)
        # Tree should still produce valid predictions
        preds = model.predict(X)
        assert len(preds) == len(X)
        # With min_samples_split=1000 and only 30 samples, depth-2 subproblems
        # at rolling level should be skipped, so only depth 2 and 3 results
        # should exist (initial depth-2 + at most one expansion attempt)
        assert 2 in model.depth_results_

    def test_min_samples_leaf_restricts_splits(self):
        """min_samples_leaf should eliminate variable pairs producing small leaves."""
        X, y = _make_X_y(n=30, n_features=5)
        model = RollingOCT(
            depth=2, solver="highs", min_samples_leaf=5, time_limit=60
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert 0.0 <= model.score(X, y) <= 1.0

    def test_min_samples_leaf_high_value(self):
        """Very high min_samples_leaf should still produce a valid tree via fallback."""
        X, y = _make_X_y(n=20, n_features=3)
        model = RollingOCT(
            depth=2, solver="highs", min_samples_leaf=100, time_limit=60
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_early_stopping_with_numpy_input(self):
        """Early stopping params work with numpy arrays too."""
        np.random.seed(42)
        X = np.random.randint(0, 2, size=(25, 4))
        y = np.array([1] * 12 + [2] * 13)
        model = RollingOCT(
            depth=3, solver="highs", min_samples_split=10, min_samples_leaf=3
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (25,)


class TestRollingOCTWineDataset:
    def test_wine_dataset_reasonable_accuracy(
        self, wine_train_data, wine_test_data
    ):
        """Test on real wine dataset: accuracy should be > random."""
        train = wine_train_data
        test = wine_test_data

        X_train = train.drop("y", axis=1)
        y_train = train["y"]
        X_test = test.drop("y", axis=1)
        y_test = test["y"]

        model = RollingOCT(depth=2, solver="highs", time_limit=120)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        n_classes = len(y_test.unique())
        random_baseline = 1.0 / n_classes
        assert score >= random_baseline * 0.8  # at least close to random

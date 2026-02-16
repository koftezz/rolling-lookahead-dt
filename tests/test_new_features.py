"""Tests for new features: sklearn protocol, predict_proba, feature importance,
tree visualization, tree inspection, model persistence, and input validation.
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rollotree import RollingOCT, export_text, export_graphviz


def _make_dataset(n=40, n_features=5, n_classes=2, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n, n_features))
    y = rng.randint(1, n_classes + 1, size=n)
    return X, y


@pytest.fixture
def fitted_model_2class():
    """A depth-2 model fitted on a small binary-class dataset."""
    X, y = _make_dataset(n=40, n_features=5, n_classes=2)
    model = RollingOCT(depth=2)
    model.fit(X, y)
    return model, X, y


# ── sklearn Protocol ──────────────────────────────────────────────────


class TestSklearnProtocol:
    def test_get_params_returns_all_init_params(self):
        model = RollingOCT(depth=3, criterion="gini", solver="highs")
        params = model.get_params()
        assert params["depth"] == 3
        assert params["criterion"] == "gini"
        assert params["solver"] == "highs"
        assert "n_jobs" in params
        assert "time_limit" in params
        assert "mip_gap" in params
        assert "big_m" in params
        assert "log_to_console" in params
        assert "min_samples_split" in params
        assert "min_samples_leaf" in params

    def test_set_params_updates_attributes(self):
        model = RollingOCT(depth=2)
        model.set_params(depth=4, criterion="misclassification")
        assert model.depth == 4
        assert model.criterion == "misclassification"

    def test_set_params_returns_self(self):
        model = RollingOCT()
        result = model.set_params(depth=3)
        assert result is model

    def test_set_params_invalid_raises(self):
        model = RollingOCT()
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(nonexistent_param=42)

    def test_get_set_roundtrip(self):
        model = RollingOCT(depth=5, criterion="misclassification", n_jobs=4)
        params = model.get_params()
        model2 = RollingOCT()
        model2.set_params(**params)
        assert model2.get_params() == params

    def test_clone_compatibility(self):
        """sklearn.base.clone uses get_params/set_params."""
        model = RollingOCT(depth=3, n_jobs=2)
        params = model.get_params()
        cloned = RollingOCT(**params)
        assert cloned.get_params() == params


# ── predict_proba ─────────────────────────────────────────────────────


class TestPredictProba:
    def test_proba_shape(self, fitted_model_2class):
        model, X, _y = fitted_model_2class
        proba = model.predict_proba(X)
        assert proba.shape == (40, 2)

    def test_proba_sums_to_one(self):
        X, y = _make_dataset(n=40, n_classes=3)
        model = RollingOCT(depth=2)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_proba_nonnegative(self, fitted_model_2class):
        model, X, _y = fitted_model_2class
        proba = model.predict_proba(X)
        assert (proba >= 0).all()

    def test_proba_multiclass(self):
        X, y = _make_dataset(n=60, n_features=6, n_classes=3)
        model = RollingOCT(depth=3)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 3
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_proba_before_fit_raises(self):
        model = RollingOCT()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((5, 3)))

    def test_proba_consistent_with_predict(self, fitted_model_2class):
        """argmax of proba should match predict."""
        model, X, _y = fitted_model_2class
        preds = model.predict(X)
        proba = model.predict_proba(X)
        proba_preds = np.array(model.classes_)[proba.argmax(axis=1)]
        np.testing.assert_array_equal(preds, proba_preds)


# ── Feature Importances ──────────────────────────────────────────────


class TestFeatureImportances:
    def test_importances_shape(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        assert model.feature_importances_.shape == (5,)

    def test_importances_sum_to_one(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-10

    def test_importances_nonnegative(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        assert (model.feature_importances_ >= 0).all()

    def test_importances_before_fit_is_none(self):
        model = RollingOCT()
        assert model.feature_importances_ is None


# ── Tree Visualization ────────────────────────────────────────────────


class TestExportText:
    def test_basic_output(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        text = export_text(model.tree_)
        assert "class:" in text
        assert "==" in text

    def test_with_feature_names(self):
        X, y = _make_dataset(n=40, n_features=3)
        model = RollingOCT(depth=2)
        model.fit(X, y)
        text = export_text(model.tree_, feature_names=["age", "income", "gender"])
        assert "age" in text or "income" in text or "gender" in text

    def test_wrong_feature_names_raises(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        with pytest.raises(ValueError, match="feature_names"):
            export_text(model.tree_, feature_names=["a", "b"])


class TestExportGraphviz:
    def test_basic_dot_output(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        dot = export_graphviz(model.tree_)
        assert "digraph Tree" in dot
        assert "class:" in dot

    def test_with_class_names(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        dot = export_graphviz(
            model.tree_,
            class_names=["negative", "positive"],
        )
        assert "negative" in dot or "positive" in dot


# ── Tree Inspection ───────────────────────────────────────────────────


class TestTreeInspection:
    def test_apply_returns_leaf_ids(self, fitted_model_2class):
        model, X, _y = fitted_model_2class
        leaf_ids = model.apply(X)
        assert leaf_ids.shape == (40,)
        for lid in np.unique(leaf_ids):
            assert lid in model.tree_.leaf_nodes

    def test_decision_path_is_sparse(self, fitted_model_2class):
        model, X, _y = fitted_model_2class
        path = model.decision_path(X)
        assert sparse.issparse(path)
        assert path.shape[0] == 40

    def test_decision_path_root_always_visited(self, fitted_model_2class):
        model, X, _y = fitted_model_2class
        path = model.decision_path(X)
        root_col = path[:, 1].toarray().flatten()
        assert (root_col == 1).all()

    def test_get_n_leaves(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        n_leaves = model.get_n_leaves()
        assert 2 <= n_leaves <= 4

    def test_get_depth(self, fitted_model_2class):
        model, _X, _y = fitted_model_2class
        assert model.get_depth() == 2

    def test_inspection_before_fit_raises(self):
        model = RollingOCT()
        X = np.zeros((5, 3))
        with pytest.raises(RuntimeError):
            model.apply(X)
        with pytest.raises(RuntimeError):
            model.decision_path(X)
        with pytest.raises(RuntimeError):
            model.get_depth()
        with pytest.raises(RuntimeError):
            model.get_n_leaves()


# ── Model Persistence ─────────────────────────────────────────────────


class TestModelPersistence:
    def test_save_load_roundtrip(self, fitted_model_2class, tmp_path):
        model, X, _y = fitted_model_2class
        preds_before = model.predict(X)

        path = str(tmp_path / "model.joblib")
        model.save(path)
        loaded = RollingOCT.load(path)
        np.testing.assert_array_equal(loaded.predict(X), preds_before)
        assert loaded.classes_ == model.classes_
        assert loaded.depth == model.depth

    def test_save_load_proba_preserved(self, fitted_model_2class, tmp_path):
        model, X, _y = fitted_model_2class
        proba_before = model.predict_proba(X)

        path = str(tmp_path / "model.joblib")
        model.save(path)
        loaded = RollingOCT.load(path)
        np.testing.assert_array_equal(loaded.predict_proba(X), proba_before)

    def test_pickle_roundtrip(self, fitted_model_2class):
        model, X, _y = fitted_model_2class
        preds_before = model.predict(X)

        loaded = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(loaded.predict(X), preds_before)

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib

        path = str(tmp_path / "not_a_model.joblib")
        joblib.dump({"not": "a model"}, path)
        with pytest.raises(TypeError, match="expected RollingOCT"):
            RollingOCT.load(path)


# ── Input Validation ──────────────────────────────────────────────────


class TestInputValidation:
    def test_non_binary_features_raises(self):
        X = np.array([[0, 1, 2], [1, 0, 3]])
        y = np.array([1, 2])
        model = RollingOCT(depth=2)
        with pytest.raises(ValueError, match="binary"):
            model.fit(X, y)

    def test_single_class_raises(self):
        X = np.array([[0, 1], [1, 0], [0, 0]])
        y = np.array([1, 1, 1])
        model = RollingOCT(depth=2)
        with pytest.raises(ValueError, match="at least 2 classes"):
            model.fit(X, y)

    def test_float_binary_accepted(self):
        """0.0 and 1.0 floats should be accepted as binary."""
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        y = np.array([1, 2, 1, 2])
        model = RollingOCT(depth=2)
        model.fit(X, y)  # Should not raise
        assert model._is_fitted

    def test_feature_names_stored_from_dataframe(self):
        X = pd.DataFrame(
            {"age": [0, 1, 0, 1], "income": [1, 0, 1, 0]},
        )
        y = np.array([1, 2, 1, 2])
        model = RollingOCT(depth=2)
        model.fit(X, y)
        np.testing.assert_array_equal(
            model.feature_names_in_, ["age", "income"]
        )

    def test_n_features_in_stored(self):
        X, y = _make_dataset(n=20, n_features=7)
        model = RollingOCT(depth=2)
        model.fit(X, y)
        assert model.n_features_in_ == 7

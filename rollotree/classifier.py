"""Public sklearn-compatible RollingOCT classifier."""

from numbers import Integral, Real
import warnings

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_consistent_length, validate_data

from rollotree.rolling.optimizer import RollingOptimizer
from rollotree.solver.base import SolverConfig
from rollotree.tree.impurity import get_criterion
from rollotree.tree.nodes import DecisionTree


class RolloTreeNotFittedError(NotFittedError, RuntimeError):
    """Backward-compatible not-fitted error for sklearn integrations."""


class RollingOCT(ClassifierMixin, BaseEstimator):
    """Rolling Optimal Classification Tree classifier.

    The estimator builds an exact depth-2 tree by default and deepens it with
    rolling exact OCT-2 subproblems. Set ``initial_depth=3`` to start from a
    globally exact complete depth-3 tree over the selected feature subset.

    Parameters
    ----------
    depth : int, default=2
        Maximum requested tree depth. Must be at least ``initial_depth``.
    criterion : {"gini", "misclassification"}, default="gini"
        Additive leaf objective optimized by each exact subproblem.
    solver : {"highs", "gurobi", "cbc"}, default="highs"
        PuLP solver backend.
    time_limit : float, default=1800
        Per-solve time limit in seconds.
    mip_gap : float or None, default=None
        Relative MIP optimality gap in the closed interval [0, 1].
    big_m : float, default=99
        Positive empty-leaf penalty used by the OCT-2 formulation.
    log_to_console : bool, default=False
        Whether solver logs are printed.
    min_samples_split : int, default=2
        Minimum samples required to attempt a rolling subproblem.
    min_samples_leaf : int, default=1
        Minimum samples required in every leaf of a complete exact subtree.
    n_jobs : int, default=1
        Process count for independent subproblems. ``-1`` uses all CPUs.
    max_features : int, float, {"sqrt", "log2"}, or None, default=None
        Number of candidate features sampled deterministically per subproblem.
        A float denotes a fraction of input features.
    random_state : int or None, default=None
        Seed controlling feature-subset selection.
    total_time_limit : float or None, default=None
        Wall-clock budget for the complete fit. The last valid tree is retained
        if the budget expires after an initial incumbent is available.
    initial_depth : {2, 3}, default=2
        Exact seed-tree depth. Depth 3 enumerates each root feature and solves
        the two independent OCT-2 child subtrees.
    """

    def __init__(
        self,
        depth: int = 2,
        criterion: str = "gini",
        solver: str = "highs",
        time_limit: float = 1800,
        mip_gap: float = None,
        big_m: float = 99,
        log_to_console: bool = False,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = 1,
        max_features=None,
        random_state: int = None,
        total_time_limit: float = None,
        initial_depth: int = 2,
    ):
        self.depth = depth
        self.criterion = criterion
        self.solver = solver
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.big_m = big_m
        self.log_to_console = log_to_console
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.random_state = random_state
        self.total_time_limit = total_time_limit
        self.initial_depth = initial_depth
        self._is_fitted = False

    @property
    def feature_importances_(self):
        """Split-frequency feature importances, or None before fitting."""
        return getattr(self, "_feature_importances", None)

    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances = value

    def fit(self, X, y) -> "RollingOCT":
        """Fit on a binary feature matrix and classification targets."""
        self._validate_parameters()
        self._is_fitted = False
        X_arr, y_arr = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=None,
            ensure_2d=True,
            ensure_min_samples=2,
        )
        check_classification_targets(y_arr)
        self._validate_binary_features(X_arr)

        if isinstance(self.max_features, Integral) and self.max_features > X_arr.shape[1]:
            raise ValueError(
                "max_features cannot exceed the number of input features "
                f"({X_arr.shape[1]}), got {self.max_features}."
            )

        self.features_ = list(range(1, X_arr.shape[1] + 1))
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError(
                f"y must have at least 2 classes, found {len(self.classes_)}."
            )

        import pandas as pd

        train_df = pd.DataFrame(X_arr, columns=self.features_)
        train_df.insert(0, "y", y_arr)

        solver_config = SolverConfig(
            solver_name=self.solver,
            time_limit=float(self.time_limit),
            mip_gap=self.mip_gap,
            log_to_console=self.log_to_console,
            big_m=float(self.big_m),
            min_samples_split=int(self.min_samples_split),
            min_samples_leaf=int(self.min_samples_leaf),
        )
        optimizer = RollingOptimizer(
            solver_config=solver_config,
            criterion=get_criterion(self.criterion),
            n_jobs=int(self.n_jobs),
            max_features=self.max_features,
            random_state=self.random_state,
            total_time_limit=self.total_time_limit,
            initial_depth=int(self.initial_depth),
        )
        self.tree_, self.depth_results_ = optimizer.build_tree(
            train_data=train_df,
            test_data=train_df,
            features=self.features_,
            classes=self.classes_.tolist(),
            target_depth=int(self.depth),
        )
        self.fit_status_ = optimizer.fit_status_
        self.fit_time_ = optimizer.fit_time_
        self.actual_depth_ = optimizer.actual_depth_
        self.subproblem_diagnostics_ = optimizer.subproblem_diagnostics_

        self.feature_importances_ = self.tree_.compute_feature_importances()
        self.tree_.store_leaf_distributions(X_arr, y_arr, self.classes_)
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        X_arr = self._validate_predict_X(X)
        return self.tree_.predict(X_arr)

    def predict_proba(self, X) -> np.ndarray:
        """Predict leaf-frequency class probabilities."""
        X_arr = self._validate_predict_X(X)
        return self.tree_.predict_proba(X_arr, self.classes_)

    def score(self, X, y) -> float:
        """Return classification accuracy."""
        predictions = self.predict(X)
        y_arr = np.asarray(y)
        check_consistent_length(predictions, y_arr)
        return float(accuracy_score(y_arr, predictions))

    def apply(self, X) -> np.ndarray:
        """Return the populated leaf node ID reached by each sample."""
        X_arr = self._validate_predict_X(X)
        return self.tree_.apply(X_arr)

    def decision_path(self, X):
        """Return a sparse indicator matrix for traversed nodes."""
        X_arr = self._validate_predict_X(X)
        return self.tree_.decision_path(X_arr)

    def get_depth(self) -> int:
        """Return the deepest populated terminal path."""
        self._check_is_fitted()
        return self.tree_.get_depth()

    def get_n_leaves(self) -> int:
        """Return the number of populated terminal leaves."""
        self._check_is_fitted()
        return self.tree_.get_n_leaves()

    def save(self, path: str):
        """Persist the fitted estimator with joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RollingOCT":
        """Load a persisted estimator, filling new defaults when possible."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected RollingOCT."
            )
        defaults = {
            "max_features": None,
            "random_state": None,
            "total_time_limit": None,
            "initial_depth": 2,
        }
        for name, value in defaults.items():
            if not hasattr(model, name):
                setattr(model, name, value)
        if getattr(model, "classes_", None) is not None:
            model.classes_ = np.asarray(model.classes_)
        if getattr(model, "tree_", None) is not None and not hasattr(
            model.tree_, "_routing_cache"
        ):
            model.tree_._routing_cache = None
        return model

    def __sklearn_is_fitted__(self):
        return bool(getattr(self, "_is_fitted", False))

    def _check_is_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise RolloTreeNotFittedError(
                "Model has not been fitted. Call fit() first."
            )

    def _validate_predict_X(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = validate_data(self, X, reset=False, dtype=None, ensure_2d=True)
        self._validate_binary_features(X_arr)
        return X_arr

    def _validate_binary_features(self, X: np.ndarray):
        try:
            valid = np.isin(X, (0, 1)).all()
        except TypeError:
            valid = False
        if not valid:
            values = np.unique(X)
            non_binary = [value for value in values if value not in (0, 1)]
            raise ValueError(
                "X must contain only binary (0/1) values. Found non-binary "
                f"values: {non_binary[:5]}. Use "
                "rollotree.preprocessing.helpers.make_data_binary() to "
                "binarize your data."
            )

        n_features = X.shape[1]
        if n_features > 200 and not self.__sklearn_is_fitted__():
            warnings.warn(
                f"X has {n_features} features. MIP complexity scales with "
                "O(P^2); consider max_features or prior feature selection.",
                UserWarning,
                stacklevel=3,
            )

    def _validate_parameters(self):
        if (
            not isinstance(self.depth, Integral)
            or isinstance(self.depth, bool)
            or self.depth < 2
        ):
            raise ValueError("depth must be an integer >= 2")
        if self.initial_depth not in (2, 3):
            raise ValueError("initial_depth must be either 2 or 3")
        if self.initial_depth > self.depth:
            raise ValueError("initial_depth cannot exceed depth")
        get_criterion(self.criterion)
        SolverConfig(solver_name=self.solver)

        self._validate_positive_real("time_limit", self.time_limit)
        self._validate_positive_real("big_m", self.big_m)
        if self.total_time_limit is not None:
            self._validate_positive_real(
                "total_time_limit", self.total_time_limit
            )
        if self.mip_gap is not None and (
            not isinstance(self.mip_gap, Real)
            or isinstance(self.mip_gap, bool)
            or not 0 <= self.mip_gap <= 1
        ):
            raise ValueError("mip_gap must be None or a number in [0, 1]")

        self._validate_minimum_integer(
            "min_samples_split", self.min_samples_split, 2
        )
        self._validate_minimum_integer(
            "min_samples_leaf", self.min_samples_leaf, 1
        )
        if (
            not isinstance(self.n_jobs, Integral)
            or isinstance(self.n_jobs, bool)
            or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be a non-zero integer")
        if self.random_state is not None and (
            not isinstance(self.random_state, Integral)
            or isinstance(self.random_state, bool)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be None or a non-negative integer")

        if self.max_features is not None:
            if isinstance(self.max_features, str):
                if self.max_features not in ("sqrt", "log2"):
                    raise ValueError(
                        "max_features must be None, a positive int, a float "
                        "in (0, 1], 'sqrt', or 'log2'"
                    )
            elif isinstance(self.max_features, Integral) and not isinstance(
                self.max_features, bool
            ):
                if self.max_features < 1:
                    raise ValueError("integer max_features must be >= 1")
            elif isinstance(self.max_features, Real) and not isinstance(
                self.max_features, bool
            ):
                if not 0 < self.max_features <= 1:
                    raise ValueError("float max_features must be in (0, 1]")
            else:
                raise ValueError(
                    "max_features must be None, a positive int, a float in "
                    "(0, 1], 'sqrt', or 'log2'"
                )

    @staticmethod
    def _validate_positive_real(name, value):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number")

    @staticmethod
    def _validate_minimum_integer(name, value, minimum):
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")


__all__ = ["RollingOCT", "RolloTreeNotFittedError"]

"""Public API: sklearn-compatible RollingOCT classifier."""

import logging
import warnings

import joblib
import numpy as np
import pandas as pd

from rollotree.tree.nodes import DecisionTree
from rollotree.tree.impurity import get_criterion
from rollotree.solver.base import SolverConfig
from rollotree.rolling.optimizer import RollingOptimizer

logger = logging.getLogger(__name__)


def _to_feature_array(X) -> np.ndarray:
    """Convert feature input to numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _to_label_array(y) -> np.ndarray:
    """Convert label input to numpy array."""
    if isinstance(y, pd.Series):
        return y.values
    return np.asarray(y)


class RollingOCT:
    """
    Rolling Optimal Classification Tree classifier.

    Uses the OCT-2 MIP formulation with rolling subtree lookahead
    to build interpretable decision trees of arbitrary depth.

    Parameters
    ----------
    depth : int, default=2
        Maximum tree depth. Must be >= 2.
    criterion : str, default="gini"
        Impurity criterion: "gini" or "misclassification".
    solver : str, default="highs"
        Solver backend: "highs" (open source, bundled with PuLP)
        or "gurobi" (commercial, requires separate install).
    time_limit : float, default=1800
        Time limit per OCT-2 subproblem, in seconds.
    mip_gap : float or None, default=None
        MIP optimality gap tolerance.
    big_m : float, default=99
        Big-M penalty for empty-leaf feature pairs.
    log_to_console : bool, default=False
        Whether the solver should print logs.
    min_samples_split : int, default=2
        Minimum number of samples required at a node to solve a
        depth-2 subproblem. Nodes with fewer samples are pruned.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.
        Feature pairs that would produce a leaf with fewer samples
        are eliminated from the formulation.
    n_jobs : int, default=1
        Number of parallel processes for solving OCT-2 subproblems
        during rolling expansion.
        1 = sequential (no multiprocessing overhead),
        -1 = use all available CPU cores,
        N = use N processes.

    Attributes
    ----------
    tree_ : DecisionTree
        The fitted decision tree (available after fit).
    depth_results_ : dict
        Accuracy and timing results for each depth level.
    classes_ : list
        Unique class labels seen during fit.
    features_ : list
        Feature indices used during fit.
    feature_importances_ : np.ndarray
        Feature importances based on split frequency (available after fit).
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : np.ndarray or None
        Feature names if X was a DataFrame during fit.

    Examples
    --------
    >>> model = RollingOCT(depth=3, criterion="gini", solver="highs")
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> accuracy = model.score(X_test, y_test)
    >>> probabilities = model.predict_proba(X_test)
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
    ):
        if depth < 2:
            raise ValueError("depth must be >= 2")
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

        # Validate solver name early
        SolverConfig(solver_name=solver)

        self.tree_: DecisionTree = None
        self.depth_results_: dict = None
        self.classes_: list = None
        self.features_: list = None
        self.feature_importances_: np.ndarray = None
        self.n_features_in_: int = None
        self.feature_names_in_: np.ndarray = None
        self._is_fitted = False

    # ── sklearn Protocol ──────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Ignored (no nested estimators), kept for sklearn API compat.

        Returns
        -------
        dict : Parameter names mapped to their values.
        """
        return {
            "depth": self.depth,
            "criterion": self.criterion,
            "solver": self.solver,
            "time_limit": self.time_limit,
            "mip_gap": self.mip_gap,
            "big_m": self.big_m,
            "log_to_console": self.log_to_console,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params) -> "RollingOCT":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
        """
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter '{key}' for RollingOCT. "
                    f"Valid parameters: {list(valid.keys())}"
                )
            setattr(self, key, value)
        return self

    # ── Core API ──────────────────────────────────────────────────────

    def fit(self, X, y) -> "RollingOCT":
        """
        Fit the rolling OCT model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix (0/1 values only).
            Can be DataFrame or numpy array.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X contains non-binary values or y has fewer than 2 classes.
        """
        X_arr = _to_feature_array(X)
        y_arr = _to_label_array(y)

        # ── Input validation ──
        self._validate_input(X_arr, y_arr)

        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = None

        self.n_features_in_ = X_arr.shape[1]
        self.features_ = list(range(1, X_arr.shape[1] + 1))
        self.classes_ = sorted(np.unique(y_arr).tolist())

        # Build internal DataFrame: y at column 0, features at columns 1..n
        train_df = pd.DataFrame(X_arr, columns=self.features_)
        train_df.insert(0, "y", y_arr)

        # For rolling optimizer we need test_data too; during fit, we pass
        # train as test to get per-depth metrics on training data.
        # Users should call score() separately for test evaluation.
        solver_config = SolverConfig(
            solver_name=self.solver,
            time_limit=self.time_limit,
            mip_gap=self.mip_gap,
            log_to_console=self.log_to_console,
            big_m=self.big_m,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        impurity = get_criterion(self.criterion)
        optimizer = RollingOptimizer(
            solver_config=solver_config, criterion=impurity, n_jobs=self.n_jobs
        )

        self.tree_, self.depth_results_ = optimizer.build_tree(
            train_data=train_df,
            test_data=train_df,
            features=self.features_,
            classes=self.classes_,
            target_depth=self.depth,
        )

        # Post-fit: compute feature importances and store leaf distributions
        self.feature_importances_ = self.tree_.compute_feature_importances()
        self.tree_.store_leaf_distributions(X_arr, y_arr, self.classes_)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        return self.tree_.predict(_to_feature_array(X))

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Probabilities are computed from the proportion of training
        samples of each class in the leaf that each sample is routed to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Columns correspond to ``self.classes_`` in sorted order.
        """
        self._check_is_fitted()
        return self.tree_.predict_proba(_to_feature_array(X), self.classes_)

    def score(self, X, y) -> float:
        """
        Return accuracy on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        float: Accuracy (fraction of correct predictions).
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == _to_label_array(y)))

    # ── Tree inspection ───────────────────────────────────────────────

    def apply(self, X) -> np.ndarray:
        """Return leaf node IDs for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with integer leaf node IDs.
        """
        self._check_is_fitted()
        return self.tree_.apply(_to_feature_array(X))

    def decision_path(self, X):
        """Return the decision path through the tree for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_samples, max_node_id + 1)
            Entry (i, j) is 1 if sample i passes through node j.
        """
        self._check_is_fitted()
        return self.tree_.decision_path(_to_feature_array(X))

    def get_depth(self) -> int:
        """Return the actual depth of the deepest active path."""
        self._check_is_fitted()
        return self.tree_.get_depth()

    def get_n_leaves(self) -> int:
        """Return the number of active (non-pruned) leaf nodes."""
        self._check_is_fitted()
        return self.tree_.get_n_leaves()

    # ── Model persistence ─────────────────────────────────────────────

    def save(self, path: str):
        """Save the fitted model to disk using joblib.

        Parameters
        ----------
        path : str
            File path to save to (e.g. "model.joblib").
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RollingOCT":
        """Load a model from disk.

        Parameters
        ----------
        path : str
            File path to load from.

        Returns
        -------
        RollingOCT : The loaded model.
        """
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, "
                f"expected RollingOCT."
            )
        return model

    # ── Private helpers ───────────────────────────────────────────────

    def _check_is_fitted(self):
        """Raise if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    @staticmethod
    def _validate_input(X: np.ndarray, y: np.ndarray):
        """Validate that features are binary and labels have >=2 classes."""
        unique_vals = np.unique(X)
        non_binary = set(unique_vals) - {0, 1, 0.0, 1.0}
        if non_binary:
            raise ValueError(
                f"X must contain only binary (0/1) values. "
                f"Found non-binary values: {sorted(non_binary)[:5]}. "
                f"Use rollotree.preprocessing.helpers.make_data_binary() "
                f"to binarize your data."
            )

        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError(
                f"y must have at least 2 classes, found {n_classes}."
            )

        n_features = X.shape[1]
        if n_features > 200:
            warnings.warn(
                f"X has {n_features} features. MIP complexity scales with "
                f"O(P^2); consider feature selection for faster solving.",
                UserWarning,
                stacklevel=3,
            )

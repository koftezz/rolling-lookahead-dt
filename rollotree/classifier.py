"""Public API: sklearn-like RollingOCT classifier."""

import logging

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

    Examples
    --------
    >>> model = RollingOCT(depth=3, criterion="gini", solver="highs")
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> accuracy = model.score(X_test, y_test)
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
        self._is_fitted = False

    def fit(self, X, y) -> "RollingOCT":
        """
        Fit the rolling OCT model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix. Can be DataFrame or numpy array.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        X_arr = _to_feature_array(X)
        y_arr = _to_label_array(y)

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
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.tree_.predict(_to_feature_array(X))

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

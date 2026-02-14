"""Public API: sklearn-like RollingOCT classifier."""

import logging

import numpy as np
import pandas as pd

from rollo_oct.tree.nodes import DecisionTree
from rollo_oct.tree.impurity import get_criterion
from rollo_oct.solver.base import SolverConfig
from rollo_oct.rolling.optimizer import RollingOptimizer

logger = logging.getLogger(__name__)


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
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        y_arr = np.asarray(y) if not isinstance(y, pd.Series) else y.values

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
        )
        impurity = get_criterion(self.criterion)
        optimizer = RollingOptimizer(
            solver_config=solver_config, criterion=impurity
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
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        return self.tree_.predict(X_arr)

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
        y_arr = np.asarray(y) if not isinstance(y, pd.Series) else y.values
        return float(np.mean(predictions == y_arr))

"""Impurity criteria for decision tree optimization."""

from abc import ABC, abstractmethod

import numpy as np


def _split_count_matrices(features: np.ndarray) -> dict:
    """Return sample-count matrices for all depth-2 leaf paths.

    Rows select the first split feature and columns select the second split
    feature.  Matrix multiplication computes every feature pair at once.
    """
    n_samples = features.shape[0]
    both_one = features.T @ features
    one_counts = features.sum(axis=0)
    return {
        (1, 1): both_one,
        (1, 0): one_counts[:, None] - both_one,
        (0, 1): one_counts[None, :] - both_one,
        (0, 0): (
            n_samples
            - one_counts[:, None]
            - one_counts[None, :]
            + both_one
        ),
    }


def _coefficient_inputs(data, features, leaf_nodes, leaf_paths, classes, y_idx):
    """Compute total counts and per-class sufficient statistics.

    The previous implementation materialized one target array for every
    ``(leaf, feature_i, feature_j)`` tuple.  This version keeps only dense
    ``p x p`` count matrices and is exact for binary features.
    """
    feature_matrix = np.asarray(data[:, features], dtype=np.int64)
    labels = np.asarray(data[:, y_idx])
    classes = np.asarray(np.unique(labels) if classes is None else classes)

    totals_by_path = _split_count_matrices(feature_matrix)
    totals = {
        leaf: totals_by_path[tuple(leaf_paths[leaf])]
        for leaf in leaf_nodes
    }
    class_counts = {leaf: [] for leaf in leaf_nodes}
    for class_label in classes:
        counts_by_path = _split_count_matrices(
            feature_matrix[labels == class_label]
        )
        for leaf in leaf_nodes:
            class_counts[leaf].append(counts_by_path[tuple(leaf_paths[leaf])])
    return totals, class_counts


def _matrix_to_pair_dict(values, totals, features) -> dict:
    """Convert non-empty entries in a coefficient matrix to pair keys."""
    rows, cols = np.where(totals > 0)
    return {
        (features[row], features[col]): values[row, col].item()
        for row, col in zip(rows, cols)
    }


class ImpurityCriterion(ABC):
    """Abstract base class for impurity calculation strategies."""

    @abstractmethod
    def compute_leaf_coefficients(
        self,
        data: np.ndarray,
        features: list,
        leaf_nodes: list,
        leaf_paths: dict,
        classes: list,
        y_idx: int = 0,
    ) -> dict:
        """Return ``leaf -> feature-pair -> objective coefficient``."""

    def combine_subproblem_objective(
        self, value: float, subset_n: int, total_n: int
    ) -> float:
        """Scale a subtree objective for comparison in a larger tree.

        The default keeps custom OCT-2 criteria written before release 2.1
        instantiable. Exact depth-3 restricts itself to the built-in criteria,
        which override this method with their known objective scale.
        """
        return float(value)


class GiniCriterion(ImpurityCriterion):
    """Weighted Gini impurity criterion (Equation 5 in the paper)."""

    def compute_leaf_coefficients(
        self, data, features, leaf_nodes, leaf_paths, classes, y_idx=0
    ):
        totals, class_counts = _coefficient_inputs(
            data, features, leaf_nodes, leaf_paths, classes, y_idx
        )
        n_samples = len(data)
        result = {}
        for leaf in leaf_nodes:
            total = totals[leaf]
            sum_squares = np.zeros_like(total, dtype=np.float64)
            for counts in class_counts[leaf]:
                sum_squares += counts.astype(np.float64) ** 2

            values = np.zeros_like(total, dtype=np.float64)
            nonempty = total > 0
            values[nonempty] = (
                total[nonempty]
                - sum_squares[nonempty] / total[nonempty]
            ) / n_samples
            result[leaf] = _matrix_to_pair_dict(values, total, features)
        return result

    def combine_subproblem_objective(self, value, subset_n, total_n):
        return float(value) * subset_n / total_n

    @staticmethod
    def _gini_index(arr, total_n, classes, y_idx=0):
        """Compute weighted Gini impurity for a subset."""
        sum_sq = sum(
            (len(arr[np.where(arr[:, y_idx] == k)]) / len(arr)) ** 2
            for k in classes
        )
        return (len(arr) / total_n) * (1 - sum_sq)


class MisclassificationCriterion(ImpurityCriterion):
    """Misclassification error criterion (Equation 4 in the paper)."""

    def compute_leaf_coefficients(
        self, data, features, leaf_nodes, leaf_paths, classes=None, y_idx=0
    ):
        totals, class_counts = _coefficient_inputs(
            data, features, leaf_nodes, leaf_paths, classes, y_idx
        )
        result = {}
        for leaf in leaf_nodes:
            total = totals[leaf]
            max_class = np.zeros_like(total, dtype=np.int64)
            for counts in class_counts[leaf]:
                np.maximum(max_class, counts, out=max_class)
            values = total - max_class
            result[leaf] = _matrix_to_pair_dict(values, total, features)
        return result

    def combine_subproblem_objective(self, value, subset_n, total_n):
        return float(value)


def get_criterion(name: str) -> ImpurityCriterion:
    """Factory function for impurity criteria."""
    criteria = {
        "gini": GiniCriterion,
        "misclassification": MisclassificationCriterion,
    }
    if name not in criteria:
        raise ValueError(
            f"Unknown criterion '{name}'. Choose from: {list(criteria.keys())}"
        )
    return criteria[name]()

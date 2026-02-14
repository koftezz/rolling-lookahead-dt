"""Impurity criteria for decision tree optimization."""

from abc import ABC, abstractmethod
import numpy as np


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
        """
        Compute the objective coefficients for each leaf node.

        For each leaf and each (feature_i, feature_j) split pair, computes
        the impurity of the data subset reaching that leaf.

        Args:
            data: Array of shape (n_samples, n_columns) with target at y_idx.
            features: List of feature column indices (P).
            leaf_nodes: List of leaf node IDs (e.g., [4, 5, 6, 7]).
            leaf_paths: Dict mapping leaf_id -> [first_split_val, second_split_val].
            classes: List of unique class labels (K).
            y_idx: Column index of the target variable.

        Returns:
            Dict of {leaf_id: {(feature_i, feature_j): coefficient}}.
        """
        ...


class GiniCriterion(ImpurityCriterion):
    """Weighted Gini impurity criterion (Equation 5 in the paper)."""

    def compute_leaf_coefficients(self, data, features, leaf_nodes,
                                  leaf_paths, classes, y_idx=0):
        n = len(data)
        result = {}
        for leaf in leaf_nodes:
            temp = {}
            first_val = leaf_paths[leaf][0]
            second_val = leaf_paths[leaf][1]
            for fi in features:
                arr = data[np.where(data[:, fi] == first_val)]
                for fj in features:
                    arr2 = arr[np.where(arr[:, fj] == second_val)]
                    if len(arr2) > 0:
                        temp[(fi, fj)] = self._gini_index(arr2, n, classes, y_idx)
            result[leaf] = temp
        return result

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

    def compute_leaf_coefficients(self, data, features, leaf_nodes,
                                  leaf_paths, classes=None, y_idx=0):
        result = {}
        for leaf in leaf_nodes:
            temp = {}
            first_val = leaf_paths[leaf][0]
            second_val = leaf_paths[leaf][1]
            for fi in features:
                arr = data[np.where(data[:, fi] == first_val)]
                for fj in features:
                    arr2 = arr[np.where(arr[:, fj] == second_val)]
                    if len(arr2) > 0:
                        values, counts = np.unique(arr2[:, y_idx],
                                                   return_counts=True)
                        temp[(fi, fj)] = len(arr2) - counts.max()
            result[leaf] = temp
        return result


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

"""Impurity criteria for decision tree optimization."""

from abc import ABC, abstractmethod
import numpy as np


def _precompute_leaf_subsets(data, features, leaf_paths, leaf_nodes, y_idx):
    """Precompute boolean masks and class labels for all (leaf, fi, fj) combos.

    Instead of nested np.where calls that create temporary arrays each time,
    precompute feature masks once and combine them with bitwise operations.

    Returns:
        Dict of {leaf_id: {(fi, fj): (count, class_counts_array)}}
        where class_counts_array maps class_label -> count.
    """
    n_samples = len(data)
    features_arr = np.array(features)

    # Precompute boolean masks: mask_eq1[fi] = (data[:, fi] == 1)
    mask_eq1 = {}
    mask_eq0 = {}
    for fi in features:
        mask_eq1[fi] = data[:, fi] == 1
        mask_eq0[fi] = ~mask_eq1[fi]

    y_col = data[:, y_idx]

    result = {}
    for leaf in leaf_nodes:
        first_val = leaf_paths[leaf][0]
        second_val = leaf_paths[leaf][1]
        leaf_data = {}

        for fi in features:
            fi_mask = mask_eq1[fi] if first_val == 1 else mask_eq0[fi]
            for fj in features:
                fj_mask = mask_eq1[fj] if second_val == 1 else mask_eq0[fj]
                combined = fi_mask & fj_mask
                count = int(np.sum(combined))
                if count > 0:
                    leaf_data[(fi, fj)] = (count, y_col[combined])
        result[leaf] = leaf_data

    return result, n_samples


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
        subsets, n = _precompute_leaf_subsets(
            data, features, leaf_paths, leaf_nodes, y_idx
        )
        classes_arr = np.array(classes)
        result = {}
        for leaf in leaf_nodes:
            temp = {}
            for (fi, fj), (count, y_subset) in subsets[leaf].items():
                # Vectorized Gini: count occurrences of each class at once
                sum_sq = 0.0
                for k in classes_arr:
                    p = np.sum(y_subset == k) / count
                    sum_sq += p * p
                temp[(fi, fj)] = (count / n) * (1.0 - sum_sq)
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
        subsets, n = _precompute_leaf_subsets(
            data, features, leaf_paths, leaf_nodes, y_idx
        )
        result = {}
        for leaf in leaf_nodes:
            temp = {}
            for (fi, fj), (count, y_subset) in subsets[leaf].items():
                _values, counts = np.unique(y_subset, return_counts=True)
                temp[(fi, fj)] = count - counts.max()
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

"""Tests for impurity criteria."""

import numpy as np
import pytest

from rollotree.tree.impurity import (
    GiniCriterion,
    MisclassificationCriterion,
    get_criterion,
)


class TestGiniCriterion:
    def test_pure_leaf_gini_zero(self):
        """All same class -> gini = 0."""
        gini = GiniCriterion()
        # 4 samples, all class 1
        arr = np.array([[1], [1], [1], [1]])
        result = gini._gini_index(arr, total_n=4, classes=[1, 2], y_idx=0)
        assert result == 0.0

    def test_even_split(self):
        """50/50 split -> gini = 0.5 * weight."""
        gini = GiniCriterion()
        arr = np.array([[1], [1], [2], [2]])
        # total_n=4, so weight = 4/4 = 1.0
        result = gini._gini_index(arr, total_n=4, classes=[1, 2], y_idx=0)
        assert abs(result - 0.5) < 1e-10

    def test_weighted_gini(self):
        """Weighted by proportion of total data."""
        gini = GiniCriterion()
        arr = np.array([[1], [2]])  # 2 samples, 50/50
        # total_n=10, weight = 2/10 = 0.2
        result = gini._gini_index(arr, total_n=10, classes=[1, 2], y_idx=0)
        assert abs(result - 0.1) < 1e-10  # 0.2 * 0.5

    def test_compute_leaf_coefficients(self):
        """Test that coefficients dict has expected structure."""
        gini = GiniCriterion()
        # Simple 4-sample dataset
        data = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 0, 1],
        ])
        features = [1, 2]
        leaf_nodes = [4, 5, 6, 7]
        leaf_paths = {4: [1, 1], 5: [1, 0], 6: [0, 1], 7: [0, 0]}
        classes = [1, 2]

        result = gini.compute_leaf_coefficients(
            data, features, leaf_nodes, leaf_paths, classes
        )
        assert isinstance(result, dict)
        for leaf in leaf_nodes:
            assert leaf in result
            assert isinstance(result[leaf], dict)


class TestMisclassificationCriterion:
    def test_pure_leaf_zero_error(self):
        """All same class -> error = 0."""
        misclass = MisclassificationCriterion()
        data = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
        ])
        features = [1, 2]
        leaf_paths = {4: [1, 1], 5: [1, 0], 6: [0, 1], 7: [0, 0]}
        result = misclass.compute_leaf_coefficients(
            data, features, [4], leaf_paths, [1]
        )
        # Leaf 4 routes samples where feat1=1 and feat2=1
        for coeff in result[4].values():
            assert coeff == 0

    def test_known_misclassification(self):
        """3 of class 1, 1 of class 2 -> error = 1."""
        misclass = MisclassificationCriterion()
        # All 4 samples have feat1=1, feat2=1 so they all route to leaf 4
        data = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
        ])
        features = [1, 2]
        leaf_paths = {4: [1, 1]}

        result = misclass.compute_leaf_coefficients(
            data, features, [4], leaf_paths, [1, 2]
        )
        # With (feat1=1, feat2=1): all 4 samples reach leaf 4
        # 3 class 1, 1 class 2 -> error = 1
        assert result[4][(1, 2)] == 1


class TestCriterionFactory:
    def test_gini(self):
        c = get_criterion("gini")
        assert isinstance(c, GiniCriterion)

    def test_misclassification(self):
        c = get_criterion("misclassification")
        assert isinstance(c, MisclassificationCriterion)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            get_criterion("entropy")

"""Tests for tree node classes and utilities."""

import numpy as np
import pytest

from rollotree.tree.utils import (
    generate_nodes,
    get_leaf_paths_depth2,
    leaf_pattern,
    parent_pattern,
    get_child,
)
from rollotree.tree.nodes import DecisionNode, LeafNode, DecisionTree


class TestGenerateNodes:
    def test_depth2(self):
        parents, leaves = generate_nodes(2)
        assert parents == [1, 2, 3]
        assert leaves == [4, 5, 6, 7]

    def test_depth3(self):
        parents, leaves = generate_nodes(3)
        assert parents == [1, 2, 3, 4, 5, 6, 7]
        assert leaves == [8, 9, 10, 11, 12, 13, 14, 15]

    def test_depth1(self):
        parents, leaves = generate_nodes(1)
        assert parents == [1]
        assert leaves == [2, 3]


class TestLeafPaths:
    def test_depth2_paths(self):
        paths = get_leaf_paths_depth2()
        assert paths == {4: [1, 1], 5: [1, 0], 6: [0, 1], 7: [0, 0]}


class TestLeafPattern:
    def test_leaf4_at_node2(self):
        # Subtree leaf 4 expanded at node 2: (4-4) + 4*2 = 8
        assert leaf_pattern(4, 2, 2) == 8

    def test_leaf5_at_node2(self):
        # Subtree leaf 5 expanded at node 2: (5-4) + 4*2 = 9
        assert leaf_pattern(5, 2, 2) == 9

    def test_leaf6_at_node3(self):
        # Subtree leaf 6 expanded at node 3: (6-4) + 4*3 = 14
        assert leaf_pattern(6, 2, 3) == 14

    def test_leaf7_at_node3(self):
        # Subtree leaf 7 expanded at node 3: (7-4) + 4*3 = 15
        assert leaf_pattern(7, 2, 3) == 15


class TestParentPattern:
    def test_root_maps_to_leaf(self):
        assert parent_pattern(1, 5) == 5

    def test_node2(self):
        # sub_leaf=2 at leaf_node=5: 2*5 + 2%2 = 10
        assert parent_pattern(2, 5) == 10

    def test_node3(self):
        # sub_leaf=3 at leaf_node=5: 2*5 + 3%2 = 11
        assert parent_pattern(3, 5) == 11

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            parent_pattern(8, 5)


class TestGetChild:
    def test_even_node(self):
        # Node 2 at depth 1, target depth 2: 2*2=4
        assert get_child(1, 2, 2) == 4

    def test_odd_node(self):
        # Node 3 at depth 1, target depth 2: 3*2+1=7
        assert get_child(1, 2, 3) == 7

    def test_same_depth(self):
        assert get_child(2, 2, 4) == 4


class TestDecisionNode:
    def test_child_ids(self):
        node = DecisionNode(node_id=2)
        assert node.left_child_id == 4
        assert node.right_child_id == 5

    def test_parent_id(self):
        node = DecisionNode(node_id=5)
        assert node.parent_id == 2

    def test_root_has_no_parent(self):
        node = DecisionNode(node_id=1)
        assert node.parent_id is None


class TestLeafNode:
    def test_parent_id(self):
        leaf = LeafNode(node_id=7)
        assert leaf.parent_id == 3


class TestDecisionTree:
    def test_init_depth2(self):
        tree = DecisionTree(depth=2, features=[1, 2, 3])
        assert len(tree.branch_nodes) == 3  # nodes 1, 2, 3
        assert len(tree.leaf_nodes) == 4  # nodes 4, 5, 6, 7

    def test_set_branch_feature(self):
        tree = DecisionTree(depth=2, features=[1, 2, 3])
        tree.set_branch_feature(1, 2)
        assert tree.branch_nodes[1].feature_index == 2
        np.testing.assert_array_equal(
            tree.branch_nodes[1].feature_vector, [0, 1, 0]
        )

    def test_predict_single(self):
        """Test routing through a manually constructed tree."""
        tree = DecisionTree(depth=2, features=[1, 2])
        # Root splits on feature 1
        tree.set_branch_feature(1, 1)
        # Left child splits on feature 2
        tree.set_branch_feature(2, 2)
        # Right child splits on feature 2
        tree.set_branch_feature(3, 2)

        tree.set_leaf_class(4, "A")  # feat1=1, feat2=1
        tree.set_leaf_class(5, "B")  # feat1=1, feat2=0
        tree.set_leaf_class(6, "C")  # feat1=0, feat2=1
        tree.set_leaf_class(7, "D")  # feat1=0, feat2=0

        # Sample [1, 1] -> left (feat1=1) -> left (feat2=1) -> leaf 4
        assert tree.predict_single(np.array([1, 1])) == "A"
        # Sample [1, 0] -> left (feat1=1) -> right (feat2=0) -> leaf 5
        assert tree.predict_single(np.array([1, 0])) == "B"
        # Sample [0, 1] -> right (feat1=0) -> left (feat2=1) -> leaf 6
        assert tree.predict_single(np.array([0, 1])) == "C"
        # Sample [0, 0] -> right (feat1=0) -> right (feat2=0) -> leaf 7
        assert tree.predict_single(np.array([0, 0])) == "D"

    def test_predict_batch(self):
        tree = DecisionTree(depth=2, features=[1, 2])
        tree.set_branch_feature(1, 1)
        tree.set_branch_feature(2, 2)
        tree.set_branch_feature(3, 2)
        tree.set_leaf_class(4, 1)
        tree.set_leaf_class(5, 2)
        tree.set_leaf_class(6, 1)
        tree.set_leaf_class(7, 2)

        X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
        preds = tree.predict(X)
        np.testing.assert_array_equal(preds, [1, 2, 1, 2])

    def test_predict_batch_preserves_full_string_labels(self):
        tree = DecisionTree(depth=2, features=[1, 2])
        tree.set_branch_feature(1, 1)
        tree.set_branch_feature(2, 2)
        tree.set_branch_feature(3, 2)
        tree.set_leaf_class(4, "a")
        tree.set_leaf_class(5, "long_label")
        tree.set_leaf_class(6, "a")
        tree.set_leaf_class(7, "long_label")

        predictions = tree.predict(
            np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
        )

        assert predictions.tolist() == ["a", "long_label", "a", "long_label"]

    def test_predict_with_pruned_nodes(self):
        tree = DecisionTree(depth=2, features=[1, 2])
        tree.set_branch_feature(1, 1)
        tree.set_branch_feature(2, 2)
        tree.set_branch_feature(3, 2)
        tree.set_leaf_class(4, 1)
        tree.set_leaf_class(5, 2)
        # Prune the right subtree: node 3's children become leaves early
        # But we need node 6 and 7 to exist with classes
        tree.set_leaf_class(6, 3)
        tree.set_leaf_class(7, 3)

        # Prune leaf 6 - it should still be reachable
        tree.prune_leaf(6)
        # Sample [0, 1] -> right -> left -> leaf 6 (pruned, but has class)
        assert tree.predict_single(np.array([0, 1])) == 3

    def test_get_misclassified_leaves(self):
        tree = DecisionTree(depth=2, features=[1, 2])
        tree.set_branch_feature(1, 1)
        tree.set_branch_feature(2, 2)
        tree.set_branch_feature(3, 2)
        tree.set_leaf_class(4, 1)
        tree.set_leaf_class(5, 2)
        tree.set_leaf_class(6, 1)
        tree.set_leaf_class(7, 2)

        X = np.array([[1, 1], [1, 1], [0, 0]])
        y = np.array([1, 2, 2])  # leaf 4 has mixed classes (1 and 2)

        misclassified = tree.get_misclassified_leaves(X, y)
        assert 4 in misclassified

    def test_extend_at_leaf(self):
        from rollotree.solver.base import OCT2Solution, SolverStatus

        tree = DecisionTree(depth=2, features=[1, 2, 3])
        tree.set_branch_feature(1, 1)
        tree.set_branch_feature(2, 2)
        tree.set_branch_feature(3, 3)
        for lid in [4, 5, 6, 7]:
            tree.set_leaf_class(lid, 1)

        # Expand at parent node 2 (which has children 4, 5)
        sub_solution = OCT2Solution(
            status=SolverStatus.OPTIMAL,
            root_feature=1,
            left_feature=2,
            right_feature=3,
            leaf_classes={4: 1, 5: 2, 6: 1, 7: 2},
        )
        leaf_map = tree.extend_at_leaf(2, sub_solution, base_depth=2)

        # After extension, node 2 should have new subtree nodes
        # Global leaf IDs: leaf_pattern(4,2,2)=8, leaf_pattern(5,2,2)=9,
        #                   leaf_pattern(6,2,2)=10, leaf_pattern(7,2,2)=11
        assert 8 in leaf_map
        assert 9 in leaf_map
        # Old leaves 4, 5 should be removed
        assert 4 not in tree.leaf_nodes
        assert 5 not in tree.leaf_nodes

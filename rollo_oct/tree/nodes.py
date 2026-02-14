"""Decision tree node classes and tree structure."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from rollo_oct.tree.utils import generate_nodes, leaf_pattern, parent_pattern


@dataclass
class DecisionNode:
    """A branching (internal) node in the decision tree."""

    node_id: int
    feature_index: Optional[int] = None
    feature_vector: Optional[np.ndarray] = None

    @property
    def left_child_id(self) -> int:
        return self.node_id * 2

    @property
    def right_child_id(self) -> int:
        return self.node_id * 2 + 1

    @property
    def parent_id(self) -> Optional[int]:
        if self.node_id == 1:
            return None
        return self.node_id // 2


@dataclass
class LeafNode:
    """A leaf (terminal) node in the decision tree."""

    node_id: int
    predicted_class: Optional[int] = None
    is_pruned: bool = False

    @property
    def parent_id(self) -> int:
        return self.node_id // 2


class DecisionTree:
    """
    Complete decision tree structure.

    Replaces the dict-based model_dict['details']['var_a'] +
    model_dict['details']['target_class'] representation.

    Uses standard binary tree numbering: root=1, left child of node i
    is 2*i, right child is 2*i+1.
    """

    def __init__(self, depth: int, features: list):
        self.depth = depth
        self.features = list(features)
        self.branch_nodes: dict = {}
        self.leaf_nodes: dict = {}
        self._pruned_node_ids: set = set()
        self._initialize_structure()

    def _initialize_structure(self):
        parent_ids, leaf_ids = generate_nodes(self.depth)
        for pid in parent_ids:
            self.branch_nodes[pid] = DecisionNode(node_id=pid)
        for lid in leaf_ids:
            self.leaf_nodes[lid] = LeafNode(node_id=lid)

    def set_branch_feature(self, node_id: int, feature_index: int):
        """Set the splitting feature for a branch node."""
        if node_id not in self.branch_nodes:
            self.branch_nodes[node_id] = DecisionNode(node_id=node_id)
        node = self.branch_nodes[node_id]
        node.feature_index = feature_index
        node.feature_vector = np.array(
            [1 if f == feature_index else 0 for f in self.features]
        )

    def set_leaf_class(self, node_id: int, predicted_class):
        """Set the predicted class for a leaf node."""
        if node_id not in self.leaf_nodes:
            self.leaf_nodes[node_id] = LeafNode(node_id=node_id)
        self.leaf_nodes[node_id].predicted_class = predicted_class

    def prune_leaf(self, node_id: int):
        """Mark a leaf as pruned (it won't be expanded further)."""
        self._pruned_node_ids.add(node_id)
        if node_id in self.leaf_nodes:
            self.leaf_nodes[node_id].is_pruned = True

    def unprune_leaf(self, node_id: int):
        """Remove pruning from a leaf."""
        self._pruned_node_ids.discard(node_id)
        if node_id in self.leaf_nodes:
            self.leaf_nodes[node_id].is_pruned = False

    def predict_single(self, x: np.ndarray) -> int:
        """Route a single sample through the tree and return prediction."""
        t = 1
        d = 0
        while d < self.depth:
            node = self.branch_nodes.get(t)
            if node is None or node.feature_vector is None:
                break
            if node.feature_vector.dot(x) == 1:
                t = t * 2  # left child
            else:
                t = t * 2 + 1  # right child
            d += 1
            if t in self._pruned_node_ids:
                break
        return self.leaf_nodes[t].predicted_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for a 2D array of samples (n_samples, n_features)."""
        return np.array([self.predict_single(X[i]) for i in range(len(X))])

    def get_misclassified_leaves(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Return leaf node IDs where routed samples have mixed classes.

        Args:
            X: Feature array (n_samples, n_features).
            y: True labels (n_samples,).
        """
        leaf_classes: dict = {}
        for i in range(len(X)):
            leaf_id = self._route_to_leaf(X[i])
            if leaf_id not in leaf_classes:
                leaf_classes[leaf_id] = set()
            leaf_classes[leaf_id].add(y[i])

        return [lid for lid, classes in leaf_classes.items() if len(classes) > 1]

    def _route_to_leaf(self, x: np.ndarray) -> int:
        """Route a sample and return the leaf node ID it reaches."""
        t = 1
        d = 0
        while d < self.depth:
            node = self.branch_nodes.get(t)
            if node is None or node.feature_vector is None:
                break
            if node.feature_vector.dot(x) == 1:
                t = t * 2
            else:
                t = t * 2 + 1
            d += 1
            if t in self._pruned_node_ids:
                break
        return t

    def extend_at_leaf(self, parent_node_id: int, subtree_solution, base_depth: int = 2):
        """
        Merge a depth-2 subtree solution into this tree at the given parent node.

        The parent_node_id was previously a leaf's parent. This method:
        1. Adds new branch nodes (mapped from subtree parents to global IDs)
        2. Adds new leaf nodes (mapped from subtree leaves to global IDs)
        3. Removes the old leaf nodes that were children of parent_node_id

        Args:
            parent_node_id: The node in this tree being expanded.
            subtree_solution: An OCT2Solution with root_feature, left_feature,
                              right_feature, and leaf_classes.
            base_depth: The depth of the subtree (always 2).

        Returns:
            dict mapping new_leaf_id -> subtree_leaf_id for tracking.
        """
        sub_parent_nodes, sub_leaf_nodes = generate_nodes(base_depth)

        # Map subtree parent nodes to global IDs and set features
        feature_map = {
            1: subtree_solution.root_feature,
            2: subtree_solution.left_feature,
            3: subtree_solution.right_feature,
        }
        for sub_parent in sub_parent_nodes:
            global_id = parent_pattern(sub_parent, parent_node_id)
            self.set_branch_feature(global_id, feature_map[sub_parent])

        # Map subtree leaf nodes to global IDs and set classes
        leaf_id_map = {}
        for sub_leaf in sub_leaf_nodes:
            global_id = leaf_pattern(sub_leaf, base_depth, parent_node_id)
            leaf_id_map[global_id] = sub_leaf
            if sub_leaf in subtree_solution.leaf_classes:
                self.set_leaf_class(global_id, subtree_solution.leaf_classes[sub_leaf])

        # Remove old leaf entries for the parent's direct children
        # (they are now branch nodes, not leaves)
        for child_id in [parent_node_id * 2, parent_node_id * 2 + 1]:
            self.leaf_nodes.pop(child_id, None)
            self._pruned_node_ids.discard(child_id)

        return leaf_id_map

    def get_var_a_dict(self) -> dict:
        """Return node_id -> feature_vector list for backward compatibility."""
        return {
            nid: node.feature_vector.tolist()
            for nid, node in self.branch_nodes.items()
            if node.feature_vector is not None
        }

    def get_target_class_dict(self) -> dict:
        """Return leaf_id -> predicted_class mapping."""
        return {
            lid: leaf.predicted_class
            for lid, leaf in self.leaf_nodes.items()
            if leaf.predicted_class is not None
        }

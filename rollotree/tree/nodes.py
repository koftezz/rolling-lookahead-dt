"""Decision tree node classes and tree structure."""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np
from scipy import sparse

from rollotree.tree.utils import generate_nodes, leaf_pattern, parent_pattern
from rollotree.tree._numba import HAS_NUMBA, _predict_batch_numba


@dataclass
class DecisionNode:
    """A branching (internal) node in the decision tree."""

    node_id: int
    feature_index: Optional[int] = None
    feature_vector: Optional[np.ndarray] = None
    feature_position: Optional[int] = None

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
    class_distribution: Optional[Dict] = field(default=None, repr=False)

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
        self._routing_cache = None
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
        # Store the positional index for direct array lookup (avoids dot product)
        node.feature_position = self.features.index(feature_index)
        self._routing_cache = None

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
        self._routing_cache = None

    def unprune_leaf(self, node_id: int):
        """Remove pruning from a leaf."""
        self._pruned_node_ids.discard(node_id)
        if node_id in self.leaf_nodes:
            self.leaf_nodes[node_id].is_pruned = False
        self._routing_cache = None

    def predict_single(self, x: np.ndarray) -> int:
        """Route a single sample through the tree and return prediction."""
        leaf_id = self._route_to_leaf(x)
        return self.leaf_nodes[leaf_id].predicted_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for a 2D array of samples (n_samples, n_features)."""
        leaf_ids = self._route_all_to_leaves(X)
        # Infer dtype from every populated leaf. Using only the first class can
        # silently truncate longer string labels (for example, ``"long"`` to
        # ``"l"`` when the first class is ``"a"``).
        populated_classes = [
            leaf.predicted_class
            for leaf in self.leaf_nodes.values()
            if leaf.predicted_class is not None
        ]
        dtype = (
            np.asarray(populated_classes).dtype
            if populated_classes
            else object
        )
        predictions = np.empty(len(X), dtype=dtype)
        assigned = np.zeros(len(X), dtype=bool)
        for lid, leaf in self.leaf_nodes.items():
            mask = leaf_ids == lid
            if mask.any() and leaf.predicted_class is not None:
                predictions[mask] = leaf.predicted_class
                assigned[mask] = True
        if not assigned.all():
            missing = sorted(set(leaf_ids[~assigned].tolist()))
            raise RuntimeError(
                "Tree routed samples to leaves without class assignments: "
                f"{missing[:5]}"
            )
        return predictions

    def get_misclassified_leaves(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Return leaf node IDs where routed samples have mixed classes.

        Args:
            X: Feature array (n_samples, n_features).
            y: True labels (n_samples,).
        """
        leaf_ids = self._route_all_to_leaves(X)
        unique_leaves = np.unique(leaf_ids)
        misclassified = []
        for lid in unique_leaves:
            mask = leaf_ids == lid
            if len(np.unique(y[mask])) > 1:
                misclassified.append(int(lid))
        return misclassified

    def _route_to_leaf(self, x: np.ndarray) -> int:
        """Route a sample and return the leaf node ID it reaches."""
        t = 1
        d = 0
        while d < self.depth:
            node = self.branch_nodes.get(t)
            if node is None or node.feature_vector is None:
                break
            if x[node.feature_position] == 1:
                t = t * 2
            else:
                t = t * 2 + 1
            d += 1
            if t in self._pruned_node_ids:
                break
        return t

    def _route_all_to_leaves(self, X: np.ndarray) -> np.ndarray:
        """Route all samples through the tree using vectorized operations.

        Uses numba JIT compilation when available for ~50-100x speedup on
        large datasets. Falls back to vectorized NumPy otherwise (~10-50x
        faster than the per-sample loop).

        Returns an array of leaf node IDs, one per sample.
        """
        # Numba cannot compile object-dtype arrays. Binary numeric objects
        # such as Decimal(0)/Decimal(1) remain valid input, so preserve the
        # NumPy implementation as a correctness fallback.
        if HAS_NUMBA and X.dtype != object:
            return self._route_all_numba(X)
        return self._route_all_numpy(X)

    def _route_all_numba(self, X: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch routing."""
        cache = getattr(self, "_routing_cache", None)
        if cache is None:
            max_id = max(self.branch_nodes.keys()) if self.branch_nodes else 0
            branch_feat_pos = np.full(max_id + 1, -1, dtype=np.int64)
            for nid, node in self.branch_nodes.items():
                if node.feature_vector is not None:
                    branch_feat_pos[nid] = node.feature_position
            pruned = np.array(sorted(self._pruned_node_ids), dtype=np.int64)
            cache = (branch_feat_pos, max_id, pruned)
            self._routing_cache = cache
        branch_feat_pos, max_id, pruned = cache

        X_contiguous = np.ascontiguousarray(X)
        return _predict_batch_numba(
            X_contiguous, branch_feat_pos, max_id, pruned,
            np.empty(0, dtype=np.int64),  # leaf_ids (unused in routing)
            np.empty(0, dtype=np.int64),  # leaf_classes (unused in routing)
            self.depth,
        )

    def _route_all_numpy(self, X: np.ndarray) -> np.ndarray:
        """Vectorized NumPy batch routing (no numba required)."""
        n = X.shape[0]
        node_ids = np.ones(n, dtype=np.int64)  # all start at root

        for _d in range(self.depth):
            active_nids = set(np.unique(node_ids))
            any_routed = False
            for nid in active_nids:
                node = self.branch_nodes.get(nid)
                if node is None or node.feature_vector is None:
                    continue
                mask = node_ids == nid
                if not mask.any():
                    continue
                any_routed = True
                feat_vals = X[mask, node.feature_position]
                left = feat_vals == 1
                new_ids = np.where(left, nid * 2, nid * 2 + 1)
                node_ids[mask] = new_ids
            if not any_routed:
                break
            # Pruned nodes won't match branch_nodes.get() so routing
            # stops naturally for samples at pruned nodes.

        return node_ids

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

        self._routing_cache = None

        return leaf_id_map

    # ── Inspection utilities ─────────────────────────────────────────

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf node IDs for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with integer leaf node IDs.
        """
        return self._route_all_to_leaves(X)

    def decision_path(self, X: np.ndarray) -> sparse.csr_matrix:
        """Return a sparse matrix indicating nodes each sample traverses.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_samples, max_node_id + 1)
            Entry (i, j) is 1 if sample i passes through node j.
        """
        n = X.shape[0]
        max_node_id = max(
            max(self.branch_nodes, default=1),
            max(self.leaf_nodes, default=1),
        )

        rows, cols = [], []
        for i in range(n):
            t = 1
            d = 0
            rows.append(i)
            cols.append(t)
            while d < self.depth:
                node = self.branch_nodes.get(t)
                if node is None or node.feature_vector is None:
                    break
                if X[i, node.feature_position] == 1:
                    t = t * 2
                else:
                    t = t * 2 + 1
                d += 1
                rows.append(i)
                cols.append(t)
                if t in self._pruned_node_ids:
                    break

        data = np.ones(len(rows), dtype=np.int8)
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(n, max_node_id + 1)
        )

    def get_n_leaves(self) -> int:
        """Return the number of populated terminal leaf nodes."""
        return sum(
            1 for leaf in self.leaf_nodes.values()
            if leaf.predicted_class is not None
        )

    def get_depth(self) -> int:
        """Return the actual depth of the deepest active path."""
        max_depth = 0
        for leaf in self.leaf_nodes.values():
            if leaf.predicted_class is None:
                continue
            # bit_length of a node id gives depth: root (1) -> 1 bit -> depth 0
            depth = leaf.node_id.bit_length() - 1
            max_depth = max(max_depth, depth)
        return max_depth

    def compute_feature_importances(self, normalize: bool = True) -> np.ndarray:
        """Compute feature importances based on split frequency.

        Returns a 1-D array of length ``len(self.features)`` with the
        number of times each feature is used as a splitting variable
        across all branch nodes.  When *normalize* is True (default)
        the values are divided by their sum so they add up to 1.
        """
        counts = np.zeros(len(self.features), dtype=np.float64)
        for node in self.branch_nodes.values():
            if node.feature_index is not None:
                counts[node.feature_position] += 1
        total = counts.sum()
        if normalize and total > 0:
            counts /= total
        return counts

    def store_leaf_distributions(self, X: np.ndarray, y: np.ndarray, classes: list):
        """Compute and store per-leaf class distributions from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        classes : list of class labels (sorted)
        """
        leaf_ids = self._route_all_to_leaves(X)
        for lid, leaf in self.leaf_nodes.items():
            if leaf.predicted_class is None:
                continue
            mask = leaf_ids == lid
            y_leaf = y[mask]
            leaf.class_distribution = {
                c: int(np.sum(y_leaf == c)) for c in classes
            }

    def predict_proba(self, X: np.ndarray, classes: list) -> np.ndarray:
        """Return class probability estimates for each sample.

        Probabilities are the proportion of training samples of each
        class in the leaf that the sample is routed to.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        classes : list of class labels (sorted, same order as columns)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        leaf_ids = self._route_all_to_leaves(X)
        proba = np.zeros((X.shape[0], len(classes)), dtype=np.float64)

        for lid, leaf in self.leaf_nodes.items():
            mask = leaf_ids == lid
            if not mask.any():
                continue

            dist = leaf.class_distribution
            if dist is None:
                # Fallback: assign probability 1.0 to predicted class
                if leaf.predicted_class is None:
                    continue
                class_positions = np.flatnonzero(
                    np.asarray(classes) == leaf.predicted_class
                )
                if len(class_positions):
                    proba[mask, class_positions[0]] = 1.0
                continue

            total = sum(dist.values())
            if total == 0:
                continue
            for j, c in enumerate(classes):
                proba[mask, j] = dist.get(c, 0) / total

        return proba

    # ── Backward compatibility ────────────────────────────────────────

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

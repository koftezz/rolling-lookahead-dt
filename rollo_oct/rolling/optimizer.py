"""Rolling Subtree (RST) optimization algorithm.

Implements Algorithm 1 from the paper: starts with a depth-2 OCT,
identifies misclassified leaves, groups them by parent, re-optimizes
depth-2 subtrees at each parent, and merges results back. Repeats
level-by-level until the target depth is reached.
"""

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rollo_oct.tree.nodes import DecisionTree
from rollo_oct.tree.impurity import ImpurityCriterion
from rollo_oct.tree.utils import (
    generate_nodes,
    get_leaf_paths_depth2,
    leaf_pattern,
    parent_pattern,
    get_child,
)
from rollo_oct.solver.base import SolverConfig, SolverStatus, OCT2Solution
from rollo_oct.solver.pulp_solver import PuLPOCT2Solver

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Results for one depth level of rolling optimization."""

    depth: int
    training_accuracy: float
    test_accuracy: float
    elapsed_time: float


def _parents_of_nodes(go_deep_nodes: dict) -> dict:
    """Group nodes-to-expand by their parent node.

    Args:
        go_deep_nodes: dict mapping leaf_id -> original_subtree_leaf_id.

    Returns:
        dict mapping parent_id -> [leaf_id, ...].
    """
    parents = {}
    for node_id in go_deep_nodes:
        parent = node_id // 2
        if parent not in parents:
            parents[parent] = []
        parents[parent].append(node_id)
    return parents


def _find_misclassified_leaves(data: pd.DataFrame) -> list:
    """Find leaf IDs where the prediction column disagrees with 'y'.

    Returns leaf IDs that have more than one unique class in the routed samples.
    """
    return list(
        data.loc[
            data.groupby("leaf")["y"].transform(lambda x: x.nunique() > 1),
            "leaf",
        ].unique()
    )


class RollingOptimizer:
    """
    Implements the Rolling Subtree (RST) algorithm.

    Builds an initial depth-2 OCT, then iteratively deepens it by
    solving depth-2 subproblems on misclassified leaf subsets.
    """

    BASE_DEPTH = 2

    def __init__(self, solver_config: SolverConfig, criterion: ImpurityCriterion):
        self.solver_config = solver_config
        self.criterion = criterion
        self._oct2_solver = PuLPOCT2Solver(
            config=solver_config, criterion=criterion
        )

    def build_tree(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        features: list,
        classes: list,
        target_depth: int,
        y_idx: int = 0,
    ) -> tuple:
        """
        Build a tree of the given target depth using rolling optimization.

        Args:
            train_data: Training DataFrame with 'y' at column y_idx and features.
            test_data: Test DataFrame (same format).
            features: List of feature column indices (P).
            classes: List of unique class labels (K).
            target_depth: Maximum tree depth (must be >= 2).
            y_idx: Column index of target variable.

        Returns:
            (DecisionTree, dict[int, DepthResult])
        """
        results = {}

        # Step 1: Solve initial depth-2 tree
        t0 = time.time()
        solution = self._oct2_solver.solve(
            data=train_data, features=features, classes=classes, y_idx=y_idx
        )
        if solution.status not in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT):
            raise RuntimeError(f"Initial OCT-2 solve failed: {solution.status}")

        tree = DecisionTree(depth=self.BASE_DEPTH, features=features)
        tree.set_branch_feature(1, solution.root_feature)
        tree.set_branch_feature(2, solution.left_feature)
        tree.set_branch_feature(3, solution.right_feature)
        for leaf_id, cls in solution.leaf_classes.items():
            tree.set_leaf_class(leaf_id, cls)

        # Evaluate initial tree
        X_train = np.array(train_data[features])
        y_train = np.array(train_data.iloc[:, y_idx])
        X_test = np.array(test_data[features])
        y_test = np.array(test_data.iloc[:, y_idx])

        train_preds = tree.predict(X_train)
        test_preds = tree.predict(X_test)
        train_acc = np.mean(train_preds == y_train)
        test_acc = np.mean(test_preds == y_test)

        results[self.BASE_DEPTH] = DepthResult(
            depth=self.BASE_DEPTH,
            training_accuracy=float(train_acc),
            test_accuracy=float(test_acc),
            elapsed_time=time.time() - t0,
        )
        logger.info(
            f"Depth {self.BASE_DEPTH}: train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

        if target_depth <= self.BASE_DEPTH:
            return tree, results

        # Step 2: Identify initial misclassified leaves
        misclassified = tree.get_misclassified_leaves(X_train, y_train)

        # Step 3: Rolling expansion
        self._rolling_expand(
            tree=tree,
            train_data=train_data,
            test_data=test_data,
            features=features,
            target_depth=target_depth,
            initial_misclassified=misclassified,
            initial_solution=solution,
            y_idx=y_idx,
            results=results,
        )

        return tree, results

    def _rolling_expand(
        self,
        tree: DecisionTree,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        features: list,
        target_depth: int,
        initial_misclassified: list,
        initial_solution: OCT2Solution,
        y_idx: int,
        results: dict,
    ):
        """Core rolling expansion loop mirroring _rolling_optimize."""
        df_arr = np.array(train_data)
        leaf_nodes_path = get_leaf_paths_depth2()

        # Track selected features per expansion (maps node -> {1: feat, 2: feat, 3: feat})
        initial_feats = {
            1: initial_solution.root_feature,
            2: initial_solution.left_feature,
            3: initial_solution.right_feature,
        }
        selected_features = {1: initial_feats}

        # to_go_deep_nodes: maps current_leaf_id -> original_subtree_leaf_id
        to_go_deep_nodes = {i: i for i in initial_misclassified}

        # pruned_nodes: leaves that won't be expanded
        pruned_node_ids = set()
        for lid in leaf_nodes_path:
            if lid not in to_go_deep_nodes:
                pruned_node_ids.add(lid)
                tree.prune_leaf(lid)

        new_train_data_dict = {}

        for level in range(1, target_depth - self.BASE_DEPTH + 1):
            logger.info(
                f"Extending tree from depth {self.BASE_DEPTH} to "
                f"{self.BASE_DEPTH + level}"
            )
            iter_time = time.time()
            next_go_deep = {}

            parents_to_optimize = _parents_of_nodes(to_go_deep_nodes)

            for parent_node in parents_to_optimize:
                leaf_ids = parents_to_optimize[parent_node]

                # Look up the features selected at the grandparent level
                grandparent = parent_node // 2
                sub_features = selected_features[grandparent]

                # Extract the data subset for this parent's region
                if level == 1:
                    first_var = leaf_nodes_path[leaf_ids[0]][0]
                    root_feat_idx = sub_features[1]  # root feature
                    arr = df_arr[
                        np.where(df_arr[:, root_feat_idx] == first_var)
                    ]
                    sub_K = list(np.unique(arr[:, y_idx]))
                    cols = list(features).copy()
                    cols.insert(y_idx, "y")
                    new_train_data_dict[parent_node] = pd.DataFrame(
                        arr, columns=cols, index=None
                    )
                else:
                    parent_data = new_train_data_dict[grandparent]
                    arr = np.array(parent_data)
                    first_var = leaf_nodes_path[
                        to_go_deep_nodes[leaf_ids[0]]
                    ][0]
                    root_feat_idx = sub_features[1]
                    arr = arr[np.where(arr[:, root_feat_idx] == first_var)]
                    sub_K = list(np.unique(arr[:, y_idx]))
                    cols = list(features).copy()
                    cols.insert(y_idx, "y")
                    new_train_data_dict[parent_node] = pd.DataFrame(
                        arr, columns=cols, index=None
                    )

                logger.info(
                    f"Processing parent node {parent_node}, "
                    f"leaves {leaf_ids}, level {level}"
                )

                # Solve depth-2 subproblem for this parent's data
                sub_solution = self._oct2_solver.solve(
                    data=new_train_data_dict[parent_node],
                    features=features,
                    classes=sub_K,
                    y_idx=y_idx,
                )

                if sub_solution.status not in (
                    SolverStatus.OPTIMAL,
                    SolverStatus.TIME_LIMIT,
                ):
                    logger.warning(
                        f"Subproblem at parent {parent_node} failed: "
                        f"{sub_solution.status}"
                    )
                    continue

                # Predict on the subproblem data to find which sub-leaves
                # are still misclassified
                sub_tree = DecisionTree(
                    depth=self.BASE_DEPTH, features=features
                )
                sub_tree.set_branch_feature(1, sub_solution.root_feature)
                sub_tree.set_branch_feature(2, sub_solution.left_feature)
                sub_tree.set_branch_feature(3, sub_solution.right_feature)
                for lid, cls in sub_solution.leaf_classes.items():
                    sub_tree.set_leaf_class(lid, cls)

                sub_X = np.array(new_train_data_dict[parent_node][features])
                sub_y = np.array(
                    new_train_data_dict[parent_node].iloc[:, y_idx]
                )
                sub_misclassified = sub_tree.get_misclassified_leaves(
                    sub_X, sub_y
                )

                # Merge subtree into main tree
                _, sub_leaf_nodes = generate_nodes(self.BASE_DEPTH)
                sub_feats = {
                    1: sub_solution.root_feature,
                    2: sub_solution.left_feature,
                    3: sub_solution.right_feature,
                }

                # Map subtree parents to global IDs and set branch features
                sub_parents, _ = generate_nodes(self.BASE_DEPTH)
                for sub_p in sub_parents:
                    global_id = parent_pattern(sub_p, parent_node)
                    tree.set_branch_feature(global_id, sub_feats[sub_p])

                # Map subtree leaves to global IDs and set classes
                for sub_leaf in sub_leaf_nodes:
                    global_leaf_id = leaf_pattern(
                        sub_leaf, self.BASE_DEPTH, parent_node
                    )
                    if sub_leaf in sub_solution.leaf_classes:
                        tree.set_leaf_class(
                            global_leaf_id,
                            sub_solution.leaf_classes[sub_leaf],
                        )
                        if sub_leaf in sub_misclassified:
                            next_go_deep[global_leaf_id] = sub_leaf
                        else:
                            pruned_node_ids.add(global_leaf_id)
                            tree.prune_leaf(global_leaf_id)

                # Remove old leaf entries that are now branch nodes
                for child_id in [parent_node * 2, parent_node * 2 + 1]:
                    tree.leaf_nodes.pop(child_id, None)
                    pruned_node_ids.discard(child_id)
                    tree._pruned_node_ids.discard(child_id)

                # Fix pruned nodes: if a parent is being re-optimized,
                # its children should be unpruned (they're now internal nodes)
                # BUG FIX: Original code had `not (A or B)` which was always False.
                # The correct logic is to unprune nodes that are direct children
                # or grandchildren of parents being optimized.
                nodes_to_unprune = set()
                for pnode in parents_to_optimize:
                    nodes_to_unprune.add(pnode * 2)
                    nodes_to_unprune.add(pnode * 2 + 1)
                    grandchild = get_child(1, 2, pnode)
                    nodes_to_unprune.add(grandchild * 2)
                    nodes_to_unprune.add(grandchild * 2 + 1)
                pruned_node_ids -= nodes_to_unprune
                for nid in nodes_to_unprune:
                    tree.unprune_leaf(nid)

                # Store selected features for this expansion
                selected_features[parent_node] = sub_feats

                # Clean prediction columns from cached data
                cached = new_train_data_dict[parent_node]
                for col in ["prediction", "leaf"]:
                    if col in cached.columns:
                        new_train_data_dict[parent_node] = cached.drop(
                            col, axis=1
                        )
                        cached = new_train_data_dict[parent_node]

            # Update tree depth for this level
            current_depth = self.BASE_DEPTH + level
            tree.depth = current_depth

            # Evaluate on full train/test data
            X_train = np.array(train_data[features])
            y_train = np.array(train_data.iloc[:, y_idx])
            X_test = np.array(test_data[features])
            y_test = np.array(test_data.iloc[:, y_idx])

            train_preds = tree.predict(X_train)
            test_preds = tree.predict(X_test)
            train_acc = float(np.mean(train_preds == y_train))
            test_acc = float(np.mean(test_preds == y_test))

            logger.info(
                f"Depth {current_depth}: train_acc={train_acc:.4f}, "
                f"test_acc={test_acc:.4f}, time={time.time() - iter_time:.2f}s"
            )

            results[current_depth] = DepthResult(
                depth=current_depth,
                training_accuracy=train_acc,
                test_accuracy=test_acc,
                elapsed_time=time.time() - iter_time,
            )

            to_go_deep_nodes = next_go_deep

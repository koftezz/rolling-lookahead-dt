"""Rolling Subtree (RST) optimization algorithm.

Implements Algorithm 1 from the paper: starts with a depth-2 OCT,
identifies misclassified leaves, groups them by parent, re-optimizes
depth-2 subtrees at each parent, and merges results back. Repeats
level-by-level until the target depth is reached.
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rollotree.tree.nodes import DecisionTree
from rollotree.tree.impurity import ImpurityCriterion
from rollotree.tree.utils import (
    generate_nodes,
    get_leaf_paths_depth2,
    leaf_pattern,
    parent_pattern,
    get_child,
)
from rollotree.solver.base import SolverConfig, SolverStatus, OCT2Solution
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.rolling.parallel import (
    SubproblemInput,
    SubproblemResult,
    _solve_subproblem,
    _resolve_n_jobs,
)

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
        parents.setdefault(parent, []).append(node_id)
    return parents


class RollingOptimizer:
    """
    Implements the Rolling Subtree (RST) algorithm.

    Builds an initial depth-2 OCT, then iteratively deepens it by
    solving depth-2 subproblems on misclassified leaf subsets.
    """

    BASE_DEPTH = 2

    def __init__(
        self,
        solver_config: SolverConfig,
        criterion: ImpurityCriterion,
        n_jobs: int = 1,
    ):
        self.solver_config = solver_config
        self.criterion = criterion
        self.n_jobs = n_jobs
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

        # Prune leaves that won't be expanded
        pruned_node_ids = {
            lid for lid in leaf_nodes_path if lid not in to_go_deep_nodes
        }
        for lid in pruned_node_ids:
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

            # ── Phase A: build subproblem inputs (sequential) ──
            inputs = []
            for parent_node in parents_to_optimize:
                leaf_ids = parents_to_optimize[parent_node]

                # Look up the features selected at the grandparent level
                grandparent = parent_node // 2
                sub_features = selected_features[grandparent]

                # Extract the data subset for this parent's region
                if level == 1:
                    source_arr = df_arr
                    leaf_key = leaf_ids[0]
                else:
                    source_arr = np.array(new_train_data_dict[grandparent])
                    leaf_key = to_go_deep_nodes[leaf_ids[0]]

                first_var = leaf_nodes_path[leaf_key][0]
                root_feat_idx = sub_features[1]
                arr = source_arr[
                    np.where(source_arr[:, root_feat_idx] == first_var)
                ]
                sub_K = list(np.unique(arr[:, y_idx]))
                cols = list(features).copy()
                cols.insert(y_idx, "y")
                parent_data = pd.DataFrame(arr, columns=cols, index=None)
                new_train_data_dict[parent_node] = parent_data

                logger.info(
                    f"Processing parent node {parent_node}, "
                    f"leaves {leaf_ids}, level {level}, "
                    f"samples={len(parent_data)}"
                )

                inputs.append(SubproblemInput(
                    parent_node=parent_node,
                    leaf_ids=leaf_ids,
                    parent_data=parent_data,
                    features=features,
                    sub_K=sub_K,
                    y_idx=y_idx,
                    solver_config=self.solver_config,
                    criterion=self.criterion,
                ))

            # ── Phase B: solve subproblems (parallel or sequential) ──
            effective_jobs = _resolve_n_jobs(self.n_jobs)
            if effective_jobs > 1 and len(inputs) > 1:
                with ProcessPoolExecutor(max_workers=effective_jobs) as pool:
                    results_list = list(pool.map(_solve_subproblem, inputs))
            else:
                results_list = [_solve_subproblem(inp) for inp in inputs]

            # ── Phase C: merge results into the tree (sequential) ──
            for result in results_list:
                parent_node = result.parent_node
                leaf_ids = result.leaf_ids

                if result.skipped:
                    logger.info(
                        f"Skipping parent {parent_node}: "
                        f"{result.n_samples} samples "
                        f"< min_samples_split="
                        f"{self.solver_config.min_samples_split}"
                    )
                    for lid in leaf_ids:
                        pruned_node_ids.add(lid)
                        tree.prune_leaf(lid)
                    continue

                sub_solution = result.sub_solution
                if sub_solution.status not in (
                    SolverStatus.OPTIMAL,
                    SolverStatus.TIME_LIMIT,
                ):
                    logger.warning(
                        f"Subproblem at parent {parent_node} failed: "
                        f"{sub_solution.status} — pruning leaves"
                    )
                    for lid in leaf_ids:
                        pruned_node_ids.add(lid)
                        tree.prune_leaf(lid)
                    continue

                # Merge subtree into main tree
                sub_parents, sub_leaf_nodes = generate_nodes(self.BASE_DEPTH)
                sub_feats = {
                    1: sub_solution.root_feature,
                    2: sub_solution.left_feature,
                    3: sub_solution.right_feature,
                }

                for sub_p in sub_parents:
                    global_id = parent_pattern(sub_p, parent_node)
                    tree.set_branch_feature(global_id, sub_feats[sub_p])

                for sub_leaf in sub_leaf_nodes:
                    global_leaf_id = leaf_pattern(
                        sub_leaf, self.BASE_DEPTH, parent_node
                    )
                    if sub_leaf in sub_solution.leaf_classes:
                        tree.set_leaf_class(
                            global_leaf_id,
                            sub_solution.leaf_classes[sub_leaf],
                        )
                        if sub_leaf in result.sub_misclassified:
                            next_go_deep[global_leaf_id] = sub_leaf
                        else:
                            pruned_node_ids.add(global_leaf_id)
                            tree.prune_leaf(global_leaf_id)

                for child_id in [parent_node * 2, parent_node * 2 + 1]:
                    tree.leaf_nodes.pop(child_id, None)
                    pruned_node_ids.discard(child_id)
                    tree._pruned_node_ids.discard(child_id)

                selected_features[parent_node] = sub_feats

                # Clean prediction columns from cached data
                parent_data = result.parent_data
                cols_to_drop = [
                    col for col in ["prediction", "leaf"]
                    if col in parent_data.columns
                ]
                if cols_to_drop:
                    new_train_data_dict[parent_node] = parent_data.drop(
                        columns=cols_to_drop
                    )

            # Unprune children and grandchildren of parents being
            # re-optimized (once per level, after all merges).
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
            if not to_go_deep_nodes:
                logger.info("No misclassified leaves remain. Stopping early.")
                break

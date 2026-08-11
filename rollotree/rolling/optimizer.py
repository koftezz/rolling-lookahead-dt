"""Rolling subtree optimization and fit diagnostics."""

from __future__ import annotations

import copy
import logging
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from rollotree.rolling.parallel import (
    SubproblemInput,
    _resolve_n_jobs,
    _solve_subproblem,
)
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import ImpurityCriterion
from rollotree.tree.nodes import DecisionTree

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Results for one accepted tree depth."""

    depth: int
    training_accuracy: float
    test_accuracy: float
    elapsed_time: float


@dataclass(frozen=True)
class SubproblemDiagnostic:
    """Stable, serializable summary of one optimization subproblem."""

    depth: int
    parent_node: int
    n_samples: int
    n_features: int
    status: str
    candidate_feature: Optional[int] = None
    objective_value: Optional[float] = None
    solver_time: Optional[float] = None
    elapsed_time: Optional[float] = None
    used_incumbent: bool = False
    skip_reason: Optional[str] = None


def _parents_of_nodes(node_ids: List[int]) -> Dict[int, List[int]]:
    parents: Dict[int, List[int]] = {}
    for node_id in node_ids:
        parents.setdefault(node_id // 2, []).append(node_id)
    return parents


def _descends_from(node_id: int, ancestor_id: int) -> bool:
    while node_id > ancestor_id:
        node_id //= 2
    return node_id == ancestor_id


def _resolve_max_features(max_features, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            count = int(math.sqrt(n_features))
        elif max_features == "log2":
            count = int(math.log2(n_features))
        else:  # validation normally catches this first
            raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")
        return max(1, count)
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    return min(int(max_features), n_features)


class RollingOptimizer:
    """Build an initial exact tree and deepen it with OCT-2 subproblems."""

    def __init__(
        self,
        solver_config: SolverConfig,
        criterion: ImpurityCriterion,
        n_jobs: int = 1,
        max_features=None,
        random_state: Optional[int] = None,
        total_time_limit: Optional[float] = None,
        initial_depth: int = 2,
    ):
        self.solver_config = solver_config
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.random_state = random_state
        self.total_time_limit = total_time_limit
        self.initial_depth = initial_depth

        self.fit_status_ = "completed"
        self.fit_time_ = 0.0
        self.actual_depth_ = 0
        self.subproblem_diagnostics_: List[SubproblemDiagnostic] = []
        self._started_at = 0.0
        self._deadline = None
        self._base_seed = None

    def _remaining_time(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return self._deadline - time.perf_counter()

    def _config_for_solve(self) -> Optional[SolverConfig]:
        remaining = self._remaining_time()
        if remaining is not None and remaining <= 0:
            return None
        limit = self.solver_config.time_limit
        if remaining is not None:
            limit = min(limit, max(0.001, remaining))
        return self.solver_config.copy_with(time_limit=limit)

    def _select_features(
        self, features: list, depth: int, parent_node: int
    ) -> list:
        count = _resolve_max_features(self.max_features, len(features))
        if count >= len(features):
            return list(features)
        seed = np.random.SeedSequence(
            [int(self._base_seed), int(depth), int(parent_node)]
        )
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(features), size=count, replace=False))
        return [features[position] for position in positions]

    @staticmethod
    def _evaluate(tree, X_train, y_train, X_test, y_test):
        train_accuracy = float(np.mean(tree.predict(X_train) == y_train))
        test_accuracy = float(np.mean(tree.predict(X_test) == y_test))
        return train_accuracy, test_accuracy

    def _warn_timeout(self):
        warnings.warn(
            "Optimization stopped at a time limit; returning the last valid tree.",
            ConvergenceWarning,
            stacklevel=3,
        )

    def _build_initial_depth2(
        self, train_data, features, classes, y_idx
    ) -> Tuple[DecisionTree, SolverStatus]:
        candidate_features = self._select_features(features, 2, 1)
        config = self._config_for_solve()
        if config is None:
            raise TimeoutError("total_time_limit expired before the initial solve")

        started = time.perf_counter()
        solution = PuLPOCT2Solver(config, self.criterion).solve(
            data=train_data,
            features=candidate_features,
            classes=classes,
            y_idx=y_idx,
        )
        elapsed = time.perf_counter() - started
        self.subproblem_diagnostics_.append(
            SubproblemDiagnostic(
                depth=2,
                parent_node=1,
                n_samples=len(train_data),
                n_features=len(candidate_features),
                status=solution.status.value,
                objective_value=solution.objective_value,
                solver_time=solution.runtime,
                elapsed_time=elapsed,
                used_incumbent=solution.status == SolverStatus.TIME_LIMIT,
            )
        )
        if solution.status not in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT):
            if solution.status == SolverStatus.INFEASIBLE:
                raise ValueError(
                    "No feasible depth-2 tree satisfies min_samples_leaf "
                    "with the selected features. Reduce min_samples_leaf or "
                    "increase max_features."
                )
            raise RuntimeError(f"Initial OCT-2 solve failed: {solution.status}")

        tree = DecisionTree(depth=2, features=features)
        tree.set_branch_feature(1, solution.root_feature)
        tree.set_branch_feature(2, solution.left_feature)
        tree.set_branch_feature(3, solution.right_feature)
        for leaf_id, class_label in solution.leaf_classes.items():
            tree.set_leaf_class(leaf_id, class_label)
        return tree, solution.status

    def _build_initial_tree(self, train_data, features, classes, y_idx):
        if self.initial_depth == 2:
            return self._build_initial_depth2(
                train_data, features, classes, y_idx
            )

        from rollotree.solver.depth3 import ExactDepth3Solver

        candidate_features = self._select_features(features, 3, 1)
        config = self._config_for_solve()
        if config is None:
            raise TimeoutError("total_time_limit expired before the initial solve")
        solver = ExactDepth3Solver(
            config=config,
            criterion=self.criterion,
            n_jobs=self.n_jobs,
            deadline=self._deadline,
        )
        solution = solver.solve(
            data=train_data,
            features=features,
            classes=classes,
            y_idx=y_idx,
            feature_subset=candidate_features,
        )
        for diagnostic in solution.candidate_diagnostics:
            self.subproblem_diagnostics_.append(
                SubproblemDiagnostic(
                    depth=3,
                    parent_node=1,
                    n_samples=(
                        diagnostic.left_samples + diagnostic.right_samples
                    ),
                    n_features=len(candidate_features),
                    status=diagnostic.status.value,
                    candidate_feature=diagnostic.root_feature,
                    objective_value=diagnostic.objective_value,
                    solver_time=diagnostic.runtime,
                    elapsed_time=diagnostic.runtime,
                    used_incumbent=(
                        diagnostic.complete
                        and diagnostic.status == SolverStatus.TIME_LIMIT
                    ),
                    skip_reason=diagnostic.reason,
                )
            )
        if solution.status not in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT):
            if solution.status == SolverStatus.INFEASIBLE:
                raise ValueError(
                    "No feasible exact depth-3 tree satisfies min_samples_leaf "
                    "with the selected features."
                )
            raise RuntimeError(f"Initial OCT-3 solve failed: {solution.status}")
        if solution.n_complete_candidates == 0:
            raise TimeoutError(
                "The time limit expired before an exact depth-3 incumbent "
                "was available. Increase total_time_limit or time_limit."
            )

        tree = DecisionTree(depth=3, features=features)
        for node_id, feature in solution.branch_features.items():
            tree.set_branch_feature(node_id, feature)
        for leaf_id, class_label in solution.leaf_classes.items():
            tree.set_leaf_class(leaf_id, class_label)
        return tree, solution.status

    def build_tree(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        features: list,
        classes: list,
        target_depth: int,
        y_idx: int = 0,
    ) -> Tuple[DecisionTree, Dict[int, DepthResult]]:
        """Build a tree and return accepted per-depth results."""
        self.fit_status_ = "completed"
        self.subproblem_diagnostics_ = []
        self._started_at = time.perf_counter()
        self._deadline = (
            None
            if self.total_time_limit is None
            else self._started_at + self.total_time_limit
        )
        if self.random_state is None:
            self._base_seed = np.random.SeedSequence().generate_state(1)[0]
        else:
            self._base_seed = int(self.random_state)

        X_train = np.asarray(train_data[features])
        y_train = np.asarray(train_data.iloc[:, y_idx])
        X_test = np.asarray(test_data[features])
        y_test = np.asarray(test_data.iloc[:, y_idx])

        initial_started = time.perf_counter()
        tree, initial_status = self._build_initial_tree(
            train_data, features, classes, y_idx
        )
        train_accuracy, test_accuracy = self._evaluate(
            tree, X_train, y_train, X_test, y_test
        )
        results = {
            self.initial_depth: DepthResult(
                depth=self.initial_depth,
                training_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                elapsed_time=time.perf_counter() - initial_started,
            )
        }

        if initial_status == SolverStatus.TIME_LIMIT:
            self.fit_status_ = "time_limit"
            self._warn_timeout()
        elif target_depth > self.initial_depth:
            tree = self._rolling_expand(
                tree,
                train_data,
                features,
                target_depth,
                y_idx,
                X_train,
                y_train,
                X_test,
                y_test,
                results,
            )

        self.actual_depth_ = tree.get_depth()
        self.fit_time_ = time.perf_counter() - self._started_at
        return tree, results

    def _rolling_expand(
        self,
        tree,
        train_data,
        features,
        target_depth,
        y_idx,
        X_train,
        y_train,
        X_test,
        y_test,
        results,
    ):
        blocked_leaves: set[int] = set()
        previous_accuracy = results[self.initial_depth].training_accuracy

        while True:
            routed_leaf_ids = tree.apply(X_train)
            mixed_leaves = tree.get_misclassified_leaves(X_train, y_train)
            eligible = [
                leaf_id
                for leaf_id in mixed_leaves
                if leaf_id not in blocked_leaves
                and (leaf_id // 2).bit_length() - 1 + 2 <= target_depth
            ]
            if not eligible:
                if tree.get_depth() < target_depth:
                    self.fit_status_ = "early_stopped"
                break

            if self._config_for_solve() is None:
                self.fit_status_ = "time_limit"
                self._warn_timeout()
                break

            parents = _parents_of_nodes(eligible)
            inputs = []
            for parent_node, leaf_ids in sorted(parents.items()):
                mask = np.fromiter(
                    (
                        _descends_from(int(leaf_id), parent_node)
                        for leaf_id in routed_leaf_ids
                    ),
                    dtype=bool,
                    count=len(routed_leaf_ids),
                )
                parent_data = train_data.loc[mask].reset_index(drop=True)
                depth = parent_node.bit_length() - 1 + 2
                candidate_features = self._select_features(
                    features, depth, parent_node
                )
                config = self._config_for_solve()
                if config is None:
                    self.fit_status_ = "time_limit"
                    break
                inputs.append(
                    SubproblemInput(
                        parent_node=parent_node,
                        leaf_ids=leaf_ids,
                        parent_data=parent_data,
                        features=candidate_features,
                        sub_K=np.unique(
                            np.asarray(parent_data.iloc[:, y_idx])
                        ).tolist(),
                        y_idx=y_idx,
                        solver_config=config,
                        criterion=self.criterion,
                        deadline=self._deadline,
                    )
                )
            if self.fit_status_ == "time_limit":
                self._warn_timeout()
                break

            level_started = time.perf_counter()
            results_list = self._solve_inputs(inputs)
            snapshot = copy.deepcopy(tree)
            merged = 0
            saw_timeout = False

            input_by_parent = {item.parent_node: item for item in inputs}
            for result in results_list:
                inp = input_by_parent[result.parent_node]
                depth = result.parent_node.bit_length() - 1 + 2
                solution = result.sub_solution
                if result.timed_out:
                    status = SolverStatus.TIME_LIMIT.value
                    skip_reason = "total_time_limit"
                    blocked_leaves.update(result.leaf_ids)
                    saw_timeout = True
                elif result.skipped:
                    status = "skipped_min_samples"
                    skip_reason = "min_samples_split"
                    blocked_leaves.update(result.leaf_ids)
                elif solution.status not in (
                    SolverStatus.OPTIMAL,
                    SolverStatus.TIME_LIMIT,
                ):
                    status = solution.status.value
                    skip_reason = "solver_failure"
                    blocked_leaves.update(result.leaf_ids)
                else:
                    status = solution.status.value
                    skip_reason = None
                    tree.extend_at_leaf(
                        result.parent_node, solution, base_depth=2
                    )
                    merged += 1
                    saw_timeout |= solution.status == SolverStatus.TIME_LIMIT

                self.subproblem_diagnostics_.append(
                    SubproblemDiagnostic(
                        depth=depth,
                        parent_node=result.parent_node,
                        n_samples=result.n_samples,
                        n_features=len(inp.features),
                        status=status,
                        objective_value=(
                            None if solution is None else solution.objective_value
                        ),
                        solver_time=(
                            None if solution is None else solution.runtime
                        ),
                        elapsed_time=result.elapsed_time,
                        used_incumbent=(
                            solution is not None
                            and solution.status == SolverStatus.TIME_LIMIT
                        ),
                        skip_reason=skip_reason,
                    )
                )

            if merged == 0:
                tree = snapshot
                remaining = self._remaining_time()
                if saw_timeout or (remaining is not None and remaining <= 0):
                    self.fit_status_ = "time_limit"
                    self._warn_timeout()
                else:
                    self.fit_status_ = "early_stopped"
                break

            tree.depth = max(
                tree.depth,
                max(
                    leaf.node_id.bit_length() - 1
                    for leaf in tree.leaf_nodes.values()
                    if leaf.predicted_class is not None
                ),
            )
            train_accuracy, test_accuracy = self._evaluate(
                tree, X_train, y_train, X_test, y_test
            )
            if train_accuracy + 1e-10 < previous_accuracy:
                tree = snapshot
                self.fit_status_ = "early_stopped"
                self.subproblem_diagnostics_.append(
                    SubproblemDiagnostic(
                        depth=tree.get_depth(),
                        parent_node=0,
                        n_samples=len(train_data),
                        n_features=len(features),
                        status="rejected",
                        skip_reason="training_accuracy_regression",
                    )
                )
                break

            actual_depth = tree.get_depth()
            elapsed = time.perf_counter() - level_started
            if actual_depth in results:
                previous = results[actual_depth]
                elapsed += previous.elapsed_time
            results[actual_depth] = DepthResult(
                depth=actual_depth,
                training_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                elapsed_time=elapsed,
            )
            previous_accuracy = train_accuracy

            remaining = self._remaining_time()
            if saw_timeout or (remaining is not None and remaining <= 0):
                self.fit_status_ = "time_limit"
                self._warn_timeout()
                break

        return tree

    def _solve_inputs(self, inputs):
        effective_jobs = min(_resolve_n_jobs(self.n_jobs), len(inputs))
        if effective_jobs <= 1:
            return [_solve_subproblem(item) for item in inputs]

        try:
            with ProcessPoolExecutor(max_workers=effective_jobs) as pool:
                return list(pool.map(_solve_subproblem, inputs))
        except (PermissionError, NotImplementedError, OSError) as exc:
            warnings.warn(
                "Parallel solving is unavailable on this platform; falling "
                f"back to sequential execution ({exc}).",
                RuntimeWarning,
                stacklevel=3,
            )
            return [_solve_subproblem(item) for item in inputs]

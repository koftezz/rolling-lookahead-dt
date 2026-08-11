"""Exact depth-3 classification trees built from OCT-2 subproblems.

Fixing the feature at the root of a depth-3 tree separates the remaining
optimization into two independent depth-2 trees.  This module enumerates the
candidate root features and delegates those two subtrees to the existing
``PuLPOCT2Solver``.  It therefore preserves the current solver backends and
both supported additive leaf criteria without constructing an impractical
``O(p**3)`` path-variable model.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import os
import time
from typing import Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import (
    GiniCriterion,
    ImpurityCriterion,
    MisclassificationCriterion,
)
from rollotree.tree.utils import leaf_pattern, parent_pattern


@dataclass(frozen=True)
class OCT3CandidateDiagnostic:
    """Outcome of evaluating one candidate feature at the top root."""

    root_feature: int
    status: SolverStatus
    left_status: Optional[SolverStatus]
    right_status: Optional[SolverStatus]
    left_samples: int
    right_samples: int
    objective_value: Optional[float]
    runtime: float
    complete: bool
    reason: Optional[str] = None


@dataclass
class OCT3Solution:
    """Result of an exact depth-3 root-enumeration solve.

    ``branch_features`` uses standard binary-tree node IDs 1 through 7 and
    ``leaf_classes`` uses leaf IDs 8 through 15, making the result directly
    consumable by :class:`rollotree.tree.nodes.DecisionTree`.
    """

    status: SolverStatus
    branch_features: dict = field(default_factory=dict)
    leaf_classes: dict = field(default_factory=dict)
    objective_value: Optional[float] = None
    runtime: Optional[float] = None
    mip_gap: Optional[float] = None
    candidate_diagnostics: Tuple[OCT3CandidateDiagnostic, ...] = field(
        default_factory=tuple
    )
    features: Tuple[int, ...] = field(default_factory=tuple)
    n_candidates: int = 0
    n_complete_candidates: int = 0


@dataclass(frozen=True)
class _CandidateInput:
    root_feature: int
    data: pd.DataFrame
    features: Tuple[int, ...]
    classes: Tuple
    y_idx: int
    config: SolverConfig
    criterion: ImpurityCriterion
    deadline: Optional[float] = None


@dataclass
class _CandidateResult:
    diagnostic: OCT3CandidateDiagnostic
    branch_features: dict = field(default_factory=dict)
    leaf_classes: dict = field(default_factory=dict)


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)


def _complete_oct2_is_feasible(
    data_arr: np.ndarray,
    features: Sequence[int],
    min_samples_leaf: int,
) -> bool:
    """Return whether any strict complete depth-2 tree is feasible.

    This precheck deliberately bypasses the OCT-2 solver's legacy fallback to
    all feature pairs.  A root is feasible only if the same split feature has
    a valid pair on both of its child branches.
    """

    if len(data_arr) < 4 * min_samples_leaf:
        return False
    valid_left, valid_right = PuLPOCT2Solver._compute_valid_pairs(
        data_arr, list(features), min_samples_leaf
    )
    left_roots = {root for root, _child in valid_left}
    right_roots = {root for root, _child in valid_right}
    return bool(left_roots & right_roots)


def _oct2_has_complete_incumbent(solution) -> bool:
    return (
        solution.status in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT)
        and solution.root_feature is not None
        and solution.left_feature is not None
        and solution.right_feature is not None
        and solution.objective_value is not None
        and set(solution.leaf_classes) == {4, 5, 6, 7}
    )


def _actual_leaf_sizes(data_arr: np.ndarray, solution) -> Tuple[int, int, int, int]:
    root_values = data_arr[:, solution.root_feature]
    left_values = data_arr[:, solution.left_feature]
    right_values = data_arr[:, solution.right_feature]
    return (
        int(np.sum((root_values == 1) & (left_values == 1))),
        int(np.sum((root_values == 1) & (left_values == 0))),
        int(np.sum((root_values == 0) & (right_values == 1))),
        int(np.sum((root_values == 0) & (right_values == 0))),
    )


def _map_subtree(solution, global_root: int) -> Tuple[dict, dict]:
    local_features = {
        1: solution.root_feature,
        2: solution.left_feature,
        3: solution.right_feature,
    }
    branch_features = {
        parent_pattern(local_node, global_root): feature
        for local_node, feature in local_features.items()
    }
    leaf_classes = {
        leaf_pattern(local_leaf, 2, global_root): predicted_class
        for local_leaf, predicted_class in solution.leaf_classes.items()
    }
    return branch_features, leaf_classes


def _candidate_failure_status(left_status, right_status) -> SolverStatus:
    statuses = {left_status, right_status}
    if SolverStatus.ERROR in statuses or SolverStatus.UNBOUNDED in statuses:
        return SolverStatus.ERROR
    if SolverStatus.TIME_LIMIT in statuses:
        return SolverStatus.TIME_LIMIT
    return SolverStatus.INFEASIBLE


def _solve_candidate(inp: _CandidateInput) -> _CandidateResult:
    """Evaluate one top-root feature; safe to dispatch to a worker process."""

    started = time.perf_counter()
    data_arr = np.asarray(inp.data)
    root_values = data_arr[:, inp.root_feature]
    left_arr = data_arr[root_values == 1]
    right_arr = data_arr[root_values == 0]
    left_n = len(left_arr)
    right_n = len(right_arr)

    def config_with_remaining_time() -> Optional[SolverConfig]:
        if inp.deadline is None:
            return inp.config
        remaining = inp.deadline - time.time()
        if remaining <= 0:
            return None
        return inp.config.copy_with(
            time_limit=min(inp.config.time_limit, max(0.001, remaining))
        )

    if config_with_remaining_time() is None:
        diagnostic = OCT3CandidateDiagnostic(
            root_feature=inp.root_feature,
            status=SolverStatus.TIME_LIMIT,
            left_status=None,
            right_status=None,
            left_samples=left_n,
            right_samples=right_n,
            objective_value=None,
            runtime=time.perf_counter() - started,
            complete=False,
            reason="global time limit expired before candidate solve",
        )
        return _CandidateResult(diagnostic=diagnostic)

    left_feasible = _complete_oct2_is_feasible(
        left_arr, inp.features, inp.config.min_samples_leaf
    )
    right_feasible = _complete_oct2_is_feasible(
        right_arr, inp.features, inp.config.min_samples_leaf
    )
    if not left_feasible or not right_feasible:
        sides = []
        if not left_feasible:
            sides.append("value-1")
        if not right_feasible:
            sides.append("value-0")
        diagnostic = OCT3CandidateDiagnostic(
            root_feature=inp.root_feature,
            status=SolverStatus.INFEASIBLE,
            left_status=(SolverStatus.INFEASIBLE if not left_feasible else None),
            right_status=(SolverStatus.INFEASIBLE if not right_feasible else None),
            left_samples=left_n,
            right_samples=right_n,
            objective_value=None,
            runtime=time.perf_counter() - started,
            complete=False,
            reason=f"no strict complete OCT-2 subtree on {', '.join(sides)} branch",
        )
        return _CandidateResult(diagnostic=diagnostic)

    columns = list(inp.data.columns)
    left_data = pd.DataFrame(left_arr, columns=columns)
    right_data = pd.DataFrame(right_arr, columns=columns)
    # Deliberately sequential inside a candidate.  Candidate roots are the
    # parallelism boundary, avoiding nested process pools and solver storms.
    left_config = config_with_remaining_time()
    if left_config is None:
        diagnostic = OCT3CandidateDiagnostic(
            root_feature=inp.root_feature,
            status=SolverStatus.TIME_LIMIT,
            left_status=None,
            right_status=None,
            left_samples=left_n,
            right_samples=right_n,
            objective_value=None,
            runtime=time.perf_counter() - started,
            complete=False,
            reason="global time limit expired before left OCT-2 solve",
        )
        return _CandidateResult(diagnostic=diagnostic)
    left_solution = PuLPOCT2Solver(
        config=left_config, criterion=inp.criterion
    ).solve(
        data=left_data,
        features=list(inp.features),
        classes=list(inp.classes),
        y_idx=inp.y_idx,
    )
    right_config = config_with_remaining_time()
    if right_config is None:
        diagnostic = OCT3CandidateDiagnostic(
            root_feature=inp.root_feature,
            status=SolverStatus.TIME_LIMIT,
            left_status=left_solution.status,
            right_status=None,
            left_samples=left_n,
            right_samples=right_n,
            objective_value=None,
            runtime=time.perf_counter() - started,
            complete=False,
            reason="global time limit expired before right OCT-2 solve",
        )
        return _CandidateResult(diagnostic=diagnostic)
    right_solution = PuLPOCT2Solver(
        config=right_config, criterion=inp.criterion
    ).solve(
        data=right_data,
        features=list(inp.features),
        classes=list(inp.classes),
        y_idx=inp.y_idx,
    )

    left_complete = _oct2_has_complete_incumbent(left_solution)
    right_complete = _oct2_has_complete_incumbent(right_solution)
    if left_complete:
        left_complete = (
            min(_actual_leaf_sizes(left_arr, left_solution))
            >= inp.config.min_samples_leaf
        )
    if right_complete:
        right_complete = (
            min(_actual_leaf_sizes(right_arr, right_solution))
            >= inp.config.min_samples_leaf
        )

    if not left_complete or not right_complete:
        status = _candidate_failure_status(
            left_solution.status, right_solution.status
        )
        diagnostic = OCT3CandidateDiagnostic(
            root_feature=inp.root_feature,
            status=status,
            left_status=left_solution.status,
            right_status=right_solution.status,
            left_samples=left_n,
            right_samples=right_n,
            objective_value=None,
            runtime=time.perf_counter() - started,
            complete=False,
            reason="one or both OCT-2 solves returned no strict complete incumbent",
        )
        return _CandidateResult(diagnostic=diagnostic)

    total_n = left_n + right_n
    objective = inp.criterion.combine_subproblem_objective(
        float(left_solution.objective_value), left_n, total_n
    ) + inp.criterion.combine_subproblem_objective(
        float(right_solution.objective_value), right_n, total_n
    )

    left_features, left_classes = _map_subtree(left_solution, global_root=2)
    right_features, right_classes = _map_subtree(right_solution, global_root=3)
    branch_features = {1: inp.root_feature, **left_features, **right_features}
    leaf_classes = {**left_classes, **right_classes}
    status = (
        SolverStatus.OPTIMAL
        if left_solution.status == right_solution.status == SolverStatus.OPTIMAL
        else SolverStatus.TIME_LIMIT
    )
    diagnostic = OCT3CandidateDiagnostic(
        root_feature=inp.root_feature,
        status=status,
        left_status=left_solution.status,
        right_status=right_solution.status,
        left_samples=left_n,
        right_samples=right_n,
        objective_value=objective,
        runtime=time.perf_counter() - started,
        complete=True,
    )
    return _CandidateResult(
        diagnostic=diagnostic,
        branch_features=branch_features,
        leaf_classes=leaf_classes,
    )


class ExactDepth3Solver:
    """Find a globally optimal complete depth-3 tree over binary features.

    Exactness is certified only when every candidate top-root feature is
    either proven infeasible or has two optimal OCT-2 subsolutions.  A
    complete incumbent may still be returned with ``TIME_LIMIT`` when an
    underlying backend exposes one.
    """

    def __init__(
        self,
        config: SolverConfig,
        criterion: ImpurityCriterion,
        n_jobs: int = 1,
        deadline: Optional[float] = None,
    ):
        if not isinstance(
            criterion, (GiniCriterion, MisclassificationCriterion)
        ):
            raise TypeError(
                "ExactDepth3Solver supports GiniCriterion and "
                "MisclassificationCriterion"
            )
        if config.mip_gap not in (None, 0, 0.0):
            raise ValueError(
                "ExactDepth3Solver requires mip_gap to be None or 0; a "
                "positive MIP gap cannot certify a globally optimal tree"
            )
        # HiGHS and Gurobi both have nonzero backend defaults. Request a zero
        # gap explicitly so OPTIMAL supports the public exactness guarantee.
        self.config = config.copy_with(mip_gap=0.0)
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.deadline = deadline

    def solve(
        self,
        data: pd.DataFrame,
        features: Sequence[int],
        classes: Sequence,
        y_idx: int = 0,
        feature_subset: Optional[Sequence[int]] = None,
    ) -> OCT3Solution:
        """Solve an exact depth-3 tree over ``feature_subset``.

        Args:
            data: Internal training frame with target at ``y_idx``.
            features: All available feature-column indices.
            classes: Class labels used by the impurity criterion.
            y_idx: Positional index of the target column.
            feature_subset: Fixed candidate set for this solve.  When omitted,
                every feature is used.  The same set is used at every node.
        """

        started = time.perf_counter()
        all_features = tuple(features)
        selected = tuple(all_features if feature_subset is None else feature_subset)
        if not selected:
            raise ValueError("feature_subset must contain at least one feature")
        if len(set(selected)) != len(selected):
            raise ValueError("feature_subset must not contain duplicates")
        unknown = [feature for feature in selected if feature not in all_features]
        if unknown:
            raise ValueError(f"feature_subset contains unknown features: {unknown}")

        inputs = [
            _CandidateInput(
                root_feature=root_feature,
                data=data,
                features=selected,
                classes=tuple(classes),
                y_idx=y_idx,
                config=self.config,
                criterion=self.criterion,
                deadline=self.deadline,
            )
            for root_feature in selected
        ]
        workers = min(_resolve_n_jobs(self.n_jobs), len(inputs))
        if workers > 1 and len(inputs) > 1:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    candidate_results = list(pool.map(_solve_candidate, inputs))
            except (OSError, PermissionError, NotImplementedError) as exc:
                warnings.warn(
                    "Depth-3 multiprocessing is unavailable; falling back "
                    f"to sequential candidate solves ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                candidate_results = [_solve_candidate(inp) for inp in inputs]
        else:
            candidate_results = [_solve_candidate(inp) for inp in inputs]

        # map() preserves input order; sorting makes that contract explicit and
        # stabilizes diagnostics if the execution strategy changes later.
        feature_order = {feature: index for index, feature in enumerate(selected)}
        candidate_results.sort(
            key=lambda result: feature_order[result.diagnostic.root_feature]
        )
        diagnostics = tuple(result.diagnostic for result in candidate_results)
        complete = [
            result for result in candidate_results
            if result.diagnostic.complete
        ]

        if not complete:
            if any(d.status == SolverStatus.ERROR for d in diagnostics):
                status = SolverStatus.ERROR
            elif any(d.status == SolverStatus.TIME_LIMIT for d in diagnostics):
                status = SolverStatus.TIME_LIMIT
            else:
                status = SolverStatus.INFEASIBLE
            return OCT3Solution(
                status=status,
                runtime=time.perf_counter() - started,
                candidate_diagnostics=diagnostics,
                features=selected,
                n_candidates=len(inputs),
                n_complete_candidates=0,
            )

        best_objective = min(
            result.diagnostic.objective_value for result in complete
        )
        tolerance = 1e-12 * max(1.0, abs(best_objective))
        tied = [
            result
            for result in complete
            if abs(result.diagnostic.objective_value - best_objective) <= tolerance
        ]

        def deterministic_signature(result):
            return tuple(
                feature_order[result.branch_features[node_id]]
                for node_id in range(1, 8)
            )

        best = min(tied, key=deterministic_signature)
        has_error = any(
            d.status in (SolverStatus.ERROR, SolverStatus.UNBOUNDED)
            for d in diagnostics
        )
        has_timeout = any(d.status == SolverStatus.TIME_LIMIT for d in diagnostics)
        if has_error:
            status = SolverStatus.ERROR
        elif has_timeout:
            status = SolverStatus.TIME_LIMIT
        else:
            status = SolverStatus.OPTIMAL

        return OCT3Solution(
            status=status,
            branch_features=dict(best.branch_features),
            leaf_classes=dict(best.leaf_classes),
            objective_value=float(best.diagnostic.objective_value),
            runtime=time.perf_counter() - started,
            mip_gap=(0.0 if status == SolverStatus.OPTIMAL else None),
            candidate_diagnostics=diagnostics,
            features=selected,
            n_candidates=len(inputs),
            n_complete_candidates=len(complete),
        )


__all__ = [
    "ExactDepth3Solver",
    "OCT3CandidateDiagnostic",
    "OCT3Solution",
]
